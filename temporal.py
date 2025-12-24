"""
Person Re-Identification System - Temporal Re-identification
======================================================================
This script implements temporal re-identification to handle people leaving
and re-entering the scene within the same video.
"""

import cv2
import numpy as np
import json
import pickle
from pathlib import Path
from collections import defaultdict
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

class TemporalGallery:
    """Dynamic gallery that tracks persons over time within a video"""
    
    def __init__(self, similarity_threshold=0.65, temporal_window=150):
        """
        Initialize temporal gallery
        
        Args:
            similarity_threshold: Minimum similarity for re-identification
            temporal_window: Number of frames before considering a person "left"
        """
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window
        
        # Active persons (currently in scene)
        self.active_persons = {}
        
        # Historical persons (left the scene)
        self.historical_persons = {}
        
        # Re-identification map: original_id -> persistent_id
        self.reid_map = {}
        
        # Next persistent ID to assign
        self.next_persistent_id = 1
        
        # Track last seen frame for each person
        self.last_seen = {}
        
        # Feature extractors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_feature_extractors()
        
    def _init_feature_extractors(self):
        """Initialize lightweight feature extractors for real-time use"""
        # ResNet for appearance
        self.appearance_model = models.resnet50(pretrained=True)
        self.appearance_model = nn.Sequential(*list(self.appearance_model.children())[:-1])
        self.appearance_model.eval()
        self.appearance_model.to(self.device)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, frame, bbox):
        """Extract quick features from a person crop"""
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0 or person_img.shape[0] < 10 or person_img.shape[1] < 10:
            return None
        
        # Extract deep features
        img_tensor = self.transform(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            deep_features = self.appearance_model(img_tensor)
        
        # Color histogram
        hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])
        
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        color_features = np.concatenate([hist_h, hist_s, hist_v])
        
        return {
            'deep': deep_features.cpu().numpy().flatten(),
            'color': color_features
        }
    
    def compute_similarity(self, features1, features2):
        """Compute similarity between two feature sets"""
        if features1 is None or features2 is None:
            return 0.0
        
        # Cosine similarity for deep features
        deep1 = features1['deep'] / (np.linalg.norm(features1['deep']) + 1e-8)
        deep2 = features2['deep'] / (np.linalg.norm(features2['deep']) + 1e-8)
        deep_sim = np.dot(deep1, deep2)
        
        # Cosine similarity for color
        color1 = features1['color'] / (np.linalg.norm(features1['color']) + 1e-8)
        color2 = features2['color'] / (np.linalg.norm(features2['color']) + 1e-8)
        color_sim = np.dot(color1, color2)
        
        # Weighted combination
        total_sim = 0.7 * deep_sim + 0.3 * color_sim
        
        return float(total_sim)
    
    def update(self, frame_num, detections, frame):
        """
        Update gallery with new detections from current frame
        
        Args:
            frame_num: Current frame number
            detections: List of detections with track_id and bbox
            frame: Current frame image
        
        Returns:
            Dictionary mapping track_id to persistent_id
        """
        current_ids = set()
        frame_reid_map = {}
        
        # Process each detection
        for detection in detections:
            track_id = detection['track_id']
            bbox = detection['bbox']
            current_ids.add(track_id)
            
            # Extract features
            features = self.extract_features(frame, bbox)
            
            if features is None:
                continue
            
            # Check if this is a new detection or existing one
            if track_id in self.active_persons:
                # Update existing active person
                persistent_id = self.reid_map[track_id]
                self.active_persons[track_id]['features'].append(features)
                self.active_persons[track_id]['last_frame'] = frame_num
                self.last_seen[track_id] = frame_num
                frame_reid_map[track_id] = persistent_id
                
            else:
                # New track_id - could be genuinely new or a re-entry
                best_match_id = None
                best_similarity = -1
                
                # Check against historical persons (who left)
                for hist_id, hist_data in self.historical_persons.items():
                    # Average historical features
                    avg_features = self._average_features(hist_data['features'])
                    similarity = self.compute_similarity(features, avg_features)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = hist_id
                
                # Decide: re-identification or new person?
                if best_similarity >= self.similarity_threshold:
                    # Re-identified! Person returned
                    persistent_id = self.historical_persons[best_match_id]['persistent_id']
                    self.reid_map[track_id] = persistent_id
                    
                    # Move from historical to active
                    self.active_persons[track_id] = {
                        'features': [features],
                        'persistent_id': persistent_id,
                        'first_frame': frame_num,
                        'last_frame': frame_num,
                        'is_reentry': True,
                        'original_track_id': best_match_id
                    }
                    self.last_seen[track_id] = frame_num
                    frame_reid_map[track_id] = persistent_id
                    
                else:
                    # Genuinely new person
                    persistent_id = self.next_persistent_id
                    self.next_persistent_id += 1
                    
                    self.reid_map[track_id] = persistent_id
                    self.active_persons[track_id] = {
                        'features': [features],
                        'persistent_id': persistent_id,
                        'first_frame': frame_num,
                        'last_frame': frame_num,
                        'is_reentry': False,
                        'original_track_id': None
                    }
                    self.last_seen[track_id] = frame_num
                    frame_reid_map[track_id] = persistent_id
        
        # Move inactive persons to historical
        inactive_ids = set(self.active_persons.keys()) - current_ids
        for track_id in inactive_ids:
            if frame_num - self.last_seen.get(track_id, frame_num) > self.temporal_window:
                # Person has been gone long enough, move to historical
                self.historical_persons[track_id] = self.active_persons[track_id]
                del self.active_persons[track_id]
        
        return frame_reid_map
    
    def _average_features(self, feature_list):
        """Average a list of feature dictionaries"""
        if not feature_list:
            return None
        
        avg_deep = np.mean([f['deep'] for f in feature_list], axis=0)
        avg_color = np.mean([f['color'] for f in feature_list], axis=0)
        
        return {
            'deep': avg_deep,
            'color': avg_color
        }
    
    def get_statistics(self):
        """Get gallery statistics"""
        total_unique = self.next_persistent_id - 1
        active = len(self.active_persons)
        historical = len(self.historical_persons)
        reentries = sum(1 for p in self.active_persons.values() if p.get('is_reentry', False))
        
        return {
            'total_unique_persons': total_unique,
            'active_persons': active,
            'historical_persons': historical,
            'detected_reentries': reentries
        }

class TemporalReIDProcessor:
    """Process videos with temporal re-identification"""
    
    def __init__(self, similarity_threshold=0.65, temporal_window=150):
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window
    
    def process_video(self, video_path, detection_file, output_path):
        """
        Process video with temporal re-identification
        
        Args:
            video_path: Path to video
            detection_file: Detection JSON from Phase 2
            output_path: Output video path
        """
        print("\n" + "=" * 60)
        print(f"Processing temporal re-ID for: {video_path}")
        print("=" * 60)
        
        # Load detections
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        detections = detection_data['detections']
        video_info = detection_data['video_info']
        
        # Initialize temporal gallery
        gallery = TemporalGallery(
            similarity_threshold=self.similarity_threshold,
            temporal_window=self.temporal_window
        )
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Setup video writer
        fps = video_info['fps']
        width, height = video_info['resolution']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"\nProcessing {len(detections)} frames...")
        
        all_reid_results = []
        reentry_events = []
        
        for frame_idx, frame_det in enumerate(detections):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num = frame_det['frame']
            
            # Update gallery and get re-identification mapping
            reid_map = gallery.update(frame_num, frame_det['detections'], frame)
            
            # Visualize
            for detection in frame_det['detections']:
                track_id = detection['track_id']
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                persistent_id = reid_map.get(track_id, track_id)
                
                # Check if this is a re-entry
                is_reentry = False
                if track_id in gallery.active_persons:
                    is_reentry = gallery.active_persons[track_id].get('is_reentry', False)
                
                # Color coding
                if is_reentry:
                    color = (0, 255, 255)  # Yellow for re-entry
                    label = f"Person {persistent_id} (RE-ENTRY)"
                else:
                    color = (0, 255, 0)  # Green for normal
                    label = f"Person {persistent_id}"
                
                # Track re-entry events
                if is_reentry and track_id not in [e['track_id'] for e in reentry_events]:
                    reentry_events.append({
                        'frame': frame_num,
                        'track_id': track_id,
                        'persistent_id': persistent_id
                    })
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add statistics overlay
            stats = gallery.get_statistics()
            info_lines = [
                f"Frame: {frame_num}",
                f"Unique Persons: {stats['total_unique_persons']}",
                f"Active: {stats['active_persons']}",
                f"Re-entries: {stats['detected_reentries']}"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(frame, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Add legend
            cv2.putText(frame, "Green=Normal | Yellow=Re-entry", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame)
            
            # Store results
            all_reid_results.append({
                'frame': frame_num,
                'reid_map': reid_map.copy(),
                'stats': stats.copy()
            })
            
            if (frame_idx + 1) % 30 == 0:
                progress = ((frame_idx + 1) / len(detections)) * 100
                print(f"Progress: {progress:.1f}% - Unique: {stats['total_unique_persons']}, Re-entries: {stats['detected_reentries']}")
        
        cap.release()
        writer.release()
        
        # Final statistics
        final_stats = gallery.get_statistics()
        
        print("\n" + "=" * 60)
        print("TEMPORAL RE-IDENTIFICATION RESULTS")
        print("=" * 60)
        print(f"Total unique persons detected: {final_stats['total_unique_persons']}")
        print(f"Detected re-entry events: {final_stats['detected_reentries']}")
        
        if reentry_events:
            print("\nRe-entry Events:")
            for event in reentry_events:
                print(f"  Frame {event['frame']}: Person {event['persistent_id']} re-entered (Track ID {event['track_id']})")
        
        # Save results
        results_file = Path(output_path).parent / f"{Path(video_path).stem}_temporal_reid.json"
        with open(results_file, 'w') as f:
            json.dump({
                'statistics': final_stats,
                'reentry_events': reentry_events,
                'frame_results': all_reid_results
            }, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        print(f"Video saved to: {output_path}")
        
        return final_stats, reentry_events

def main():
    """Main function to run Phase 7"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Temporal Re-ID  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Initialize processor
    processor = TemporalReIDProcessor(
        similarity_threshold=0.65,
        temporal_window=150  # ~5 seconds at 30fps
    )
    
    # Find videos and detections
    video_dir = Path("data/videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print("\nNo video files found!")
        return
    
    detection_dir = Path("data/sample_frames")
    output_dir = Path("output/videos")
    
    print(f"\nFound {len(video_files)} video(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    
    # Process each video
    for video_file in video_files:
        video_name = video_file.stem
        detection_file = detection_dir / f"{video_name}_detections.json"
        
        if not detection_file.exists():
            print(f"\nNo detection file for {video_name}")
            continue
        
        output_path = output_dir / f"{video_name}_temporal_reid.mp4"
        
        stats, events = processor.process_video(
            video_file,
            detection_file,
            output_path
        )
    
    print("\nOutputs:")
    print("1. Temporal re-ID videos in output/videos/")
    print("2. Re-entry event logs (JSON)")
    print("3. Statistics on unique persons and re-entries")
    print("\nThe system now handles:")
    print("- People leaving and returning to scene")
    print("- Persistent IDs across temporal gaps")
    print("- Visual indicators for re-entry events")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()