"""
Person Re-Identification System - Body Features
====================================================================
This script implements appearance-based re-identification features using deep learning
and traditional computer vision techniques.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from pathlib import Path
import json
import pickle
from collections import defaultdict

class AppearanceFeatureExtractor:
    """Extract appearance-based features for person re-identification"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print("Loading appearance models...")
        
        # Use ResNet50 as backbone for appearance features
        self.appearance_model = models.resnet50(pretrained=True)
        # Remove final classification layer
        self.appearance_model = nn.Sequential(*list(self.appearance_model.children())[:-1])
        self.appearance_model.eval()
        self.appearance_model.to(self.device)
        
        print("ResNet50 appearance model loaded!")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID dimensions
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_deep_features(self, frame, bbox):
        """
        Extract deep appearance features from person crop
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2] person bounding box
        
        Returns:
            Feature vector (2048-dim from ResNet50)
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Crop person
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0 or person_img.shape[0] < 10 or person_img.shape[1] < 10:
            return None
        
        # Preprocess
        img_tensor = self.transform(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.appearance_model(img_tensor)
        
        return features.cpu().numpy().flatten()
    
    def extract_color_histogram(self, frame, bbox, bins=32):
        """
        Extract color histogram features
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2] person bounding box
            bins: Number of bins per channel
        
        Returns:
            Color histogram feature vector
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        # Crop person
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return None
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        # Concatenate all channels
        color_hist = np.concatenate([hist_h, hist_s, hist_v])
        
        return color_hist
    
    def extract_body_structure(self, frame, bbox):
        """
        Extract simple body structure features (aspect ratio, size)
        
        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2] person bounding box
        
        Returns:
            Structure feature vector
        """
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        
        # Aspect ratio
        aspect_ratio = height / (width + 1e-6)
        
        # Relative size, normalized by frame size
        rel_width = width / frame.shape[1]
        rel_height = height / frame.shape[0]
        rel_area = (width * height) / (frame.shape[0] * frame.shape[1])
        
        return np.array([aspect_ratio, rel_width, rel_height, rel_area])

class PersonReIDFeatureExtractor:
    """Comprehensive person re-identification feature extraction"""
    
    def __init__(self):
        self.appearance_extractor = AppearanceFeatureExtractor()
        self.person_features = defaultdict(lambda: {
            'deep_features': [],
            'color_features': [],
            'structure_features': [],
            'frames': []
        })
    
    def process_video_features(self, video_path, detection_file):
        """
        Extract all re-identification features from video
        
        Args:
            video_path: Path to video file
            detection_file: Path to detection JSON from Phase 2
        """
        print("\n" + "=" * 60)
        print(f"Extracting ReID features for: {video_path}")
        print("=" * 60)
        
        # Load detections
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        detections = detection_data['detections']
        video_info = detection_data['video_info']
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        print(f"\nProcessing {len(detections)} frames for appearance features...")
        
        total_features = 0
        
        for frame_idx, frame_det in enumerate(detections):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process each person detection
            for detection in frame_det['detections']:
                track_id = detection['track_id']
                bbox = detection['bbox']
                
                # Extract deep appearance features
                deep_feat = self.appearance_extractor.extract_deep_features(frame, bbox)
                
                # Extract color histogram
                color_feat = self.appearance_extractor.extract_color_histogram(frame, bbox)
                
                # Extract body structure
                structure_feat = self.appearance_extractor.extract_body_structure(frame, bbox)
                
                if deep_feat is not None and color_feat is not None:
                    self.person_features[track_id]['deep_features'].append(deep_feat)
                    self.person_features[track_id]['color_features'].append(color_feat)
                    self.person_features[track_id]['structure_features'].append(structure_feat)
                    self.person_features[track_id]['frames'].append(frame_det['frame'])
                    total_features += 1
            
            # Progress indicator
            if (frame_idx + 1) % 30 == 0:
                progress = ((frame_idx + 1) / len(detections)) * 100
                print(f"Progress: {progress:.1f}% - Features extracted: {total_features}")
        
        cap.release()
        
        # Compute average features for each person
        person_reid_features = {}
        
        for track_id, features in self.person_features.items():
            if features['deep_features']:
                # Average all features
                avg_deep = np.mean(features['deep_features'], axis=0)
                avg_color = np.mean(features['color_features'], axis=0)
                avg_structure = np.mean(features['structure_features'], axis=0)
                
                person_reid_features[track_id] = {
                    'deep_features': avg_deep,
                    'color_features': avg_color,
                    'structure_features': avg_structure,
                    'num_samples': len(features['deep_features']),
                    'frames': features['frames']
                }
        
        print("\n" + "=" * 60)
        print("RE-ID FEATURE STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {len(detections)}")
        print(f"Total features extracted: {total_features}")
        print(f"Unique persons with features: {len(person_reid_features)}")
        
        for track_id, feat in person_reid_features.items():
            print(f"  - Person ID {track_id}: {feat['num_samples']} samples")
        
        # Save features
        output_file = Path(detection_file).parent / f"{Path(video_path).stem}_reid_features.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'person_features': person_reid_features,
                'video_info': video_info
            }, f)
        
        print(f"\nReID features saved to: {output_file}")
        
        return person_reid_features
    
    def compute_feature_similarity(self, features1, features2, weights=None):
        """
        Compute similarity between two feature sets
        
        Args:
            features1: First feature dict
            features2: Second feature dict
            weights: Optional weights for different feature types
        
        Returns:
            Similarity score (0-1)
        """
        if weights is None:
            weights = {
                'deep': 0.6,      # Deep features most important
                'color': 0.3,     # Color somewhat important
                'structure': 0.1  # Structure least important (can change with clothing)
            }
        
        # Compute cosine similarity for each feature type
        def cosine_sim(a, b):
            a_norm = a / (np.linalg.norm(a) + 1e-8)
            b_norm = b / (np.linalg.norm(b) + 1e-8)
            return np.dot(a_norm, b_norm)
        
        deep_sim = cosine_sim(features1['deep_features'], features2['deep_features'])
        color_sim = cosine_sim(features1['color_features'], features2['color_features'])
        structure_sim = cosine_sim(features1['structure_features'], features2['structure_features'])
        
        # Weighted combination
        total_sim = (weights['deep'] * deep_sim + 
                    weights['color'] * color_sim + 
                    weights['structure'] * structure_sim)
        
        return float(total_sim), {
            'deep': float(deep_sim),
            'color': float(color_sim),
            'structure': float(structure_sim)
        }
    
    def visualize_features(self, video_path, detection_file, output_path):
        """
        Create visualization showing feature extraction
        """
        print("\n" + "=" * 60)
        print(f"Creating feature visualization for: {video_path}")
        print("=" * 60)
        
        # Load detections
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        detections = detection_data['detections']
        video_info = detection_data['video_info']
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Setup video writer
        fps = video_info['fps']
        width, height = video_info['resolution']
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print("\nGenerating visualization...")
        
        for frame_idx, frame_det in enumerate(detections):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process each person
            for detection in frame_det['detections']:
                track_id = detection['track_id']
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Draw person bbox
                color = self.get_color_for_id(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Extract color histogram for visualization
                color_feat = self.appearance_extractor.extract_color_histogram(frame, bbox, bins=8)
                
                if color_feat is not None:
                    # Show dominant colors
                    cv2.putText(frame, f"ID: {track_id} [Features OK]", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add frame info
            info_text = f"Frame: {frame_det['frame']} | ReID Features: Active"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            writer.write(frame)
            
            if (frame_idx + 1) % 30 == 0:
                progress = ((frame_idx + 1) / len(detections)) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        writer.release()
        
        print(f"\nVisualization saved to: {output_path}")
    
    @staticmethod
    def get_color_for_id(track_id):
        """Generate consistent color for each track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

def main():
    """Main function to run Phase 4"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Appearance Features  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Initialize feature extractor
    extractor = PersonReIDFeatureExtractor()
    
    # Find detection files from Phase 2
    detection_dir = Path("data/sample_frames")
    detection_files = list(detection_dir.glob("*_detections.json"))
    
    if not detection_files:
        print("\nNo detection files found!")
        print("Please run detection.py first.")
        return
    
    print(f"\nFound {len(detection_files)} detection file(s):")
    for i, df in enumerate(detection_files, 1):
        print(f"  {i}. {df.name}")
    
    # Process each video
    video_dir = Path("data/videos")
    output_dir = Path("output/videos")
    
    for detection_file in detection_files:
        # Find corresponding video
        video_name = detection_file.stem.replace("_detections", "")
        video_files = list(video_dir.glob(f"{video_name}.*"))
        
        if not video_files:
            print(f"\nVideo not found for {detection_file.name}")
            continue
        
        video_file = video_files[0]
        
        # Extract ReID features
        person_features = extractor.process_video_features(
            video_file,
            detection_file
        )
        
        # Create visualization
        output_path = output_dir / f"{video_name}_reid_features.mp4"
        extractor.visualize_features(video_file, detection_file, output_path)
    
    print("\nOutputs:")
    print("1. ReID feature files (.pkl) in data/sample_frames/")
    print("2. Feature visualization videos in output/videos/")
    print("\nNext Steps:")
    print("Run feature_matching.py")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()