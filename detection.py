"""
Person Re-Identification System - Detection & Tracking
================================================================
This script implements person detection using YOLOv5 and basic tracking using SORT algorithm.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict

# SORT Tracker Implementation - Simple Online and Realtime Tracking
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    
    def __init__(self, bbox):
        """Initialize a tracker using initial bounding box"""
        # Define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def update(self, bbox):
        """Update the state vector with observed bbox"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        
    def predict(self):
        """Advance the state vector and return the predicted bounding box estimate"""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]
        
    def get_state(self):
        """Return the current bounding box estimate"""
        return self.convert_x_to_bbox(self.kf.x)
        
    @staticmethod
    def convert_bbox_to_z(bbox):
        """Convert [x1,y1,x2,y2] to [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))
        
    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """Convert [x,y,s,r] to [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))

def iou_batch(bb_test, bb_gt):
    """Compute IOU between two bboxes in the form [x1,y1,x2,y2]"""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o

class Sort:
    """Simple Online and Realtime Tracking"""
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        
    def update(self, dets=np.empty((0, 5))):
        """Update trackers with detections"""
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1))
            i -= 1
            # Remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))
        
    def associate_detections_to_trackers(self, detections, trackers):
        """Assign detections to tracked object"""
        if len(trackers) == 0:
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
            
        iou_matrix = iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = self.linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0,2))
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:,0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:,1]:
                unmatched_trackers.append(t)
                
        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
                
        if len(matches) == 0:
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
        
    def linear_assignment(self, cost_matrix):
        """Linear assignment using Hungarian algorithm"""
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return np.column_stack((row_ind, col_ind))

class PersonDetectorTracker:
    """Person detection and tracking pipeline"""
    
    def __init__(self, conf_threshold=0.5, iou_threshold=0.3):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print("Loading YOLOv5 model...")
        
        # Load YOLOv5 from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to(self.device)
        self.model.conf = conf_threshold
        self.model.classes = [0]  # Only detect persons (class 0)
        
        print("YOLOv5 model loaded successfully!")
        
        # Initialize SORT tracker with simplified parameters
        self.tracker = Sort(max_age=5, min_hits=1, iou_threshold=iou_threshold)
        
        # Statistics
        self.detection_history = defaultdict(list)
        
    def detect_persons(self, frame):
        """Detect persons in a frame"""
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]
        
        # Filter for person class and confidence
        person_detections = detections[detections[:, 5] == 0]  # class 0 is person
        
        return person_detections
    
    def process_video(self, video_path, output_path=None, visualize=True, save_detections=True):
        """Process video for person detection and tracking"""
        
        print("\n" + "=" * 60)
        print(f"Processing: {video_path}")
        print("=" * 60)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer if visualization requested
        writer = None
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_num = 0
        all_detections = []
        unique_ids = set()
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Detect persons
            detections = self.detect_persons(frame)
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Store detections
            frame_detections = {
                'frame': frame_num,
                'timestamp': frame_num / fps,
                'detections': []
            }
            
            # Draw and record tracked objects
            for track in tracked_objects:
                x1, y1, x2, y2, track_id = track
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                track_id = int(track_id)
                
                unique_ids.add(track_id)
                
                # Store detection info
                detection_info = {
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                frame_detections['detections'].append(detection_info)
                
                # Visualize
                if visualize:
                    # Draw bounding box
                    color = self.get_color_for_id(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw ID label
                    label = f"ID: {track_id}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add frame info
            if visualize:
                info_text = f"Frame: {frame_num}/{total_frames} | Tracked: {len(tracked_objects)} | Unique IDs: {len(unique_ids)}"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            all_detections.append(frame_detections)
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Progress indicator
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames}) - Unique IDs: {len(unique_ids)}")
        
        cap.release()
        if writer:
            writer.release()
        
        # Save detection data
        if save_detections:
            detection_file = Path(video_path).stem + "_detections.json"
            detection_path = Path("data") / "sample_frames" / detection_file
            
            with open(detection_path, 'w') as f:
                json.dump({
                    'video_info': {
                        'path': str(video_path),
                        'fps': fps,
                        'resolution': [width, height],
                        'total_frames': total_frames
                    },
                    'statistics': {
                        'unique_persons': len(unique_ids),
                        'total_detections': sum(len(f['detections']) for f in all_detections)
                    },
                    'detections': all_detections
                }, f, indent=2)
            
            print(f"\nDetections saved to: {detection_path}")
        
        print("\n" + "=" * 60)
        print("TRACKING STATISTICS")
        print("=" * 60)
        print(f"Total unique persons tracked: {len(unique_ids)}")
        print(f"Total frames processed: {frame_num}")
        print(f"Average detections per frame: {sum(len(f['detections']) for f in all_detections) / frame_num:.2f}")
        
        if output_path:
            print(f"Output video saved to: {output_path}")
        
        return all_detections, unique_ids
    
    @staticmethod
    def get_color_for_id(track_id):
        """Generate consistent color for each track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())

def main():
    """Main function to run Phase 2"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Detection & Tracking  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Initialize detector
    detector = PersonDetectorTracker(conf_threshold=0.5, iou_threshold=0.3)
    
    # Find video files
    video_dir = Path("data/videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print("\nNo video files found in data/videos/")
        print("Please add video files and try again.")
        return
    
    print(f"\nFound {len(video_files)} video file(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    
    # Process each video
    output_dir = Path("output/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_file in video_files:
        output_path = output_dir / f"{video_file.stem}_tracked.mp4"
        detector.process_video(
            video_path=video_file,
            output_path=output_path,
            visualize=True,
            save_detections=True
        )
    
    print("\nNext Steps:")
    print("1. Check output/videos/ for tracked videos")
    print("2. Check data/sample_frames/ for detection JSON files")
    print("3. Run recognition.py face recognition")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()