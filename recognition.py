"""
Person Re-Identification System - Face Recognition
============================================================
This script implements face detection and embedding extraction for person re-identification.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path
import json
from collections import defaultdict
import pickle

class FaceRecognitionModule:
    """Face detection and embedding extraction"""
    
    def __init__(self, conf_threshold=0.7):
        self.conf_threshold = conf_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print("Loading face detection model...")
        
        # Load Haar Cascade for face detection (lightweight and fast)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        print("Face detector loaded successfully!")
        print("Loading face recognition model (FaceNet)...")
        
        # Load InceptionResnetV1 (FaceNet) for face embeddings
        try:
            from facenet_pytorch import InceptionResnetV1
            self.face_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.use_facenet = True
            print("FaceNet model loaded successfully!")
        except ImportError:
            print("facenet-pytorch not available, using alternative method")
            print("Installing facenet-pytorch...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "facenet-pytorch"])
            from facenet_pytorch import InceptionResnetV1
            self.face_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.use_facenet = True
            print("FaceNet model loaded successfully!")
        
        # Face preprocessing
        self.face_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def detect_faces(self, frame, person_bbox=None):
        """
        Detect faces in frame, optionally within a person bounding box
        
        Args:
            frame: Input frame
            person_bbox: [x1, y1, x2, y2] optional bounding box to search within
        
        Returns:
            List of face bounding boxes [x, y, w, h]
        """
        # If person bbox provided, crop to that region
        if person_bbox is not None:
            x1, y1, x2, y2 = person_bbox
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
            search_region = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            search_region = frame
            offset = (0, 0)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Adjust coordinates to full frame
        adjusted_faces = []
        for (x, y, w, h) in faces:
            adjusted_faces.append([
                x + offset[0],
                y + offset[1],
                w,
                h
            ])
        
        return adjusted_faces
    
    def extract_face_embedding(self, frame, face_bbox):
        """
        Extract face embedding from face region
        
        Args:
            frame: Input frame
            face_bbox: [x, y, w, h] face bounding box
        
        Returns:
            Face embedding vector (512-dim)
        """
        x, y, w, h = face_bbox
        
        # Add padding
        padding = int(0.2 * max(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        # Crop face
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size == 0:
            return None
        
        # Preprocess
        face_tensor = self.face_transform(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.face_model(face_tensor)
        
        return embedding.cpu().numpy().flatten()
    
    def compute_similarity(self, embedding1, embedding2):
        """
        Compute cosine similarity between two embeddings
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)

class FaceFeatureExtractor:
    """Extract and manage face features for tracked persons"""
    
    def __init__(self):
        self.face_module = FaceRecognitionModule()
        self.person_face_embeddings = defaultdict(list)
        
    def process_video_with_faces(self, video_path, detection_file):
        """
        Process video to extract face features for tracked persons
        
        Args:
            video_path: Path to video file
            detection_file: Path to detection JSON from Phase 2
        """
        print("\n" + "=" * 60)
        print(f"Processing faces for: {video_path}")
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
        
        print(f"\nExtracting face features from {len(detections)} frames...")
        
        face_data = []
        frames_with_faces = 0
        total_faces_detected = 0
        
        for frame_idx, frame_det in enumerate(detections):
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_faces = []
            
            # Process each person detection in this frame
            for detection in frame_det['detections']:
                track_id = detection['track_id']
                bbox = detection['bbox']
                
                # Detect faces within person bbox
                faces = self.face_module.detect_faces(frame, bbox)
                
                if faces:
                    # Take the largest face (most likely to be the person's face)
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    
                    # Extract embedding
                    embedding = self.face_module.extract_face_embedding(frame, largest_face)
                    
                    if embedding is not None:
                        # Store embedding for this person
                        self.person_face_embeddings[track_id].append({
                            'frame': frame_det['frame'],
                            'embedding': embedding,
                            'bbox': largest_face
                        })
                        
                        frame_faces.append({
                            'track_id': track_id,
                            'face_bbox': largest_face,
                            'has_embedding': True
                        })
                        
                        total_faces_detected += 1
                else:
                    frame_faces.append({
                        'track_id': track_id,
                        'face_bbox': None,
                        'has_embedding': False
                    })
            
            if frame_faces:
                frames_with_faces += 1
            
            face_data.append({
                'frame': frame_det['frame'],
                'faces': frame_faces
            })
            
            # Progress indicator
            if (frame_idx + 1) % 30 == 0:
                progress = ((frame_idx + 1) / len(detections)) * 100
                print(f"Progress: {progress:.1f}% - Faces detected: {total_faces_detected}")
        
        cap.release()
        
        # Compute average embeddings for each person
        person_face_features = {}
        for track_id, embeddings in self.person_face_embeddings.items():
            if embeddings:
                # Average all embeddings for this person
                avg_embedding = np.mean([e['embedding'] for e in embeddings], axis=0)
                person_face_features[track_id] = {
                    'avg_embedding': avg_embedding,
                    'num_samples': len(embeddings),
                    'frames_with_face': [e['frame'] for e in embeddings]
                }
        
        print("\n" + "=" * 60)
        print("FACE RECOGNITION STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {len(detections)}")
        print(f"Frames with faces: {frames_with_faces}")
        print(f"Total faces detected: {total_faces_detected}")
        print(f"Unique persons with faces: {len(person_face_features)}")
        
        for track_id, features in person_face_features.items():
            print(f"  - Person ID {track_id}: {features['num_samples']} face samples")
        
        # Save face features
        output_file = Path(detection_file).parent / f"{Path(video_path).stem}_face_features.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump({
                'person_features': person_face_features,
                'frame_data': face_data,
                'video_info': video_info
            }, f)
        
        print(f"\nFace features saved to: {output_file}")
        
        return person_face_features, face_data
    
    def visualize_faces(self, video_path, detection_file, output_path):
        """
        Create visualization with face detection overlays
        """
        print("\n" + "=" * 60)
        print(f"Creating face visualization for: {video_path}")
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
                
                # Draw person bbox (blue)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Detect and draw face
                faces = self.face_module.detect_faces(frame, bbox)
                if faces:
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    fx, fy, fw, fh = largest_face
                    
                    # Draw face bbox (green)
                    cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                    cv2.putText(frame, "FACE", (fx, fy - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_det['frame']} | Persons: {len(frame_det['detections'])}"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            writer.write(frame)
            
            if (frame_idx + 1) % 30 == 0:
                progress = ((frame_idx + 1) / len(detections)) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        writer.release()
        
        print(f"\nVisualization saved to: {output_path}")

def main():
    """Main function to run Phase 3"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Face Recognition  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Initialize face extractor
    extractor = FaceFeatureExtractor()
    
    # Find detection files from Phase 2
    detection_dir = Path("data/sample_frames")
    detection_files = list(detection_dir.glob("*_detections.json"))
    
    if not detection_files:
        print("\nNo detection files found!")
        print("Please run Phase 2 first.")
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
        
        # Extract face features
        person_features, face_data = extractor.process_video_with_faces(
            video_file,
            detection_file
        )
        
        # Create visualization
        output_path = output_dir / f"{video_name}_faces.mp4"
        extractor.visualize_faces(video_file, detection_file, output_path)
    
    print("\nOutputs:")
    print("1. Face feature files (.pkl) in data/sample_frames/")
    print("2. Face visualization videos in output/videos/")
    print("\nNext Steps:")
    print("Run feature_extraction.py to implement body re-identification features")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()