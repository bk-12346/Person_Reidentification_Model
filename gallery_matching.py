"""
Person Re-Identification System - Gallery Matching
============================================================
This script implements the matching logic to identify persons against a gallery
and handle re-identification across time and videos.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
import cv2

class GalleryMatcher:
    """Match detected persons against a gallery of known signatures"""
    
    def __init__(self, gallery_path, similarity_threshold=0.6):
        """
        Initialize matcher with a gallery
        
        Args:
            gallery_path: Path to gallery JSON file
            similarity_threshold: Minimum similarity to consider a match
        """
        self.similarity_threshold = similarity_threshold
        self.gallery = self.load_gallery(gallery_path)
        
        print(f"Gallery loaded with {len(self.gallery)} signatures")
        print(f"Match threshold: {similarity_threshold}")
        
    def load_gallery(self, gallery_path):
        """Load gallery from JSON"""
        with open(gallery_path, 'r') as f:
            data = json.load(f)
        
        # Import PersonSignature from Phase 5
        from collections import namedtuple
        
        gallery = {}
        for key_str, sig_dict in data.items():
            key = eval(key_str)
            
            # Create simplified signature object
            sig = type('Signature', (), {})()
            sig.person_id = sig_dict['person_id']
            sig.video_source = sig_dict['video_source']
            sig.face_embedding = np.array(sig_dict['face_embedding']) if sig_dict['face_embedding'] else None
            sig.face_available = sig_dict['face_available']
            sig.deep_features = np.array(sig_dict['deep_features']) if sig_dict['deep_features'] else None
            sig.color_features = np.array(sig_dict['color_features']) if sig_dict['color_features'] else None
            sig.structure_features = np.array(sig_dict['structure_features']) if sig_dict['structure_features'] else None
            sig.appearance_available = sig_dict['appearance_available']
            sig.fusion_weights = sig_dict['fusion_weights']
            
            gallery[key] = sig
        
        return gallery
    
    def compute_similarity(self, query_sig, gallery_sig):
        """
        Compute similarity between query and gallery signature
        
        Args:
            query_sig: Query signature
            gallery_sig: Gallery signature
        
        Returns:
            Similarity score and breakdown
        """
        total_sim = 0.0
        details = {}
        
        # Average weights from both signatures
        weights = {
            k: (query_sig.fusion_weights[k] + gallery_sig.fusion_weights[k]) / 2
            for k in query_sig.fusion_weights.keys()
        }
        
        # Face similarity
        if query_sig.face_available and gallery_sig.face_available:
            face_sim = self._cosine_similarity(query_sig.face_embedding, gallery_sig.face_embedding)
            details['face'] = float(face_sim)
            total_sim += weights['face'] * face_sim
        else:
            details['face'] = None
        
        # Appearance similarities
        if query_sig.appearance_available and gallery_sig.appearance_available:
            deep_sim = self._cosine_similarity(query_sig.deep_features, gallery_sig.deep_features)
            color_sim = self._cosine_similarity(query_sig.color_features, gallery_sig.color_features)
            structure_sim = self._cosine_similarity(query_sig.structure_features, gallery_sig.structure_features)
            
            details['deep'] = float(deep_sim)
            details['color'] = float(color_sim)
            details['structure'] = float(structure_sim)
            
            total_sim += weights['deep_appearance'] * deep_sim
            total_sim += weights['color'] * color_sim
            total_sim += weights['structure'] * structure_sim
        else:
            details['deep'] = None
            details['color'] = None
            details['structure'] = None
        
        return total_sim, details
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Compute cosine similarity"""
        if vec1 is None or vec2 is None:
            return 0.0
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return np.dot(vec1_norm, vec2_norm)
    
    def find_best_match(self, query_signature):
        """
        Find best matching signature in gallery
        
        Args:
            query_signature: Signature to match
        
        Returns:
            Best match key, similarity score, and details
        """
        best_match = None
        best_similarity = -1
        best_details = None
        
        for gallery_key, gallery_sig in self.gallery.items():
            similarity, details = self.compute_similarity(query_signature, gallery_sig)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = gallery_key
                best_details = details
        
        # Check if best match exceeds threshold
        if best_similarity >= self.similarity_threshold:
            return best_match, best_similarity, best_details
        else:
            return None, best_similarity, best_details
    
    def match_all_signatures(self, query_gallery):
        """
        Match all signatures from a query video against the gallery
        
        Args:
            query_gallery: Dictionary of query signatures
        
        Returns:
            Matching results
        """
        results = {}
        
        for query_key, query_sig in query_gallery.items():
            match_key, similarity, details = self.find_best_match(query_sig)
            
            results[query_key] = {
                'matched': match_key is not None,
                'match_key': match_key,
                'similarity': similarity,
                'details': details,
                'query_video': query_sig.video_source,
                'query_person_id': query_sig.person_id
            }
        
        return results

class ReIdentificationEngine:
    """End-to-end re-identification engine"""
    
    def __init__(self, gallery_path, similarity_threshold=0.6):
        self.matcher = GalleryMatcher(gallery_path, similarity_threshold)
        
    def test_cross_video_matching(self, video1_path, video2_path, detection_dir):
        """
        Test re-identification between two videos
        
        Args:
            video1_path: First video (for gallery)
            video2_path: Second video (for queries)
            detection_dir: Directory with feature files
        """
        print("\n" + "=" * 60)
        print("CROSS-VIDEO RE-IDENTIFICATION TEST")
        print("=" * 60)
        
        video1_name = Path(video1_path).stem
        video2_name = Path(video2_path).stem
        
        print(f"\nGallery Video: {video1_name}")
        print(f"Query Video: {video2_name}")
        
        # Build query gallery from video 2
        print("\nBuilding query signatures from video 2...")
        query_gallery = self._build_query_gallery(video2_path, detection_dir)
        
        print(f"Query gallery: {len(query_gallery)} persons")
        
        # Match against main gallery
        print("\nMatching query signatures against gallery...")
        results = self.matcher.match_all_signatures(query_gallery)
        
        # Analyze results
        self._analyze_matching_results(results)
        
        return results
    
    def _build_query_gallery(self, video_path, detection_dir):
        """Build a temporary gallery from a video for matching"""
        video_name = Path(video_path).stem
        
        # Load features
        face_file = detection_dir / f"{video_name}_face_features.pkl"
        reid_file = detection_dir / f"{video_name}_reid_features.pkl"
        
        face_data = None
        reid_data = None
        
        if face_file.exists():
            with open(face_file, 'rb') as f:
                face_data = pickle.load(f)
        
        if reid_file.exists():
            with open(reid_file, 'rb') as f:
                reid_data = pickle.load(f)
        
        # Get all person IDs
        person_ids = set()
        if face_data:
            person_ids.update(face_data['person_features'].keys())
        if reid_data:
            person_ids.update(reid_data['person_features'].keys())
        
        # Create signatures
        query_gallery = {}
        
        for person_id in person_ids:
            face_feat = face_data['person_features'].get(person_id) if face_data else None
            reid_feat = reid_data['person_features'].get(person_id) if reid_data else None
            
            # Create signature
            sig = type('Signature', (), {})()
            sig.person_id = person_id
            sig.video_source = video_name
            sig.face_available = False
            sig.appearance_available = False
            
            # Add face features
            if face_feat:
                sig.face_embedding = face_feat['avg_embedding']
                sig.face_available = True
            else:
                sig.face_embedding = None
            
            # Add appearance features
            if reid_feat:
                sig.deep_features = reid_feat['deep_features']
                sig.color_features = reid_feat['color_features']
                sig.structure_features = reid_feat['structure_features']
                sig.appearance_available = True
            else:
                sig.deep_features = None
                sig.color_features = None
                sig.structure_features = None
            
            # Set fusion weights
            sig.fusion_weights = self._compute_fusion_weights(sig)
            
            key = (video_name, person_id)
            query_gallery[key] = sig
        
        return query_gallery
    
    @staticmethod
    def _compute_fusion_weights(sig):
        """Compute fusion weights for a signature"""
        weights = {
            'face': 0.0,
            'deep_appearance': 0.0,
            'color': 0.0,
            'structure': 0.0
        }
        
        if sig.face_available:
            weights['face'] = 0.5
        
        if sig.appearance_available:
            remaining = 1.0 - weights['face']
            weights['deep_appearance'] = 0.5 * remaining
            weights['color'] = 0.3 * remaining
            weights['structure'] = 0.2 * remaining
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def _analyze_matching_results(self, results):
        """Analyze and display matching results"""
        print("\n" + "=" * 60)
        print("MATCHING RESULTS")
        print("=" * 60)
        
        total_queries = len(results)
        matched = sum(1 for r in results.values() if r['matched'])
        unmatched = total_queries - matched
        
        print(f"\nTotal queries: {total_queries}")
        print(f"Matched: {matched}")
        print(f"Unmatched (new persons): {unmatched}")
        print(f"Match rate: {matched/total_queries*100:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 60)
        
        for query_key, result in results.items():
            video_name, person_id = query_key
            
            if result['matched']:
                match_video, match_id = result['match_key']
                print(f"\nMATCH: {video_name} Person {person_id}")
                print(f"\nMatched to: {match_video} Person {match_id}")
                print(f"\nSimilarity: {result['similarity']:.3f}")
                
                details = result['details']
                if details['face'] is not None:
                    print(f"  Face: {details['face']:.3f}")
                if details['deep'] is not None:
                    print(f"  Appearance: {details['deep']:.3f}, Color: {details['color']:.3f}")
            else:
                print(f"\nNO MATCH: {video_name} Person {person_id}")
                print(f"  Best similarity: {result['similarity']:.3f} (below threshold)")
    
    def visualize_matches(self, video_path, detection_file, results, output_path):
        """
        Create visualization showing re-identification results
        
        Args:
            video_path: Path to query video
            detection_file: Detection JSON file
            results: Matching results
            output_path: Output video path
        """
        print("\n" + "=" * 60)
        print(f"Creating match visualization for: {video_path}")
        print("=" * 60)
        
        # Load detections
        with open(detection_file, 'r') as f:
            detection_data = json.load(f)
        
        detections = detection_data['detections']
        video_info = detection_data['video_info']
        
        # Create match lookup
        video_name = Path(video_path).stem
        match_lookup = {}
        for query_key, result in results.items():
            if query_key[0] == video_name:
                person_id = query_key[1]
                if result['matched']:
                    match_id = result['match_key'][1]
                    match_lookup[person_id] = {
                        'match_id': match_id,
                        'similarity': result['similarity']
                    }
        
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
            
            # Draw each detection
            for detection in frame_det['detections']:
                track_id = detection['track_id']
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Check if matched
                if track_id in match_lookup:
                    match_info = match_lookup[track_id]
                    color = (0, 255, 0)  # Green for matched
                    label = f"ID {track_id} → Match: ID {match_info['match_id']} ({match_info['similarity']:.2f})"
                else:
                    color = (0, 0, 255)  # Red for unmatched
                    label = f"ID {track_id} (New Person)"
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add legend
            legend_y = 30
            cv2.putText(frame, "Green = Matched | Red = New Person", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            writer.write(frame)
            
            if (frame_idx + 1) % 30 == 0:
                progress = ((frame_idx + 1) / len(detections)) * 100
                print(f"Progress: {progress:.1f}%")
        
        cap.release()
        writer.release()
        
        print(f"\nVisualization saved to: {output_path}")

def main():
    """Main function to run gallery_matching.py"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Gallery Matching  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Load gallery
    gallery_path = Path("gallery/person_gallery.json")
    
    if not gallery_path.exists():
        print("\nGallery not found!")
        print("Please run feature_matching.py first.")
        return
    
    # Initialize re-identification engine
    reid_engine = ReIdentificationEngine(gallery_path, similarity_threshold=0.6)
    
    # Find videos
    video_dir = Path("data/videos")
    video_files = sorted(list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")))
    
    if len(video_files) < 2:
        print("\nNeed at least 2 videos for cross-video matching")
        print("Testing with available videos...")
    
    print(f"\nFound {len(video_files)} video(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    
    # Test cross-video matching (use first video as gallery, second as query)
    if len(video_files) >= 2:
        detection_dir = Path("data/sample_frames")
        
        results = reid_engine.test_cross_video_matching(
            video1_path=video_files[0],
            video2_path=video_files[1],
            detection_dir=detection_dir
        )
        
        # Create visualization
        detection_file = detection_dir / f"{video_files[1].stem}_detections.json"
        output_path = Path("output/videos") / f"{video_files[1].stem}_reid_results.mp4"
        
        reid_engine.visualize_matches(
            video_files[1],
            detection_file,
            results,
            output_path
        )
    
    print("\nOutputs:")
    print("1. Matching results with similarity scores")
    print("2. Visualization video showing matches in output/videos/")
    print("\nNext Steps:")
    print("Run temporal.py to implement temporal re-identification logic")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()