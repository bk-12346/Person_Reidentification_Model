"""
Person Re-Identification System - Feature Fusion & Signatures
=======================================================================
This script combines face and appearance features into unified person signatures
for robust re-identification.
"""

import numpy as np
import pickle
from pathlib import Path
import json
from collections import defaultdict

class PersonSignature:
    """Unified person signature combining multiple feature types"""
    
    def __init__(self, person_id, video_source):
        self.person_id = person_id
        self.video_source = video_source
        
        # Feature components
        self.face_embedding = None
        self.face_available = False
        self.face_confidence = 0.0
        
        self.deep_features = None
        self.color_features = None
        self.structure_features = None
        self.appearance_available = False
        
        # Metadata
        self.num_face_samples = 0
        self.num_appearance_samples = 0
        self.frames_seen = []
        
    def set_face_features(self, face_embedding, num_samples, frames):
        """Add face features to signature"""
        self.face_embedding = face_embedding
        self.face_available = True
        self.num_face_samples = num_samples
        self.face_confidence = min(1.0, num_samples / 10.0)  # Confidence based on sample count
        
    def set_appearance_features(self, deep_feat, color_feat, structure_feat, num_samples, frames):
        """Add appearance features to signature"""
        self.deep_features = deep_feat
        self.color_features = color_feat
        self.structure_features = structure_feat
        self.appearance_available = True
        self.num_appearance_samples = num_samples
        
    def get_fusion_weights(self):
        """
        Determine fusion weights based on feature availability and quality
        
        Returns:
            Dictionary of weights for each feature type
        """
        weights = {
            'face': 0.0,
            'deep_appearance': 0.0,
            'color': 0.0,
            'structure': 0.0
        }
        
        # If face is available and confident, prioritize it
        if self.face_available:
            weights['face'] = 0.5 * self.face_confidence
        
        # Appearance features always contribute
        if self.appearance_available:
            remaining_weight = 1.0 - weights['face']
            weights['deep_appearance'] = 0.5 * remaining_weight
            weights['color'] = 0.3 * remaining_weight
            weights['structure'] = 0.2 * remaining_weight
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def to_dict(self):
        """Convert signature to dictionary for storage"""
        return {
            'person_id': self.person_id,
            'video_source': self.video_source,
            'face_embedding': self.face_embedding.tolist() if self.face_embedding is not None else None,
            'face_available': self.face_available,
            'face_confidence': self.face_confidence,
            'deep_features': self.deep_features.tolist() if self.deep_features is not None else None,
            'color_features': self.color_features.tolist() if self.color_features is not None else None,
            'structure_features': self.structure_features.tolist() if self.structure_features is not None else None,
            'appearance_available': self.appearance_available,
            'num_face_samples': self.num_face_samples,
            'num_appearance_samples': self.num_appearance_samples,
            'fusion_weights': self.get_fusion_weights()
        }
    
    @classmethod
    def from_dict(cls, data):
        """Reconstruct signature from dictionary"""
        signature = cls(data['person_id'], data['video_source'])
        
        if data['face_embedding'] is not None:
            signature.face_embedding = np.array(data['face_embedding'])
            signature.face_available = data['face_available']
            signature.face_confidence = data['face_confidence']
            signature.num_face_samples = data['num_face_samples']
        
        if data['deep_features'] is not None:
            signature.deep_features = np.array(data['deep_features'])
            signature.color_features = np.array(data['color_features'])
            signature.structure_features = np.array(data['structure_features'])
            signature.appearance_available = data['appearance_available']
            signature.num_appearance_samples = data['num_appearance_samples']
        
        return signature

class FeatureFusionEngine:
    """Engine for fusing face and appearance features"""
    
    def __init__(self):
        self.signatures = {}
        
    def create_signature(self, person_id, video_source, face_features=None, reid_features=None):
        """
        Create a unified person signature from available features
        
        Args:
            person_id: Unique person identifier
            video_source: Source video name
            face_features: Dictionary with face feature data
            reid_features: Dictionary with ReID feature data
        
        Returns:
            PersonSignature object
        """
        signature = PersonSignature(person_id, video_source)
        
        # Add face features if available
        if face_features is not None:
            signature.set_face_features(
                face_embedding=face_features['avg_embedding'],
                num_samples=face_features['num_samples'],
                frames=face_features['frames_with_face']
            )
        
        # Add appearance features if available
        if reid_features is not None:
            signature.set_appearance_features(
                deep_feat=reid_features['deep_features'],
                color_feat=reid_features['color_features'],
                structure_feat=reid_features['structure_features'],
                num_samples=reid_features['num_samples'],
                frames=reid_features['frames']
            )
        
        return signature
    
    def compute_similarity(self, sig1, sig2):
        """
        Compute similarity between two person signatures using weighted fusion
        
        Args:
            sig1: First PersonSignature
            sig2: Second PersonSignature
        
        Returns:
            Overall similarity score (0-1) and detailed breakdown
        """
        similarities = {}
        weights1 = sig1.get_fusion_weights()
        weights2 = sig2.get_fusion_weights()
        
        # Average the weights from both signatures
        avg_weights = {
            k: (weights1[k] + weights2[k]) / 2 
            for k in weights1.keys()
        }
        
        total_similarity = 0.0
        
        # Face similarity
        if sig1.face_available and sig2.face_available:
            face_sim = self._cosine_similarity(sig1.face_embedding, sig2.face_embedding)
            similarities['face'] = float(face_sim)
            total_similarity += avg_weights['face'] * face_sim
        else:
            similarities['face'] = None
        
        # Deep appearance similarity
        if sig1.appearance_available and sig2.appearance_available:
            deep_sim = self._cosine_similarity(sig1.deep_features, sig2.deep_features)
            similarities['deep_appearance'] = float(deep_sim)
            total_similarity += avg_weights['deep_appearance'] * deep_sim
            
            # Color similarity
            color_sim = self._cosine_similarity(sig1.color_features, sig2.color_features)
            similarities['color'] = float(color_sim)
            total_similarity += avg_weights['color'] * color_sim
            
            # Structure similarity
            structure_sim = self._cosine_similarity(sig1.structure_features, sig2.structure_features)
            similarities['structure'] = float(structure_sim)
            total_similarity += avg_weights['structure'] * structure_sim
        else:
            similarities['deep_appearance'] = None
            similarities['color'] = None
            similarities['structure'] = None
        
        return total_similarity, similarities, avg_weights
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors"""
        if vec1 is None or vec2 is None:
            return 0.0
        
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        return np.dot(vec1_norm, vec2_norm)
    
    def build_gallery(self, video_paths, detection_dir):
        """
        Build a gallery of person signatures from processed videos
        
        Args:
            video_paths: List of video file paths
            detection_dir: Directory containing feature files
        
        Returns:
            Dictionary of signatures keyed by (video_name, person_id)
        """
        print("\n" + "=" * 60)
        print("BUILDING PERSON SIGNATURE GALLERY")
        print("=" * 60)
        
        gallery = {}
        
        for video_path in video_paths:
            video_name = Path(video_path).stem
            print(f"\nProcessing signatures from: {video_name}")
            
            # Load face features
            face_file = detection_dir / f"{video_name}_face_features.pkl"
            face_data = None
            if face_file.exists():
                with open(face_file, 'rb') as f:
                    face_data = pickle.load(f)
                print(f"Loaded face features: {len(face_data['person_features'])} persons")
            else:
                print(f"No face features found")
            
            # Load ReID features
            reid_file = detection_dir / f"{video_name}_reid_features.pkl"
            reid_data = None
            if reid_file.exists():
                with open(reid_file, 'rb') as f:
                    reid_data = pickle.load(f)
                print(f"Loaded ReID features: {len(reid_data['person_features'])} persons")
            else:
                print(f"No ReID features found")
            
            # Get all unique person IDs
            person_ids = set()
            if face_data:
                person_ids.update(face_data['person_features'].keys())
            if reid_data:
                person_ids.update(reid_data['person_features'].keys())
            
            # Create signatures for each person
            for person_id in person_ids:
                face_feat = face_data['person_features'].get(person_id) if face_data else None
                reid_feat = reid_data['person_features'].get(person_id) if reid_data else None
                
                signature = self.create_signature(
                    person_id=person_id,
                    video_source=video_name,
                    face_features=face_feat,
                    reid_features=reid_feat
                )
                
                key = (video_name, person_id)
                gallery[key] = signature
                
                # Print signature info
                features = []
                if signature.face_available:
                    features.append(f"Face({signature.num_face_samples} samples)")
                if signature.appearance_available:
                    features.append(f"Appearance({signature.num_appearance_samples} samples)")
                
                print(f"Person {person_id}: {', '.join(features)}")
        
        print("\n" + "=" * 60)
        print(f"Gallery built with {len(gallery)} person signatures")
        print("=" * 60)
        
        return gallery
    
    def save_gallery(self, gallery, output_path):
        """Save gallery to file"""
        serialized_gallery = {
            str(k): v.to_dict() for k, v in gallery.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(serialized_gallery, f, indent=2)
        
        print(f"\nGallery saved to: {output_path}")
    
    def load_gallery(self, gallery_path):
        """Load gallery from file"""
        with open(gallery_path, 'r') as f:
            data = json.load(f)
        
        gallery = {}
        for key_str, sig_dict in data.items():
            # Parse key
            key = eval(key_str)  # Convert string tuple back to tuple
            gallery[key] = PersonSignature.from_dict(sig_dict)
        
        return gallery

def main():
    """Main function to run feature_matching"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Feature Fusion  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Initialize fusion engine
    fusion_engine = FeatureFusionEngine()
    
    # Find videos
    video_dir = Path("data/videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print("\nNo video files found!")
        return
    
    print(f"\nFound {len(video_files)} video file(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    
    # Build gallery
    detection_dir = Path("data/sample_frames")
    gallery = fusion_engine.build_gallery(video_files, detection_dir)
    
    # Analyze gallery
    print("\n" + "=" * 60)
    print("GALLERY ANALYSIS")
    print("=" * 60)
    
    total_with_face = sum(1 for sig in gallery.values() if sig.face_available)
    total_with_appearance = sum(1 for sig in gallery.values() if sig.appearance_available)
    total_with_both = sum(1 for sig in gallery.values() if sig.face_available and sig.appearance_available)
    
    print(f"Total signatures: {len(gallery)}")
    print(f"With face features: {total_with_face}")
    print(f"With appearance features: {total_with_appearance}")
    print(f"With both: {total_with_both}")
    
    # Show feature weights for each signature
    print("\nFeature Weights per Signature:")
    for (video_name, person_id), sig in gallery.items():
        weights = sig.get_fusion_weights()
        print(f"  {video_name} - Person {person_id}:")
        print(f"    Face: {weights['face']:.2f}, Deep: {weights['deep_appearance']:.2f}, "
              f"Color: {weights['color']:.2f}, Structure: {weights['structure']:.2f}")
    
    # Save gallery
    gallery_dir = Path("gallery")
    gallery_dir.mkdir(exist_ok=True)
    gallery_path = gallery_dir / "person_gallery.json"
    
    fusion_engine.save_gallery(gallery, gallery_path)
    
    # Demo: Compute similarity between all pairs (if more than one signature)
    if len(gallery) > 1:
        print("\n" + "=" * 60)
        print("SIMILARITY MATRIX (Sample)")
        print("=" * 60)
        
        signatures = list(gallery.items())
        
        # Show a few comparisons
        for i in range(min(3, len(signatures))):
            for j in range(i+1, min(3, len(signatures))):
                key1, sig1 = signatures[i]
                key2, sig2 = signatures[j]
                
                sim, details, weights = fusion_engine.compute_similarity(sig1, sig2)
                
                print(f"\n{key1} vs {key2}:")
                print(f"  Overall Similarity: {sim:.3f}")
                print(f"  Face: {details['face']:.3f}" if details['face'] is not None else "  Face: N/A")
                print(f"  Deep Appearance: {details['deep_appearance']:.3f}" if details['deep_appearance'] is not None else "  Deep Appearance: N/A")
    
    print("\nOutputs:")
    print("1. Person gallery (JSON) in gallery/")
    print("2. Unified signatures combining face + appearance features")
    print("\nNext Steps:")
    print("Run gallery_matching.py to implement gallery matching and re-identification logic")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()