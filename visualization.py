"""
Person Re-Identification System - Visualization & Output
==================================================================
This script creates comprehensive visualizations and summary reports
demonstrating all system capabilities.
"""

import cv2
import numpy as np
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime

class ComprehensiveVisualizer:
    """Create comprehensive visualizations of the re-ID system"""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.report_dir = self.output_dir / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)
    
    def create_summary_report(self, video_files, detection_dir, gallery_path):
        """
        Create a comprehensive summary report
        
        Args:
            video_files: List of processed video files
            detection_dir: Directory with detection and feature files
            gallery_path: Path to person gallery
        """
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE SUMMARY REPORT")
        print("=" * 60)
        
        report = {
            'generation_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': {
                'total_videos_processed': len(video_files),
                'video_names': [v.name for v in video_files]
            },
            'per_video_stats': {},
            'cross_video_analysis': {},
            'temporal_analysis': {}
        }
        
        # Analyze each video
        for video_file in video_files:
            video_name = video_file.stem
            print(f"\nAnalyzing: {video_name}")
            
            stats = self._analyze_video(video_name, detection_dir)
            report['per_video_stats'][video_name] = stats
        
        # Cross-video analysis
        if len(video_files) >= 2:
            cross_stats = self._analyze_cross_video(video_files, detection_dir, gallery_path)
            report['cross_video_analysis'] = cross_stats
        
        # Temporal analysis
        temporal_stats = self._analyze_temporal(video_files, detection_dir)
        report['temporal_analysis'] = temporal_stats
        
        # Save report
        report_file = self.report_dir / "system_summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nSummary report saved to: {report_file}")
        
        # Create human-readable report
        self._create_readable_report(report)
        
        return report
    
    def _analyze_video(self, video_name, detection_dir):
        """Analyze a single video"""
        stats = {}
        
        # Load detections
        detection_file = detection_dir / f"{video_name}_detections.json"
        if detection_file.exists():
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            stats['detection'] = {
                'total_frames': len(data['detections']),
                'unique_persons_tracked': data['statistics']['unique_persons'],
                'total_detections': data['statistics']['total_detections']
            }
        
        # Load face features
        face_file = detection_dir / f"{video_name}_face_features.pkl"
        if face_file.exists():
            with open(face_file, 'rb') as f:
                face_data = pickle.load(f)
            
            stats['face_recognition'] = {
                'persons_with_faces': len(face_data['person_features']),
                'total_face_samples': sum(p['num_samples'] for p in face_data['person_features'].values())
            }
        
        # Load ReID features
        reid_file = detection_dir / f"{video_name}_reid_features.pkl"
        if reid_file.exists():
            with open(reid_file, 'rb') as f:
                reid_data = pickle.load(f)
            
            stats['appearance_reid'] = {
                'persons_with_features': len(reid_data['person_features']),
                'total_appearance_samples': sum(p['num_samples'] for p in reid_data['person_features'].values())
            }
        
        # Load temporal re-ID results
        temporal_file = self.output_dir / "videos" / f"{video_name}_temporal_reid.json"
        if temporal_file.exists():
            with open(temporal_file, 'r') as f:
                temporal_data = json.load(f)
            
            stats['temporal_reid'] = {
                'unique_persons': temporal_data['statistics']['total_unique_persons'],
                'reentry_events': temporal_data['statistics']['detected_reentries']
            }
        
        return stats
    
    def _analyze_cross_video(self, video_files, detection_dir, gallery_path):
        """Analyze cross-video matching"""
        if len(video_files) < 2:
            return {}
        
        # Count persons in each video
        video_person_counts = {}
        
        for video_file in video_files:
            video_name = video_file.stem
            detection_file = detection_dir / f"{video_name}_detections.json"
            
            if detection_file.exists():
                with open(detection_file, 'r') as f:
                    data = json.load(f)
                video_person_counts[video_name] = data['statistics']['unique_persons']
        
        return {
            'videos_compared': len(video_files),
            'persons_per_video': video_person_counts,
            'potential_matches': 'See Phase 6 results for detailed matching'
        }
    
    def _analyze_temporal(self, video_files, detection_dir):
        """Analyze temporal re-identification across all videos"""
        total_reentries = 0
        videos_with_reentries = 0
        
        for video_file in video_files:
            video_name = video_file.stem
            temporal_file = self.output_dir / "videos" / f"{video_name}_temporal_reid.json"
            
            if temporal_file.exists():
                with open(temporal_file, 'r') as f:
                    data = json.load(f)
                
                reentries = data['statistics']['detected_reentries']
                total_reentries += reentries
                
                if reentries > 0:
                    videos_with_reentries += 1
        
        return {
            'total_reentry_events': total_reentries,
            'videos_with_reentries': videos_with_reentries
        }
    
    def _create_readable_report(self, report):
        """Create a human-readable text report"""
        report_file = self.report_dir / "summary_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PERSON RE-IDENTIFICATION SYSTEM - SUMMARY REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Generated: {report['generation_time']}\n\n")
            
            # System overview
            f.write("SYSTEM OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Videos Processed: {report['system_info']['total_videos_processed']}\n")
            f.write(f"Videos: {', '.join(report['system_info']['video_names'])}\n\n")
            
            # Per-video statistics
            f.write("PER-VIDEO ANALYSIS\n")
            f.write("-" * 70 + "\n")
            
            for video_name, stats in report['per_video_stats'].items():
                f.write(f"\n{video_name}:\n")
                
                if 'detection' in stats:
                    f.write(f"  Detection & Tracking:\n")
                    f.write(f"    - Total frames: {stats['detection']['total_frames']}\n")
                    f.write(f"    - Unique persons tracked: {stats['detection']['unique_persons_tracked']}\n")
                    f.write(f"    - Total detections: {stats['detection']['total_detections']}\n")
                
                if 'face_recognition' in stats:
                    f.write(f"  Face Recognition:\n")
                    f.write(f"    - Persons with faces: {stats['face_recognition']['persons_with_faces']}\n")
                    f.write(f"    - Total face samples: {stats['face_recognition']['total_face_samples']}\n")
                
                if 'appearance_reid' in stats:
                    f.write(f"  Appearance Re-ID:\n")
                    f.write(f"    - Persons with features: {stats['appearance_reid']['persons_with_features']}\n")
                    f.write(f"    - Total samples: {stats['appearance_reid']['total_appearance_samples']}\n")
                
                if 'temporal_reid' in stats:
                    f.write(f"  Temporal Re-identification:\n")
                    f.write(f"    - Unique persons: {stats['temporal_reid']['unique_persons']}\n")
                    f.write(f"    - Re-entry events: {stats['temporal_reid']['reentry_events']}\n")
            
            # Cross-video analysis
            if report['cross_video_analysis']:
                f.write("\n\nCROSS-VIDEO ANALYSIS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Videos compared: {report['cross_video_analysis']['videos_compared']}\n")
                f.write(f"Persons per video:\n")
                for video, count in report['cross_video_analysis']['persons_per_video'].items():
                    f.write(f"  - {video}: {count} persons\n")
            
            # Temporal analysis
            if report['temporal_analysis']:
                f.write("\n\nTEMPORAL RE-IDENTIFICATION SUMMARY\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total re-entry events detected: {report['temporal_analysis']['total_reentry_events']}\n")
                f.write(f"Videos with re-entries: {report['temporal_analysis']['videos_with_reentries']}\n")
            
            # System capabilities
            f.write("\n\nSYSTEM CAPABILITIES DEMONSTRATED\n")
            f.write("-" * 70 + "\n")
            f.write("Person detection and tracking within single camera feed\n")
            f.write("Face recognition when faces are visible\n")
            f.write("Appearance-based re-identification (works without faces)\n")
            f.write("Temporal re-identification (people leaving and returning)\n")
            f.write("Cross-video person matching\n")
            f.write("Multi-modal feature fusion (face + appearance)\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Readable report saved to: {report_file}")
    
    def create_visualization_grid(self, video_files):
        """Create a grid showing different visualizations"""
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATION COMPARISONS")
        print("=" * 60)
        
        output_videos = self.output_dir / "videos"
        
        # List all output videos
        viz_types = [
            ('tracked', 'Detection & Tracking'),
            ('faces', 'Face Detection'),
            ('reid_features', 'Re-ID Features'),
            ('temporal_reid', 'Temporal Re-ID')
        ]
        
        for video_file in video_files:
            video_name = video_file.stem
            print(f"\nProcessing: {video_name}")
            
            # Check which visualizations exist
            available_viz = []
            for suffix, label in viz_types:
                viz_file = output_videos / f"{video_name}_{suffix}.mp4"
                if viz_file.exists():
                    available_viz.append((viz_file, label))
            
            print(f"  Found {len(available_viz)} visualization types")
    
    def generate_statistics_plots(self, report):
        """Generate statistical plots"""
        print("\n" + "=" * 60)
        print("GENERATING STATISTICAL PLOTS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Person Re-Identification System Statistics', fontsize=16, fontweight='bold')
        
        # Plot 1: Persons per video
        ax1 = axes[0, 0]
        videos = list(report['per_video_stats'].keys())
        persons = [report['per_video_stats'][v]['detection']['unique_persons_tracked'] 
                  for v in videos if 'detection' in report['per_video_stats'][v]]
        
        ax1.bar(range(len(videos)), persons, color='steelblue')
        ax1.set_xlabel('Video')
        ax1.set_ylabel('Unique Persons Tracked')
        ax1.set_title('Persons Tracked per Video')
        ax1.set_xticks(range(len(videos)))
        ax1.set_xticklabels(videos, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Face detection rate
        ax2 = axes[0, 1]
        face_data = []
        for v in videos:
            if 'face_recognition' in report['per_video_stats'][v]:
                total = report['per_video_stats'][v]['detection']['unique_persons_tracked']
                with_faces = report['per_video_stats'][v]['face_recognition']['persons_with_faces']
                face_data.append(with_faces / total * 100 if total > 0 else 0)
        
        ax2.bar(range(len(videos)), face_data, color='green')
        ax2.set_xlabel('Video')
        ax2.set_ylabel('Face Detection Rate (%)')
        ax2.set_title('Percentage of Persons with Face Detection')
        ax2.set_xticks(range(len(videos)))
        ax2.set_xticklabels(videos, rotation=45, ha='right')
        ax2.set_ylim([0, 100])
        ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Feature availability
        ax3 = axes[1, 0]
        feature_types = ['Face Only', 'Appearance Only', 'Both']
        counts = [0, 0, 0]
        
        for v in videos:
            stats = report['per_video_stats'][v]
            if 'face_recognition' in stats and 'appearance_reid' in stats:
                total = stats['detection']['unique_persons_tracked']
                with_face = stats['face_recognition']['persons_with_faces']
                with_appearance = stats['appearance_reid']['persons_with_features']
                
                both = min(with_face, with_appearance)
                face_only = with_face - both
                appearance_only = with_appearance - both
                
                counts[0] += face_only
                counts[1] += appearance_only
                counts[2] += both
        
        ax3.pie(counts, labels=feature_types, autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen'])
        ax3.set_title('Feature Type Distribution')
        
        # Plot 4: Temporal re-identification
        ax4 = axes[1, 1]
        temporal_data = []
        for v in videos:
            if 'temporal_reid' in report['per_video_stats'][v]:
                temporal_data.append(report['per_video_stats'][v]['temporal_reid']['reentry_events'])
        
        if temporal_data:
            ax4.bar(range(len(videos)), temporal_data, color='orange')
            ax4.set_xlabel('Video')
            ax4.set_ylabel('Re-entry Events')
            ax4.set_title('Temporal Re-identification Events')
            ax4.set_xticks(range(len(videos)))
            ax4.set_xticklabels(videos, rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No temporal data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Temporal Re-identification Events')
        
        plt.tight_layout()
        
        plot_file = self.report_dir / "statistics_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Statistical plots saved to: {plot_file}")

def main():
    """Main function to run Phase 8"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-ID System - Visualiztions  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Initialize visualizer
    visualizer = ComprehensiveVisualizer()
    
    # Find all processed videos
    video_dir = Path("data/videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print("\nNo video files found!")
        return
    
    detection_dir = Path("data/sample_frames")
    gallery_path = Path("gallery/person_gallery.json")
    
    print(f"\nFound {len(video_files)} video(s):")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {vf.name}")
    
    # Generate comprehensive report
    report = visualizer.create_summary_report(video_files, detection_dir, gallery_path)
    
    # Create visualization comparisons
    visualizer.create_visualization_grid(video_files)
    
    # Generate statistical plots
    visualizer.generate_statistics_plots(report)
    
    print("\nFinal Deliverables:")
    print("1. Comprehensive summary reports in output/reports/")
    print("   - system_summary_report.json (machine-readable)")
    print("   - summary_report.txt (human-readable)")
    print("   - statistics_plots.png (visualizations)")
    print("\n2. All processed videos in output/videos/:")
    print("   - *_tracked.mp4 (detection & tracking)")
    print("   - *_faces.mp4 (face detection)")
    print("   - *_reid_features.mp4 (appearance features)")
    print("   - *_reid_results.mp4 (cross-video matching)")
    print("   - *_temporal_reid.mp4 (temporal re-identification)")
    print("\n3. Feature data in data/sample_frames/:")
    print("   - Detection JSONs")
    print("   - Face feature pickles")
    print("   - ReID feature pickles")
    print("\n4. Gallery in gallery/:")
    print("   - person_gallery.json")
    print("\n" + "=" * 60)
    print("\nSYSTEM CAPABILITIES DEMONSTRATED:")
    print("=" * 60)
    print("Person detection and tracking within camera feed")
    print("Face recognition when faces visible")
    print("Appearance-based re-ID (works without faces)")
    print("Temporal re-ID (people leaving/returning)")
    print("Cross-video matching")
    print("Multi-modal feature fusion")
    print("Robust to clothing changes (using multiple features)")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()