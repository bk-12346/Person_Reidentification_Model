"""
Person Re-Identification System - Environment Setup
================================================================
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install required packages"""
    print("=" * 60)
    print("INSTALLING DEPENDENCIES")
    print("=" * 60)
    
    dependencies = [
        "opencv-python",
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "filterpy",  # For Kalman filter in tracking
        "lap",  # For linear assignment problem in tracking
    ]
    
    print("\nInstalling dependencies...")
    for package in dependencies:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("\nAll dependencies installed successfully!")

def setup_directories():
    """Create necessary directories for the project"""
    print("\n" + "=" * 60)
    print("CREATING PROJECT DIRECTORIES")
    print("=" * 60)
    
    directories = [
        "data/videos",
        "data/sample_frames",
        "models",
        "output/videos",
        "output/visualizations",
        "gallery",  # For storing person signatures
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}/")
    
    print("\nProject structure created!")

def verify_setup():
    """Verify that the setup is complete"""
    print("\n" + "=" * 60)
    print("VERIFYING SETUP")
    print("=" * 60)
    
    # Check imports
    try:
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        import torchvision
        print(f"TorchVision version: {torchvision.__version__}")
        
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        print("\nAll core libraries imported successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    
    # Check directories
    required_dirs = ["data/videos", "models", "output", "gallery"]
    all_exist = all(os.path.exists(d) for d in required_dirs)
    
    if all_exist:
        print("All required directories exist!")
    else:
        print("Some directories are missing")
        return False
    
    # Check for video files
    video_dir = "data/videos"
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))] if os.path.exists(video_dir) else []
    
    if video_files:
        print(f"Found {len(video_files)} video file(s) in data/videos/")
        for vf in video_files:
            print(f"  - {vf}")
    else:
        print("No video files found in data/videos/")
        print("  Please download sample videos manually:")
        print("  1. Visit https://www.pexels.com/search/videos/people%20walking/")
        print("  2. Download 1-2 videos")
        print("  3. Save to data/videos/")
    
    return True

def main():
    """Run the complete environment setup"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Person Re-Identification System - Environment Setup  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        install_dependencies()
        setup_directories()
        verify_setup()
        
        print("\n" + "=" * 60)
        print("Environment setup complete!")
        print("=" * 60)
        print("\nNext Steps:")
        print("1. Download 2 videos from Pexels to data/videos/")
        print("   - Video 1: For initial tracking/training")
        print("   - Video 2: For testing re-identification")
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()