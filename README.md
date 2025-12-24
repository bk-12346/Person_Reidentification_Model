Person Re-Identification System
===============================

A comprehensive computer vision system for recognizing and re-identifying people in CCTV video footage across time and different camera views.

Table of Contents
-----------------

-   Overview
-   System Capabilities
-   Architecture
-   Installation
-   Usage
-   Step-by-Step Implementation
-   Results and Deliverables
-   Technical Details

* * * * *

Overview
--------

This system implements a multi-modal person re-identification pipeline that can:

-   Track people within a single camera feed
-   Recognize people when they leave and re-enter the scene
-   Match people across different videos
-   Work even when faces are not visible or clothing changes
-   Utilize both face recognition and appearance-based features

**Use Case**: Data Science position case study demonstrating computer vision and ML skills

* * * * *

System Capabilities
-------------------

### Single Camera Tracking

-   Real-time person detection using YOLOv5
-   Frame-to-frame tracking using SORT algorithm
-   Persistent ID assignment within scenes

### Face Recognition

-   Face detection using Haar Cascades
-   Face embedding extraction using FaceNet (InceptionResnetV1)
-   512-dimensional face embeddings for matching

### Appearance-Based Re-identification

-   Deep appearance features using ResNet50 (2048-dim)
-   Color histogram features (96-dim HSV)
-   Body structure features (aspect ratio, size)
-   Works when faces are occluded or not visible

### Temporal Re-identification

-   Handles people leaving and returning to scene
-   Maintains persistent IDs across temporal gaps
-   Dynamic gallery management

### Cross-Video Matching

-   Match people across different camera feeds
-   Similarity-based matching with configurable thresholds
-   Multi-modal feature fusion

### Robust to Changes

-   Clothing changes handled through multiple feature types
-   Weighted fusion based on feature availability
-   Works with partial occlusions

* * * * *

Architecture
------------

```
Input Videos → Detection → Feature Extraction → Feature Fusion → Matching → Re-identification
                  ↓              ↓                    ↓              ↓             ↓
              YOLO+SORT    Face+Appearance      Multi-Modal      Gallery      Temporal
                                               Signatures       Matching      Re-entry

```

### Key Components

1.  **Detection & Tracking**: YOLOv5 + SORT
2.  **Face Recognition**: Haar Cascade + FaceNet
3.  **Appearance Features**: ResNet50 + Color Histograms
4.  **Feature Fusion**: Weighted combination based on availability
5.  **Gallery Management**: Dynamic database of person signatures
6.  **Matching Engine**: Cosine similarity with thresholds

* * * * *

Installation
------------

### Prerequisites

-   Python 3.10+
-   CUDA-capable GPU (optional, for faster processing)

### Setup

```
# Run environment setup script
python setup.py

```

This will install all required dependencies:

-   opencv-python
-   torch & torchvision
-   facenet-pytorch
-   numpy, scipy, scikit-learn
-   matplotlib
-   filterpy, lap

### Directory Structure

After setup:

```
project/
├── data/
│   ├── videos/              # Place input videos here
│   └── sample_frames/       # Extracted features stored here
├── models/                  # Model weights
├── output/
│   ├── videos/             # Output visualizations
│   └── reports/            # Summary reports
├── gallery/                # Person signature gallery
└── logs/                   # System logs

```

* * * * *

Usage
-----

### Quick Start

1.  **Download Videos**: Place 2 videos in `data/videos/`

    -   Recommended: Download from [Pexels](https://www.pexels.com/search/videos/people%20walking/)
2.  **Run All Files Sequentially**:

```
# Step 1: Setup environment
python setup.py

# Step 2: Detection and tracking
python detection.py

# Step 3: Face recognition
python recognition.py

# Step 4: Appearance features
python feature_extraction.py

# Step 5: Feature fusion
python feature_matching.py

# Step 6: Gallery matching
python gallery_matching.py

# Step 7: Temporal re-identification
python temporal.py

# Step 8: Final visualizations and reports
python visualization.py

```

1.  **Check Results**:
    -   Videos in `output/videos/`
    -   Reports in `output/reports/`

* * * * *

Step-by-Step Implementation
-----------------------------

### Step 1: Environment Setup

**Purpose**: Install dependencies and create project structure

**What I Did**:

-   Installed OpenCV, PyTorch, and specialized libraries
-   Created organized directory structure
-   Verified all dependencies are working

**Key Technologies**: pip, system setup

* * * * *

### Step 2: Person Detection & Tracking

**Purpose**: Detect people and maintain consistent IDs within frames

**What I Did**:

-   Implemented YOLOv5 for person detection (pre-trained on COCO)
-   Integrated SORT (Simple Online and Realtime Tracking) algorithm
-   Used Kalman filters for motion prediction
-   Implemented Hungarian algorithm for detection-to-track association

**How It Works**:

1.  YOLOv5 detects all persons in each frame (bounding boxes)
2.  SORT matches detections across frames using IoU (Intersection over Union)
3.  Each person gets a unique track ID while in the scene
4.  IDs maintained through temporary occlusions

**Key Technologies**: YOLOv5, SORT, Kalman Filter, Hungarian Algorithm

**Output**:

-   Tracked videos with bounding boxes and IDs
-   Detection JSON files with frame-by-frame data

* * * * *

### Step 3: Face Recognition

**Purpose**: Extract face embeddings for identity matching

**What We Did**:

-   Implemented Haar Cascade for fast face detection
-   Integrated FaceNet (InceptionResnetV1) for face embeddings
-   Extracted 512-dimensional face vectors
-   Aggregated multiple face samples per person

**How It Works**:

1.  Search for faces within person bounding boxes
2.  Extract and preprocess face regions (160x160)
3.  Generate embeddings using FaceNet (trained on VGGFace2)
4.  Average embeddings across multiple frames for robustness
5.  Store per-person face signatures

**Key Technologies**: Haar Cascade, FaceNet (InceptionResnetV1), VGGFace2

**Why This Works**:

-   FaceNet embeddings cluster by identity
-   Cosine similarity >0.6 indicates same person
-   Multiple samples increase reliability

**Output**:

-   Face feature files (.pkl) with embeddings
-   Visualization videos showing face detections

* * * * *

### Step 4: Body/Appearance Re-identification

**Purpose**: Enable re-identification when faces aren't visible

**What We Did**:

-   Implemented ResNet50 for deep appearance features (2048-dim)
-   Extracted HSV color histograms (96-dim)
-   Computed body structure features (4-dim)
-   Combined features for robust appearance signatures

**How It Works**:

1.  **Deep Features**: ResNet50 processes full body crops → 2048-dim vector
2.  **Color Features**: HSV histograms capture clothing colors → 96-dim vector
3.  **Structure Features**: Body proportions (aspect ratio, size) → 4-dim vector
4.  Average features across frames for stability

**Why Multiple Features**:

-   Deep features: Overall appearance patterns
-   Color features: Clothing color (robust to pose)
-   Structure features: Body proportions (robust to clothing changes)

**Key Technologies**: ResNet50, HSV color space, histogram analysis

**Output**:

-   ReID feature files (.pkl)
-   Visualization videos with feature indicators

* * * * *

### Step 5: Feature Fusion & Signature Creation

**Purpose**: Combine face and appearance into unified signatures

**What We Did**:

-   Created PersonSignature class to encapsulate all features
-   Implemented adaptive weight fusion based on feature availability
-   Built a gallery of person signatures
-   Implemented similarity computation across signatures

**How It Works**:

**Weight Distribution**:

-   **With Face**: Face (50%), Deep Appearance (25%), Color (15%), Structure (10%)
-   **Without Face**: Deep Appearance (50%), Color (30%), Structure (20%)

**Similarity Computation**:

1.  Compute cosine similarity for each feature type
2.  Apply adaptive weights based on both signatures
3.  Combine into overall similarity score (0-1)

**Why Adaptive Fusion**:

-   Prioritizes most reliable features
-   Handles missing face data gracefully
-   Balances multiple modalities

**Key Technologies**: Feature normalization, cosine similarity, weighted fusion

**Output**:

-   Person gallery (JSON) with unified signatures
-   Feature weight distributions per person

* * * * *

### Step 6: Gallery Management & Matching

**Purpose**: Match detected persons against known gallery

**What We Did**:

-   Implemented gallery-based matching system
-   Set similarity threshold for match decisions (default: 0.6)
-   Tested cross-video person matching
-   Generated detailed match reports

**How It Works**:

1.  Load gallery of known persons (from Video 1)
2.  Extract features from query persons (Video 2)
3.  Compute similarity against all gallery entries
4.  Select best match above threshold
5.  If no match → new person

**Matching Logic**:

```
for query_person in video2:
    best_match = None
    best_score = 0

    for gallery_person in gallery:
        score = compute_similarity(query_person, gallery_person)
        if score > best_score:
            best_score = score
            best_match = gallery_person

    if best_score >= threshold:
        return best_match  # Re-identified!
    else:
        return NEW_PERSON  # Not in gallery

```

**Key Technologies**: Similarity search, threshold-based decisions

**Output**:

-   Cross-video matching results
-   Visualization with color-coded matches (green=match, red=new)

* * * * *

### Step 7: Temporal Re-identification

**Purpose**: Handle people leaving and returning within same video

**What We Did**:

-   Implemented dynamic temporal gallery
-   Tracked active persons vs historical persons
-   Detected re-entry events when people return
-   Maintained persistent IDs across temporal gaps

**How It Works**:

**Gallery States**:

-   **Active**: Currently visible in scene
-   **Historical**: Left scene but stored for re-identification

**Re-entry Detection**:

1.  Person leaves scene → moved to historical gallery (after 150 frames / ~5 sec)
2.  New detection appears → extract features
3.  Compare against historical gallery
4.  If similarity > threshold → RE-IDENTIFIED (same persistent ID)
5.  If similarity < threshold → NEW PERSON (new persistent ID)

**Example Timeline**:

```
Frame 1-100:   Person A appears → Track ID 1, Persistent ID 1
Frame 101-200: Person A leaves → Moved to historical
Frame 201:     New detection → Extract features
               Compare to historical → Match found!
               Assign: Track ID 2, Persistent ID 1 (same person)

```

**Key Technologies**: Dynamic gallery management, temporal windowing

**Output**:

-   Temporal re-ID videos (yellow boxes for re-entries)
-   Re-entry event logs
-   Persistent ID tracking

* * * * *

### Step 8: Visualization & Output

**Purpose**: Generate comprehensive reports and visualizations

**What We Did**:

-   Created summary reports (JSON + text)
-   Generated statistical plots with matplotlib
-   Compiled all metrics and results
-   Produced final deliverables

**Reports Include**:

-   Per-video statistics (detections, faces, features)
-   Cross-video analysis
-   Temporal re-identification summary
-   Feature availability breakdown
-   System capability checklist

**Visualizations**:

-   Bar charts: Persons per video, detection rates
-   Pie chart: Feature type distribution
-   Statistical summaries

**Key Technologies**: JSON, matplotlib, report generation

**Output**:

-   system_summary_report.json
-   summary_report.txt
-   statistics_plots.png

* * * * *

Results and Deliverables
------------------------

### Generated Files

**Videos** (`output/videos/`):

-   `*_tracked.mp4`: Basic detection and tracking
-   `*_faces.mp4`: Face detection overlays
-   `*_reid_features.mp4`: Appearance features
-   `*_reid_results.mp4`: Cross-video matching
-   `*_temporal_reid.mp4`: Temporal re-identification with re-entries

**Data** (`data/sample_frames/`):

-   `*_detections.json`: Frame-by-frame detection data
-   `*_face_features.pkl`: Face embeddings per person
-   `*_reid_features.pkl`: Appearance features per person
-   `*_temporal_reid.json`: Temporal re-ID events

**Gallery** (`gallery/`):

-   `person_gallery.json`: Complete person signature database

**Reports** (`output/reports/`):

-   `system_summary_report.json`: Complete system metrics
-   `summary_report.txt`: Human-readable summary
-   `statistics_plots.png`: Visual statistics

* * * * *

Technical Details
-----------------

### Model Specifications

| Component | Model | Input Size | Output Dim | Pre-training |
| --- | --- | --- | --- | --- |
| Detection | YOLOv5s | 640x640 | Bboxes | COCO |
| Face Detection | Haar Cascade | Variable | Bboxes | OpenCV |
| Face Embedding | InceptionResnetV1 | 160x160 | 512 | VGGFace2 |
| Appearance | ResNet50 | 256x128 | 2048 | ImageNet |
| Color | HSV Histogram | Variable | 96 | N/A |

### Feature Fusion Weights

**Default Configuration**:

-   Face available: Face (50%), Deep (25%), Color (15%), Structure (10%)
-   Face unavailable: Deep (50%), Color (30%), Structure (20%)

### Thresholds

-   **Face matching**: 0.6 (60% similarity)
-   **Cross-video matching**: 0.6 (60% similarity)
-   **Temporal re-ID**: 0.65 (65% similarity, stricter for temporal gaps)
-   **Temporal window**: 150 frames (~5 seconds at 30fps)

### Performance Considerations

**Processing Speed** (approximate, CPU):

-   Detection: ~15-20 FPS
-   Face extraction: ~10-15 FPS
-   Appearance features: ~5-10 FPS
-   Matching: Real-time

**GPU Acceleration**:

-   All deep learning models support CUDA
-   3-5x speedup with GPU

* * * * *

System Validation
-----------------

### How to Verify System Works

1.  **Tracking**: Same person maintains ID while in frame
2.  **Face Recognition**: Green boxes appear over detected faces
3.  **Temporal Re-ID**: Yellow boxes when person returns (check frame numbers)
4.  **Cross-Video**: Matched persons show in gallery_matching.py results
5.  **Feature Fusion**: Check `person_gallery.json` for multi-modal signatures

### Expected Performance

-   **Detection Accuracy**: >90% for clearly visible persons
-   **Face Detection Rate**: 60-80% (depends on video quality and angles)
-   **Cross-Video Matching**: 70-85% for same persons in similar conditions
-   **Temporal Re-ID**: 75-90% for re-entries within 30 seconds

* * * * *

Troubleshooting
---------------

### Common Issues

**Issue**: "No video files found"

-   **Solution**: Ensure videos are in `data/videos/` with .mp4 or .avi extension

**Issue**: "CUDA out of memory"

-   **Solution**: System will automatically fall back to CPU

**Issue**: "No faces detected"

-   **Solution**: Normal if people face away from camera; appearance features still work

**Issue**: "Low matching scores"

-   **Solution**: Adjust thresholds in respective scripts

* * * * *

Citation and Credits
--------------------

### Libraries and Models Used

-   **YOLOv5**: Ultralytics (Glenn Jocher)
-   **FaceNet**: David Sandberg / Pytorch implementation by Tim Esler
-   **SORT**: Alex Bewley et al.
-   **ResNet**: Kaiming He et al. / torchvision
-   **OpenCV**: Intel Corporation and contributors

### References

1.  Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement
2.  Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition
3.  Bewley, A., et al. (2016). Simple Online and Realtime Tracking
4.  He, K., et al. (2016). Deep Residual Learning for Image Recognition
5.  Zheng, L., et al. (2015). Scalable Person Re-identification: A Benchmark

* * * * *

License
-------

This project is created for educational and case study purposes.

* * * * *

Contact
-------

For questions about implementation or results, please refer to the generated reports in `output/reports/`.


