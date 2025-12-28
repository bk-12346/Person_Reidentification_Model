##### detection.py #####
### 1. YOLO ###
# single stage object detector that predicts bounding boxes and class probabilities directlt from images in one pass
## Better because:
# -> real-time -> pre-trained on COCO, so already knows 'person' class -> easy to use, PyTorch Hub

self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
self.model.conf = conf_threshold  # Confidence threshold (e.g., 0.5)
self.model.classes = [0]  # Only detect persons (class 0 in COCO)

# Detection
results = self.model(frame)
detections = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, class]

## Expalanation:
# used YOLOv5s - the 's' means small/fast variant
# It's pre-trained on COCO dataset which has 'person' as class 0
# filter for only person detections and set a confidence threshold of 0.5 to reduce false positives. 
# The model returns bounding boxes in [x1, y1, x2, y2] format - top-left and bottom-right corners.
## Key parameters ##
# conf_threshold=0.5`: Only accept detections with >50% confidence
# classes=[0]`: Filter for person class only

### 2. SORT Tracking Algorithm (Simple Online and Realtime Tracking)**
# SORT associates detections across frames to maintain consistent IDs. 
# It uses **Kalman Filter** for motion prediction and **Hungarian Algorithm** for matching.
## The tracking pipeline:
# Frame N detections → Predict positions → Match to existing tracks → Update tracks

## KALMAN FILTER: Predicts where a person will be in the next frame based o their velocity
class KalmanBoxTracker:
    def __init__(self, bbox):
        # State: [x, y, s, r, vx, vy, vs]
        # x, y = center position of BB
        # s = scale (area) of BB
        # r = aspect ratio
        # vx, vy, vs = velocities
        
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([...])  # State transition matrix
        self.kf.H = np.array([...])  # Measurement matrix

## IOU Metric: measures overlap between 2 BB
# 1.0 means perfect overlap, 0.0 means no overlap
# use IoU of 0.3 as threshold - if a detection and predicted track overlap by >30%, they're likely the same person

# Visual example:
# Box A:  [10, 10, 50, 50]  (area = 1600)
# Box B:  [30, 30, 70, 70]  (area = 1600)
# Intersection: [30, 30, 50, 50] (area = 400)
# Union: 1600 + 1600 - 400 = 2800
# IoU: 400/2800 = 0.14

def iou_batch(bb_test, bb_gt):
    # Compute intersection
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    intersection = w * h
    
    # IoU = intersection / union
    union = area1 + area2 - intersection
    return intersection / union

## Hungarian Algorithm: Finds optimal assignment between detections and tracks to minimize cost
# The problem: -> Frame N has 3 existing tracks, -> Frame N+1 has 4 new detections, -> Which detection matches which track?
# The Hungarian algorithm solves the assignment problem optimally. 
# Given N detections and M tracks, it finds the best one-to-one matching that maximizes total IoU. 
# Scipy's linear_sum_assignment implements this in O(n³) time

# Example:
# Cost Matrix (IoU):
#            Track1  Track2  Track3
# Detection1   0.7    0.1    0.2
# Detection2   0.2    0.8    0.1
# Detection3   0.1    0.2    0.7

# Hungarian assigns:
# Detection1 → Track1 (0.7)
# Detection2 → Track2 (0.8)
# Detection3 → Track3 (0.7)

def associate_detections_to_trackers(self, detections, trackers):
    # Compute IoU cost matrix (higher IoU = lower cost)
    iou_matrix = iou_batch(detections, trackers)
    
    # Hungarian algorithm finds optimal assignment
    matched_indices = linear_sum_assignment(-iou_matrix)  # Negative for max
    
    # Filter matches below threshold
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] >= self.iou_threshold:
            matches.append(m)
    
    return matches, unmatched_detections, unmatched_trackers

### SORT UPDATE CYCLE ###
# SORT runs in a loop each frame. 
# First, it predicts where existing people should be using Kalman filters. 
# Then it matches new detections to predictions using IoU and Hungarian algorithm. 
# Matched tracks get updated, unmatched detections become new tracks, and tracks not seen for 5 frames get deleted. 
# This gives consistent IDs while people are visible.

# Key Parameters
max_age = 5          # Delete track after 5 frames without detection
min_hits = 1         # Assign ID after 1 detection (immediate)
iou_threshold = 0.3  # Minimum overlap for matching

# Cycle
def update(self, dets):
    # 1. PREDICT: Where will existing tracks be?
    for tracker in self.trackers:
        predicted_pos = tracker.predict()  # Kalman prediction
    
    # 2. MATCH: Which detections match which tracks?
    matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
        detections, predicted_positions
    )
    
    # 3. UPDATE: Update matched tracks with new measurements
    for match in matched:
        self.trackers[match[1]].update(detections[match[0]])
    
    # 4. CREATE: New tracks for unmatched detections
    for i in unmatched_dets:
        new_tracker = KalmanBoxTracker(detections[i])
        self.trackers.append(new_tracker)
    
    # 5. DELETE: Remove lost tracks (not seen for max_age frames)
    self.trackers = [t for t in self.trackers 
                     if t.time_since_update < self.max_age]
    
    return active_tracks
####################################################################################################################
####################################################################################################################

##### recognition.py #####
### 1. Face Detection with Haar Cascades
# Haar Cascades are classical CV features that detect faces using a cascade of simple classifiers
# -> Haar Cascades use simple rectangle features to detect faces. 
# -> The cascade is a series of classifiers - if a region passes all stages, it's classified as a face. 
# -> The parameters control sensitivity: scaleFactor=1.1 means we check the image at multiple scales (1.0x, 1.1x, 1.21x...) to find faces of different sizes
# -> minNeighbors=5 means we need 5 overlapping detections to confirm a face, reducing false positives

# Key Parameters:
scaleFactor=1.1     #Check image at 10% size increments
minNeighbors=5      #Require 5 overlapping detections (higher = fewer false positives)
minSize=(30, 30)    #Ignore faces smaller than 30x30 pixels

### 2. Face Embeddings with FaceNet
# FaceNet maps face images to a 512-dimensional vector space where distance represents similarity
# -> Same person → close embeddings (small distance)
# -> Different people → far embeddings (large distance)
# ->> FaceNet uses triplet loss during training. 
# ->> For each anchor face, it finds a positive example (same person) and negative example (different person). 
# ->> The loss encourages the network to make same-person distance smaller than different-person distance by at least a margin. 
# ->> This creates an embedding space where cosine similarity directly measures identity similarity.

### 3. Triplet Loss Training
# FaceNet was trained to minimize distance between same-person faces (anchor-positive) and maximize distance to different people (anchor-negative).
# Loss = max(0, ||anchor - positive||² - ||anchor - negative||² + margin)

# **Example**:
# Anchor:   Person A, Photo 1
# Positive: Person A, Photo 2  (same person)
# Negative: Person B, Photo 1  (different person)
# Goal: dist(A1, A2) < dist(A1, B1) + margin

# Load pre-trained FaceNet
from facenet_pytorch import InceptionResnetV1
self.face_model = InceptionResnetV1(pretrained='vggface2').eval()

# Preprocessing
self.face_transform = transforms.Compose([
    transforms.Resize((160, 160)),  # FaceNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Extract embedding
def extract_face_embedding(self, frame, face_bbox):
    face_img = frame[y1:y2, x1:x2]  # Crop face
    face_tensor = self.face_transform(face_img)
    
    with torch.no_grad():
        embedding = self.face_model(face_tensor)  # → 512-dim vector
    
    return embedding.cpu().numpy().flatten()

### 4. Cosine Similarity for Matching
# Why cosine over euclidean distance:
# -> Scale invariant: Only cares about direction, not magnitude
# -> Bounded: Returns values in [-1, 1]
# -> Intuitive: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite

# ->> Cosine similarity measures the angle between two vectors, not their magnitude. 
# ->> For face embeddings, direction in the 512-D space represents identity, so cosine similarity is perfect. 
# ->> Values above 0.6 typically indicate the same person. 
# ->> I normalize embeddings to unit length before computing the dot product.

# **Visual intuition**:
# Same person:     cos(θ) ≈ 0.8-0.95
# Similar people:  cos(θ) ≈ 0.5-0.7
# Different people: cos(θ) ≈ 0.2-0.5

def compute_similarity(self, embedding1, embedding2):
    # Normalize to unit vectors
    emb1_norm = embedding1 / np.linalg.norm(embedding1)
    emb2_norm = embedding2 / np.linalg.norm(embedding2)
    
    # Cosine similarity = dot product of normalized vectors
    similarity = np.dot(emb1_norm, emb2_norm)
    
    return float(similarity)  # Range: [-1, 1], typically [0, 1] for faces

### 5. Aggregating Multiple Face Samples
# ->> I collect face embeddings from multiple frames and average them. 
# ->> This is valid because FaceNet's embedding space is approximately linear - the average of same-person embeddings is still close to that person's embedding. 
# ->> This makes the signature more robust to pose variations, lighting changes, and occasional misdetections.

# Why averaging works:
# -> 1. Reduces noise: Variations from lighting, angle, expression get averaged out
# -> 2. More robust: Single bad frame doesn't dominate
# -> 3. Stays in embedding space: Linear combination of faces is still a face

### 6. Handling Missing Faces:
# ->> Face detection doesn't work every frame - people turn away, look down, or are partially occluded. 
# ->> I detect faces within person bounding boxes and only extract embeddings when faces are found. 
# ->> If multiple faces are detected in one person's box (rare), I take the largest one as it's most likely their actual face. 
# ->> Missing faces are fine because I aggregate across frames and also have appearance-based features in feature_extraction.py.

faces = self.detect_faces(frame, person_bbox)

if faces:
    # Extract and store embedding
    largest_face = max(faces, key=lambda f: f[2] * f[3])
    embedding = self.extract_face_embedding(frame, largest_face)
    self.person_face_embeddings[track_id].append(...)
else:
    # No face detected - skip this frame
    pass

### 7. Search Region Optimization
# ->> Instead of detecting faces in the entire frame, I crop to each person's bounding box first.
# ->> This is faster since Haar Cascades scale with image area.
# ->> More importantly, it prevents associating faces with the wrong person - if person A's box contains person B's face in the background, we'd get a mismatch. 
# ->> By searching within boxes, each face is correctly associated with its person.

def detect_faces(self, frame, person_bbox=None):
    if person_bbox is not None:
        x1, y1, x2, y2 = person_bbox
        search_region = frame[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        search_region = frame
        offset = (0, 0)
    
    # Detect in search region
    faces = self.face_cascade.detectMultiScale(search_region, ...)
    
    # Adjust coordinates back to full frame
    adjusted_faces = [[x + offset[0], y + offset[1], w, h] 
                      for (x, y, w, h) in faces]

### 8. Face Preprocessing
# ->> Preprocessing is critical for neural networks. 
# ->> FaceNet was trained on 160x160 images normalized to [-1, 1].
# ->> The normalization formula is: output = (input - mean) / std. 
# ->> With mean=0.5 and std=0.5, an input of 1.0 becomes (1-0.5)/0.5 = 1.0, and 0.0 becomes (0-0.5)/0.5 = -1.0. 
# ->> This matches the training distribution.

self.face_transform = transforms.Compose([
    transforms.ToPILImage(),           # OpenCV → PIL,              torchvision transforms work on PIL images
    transforms.Resize((160, 160)),     # FaceNet expects 160x160,   FaceNet input is exactly 160×160
    transforms.ToTensor(),             # → [0,1] tensor,            Convert to PyTorch tensor, scales to [0, 1]
    transforms.Normalize(              # → [-1,1] range,            FaceNet was trained on [-1, 1] range
        mean=[0.5, 0.5, 0.5], 
        std=[0.5, 0.5, 0.5]
    )
])
############################################################################################################################
###################################################################################################################################

##### feature_extraction.py #####
### 1. Deep Appearance with ResNet50
# ->> ResNet50 is a 50-layer convolutional network pre-trained on ImageNet. 
# ->> I remove the final classification layer and use it as a feature extractor. 
# ->> The network was trained on 1000 object classes, so it learned general visual features like textures, colors, and shapes. 
# ->> These transfer well to person appearance. 
# ->> The output is a 2048-dimensional vector that captures the person's overall appearance.
# # The model outputs 2048-dimensional features
# Shape: [batch_size, 2048, 1, 1] → flatten to [2048]

# **What happens inside ResNet50**:
# Input (256×128×3) 
#   → Conv layers (learn low-level features: edges, textures)
#   → Residual blocks (learn mid-level: patterns, shapes)
#   → Deeper blocks (learn high-level: clothing types, body structure)
#   → Global average pooling
#   → 2048-dimensional feature vector

# Load ResNet50 and remove final classification layer
self.appearance_model = models.resnet50(pretrained=True)
self.appearance_model = nn.Sequential(*list(self.appearance_model.children())[:-1])
self.appearance_model.eval()

### 2. Residual Connections
# The vanishing gradient problem: Deep networks are hard to train because gradients get smaller as they backpropagate.
# ResNet's solution: Add skip connections that allow gradients to flow directly
# **Visual representation**:
# x ──┬──→ Conv → Conv ──┬──→ output
#     │                   │
#     └───────────────────┘
#     (identity shortcut)

# ->> ResNet introduced residual connections - shortcuts that bypass layers.
# ->> Instead of learning a mapping H(x), layers learn the residual F(x) = H(x) - x. 
# ->> This makes training easier because if F(x) is zero, the layer just passes the input through unchanged.
# ->> It's why we can train very deep networks (50, 101, even 152 layers) without vanishing gradients.

# Traditional layer
output = F(x)
# Residual block
output = F(x) + x  # Add input to output

### 3. Image Pre-Processing for Re-ID
# ->> Person ReID typically uses 256×128 resolution - it's tall and narrow to match human body proportions. 
# ->> This aspect ratio preserves important details like clothing patterns while being computationally efficient. 
# ->> I normalize using ImageNet statistics because ResNet was trained on ImageNet, so the input distribution must match.

# **Why 256×128 resolution**:
# -> Aspect ratio ~2:1: Matches human body proportions
# -> Standard in ReID: Most ReID datasets use this size
# -> Computational efficiency: Smaller than original but preserves detail

# **Why ImageNet normalization**:
# -> ResNet was trained on ImageNet with these statistics
# -> Must use same normalization for inference

self.transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),  # Standard ReID aspect ratio
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])

### 5. Color Histogram Features
# ->> Color histograms count pixel frequencies in different color ranges. 
# ->> I use HSV space because hue is more robust to lighting changes than RGB. 
# ->> With 32 bins per channel, I get a 96-dimensional vector that captures the color distribution. 
# ->> For example, someone wearing a red shirt will have high values in red hue bins. 
# ->> I normalize histograms so they're comparable regardless of person size.

# **Why color matters**:
# - Clothing color is distinctive
# - Relatively invariant to pose changes
# - Simple but effective

# **HSV color space** (vs RGB):
# RGB: Red, Green, Blue (how monitors display)
# HSV: Hue, Saturation, Value (how humans perceive)

# HSV advantages:
# - Hue: Pure color (red, blue, green) - robust to lighting
# - Saturation: Color intensity
# - Value: Brightness

# **What a histogram represents**:
# Histogram bins (simplified, 8 bins):
# Bin 0: Count of pixels with hue 0-22.5°
# Bin 1: Count of pixels with hue 22.5-45°
# ...
# Bin 7: Count of pixels with hue 157.5-180°

# Example person:
# - Blue shirt: Many pixels in blue hue bins
# - Black pants: Many pixels in low saturation bins
# - White shoes: Many pixels in high value bins

def extract_color_histogram(self, frame, bbox, bins=32):
    person_img = frame[y1:y2, x1:x2]
    
    # Convert to HSV
    hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
    
    # Compute histogram for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])  # Hue: 0-180
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])  # Sat: 0-255
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])  # Val: 0-255
    
    # Normalize each histogram
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    # Concatenate: 32 + 32 + 32 = 96 dimensions
    color_hist = np.concatenate([hist_h, hist_s, hist_v])
    
    return color_hist

### 6. Body Structure Features
# ->> Structure features capture body geometry. 
# ->> Aspect ratio (height/width) reflects body type - taller people have higher ratios. 
# ->> I normalize by frame size to handle different video resolutions. 
# ->> These 4 features are the weakest signal but help when clothing changes completely. 
# ->> A tall thin person will have different proportions than a short wide person regardless of what they wear.

def extract_body_structure(self, frame, bbox):
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    # Aspect ratio (height/width) - relatively constant per person
    aspect_ratio = height / (width + 1e-6)
    
    # Relative size (normalized by frame size)
    rel_width = width / frame.shape[1]
    rel_height = height / frame.shape[0]
    rel_area = (width * height) / (frame.shape[0] * frame.shape[1])
    
    return np.array([aspect_ratio, rel_width, rel_height, rel_area])

### 7. Feature Aggregation Across Frames
# ->> I extract features from every frame where a person appears and average them. 
# ->> This is crucial because a single frame might have poor lighting, motion blur, or an unusual pose. 
# ->> Averaging across 10-100 frames gives a robust signature that represents the person's typical appearance. 
# ->> For ResNet features, this works because the embedding space is approximately linear.

for frame_idx, frame_det in enumerate(detections):
    for detection in frame_det['detections']:
        track_id = detection['track_id']
        
        # Extract features
        deep_feat = self.extract_deep_features(frame, bbox)
        color_feat = self.extract_color_histogram(frame, bbox)
        structure_feat = self.extract_body_structure(frame, bbox)
        
        # Accumulate
        self.person_features[track_id]['deep_features'].append(deep_feat)
        self.person_features[track_id]['color_features'].append(color_feat)
        self.person_features[track_id]['structure_features'].append(structure_feat)

# After processing all frames
avg_deep = np.mean(features['deep_features'], axis=0)
avg_color = np.mean(features['color_features'], axis=0)
avg_structure = np.mean(features['structure_features'], axis=0)

### 8. Multi-scale Feature Vector
# ->> I use multi-scale features because different aspects matter in different situations.
# ->> Deep features from ResNet capture complex patterns like clothing textures and styles. 
# ->> Color histograms are simpler but very robust to pose changes. 
# ->> Structure features are the weakest but help when clothing changes completely. 
# ->> By combining all three, the system is more robust than using any single feature type.

# Complete Feature Vector:
# Person Appearance Signature:
# ├── Deep features: 2048 dim (high-level appearance)
# ├── Color features: 96 dim (color distribution)
# └── Structure features: 4 dim (body geometry)
# Total: 2148 dimensions
##########################################################################################################################
#########################################################################################################################

##### feature_extraction.py #####
# Different features have different strengths.
# -> Face embeddings are most distinctive but only available 60-70% of the time. 
# -> Appearance features always work but are less unique. 
# -> I implemented adaptive fusion that weighs features based on availability and quality. 
# -> When faces are visible, they get 50% weight. 
# -> Without faces, appearance features are weighted more heavily.

### 1. Person Signature Class
# -> I created a PersonSignature class to encapsulate all features for one person. 
# -> It stores face embeddings, appearance features, and metadata like how many samples we have. 
# -> The class has methods to compute fusion weights and similarities. 
# -> This object-oriented approach makes the code cleaner and easier to maintain than passing around dictionaries.

class PersonSignature:
    def __init__(self, person_id, video_source):
        self.person_id = person_id
        self.video_source = video_source
        
        # Face features
        self.face_embedding = None      # 512-dim
        self.face_available = False
        self.face_confidence = 0.0
        
        # Appearance features
        self.deep_features = None       # 2048-dim
        self.color_features = None      # 96-dim
        self.structure_features = None  # 4-dim
        self.appearance_available = False
        
        # Metadata
        self.num_face_samples = 0
        self.num_appearance_samples = 0

### 2. Adaptive Weight Fusion
# -> The fusion weights adapt based on what's available. 
# -> Face gets up to 50% weight, modulated by confidence (based on sample count).
# -> If we only have 3 face samples versus 100 appearance samples, face gets less weight. 
# -> The remaining weight is distributed to appearance features with deep features getting the most (50%), then color (30%), then structure (20%). 
# -> This hierarchy reflects their relative discriminative power.

def get_fusion_weights(self):
    weights = {
        'face': 0.0,
        'deep_appearance': 0.0,
        'color': 0.0,
        'structure': 0.0
    }
    
    # If face available, it gets priority
    if self.face_available:
        weights['face'] = 0.5 * self.face_confidence
    
    # Remaining weight distributed to appearance
    if self.appearance_available:
        remaining_weight = 1.0 - weights['face']
        weights['deep_appearance'] = 0.5 * remaining_weight
        weights['color'] = 0.3 * remaining_weight
        weights['structure'] = 0.2 * remaining_weight
    
    # Normalize to sum to 1.0
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return weights

# **Example scenarios**:
# Scenario 1: Face available with 10 samples
# face_confidence = min(1.0, 10/10) = 1.0
# weights:
#   face: 0.5 × 1.0 = 0.50
#   remaining: 0.50
#   deep: 0.5 × 0.50 = 0.25
#   color: 0.3 × 0.50 = 0.15
#   structure: 0.2 × 0.50 = 0.10

# **Scenario 2: No face, only appearance**
# face: 0.0
# remaining: 1.0
# weights:
#   face: 0.00
#   deep: 0.5 × 1.0 = 0.50
#   color: 0.3 × 1.0 = 0.30
#   structure: 0.2 × 1.0 = 0.20

# **Scenario 3: Face with only 3 samples (low confidence)**
# face_confidence = min(1.0, 3/10) = 0.3
# weights:
#   face: 0.5 × 0.3 = 0.15
#   remaining: 0.85
#   deep: 0.5 × 0.85 = 0.425
#   color: 0.3 × 0.85 = 0.255
#   structure: 0.2 × 0.85 = 0.170

### 3. Feature Confidence Estimation
# -> Confidence reflects data quality. 
# -> One face detection could be a false positive or poor angle. 
# -> Ten detections means we consistently detected a face across many frames, so we trust it more. 
# -> I use a simple linear ramp capping at 10 samples - more than that doesn't add much information.
# -> This confidence modulates the face weight, so unreliable faces don't dominate the signature.

self.face_confidence = min(1.0, num_samples / 10.0)

# **Why confidence matters**:
# - 1 face sample: Could be a glitch or poor angle → low confidence
# - 10+ face samples: Strong evidence → high confidence

# **The formula**:
# confidence = min(1.0, samples / 10)
# 1 sample  → 0.1 confidence
# 5 samples → 0.5 confidence
# 10 samples → 1.0 confidence
# 20 samples → 1.0 confidence (capped)

### 4. Similarity Computation Between Signatures
# -> Similarity computation uses weighted cosine similarity across all features. 
# -> Crucially, I average the fusion weights from both signatures. 
# -> If person A has a face but person B doesn't, we can't rely on face matching, so face weight is reduced. 
# -> This makes comparisons fair regardless of feature availability. 
# -> The total similarity is the weighted sum of individual feature similarities.

# **Why average weights from both signatures**:
# Person A: Has face (weight=0.5) + appearance
# Person B: No face (weight=0.0) + appearance

# Average weights:
#   face: (0.5 + 0.0) / 2 = 0.25
#   appearance: Scaled accordingly

# This is fair - we can't match faces if only one person has them,
# so we reduce face weight to 0.25 and rely more on appearance.

def compute_similarity(self, sig1, sig2):
    total_sim = 0.0
    details = {}
    
    # Average weights from both signatures
    weights = {
        k: (sig1.fusion_weights[k] + sig2.fusion_weights[k]) / 2
        for k in sig1.fusion_weights.keys()
    }
    
    # Face similarity (if both have faces)
    if sig1.face_available and sig2.face_available:
        face_sim = self._cosine_similarity(sig1.face_embedding, sig2.face_embedding)
        details['face'] = float(face_sim)
        total_sim += weights['face'] * face_sim
    
    # Appearance similarities (if both have appearance)
    if sig1.appearance_available and sig2.appearance_available:
        deep_sim = self._cosine_similarity(sig1.deep_features, sig2.deep_features)
        color_sim = self._cosine_similarity(sig1.color_features, sig2.color_features)
        structure_sim = self._cosine_similarity(sig1.structure_features, sig2.structure_features)
        
        total_sim += weights['deep_appearance'] * deep_sim
        total_sim += weights['color'] * color_sim
        total_sim += weights['structure'] * structure_sim
    
    return total_sim, details

### 5. Cosine Similarity Implementation
# -> Cosine similarity measures the angle between vectors, ignoring magnitude. 
# -> I first normalize both vectors to unit length (divide by L2 norm), then compute their dot product.
# -> The result is between -1 and 1, but for our features it's typically 0 to 1.
# -> Values above 0.6 indicate the same person. 
# -> I add epsilon (1e-8) to prevent division by zero edge cases.

# Why add 1e-8
# -> Prevent division by zero if a vector is all zeros
# -> Numerical stability for very small values

# **Mathematical background**:
# Cosine similarity = (A · B) / (||A|| × ||B||)

# Where:
# A · B = sum of element-wise products (dot product)
# ||A|| = sqrt(sum of squared elements) (L2 norm)

# For unit vectors (||A|| = ||B|| = 1):
# Cosine similarity = A · B (just the dot product)

@staticmethod
def _cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0
    
    # L2 normalize (convert to unit vectors)
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
    
    # Dot product of unit vectors = cosine of angle
    return np.dot(vec1_norm, vec2_norm)

### 6. Gallery Management
# The gallery is a database of person signatures indexed by (video_name, person_id). 
# After processing videos, I build the gallery by loading face and appearance features for each person and creating their signature. 
# The gallery is saved as JSON so it persists across runs. 
# Later steps query this gallery to find matches for new detections.

# Key operations:
# 1. Build: Create signatures from all processed videos
# 2. Save: Serialize to JSON for persistence
# 3. Load: Deserialize for matching
# 4. Query: Find best match for a new signature

gallery = {
    ('video1', 1): PersonSignature(...),  # Video 1, Person ID 1
    ('video1', 2): PersonSignature(...),  # Video 1, Person ID 2
    ('video2', 1): PersonSignature(...),  # Video 2, Person ID 1
    ...
}

### 7. Serialization Strategy
# -> To save signatures as JSON, I convert numpy arrays to lists with .tolist(). 
# -> When loading, I convert back to numpy arrays.
# -> This serialization allows the gallery to be saved as human-readable JSON that can be inspected or shared. 
# -> I use class methods to_dict and from_dict for clean serialization/deserialization logic.

def to_dict(self):
    return {
        'person_id': self.person_id,
        'video_source': self.video_source,
        'face_embedding': self.face_embedding.tolist() if self.face_embedding is not None else None,
        'deep_features': self.deep_features.tolist() if self.deep_features is not None else None,
        # ... etc
    }

@classmethod
def from_dict(cls, data):
    signature = cls(data['person_id'], data['video_source'])
    if data['face_embedding'] is not None:
        signature.face_embedding = np.array(data['face_embedding'])
    # ... etc
    return signature
#########################################################################################################
#########################################################################################################

##### gallery_matching.py #####
### 1. Similarity Threshold Decision
# -> I set the matching threshold at 0.6 (60% similarity). 
# -> This was chosen empirically - same-person comparisons typically score above 0.7, while different people score below 0.5. 
# -> A threshold of 0.6 is conservative, reducing false matches while catching most true matches. 
# -> In production, I'd tune this on validation data using ROC curves to optimize precision-recall trade-off.

# Threshold too low (0.3):
#   → Many false matches (high recall, low precision)
#   → Different people incorrectly matched
  
# Threshold too high (0.9):
#   → Miss real matches (low recall, high precision)
#   → Same person not recognized
  
# Sweet spot (0.6-0.7):
#   → Balanced trade-off

class GalleryMatcher:
    def __init__(self, gallery_path, similarity_threshold=0.6):
        self.similarity_threshold = similarity_threshold

# **How similarity scores distribute**:
# Same person:      0.7 - 0.95 (high similarity)
# Similar people:   0.5 - 0.7  (medium)
# Different people: 0.2 - 0.5  (low)
# Threshold 0.6 splits same/different

### 2. The Matching Algorithm
# The matching algorithm is straightforward exhaustive search. 
# For each query person, I compute similarity to every gallery entry and select the highest scoring match. 
# If that score exceeds the threshold, we have a match. 
# Otherwise, it's a new person. 
# This is O(N) complexity, which is fine for small galleries (<1000 people). 
# For large-scale systems, I'd use approximate nearest neighbor search like FAISS or annoy to speed this up.

# **This is exhaustive search**: Compare query to every gallery entry.
# **Complexity**: O(N) where N = gallery size

def find_best_match(self, query_signature):
    best_match = None
    best_similarity = -1
    best_details = None
    
    # 1. Compare query against ALL gallery signatures
    for gallery_key, gallery_sig in self.gallery.items():
        similarity, details = self.compute_similarity(query_signature, gallery_sig)
        
        # 2. Track the best match
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = gallery_key
            best_details = details
    
    # 3. Apply threshold decision
    if best_similarity >= self.similarity_threshold:
        return best_match, best_similarity, best_details  # MATCH
    else:
        return None, best_similarity, best_details  # NEW PERSON

### 3. Similarity Breakdown
# -> I return not just overall similarity but a breakdown by feature type. 
# -> This is crucial for debugging and explainability.
# -> If a match has high face similarity (0.9) but low appearance similarity (0.4), it might be the same person who changed clothes.
# -> If face is low (0.3) but appearance is high (0.8), maybe the face wasn't detected well but clothing matches.
# -> This granularity helps build trust in the system.

similarity, details = self.compute_similarity(query_sig, gallery_sig)

# similarity: 0.73 (overall)
# details: {
#     'face': 0.85,        # Strong face match
#     'deep': 0.72,        # Good appearance match
#     'color': 0.68,       # Decent color match
#     'structure': 0.45    # Weak structure match
# }

### 4. Cross-Video Matching Workflow
# -> Cross-video matching tests if people appear in multiple videos. 
# -> I treat Video 1 as the gallery (known people) and Video 2 as queries (unknowns).
# -> Each person from Video 2 is compared against all people from Video 1. 
# -> If someone from Video 2 matches someone from Video 1 with high similarity, they're likely the same person. 
# -> This simulates a multi-camera scenario where we track people across different camera feeds.

def test_cross_video_matching(self, video1_path, video2_path, detection_dir):
    # 1. Gallery (Video 1) is already loaded
    print(f"Gallery Video: {video1_name}")
    
    # 2. Build query gallery from Video 2
    query_gallery = self._build_query_gallery(video2_path, detection_dir)
    print(f"Query Video: {video2_name} - {len(query_gallery)} persons")
    
    # 3. Match each query person against gallery
    results = {}
    for query_key, query_sig in query_gallery.items():
        match_key, similarity, details = self.find_best_match(query_sig)
        results[query_key] = {
            'matched': match_key is not None,
            'match_key': match_key,
            'similarity': similarity,
            'details': details
        }
    
    # 4. Analyze results
    matched = sum(1 for r in results.values() if r['matched'])
    print(f"Matched: {matched}/{len(results)}")
    
    return results

# **Example scenario**:
# Video 1 (Gallery):
#   - Person 1: John (enters at 0:10, leaves at 0:45)
#   - Person 2: Mary (enters at 0:20, leaves at 1:00)
#   - Person 3: Bob (enters at 0:50, leaves at 1:30)

# Video 2 (Query):
#   - Person 1: ??? (appears at 0:05)
#   - Person 2: ??? (appears at 0:30)

# Matching:
#   Video2-Person1 vs Gallery → Best match: Video1-Person2 (Mary), similarity 0.78 → MATCH
#   Video2-Person2 vs Gallery → Best match: Video1-Person1 (John), similarity 0.45 → NO MATCH (new person)

### 5. Building Query Gallery On-The-Fly
# -> When matching a new video, I build a temporary query gallery using the same signature creation logic from feature_matching.py. 
# -> I load the face and appearance features extracted for that video and create PersonSignature objects.
# -> This query gallery is then compared against the persistent main gallery. 
# -> This design allows the system to match new videos against any existing gallery without reprocessing everything.

def _build_query_gallery(self, video_path, detection_dir):
    video_name = Path(video_path).stem
    
    # Load features for this video
    face_file = detection_dir / f"{video_name}_face_features.pkl"
    reid_file = detection_dir / f"{video_name}_reid_features.pkl"
    
    face_data = pickle.load(open(face_file, 'rb')) if face_file.exists() else None
    reid_data = pickle.load(open(reid_file, 'rb')) if reid_file.exists() else None
    
    # Create signatures for each person
    query_gallery = {}
    for person_id in all_person_ids:
        sig = create_signature(person_id, face_data, reid_data)
        query_gallery[(video_name, person_id)] = sig
    
    return query_gallery

### 6. Match Result Visualization
# -> I visualize matches with color coding: green boxes for matched persons showing which gallery entry they matched and the similarity score, red boxes for new people not in the gallery. 
# -> This makes it immediately clear which detections are re-identifications versus new entries. 
# -> The similarity score helps assess confidence - 0.9 is very confident, 0.6 is borderline.

if result['matched']:
    color = (0, 255, 0)  # GREEN = matched to someone in gallery
    label = f"ID {track_id} → Match: ID {match_id} ({similarity:.2f})"
else:
    color = (0, 0, 255)  # RED = new person (not in gallery)
    label = f"ID {track_id} (New Person)"

# **Why this is effective**:
# - **Instant visual feedback**: Green/red immediately shows matches
# - **Detailed info**: Shows which gallery person matched
# - **Confidence score**: Similarity value builds trust

### 7. Match Statistics and Analysis
# -> I compute match rate as the percentage of query persons that matched someone in the gallery. 
# -> This metric helps evaluate system performance. 
# -> A high match rate on videos of the same location taken at different times validates the re-identification works. 
# -> A low match rate might indicate the videos show different populations, or the threshold is too strict. 
# -> This feedback helps tune the system.

total_queries = len(results)
matched = sum(1 for r in results.values() if r['matched'])
unmatched = total_queries - matched

match_rate = matched / total_queries * 100

print(f"Match rate: {match_rate:.1f}%")
###################################################################################################################################
###################################################################################################################################

##### temporal.py #####
# -> Temporal re-identification handles people leaving and returning to the same scene. 
# -> SORT tracking only maintains IDs while people are continuously visible.
# -> When someone leaves frame and returns later, SORT assigns a new track ID because it has no memory.
# -> My temporal re-identification layer sits above SORT, maintaining persistent IDs across these gaps by matching appearance features.

### 1. 2-Tier Gallery System ###
# -> Maintain two galleries: active persons currently in the scene, and historical persons who've left. 
# -> When someone exits, they're moved to historical storage after 150 frames (about 5 seconds). 
# -> When a new detection appears, I first check if it matches anyone in the historical gallery. 
# -> If yes, it's a re-entry, assign the same persistent ID.
# -> If no, it's a genuinely new person. This gives persistent IDs that survive temporal gaps.

class TemporalGallery:
    def __init__(self):
        # Tier 1: Active persons (currently visible)
        self.active_persons = {}      # track_id → {features, persistent_id, ...}
        
        # Tier 2: Historical persons (left scene)
        self.historical_persons = {}  # track_id → {features, persistent_id, ...}
        
        # Mapping: track_id → persistent_id
        self.reid_map = {}
        
        # Counter for new persistent IDs
        self.next_persistent_id = 1

# **State transitions**:
# NEW DETECTION → Create active entry, assign persistent_id
# CONTINUOUS TRACKING → Update active entry
# EXIT SCENE → Move to historical after temporal_window frames
# RE-ENTER → Match against historical, reuse persistent_id

### 2. Temporal Window Parameter ###
# -> The temporal window is 150 frames, about 5 seconds at 30fps. 
# -> This is the grace period - if someone isn't detected for 150 frames, we assume they've left and move them to historical gallery.
# -> This balances two competing needs: (1) not treating brief occlusions as exits, (2) not keeping every person in active memory forever. 
# -> Five seconds is enough to handle someone walking behind a pillar but not so long that we accumulate stale entries.

temporal_window = 150  # frames (~5 seconds at 30fps)

# **What it controls**:
# Person leaves frame → Start counting
# Frame 1-149:   Still in "active" state (might return any second)
# Frame 150+:    Moved to "historical" (probably gone for a while)

### 3. Update Cycle ###
# -> Each frame, process all detections. 
# -> For known track IDs in active gallery, simply update their features.
# -> For new track IDs, extract features and match against the historical gallery.
# -> If similarity exceeds 0.65, it's a re-entry, assign the same persistent ID they had before.
# -> Otherwise, it's a new person getting a new persistent ID.
# -> Finally, move people who haven't been seen for 150 frames from active to historical gallery. 
# -> This cycle maintains persistent IDs across temporal gaps.

def update(self, frame_num, detections, frame):
    current_ids = set()
    frame_reid_map = {}
    
    # Step 1: Process each detection in this frame
    for detection in detections:
        track_id = detection['track_id']
        current_ids.add(track_id)
        
        # Extract features for this detection
        features = self.extract_features(frame, detection['bbox'])
        
        # Step 2: Check if this track_id is known (active) or new
        if track_id in self.active_persons:
            # Known active person - just update
            persistent_id = self.reid_map[track_id]
            self.active_persons[track_id]['features'].append(features)
            self.active_persons[track_id]['last_frame'] = frame_num
            
        else:
            # New track_id - could be genuinely new or re-entry
            
            # Step 3: Try to match against historical persons
            best_match_id, best_similarity = self.match_against_historical(features)
            
            if best_similarity >= self.threshold:
                # RE-ENTRY! Reuse old persistent_id
                persistent_id = self.historical_persons[best_match_id]['persistent_id']
                self.active_persons[track_id] = {
                    'persistent_id': persistent_id,
                    'is_reentry': True,
                    'original_track_id': best_match_id
                }
            else:
                # NEW PERSON - assign new persistent_id
                persistent_id = self.next_persistent_id
                self.next_persistent_id += 1
                self.active_persons[track_id] = {
                    'persistent_id': persistent_id,
                    'is_reentry': False
                }
            
            self.reid_map[track_id] = persistent_id
        
        frame_reid_map[track_id] = persistent_id
    
    # Step 4: Move inactive persons to historical
    inactive_ids = set(self.active_persons.keys()) - current_ids
    for track_id in inactive_ids:
        if frame_num - self.last_seen[track_id] > self.temporal_window:
            self.historical_persons[track_id] = self.active_persons[track_id]
            del self.active_persons[track_id]
    
    return frame_reid_map

# **Key decisions in the algorithm**:
# 1. Known track_id: Easy, just update features
# 2. New track_id: The critical decision - new person or re-entry?
# 3. Matching threshold: 0.65 (stricter than cross-video 0.6)
# 4. Cleanup: Move inactive persons to historical

### 4. Stricter Threshold for Temporal (0.65) ###
# Reasoning:
# Cross-video (0.6):
#   - Different cameras, lighting, angles
#   - More variation expected
#   - Lower threshold to catch difficult matches

# Temporal (0.65):
#   - Same camera, same lighting
#   - Less variation expected
#   - Higher threshold to avoid false re-entries

# The risk of being too lenient:
# Person A leaves (moved to historical)
# Person B enters (looks similar)
# With threshold 0.6: Might incorrectly match → wrong persistent_id
# With threshold 0.65: Less likely to false match

### 5. Real-time Feature Extraction
# -> Unlike feature_extraction.py and feature_matching.py which process videos offline, temporal.py extracts features in real-time for each detection.
# -> This is necessary because we don't know in advance who will return.
# -> I use ResNet50 and color histograms for speed - face detection is skipped since it's slower and we need quick matching. 
# -> On GPU, this runs near real-time. 
# -> For true real-time on CPU, I'd skip frames or use lighter models like MobileNet.

def extract_features(self, frame, bbox):
    # Quick ResNet50 + color histogram extraction
    person_img = frame[y1:y2, x1:x2]
    
    # Deep features
    img_tensor = self.transform(person_img)
    with torch.no_grad():
        deep_features = self.appearance_model(img_tensor).flatten()
    
    # Color features
    hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
    hist = compute_histogram(hsv)
    
    return {'deep': deep_features, 'color': hist}

### 6. Handling Re-Entry Events
# -> Explicitly track re-entry events - when someone returns and is matched to their historical entry. 
# -> These are visualized with yellow boxes and logged for analysis. 
# -> This serves multiple purposes: (1) validates the system is working correctly, (2) provides metrics like 'X people returned', (3) helps debug by showing exactly when re-identifications occurred.
# -> In the final video, yellow boxes highlight re-entries, making it clear the system recognized the person.

if is_reentry and track_id not in [e['track_id'] for e in reentry_events]:
    reentry_events.append({
        'frame': frame_num,
        'track_id': track_id,
        'persistent_id': persistent_id
    })

if is_reentry:
    color = (0, 255, 255)  # YELLOW for re-entry
    label = f"Person {persistent_id} (RE-ENTRY)"
else:
    color = (0, 255, 0)    # GREEN for normal
    label = f"Person {persistent_id}"

### 7. Memory Management ###
# -> Memory can grow unbounded if we store every historical person forever. 
# -> I implement a temporal window to move people from active to historical, but even historical can grow large.
# ->  For production, I'd add: (1) Maximum historical size with FIFO eviction, (2) Only store averaged features not all samples, (3) Periodically purge entries older than 1 hour. 
# -> The key is balancing re-identification capability (need historical entries) with memory constraints (can't store everything).

# 1. Move old entries to historical (already implemented)
if not_seen_for > temporal_window:
    move_to_historical()

# 2. Limit historical size
MAX_HISTORICAL = 1000
if len(self.historical_persons) > MAX_HISTORICAL:
    # Remove oldest entries (FIFO)
    oldest_id = min(self.historical_persons.keys(), 
                    key=lambda x: self.historical_persons[x]['last_frame'])
    del self.historical_persons[oldest_id]

# 3. Feature compression
# Store only averaged features, not all samples
avg_features = np.mean(all_features, axis=0)
####################################################################################################################
####################################################################################################################

##### 8. Visualization #####
### 1. Report Generation Pipeline
## ->> 3 types of output
# 1. Machine-readable: JSON with all data
#    → For automated processing, databases, APIs
# 2. Human-readable: Text reports
#    → For stakeholders, documentation
# 3. Visual: Plots and annotated videos
#    → For presentations, quick understanding

## ->> Architecture
class ComprehensiveVisualizer:
    def create_summary_report(self):
        # 1. Collect data from all phases
        stats = self._analyze_all_videos()
        
        # 2. Generate machine-readable report (JSON)
        self._save_json_report(stats)
        
        # 3. Generate human-readable report (TXT)
        self._create_readable_report(stats)
        
        # 4. Generate visualizations (PNG)
        self.generate_statistics_plots(stats)

### 2. Per-Video Analysis
# Detection rate: How many frames had detections
# Face detection rate: % of tracked persons with faces
# Re-entry rate: % of persons who returned

def _analyze_video(self, video_name, detection_dir):
    stats = {}
    
    # Detection metrics (detection.py)
    detection_data = load_json(f"{video_name}_detections.json")
    stats['detection'] = {
        'total_frames': len(detection_data['detections']),
        'unique_persons_tracked': detection_data['statistics']['unique_persons'],
        'total_detections': detection_data['statistics']['total_detections'],
        'avg_detections_per_frame': total_detections / total_frames
    }
    
    # Face recognition metrics (recognition.py)
    face_data = load_pickle(f"{video_name}_face_features.pkl")
    stats['face_recognition'] = {
        'persons_with_faces': len(face_data['person_features']),
        'total_face_samples': sum(p['num_samples'] for p in face_data.values()),
        'face_detection_rate': persons_with_faces / unique_persons * 100
    }
    
    # ReID metrics (feature_extraction.py)
    reid_data = load_pickle(f"{video_name}_reid_features.pkl")
    stats['appearance_reid'] = {
        'persons_with_features': len(reid_data['person_features']),
        'total_appearance_samples': sum(p['num_samples'] for p in reid_data.values())
    }
    
    # Temporal metrics (temporal.py)
    temporal_data = load_json(f"{video_name}_temporal_reid.json")
    stats['temporal_reid'] = {
        'unique_persons': temporal_data['statistics']['total_unique_persons'],
        'reentry_events': temporal_data['statistics']['detected_reentries'],
        'reentry_rate': reentry_events / unique_persons * 100
    }
    
    return stats

### 3. Cross-Video Analysis
# -> Cross-video analysis shows the big picture. 
# -> If Video 1 has 15 people and Video 2 has 12 people, but 8 matched between them, then we actually saw 19 unique individuals (15 + 12 - 8). 
# -> The match rate indicates system performance - high rate (>70%) means the system successfully re-identified people across videos. 
# -> Low rate (<30%) might mean different populations or poor matching. 
# -> This metric is crucial for multi-camera scenarios.

def _analyze_cross_video(self, video_files, detection_dir, gallery_path):
    # Count persons in each video
    video_person_counts = {}
    for video_file in video_files:
        video_name = video_file.stem
        detection_data = load_json(f"{video_name}_detections.json")
        video_person_counts[video_name] = detection_data['statistics']['unique_persons']
    
    # If we have matching results from Phase 6
    if matching_results_exist():
        matched_count = count_cross_video_matches()
        cross_video_match_rate = matched_count / sum(video_person_counts.values()) * 100
    
    return {
        'videos_compared': len(video_files),
        'persons_per_video': video_person_counts,
        'total_unique_across_videos': sum(video_person_counts.values()),
        'cross_video_match_rate': cross_video_match_rate
    }

### 4. Temporal Analysis Aggregation
# -> Temporal analysis aggregates re-entry events across all videos. 
# -> Total re-entries shows how often people returned. 
# -> Re-entry rate indicates traffic patterns - a store might have 40% re-entry rate (people browse and come back), while a hallway has 5% (people just pass through).
# -> I also track which specific frames re-entries occurred for detailed validation. 
# -> This helps understand not just if the system works, but what it's detecting.

def _analyze_temporal(self, video_files, detection_dir):
    total_reentries = 0
    videos_with_reentries = 0
    
    reentry_details = []
    
    for video_file in video_files:
        temporal_data = load_json(f"{video_name}_temporal_reid.json")
        reentries = temporal_data['statistics']['detected_reentries']
        
        total_reentries += reentries
        if reentries > 0:
            videos_with_reentries += 1
            
        # Collect individual re-entry events
        for event in temporal_data['reentry_events']:
            reentry_details.append({
                'video': video_name,
                'frame': event['frame'],
                'person_id': event['persistent_id']
            })
    
    return {
        'total_reentry_events': total_reentries,
        'videos_with_reentries': videos_with_reentries,
        'avg_reentries_per_video': total_reentries / len(video_files),
        'reentry_details': reentry_details
    }

### 5. Human-Readable Report Generation
# -> Human-readable reports use clear structure with headers and sections. 
# -> I format numbers with units (75%, 150 frames) and provide context ('Detection rate of 75% is good for CCTV'). 
# -> The report starts with high-level overview, then drills into per-video details, then summarizes capabilities. 
# -> This inverted pyramid structure lets readers get the main points quickly or dive deep if needed. 
# -> It's the same principle as writing scientific papers or news articles.

def _create_readable_report(self, report):
    with open('summary_report.txt', 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("PERSON RE-IDENTIFICATION SYSTEM - SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        # System overview
        f.write("SYSTEM OVERVIEW\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Videos Processed: {len(video_files)}\n")
        f.write(f"Videos: {', '.join(video_names)}\n\n")
        
        # Per-video breakdown
        f.write("PER-VIDEO ANALYSIS\n")
        f.write("-" * 70 + "\n")
        for video_name, stats in report['per_video_stats'].items():
            f.write(f"\n{video_name}:\n")
            f.write(f"  Detection & Tracking:\n")
            f.write(f"    - Total frames: {stats['detection']['total_frames']}\n")
            f.write(f"    - Unique persons: {stats['detection']['unique_persons_tracked']}\n")
            f.write(f"  Face Recognition:\n")
            f.write(f"    - Persons with faces: {stats['face']['persons_with_faces']}\n")
            f.write(f"    - Detection rate: {stats['face']['face_detection_rate']:.1f}%\n")
        
        # Summary of capabilities
        f.write("\n\nSYSTEM CAPABILITIES DEMONSTRATED\n")
        f.write("-" * 70 + "\n")
        f.write("Person detection and tracking within single camera feed\n")
        f.write("Face recognition when faces are visible\n")
        f.write("Appearance-based re-identification (works without faces)\n")
        f.write("Temporal re-identification (people leaving and returning)\n")
        f.write("Cross-video person matching\n")
    
### 6. Validation
# -> Validation is critical for ML systems. 
# -> I use multiple strategies: 
# ->> (1) Visual inspection of output videos - do the boxes make sense? 
# ->> (2) Sanity checks on metrics - detection rate can't exceed 100%. 
# ->> (3) Edge case testing - what if video has no people?
# ->> (4) Cross-validation - if system says A matches B, manually verify similarity is high. 
# -> These checks catch bugs and build confidence the system actually works.

# 1. Visual inspection
# Open output videos, manually verify:
# - Are bounding boxes around people?
# - Do IDs stay consistent?
# - Are re-entries marked in yellow?

# 2. Sanity checks on metrics
assert face_detection_rate <= 100, "Rate can't exceed 100%"
assert reentry_count <= total_persons, "Can't have more re-entries than people"
assert all(similarity >= 0 and similarity <= 1 for sim in all_similarities)

# 3. Edge case testing
# - Empty video (no people) → should have 0 detections
# - Single person video → should have 1 unique person
# - Same person in both videos → should match with high similarity

# 4. Cross-validation
# If person A matched person B:
#   similarity(A, B) should be high (>0.6)
# If person A didn't match person C:
#   similarity(A, C) should be low (<0.6)
