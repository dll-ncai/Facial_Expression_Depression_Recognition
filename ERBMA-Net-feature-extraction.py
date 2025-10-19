# The author of this code is Dr. Muhammad Turyalai Khan. Cite his paper if you use it.

import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from torch.nn import functional as F
from mtcnn import MTCNN

# Step 1: Create Directories for Data Organization
def create_directories(base_dir):
    """Create directories for face, eyes, mouth, and frames across splits and categories."""
    for region in ['face', 'eyes', 'mouth']:
        for split in ['train', 'dev', 'test']:
            for category in ['Freeform', 'Northwind']:
                category_path = os.path.join(base_dir, region, split, category)
                os.makedirs(category_path, exist_ok=True)

# Step 2: Augmentation for Training Set
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    channel_shift_range=20.0,
    vertical_flip=True
)

def augment_image(image, generator, augment_count=10):
    """Generate augmented images."""
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return [next(generator.flow(image, batch_size=1))[0].astype(np.uint8) for _ in range(augment_count)]

# EfficientNetV2 for Global Features with Squeeze-and-Excitation (SE)
class EfficientNetV2(nn.Module):
    def __init__(self, model_cnf=[[2, 3, 1, 1, 24, 24, 0, 0], [2, 3, 2, 4, 24, 64, 0, 0]]):
        super(EfficientNetV2, self).__init__()
        self.stem = ConvBNAct(3, model_cnf[0][4], kernel_size=3, stride=2)
        total_blocks = sum([i[0] for i in model_cnf])
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            for i in range(repeats):
                stride = cnf[2] if i == 0 else 1
                input_c = cnf[4] if i == 0 else cnf[5]
                block = FusedMBConv(
                    kernel_size=cnf[1],
                    input_c=input_c,
                    out_c=cnf[5],
                    expand_ratio=cnf[3],
                    stride=stride
                )
                blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.se_block = SqueezeExcite(input_channels=cnf[5])
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.se_block(x)
        return x

# Squeeze-and-Excitation Block
class SqueezeExcite(nn.Module):
    def __init__(self, input_channels: int):
        super(SqueezeExcite, self).__init__()
        squeeze_channels = int(input_channels * 0.25)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, 1)
        x_avg = self.fc1(x_avg)
        x_avg = self.act(x_avg)
        x_avg = self.fc2(x_avg)
        x_avg = torch.sigmoid(x_avg)
        return x * x_avg

# FusedMBConv Block
class FusedMBConv(nn.Module):
    def __init__(self, kernel_size, input_c, out_c, expand_ratio, stride=None):
        super(FusedMBConv, self).__init__()
        self.expand_conv = (
            ConvBNAct(input_c, input_c * expand_ratio, kernel_size, stride)
            if expand_ratio != 1 else nn.Identity()
        )
        self.project_conv = ConvBNAct(input_c * expand_ratio, out_c, kernel_size=1, stride=1)
        self.se = SqueezeExcite(input_channels=out_c)

    def forward(self, x):
        out = self.expand_conv(x)
        out = self.project_conv(out)
        out = self.se(out)
        return out

# Convolution + BatchNorm + Activation Block
class ConvBNAct(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(ConvBNAct, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
# Step 5: RBC with improved Local Binary Convolutional Neural Network (LBCNN)
class RandomBinaryConv(nn.Module):
    """Random Binary Convolution Layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(RandomBinaryConv, self).__init__()
        self.weight = torch.randint(0, 2, (out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32) * 2 - 1
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        weight = self.weight.to(x.device)  # Move weights to the same device as input
        return F.conv2d(x, weight, stride=self.stride, padding=self.padding)

class ImprovedLBCNN(nn.Module):
    """Improved LBCNN with Random Binary Convolution, BatchNorm, and 1x1 Convolution."""
    def __init__(self, input_channels, output_channels=512):
        super(ImprovedLBCNN, self).__init__()
        self.rbc = RandomBinaryConv(input_channels, 64, kernel_size=3, padding=1)  # Random Binary Conv
        self.bn = nn.BatchNorm2d(64)  # BatchNorm after RBC
        self.relu = nn.LeakyReLU(negative_slope=0.2)  # Activation
        self.refine = nn.Conv2d(64, output_channels, kernel_size=1)  # 1x1 Conv for refinement

    def forward(self, x):
        x = self.rbc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.refine(x)
        return x  # Feature maps with 512 channels

# Step 6: ResNet with Coordinate Attention (CA) for Local Features (Eyes/Mouth)
class ResNetWithCA(nn.Module):
    def __init__(self):
        super(ResNetWithCA, self).__init__()
        self.local_blocks = nn.Sequential(
            CABottleneck(3, 64),
            CABottleneck(64, 128),
            CABottleneck(128, 256),
            CABottleneck(256, 512),
        )
        
    def forward(self, x):
        x = self.local_blocks(x)
        return x

# Coordinate Attention Bottleneck Block
class CABottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.ca = CoordAtt(out_channels, out_channels)  # Coordinate Attention
        self.relu = nn.ReLU()

        # Add projection layer if input and output channel dimensions mismatch
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False) \
            if in_channels != out_channels else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)

        if self.projection:
            identity = self.projection(identity)  # Match dimensions

        out += identity
        return self.relu(out)

# Coordinate Attention Mechanism
class CoordAtt(nn.Module):
    def __init__(self, inp, oup):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, oup // 2, kernel_size=1)
        self.bn_h = nn.BatchNorm2d(oup // 2)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(oup // 2, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(oup // 2, oup, kernel_size=1)

    def forward(self, x):
        identity = x
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn_h(self.conv1(y)))
        a_h, a_w = torch.split(y, [identity.size(2), identity.size(3)], dim=2)
        a_w = a_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(a_h).sigmoid()
        a_w = self.conv_w(a_w).sigmoid()
        return identity * a_h * a_w
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling
        combined = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(combined))
        return x * attention_map

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch, channels, height, width = x.size()
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)  # (B, H*W, C//8)
        key = self.key(x).view(batch, -1, height * width)  # (B, C//8, H*W)
        attention = torch.bmm(query, key)  # (B, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        value = self.value(x).view(batch, -1, height * width)  # (B, C, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch, channels, height, width)
        return self.gamma * out + x

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x3 = self.relu(self.conv3x3(x))
        x5 = self.relu(self.conv5x5(x))
        x7 = self.relu(self.conv7x7(x))
        return x3 + x5 + x7

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return x + residual

# Step 7: Combined Feature Extractor
class CombinedFeatureExtractor(nn.Module):
    def __init__(self):
        super(CombinedFeatureExtractor, self).__init__()
        self.global_extractor = EfficientNetV2()  # Face
        self.local_extractor = ResNetWithCA()    # Eyes/Mouth

        # Enhanced LBCNN for feature map refinement
        self.lbcnn_face = ImprovedLBCNN(input_channels=64)  # For face feature maps
        self.lbcnn_eyes_mouth = ImprovedLBCNN(input_channels=512)  # For eyes and mouth feature maps

        # Spatial-attention mechanism
        self.spatial_attention_face = SpatialAttention()
        self.spatial_attention_eyes = SpatialAttention()
        self.spatial_attention_mouth = SpatialAttention()

        # Multi-scale feature extraction
        self.multi_scale_face = MultiScaleFeatureExtractor(512, 512)
        self.multi_scale_eyes = MultiScaleFeatureExtractor(512, 512)
        self.multi_scale_mouth = MultiScaleFeatureExtractor(512, 512)

        # Residual connections
        self.residual_face = ResidualBlock(512)
        self.residual_eyes = ResidualBlock(512)
        self.residual_mouth = ResidualBlock(512)

        # Self-attention mechanism
        self.self_attention = SelfAttention(1536)  # Since concatenation will give 1536 channels

        # Three-stage average pooling to maintain a balance between detail and dimensionality
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.avg_pool3 = nn.AvgPool2d(kernel_size=4, stride=4)
        
    def forward(self, face, eyes, mouth):
        # Step 1: Extract raw feature maps
        face_features = self.global_extractor(face)  # EfficientNetV2 for face
        eyes_features = self.local_extractor(eyes)  # ResNet for eyes
        mouth_features = self.local_extractor(mouth)  # ResNet for mouth

        print(f"global face features: {face_features.shape}")
        print(f"local eyes features: {eyes_features.shape}")
        print(f"local mouth features: {mouth_features.shape}")

        # Step 2: Apply Enhanced LBCNN
        face_features = self.lbcnn_face(face_features)
        eyes_features = self.lbcnn_eyes_mouth(eyes_features)
        mouth_features = self.lbcnn_eyes_mouth(mouth_features)

        # Step 3: Apply spatial attention (before concatenation)
        face_features = self.spatial_attention_face(face_features)
        eyes_features = self.spatial_attention_eyes(eyes_features)
        mouth_features = self.spatial_attention_mouth(mouth_features)

        # Step 4: Apply multi-scale feature extraction
        face_features = self.multi_scale_face(face_features)
        eyes_features = self.multi_scale_eyes(eyes_features)
        mouth_features = self.multi_scale_mouth(mouth_features)

        # Step 5: Apply residual connections
        face_features = self.residual_face(face_features)
        eyes_features = self.residual_eyes(eyes_features)
        mouth_features = self.residual_mouth(mouth_features)

        # Debug: Print feature shapes after residual connections
        print(f"Face features after residual connection: {face_features.shape}")
        print(f"Eyes features after residual connection: {eyes_features.shape}")
        print(f"Mouth features after residual connection: {mouth_features.shape}")

        # Step 6: Resize feature maps to a common size
        target_size = (96, 192)  # Choose target size for resizing
        face_features_resized = F.interpolate(face_features, size=target_size, mode='bilinear', align_corners=False)
        eyes_features_resized = F.interpolate(eyes_features, size=target_size, mode='bilinear', align_corners=False)
        mouth_features_resized = F.interpolate(mouth_features, size=target_size, mode='bilinear', align_corners=False)

        # Debug: Print resized feature shapes
        print(f"Resized face features shape: {face_features_resized.shape}")
        print(f"Resized eyes features shape: {eyes_features_resized.shape}")
        print(f"Resized mouth features shape: {mouth_features_resized.shape}")

        # Step 7: Concatenate features across channels (face + eyes + mouth)
        combined_features = torch.cat([face_features_resized, eyes_features_resized, mouth_features_resized], dim=1)

        # Step 8: Apply self-attention on the concatenated features
        attention_output = self.self_attention(combined_features)

        # Debug: Print feature shapes after self-attention
        print(f"Feature map after self-attention: {attention_output.shape}")

        # Step 9: Apply three-stage avg pooling
        pooled_features = self.avg_pool1(attention_output)
        pooled_features = self.avg_pool2(pooled_features)
        pooled_features = self.avg_pool3(pooled_features)

        # Step 10: Direct Flattening to preserve spatial information
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Debug: Print shape after flattening
        print(f"Flattened feature vector shape: {flattened_features.shape}")

        return flattened_features

# Step 7: Extract face regions from video using MTCNN and handle padding for smaller crops
def crop_face_regions(video_path):
    """Crop face, eyes, and mouth regions from video frames using MTCNN."""
    detector = MTCNN()
    video_capture = cv2.VideoCapture(video_path)

    frames = []
    frame_idx = 0
    skip_frames = 150  # Adjust based on your video's frame rate for every ~5 seconds

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:  # Process only every 'skip_frames' frame
            results = detector.detect_faces(frame)

            for result in results:
                if 'keypoints' in result:
                    keypoints = result['keypoints']
                    left_eye_x, left_eye_y = keypoints['left_eye']
                    right_eye_x, right_eye_y = keypoints['right_eye']
                    mouth_left_x, mouth_left_y = keypoints['mouth_left']
                    mouth_right_x, mouth_right_y = keypoints['mouth_right']

                    # Crop face with fixed dimensions (224x224)
                    x1_face, y1_face, width_face, height_face = result['box']
                    # Adjust the bounding box to get 224x224 face crop
                    face_width = 224
                    face_height = 224

                    # Center the crop around the face bounding box
                    center_x = x1_face + width_face // 2
                    center_y = y1_face + height_face // 2

                    # Calculate new crop coordinates to ensure 224x224 size
                    x1_face_new = max(center_x - face_width // 2, 0)
                    y1_face_new = max(center_y - face_height // 2, 0)

                    # Ensure the crop fits within the image boundaries
                    x2_face_new = min(x1_face_new + face_width, frame.shape[1])
                    y2_face_new = min(y1_face_new + face_height, frame.shape[0])

                    # Crop the face image
                    face_image = frame[y1_face_new:y2_face_new, x1_face_new:x2_face_new]

                    # Check if the face image has the desired dimensions
                    if face_image.shape[0] != 224 or face_image.shape[1] != 224:
                        continue  # Skip this frame if face dimensions are not 224x224

                    # Crop eyes with fixed dimensions (192x96)
                    eye_width = right_eye_x - left_eye_x + 60
                    eye_height = 96
                    x1_eyes = max(0, left_eye_x - 30)
                    y1_eyes = max(0, left_eye_y - 20)
                    eyes_region = frame[y1_eyes:y1_eyes + eye_height, x1_eyes:x1_eyes + 192]

                    # Crop mouth with fixed dimensions (128x96)
                    mouth_width = mouth_right_x - mouth_left_x + 60
                    mouth_height = 96
                    x1_mouth = max(0, mouth_left_x - 30)
                    y1_mouth = max(0, mouth_left_y - 20)
                    mouth_region = frame[y1_mouth:y1_mouth + mouth_height, x1_mouth:x1_mouth + 128]

                    # Check if the mouth region has the desired dimensions
                    if mouth_region.shape[0] != 96 or mouth_region.shape[1] != 128:
                        continue  # Skip this frame if mouth dimensions are not 128x96

                    # Check if the eyes region has the desired dimensions
                    if eyes_region.shape[0] != 96 or eyes_region.shape[1] != 192:
                        continue  # Skip this frame if eyes dimensions are not 192x96

                    frames.append((face_image, eyes_region, mouth_region))

        frame_idx += 1

    video_capture.release()
    return frames

# Step 8: Save Frames and Features (including augmented images)
def save_frames_and_features(frames, output_dir, split, category, video_name, model, split_csv, augment, augment_count):
    face_dir = os.path.join(output_dir, 'face', split, category, video_name)
    eyes_dir = os.path.join(output_dir, 'eyes', split, category, video_name)
    mouth_dir = os.path.join(output_dir, 'mouth', split, category, video_name)

    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(eyes_dir, exist_ok=True)
    os.makedirs(mouth_dir, exist_ok=True)

    for frame_idx, (face, eyes, mouth) in enumerate(frames):
        # Save original images
        face_path = os.path.join(face_dir, f"face_{frame_idx}.jpg")
        eyes_path = os.path.join(eyes_dir, f"eyes_{frame_idx}.jpg")
        mouth_path = os.path.join(mouth_dir, f"mouth_{frame_idx}.jpg")

        cv2.imwrite(face_path, face)
        cv2.imwrite(eyes_path, eyes)
        cv2.imwrite(mouth_path, mouth)

        # Convert images to tensors
        face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float()
        eyes_tensor = torch.tensor(eyes).permute(2, 0, 1).unsqueeze(0).float()
        mouth_tensor = torch.tensor(mouth).permute(2, 0, 1).unsqueeze(0).float()

        face_tensor, eyes_tensor, mouth_tensor = face_tensor.to(model.device), eyes_tensor.to(model.device), mouth_tensor.to(model.device)
        features = model(face_tensor, eyes_tensor, mouth_tensor).detach().cpu().numpy().flatten()

        # Save original features to CSV
        save_features_to_csv([[video_name] + features.tolist()], split_csv)

        # Augmentation for training set
        if augment:
            augmented_faces = augment_image(face, data_gen, augment_count)
            augmented_eyes = augment_image(eyes, data_gen, augment_count)
            augmented_mouths = augment_image(mouth, data_gen, augment_count)

            for aug_idx, (aug_face, aug_eye, aug_mouth) in enumerate(zip(augmented_faces, augmented_eyes, augmented_mouths)):
                # Save augmented images
                aug_face_path = os.path.join(face_dir, f"face_aug_{frame_idx}_{aug_idx}.jpg")
                aug_eyes_path = os.path.join(eyes_dir, f"eyes_aug_{frame_idx}_{aug_idx}.jpg")
                aug_mouth_path = os.path.join(mouth_dir, f"mouth_aug_{frame_idx}_{aug_idx}.jpg")

                cv2.imwrite(aug_face_path, aug_face)
                cv2.imwrite(aug_eyes_path, aug_eye)
                cv2.imwrite(aug_mouth_path, aug_mouth)

                # Convert augmented images to tensors
                aug_face_tensor = torch.tensor(aug_face).permute(2, 0, 1).unsqueeze(0).float().to(model.device)
                aug_eye_tensor = torch.tensor(aug_eye).permute(2, 0, 1).unsqueeze(0).float().to(model.device)
                aug_mouth_tensor = torch.tensor(aug_mouth).permute(2, 0, 1).unsqueeze(0).float().to(model.device)

                augmented_features = model(aug_face_tensor, aug_eye_tensor, aug_mouth_tensor).detach().cpu().numpy().flatten()

                # Save augmented features to CSV
                save_features_to_csv([[video_name] + augmented_features.tolist()], split_csv)

# Step 9: Save extracted features to a CSV file
def save_features_to_csv(features, output_csv):
    """Save 1536-dimensional features (face, eyes, mouth) to a CSV file."""
    face_columns = [f"face_feature_{i}" for i in range(9216)] #16384 for 3 by 3
    eyes_columns = [f"eyes_feature_{i}" for i in range(9216)]
    mouth_columns = [f"mouth_feature_{i}" for i in range(9216)]
    all_columns = ["filename"] + face_columns + eyes_columns + mouth_columns

    df = pd.DataFrame(features, columns=all_columns)

    if not os.path.exists(output_csv):  # Create file if it doesn't exist
        df.to_csv(output_csv, index=False)
    else:  # Append to existing file
        df.to_csv(output_csv, mode='a', header=False, index=False)

    print(f"Features saved to {output_csv}")

# Step 10: Process Videos and Save Frames and Features
def process_videos(input_dir, output_dir, model, output_csv_files, augment=True, augment_count=10):
    """Process videos to extract face, eyes, and mouth features, save images, and save features to CSV."""
    create_directories(output_dir)

    for split in ['train', 'dev', 'test']:
        split_csv = output_csv_files[split]

        for category in ['Freeform', 'Northwind']:
            category_path = os.path.join(input_dir, split, category)

            for filename in os.listdir(category_path):
                if filename.endswith('.mp4'):
                    video_path = os.path.join(category_path, filename)
                    print(f"Processing video: {video_path}")

                    frames = crop_face_regions(video_path)
                    video_name = filename.replace('_Freeform_video', '').replace('_Northwind_video', '').replace('.mp4', '')

                    # Create directories for saving images
                    face_dir = os.path.join(output_dir, 'face', split, category, video_name)
                    eyes_dir = os.path.join(output_dir, 'eyes', split, category, video_name)
                    mouth_dir = os.path.join(output_dir, 'mouth', split, category, video_name)

                    os.makedirs(face_dir, exist_ok=True)
                    os.makedirs(eyes_dir, exist_ok=True)
                    os.makedirs(mouth_dir, exist_ok=True)

                    # Save frames for debugging and visualization
                    save_frames_and_features(frames, output_dir, split, category, video_name, model, split_csv, augment if split == 'train' else False, augment_count)
if __name__ == "__main__":
    input_base_directory = r"C:\Users\Turyal\Desktop\Depression recognition using facial features\MyAVEC2014\AVEC2014"
    output_directory = r"C:\Users\Turyal\Desktop\MTK\check"

    output_csv_files = {
        'train': os.path.join(output_directory, 'train_features.csv'),
        'dev': os.path.join(output_directory, 'dev_features.csv'),
        'test': os.path.join(output_directory, 'test_features.csv'),
    }

    # Initialize the feature extraction model
    model = CombinedFeatureExtractor()
    model.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(model.device)

    process_videos(
        input_dir=input_base_directory,
        output_dir=output_directory,
        model=model,
        output_csv_files=output_csv_files,
        augment=True,  # Enable augmentation for training
        augment_count=10  # Number of augmentations per frame
    )

    print("Processing complete.")