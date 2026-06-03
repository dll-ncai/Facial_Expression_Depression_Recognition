import torch
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Frame-level feature extraction
# ---------------------------------------------------------------------------

def extract_video_features(frames, model, device="cuda", batch_size=32):
    """
    Forward-pass ``frames`` through ``model`` in mini-batches.

    Args:
        frames:     Tensor of shape ``(N, C, H, W)``.
        model:      Feature extractor (output squeezed to ``(N, D)``).
        device:     Torch device string.
        batch_size: Number of frames per forward pass.

    Returns:
        Tensor of shape ``(N, D)``.
    """
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size].to(device)
            out = model(batch).view(batch.size(0), -1)
            features.append(out.cpu())
    return torch.cat(features, dim=0)


def select_expressive_frames(frames, model, device="cuda", num_frames=100):
    """
    Select the ``num_frames`` most expressive frames using a composite score
    based on feature magnitude, deviation from the mean, and channel variance.

    Args:
        frames:     Tensor of shape ``(N, C, H, W)``.
        model:      Feature extractor.
        device:     Torch device string.
        num_frames: Number of frames to retain.

    Returns:
        Tuple of ``(selected_frames, scores)`` where ``selected_frames`` has
        shape ``(num_frames, C, H, W)``.
    """
    features = extract_video_features(frames, model, device)

    magnitude = torch.norm(features, p=2, dim=1)
    deviation = torch.norm(features - features.mean(dim=0), p=2, dim=1)
    variance  = features[:, : min(100, features.shape[1])].std(dim=1)

    scores = 0.5 * magnitude + 0.3 * deviation + 0.2 * variance
    k = min(num_frames, len(frames))
    top_idx = torch.topk(scores, k).indices
    return frames[top_idx], scores


# ---------------------------------------------------------------------------
# Facial-region decomposition
# ---------------------------------------------------------------------------

def extract_facial_regions(frames):
    """
    Partition each frame into four anatomical regions.

    Regions and their approximate facial coverage:

    * **eyes**        – upper half of the face.
    * **mouth**       – lower third of the face.
    * **left_cheek**  – middle third, left half.
    * **right_cheek** – middle third, right half.

    Args:
        frames: Tensor of shape ``(N, C, H, W)``.

    Returns:
        Dict mapping region name to a tensor of the same dtype/device.
    """
    H, W = frames.shape[2], frames.shape[3]
    return {
        "eyes":        frames[:, :, : H // 2, :],
        "mouth":       frames[:, :, 2 * H // 3 :, :],
        "left_cheek":  frames[:, :, H // 3 : 2 * H // 3, : W // 2],
        "right_cheek": frames[:, :, H // 3 : 2 * H // 3, W // 2 :],
    }


def extract_multi_region_features(frames, model, device):
    """
    Extract features independently for each facial region.

    Each region is resized to ``224×224`` before the forward pass.

    Args:
        frames: Tensor of shape ``(N, C, H, W)``.
        model:  Feature extractor.
        device: Torch device string.

    Returns:
        Dict mapping region name to a feature tensor of shape ``(N, D)``.
    """
    regions = extract_facial_regions(frames)
    region_features = {}
    for name, region in regions.items():
        region_resized = F.interpolate(region, size=(224, 224))
        region_features[name] = extract_video_features(region_resized, model, device)
    return region_features


# ---------------------------------------------------------------------------
# Weighted temporal pooling
# ---------------------------------------------------------------------------

# Facial-region weights derived from depression literature (AUs 1, 4, 6, 12,
# 17 cluster around eyes and mouth, with lower weight on cheeks).
REGION_WEIGHTS = {
    "eyes":        0.35,
    "mouth":       0.30,
    "right_cheek": 0.20,
    "left_cheek":  0.15,
}


def enhanced_weighted_pooling(region_features):
    """
    Aggregate per-frame region features into a single video-level descriptor.

    For each region:
    1. Compute per-frame temporal importance via channel standard deviation.
    2. Apply softmax over frames to obtain attention weights.
    3. Compute the attended feature vector.
    4. Scale by the anatomical region weight.

    The four weighted region vectors are concatenated to form the final
    descriptor.

    Args:
        region_features: Dict mapping region name to tensor ``(N, D)``.

    Returns:
        1-D tensor of length ``4 * D``.
    """
    pooled = []
    for name, feats in region_features.items():
        temporal_importance = feats.std(dim=1)
        attn = torch.softmax(temporal_importance, dim=0)
        attended = (feats * attn.unsqueeze(1)).sum(dim=0)
        pooled.append(attended * REGION_WEIGHTS[name])
    return torch.cat(pooled, dim=0)