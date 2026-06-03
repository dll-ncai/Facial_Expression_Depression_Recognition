"""
ResNet-50 video branch for LightFusionNet.

Extracts multi-region, temporally-attended features from raw video frames
and trains an SVR + Random Forest voting ensemble to predict BDI-II scores.

Usage::

    python models/video_resnet.py \\
        --root  /path/to/dataset_preprocessed \\
        --labels /path/to/dataset_preprocessed/labels.csv \\
        --output /path/to/saved_models
"""

import os
import argparse
import numpy as np
import joblib

import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR

from utils import (
    AVEC2014Dataset,
    IMAGE_TRANSFORM,
    select_expressive_frames,
    extract_multi_region_features,
    enhanced_weighted_pooling,
    compute_all_metrics,
    bdi_to_severity,
)


def build_feature_extractor():
    """Return a frozen ResNet-50 truncated before the classification head."""
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in backbone.parameters():
        param.requires_grad = False
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    extractor.eval()
    return extractor


def precompute_features(dataset, extractor, device, max_expressive=100):
    """
    Iterate over a dataset and return stacked feature vectors.

    For each video:
    1. Optionally sub-select the most expressive frames.
    2. Extract multi-region features with the frozen extractor.
    3. Pool into a single descriptor via weighted temporal attention.

    Returns:
        Tuple of ``(X, y, video_ids)`` as numpy arrays.
    """
    X, y, video_ids = [], [], []
    for i, (frames, label, video_id) in enumerate(dataset):
        print(f"  Video {i + 1}/{len(dataset)}", end="\r")
        if len(frames) > max_expressive:
            frames, _ = select_expressive_frames(frames, extractor, device, max_expressive)
        region_feats = extract_multi_region_features(frames, extractor, device)
        pooled = enhanced_weighted_pooling(region_feats)
        X.append(pooled.numpy())
        y.append(label.item())
        video_ids.append(video_id)
    print()
    return np.array(X), np.array(y), video_ids


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    extractor = build_feature_extractor().to(device)

    # ------------------------------------------------------------------ data
    train_ds = AVEC2014Dataset(args.root, args.labels, "Training",  IMAGE_TRANSFORM, 500)
    dev_ds   = AVEC2014Dataset(args.root, args.labels, "Development", IMAGE_TRANSFORM, 500)
    test_ds  = AVEC2014Dataset(args.root, args.labels, "Testing",   IMAGE_TRANSFORM, 500)

    print("Extracting training features…")
    X_train, y_train, train_ids = precompute_features(train_ds, extractor, device)
    print("Extracting development features…")
    X_dev,   y_dev,   dev_ids   = precompute_features(dev_ds,   extractor, device)
    print("Extracting test features…")
    X_test,  y_test,  test_ids  = precompute_features(test_ds,  extractor, device)

    # --------------------------------------------------------- feature selection
    selector = SelectKBest(f_regression, k=1000)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_dev_sel   = selector.transform(X_dev)
    X_test_sel  = selector.transform(X_test)
    print(f"Features: {X_train.shape[1]} → {X_train_sel.shape[1]}")

    # -------------------------------------------------------------- training
    svr = SVR(kernel="linear", C=0.1)
    rf  = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, random_state=42)
    ensemble = VotingRegressor([("svr", svr), ("rf", rf)])
    ensemble.fit(X_train_sel, y_train)

    # --------------------------------------------------------------- evaluation
    for tag, X_sel, y_true in [
        ("Train",       X_train_sel, y_train),
        ("Development", X_dev_sel,   y_dev),
        ("Test",        X_test_sel,  y_test),
    ]:
        preds = ensemble.predict(X_sel)
        m = compute_all_metrics(y_true, preds)
        print(f"{tag:12s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
              f"PCC={m['PCC']:.4f}  CCC={m['CCC']:.4f}")

    test_preds = ensemble.predict(X_test_sel)
    sev_true = [bdi_to_severity(s) for s in y_test]
    sev_pred = [bdi_to_severity(s) for s in test_preds]
    acc = np.mean(np.array(sev_true) == np.array(sev_pred))
    print(f"Severity accuracy: {acc:.2%}")
    labels = ["Minimal", "Mild", "Moderate", "Severe"]
    print("Confusion matrix:")
    print(confusion_matrix(sev_true, sev_pred, labels=labels))

    # --------------------------------------------------------------- saving
    os.makedirs(args.output, exist_ok=True)
    joblib.dump(ensemble, os.path.join(args.output, "video_resnet_model.pkl"))
    joblib.dump(selector, os.path.join(args.output, "video_resnet_selector.pkl"))

    for split_tag, ids, preds, labels_arr in [
        ("train", train_ids, ensemble.predict(X_train_sel), y_train),
        ("dev",   dev_ids,   ensemble.predict(X_dev_sel),   y_dev),
        ("test",  test_ids,  test_preds,                    y_test),
    ]:
        np.save(os.path.join(args.output, f"frames_{split_tag}_predictions.npy"), preds)
        np.save(os.path.join(args.output, f"frames_{split_tag}_labels.npy"),      labels_arr)
        np.save(os.path.join(args.output, f"frames_{split_tag}_video_ids.npy"),   np.array(ids))

    print(f"Saved model and predictions to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResNet-50 video branch")
    parser.add_argument("--root",   required=True, help="Dataset root directory")
    parser.add_argument("--labels", required=True, help="Path to labels.csv")
    parser.add_argument("--output", default="saved_models", help="Output directory")
    main(parser.parse_args())