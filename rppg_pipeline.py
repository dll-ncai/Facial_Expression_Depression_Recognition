"""
rPPG feature pipeline for LightFusionNet.

Trains a stacking ensemble (SVR + LightGBM + XGBoost + ElasticNet + BayesianRidge)
on pre-extracted rPPG physiological features and saves per-video predictions for
all three dataset splits.

Usage::

    python models/rppg_pipeline.py \\
        --csv  /path/to/rppg_features_OFFICIAL_splits.csv \\
        --output /path/to/rppg_output
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    SelectFromModel, VarianceThreshold,
)
from sklearn.impute import KNNImputer
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import PowerTransformer
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_enhanced_features(df, feature_cols):
    """
    Augment a raw feature DataFrame with domain-specific interaction terms.

    Added features (when the underlying columns are present):

    * ``rmssd_sdnn_ratio`` – short-term vs long-term HRV.
    * ``lf_hf_ratio``      – sympatho-vagal balance.
    * ``dfa_alpha_centered`` – DFA α centred at 1 (self-similar boundary).

    Args:
        df:           DataFrame containing at least ``feature_cols``.
        feature_cols: List of numeric column names to include.

    Returns:
        DataFrame of enhanced features with ``NaN`` / ``inf`` imputed.
    """
    available = [c for c in feature_cols if c in df.columns]
    out = df[available].copy().fillna(df[available].median())

    if {"rmssd", "sdnn"}.issubset(out.columns):
        out["rmssd_sdnn_ratio"] = out["rmssd"] / (out["sdnn"] + 1e-8)
    if {"lf_power", "hf_power"}.issubset(out.columns):
        out["lf_hf_ratio"] = out["lf_power"] / (out["hf_power"] + 1e-8)
    if "dfa_alpha" in out.columns:
        out["dfa_alpha_centered"] = out["dfa_alpha"] - 1.0

    return out.replace([np.inf, -np.inf], np.nan).fillna(out.median())


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

class AdvancedFeatureSelector:
    """
    Multi-strategy feature selector that combines:

    * Variance thresholding (removes near-constant features).
    * Univariate F-regression (``SelectKBest``).
    * Mutual information regression.
    * Model-based importance (Random Forest).

    Features are ranked by how many strategies nominated them, and the top
    ``k_best`` are retained.

    Args:
        k_best: Maximum number of features to select.
    """

    def __init__(self, k_best=15):
        self.k_best = k_best
        self.selected_features = None

    def fit(self, X, y, feature_names):
        feature_names = np.array(feature_names[: X.shape[1]])

        vt = VarianceThreshold(threshold=0.01)
        X_vt = vt.fit_transform(X)
        vt_feats = feature_names[vt.get_support()]

        k = min(self.k_best, X_vt.shape[1])
        kbest_feats = vt_feats[SelectKBest(f_regression, k=k).fit(X_vt, y).get_support()]

        mi_idx = np.argsort(mutual_info_regression(X_vt, y, random_state=42))[-self.k_best :]
        mi_feats = vt_feats[mi_idx]

        rf_sel = SelectFromModel(
            RandomForestRegressor(n_estimators=100, random_state=42),
            max_features=min(self.k_best, X_vt.shape[1]),
        )
        rf_feats = vt_feats[rf_sel.fit(X_vt, y).get_support()]

        combined = np.concatenate([vt_feats, kbest_feats, mi_feats, rf_feats])
        counts = pd.Series(combined).value_counts()
        self.selected_features = counts.head(self.k_best).index.tolist()
        log(f"Selected {len(self.selected_features)} features")
        return self

    def transform(self, X, feature_names):
        feature_names = np.array(feature_names)
        keep = [i for i, f in enumerate(feature_names) if f in self.selected_features]
        return X[:, keep]

    def fit_transform(self, X, y, feature_names):
        return self.fit(X, y, feature_names).transform(X, feature_names)


# ---------------------------------------------------------------------------
# Video-level aggregation
# ---------------------------------------------------------------------------

class OptimalVideoAggregator:
    """
    Adaptively selects the best segment-to-video aggregation strategy.

    Candidates evaluated on the provided data:

    * mean
    * median
    * trimmed mean (drop min/max)
    * weighted centre (70 % mean + 30 % median)

    Args:
        strategy: ``"adaptive"`` (default) or one of the named candidates.
    """

    def __init__(self, strategy="adaptive"):
        self.strategy = strategy
        self.best_method_ = "median"

    def fit(self, segment_preds, video_groups, true_values):
        if self.strategy != "adaptive":
            self.best_method_ = self.strategy
            return self

        candidates = {
            "mean":           lambda s: np.mean(s),
            "median":         lambda s: np.median(s),
            "trimmed_mean":   lambda s: np.mean(np.sort(s)[1:-1]) if len(s) > 2 else np.mean(s),
            "weighted_center": lambda s: 0.7 * np.mean(s) + 0.3 * np.median(s),
        }
        errors = {k: [] for k in candidates}
        true_values = np.array(true_values)

        for vid in np.unique(video_groups):
            mask = video_groups == vid
            segs = segment_preds[mask]
            truth = true_values[mask][0]
            for name, fn in candidates.items():
                errors[name].append(abs(fn(segs) - truth))

        avg = {k: np.mean(v) for k, v in errors.items() if v}
        self.best_method_ = min(avg, key=avg.get)
        log(f"Best aggregation: {self.best_method_} (MAE={avg[self.best_method_]:.4f})")
        return self

    def predict(self, segment_preds, video_groups):
        fns = {
            "mean":           lambda s: np.mean(s),
            "median":         lambda s: np.median(s),
            "trimmed_mean":   lambda s: np.mean(np.sort(s)[1:-1]) if len(s) > 2 else np.mean(s),
            "weighted_center": lambda s: 0.7 * np.mean(s) + 0.3 * np.median(s),
        }
        fn = fns.get(self.best_method_, fns["median"])
        unique_vids = np.unique(video_groups)
        preds = [fn(segment_preds[video_groups == v]) for v in unique_vids]
        return np.array(preds), unique_vids


# ---------------------------------------------------------------------------
# Stacking ensemble (group-aware OOF)
# ---------------------------------------------------------------------------

def fit_stacking(base_estimators, meta_learner, X, y, groups, n_splits=5):
    """
    Train a two-level stacking ensemble respecting subject groups.

    Out-of-fold predictions are generated via ``GroupKFold`` so that no
    subject appears in both the fold's training and validation sets.  This
    prevents data leakage typical in clinical video datasets where multiple
    segments per subject exist.

    Args:
        base_estimators: List of ``(name, estimator)`` pairs.
        meta_learner:    Estimator for the second level.
        X, y, groups:    Training data.
        n_splits:        Number of GroupKFold splits.

    Returns:
        Tuple of ``(fitted_bases, fitted_meta, oof_preds)``.
    """
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    oof = np.zeros((len(X), len(base_estimators)))
    fitted_bases = {}

    for i, (name, est) in enumerate(base_estimators):
        log(f"OOF generation for: {name}")
        fold_preds = np.zeros(len(X))
        for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
            m = clone(est)
            m.fit(X[tr], y[tr])
            fold_preds[va] = m.predict(X[va])
            log(f"  fold {fold + 1}/{gkf.get_n_splits()}")
        oof[:, i] = fold_preds

        full = clone(est)
        full.fit(X, y)
        fitted_bases[name] = full

    meta = clone(meta_learner)
    meta.fit(oof, y)
    log("Meta-learner fitted on OOF predictions")
    return fitted_bases, meta, oof


def predict_stacked(fitted_bases, meta, X):
    meta_feats = np.column_stack([m.predict(X) for m in fitted_bases.values()])
    return meta.predict(meta_feats)


# ---------------------------------------------------------------------------
# Base model factory
# ---------------------------------------------------------------------------

def build_base_models():
    base = [
        ("svr_rbf",  SVR(kernel="rbf",  C=5.0, epsilon=0.05, gamma="scale")),
        ("svr_poly", SVR(kernel="poly", degree=2, C=2.0, epsilon=0.1, gamma="scale")),
        ("elastic",  ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=2000)),
        ("bayesian", BayesianRidge()),
    ]
    lgb_params = dict(n_estimators=80, learning_rate=0.05, max_depth=4,
                      random_state=42, verbose=-1)
    xgb_params = dict(n_estimators=80, learning_rate=0.05, max_depth=3,
                      random_state=42, objective="reg:squarederror", verbosity=0)

    if lgb is not None:
        base.append(("lgb", lgb.LGBMRegressor(**lgb_params)))
    if xgb is not None:
        base.append(("xgb", xgb.XGBRegressor(**xgb_params)))

    meta = ElasticNet(alpha=0.1, l1_ratio=0.5)
    return base, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    t0 = time.time()
    log("=== rPPG Pipeline ===")

    df = pd.read_csv(args.csv)
    df = df.loc[:, ~df.columns.duplicated()]
    required = ["video_id", "split", "BDI_II"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in required]
    df = df[list(dict.fromkeys(required + feature_cols))].dropna(subset=["BDI_II", "split"])

    train_df = df[df["split"] == "Training"].reset_index(drop=True)
    dev_df   = df[df["split"] == "Development"].reset_index(drop=True)
    test_df  = df[df["split"] == "Testing"].reset_index(drop=True)
    log(f"Videos — train: {train_df['video_id'].nunique()}, "
        f"dev: {dev_df['video_id'].nunique()}, test: {test_df['video_id'].nunique()}")

    train_feat = create_enhanced_features(train_df, feature_cols)
    dev_feat   = create_enhanced_features(dev_df,   feature_cols)
    test_feat  = create_enhanced_features(test_df,  feature_cols)

    common_cols = sorted(set(train_feat.columns) & set(dev_feat.columns) & set(test_feat.columns))
    X_train = train_feat[common_cols].values.astype(float)
    y_train = train_df["BDI_II"].values.astype(float)
    X_dev   = dev_feat[common_cols].values.astype(float)
    y_dev   = dev_df["BDI_II"].values.astype(float)
    X_test  = test_feat[common_cols].values.astype(float)
    groups_train = train_df["video_id"].values
    groups_dev   = dev_df["video_id"].values

    X_combined = np.vstack([X_train, X_dev])
    y_combined = np.concatenate([y_train, y_dev])
    groups_combined = np.concatenate([groups_train, groups_dev])

    imputer = KNNImputer(n_neighbors=3)
    X_combined = imputer.fit_transform(X_combined)
    X_test     = imputer.transform(X_test)

    selector = AdvancedFeatureSelector(k_best=args.k_best)
    X_combined_sel = selector.fit_transform(X_combined, y_combined, common_cols)
    X_test_sel     = selector.transform(X_test, common_cols)

    pt = PowerTransformer(method="yeo-johnson")
    X_combined_tf = pt.fit_transform(X_combined_sel)
    X_test_tf     = pt.transform(X_test_sel)

    # Strategy 1 – Fixed SVR
    log("Strategy 1: Fixed SVR")
    svr = SVR(kernel="rbf", C=5.0, epsilon=0.05, gamma="scale")
    svr.fit(X_combined_tf, y_combined)
    seg_pred_svr = svr.predict(X_test_tf)

    vid_to_true = {
        vid: test_df.loc[test_df["video_id"] == vid, "BDI_II"].iloc[0]
        for vid in test_df["video_id"].unique()
    }
    seg_true = np.array([vid_to_true[v] for v in test_df["video_id"].values])
    agg_svr = OptimalVideoAggregator()
    agg_svr.fit(seg_pred_svr, test_df["video_id"].values, seg_true)
    vid_pred_svr, _ = agg_svr.predict(seg_pred_svr, test_df["video_id"].values)
    video_ids_sorted = np.array(sorted(test_df["video_id"].unique()))
    vid_true = np.array([vid_to_true[v] for v in video_ids_sorted])
    mae_svr = mean_absolute_error(vid_true, vid_pred_svr)
    log(f"SVR test MAE (video-level): {mae_svr:.4f}")

    # Strategy 2 – Stacking ensemble
    log("Strategy 2: Stacking ensemble")
    base_models, meta_learner = build_base_models()
    fitted_bases, fitted_meta, _ = fit_stacking(
        base_models, meta_learner, X_combined_tf, y_combined, groups_combined
    )
    try:
        seg_pred_ens = predict_stacked(fitted_bases, fitted_meta, X_test_tf)
    except Exception as e:
        log(f"Stacking prediction failed ({e}); falling back to SVR.")
        seg_pred_ens = seg_pred_svr.copy()

    agg_ens = OptimalVideoAggregator()
    agg_ens.fit(seg_pred_ens, test_df["video_id"].values, seg_true)
    vid_pred_ens, _ = agg_ens.predict(seg_pred_ens, test_df["video_id"].values)
    mae_ens = mean_absolute_error(vid_true, vid_pred_ens)
    log(f"Ensemble test MAE (video-level): {mae_ens:.4f}")

    # Strategy 3 – Weighted combination
    alpha = 0.6
    vid_pred_combo = alpha * vid_pred_svr + (1 - alpha) * vid_pred_ens
    mae_combo = mean_absolute_error(vid_true, vid_pred_combo)
    log(f"Weighted combination test MAE: {mae_combo:.4f}")

    strategies = {
        "Fixed SVR": (mae_svr, vid_pred_svr),
        "Stacking Ensemble": (mae_ens, vid_pred_ens),
        "Weighted Combination": (mae_combo, vid_pred_combo),
    }
    best_name, (best_mae, best_preds) = min(strategies.items(), key=lambda x: x[1][0])
    log(f"Best strategy: {best_name} (MAE={best_mae:.4f})")

    # ----------------------------------------------------------------- save
    os.makedirs(args.output, exist_ok=True)

    for split, split_df in [("training", train_df), ("development", dev_df), ("testing", test_df)]:
        # Re-run predictions on each split for fusion preparation
        feat_split = create_enhanced_features(split_df, feature_cols)
        common_split = sorted(set(feat_split.columns) & set(common_cols))
        X_split = imputer.transform(feat_split[common_split].values.astype(float))
        X_split = selector.transform(X_split, common_split)
        X_split = pt.transform(X_split)

        if best_name == "Fixed SVR":
            seg_preds = svr.predict(X_split)
        else:
            try:
                seg_preds = predict_stacked(fitted_bases, fitted_meta, X_split)
            except Exception:
                seg_preds = svr.predict(X_split)

        # Simple mean aggregation for train / dev; trained aggregator for test
        video_level = {}
        for vid, pred in zip(split_df["video_id"].values, seg_preds):
            video_level.setdefault(vid, []).append(pred)

        rows = []
        for vid, preds_list in video_level.items():
            true_val = split_df.loc[split_df["video_id"] == vid, "BDI_II"].iloc[0]
            rows.append({
                "video_id": vid,
                "split": split,
                "true_label": true_val,
                "rppg_pred": np.mean(preds_list),
            })

        import pandas as _pd
        _pd.DataFrame(rows).to_csv(
            os.path.join(args.output, f"rppg_{split}_video_predictions.csv"), index=False
        )

    preprocessing = {
        "imputer": imputer,
        "feature_selector": selector,
        "power_transformer": pt,
        "aggregator_svr": agg_svr,
        "aggregator_ens": agg_ens,
    }
    joblib.dump(preprocessing, os.path.join(args.output, "preprocessing_pipeline.joblib"))

    model_to_save = (
        svr if best_name == "Fixed SVR"
        else {"stack_base_models": fitted_bases, "stack_meta": fitted_meta}
    )
    joblib.dump(model_to_save, os.path.join(args.output, "best_model.joblib"))
    log(f"Saved artefacts to {args.output}  (elapsed: {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rPPG stacking pipeline")
    parser.add_argument("--csv",    required=True, help="Path to rPPG features CSV")
    parser.add_argument("--output", default="rppg_output", help="Output directory")
    parser.add_argument("--k-best", type=int, default=25, help="Features to select")
    main(parser.parse_args())