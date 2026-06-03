import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr


def concordance_ccc(y_true, y_pred):
    """
    Concordance Correlation Coefficient (CCC).

    CCC measures the agreement between two variables by considering both
    their correlation and how close they are to the 45-degree line of
    perfect concordance.

    Args:
        y_true: Array-like of ground-truth values.
        y_pred: Array-like of predicted values.

    Returns:
        float: CCC in the range [-1, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cov = np.mean((y_true - y_true.mean()) * (y_pred - y_pred.mean()))
    return (2 * cov) / (np.var(y_true) + np.var(y_pred) + (y_true.mean() - y_pred.mean()) ** 2)


def compute_all_metrics(y_true, y_pred):
    """
    Compute the full suite of regression metrics used in the paper.

    Args:
        y_true: Array-like of ground-truth BDI-II scores.
        y_pred: Array-like of predicted BDI-II scores.

    Returns:
        Dict with keys ``MAE``, ``RMSE``, ``PCC``, ``CCC``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    pcc, _ = pearsonr(y_true, y_pred)
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "PCC":  float(pcc) if not np.isnan(pcc) else 0.0,
        "CCC":  concordance_ccc(y_true, y_pred),
    }


def bdi_to_severity(score):
    """
    Map a continuous BDI-II score to a clinical severity label.

    Thresholds follow Beck et al. (1996):

    * 0–13   → Minimal
    * 14–19  → Mild
    * 20–28  → Moderate
    * 29+    → Severe

    Args:
        score: Numeric BDI-II score.

    Returns:
        str: One of ``"Minimal"``, ``"Mild"``, ``"Moderate"``, ``"Severe"``.
    """
    if score <= 13:
        return "Minimal"
    elif score <= 19:
        return "Mild"
    elif score <= 28:
        return "Moderate"
    return "Severe"