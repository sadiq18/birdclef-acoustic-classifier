import numpy as np
from sklearn.metrics import roc_auc_score


def compute_macro_auc(y_true, y_pred, mask=None, class_mask=None):
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    if class_mask is not None:
        y_true, y_pred = y_true[:, class_mask], y_pred[:, class_mask]
    aucs = []
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        if col.sum() == 0 or col.sum() == len(col):
            continue
        try:
            aucs.append(roc_auc_score(col, y_pred[:, c]))
        except ValueError:
            continue
    return (np.mean(aucs) if aucs else float("nan")), len(aucs)


def full_eval(y_true, y_pred, ns22, taxon_masks):
    results = {}
    a, n = compute_macro_auc(y_true, y_pred)
    results["macro_auc_all"], results["n_all"] = round(a, 4), n
    a, n = compute_macro_auc(y_true, y_pred, mask=ns22)
    results["non_s22_macro"], results["n_ns22"] = round(a, 4), n
    for t, cm in taxon_masks.items():
        if t == "Reptilia":
            continue
        a, _ = compute_macro_auc(y_true, y_pred, mask=ns22, class_mask=cm)
        results[f"non_s22_{t}"] = round(a, 4)
    return results
