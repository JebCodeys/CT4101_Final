import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

BENIGN = "Benign"


def scores(y, pred, p_attack):
    benign = y == BENIGN
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "bal_acc": balanced_accuracy_score(y, pred),
        "acc": accuracy_score(y, pred),
        "benign_fpr": float(np.mean(pred[benign] != BENIGN)),
        "attack_recall": float(np.mean(pred[~benign] != BENIGN)),
        "pr_auc": average_precision_score(~benign, p_attack),
    }


def p_attack(model, X):
    proba = model.predict_proba(X)
    return 1.0 - proba[:, list(model.classes_).index(BENIGN)]


def cv_evaluate(model, X, y, seed, n_splits=5, train_scores=True, sample_weight=None):
    """Stratified k-fold CV. `sample_weight` (per row of X) is passed to the final
    pipeline step's `fit` for the training fold only; scoring is always unweighted."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.empty(len(y), dtype=object)
    oof_pa = np.empty(len(y))
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        t = time.time()
        fit_params = {} if sample_weight is None else {"model__sample_weight": np.asarray(sample_weight)[tr]}
        m = clone(model).fit(X.iloc[tr], y[tr], **fit_params)
        pred, pa = m.predict(X.iloc[va]), p_attack(m, X.iloc[va])
        oof_pred[va], oof_pa[va] = pred, pa
        r = {"fold": k, **{f"val_{s}": v for s, v in scores(y[va], pred, pa).items()}}
        if train_scores:
            r["train_macro_f1"] = f1_score(y[tr], m.predict(X.iloc[tr]), average="macro")
        r["sec"] = time.time() - t
        rows.append(r)
    return pd.DataFrame(rows), oof_pred, oof_pa


def weighted_class_report(y, pred, w, labels):
    """Per-class precision / recall / F1 where every row counts `w` times.

    With importance weights from `background.scenario_weights`, this is the
    classification report the model would get on a day with that traffic mix.
    Recall is unchanged by re-weighting within a class; precision and F1 move."""
    y, pred, w = np.asarray(y), np.asarray(pred), np.asarray(w, dtype=np.float64)
    rows = {}
    for c in labels:
        tp = w[(y == c) & (pred == c)].sum()
        fp = w[(y != c) & (pred == c)].sum()
        fn = w[(y == c) & (pred != c)].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows[c] = {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
                   "weighted_support": tp + fn, "false_alarms_from_benign": w[(y == BENIGN) & (pred == c)].sum()}
    out = pd.DataFrame(rows).T
    out.attrs["macro_f1"] = float(out["f1"].mean())
    return out


def fit_holdout_benign(model, X, y, session, holdout, sample_weight=None):
    """Fit on every row except the benign flows of `holdout`; return their predicted
    attack probabilities and labels. Measures how the benign model transfers to a
    day it never saw."""
    session = np.asarray(session)
    y = np.asarray(y)
    held = (session == holdout) & (y == BENIGN)
    fit_params = {} if sample_weight is None else {"model__sample_weight": np.asarray(sample_weight)[~held]}
    m = clone(model).fit(X.iloc[np.flatnonzero(~held)], y[~held], **fit_params)
    Xh = X.iloc[np.flatnonzero(held)]
    return m.predict(Xh), p_attack(m, Xh)
