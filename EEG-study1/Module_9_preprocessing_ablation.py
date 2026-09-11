"""

Module 9 : Module_9_preprocessing_ablation.py

Does the preprocessing actually earn its place?

Four conditions, everything downstream held identical:

    A. raw            : no band-pass, no Euclidean Alignment
    B. band-pass only : 8-30 Hz
    C. alignment only : Euclidean Alignment on unfiltered data
    D. both           : 8-30 Hz + Euclidean Alignment   (the pipeline default)

All four use the SAME global amplitude scaling, so the comparison isolates the
filtering and the alignment rather than a scale artefact.

    - LH/RH classification accuracy on unseen subjects
    - leakage-free confidence AUROC2
    - LH/RH bias of the confidence: mean shift (Cohen's d) AND distributional
      separability (AUC, permutation p), since these can disagree

"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib

import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure
from Module_6_feature_selection import select_confidence_variables

FS = 250.0
MI_START, MI_END = 125, 1000
K, S = 3, 3


# ----------------------------------------------------------------- helpers
def fast_auc(score, label):
    r = rankdata(score)
    n1 = int(np.sum(label == 1)); n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def perm_p_auc(score, label, groups, B=3000, seed=0):
    rng = np.random.default_rng(seed)
    r = rankdata(score)
    n1 = int(np.sum(label == 1)); n0 = len(label) - n1
    obs = abs((r[label == 1].sum() - n1*(n1+1)/2) / (n1*n0) - 0.5)
    gi = [np.where(groups == g)[0] for g in np.unique(groups)]
    cnt = 0; L = label.copy()
    for _ in range(B):
        for ii in gi:
            L[ii] = rng.permutation(label[ii])
        cnt += (abs((r[L == 1].sum() - n1*(n1+1)/2) / (n1*n0) - 0.5) >= obs)
    return (cnt + 1) / (B + 1)


def quad_expand(Z):
    idx = list(combinations_with_replacement(range(Z.shape[1]), 2))
    i = np.array([a for a, b in idx]); j = np.array([b for a, b in idx])
    return np.hstack([Z, Z[:, i] * Z[:, j]])


def subject_folds(subjects, seed=0):
    ids = np.unique(subjects)
    return np.array_split(np.random.default_rng(seed).permutation(ids), 3)


# ------------------------------------------------------- preprocessing steps
def bandpass(X, band=(8, 30)):
    sos = butter(4, list(band), btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def euclidean_align(X, subjects):
    Xa = np.empty_like(X)
    for s in np.unique(subjects):
        m = subjects == s
        Xs = X[m].astype(np.float64)
        R = np.einsum("nct,ndt->cd", Xs, Xs) / (Xs.shape[0] * Xs.shape[2])
        w, V = np.linalg.eigh(R)
        Rm12 = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        Xa[m] = np.einsum("cd,ndt->nct", Rm12, Xs).astype(X.dtype)
    return Xa


def global_scale(X):
    """Same amplitude normalisation in every condition, so the comparison is
    about filtering and alignment, not about scale."""
    return X / np.sqrt((X.astype(np.float64) ** 2).mean())


def ngrc_state(X, k=K, s=S):
    start = max(MI_START, (k - 1) * s)
    D = np.stack([X[:, :, start - j*s: MI_END - j*s] for j in range(k)], 1)
    D = D.reshape(len(X), k * X.shape[1], -1).astype(np.float64)
    q = np.einsum("nit,njt->nij", D, D) / D.shape[-1]
    iu = np.triu_indices(k * X.shape[1])
    R = np.hstack([np.ones((len(X), 1)), D.mean(-1), q[:, iu[0], iu[1]]])
    scale = np.median(np.abs(R), 0, keepdims=True) + 1e-12
    R = np.sign(R) * np.log1p(np.abs(R) / scale)
    R[:, 0] = 1.0
    return R


# --------------------------------------------------- leakage-free evaluation
def evaluate_condition(R, y, subjects):
    ids = np.unique(subjects)

    def delta_from(fit_m, app_m):
        sc = StandardScaler().fit(R[fit_m])
        clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(R[fit_m]), y[fit_m])
        lh = list(clf.classes_).index(0)
        return 2 * clf.predict_proba(sc.transform(R[app_m]))[:, lh] - 1

    delta = np.full(len(y), np.nan)
    C_hat = np.full(len(y), np.nan)

    for unseen in subject_folds(subjects):
        tr = ~np.isin(subjects, unseen); te = np.isin(subjects, unseen)
        tr_subs = np.setdiff1d(ids, unseen)
        delta[te] = delta_from(tr, te)

        d_in = np.full(len(y), np.nan)
        for ho in tr_subs:
            ho_m = subjects == ho
            d_in[ho_m] = delta_from(tr & ~ho_m, ho_m)
        C_in = np.abs(d_in)

        sel = select_confidence_variables(R[tr], C_in[tr], topk=26)
        sc = StandardScaler().fit(R[np.ix_(tr, sel)])
        Xq = quad_expand(sc.transform(R[:, sel]))
        sc2 = StandardScaler().fit(Xq[tr])
        m = Ridge(alpha=100.0).fit(sc2.transform(Xq[tr]), C_in[tr])
        C_hat[te] = m.predict(sc2.transform(Xq[te]))

    correct = ((delta < 0).astype(int) == y).astype(int)
    a, b = C_hat[y == 0], C_hat[y == 1]
    return {"accuracy": correct.mean(),
            "AUROC2": fast_auc(C_hat, correct),
            "Spearman_vs_correct": spearmanr(C_hat, correct)[0],
            "bias_d": cohens_d(a, b),
            "bias_AUC": fast_auc(C_hat, y),
            "bias_perm_p": perm_p_auc(C_hat, y, subjects)}


if __name__ == "__main__":
    apply_style()
    from Module_1_eeg_data import EEGData

    # The raw file still holds all 3 channels and all 9 subjects, so apply the
    # two changes that define this study before anything else, exactly as
    # Module 1 does.
    X0 = np.load("Data/X_epochs.npy").astype(np.float32) * 1e6
    subj_raw = pd.read_csv("Data/epoch_labels.csv")["Subject"].values
    X0 = X0[:, EEGData.KEEP_CHANNELS, :]                    # drop Cz
    keep = subj_raw != EEGData.DROP_SUBJECT                  # drop Subject 4
    X0 = X0[keep]
    y = np.load("Data/eeg_labels.npy")
    subjects = np.load("Data/eeg_subjects.npy")
    assert len(X0) == len(y) == len(subjects), "trial counts disagree"
    print(f"  ablation runs on {X0.shape[1]} channels, "
          f"{len(np.unique(subjects))} subjects, {len(y)} trials\n")

    conditions = {
        "A. raw (no filter, no EA)":      lambda: global_scale(X0),
        "B. band-pass only (8-30 Hz)":    lambda: global_scale(bandpass(X0)),
        "C. alignment only (EA)":         lambda: global_scale(euclidean_align(X0, subjects)),
        "D. band-pass + EA (default)":    lambda: global_scale(euclidean_align(bandpass(X0), subjects)),
    }

    rows = []
    for name, fn in conditions.items():
        print(f"\n--- {name} ---")
        R = ngrc_state(fn())
        m = evaluate_condition(R, y, subjects)
        m["Condition"] = name
        rows.append(m)
        print(f"  accuracy {m['accuracy']:.4f} | confidence AUROC2 {m['AUROC2']:.4f} "
              f"| bias d {m['bias_d']:+.4f} (AUC {m['bias_AUC']:.4f}, perm p {m['bias_perm_p']:.4f})")

    df = pd.DataFrame(rows)[["Condition", "accuracy", "AUROC2", "Spearman_vs_correct",
                              "bias_d", "bias_AUC", "bias_perm_p"]]
    df.to_csv("Results/preprocessing_ablation.csv", index=False)
    print("\n" + "=" * 78)
    print("PREPROCESSING ABLATION  (all leakage-free, unseen subjects)")
    print("=" * 78)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    base = df[df.Condition.str.startswith("A")].iloc[0]
    best = df[df.Condition.str.startswith("D")].iloc[0]
    print(f"\n  band-pass + EA vs raw : accuracy {base.accuracy:.4f} -> {best.accuracy:.4f} "
          f"({100*(best.accuracy-base.accuracy):+.1f} pp)")
    print(f"                          AUROC2   {base.AUROC2:.4f} -> {best.AUROC2:.4f} "
          f"({best.AUROC2-base.AUROC2:+.4f})")
    print(f"                          |bias d| {abs(base.bias_d):.4f} -> {abs(best.bias_d):.4f}")

    print("\nSaved Results/preprocessing_ablation.csv")
    print("\nModule 9 Completed Successfully.")
