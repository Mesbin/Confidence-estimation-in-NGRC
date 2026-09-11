"""

Module 11 : Module_11_study_comparison.py      [DROP-IN, changes nothing else]

Compares the two studies side by side in a single figure:

    STUDY 1 (original)  C3 + Cz + C4,  9 subjects,  55 reservoir variables
    STUDY 2 (new)       C3 + Cz + C4,  8 subjects,  55 reservoir variables
                        Cz KEPT, Subject 4 removed

This file is self-contained. It rebuilds BOTH configurations internally from
the raw data, so it can be dropped into either pipeline folder and will give
the same answer in both. It imports only the figure style from Module 1 and the
canonical selection criteria from Module 6, so both studies are evaluated with
byte-for-byte identical rules.

Every stage is the pipeline's own: band-pass 8-30 Hz, Euclidean Alignment,
delay embedding (k = 3, s = 3), signed-log compression, subject-level folds,
logistic readout, inner leave-one-subject-out confidence target, Spearman +
Kendall + mutual information selection, quadratic ridge estimator.

Output : Results/study_comparison.csv and ONE figure,
         Results/study_comparison.png
"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from Module_1_eeg_data import apply_style, save_figure
from Module_6_feature_selection import select_confidence_variables

FS = 250.0
BAND = (8, 30)
MI_START, MI_END = 125, 1000
K, S = 3, 3
CH_NAMES = ("C3", "Cz", "C4")

STUDIES = [
    ("Study 1  (C3+Cz+C4, 9 subjects)", [0, 1, 2], None),
    ("Study 2  (C3+Cz+C4, 8 subjects)", [0, 1, 2], 4),
]


# ------------------------------------------------------------------ helpers
def fast_auc(score, label):
    r = rankdata(score)
    n1 = int(np.sum(label == 1)); n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def quad_expand(Z):
    idx = list(combinations_with_replacement(range(Z.shape[1]), 2))
    i = np.array([a for a, b in idx]); j = np.array([b for a, b in idx])
    return np.hstack([Z, Z[:, i] * Z[:, j]])


def subject_folds(subjects, seed=0):
    ids = np.unique(subjects)
    return np.array_split(np.random.default_rng(seed).permutation(ids), 3)


def bandpass(X, band=BAND):
    sos = butter(4, list(band), btype="band", fs=FS, output="sos")
    Xf = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
    return Xf / np.sqrt((Xf ** 2).mean())


def euclidean_align(X, subjects):
    Xa = np.empty_like(X)
    for s in np.unique(subjects):
        m = subjects == s
        Xs = X[m].astype(np.float64)
        R = np.einsum("nct,ndt->cd", Xs, Xs) / (Xs.shape[0] * Xs.shape[2])
        w, V = np.linalg.eigh(R)
        Xa[m] = np.einsum("cd,ndt->nct",
                          V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T,
                          Xs).astype(X.dtype)
    return Xa


def build_reservoir(X, channels, k=K, s=S):
    Xc = X[:, channels, :]
    nch = len(channels)
    start = max(MI_START, (k - 1) * s)
    D = np.stack([Xc[:, :, start - j*s: MI_END - j*s] for j in range(k)], 1)
    D = D.reshape(len(Xc), k * nch, -1).astype(np.float64)
    kc = k * nch
    q = np.einsum("nit,njt->nij", D, D) / D.shape[-1]
    iu = np.triu_indices(kc)
    R = np.hstack([np.ones((len(Xc), 1)), D.mean(-1), q[:, iu[0], iu[1]]])
    scale = np.median(np.abs(R), 0, keepdims=True) + 1e-12
    R = np.sign(R) * np.log1p(np.abs(R) / scale); R[:, 0] = 1.0
    return R


def run_study(R, y, subjects):
    folds = subject_folds(subjects)
    n = len(y)
    delta = np.full(n, np.nan)
    for unseen in folds:
        te = np.isin(subjects, unseen); tr = ~te
        sc = StandardScaler().fit(R[tr])
        clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(R[tr]), y[tr])
        lh = list(clf.classes_).index(0)
        delta[te] = 2 * clf.predict_proba(sc.transform(R[te]))[:, lh] - 1
    correct = ((delta < 0).astype(int) == y).astype(int)
    conf = np.abs(delta)
    fold_acc = np.array([correct[np.isin(subjects, u)].mean() for u in folds])
    sub_acc = np.array([correct[subjects == s].mean() for s in np.unique(subjects)])

    hat = np.full(n, np.nan)
    for unseen in folds:
        te = np.isin(subjects, unseen); tr = ~te
        conf_in = np.full(n, np.nan)
        for hs in np.unique(subjects[tr]):
            held = subjects == hs
            fit = tr & ~held
            sc = StandardScaler().fit(R[fit])
            c = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(R[fit]), y[fit])
            lh = list(c.classes_).index(0)
            conf_in[held] = np.abs(2 * c.predict_proba(sc.transform(R[held]))[:, lh] - 1)
        sel = select_confidence_variables(R[tr], conf_in[tr])
        s1 = StandardScaler().fit(R[np.ix_(tr, sel)])
        Q = quad_expand(s1.transform(R[:, sel]))
        s2 = StandardScaler().fit(Q[tr])
        g = Ridge(alpha=100.0).fit(s2.transform(Q[tr]), conf_in[tr])
        hat[te] = g.predict(s2.transform(Q[te]))

    order = np.argsort(-hat)
    return dict(n_subjects=len(np.unique(subjects)), n_trials=n,
                n_features=R.shape[1],
                accuracy=correct.mean(), fold_sd=fold_acc.std(ddof=1),
                subject_sd=sub_acc.std(ddof=1),
                ref_AUROC2=fast_auc(conf, correct), AUROC2=fast_auc(hat, correct),
                acc25=correct[order][:n // 4].mean(),
                abs_bias=abs(cohens_d(hat[y == 0], hat[y == 1])))


if __name__ == "__main__":
    apply_style()
    print("=" * 78)
    print("MODULE 11 : STUDY 1 versus STUDY 2")
    print("=" * 78)

    X0 = np.load("Data/X_epochs.npy").astype(np.float32) * 1e6
    subj_raw = pd.read_csv("Data/epoch_labels.csv")["Subject"].values
    y_raw = np.load("Data/y_labels.npy")

    rows = []
    for label, chans, drop in STUDIES:
        keep = np.ones(len(y_raw), bool) if drop is None else subj_raw != drop
        Xs, ys, ss = X0[keep], y_raw[keep], subj_raw[keep]
        Xa = euclidean_align(bandpass(Xs), ss)
        r = run_study(build_reservoir(Xa, chans), ys, ss)
        r["Study"] = label
        rows.append(r)
        print(f"\n  {label}")
        print(f"    {r['n_subjects']} subjects, {r['n_trials']} trials, "
              f"{r['n_features']} reservoir variables")
        print(f"    accuracy           {r['accuracy']*100:.2f}%  "
              f"+/- {r['fold_sd']*100:.2f} (folds)  +/- {r['subject_sd']*100:.2f} (subjects)")
        print(f"    reference AUROC2   {r['ref_AUROC2']:.4f}")
        print(f"    confidence AUROC2  {r['AUROC2']:.4f}")
        print(f"    accuracy @25% cov  {r['acc25']*100:.1f}%")
        print(f"    |LH/RH bias d|     {r['abs_bias']:.4f}")

    df = pd.DataFrame(rows)[["Study", "n_subjects", "n_trials", "n_features",
                             "accuracy", "fold_sd", "subject_sd", "ref_AUROC2",
                             "AUROC2", "acc25", "abs_bias"]]
    df.to_csv("Results/study_comparison.csv", index=False)

    # =================================================================== FIGURE
    # Six quantities on one axis by expressing each as a percentage of Study 1,
    # so improvements and costs are directly comparable. The real values are
    # printed on the bars, and each label states which direction is better.
    a, b = rows[0], rows[1]
    metrics = [
        ("Accuracy\n(higher better)",          a["accuracy"],   b["accuracy"],   "{:.1%}", +1),
        ("Confidence AUROC2\n(higher better)", a["AUROC2"],     b["AUROC2"],     "{:.3f}", +1),
        ("Accuracy @25% cov\n(higher better)", a["acc25"],      b["acc25"],      "{:.1%}", +1),
        ("SD across folds\n(lower better)",    a["fold_sd"],    b["fold_sd"],    "{:.1%}", -1),
        ("SD across subjects\n(lower better)", a["subject_sd"], b["subject_sd"], "{:.1%}", -1),
        ("|LH/RH bias d|\n(lower better)",     a["abs_bias"],   b["abs_bias"],   "{:.3f}", -1),
    ]

    x = np.arange(len(metrics)); w = 0.36
    rel = [100.0 * m[2] / m[1] for m in metrics]
    better = [(m[4] > 0 and r > 100) or (m[4] < 0 and r < 100) for m, r in zip(metrics, rel)]

    # One bar can run far above the rest (the bias ratio), which would squash
    # everything else, so the axis is capped and any overflowing bar is drawn
    # clipped with its true value marked.
    CAP = 165.0
    drawn = [min(r, CAP - 8) for r in rel]

    plt.figure(figsize=(13.5, 6.8))
    plt.bar(x - w/2, [100] * len(metrics), w, color="#9aa4ad", label=rows[0]["Study"])
    plt.bar(x + w/2, drawn, w,
            color=["#1b7f5a" if ok else "#b3202c" for ok in better],
            label=rows[1]["Study"] + "   (green = better, red = worse)")
    plt.axhline(100, color="black", linewidth=1.4)

    for i, (m, r) in enumerate(zip(metrics, rel)):
        plt.text(i - w/2, 102, m[3].format(m[1]), ha="center", fontsize=12, color="#3f4a57")
        clipped = r > CAP - 8
        plt.text(i + w/2, drawn[i] + 3,
                 m[3].format(m[2]) + (f"\n$\\uparrow$ {r:.0f}%" if clipped else ""),
                 ha="center", fontsize=12, fontweight="bold",
                 color="#1b7f5a" if better[i] else "#b3202c")
        if clipped:                       # break marker on the clipped bar
            plt.plot([i + w/2 - w/3, i + w/2 + w/3],
                     [drawn[i] - 3, drawn[i] + 1], color="white", lw=3)
            plt.plot([i + w/2 - w/3, i + w/2 + w/3],
                     [drawn[i] - 7, drawn[i] - 3], color="white", lw=3)

    plt.xticks(x, [m[0] for m in metrics], fontsize=12)
    plt.ylabel("Value as a percentage of Study 1")
    plt.ylim(0, CAP)
    plt.title("Removing Subject 4 (Cz kept): what improves and what it costs\n"
              f"Study 1: {a['n_features']} variables, {a['n_subjects']} subjects   |   "
              f"Study 2: {b['n_features']} variables, {b['n_subjects']} subjects")
    plt.legend(fontsize=12, loc="upper left")
    save_figure("Results/study_comparison.png")

    print("\n" + "=" * 78)
    for m, r, ok in zip(metrics, rel, better):
        name = m[0].split("\n")[0]
        print(f"  {name:22s} {m[3].format(m[1]):>8s} -> {m[3].format(m[2]):>8s}   "
              f"{r-100:+6.1f}%   {'better' if ok else 'worse'}")
    print("\nSaved Results/study_comparison.csv and Results/study_comparison.png")
    print("\nModule 11 Completed Successfully.")
