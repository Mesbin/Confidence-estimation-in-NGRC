"""
=====================================================================
Study 2, Module 3 : S2_Module_3_confidence_over_time.py

Estimates the confidence continuously from the reservoir state:

    X_bar(t)  ->  C_hat(t)

Study 1 estimated one confidence per trial. Here the same estimator form is
applied at every decision opportunity, so the confidence can be monitored while
the trial is still running.

Leakage control, unchanged in spirit from Study 1 and applied per fold:
    1. the classifier is trained on the 6 training subjects only
    2. their reference confidence C_ref(t) comes from an INNER leave-one-
       subject-out loop inside those 6, so no training target was produced by a
       model that had seen the 3 unseen subjects
    3. the confidence variables are selected on the training subjects only
    4. the estimator is fitted on the training subjects only
    5. the unseen subjects are touched once, for scoring

Selection and estimator form are those of Study 1: Spearman identifies,
Kendall confirms, mutual information adds a nonlinear criterion, and the
estimator is a ridge on the quadratic expansion of the selected variables. The
only difference is that the training rows are (trial, time) pairs rather than
one row per trial, so the estimator is valid at any point in the trial.

Output : Data/rt_chat.npy   (n_trials, n_timepoints)
=====================================================================
"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.stats import spearmanr, kendalltau, rankdata, false_discovery_control
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from S2_Module_1_realtime_reservoir import (apply_style, save_figure, subject_folds,
                                            BLUE, ORANGE, GREEN, GREY)
from S2_Module_2_realtime_readout import fit_readout

TOPK = 26
RIDGE_ALPHA = 100.0
SUBSAMPLE = 5          # rows used for the selection statistics, for speed


def fast_auc(score, label):
    r = rankdata(score)
    n1 = int(np.sum(label == 1)); n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def quad_expand(Z):
    idx = list(combinations_with_replacement(range(Z.shape[1]), 2))
    i = np.array([a for a, b in idx]); j = np.array([b for a, b in idx])
    return np.hstack([Z, Z[:, i] * Z[:, j]])


def select_confidence_variables(R, C, spearman_threshold=0.05, q_threshold=0.05,
                                mi_percentile=25, topk=TOPK, random_state=0):
    """The Study 1 criteria, applied to (trial, time) rows."""
    P = R.shape[1]
    sp = np.zeros(P); spp = np.ones(P); ktp = np.ones(P)
    for i in range(P):
        if R[:, i].std() > 0:
            sp[i], spp[i] = spearmanr(R[:, i], C)
            ktp[i] = kendalltau(R[:, i], C)[1]
    sp = np.nan_to_num(sp)
    spq = false_discovery_control(np.nan_to_num(spp, nan=1.0))
    ktq = false_discovery_control(np.nan_to_num(ktp, nan=1.0))
    mi = mutual_info_regression(R, C, random_state=random_state)
    mi_cut = np.percentile(mi, mi_percentile)
    keep = np.where((np.abs(sp) >= spearman_threshold) & (spq < q_threshold)
                    & (ktq < q_threshold) & (mi >= mi_cut))[0]
    if len(keep) < 5:
        keep = np.argsort(-np.abs(sp))[:topk]
    return keep[np.argsort(-np.abs(sp[keep]))][:topk]


if __name__ == "__main__":
    apply_style()
    print("=" * 70)
    print("STUDY 2, MODULE 3 : confidence estimated at every time point")
    print("=" * 70)

    state = np.load("Data/rt_state.npy")
    t = np.load("Data/rt_time.npy")
    y = np.load("Data/labels.npy")
    subjects = np.load("Data/subjects.npy")
    correct = np.load("Data/rt_correct.npy")
    n, nT, P = state.shape

    chat = np.full((n, nT), np.nan, np.float32)

    for fold, unseen in enumerate(subject_folds(subjects)):
        te = np.isin(subjects, unseen); tr = ~te

        # --- step 2 : inner leave-one-subject-out reference confidence
        cref_in = np.full((n, nT), np.nan, np.float32)
        for hs in np.unique(subjects[tr]):
            held = subjects == hs
            cref_in[held] = np.abs(2 * fit_readout(state, y, tr & ~held, held) - 1)

        # --- step 3 : selection on the training subjects only
        Ztr = state[tr].reshape(-1, P)
        ctr = cref_in[tr].reshape(-1)
        sel = select_confidence_variables(Ztr[::SUBSAMPLE], ctr[::SUBSAMPLE])

        # --- step 4 : estimator on the training subjects only
        s1 = StandardScaler().fit(Ztr[:, sel])
        Q = quad_expand(s1.transform(Ztr[:, sel]))
        s2 = StandardScaler().fit(Q)
        g = Ridge(alpha=RIDGE_ALPHA).fit(s2.transform(Q), ctr)

        # --- step 5 : apply once to the unseen subjects
        Zte = state[te].reshape(-1, P)
        chat[te] = g.predict(s2.transform(quad_expand(s1.transform(Zte[:, sel])))).reshape(
            int(te.sum()), nT)
        print(f"  fold {fold}: {len(sel)} variables selected from "
              f"{int(tr.sum())*nT:,} (trial, time) training rows "
              f"-> applied to subjects {sorted(unseen.tolist())}")

    np.save("Data/rt_chat.npy", chat)

    auc_t = np.array([fast_auc(chat[:, j], correct[:, j]) for j in range(nT)])
    print(f"\n  AUROC2 of C_hat(t) against 'would the decision at t be correct':")
    print(f"  {'t (s)':>6} {'AUROC2':>8}")
    for j in range(0, nT, max(1, nT // 9)):
        print(f"  {t[j]:6.2f} {auc_t[j]:8.4f}")
    print(f"\n  mean over the trial : {np.nanmean(auc_t):.4f}   "
          f"(Study 1, full trial only: 0.6149)")

    pd.DataFrame({"time_s": t, "AUROC2": auc_t}).to_csv(
        "Results/S2_confidence_auroc_over_time.csv", index=False)

    # --- figure 5 : how informative the confidence estimate is at each time
    plt.figure(figsize=(9, 5.6))
    plt.plot(t, auc_t, color=BLUE)
    plt.axhline(0.5, ls=":", color=GREY, label="chance")
    plt.xlabel("Time within trial (s)"); plt.ylabel("AUROC2 of $\\hat{C}(t)$")
    plt.title("The confidence estimate is informative\nthroughout the trial")
    plt.legend(fontsize=12)
    save_figure("Results/S2_fig05_confidence_auroc.png")

    # --- figure 6 : estimated confidence, split by eventual correctness
    plt.figure(figsize=(9, 5.6))
    for lab, colour, name in [(1, GREEN, "trials decided correctly"),
                              (0, ORANGE, "trials decided wrongly")]:
        m = correct[:, -1] == lab
        mu = np.nanmean(chat[m], axis=0)
        se = np.nanstd(chat[m], axis=0) / np.sqrt(m.sum())
        plt.plot(t, mu, color=colour, label=name)
        plt.fill_between(t, mu - se, mu + se, color=colour, alpha=0.25, linewidth=0)
    plt.xlabel("Time within trial (s)"); plt.ylabel(r"$\hat{C}(t)$")
    plt.title("Estimated confidence separates\ncorrect from incorrect trials early")
    plt.legend(fontsize=12)
    save_figure("Results/S2_fig06_confidence_by_outcome.png")

    print("\nStudy 2, Module 3 Completed Successfully.")
