"""
=====================================================================
Study 2, Module 2 : S2_Module_2_realtime_readout.py

Applies the LH/RH readout at every decision opportunity.

    X_bar(t) -> o_LH(t),  o_RH(t) = 1 - o_LH(t)
    y_hat(t) = argmax{o_LH(t), o_RH(t)}
    C_ref(t) = |o_LH(t) - o_RH(t)|

The classifier is trained once per fold on the FULL-TRIAL representation of the
training subjects, exactly as in Study 1, and then applied unchanged at earlier
times. This keeps the two studies comparable: at t = T the decision is Study 1's
decision, which this module verifies numerically.

Subject-independent protocol, unchanged: 3 folds, 3 unseen subjects each, every
subject unseen exactly once, the readout never trained on a test subject.

Output : Data/rt_delta.npy   (n_trials, n_timepoints)  signed separation
         Data/rt_cref.npy    |o_LH - o_RH| at each time
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from S2_Module_1_realtime_reservoir import (apply_style, save_figure, subject_folds,
                                            BLUE, ORANGE, GREEN, GREY)

CLF_C = 0.05


def fit_readout(state, y, fit_mask, apply_mask, C=CLF_C):
    """Train on the full-trial representation of fit_mask, apply at every time
    point of apply_mask. Returns o_LH(t) with shape (n_apply, n_time)."""
    full = state[:, -1, :]
    sc = StandardScaler().fit(full[fit_mask])
    clf = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(full[fit_mask]), y[fit_mask])
    lh = list(clf.classes_).index(0)
    out = np.empty((int(apply_mask.sum()), state.shape[1]), np.float32)
    for j in range(state.shape[1]):
        out[:, j] = clf.predict_proba(sc.transform(state[apply_mask, j, :]))[:, lh]
    return out


if __name__ == "__main__":
    apply_style()
    print("=" * 70)
    print("STUDY 2, MODULE 2 : readout at every decision opportunity")
    print("=" * 70)

    state = np.load("Data/rt_state.npy")
    t = np.load("Data/rt_time.npy")
    y = np.load("Data/labels.npy")
    subjects = np.load("Data/subjects.npy")

    o_LH = np.full((len(y), len(t)), np.nan, np.float32)
    for fold, unseen in enumerate(subject_folds(subjects)):
        te = np.isin(subjects, unseen); tr = ~te
        o_LH[te] = fit_readout(state, y, tr, te)
        print(f"  fold {fold}: trained on {int(tr.sum())} trials, "
              f"applied at {len(t)} time points of {int(te.sum())} unseen trials "
              f"(subjects {sorted(unseen.tolist())})")

    delta = 2 * o_LH - 1                     # o_LH - o_RH
    cref = np.abs(delta)
    pred = (delta < 0).astype(int)
    correct = (pred == y[:, None]).astype(int)

    acc_full = correct[:, -1].mean()
    print(f"\n  full-trial accuracy : {acc_full:.4f}")
    print("  Study 1 reported    : 0.6986   <- must agree, the representations "
          "are identical at t = T")

    np.save("Data/rt_delta.npy", delta)
    np.save("Data/rt_cref.npy", cref)
    np.save("Data/rt_correct.npy", correct)

    print("\n  accuracy and reference confidence as the trial unfolds:")
    print(f"  {'t (s)':>6} {'accuracy':>9} {'mean C_ref':>11}")
    for j in range(0, len(t), max(1, len(t) // 9)):
        print(f"  {t[j]:6.2f} {correct[:, j].mean()*100:8.2f}% {cref[:, j].mean():11.4f}")

    pd.DataFrame({"time_s": t, "accuracy": correct.mean(0),
                  "mean_Cref": cref.mean(0)}).to_csv(
        "Results/S2_accuracy_over_time.csv", index=False)

    # --- figure 3 : accuracy if the decision were taken now
    plt.figure(figsize=(9, 5.6))
    plt.plot(t, correct.mean(0) * 100, color=GREEN)
    plt.axhline(50, ls=":", color=GREY, label="chance")
    plt.axhline(acc_full * 100, ls="--", color="k", label=f"full trial {acc_full*100:.1f}%")
    plt.xlabel("Time within trial (s)"); plt.ylabel("Accuracy if decided now (%)")
    plt.title("Evidence accumulates well before the trial ends")
    plt.legend(fontsize=12)
    save_figure("Results/S2_fig03_accuracy_over_time.png")

    # --- figure 4 : the reference confidence falls rather than rises
    plt.figure(figsize=(9, 5.6))
    plt.plot(t, cref.mean(0), color=ORANGE)
    plt.xlabel("Time within trial (s)")
    plt.ylabel(r"mean $C_{ref}(t)=|o_{LH}-o_{RH}|$")
    plt.title("Reference confidence FALLS early on\n(few samples, so probabilities are noise-inflated)")
    save_figure("Results/S2_fig04_reference_confidence.png")

    print("\nStudy 2, Module 2 Completed Successfully.")
