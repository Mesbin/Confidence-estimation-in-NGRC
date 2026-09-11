"""
=====================================================================
Study 2, Module 5 : S2_Module_5_results.py

=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from S2_Module_1_realtime_reservoir import apply_style, save_figure, BLUE, ORANGE, GREY
from S2_Module_4_early_decision import sustained_commit, evaluate, TARGET_ACCURACY

if __name__ == "__main__":
    apply_style()
    print("=" * 70)
    print("STUDY 2, MODULE 5 : results")
    print("=" * 70)

    t = np.load("Data/rt_time.npy")
    correct = np.load("Data/rt_correct.npy")
    idx = np.load("Data/rt_decision_idx.npy")
    sweep = pd.read_csv("Results/S2_threshold_sweep.csv")
    summary = pd.read_csv("Results/S2_operating_point.csv").iloc[0]
    thr = float(summary.threshold_fold0)
    dwell = float(summary.dwell_fold0)
    full_acc = float(summary.full_trial_accuracy)

    # --- figure 7 : accuracy and coverage against the confidence threshold
    plt.figure(figsize=(9, 5.6))
    plt.plot(sweep.threshold, sweep.accuracy * 100, "o-", color=BLUE, label="accuracy")
    plt.plot(sweep.threshold, sweep.coverage * 100, "s--", color=ORANGE, label="coverage")
    plt.axhline(full_acc * 100, ls=":", color=GREY,
                label=f"full-trial accuracy {full_acc*100:.1f}%")
    plt.axvline(thr, ls="--", color="k", lw=2, label=f"chosen threshold {thr:.2f}")
    plt.xlabel("Confidence threshold $C_{th}$"); plt.ylabel("Per cent")
    plt.title(f"Higher confidence buys accuracy and costs coverage\n(dwell D = {dwell:.2f} s)")
    plt.legend(fontsize=12)
    save_figure("Results/S2_fig07_threshold.png")

    # --------------------------------------------------------------- summary
    out = pd.DataFrame([
        ("Full-trial accuracy (Study 1 protocol)", f"{full_acc*100:.2f}%"),
        ("Trial length", f"{t[-1]:.2f} s"),
        ("Reliability target", f"{TARGET_ACCURACY:.0%}"),
        ("Chosen dwell D (nested, per fold)",
         f"{summary.dwell_fold0:.2f} / {summary.dwell_fold1:.2f} / {summary.dwell_fold2:.2f} s"),
        ("Chosen threshold (nested, per fold)",
         f"{summary.threshold_fold0:.2f} / {summary.threshold_fold1:.2f} / "
         f"{summary.threshold_fold2:.2f}"),
        ("Wrong-command rate, full trial", f"{(1-full_acc)*100:.1f}%"),

        ("Coverage (commands issued)", f"{float(summary.coverage)*100:.1f}%"),
        ("Accuracy of issued commands", f"{float(summary.accuracy)*100:.1f}%"),
        ("Wrong-command rate (of all trials)", f"{float(summary.wrong_rate)*100:.1f}%"),
        ("No command (of all trials)", f"{float(summary.rejection)*100:.1f}%"),
        ("Median decision time", f"{float(summary.median_T):.2f} s"),
        ("Spread of decision times", f"SD {float(summary.decision_time_sd):.2f} s"),
        ("Mean decision time", f"{float(summary.mean_T):.2f} s"),
        ("Time saved per command", f"{float(summary.time_saved):.2f} s"),
    ], columns=["Quantity", "Value"])
    out.to_csv("Results/S2_final_summary.csv", index=False)
    print(out.to_string(index=False))
    print("\nStudy 2, Module 5 Completed Successfully.")
