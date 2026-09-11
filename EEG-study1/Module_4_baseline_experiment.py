"""

Module 4 : Module_4_baseline_experiment.py

Baseline performance of the NGRC classifier on UNSEEN subjects, and the
verification that the whole method rests on:

    when the two readout activities are CLOSER, the decision is MORE often wrong

If that relationship did not hold, |o_LH - o_RH| would be worthless as a
confidence signal and there would be nothing to estimate in Module 5 onward.

Equivalent role to Module 4 in the Lorenz-63 pipeline (prediction vs truth,
error over time) but with classification metrics.

"""

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure
from sklearn.metrics import confusion_matrix


def plot_accuracy_per_subject(subjects, correct, filename="Results/baseline_accuracy.png"):
    subs = np.unique(subjects)
    acc = np.array([correct[subjects == s].mean() for s in subs])
    n = np.array([(subjects == s).sum() for s in subs])
    se = np.sqrt(acc * (1 - acc) / n)          # binomial standard error per subject
    plt.figure(figsize=(11, 5.5))
    bars = plt.bar([str(s) for s in subs], acc, yerr=se, color="steelblue")
    plt.axhline(0.5, color="grey", linestyle=":", label="chance")
    plt.axhline(correct.mean(), color="black", linestyle="--",
                label=f"mean {correct.mean()*100:.1f}%")
    for b, a, e in zip(bars, acc, se):
        plt.text(b.get_x() + b.get_width()/2, a + e + 0.015, f"{a*100:.0f}%",
                 ha="center", fontsize=12)
    plt.ylim(0.4, 1.0)
    plt.xlabel("Subject (held out as UNSEEN)"); plt.ylabel("Accuracy")
    plt.title("NGRC Baseline : LH/RH Classification on Unseen Subjects")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved {filename}")


def plot_separation_vs_accuracy(delta, correct, filename="Results/separation_vs_accuracy.png"):
    """THE key premise plot: closer readouts -> more errors."""
    sep = np.abs(delta)
    q = pd.qcut(sep, 10, labels=False)
    acc = np.array([correct[q == i].mean() for i in range(10)])
    se = np.array([correct[q == i].std() / np.sqrt((q == i).sum()) for i in range(10)])
    plt.figure(figsize=(11, 5.5))
    plt.errorbar(range(1, 11), acc, yerr=se, marker="o", linewidth=1.5, color="seagreen")
    plt.axhline(correct.mean(), color="black", linestyle="--",
                label=f"overall {correct.mean()*100:.1f}%")
    plt.xlabel(r"Decile of $|\Delta|$  (1 = readouts closest together)")
    plt.ylabel("Accuracy")
    plt.title("Closer Readout Activities -> Lower Accuracy\n(the premise of the confidence method, verified)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Saved {filename}")


if __name__ == "__main__":
    apply_style()
    delta = np.load("Data/readout_delta.npy")
    y = np.load("Data/eeg_labels.npy")
    subjects = np.load("Data/eeg_subjects.npy")

    pred = (delta < 0).astype(int)          # Delta > 0 -> LH (label 0)
    correct = (pred == y).astype(int)

    acc = correct.mean()
    sep = np.abs(delta)
    lh_acc = correct[y == 0].mean()
    rh_acc = correct[y == 1].mean()

    fold_id = np.load("Data/fold_id.npy")
    fold_acc = np.array([correct[fold_id == f].mean() for f in np.unique(fold_id)])
    sub_acc = np.array([correct[subjects == s].mean() for s in np.unique(subjects)])

    print("Baseline NGRC Classification Performance (UNSEEN subjects)")
    print("  Accuracy          :", round(acc, 4))
    print(f"  across 3 folds    : {fold_acc.mean()*100:.2f}% "
          f"+/- {fold_acc.std(ddof=1)*100:.2f} (SD), "
          f"SEM {fold_acc.std(ddof=1)/np.sqrt(len(fold_acc))*100:.2f}")
    print(f"  across 9 subjects : {sub_acc.mean()*100:.2f}% "
          f"+/- {sub_acc.std(ddof=1)*100:.2f} (SD), "
          f"SEM {sub_acc.std(ddof=1)/np.sqrt(len(sub_acc))*100:.2f}")
    print(f"  subject range     : {sub_acc.min()*100:.1f}% to {sub_acc.max()*100:.1f}%")
    print("  LH accuracy       :", round(lh_acc, 4))
    print("  RH accuracy       :", round(rh_acc, 4))
    print("  Mean |Delta|      :", round(sep.mean(), 4))
    print("  |Delta| correct   :", round(sep[correct == 1].mean(), 4))
    print("  |Delta| errors    :", round(sep[correct == 0].mean(), 4))

    np.save("Data/correct.npy", correct)

    pd.DataFrame([{"Accuracy": acc, "LH_accuracy": lh_acc, "RH_accuracy": rh_acc,
                   "mean_abs_delta": sep.mean(),
                   "mean_abs_delta_correct": sep[correct == 1].mean(),
                   "mean_abs_delta_error": sep[correct == 0].mean(),
                   "SD_across_folds": fold_acc.std(ddof=1),
                   "SEM_across_folds": fold_acc.std(ddof=1)/np.sqrt(len(fold_acc)),
                   "SD_across_subjects": sub_acc.std(ddof=1),
                   "SEM_across_subjects": sub_acc.std(ddof=1)/np.sqrt(len(sub_acc))}]).to_csv(
        "Results/baseline_metrics.csv", index=False)
    print("Saved Results/baseline_metrics.csv")

    per_sub = pd.DataFrame({"subject": np.unique(subjects),
                            "accuracy": [correct[subjects == s].mean()
                                         for s in np.unique(subjects)]})
    per_sub.to_csv("Results/baseline_per_subject.csv", index=False)
    print("Saved Results/baseline_per_subject.csv")

    plot_accuracy_per_subject(subjects, correct)
    plot_separation_vs_accuracy(delta, correct)

    print("\nModule 4 Completed Successfully.")
