"""

Module 6 : Module_6_feature_selection.py

*** DISCOVERY STAGE -- uses all 9 subjects (see the note in Module 5). ***
The selection produced here answers "which variables encode confidence in this
dataset?". It is NOT used to produce the headline performance number: Module 7
re-runs this exact selection procedure inside the training subjects of each
fold, so that the unseen subjects never influence which variables are chosen.

Selects the reservoir variables that encode confidence, mirroring Module 6 of
the Lorenz-63 pipeline but with three changes suited to the EEG task:

  1. Selection uses SPEARMAN (identify) confirmed by KENDALL (verify), both
     FDR-corrected, instead of Pearson. The confidence distribution is not
     normal, and the rank-based ranking is invariant to the monotone form of
     the confidence definition, whereas the Pearson ranking is not.

  2. Mutual information is kept as an additional nonlinear criterion.

  3. A CLASS-AGNOSTICISM filter is added, which the Lorenz-63 version has no
     need for. A variable is TRUE_CONF if it is confidence-encoding AND NOT
     class-encoding. This is the supervisor's definition of a confidence
     variable turned directly into a filter:
         "a true confidence-encoded neuron is agnostic to whether the person
          imagines moving left or right arm"

"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau, false_discovery_control
from sklearn.feature_selection import mutual_info_regression
import matplotlib

import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure

# ============================================================================
# THE CANONICAL CONFIDENCE-VARIABLE SELECTION
#
# This lives here, in the selection module, and is imported by Modules 7, 8 and
# 9 rather than being reimplemented. The discovery selection (all subjects,
# below) and the nested selection (training subjects only, Module 7) therefore
# apply byte-for-byte identical criteria and differ only in which subjects'
# data they see. Without that guarantee the leakage estimate in Module 7 would
# be comparing two different procedures.
#
# Criteria (all applied together):
#     |Spearman rho| >= spearman_threshold        identify
#     Spearman FDR q  <  q_threshold              multiplicity-corrected
#     Kendall  FDR q  <  q_threshold              confirm with a second rank test
#     mutual information >= percentile cut        nonlinear dependence
# ============================================================================


def selection_stats(R, C, random_state=0, compute_mi=True):
    """Per-variable association with the confidence target."""
    P = R.shape[1]
    sp = np.zeros(P); spp = np.ones(P)
    kt = np.zeros(P); ktp = np.ones(P)
    for i in range(P):
        if R[:, i].std() > 0:
            sp[i], spp[i] = spearmanr(R[:, i], C)
            kt[i], ktp[i] = kendalltau(R[:, i], C)
    sp = np.nan_to_num(sp); kt = np.nan_to_num(kt)
    spq = false_discovery_control(np.nan_to_num(spp, nan=1.0))
    ktq = false_discovery_control(np.nan_to_num(ktp, nan=1.0))
    mi = (mutual_info_regression(R, C, random_state=random_state)
          if compute_mi else np.full(P, np.inf))
    return {"spearman": sp, "spearman_q": spq,
            "kendall": kt, "kendall_q": ktq, "mi": mi}


def select_confidence_variables(R, C, spearman_threshold=0.05, q_threshold=0.05,
                                mi_percentile=25, topk=26, random_state=0,
                                compute_mi=True, return_stats=False):
    """Canonical selection. Returns indices ordered by |Spearman|, capped at topk.

    R : (n_trials, n_variables) reservoir state
    C : (n_trials,) confidence target

    Both must be restricted to the TRAINING subjects when used for validation.
    """
    st = selection_stats(R, C, random_state=random_state, compute_mi=compute_mi)
    mi_cut = (np.percentile(st["mi"], mi_percentile) if compute_mi else -np.inf)

    keep = np.where((np.abs(st["spearman"]) >= spearman_threshold)
                    & (st["spearman_q"] < q_threshold)
                    & (st["kendall_q"] < q_threshold)
                    & (st["mi"] >= mi_cut))[0]

    if len(keep) < 5:                      # identical fallback in every module
        keep = np.argsort(-np.abs(st["spearman"]))[:topk]

    idx = keep[np.argsort(-np.abs(st["spearman"][keep]))][:topk]
    return (idx, st, mi_cut) if return_stats else idx


class FeatureSelector:

    def __init__(self, spearman_threshold=0.05, q_threshold=0.05, mi_method="p25"):
        self.spearman_threshold = spearman_threshold
        self.q_threshold = q_threshold
        self.mi_method = mi_method

    # ==========================================================
    def load(self, folder="Data"):
        self.ranking = pd.read_csv("Results/feature_ranking.csv")
        self.reservoir = np.load(f"{folder}/ngrc_reservoir.npy")
        self.y = np.load(f"{folder}/eeg_labels.npy")
        print("Feature Ranking Loaded :", self.ranking.shape)
        print("Reservoir Loaded       :", self.reservoir.shape)

    # ==========================================================
    @staticmethod
    def _auc(score, label):
        r = rankdata(score)
        n1 = int(np.sum(label == 1)); n0 = len(label) - n1
        return (r[label == 1].sum() - n1*(n1+1)/2) / (n1*n0)

    # ==========================================================
    def class_selectivity(self):
        """AUC of each reservoir variable for predicting LH vs RH.
        0.5 means the variable cannot see the class at all."""
        auc = np.array([self._auc(self.reservoir[:, i], self.y)
                        if self.reservoir[:, i].std() > 0 else 0.5
                        for i in range(self.reservoir.shape[1])])
        n_min = min((self.y == 0).sum(), (self.y == 1).sum())
        band = 1.96 * np.sqrt(1.0 / (12 * n_min))
        self.ranking["Class_AUC"] = auc[self.ranking["Index"].values]
        self.ranking["Class_Encoding"] = np.abs(self.ranking["Class_AUC"] - 0.5) > band
        print(f"\nClass-selectivity band : |AUC - 0.5| > {band:.4f}")
        print(f"  class-encoding variables : {int(self.ranking['Class_Encoding'].sum())}"
              f" / {len(self.ranking)}")
        return band

    # ==========================================================
    def mi_threshold(self):
        if self.mi_method == "median":
            t = np.median(self.ranking["Mutual_Information"])
        elif self.mi_method == "p25":
            t = np.percentile(self.ranking["Mutual_Information"], 25)
        elif self.mi_method == "mean":
            t = np.mean(self.ranking["Mutual_Information"])
        else:
            t = float(self.mi_method)
        print("Mutual Information Threshold :", round(float(t), 6))
        return t

    # ==========================================================
    def select(self):
        """DISCOVERY selection, using the canonical criteria defined at the top
        of this module -- the identical function Modules 7, 8 and 9 apply inside
        training subjects only. Same code, different data."""
        C = np.load("Data/reference_confidence.npy")
        idx, st, mi_cut = select_confidence_variables(
            self.reservoir, C, spearman_threshold=self.spearman_threshold,
            q_threshold=self.q_threshold, mi_percentile=25, topk=26,
            return_stats=True)
        print("Mutual Information cut (25th pct) :", round(float(mi_cut), 6))

        conf_encoding = ((np.abs(st["spearman"]) >= self.spearman_threshold)
                         & (st["spearman_q"] < self.q_threshold)
                         & (st["kendall_q"] < self.q_threshold)
                         & (st["mi"] >= mi_cut))
        order = self.ranking["Index"].values
        self.ranking["Conf_Encoding"] = conf_encoding[order]
        self.ranking["TRUE_CONF"] = self.ranking["Conf_Encoding"] & ~self.ranking["Class_Encoding"]

        self.selected_indices = idx.tolist()
        self.selected = self.ranking.set_index("Index").loc[self.selected_indices].reset_index()

        # ---- SIGN OF THE ASSOCIATION -----------------------------------
        # The selection thresholds |rho|, so it keeps both directions. Split
        # them, because they mean opposite things:
        #     rho > 0 : variable grows with CONFIDENCE
        #     rho < 0 : variable grows with UNCERTAINTY
        rho_all = st["spearman"]
        conf_idx = [i for i in np.where(conf_encoding)[0] if rho_all[i] > 0]
        unc_idx = [i for i in np.where(conf_encoding)[0] if rho_all[i] < 0]
        self.conf_idx, self.unc_idx = conf_idx, unc_idx
        names_all = self.ranking.set_index("Index")["Feature"]

        print("\n" + "-" * 66)
        print("  CONFIDENCE-ENCODING versus UNCERTAINTY-ENCODING VARIABLES")
        print("-" * 66)
        print(f"  confidence-encoding  (rho > 0) : {len(conf_idx):3d} variables")
        for i in sorted(conf_idx, key=lambda i: -rho_all[i]):
            print(f"      {names_all[i]:24s} rho = {rho_all[i]:+.4f}")
        print(f"\n  uncertainty-encoding (rho < 0) : {len(unc_idx):3d} variables")
        for i in sorted(unc_idx, key=lambda i: rho_all[i]):
            print(f"      {names_all[i]:24s} rho = {rho_all[i]:+.4f}")
        if not unc_idx:
            print("      (none: every selected variable grows WITH confidence)")
        print("-" * 66)

        pd.DataFrame({
            "Feature": [names_all[i] for i in conf_idx + unc_idx],
            "Index": conf_idx + unc_idx,
            "Spearman": [rho_all[i] for i in conf_idx + unc_idx],
            "Encoding": (["confidence"] * len(conf_idx) + ["uncertainty"] * len(unc_idx)),
        }).sort_values("Spearman", ascending=False).to_csv(
            "Results/confidence_vs_uncertainty_variables.csv", index=False)
        print("Saved Results/confidence_vs_uncertainty_variables.csv")

        n_true = int(self.ranking["TRUE_CONF"].sum())
        print(f"\nConfidence-encoding (Spearman + Kendall + MI) : {int(conf_encoding.sum())}")
        print(f"Also class-encoding (NOT agnostic)            : "
              f"{int((self.ranking['Conf_Encoding'] & self.ranking['Class_Encoding']).sum())}")
        print(f"TRUE_CONF (confidence but NOT class)          : {n_true}")
        if n_true == 0:
            print("\n  FINDING: no SINGLE reservoir variable is both confidence-encoding and")
            print("  class-agnostic. This is expected, not a bug: the individual variables are")
            print("  C3/C4 band-power terms, which are inherently lateralised. Confidence is")
            print("  therefore carried by a QUADRATIC COMBINATION of them, which Module 7 builds.")
        print(f"\nNumber Selected for the estimator            : {len(self.selected_indices)}")
        print("\nTop selected features:")
        print(self.selected[["Feature", "Spearman", "Kendall", "Class_AUC", "TRUE_CONF"]]
              .head(10).to_string(index=False))

    # ==========================================================
    def extract_and_save(self, folder="Data"):
        self.selected_reservoir = self.reservoir[:, self.selected_indices]
        true_conf_idx = self.ranking[self.ranking["TRUE_CONF"]]["Index"].astype(int).tolist()
        np.save("Results/selected_reservoir.npy", self.selected_reservoir)
        np.save("Results/selected_indices.npy", np.array(self.selected_indices))
        np.save("Results/true_conf_indices.npy", np.array(true_conf_idx))
        self.selected.to_csv("Results/selected_features.csv", index=False)
        self.ranking.to_csv("Results/feature_ranking.csv", index=False)
        print("\nSelected Reservoir Shape :", self.selected_reservoir.shape)
        print("Saved selected_reservoir.npy, selected_indices.npy, true_conf_indices.npy")

    # ==========================================================
    def plot_confidence_vs_class(self, band, filename="Results/confidence_vs_class_coding.png"):
        """Signed rho on the y axis, so confidence-encoding and
        uncertainty-encoding variables separate above and below zero."""
        r = self.ranking
        x = np.abs(r["Class_AUC"] - 0.5)
        y = r["Spearman"]                      # SIGN RETAINED
        thr = self.spearman_threshold

        plt.figure(figsize=(10, 6.8))
        ax = plt.gca()
        x_max = x.max() * 1.05
        y_lim = np.abs(y).max() * 1.15

        ax.fill_between([0, band], thr, y_lim, color="limegreen", alpha=0.18)
        ax.fill_between([0, band], -y_lim, -thr, color="indianred", alpha=0.18)
        ax.fill_between([band, x_max], thr, y_lim, color="gray", alpha=0.15)
        ax.fill_between([band, x_max], -y_lim, -thr, color="gray", alpha=0.15)

        colour = ["#1b7f5a" if v > 0 else "#b3202c" for v in y]
        plt.scatter(x, y, c=colour, s=45, linewidth=0.4)
        plt.axhline(0, color="black", linewidth=1.2)
        plt.axvline(band, color="indianred", linestyle="--", label="class-encoding threshold")
        plt.axhline(thr, color="steelblue", linestyle=":", linewidth=2)
        plt.axhline(-thr, color="steelblue", linestyle=":", linewidth=2,
                    label="confidence-encoding threshold")
        plt.xlim(0, x_max); plt.ylim(-y_lim, y_lim)
        plt.text(band * 0.05, y_lim * 0.82, "confidence-encoding\nand class-agnostic",
                 fontsize=11, color="#1b7f5a")
        plt.text(band * 0.05, -y_lim * 0.92, "uncertainty-encoding\nand class-agnostic",
                 fontsize=11, color="#b3202c")
        plt.xlabel("|AUC - 0.5|   (class selectivity : can it see LH vs RH?)")
        plt.ylabel(r"Spearman $\rho$ with confidence   (sign retained)")
        plt.title("Confidence Coding vs Class Coding per Reservoir Variable\n"
                  "green above zero = confidence, red below zero = uncertainty")
        plt.legend(fontsize=12, loc="upper right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")


if __name__ == "__main__":
    apply_style()
    print("=" * 68)
    print("DISCOVERY STAGE : selection uses ALL subjects.")
    print("Module 7 repeats it inside training subjects only (leakage-free).")
    print("=" * 68)
    fs = FeatureSelector(spearman_threshold=0.05, q_threshold=0.05, mi_method="p25")
    fs.load()
    band = fs.class_selectivity()
    fs.select()
    fs.extract_and_save()
    fs.plot_confidence_vs_class(band)
    print("\nModule 6 Completed Successfully.")
