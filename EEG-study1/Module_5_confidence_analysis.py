"""

Module 5 : Module_5_confidence_analysis.py

*** THIS MODULE IS THE DISCOVERY STAGE ***
It uses ALL 9 subjects to ask "which NGRC variables encode confidence in this
dataset?". That is a legitimate descriptive question and the answer is the
scientific content of the study. But because every subject contributes to the
correlations, any performance number derived from this selection is OPTIMISTIC.
Module 7 repeats the selection inside the training subjects only and reports the
leakage-free number. Quote Module 7, cite Module 5 for interpretation.

The confidence step, mirroring Module 5 of the Lorenz-63 pipeline.

Lorenz-63 (regression)    : E(t) = |y - y_hat| ,  C(t) = 1 / (1 + E(t))
Here      (classification): C(t) = |o_LH - o_RH| = |Delta|         <- LABEL-FREE

Why label-free is the primary target:
    The Lorenz-style target needs the true label to build. If confidence is fitted
    to a label-derived target, then testing whether the result is LH/RH-agnostic
    is circular. Using |o_LH - o_RH| keeps the label out of the target, the
    feature selection AND the estimator, so the agnosticism test in Module 8 is
    an INDEPENDENT test.

    It also states the requirement literally: the closer the two readout
    activities, the lower the confidence.

Cost, stated plainly: |Delta| measures DECISIVENESS, not probability-correct.
A confidently wrong trial scores high. That is why correctness-based validation
(labels used only at scoring time) is kept as the external check in Modules 7-8.

The label-dependent definition is still computed here and carried forward as a
SENSITIVITY comparison.

"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, false_discovery_control
from sklearn.feature_selection import mutual_info_regression
import matplotlib

import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure


class ConfidenceAnalysis:

    def __init__(self):
        self.reservoir = None
        self.delta = None
        self.confidence = None
        self.confidence_labeldep = None
        self.feature_names = None
        self.ranking = None

    # ==========================================================
    def load_data(self, folder="Data"):
        self.reservoir = np.load(f"{folder}/ngrc_reservoir.npy")
        self.delta = np.load(f"{folder}/readout_delta.npy")
        self.y = np.load(f"{folder}/eeg_labels.npy")
        self.correct = np.load(f"{folder}/correct.npy")
        self.feature_names = list(np.load(f"{folder}/feature_names.npy", allow_pickle=True))
        print("Reservoir Shape  :", self.reservoir.shape)
        print("Delta Shape      :", self.delta.shape)

    # ==========================================================
    def compute_confidence(self, folder="Data"):
        """PRIMARY  : C = |o_LH - o_RH|            (label-free)
           SECONDARY: C = 1/(1+|y_signed - Delta|) (Lorenz-style, for sensitivity)"""
        self.confidence = np.abs(self.delta)

        y_signed = np.where(self.y == 0, 1.0, -1.0)
        error = np.abs(y_signed - self.delta)
        self.confidence_labeldep = 1.0 / (1.0 + error)

        print("\nReference Confidence computed (label-free, PRIMARY)")
        print("  Mean :", round(float(np.mean(self.confidence)), 4))
        print("  Min  :", round(float(np.min(self.confidence)), 4))
        print("  Max  :", round(float(np.max(self.confidence)), 4))

        auc = self._auc(self.confidence, self.correct)
        print(f"  Sanity check: confidence separates correct from error trials, AUROC2 = {auc:.4f}")

        np.save(f"{folder}/reference_confidence.npy", self.confidence)
        np.save(f"{folder}/reference_confidence_labeldep.npy", self.confidence_labeldep)
        np.save(f"{folder}/prediction_error.npy", error)
        print(f"Saved {folder}/reference_confidence.npy")

    # ==========================================================
    @staticmethod
    def _auc(score, label):
        from scipy.stats import rankdata
        r = rankdata(score)
        n1 = int(np.sum(label == 1)); n0 = len(label) - n1
        return (r[label == 1].sum() - n1*(n1+1)/2) / (n1*n0)

    # ==========================================================
    def compute_correlations(self):
        """Correlate every reservoir variable with the confidence.
        Spearman IDENTIFIES candidates, Kendall CONFIRMS them, mutual information
        adds a nonlinear criterion. Pearson is reported but not used for
        selection: the confidence distribution is not normal."""
        n_features = self.reservoir.shape[1]
        pear = np.zeros(n_features); pear_p = np.ones(n_features)
        spear = np.zeros(n_features); spear_p = np.ones(n_features)
        kend = np.zeros(n_features); kend_p = np.ones(n_features)

        for i in range(n_features):
            col = self.reservoir[:, i]
            if np.std(col) == 0:
                continue
            pear[i], pear_p[i] = pearsonr(col, self.confidence)
            spear[i], spear_p[i] = spearmanr(col, self.confidence)
            kend[i], kend_p[i] = kendalltau(col, self.confidence)

        print("\nComputing Mutual Information...")
        mi = mutual_info_regression(self.reservoir, self.confidence, random_state=0)

        spear_q = false_discovery_control(np.nan_to_num(spear_p, nan=1.0))
        kend_q = false_discovery_control(np.nan_to_num(kend_p, nan=1.0))

        agreement = spearmanr(np.abs(spear), np.abs(kend))[0]
        print(f"Spearman / Kendall ranking agreement : {agreement:.4f}")

        np.save("Results/pearson_results.npy", pear)
        np.save("Results/spearman_results.npy", spear)
        np.save("Results/kendall_results.npy", kend)
        np.save("Results/mutual_information.npy", mi)

        self.ranking = pd.DataFrame({
            "Feature": self.feature_names,
            "Index": np.arange(n_features),
            "Pearson": pear, "Pearson_p": pear_p,
            "Spearman": spear, "Spearman_p": spear_p, "Spearman_q": spear_q,
            "Kendall": kend, "Kendall_p": kend_p, "Kendall_q": kend_q,
            "Mutual_Information": mi,
        })
        self.ranking = self.ranking.reindex(
            self.ranking["Spearman"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        print("\nTop 10 features by |Spearman| with confidence:")
        print(self.ranking[["Feature", "Spearman", "Kendall", "Mutual_Information"]]
              .head(10).to_string(index=False))

        self.ranking.to_csv("Results/feature_ranking.csv", index=False)
        print("Saved Results/feature_ranking.csv")
        return self.ranking

    # ==========================================================
    def compute_correlation_matrix(self):
        cm = np.corrcoef(self.reservoir.T)
        cm = np.nan_to_num(cm, nan=0.0)
        np.save("Results/correlation_matrix.npy", cm)
        print("Saved Results/correlation_matrix.npy")
        return cm

    # ==========================================================
    def plot_feature_ranking(self, filename="Results/feature_ranking_barplot.png", top_n=15):
        """SIGNED correlations, not absolute values.

        The absolute value hides the direction of the association. Keeping the
        sign separates the two kinds of variable:
            rho > 0  the variable grows as the decision becomes more CONFIDENT
            rho < 0  the variable grows as the decision becomes more UNCERTAIN
        Both are selected by the pipeline (which thresholds |rho|), but they
        mean opposite things and should be read separately.
        """
        top = self.ranking.head(top_n)
        x = np.arange(len(top)); w = 0.38
        sp, kd = top["Spearman"].values, top["Kendall"].values

        plt.figure(figsize=(13, 6.4))
        plt.bar(x - w/2, sp, w, label=r"Spearman $\rho$",
                color=["#1b7f5a" if v > 0 else "#b3202c" for v in sp])
        plt.bar(x + w/2, kd, w, label=r"Kendall $\tau$",
                color=["#7fc4a8" if v > 0 else "#dd8d97" for v in kd])
        plt.axhline(0, color="black", linewidth=1.2)
        plt.axhline(0.05, color="#7a7a7a", linestyle=":", linewidth=2)
        plt.axhline(-0.05, color="#7a7a7a", linestyle=":", linewidth=2,
                    label="selection threshold")
        plt.xticks(x, top["Feature"], rotation=45, ha="right", fontsize=11)
        plt.xlabel("Reservoir Feature")
        plt.ylabel(r"correlation with confidence (sign retained)")
        plt.title(f"Top {top_n} Reservoir Features by Confidence Correlation\n"
                  "green = confidence-encoding, red = uncertainty-encoding")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")

    # ==========================================================
    def plot_signed_distribution(self, filename="Results/signed_correlation_distribution.png"):
        """Where the whole reservoir sits on the signed axis."""
        rho = self.ranking["Spearman"].values
        plt.figure(figsize=(9.5, 5.6))
        plt.hist(rho[rho >= 0], bins=15, color="#1b7f5a", label="confidence-encoding")
        plt.hist(rho[rho < 0], bins=15, color="#b3202c", label="uncertainty-encoding")
        plt.axvline(0, color="black", linewidth=1.2)
        plt.axvline(0.05, color="#7a7a7a", linestyle=":", linewidth=2)
        plt.axvline(-0.05, color="#7a7a7a", linestyle=":", linewidth=2,
                    label="selection threshold")
        plt.xlabel(r"Spearman $\rho$ with confidence")
        plt.ylabel("Reservoir variables")
        plt.title("Signed correlations across the whole reservoir")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")


if __name__ == "__main__":
    apply_style()
    print("=" * 68)
    print("DISCOVERY STAGE : correlations computed on ALL subjects.")
    print("Descriptive only -- see Module 7 for the leakage-free validation.")
    print("=" * 68)
    ca = ConfidenceAnalysis()
    ca.load_data()
    ca.compute_confidence()
    ranking = ca.compute_correlations()
    cm = ca.compute_correlation_matrix()
    ca.plot_feature_ranking()
    ca.plot_signed_distribution()

    print("\nModule 5 Completed Successfully.")
