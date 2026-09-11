"""

Module 5 : Module_5_confidence_analysis.py

"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression
import matplotlib

import matplotlib.pyplot as plt
from Module_1_lorenz63 import apply_style, save_figure


class ConfidenceAnalysis:

    def __init__(self):
        self.reservoir = None
        self.target = None
        self.prediction = None
        self.error = None
        self.confidence = None
        self.feature_names = None

    # ==========================================================
    def load_data(self, k=2, s=1):
        self.reservoir = np.load("Data/ngrc_reservoir.npy")
        dataset = np.load("Data/lorenz63_dataset.npy")
        self.prediction = np.load("Data/prediction.npy")

        start = k * s
        self.target = dataset[start + 1:, 1:]
        self.reservoir = self.reservoir[:len(self.target)]

        print("Reservoir Shape  :", self.reservoir.shape)
        print("Prediction Shape :", self.prediction.shape)
        print("Target Shape     :", self.target.shape)

    # ==========================================================
    def compute_error_and_confidence(self):
        """ E(t) = mean_dim |y - y_hat| ; C(t) = 1 / (1 + E(t)) """
        self.error = np.mean(np.abs(self.target - self.prediction), axis=1)
        self.confidence = 1.0 / (1.0 + self.error)

        print("\nReference Confidence computed")
        print("Mean :", np.mean(self.confidence))
        print("Min  :", np.min(self.confidence))
        print("Max  :", np.max(self.confidence))

        np.save("Data/reference_confidence.npy", self.confidence)
        print("Saved Data/reference_confidence.npy")

    # ==========================================================
    def set_feature_names(self):
        self.feature_names = [f"x{i}" for i in range(self.reservoir.shape[1])]

    # ==========================================================
    def compute_correlations(self):
        if self.feature_names is None:
            self.set_feature_names()

        n_features = self.reservoir.shape[1]
        pearson_r = np.zeros(n_features)
        pearson_p = np.zeros(n_features)
        spearman_r = np.zeros(n_features)
        spearman_p = np.zeros(n_features)

        for i in range(n_features):
            col = self.reservoir[:, i]
            if np.std(col) == 0:
                pearson_r[i], pearson_p[i] = 0.0, 1.0
                spearman_r[i], spearman_p[i] = 0.0, 1.0
                continue
            pr, pp = pearsonr(col, self.confidence)
            sr, sp = spearmanr(col, self.confidence)
            pearson_r[i], pearson_p[i] = pr, pp
            spearman_r[i], spearman_p[i] = sr, sp

        print("\nComputing Mutual Information...")
        mi = mutual_info_regression(self.reservoir, self.confidence, random_state=0)

        np.save("Results/pearson_results.npy", pearson_r)
        np.save("Results/spearman_results.npy", spearman_r)
        np.save("Results/mutual_information.npy", mi)

        self.ranking = pd.DataFrame({
            "Feature": self.feature_names,
            "Pearson": pearson_r,
            "Pearson_p": pearson_p,
            "Spearman": spearman_r,
            "Spearman_p": spearman_p,
            "Mutual_Information": mi,
        })
        self.ranking = self.ranking.reindex(
            self.ranking["Pearson"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)

        print("\nTop 10 features by |Pearson|:")
        print(self.ranking.head(10).to_string(index=False))

        self.ranking.to_csv("Results/feature_ranking.csv", index=False)
        print("Saved Results/feature_ranking.csv")

        return self.ranking

    # ==========================================================
    def compute_correlation_matrix(self):
        """ Pairwise correlation matrix among reservoir features themselves. """
        corr_matrix = np.corrcoef(self.reservoir.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)  # bias column has zero variance
        np.save("Results/correlation_matrix.npy", corr_matrix)
        print("Saved Results/correlation_matrix.npy")
        return corr_matrix

    # ==========================================================
    def plot_correlation_heatmap(self, corr_matrix, filename="Results/correlation_heatmap.png"):
        plt.figure(figsize=(10, 8))
        im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, label="Pearson correlation")
        plt.title("Reservoir Feature Correlation Matrix")
        plt.xlabel("Feature index"); plt.ylabel("Feature index")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")

    # ==========================================================
    def plot_feature_ranking(self, filename="Results/feature_ranking_barplot.png", top_n=15):
        top = self.ranking.head(top_n)
        plt.figure(figsize=(10, 6))
        plt.bar(top["Feature"], top["Pearson"].abs())
        plt.xlabel("Reservoir Feature")
        plt.ylabel("|Pearson correlation| with Confidence")
        plt.title(f"Top {top_n} Reservoir Features by Confidence Correlation")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")

    # ==========================================================
    def plot_confidence_distribution(self, filename="Results/confidence_distribution.png"):
        plt.figure(figsize=(7, 5))
        plt.hist(self.confidence, bins=50, color="steelblue", edgecolor="black")
        plt.xlabel("Reference Confidence C(t)")
        plt.ylabel("Frequency")
        plt.title("Distribution of Reference Confidence")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")


if __name__ == "__main__":
    apply_style()
    ca = ConfidenceAnalysis()
    ca.load_data()
    ca.compute_error_and_confidence()
    ca.set_feature_names()
    ranking = ca.compute_correlations()
    corr_matrix = ca.compute_correlation_matrix()
    ca.plot_correlation_heatmap(corr_matrix)
    ca.plot_feature_ranking()
    ca.plot_confidence_distribution()

    print("\nModule 5 Completed Successfully.")
