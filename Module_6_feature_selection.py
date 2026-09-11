"""
Module 6 : Module_6_feature_selection.py

Objective:
    Perform leakage-free confidence-related feature selection.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression

from Module_1_lorenz63 import apply_style, save_figure


class FeatureSelector:

    def __init__(self, pearson_threshold=0.30, pvalue_threshold=0.05,
                 mi_method="median", train_fraction=0.80):

        self.pearson_threshold = pearson_threshold
        self.pvalue_threshold = pvalue_threshold
        self.mi_method = mi_method
        self.train_fraction = train_fraction

        self.reservoir = None
        self.feature_names = None
        self.confidence = None

        self.train_reservoir = None
        self.test_reservoir = None
        self.train_confidence = None
        self.test_confidence = None

        self.ranking = None
        self.selected = None
        self.selected_indices = None
        self.selected_reservoir = None

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    def load_reservoir(self, filename="Data/ngrc_reservoir.npy"):

        self.reservoir = np.load(filename)

        self.feature_names = np.load(
            "Data/feature_names.npy",
            allow_pickle=True
        )

        print("\nReservoir Loaded. Shape:", self.reservoir.shape)
        print("Feature Names Loaded:", len(self.feature_names))

    def load_confidence(
            self,
            filename="Data/reference_confidence.npy"):

        self.confidence = np.load(filename)

        print(
            "Reference Confidence Loaded. Shape:",
            self.confidence.shape
        )

    # ---------------------------------------------------------
    # Train / test split
    # ---------------------------------------------------------

    def split_data(self):

        n = min(
            len(self.reservoir),
            len(self.confidence)
        )

        self.reservoir = self.reservoir[:n]
        self.confidence = self.confidence[:n]

        n_train = int(
            n * self.train_fraction
        )

        self.train_reservoir = self.reservoir[:n_train]
        self.test_reservoir = self.reservoir[n_train:]

        self.train_confidence = self.confidence[:n_train]
        self.test_confidence = self.confidence[n_train:]

        print("\nLeakage-Free Feature Selection Split")
        print("------------------------------------")
        print("Total samples    :", n)
        print(
            "Training samples :",
            len(self.train_reservoir)
        )
        print(
            "Test samples     :",
            len(self.test_reservoir)
        )

    # ---------------------------------------------------------
    # Feature ranking
    # ---------------------------------------------------------

    def compute_feature_ranking(self):

        X = self.train_reservoir
        C = self.train_confidence

        n_features = X.shape[1]

        pearson_values = []
        pearson_pvalues = []

        print(
            "\nCalculating Pearson correlations "
            "using TRAINING DATA ONLY..."
        )

        for i in range(n_features):

            r, p = pearsonr(
                X[:, i],
                C
            )

            pearson_values.append(r)
            pearson_pvalues.append(p)

        print(
            "Calculating Mutual Information "
            "using TRAINING DATA ONLY..."
        )

        mi_values = mutual_info_regression(
            X,
            C,
            random_state=0
        )

        self.ranking = pd.DataFrame({
            "Index": np.arange(n_features),
            "Feature": self.feature_names,
            "Pearson": pearson_values,
            "Pearson_p": pearson_pvalues,
            "Mutual_Information": mi_values
        })

        self.ranking = self.ranking.reindex(
            self.ranking["Pearson"]
            .abs()
            .sort_values(ascending=False)
            .index
        ).reset_index(drop=True)

        print("\nFeature Ranking Created")
        print("------------------------------------")
        print(self.ranking.head(10))

        self.ranking.to_csv(
            "Results/feature_ranking.csv",
            index=False
        )

    # ---------------------------------------------------------
    # Mutual information threshold
    # ---------------------------------------------------------

    def mi_threshold(self):

        if self.mi_method == "median":

            threshold = np.median(
                self.ranking["Mutual_Information"]
            )

        elif self.mi_method == "mean":

            threshold = np.mean(
                self.ranking["Mutual_Information"]
            )

        else:

            threshold = float(
                self.mi_method
            )

        print(
            "\nMutual Information Threshold:",
            threshold
        )

        return threshold

    # ---------------------------------------------------------
    # Feature selection
    # ---------------------------------------------------------

    def select(self):

        mi_threshold = self.mi_threshold()

        selected = self.ranking[
            (self.ranking["Pearson"].abs()
             >= self.pearson_threshold)
            &
            (self.ranking["Pearson_p"]
             < self.pvalue_threshold)
            &
            (self.ranking["Mutual_Information"]
             >= mi_threshold)
        ]

        if len(selected) == 0:

            print(
                "\nWarning: nothing passed thresholds."
            )

            print(
                "Falling back to top 10 "
                "features by |Pearson|."
            )

            selected = self.ranking.head(10)

        self.selected = selected.copy()

        self.selected_indices = (
            self.selected["Index"]
            .astype(int)
            .tolist()
        )

        print("\nSelected Features")
        print("------------------------------------")

        print(
            self.selected[
                [
                    "Index",
                    "Feature",
                    "Pearson",
                    "Pearson_p",
                    "Mutual_Information"
                ]
            ]
        )

        print(
            "\nNumber Selected:",
            len(self.selected_indices)
        )

    # ---------------------------------------------------------
    # Extract selected reservoir features
    # ---------------------------------------------------------

    def extract(self):

        self.selected_reservoir = (
            self.reservoir[
                :,
                self.selected_indices
            ]
        )

        print(
            "\nSelected Reservoir Shape:",
            self.selected_reservoir.shape
        )

        print(
            "Selected features determined "
            "using training data only."
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    def save(self):

        np.save(
            "Results/selected_reservoir.npy",
            self.selected_reservoir
        )

        np.save(
            "Results/selected_indices.npy",
            np.array(self.selected_indices)
        )

        self.selected.to_csv(
            "Results/selected_features.csv",
            index=False
        )

        print("\nSaved:")
        print(
            "  Results/selected_reservoir.npy"
        )
        print(
            "  Results/selected_indices.npy"
        )
        print(
            "  Results/selected_features.csv"
        )

    # ---------------------------------------------------------
    # Plot selected features
    # ---------------------------------------------------------

    def plot_selected_features(
            self,
            filename="Results/selected_features_plot.png"):

        # Keep the same selected-feature ordering
        features = self.selected.copy()

        x = np.arange(
            len(features)
        )

        # Positive = confidence
        # Negative = uncertainty
        colors = np.where(
            features["Pearson"] >= 0,
            "green",
            "red"
        )

        plt.figure(
            figsize=(12, 7)
        )

        plt.bar(
            x,
            features["Pearson"],
            color=colors
        )

        plt.axhline(
            0,
            color="black",
            linewidth=1
        )

        plt.axhline(
            self.pearson_threshold,
            color="gray",
            linestyle=":",
            linewidth=1.5
        )

        plt.axhline(
            -self.pearson_threshold,
            color="gray",
            linestyle=":",
            linewidth=1.5
        )

        plt.xticks(
            x,
            features["Feature"],
            rotation=45,
            ha="right"
        )

        plt.xlabel(
            "Selected Reservoir Feature"
        )

        plt.ylabel(
            "Pearson Correlation with Confidence"
        )

        plt.title(
            f"Selected Reservoir Features "
            f"(n={len(self.selected_indices)})\n"
            "green = confidence-encoding, "
            "red = uncertainty-encoding"
        )

        from matplotlib.patches import Patch

        legend = [
            Patch(
                facecolor="green",
                label="Confidence-encoding"
            ),
            Patch(
                facecolor="red",
                label="Uncertainty-encoding"
            )
        ]

        plt.legend(
            handles=legend
        )

        plt.tight_layout()

        save_figure(
            filename
        )

        print(
            f"Saved {filename}"
        )


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":

    apply_style()

    fs = FeatureSelector(
        pearson_threshold=0.30,
        pvalue_threshold=0.05,
        mi_method="median",
        train_fraction=0.80
    )

    fs.load_reservoir()
    fs.load_confidence()
    fs.split_data()
    fs.compute_feature_ranking()
    fs.select()
    fs.extract()
    fs.save()
    fs.plot_selected_features()

    print(
        "\nModule 6 Completed Successfully."
    )