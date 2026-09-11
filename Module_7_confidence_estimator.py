"""
Module 7 : Module_7_confidence_estimator.py

Objective:
    Estimate confidence directly from the selected reservoir features.

Leakage-free workflow:
    - Load selected reservoir
    - Load reference confidence
    - Split into training and test sets
    - Train confidence estimator using training data only
    - Predict confidence on unseen test data
    - Evaluate: RMSE, MAE, R2, Pearson correlation
    - Generate plots
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Module_1_lorenz63 import apply_style, save_figure


class ConfidenceEstimator:

    def __init__(self, alpha=1e-3):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)

    def load_data(self):

        S = np.load("Results/selected_reservoir.npy")
        C = np.load("Data/reference_confidence.npy")

        n = min(len(S), len(C))

        S = S[:n]
        C = C[:n]

        print("\nConfidence Estimator Data")
        print("------------------------------------")
        print("Selected reservoir shape :", S.shape)
        print("Reference confidence shape:", C.shape)
        print("Total samples            :", n)

        return S, C

    def split(self, S, C, train_fraction=0.80):

        n_train = int(len(S) * train_fraction)

        S_train = S[:n_train]
        S_test = S[n_train:]

        C_train = C[:n_train]
        C_test = C[n_train:]

        print("\nTrain / Test Split")
        print("------------------------------------")
        print("Training samples :", len(S_train))
        print("Test samples     :", len(S_test))

        return S_train, S_test, C_train, C_test

    def train(self, S_train, C_train):

        # Selected reservoir features → reference confidence
        self.model.fit(S_train, C_train)

        print("\nConfidence Estimator Trained")
        print("Model: Ridge")
        print("Alpha:", self.alpha)

    def predict(self, S):

        return self.model.predict(S)

    def evaluate(self, C_true, C_hat):

        rmse = np.sqrt(mean_squared_error(C_true, C_hat))
        mae = mean_absolute_error(C_true, C_hat)
        r2 = r2_score(C_true, C_hat)
        r, p = pearsonr(C_true, C_hat)

        metrics = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Pearson_r": r,
            "Pearson_p": p
        }

        print("\nConfidence Estimator Evaluation")
        print("------------------------------------")

        for key, value in metrics.items():
            print(f"{key:12s}: {value:.6f}")

        return metrics

    def save_model(self, filename="Results/confidence_model.pkl"):

        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

        print("\nModel saved to:", filename)

    def save_metrics(self, metrics,
                     filename="Results/confidence_metrics.csv"):

        pd.DataFrame([metrics]).to_csv(filename, index=False)

        print("Metrics saved to:", filename)

    def save_predictions(self, C_true, C_hat,
                         filename="Results/confidence_prediction.csv"):

        prediction_table = pd.DataFrame({
            "test_time_index": np.arange(len(C_true)),
            "reference_confidence": C_true,
            "estimated_confidence": C_hat
        })

        prediction_table.to_csv(filename, index=False)

        print("Predictions saved to:", filename)

    def plot_comparison(self, C_true, C_hat, n=1000):

        n_plot = min(n, len(C_true))

        # Reference vs Estimated Confidence
        plt.figure(figsize=(11, 5))

        plt.plot(
            C_true[:n_plot],
            label="Reference Confidence",
            linewidth=2.2
        )

        plt.plot(
            C_hat[:n_plot],
            "--",
            label="Estimated Confidence",
            linewidth=2.2
        )

        plt.xlabel("Test time step")
        plt.ylabel("Confidence")
        plt.title("Reference vs Estimated Confidence")
        plt.legend()
        plt.tight_layout()

        save_figure("Results/confidence_timeseries.png")

        # Scatter plot
        plt.figure(figsize=(7.5, 6.5))

        plt.scatter(
            C_true,
            C_hat,
            s=6,
            alpha=0.35,
            edgecolor="none"
        )

        lims = [
            min(C_true.min(), C_hat.min()),
            max(C_true.max(), C_hat.max())
        ]

        plt.plot(
            lims,
            lims,
            "r--",
            linewidth=2,
            label="Perfect fit"
        )

        plt.xlabel("Reference Confidence")
        plt.ylabel("Estimated Confidence")
        plt.title("Estimated vs Reference Confidence")
        plt.legend()
        plt.tight_layout()

        save_figure("Results/confidence_scatter.png")


if __name__ == "__main__":

    apply_style()

    est = ConfidenceEstimator(alpha=1e-3)

    S, C = est.load_data()

    S_train, S_test, C_train, C_test = est.split(
        S, C, train_fraction=0.80
    )

    est.train(S_train, C_train)

    C_hat = est.predict(S_test)

    metrics = est.evaluate(C_test, C_hat)

    est.save_model()
    est.save_metrics(metrics)
    est.save_predictions(C_test, C_hat)
    est.plot_comparison(C_test, C_hat)

    print("\nModule 7 Completed Successfully.")

    # Diagnostic
    print("\nDIAGNOSTIC")
    print("-----------------------------")

    print("C_test first 10:")
    print(C_test[:10])

    print("\nC_hat first 10:")
    print(C_hat[:10])

    print("\nAre they exactly identical?")
    print(np.array_equal(C_test, C_hat))

    print("\nMaximum absolute difference:")
    print(np.max(np.abs(C_test - C_hat)))

    print("\nMean absolute difference:")
    print(np.mean(np.abs(C_test - C_hat)))

    print("\nCorrelation:")
    print(np.corrcoef(C_test, C_hat)[0, 1])