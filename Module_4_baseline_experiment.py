"""

Module 4 : Module_4_baseline_experiment.py

"""

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from Module_1_lorenz63 import apply_style, save_figure


def create_target(data, k=2, s=1):
    start = k * s
    return data[start + 1:, 1:]


def align_reservoir(reservoir, target):
    return reservoir[:len(target)]


def plot_prediction_vs_true(target, prediction, n=1000):
    """One separate figure per state variable."""
    for i, lab in enumerate(["x", "y", "z"]):
        plt.figure(figsize=(11, 4.6))
        plt.plot(target[:n, i], label="True", linewidth=2.2)
        plt.plot(prediction[:n, i], "--", label="Predicted", linewidth=2.2)
        plt.xlabel("Time step"); plt.ylabel(lab)
        plt.title(f"NGRC Baseline: Prediction vs True Trajectory  ({lab})")
        plt.legend()
        save_figure(f"Results/prediction_vs_true_{lab}.png")


if __name__ == "__main__":
    apply_style()
    reservoir = np.load("Data/ngrc_reservoir.npy")
    dataset = np.load("Data/lorenz63_dataset.npy")
    Wout = np.load("Data/Wout.npy")

    target = create_target(dataset, k=2, s=1)
    reservoir = align_reservoir(reservoir, target)

    prediction = (Wout @ reservoir.T).T
    error = np.abs(target - prediction)

    mse = np.mean((prediction - target) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(prediction - target))

    print("Baseline NGRC Performance")
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("MAE  :", mae)

    np.save("Data/prediction.npy", prediction)
    np.save("Data/prediction_error.npy", error)

    pd.DataFrame([{"MSE": mse, "RMSE": rmse, "MAE": mae}]).to_csv(
        "Results/baseline_metrics.csv", index=False
    )
    print("Saved Results/baseline_metrics.csv")

    plot_prediction_vs_true(target, prediction)

    print("\nModule 4 Completed Successfully.")
