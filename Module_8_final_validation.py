"""

Module 8 : Module_8_final_validation.py


"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib

import matplotlib.pyplot as plt
from Module_1_lorenz63 import apply_style, save_figure


# ----------------------------------------------------------------
def evaluate(C_true, C_hat):
    rmse = np.sqrt(mean_squared_error(C_true, C_hat))
    mae = mean_absolute_error(C_true, C_hat)
    r2 = r2_score(C_true, C_hat)
    r, p = pearsonr(C_true, C_hat)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "Pearson_r": r, "Pearson_p": p}


def split(X, y, train_fraction=0.8):
    n_train = int(len(X) * train_fraction)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


# ==================================================================
# 1. Different reference confidence definitions
# ==================================================================
def compare_confidence_definitions(selected_reservoir, error):
    print("\n=== Confidence Definitions Comparison ===")

    definitions = {
        "inverse":     1.0 / (1.0 + error),
        "exponential": np.exp(-error),
        "gaussian":    np.exp(-(error ** 2) / (2 * np.var(error))),
    }

    results = []
    for name, C in definitions.items():
        S_train, S_test, C_train, C_test = split(selected_reservoir, C)
        model = Ridge(alpha=1e-3).fit(S_train, C_train)
        C_hat = model.predict(S_test)
        metrics = evaluate(C_test, C_hat)
        metrics["Definition"] = name
        results.append(metrics)
        print(f"  {name:12s} R2={metrics['R2']:.4f}  r={metrics['Pearson_r']:.4f}")

    df = pd.DataFrame(results)[["Definition", "RMSE", "MAE", "R2", "Pearson_r", "Pearson_p"]]
    df.to_csv("Results/confidence_definitions_comparison.csv", index=False)

    print("Saved confidence_definitions_comparison.csv")
    return df


# ==================================================================
# 2. Compare estimators: Ridge vs Linear Regression + significance test
# ==================================================================
def compare_estimators(S, C):
    print("\n=== Estimator Comparison: Ridge vs Linear Regression ===")

    S_train, S_test, C_train, C_test = split(S, C)

    ridge = Ridge(alpha=1e-3).fit(S_train, C_train)
    linreg = LinearRegression().fit(S_train, C_train)

    C_hat_ridge = ridge.predict(S_test)
    C_hat_lin = linreg.predict(S_test)

    m_ridge = evaluate(C_test, C_hat_ridge)
    m_lin = evaluate(C_test, C_hat_lin)

    m_ridge["Model"] = "Ridge"
    m_lin["Model"] = "LinearRegression"

    df = pd.DataFrame([m_ridge, m_lin])[["Model", "RMSE", "MAE", "R2", "Pearson_r", "Pearson_p"]]
    print(df.to_string(index=False))

    # Statistical significance: paired test on absolute errors of the two models
    err_ridge = np.abs(C_test - C_hat_ridge)
    err_lin = np.abs(C_test - C_hat_lin)

    t_stat, t_p = ttest_rel(err_ridge, err_lin)
    try:
        w_stat, w_p = wilcoxon(err_ridge, err_lin)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    sig_df = pd.DataFrame([{
        "Comparison": "Ridge vs LinearRegression (|error|)",
        "Paired_t_stat": t_stat, "Paired_t_p": t_p,
        "Wilcoxon_stat": w_stat, "Wilcoxon_p": w_p,
    }])
    sig_df.to_csv("Results/statistical_significance.csv", index=False)

    print(f"\nPaired t-test  : t={t_stat:.4f}, p={t_p:.4e}")
    print(f"Wilcoxon test  : W={w_stat:.4f}, p={w_p:.4e}")
    print("Saved statistical_significance.csv")

    return df


# ==================================================================
# 3. Ablation: with vs without feature selection
# ==================================================================
def ablation_study(full_reservoir, selected_reservoir, C):
    print("\n=== Ablation Study: With vs Without Feature Selection ===")

    results = []
    for name, X in [("Without_Selection (full reservoir)", full_reservoir),
                     ("With_Selection (reduced reservoir)", selected_reservoir)]:
        X_train, X_test, C_train, C_test = split(X, C)
        model = Ridge(alpha=1e-3).fit(X_train, C_train)
        C_hat = model.predict(X_test)
        metrics = evaluate(C_test, C_hat)
        metrics["Setting"] = name
        metrics["n_features"] = X.shape[1]
        results.append(metrics)
        print(f"  {name:38s} n_features={X.shape[1]:3d}  R2={metrics['R2']:.4f}")

    df = pd.DataFrame(results)[["Setting", "n_features", "RMSE", "MAE", "R2", "Pearson_r"]]
    df.to_csv("Results/ablation_results.csv", index=False)

    print("Saved ablation_results.csv")
    return df


# ==================================================================
# 4. Sensitivity analysis: effect of the Pearson threshold delta
# ==================================================================
def sensitivity_analysis(ranking, full_reservoir, C, thresholds=None):
    print("\n=== Sensitivity Analysis: Pearson Threshold ===")

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = []
    for delta in thresholds:
        idx = ranking[ranking["Pearson"].abs() > delta]["Feature"].apply(
            lambda f: int(f.replace("x", ""))
        ).tolist()

        if len(idx) == 0:
            continue

        X = full_reservoir[:, idx]
        X_train, X_test, C_train, C_test = split(X, C)
        model = Ridge(alpha=1e-3).fit(X_train, C_train)
        C_hat = model.predict(X_test)
        metrics = evaluate(C_test, C_hat)

        results.append({"delta": delta, "n_features": len(idx), "R2": metrics["R2"],
                         "RMSE": metrics["RMSE"], "Pearson_r": metrics["Pearson_r"]})
        print(f"  delta={delta:.2f}  n_features={len(idx):3d}  R2={metrics['R2']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("Results/sensitivity_results.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df["delta"], df["R2"], "o-", color="steelblue", label="R2")
    ax1.set_xlabel("Pearson threshold (delta)")
    ax1.set_ylabel("R2", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(df["delta"], df["n_features"], "s--", color="darkorange", label="n_features")
    ax2.set_ylabel("Number of selected features", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    plt.title("Sensitivity of Confidence Estimator to Feature Selection Threshold")
    save_figure("Results/sensitivity_plot.png")

    print("Saved sensitivity_results.csv + plot")
    return df


# ==================================================================
if __name__ == "__main__":
    apply_style()
    full_reservoir = np.load("Data/ngrc_reservoir.npy")
    selected_reservoir = np.load("Results/selected_reservoir.npy")
    C = np.load("Data/reference_confidence.npy")
    error = np.load("Data/prediction_error.npy")
    error_scalar = np.mean(error, axis=1)
    ranking = pd.read_csv("Results/feature_ranking.csv")

    n = min(len(full_reservoir), len(selected_reservoir), len(C), len(error_scalar))
    full_reservoir = full_reservoir[:n]
    selected_reservoir = selected_reservoir[:n]
    C = C[:n]
    error_scalar = error_scalar[:n]

    def_df = compare_confidence_definitions(selected_reservoir, error_scalar)
    est_df = compare_estimators(selected_reservoir, C)
    abl_df = ablation_study(full_reservoir, selected_reservoir, C)
    sens_df = sensitivity_analysis(ranking, full_reservoir, C)

    # ---- final combined comparison table ----
    final_table = pd.concat([
        def_df.assign(Analysis="Confidence Definition"),
        est_df.assign(Analysis="Estimator Comparison"),
        abl_df.rename(columns={"Setting": "Definition"}).assign(Analysis="Ablation"),
    ], ignore_index=True, sort=False)

    final_table.to_csv("Results/final_comparison_table.csv", index=False)
    print("\nSaved Results/final_comparison_table.csv")

    print("\nModule 8 Completed Successfully.")
