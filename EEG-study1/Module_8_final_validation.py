"""

Module 8 : Module_8_final_validation.py

Final validation, mirroring Module 8 of the Lorenz-63 pipeline (definitions,
estimator comparison, ablation, sensitivity) and adding the two analyses the
EEG task specifically requires:

    5. LH/RH AGNOSTICISM  -- does the confidence encode the DECISION?
       This is the supervisor's central question. Because the confidence target
       |o_LH - o_RH| is label-free, no label entered the target, the selection
       or the estimator, so this is an INDEPENDENT test, not a circular one.

    6. NGRC RANDOMISATION -- NGRC has no random weights, so "different random
       initialisation" means the constructive choices it does have: the delay
       depth k and spacing s. The whole pipeline is repeated across several.

All numbers are computed on UNSEEN SUBJECTS.

"""

import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.stats import (pearsonr, spearmanr, kendalltau, ttest_rel, wilcoxon,
                          rankdata, levene)
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib

import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure
from scipy.stats import ks_2samp
from Module_6_feature_selection import select_confidence_variables


# ----------------------------------------------------------------- helpers
def fast_auc(score, label):
    r = rankdata(score)
    n1 = int(np.sum(label == 1)); n0 = len(label) - n1
    return (r[label == 1].sum() - n1*(n1+1)/2) / (n1*n0)


def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)


def d_ci(a, b, B=2000, seed=0):
    rng = np.random.default_rng(seed); out = np.empty(B)
    for i in range(B):
        out[i] = cohens_d(a[rng.integers(0, len(a), len(a))],
                          b[rng.integers(0, len(b), len(b))])
    return np.percentile(out, [2.5, 97.5])


def perm_p_auc(score, label, groups, B=5000, seed=0):
    """Permutation p for AUC != 0.5, labels shuffled WITHIN subject."""
    rng = np.random.default_rng(seed)
    r = rankdata(score)
    n1 = int(np.sum(label == 1)); n0 = len(label) - n1
    obs = abs((r[label == 1].sum() - n1*(n1+1)/2) / (n1*n0) - 0.5)
    gi = [np.where(groups == g)[0] for g in np.unique(groups)]
    cnt = 0; L = label.copy()
    for _ in range(B):
        for ii in gi:
            L[ii] = rng.permutation(label[ii])
        cnt += (abs((r[L == 1].sum() - n1*(n1+1)/2) / (n1*n0) - 0.5) >= obs)
    return (cnt + 1) / (B + 1)


def quad_expand(Z):
    idx = list(combinations_with_replacement(range(Z.shape[1]), 2))
    i_arr = np.array([i for i, j in idx]); j_arr = np.array([j for i, j in idx])
    return np.hstack([Z, Z[:, i_arr] * Z[:, j_arr]])


def subject_folds(subjects, n_folds=3, seed=0):
    ids = np.unique(subjects)
    return np.array_split(np.random.default_rng(seed).permutation(ids), n_folds)


def fit_unseen(X, target, subjects, alpha=100.0, quadratic=True):
    """Fit on training subjects, predict on unseen. Returns predictions for all trials."""
    out = np.full(len(target), np.nan)
    for unseen in subject_folds(subjects):
        tr = ~np.isin(subjects, unseen); te = np.isin(subjects, unseen)
        sc = StandardScaler().fit(X[tr]); Z = sc.transform(X)
        F = quad_expand(Z) if quadratic else Z
        sc2 = StandardScaler().fit(F[tr])
        m = Ridge(alpha=alpha).fit(sc2.transform(F[tr]), target[tr])
        out[te] = m.predict(sc2.transform(F[te]))
    return out


def evaluate(C_true, C_hat, correct):
    return {"RMSE": np.sqrt(mean_squared_error(C_true, C_hat)),
            "MAE": mean_absolute_error(C_true, C_hat),
            "R2": r2_score(C_true, C_hat),
            "Spearman_r": spearmanr(C_true, C_hat)[0],
            "Kendall_tau": kendalltau(C_true, C_hat)[0],
            "AUROC2_vs_correct": fast_auc(C_hat, correct)}


# ==================================================================
# 1. Different reference confidence definitions
# ==================================================================
def compare_confidence_definitions(S, delta, y, subjects, correct):
    print("\n=== 1. Confidence Definitions Comparison ===")
    y_signed = np.where(y == 0, 1.0, -1.0)
    error = np.abs(y_signed - delta)
    sep = np.abs(delta)

    definitions = {
        "|o_LH - o_RH|  (label-free, PRIMARY)": sep,
        "inverse  1/(1+E)  (label-dependent)": 1.0 / (1.0 + error),
        "exponential  exp(-E)  (label-dep.)": np.exp(-error),
        "gaussian  (label-dependent)": np.exp(-(error ** 2) / (2 * np.var(error))),
    }

    results = []
    for name, C in definitions.items():
        C_hat = fit_unseen(S, C, subjects)
        m = evaluate(C, C_hat, correct)
        m["Definition"] = name
        m["uses_label"] = "label-free" not in name
        results.append(m)
        print(f"  {name:38s} R2={m['R2']:.4f}  AUROC2={m['AUROC2_vs_correct']:.4f}")

    df = pd.DataFrame(results)[["Definition", "uses_label", "RMSE", "MAE", "R2",
                                 "Spearman_r", "Kendall_tau", "AUROC2_vs_correct"]]
    df.to_csv("Results/confidence_definitions_comparison.csv", index=False)

    print("Saved confidence_definitions_comparison.csv")
    return df


# ==================================================================
# 2. Estimator comparison: LINEAR vs QUADRATIC (the critical one)
# ==================================================================
def compare_estimators(S, C, subjects, correct):
    print("\n=== 2. Estimator Comparison: Linear vs Quadratic readout ===")
    C_lin = fit_unseen(S, C, subjects, quadratic=False)
    C_quad = fit_unseen(S, C, subjects, quadratic=True)

    m_lin = evaluate(C, C_lin, correct); m_lin["Model"] = "Linear (Lorenz-63 style)"
    m_quad = evaluate(C, C_quad, correct); m_quad["Model"] = "Quadratic (pairwise products)"

    df = pd.DataFrame([m_lin, m_quad])[["Model", "RMSE", "MAE", "R2", "Spearman_r",
                                         "Kendall_tau", "AUROC2_vs_correct"]]
    print(df.to_string(index=False))

    err_lin = np.abs(C - C_lin); err_quad = np.abs(C - C_quad)
    t_stat, t_p = ttest_rel(err_quad, err_lin)
    try:
        w_stat, w_p = wilcoxon(err_quad, err_lin)
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    pd.DataFrame([{"Comparison": "Quadratic vs Linear (|error|)",
                   "Paired_t_stat": t_stat, "Paired_t_p": t_p,
                   "Wilcoxon_stat": w_stat, "Wilcoxon_p": w_p}]).to_csv(
        "Results/statistical_significance.csv", index=False)
    print(f"\nPaired t-test : t={t_stat:.4f}, p={t_p:.4e}")
    print(f"Wilcoxon      : W={w_stat:.4f}, p={w_p:.4e}")
    print("Saved statistical_significance.csv")

    df.to_csv("Results/estimator_comparison.csv", index=False)
    return df, C_quad, C_lin


# ==================================================================
# 3. Ablation: with vs without feature selection
# ==================================================================
def ablation_study(full_reservoir, selected_reservoir, C, subjects, correct):
    print("\n=== 3. Ablation: With vs Without Feature Selection ===")
    results = []
    for name, X in [("Without_Selection (full reservoir)", full_reservoir),
                     ("With_Selection (reduced reservoir)", selected_reservoir)]:
        C_hat = fit_unseen(X, C, subjects)
        m = evaluate(C, C_hat, correct)
        m["Setting"] = name; m["n_features"] = X.shape[1]
        results.append(m)
        print(f"  {name:38s} n={X.shape[1]:3d}  AUROC2={m['AUROC2_vs_correct']:.4f}")

    df = pd.DataFrame(results)[["Setting", "n_features", "RMSE", "R2",
                                 "Spearman_r", "AUROC2_vs_correct"]]
    df.to_csv("Results/ablation_results.csv", index=False)

    print("Saved ablation_results.csv")
    return df


# ==================================================================
# 4. Sensitivity to the selection threshold
# ==================================================================
def sensitivity_analysis(ranking, full_reservoir, C, subjects, correct):
    print("\n=== 4. Sensitivity: Spearman Selection Threshold ===")
    results = []
    for delta_t in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        idx = ranking[ranking["Spearman"].abs() > delta_t]["Index"].astype(int).tolist()
        if len(idx) < 3:
            continue
        C_hat = fit_unseen(full_reservoir[:, idx], C, subjects)
        m = evaluate(C, C_hat, correct)
        results.append({"delta": delta_t, "n_features": len(idx), "R2": m["R2"],
                         "Spearman_r": m["Spearman_r"],
                         "AUROC2_vs_correct": m["AUROC2_vs_correct"]})
        print(f"  threshold={delta_t:.2f}  n={len(idx):3d}  AUROC2={m['AUROC2_vs_correct']:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("Results/sensitivity_results.csv", index=False)

    print("Saved sensitivity_results.csv")
    return df


# ==================================================================
# 5. LH/RH AGNOSTICISM  (the supervisor's central question)
# ==================================================================
def agnosticism_test(C_quad, C_lin, delta, full_reservoir, y, subjects):
    """Two DIFFERENT questions, reported separately:

       (a) MEAN shift        -- Cohen's d. Is confidence on average higher for
                                one hand than the other?
       (b) DISTRIBUTIONAL    -- AUC + permutation p, Levene, Kolmogorov-Smirnov.
                                Can LH and RH be told apart from the confidence
                                value AT ALL, by any feature of its distribution?

    These can disagree: a negligible mean shift can coexist with a detectable
    difference in spread or shape. Reporting only (a) would overstate the claim.
    """
    print("\n=== 5. LH/RH AGNOSTICISM (independent test) ===")
    print("  Reported as TWO separate questions -- mean shift, and distributional")
    print("  separability. A tiny mean shift does NOT by itself license a claim")
    print("  of full agnosticism.\n")

    rows = []
    for name, v in [("Estimated confidence (LEAKAGE-FREE)", C_quad),
                     ("Estimated confidence (linear)", C_lin),
                     ("Reference |o_LH - o_RH|", np.abs(delta))]:
        a, b = v[y == 0], v[y == 1]
        d = cohens_d(a, b); lo, hi = d_ci(a, b)
        auc = fast_auc(v, y)
        pp = perm_p_auc(v, y, subjects)
        lev_p = levene(a, b).pvalue
        ks_stat, ks_p = ks_2samp(a, b)
        pct = 100 * (a.mean() - b.mean()) / np.abs(v).mean()

        mean_ok = bool(abs(lo) < 0.2 and abs(hi) < 0.2)
        dist_detected = bool(pp < 0.05)
        if mean_ok and not dist_detected:
            verdict = "no detectable bias"
        elif mean_ok and dist_detected:
            verdict = "mean shift negligible, but distribution differs"
        else:
            verdict = "biased"

        rows.append({"Variable": name, "mean_LH": a.mean(), "mean_RH": b.mean(),
                     "pct_bias": pct, "Cohens_d": d, "d_lo": lo, "d_hi": hi,
                     "mean_equivalent_0.2SD": mean_ok,
                     "AUC_LH_vs_RH": auc, "perm_p": pp,
                     "levene_p": lev_p, "KS_stat": ks_stat, "KS_p": ks_p,
                     "distributional_difference_detected": dist_detected,
                     "Verdict": verdict})
        print(f"  {name}")
        print(f"     mean shift    : d={d:+.4f} [{lo:+.3f},{hi:+.3f}]  "
              f"({'within' if mean_ok else 'OUTSIDE'} +/-0.2 SD)")
        print(f"     distributional: AUC={auc:.4f}  perm p={pp:.4f}  "
              f"Levene p={lev_p:.4f}  KS p={ks_p:.2e}")
        print(f"     verdict       : {verdict}\n")

    df = pd.DataFrame(rows)
    df.to_csv("Results/agnosticism_results.csv", index=False)

    # structural control: which subspace carries the class information?
    print("  Structural control (C3<->C4 mirror symmetry):")
    # channel count inferred from the reservoir width, so the mirror works for
    # any number of electrodes: 1 + kc + kc(kc+1)/2 = P  with kc = k*nch
    k = 3
    P = full_reservoir.shape[1]
    kc = int((np.sqrt(9 + 8 * (P - 1)) - 3) / 2)
    nch = kc // k
    m = np.arange(k*nch)
    for j in range(k):
        a_, b_ = j*nch + 0, j*nch + (nch - 1)     # C3 first, C4 last
        m[a_], m[b_] = m[b_], m[a_]
    iu = np.triu_indices(k*nch)
    pos = {(i, j): p for p, (i, j) in enumerate(zip(iu[0], iu[1]))}
    qmap = [pos[(min(m[i], m[j]), max(m[i], m[j]))] for i, j in zip(iu[0], iu[1])]
    perm = np.concatenate([[0], 1 + m, 1 + k*nch + np.array(qmap)]).astype(int)

    F = full_reservoir
    subspaces = {"Full state": F,
                 "Symmetric (C3<->C4 invariant)": 0.5 * (F + F[:, perm]),
                 "Antisymmetric (lateralised)": 0.5 * (F - F[:, perm])}
    struct = []
    for nm, M in subspaces.items():
        accs = []
        for unseen in subject_folds(subjects):
            tr = ~np.isin(subjects, unseen); te = np.isin(subjects, unseen)
            sc = StandardScaler().fit(M[tr])
            clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(M[tr]), y[tr])
            accs.append((clf.predict(sc.transform(M[te])) == y[te]).mean())
        struct.append({"Subspace": nm, "LH_RH_accuracy": np.mean(accs)})
        print(f"    {nm:32s} LH/RH accuracy = {np.mean(accs):.4f}")
    sdf = pd.DataFrame(struct)
    sdf.to_csv("Results/structural_control.csv", index=False)

    yy = np.arange(len(df))
    plt.figure(figsize=(9.5, 5.6))
    colors = ["seagreen" if e else "darkorange" for e in df["mean_equivalent_0.2SD"]]
    plt.barh(yy, df["Cohens_d"],
             xerr=[df["Cohens_d"] - df["d_lo"], df["d_hi"] - df["Cohens_d"]],
             color=colors, height=0.55)
    plt.axvspan(-0.2, 0.2, color="seagreen", alpha=0.12)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(yy, df["Variable"], fontsize=11)
    plt.xlabel("Cohen's d  (LH - RH),  95% CI")
    plt.title("MEAN shift\nshaded = within $\\pm$0.2 SD")
    save_figure("Results/agnosticism_mean_shift.png")

    plt.figure(figsize=(9.5, 5.6))
    plt.barh(yy, df["AUC_LH_vs_RH"] - 0.5,
             color=["darkorange" if p < 0.05 else "seagreen" for p in df["perm_p"]],
             height=0.55)
    plt.axvline(0, color="black", linewidth=1)
    plt.yticks(yy, df["Variable"], fontsize=11)
    plt.xlabel("AUC - 0.5   (LH vs RH separability)")
    plt.title("DISTRIBUTIONAL separability\norange = detectable (perm p < 0.05)")
    for i, (a_, p_) in enumerate(zip(df["AUC_LH_vs_RH"], df["perm_p"])):
        plt.text(a_ - 0.5, i, f"  p={p_:.3f}", va="center", fontsize=11)
    save_figure("Results/agnosticism_distributional.png")

    plt.figure(figsize=(7.5, 5.6))
    bars = plt.bar(["Full", "Symmetric", "Antisymmetric"], sdf["LH_RH_accuracy"],
                   color=["steelblue", "seagreen", "indianred"])
    plt.axhline(0.5, color="grey", linestyle=":", label="chance")
    plt.ylim(0.4, 0.85); plt.ylabel("LH/RH decoding accuracy")
    plt.title("Structural control : the symmetric\nsubspace cannot see the decision")
    for b, v in zip(bars, sdf["LH_RH_accuracy"]):
        plt.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v*100:.1f}%",
                 ha="center", fontsize=12)
    plt.legend()
    save_figure("Results/agnosticism_structural.png")

    print("Saved agnosticism_results.csv, structural_control.csv + plot")
    return df, sdf


# ==================================================================
# 6. NGRC randomisation robustness
# ==================================================================
def ngrc_randomisation(y, subjects, correct_ref):
    """NGRC has no random weights; we vary the constructive choices (k, s).

    LEAKAGE-FREE: for every configuration AND every fold, the classifier, the
    confidence target (inner leave-one-subject-out), the variable selection and
    the estimator are all built from the training subjects only.
    """
    print("\n=== 6. NGRC Randomisation Robustness (leakage-free) ===")
    print("  NGRC has no random weights; we vary the constructive choices (k, s).")
    X = np.load("Data/eeg_preprocessed.npy")
    MI_START, MI_END = 125, 1000
    configs = [(3, 3), (3, 2), (4, 4), (5, 2), (5, 4), (5, 5)]
    ids = np.unique(subjects)

    def build_state(k, s):
        start = max(MI_START, (k-1)*s)
        D = np.stack([X[:, :, start - j*s: MI_END - j*s] for j in range(k)], 1)
        D = D.reshape(len(X), k*X.shape[1], -1).astype(np.float64)
        q = np.einsum("nit,njt->nij", D, D) / D.shape[-1]
        iu = np.triu_indices(k*X.shape[1])
        R = np.hstack([np.ones((len(X), 1)), D.mean(-1), q[:, iu[0], iu[1]]])
        scale = np.median(np.abs(R), 0, keepdims=True) + 1e-12
        R = np.sign(R) * np.log1p(np.abs(R) / scale); R[:, 0] = 1.0
        return R

    def delta_from(R, fit_m, app_m):
        sc = StandardScaler().fit(R[fit_m])
        clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(R[fit_m]), y[fit_m])
        lh = list(clf.classes_).index(0)
        return 2*clf.predict_proba(sc.transform(R[app_m]))[:, lh] - 1

    rows = []
    for (k, s) in configs:
        R = build_state(k, s)
        delta = np.full(len(y), np.nan)
        C_hat = np.full(len(y), np.nan)

        for unseen in subject_folds(subjects):
            tr = ~np.isin(subjects, unseen); te = np.isin(subjects, unseen)
            tr_subs = np.setdiff1d(ids, unseen)
            delta[te] = delta_from(R, tr, te)

            # inner LOSO inside the training subjects -> their confidence target
            d_in = np.full(len(y), np.nan)
            for ho in tr_subs:
                ho_m = subjects == ho
                d_in[ho_m] = delta_from(R, tr & ~ho_m, ho_m)
            C_in = np.abs(d_in)

            # selection + estimator, training subjects only
            sel = select_confidence_variables(R[tr], C_in[tr], topk=26)
            sc = StandardScaler().fit(R[np.ix_(tr, sel)])
            Xq = quad_expand(sc.transform(R[:, sel]))
            sc2 = StandardScaler().fit(Xq[tr])
            m = Ridge(alpha=100.0).fit(sc2.transform(Xq[tr]), C_in[tr])
            C_hat[te] = m.predict(sc2.transform(Xq[te]))

        corr = ((delta < 0).astype(int) == y).astype(int)
        rows.append({"k": k, "skip": s, "n_features": R.shape[1],
                     "accuracy": corr.mean(),
                     "AUROC2": fast_auc(C_hat, corr),
                     "Spearman_vs_correct": spearmanr(C_hat, corr)[0],
                     "bias_d": cohens_d(C_hat[y == 0], C_hat[y == 1]),
                     "bias_AUC": fast_auc(C_hat, y)})
        print(f"  k={k}, s={s} : {R.shape[1]:3d} features, acc={corr.mean():.4f}, "
              f"AUROC2={rows[-1]['AUROC2']:.4f}, bias d={rows[-1]['bias_d']:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv("Results/ngrc_randomisation.csv", index=False)
    print(f"\n  Across randomisations : accuracy {df.accuracy.mean():.4f} +/- {df.accuracy.std():.4f}"
          f" | AUROC2 {df.AUROC2.mean():.4f} +/- {df.AUROC2.std():.4f}"
          f" | bias d {df.bias_d.mean():+.4f} +/- {df.bias_d.std():.4f}")

    print("Saved ngrc_randomisation.csv")
    return df


# ==================================================================
if __name__ == "__main__":
    apply_style()
    full_reservoir = np.load("Data/ngrc_reservoir.npy")
    selected_reservoir = np.load("Results/selected_reservoir.npy")
    C = np.load("Data/reference_confidence.npy")
    delta = np.load("Data/readout_delta.npy")
    y = np.load("Data/eeg_labels.npy")
    subjects = np.load("Data/eeg_subjects.npy")
    correct = np.load("Data/correct.npy")
    ranking = pd.read_csv("Results/feature_ranking.csv")

    # The agnosticism test uses the LEAKAGE-FREE estimate from Module 7.
    C_nested = np.load("Data/estimated_confidence.npy")

    def_df = compare_confidence_definitions(selected_reservoir, delta, y, subjects, correct)
    est_df, C_quad, C_lin = compare_estimators(selected_reservoir, C, subjects, correct)
    abl_df = ablation_study(full_reservoir, selected_reservoir, C, subjects, correct)
    sens_df = sensitivity_analysis(ranking, full_reservoir, C, subjects, correct)
    agn_df, struct_df = agnosticism_test(C_nested, C_lin, delta, full_reservoir, y, subjects)
    rand_df = ngrc_randomisation(y, subjects, correct)

    # ---- final combined summary ----
    summary = pd.DataFrame([
        {"Quantity": "LH/RH classification accuracy (unseen subjects)",
         "Value": f"{correct.mean()*100:.1f}%"},
        {"Quantity": "Confidence AUROC2, LEAKAGE-FREE (report this)",
         "Value": f"{fast_auc(C_nested, correct):.4f}"},
        {"Quantity": "Confidence AUROC2, discovery protocol (optimistic)",
         "Value": f"{fast_auc(C_quad, correct):.4f}"},
        {"Quantity": "Confidence AUROC2 (linear readout)",
         "Value": f"{fast_auc(C_lin, correct):.4f}"},
        {"Quantity": "Spearman(confidence, correctness), leakage-free",
         "Value": f"{spearmanr(C_nested, correct)[0]:.4f}"},
        {"Quantity": "Kendall(confidence, reference)",
         "Value": f"{kendalltau(C, C_quad)[0]:.4f}"},
        {"Quantity": "LH/RH bias Cohen's d",
         "Value": f"{agn_df.iloc[0]['Cohens_d']:+.4f} "
                  f"[{agn_df.iloc[0]['d_lo']:+.3f}, {agn_df.iloc[0]['d_hi']:+.3f}]"},
        {"Quantity": "LH/RH mean shift within +/-0.2 SD",
         "Value": str(agn_df.iloc[0]["mean_equivalent_0.2SD"])},
        {"Quantity": "LH/RH distributional difference detected",
         "Value": f'{agn_df.iloc[0]["distributional_difference_detected"]} '
                  f'(AUC={agn_df.iloc[0]["AUC_LH_vs_RH"]:.4f}, '
                  f'perm p={agn_df.iloc[0]["perm_p"]:.4f})'},
        {"Quantity": "Agnosticism verdict",
         "Value": agn_df.iloc[0]["Verdict"]},
        {"Quantity": "Symmetric subspace LH/RH accuracy",
         "Value": f"{struct_df.iloc[1]['LH_RH_accuracy']*100:.1f}%"},
        {"Quantity": "Accuracy at 25% coverage (leakage-free)",
         "Value": f"{correct[np.argsort(-C_nested)][:len(C_nested)//4].mean()*100:.1f}%"},
    ])
    summary.to_csv("Results/final_summary.csv", index=False)
    print("\n" + "="*66)
    print("FINAL SUMMARY")
    print("="*66)
    print(summary.to_string(index=False))
    print("\nSaved Results/final_summary.csv")

    print("\nModule 8 Completed Successfully.")
