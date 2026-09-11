"""
=====================================================================
Module 7 : Module_7_confidence_estimator.py
=====================================================================
DISCOVERY: whole-dataset confidence-variable selection; descriptive/optimistic.
VALIDATION: nested subject-based CV; unseen subjects never enter fitting.
Nested protocol:
1. classifier trained on training subjects only
2. inner LOSO confidence target
3. Spearman + Kendall selection on training subjects only
4. quadratic ridge fitted on training subjects only
5. unseen subjects used only for final scoring
=====================================================================
"""

import numpy as np
import pandas as pd
import pickle
from itertools import combinations_with_replacement
from scipy.stats import spearmanr, kendalltau, rankdata
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure
from Module_6_feature_selection import select_confidence_variables


class ConfidenceEstimator:

    def __init__(self, alpha=100.0, clf_C=0.05, n_folds=3, seed=0, spearman_threshold=0.05, q_threshold=0.05, topk=26):
        self.alpha, self.clf_C, self.n_folds, self.seed = alpha, clf_C, n_folds, seed
        self.spearman_threshold, self.q_threshold, self.topk = spearman_threshold, q_threshold, topk

    def load_data(self, folder="Data"):
        self.R = np.load(f"{folder}/ngrc_reservoir.npy")
        self.y = np.load(f"{folder}/eeg_labels.npy")
        self.subjects = np.load(f"{folder}/eeg_subjects.npy")
        self.delta_outer = np.load(f"{folder}/readout_delta.npy")
        self.correct = np.load(f"{folder}/correct.npy")
        self.C_outer = np.abs(self.delta_outer)
        self.sel_discovery = np.load("Results/selected_indices.npy")
        print("Reservoir            :", self.R.shape)
        print("Discovery selection  :", len(self.sel_discovery), "variables (from Module 6)")

    def folds(self):
        ids = np.unique(self.subjects)
        return np.array_split(np.random.default_rng(self.seed).permutation(ids), self.n_folds)

    @staticmethod
    def _auc(score, label):
        r = rankdata(score); n1 = int(np.sum(label == 1)); n0 = len(label) - n1
        return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    @staticmethod
    def quad_expand(Z):
        idx = list(combinations_with_replacement(range(Z.shape[1]), 2))
        i = np.array([a for a, b in idx]); j = np.array([b for a, b in idx])
        return np.hstack([Z, Z[:, i] * Z[:, j]])

    def _delta(self, fit_mask, apply_mask):
        sc = StandardScaler().fit(self.R[fit_mask])
        clf = LogisticRegression(C=self.clf_C, max_iter=3000)
        clf.fit(sc.transform(self.R[fit_mask]), self.y[fit_mask])
        lh = list(clf.classes_).index(0)
        return 2 * clf.predict_proba(sc.transform(self.R[apply_mask]))[:, lh] - 1

    def _select(self, Rm, Cm):
        return select_confidence_variables(Rm, Cm, spearman_threshold=self.spearman_threshold,
                                           q_threshold=self.q_threshold, mi_percentile=25, topk=self.topk)

    def _fit_apply(self, sel, target, tr, te):
        sc = StandardScaler().fit(self.R[np.ix_(tr, sel)])
        X = self.quad_expand(sc.transform(self.R[:, sel]))
        sc2 = StandardScaler().fit(X[tr])
        self.model = Ridge(alpha=self.alpha).fit(sc2.transform(X[tr]), target[tr])
        return self.model.predict(sc2.transform(X[te]))

    def run_discovery(self):
        out = np.full(len(self.y), np.nan)
        for unseen in self.folds():
            tr = ~np.isin(self.subjects, unseen); te = np.isin(self.subjects, unseen)
            out[te] = self._fit_apply(self.sel_discovery, self.C_outer, tr, te)
        return out

    def run_nested(self, inner="loso", verbose=True):
        out = np.full(len(self.y), np.nan); ids = np.unique(self.subjects); self.fold_selections = []
        for fi, unseen in enumerate(self.folds()):
            tr = ~np.isin(self.subjects, unseen); te = np.isin(self.subjects, unseen)
            tr_subs = np.setdiff1d(ids, unseen); d_in = np.full(len(self.y), np.nan)
            groups = [np.array([s]) for s in tr_subs] if inner == "loso" else np.array_split(np.random.default_rng(1).permutation(tr_subs), 3)
            for ho in groups:
                ho_m = np.isin(self.subjects, ho); d_in[ho_m] = self._delta(tr & ~ho_m, ho_m)
            C_in = np.abs(d_in); sel = self._select(self.R[tr], C_in[tr]); self.fold_selections.append(sel)
            out[te] = self._fit_apply(sel, C_in, tr, te)
            if verbose:
                ov = len(set(sel.tolist()) & set(self.sel_discovery.tolist())) / len(sel)
                print(f"  fold {fi}: train {sorted(tr_subs.tolist())} -> test {sorted(unseen.tolist())} | {len(sel)} variables selected ({ov*100:.0f}% overlap with discovery set)")
        return out

    def evaluate(self, C_hat, name):
        ok = np.isfinite(C_hat)
        return {
            "Protocol": name,
            "RMSE": np.sqrt(mean_squared_error(self.C_outer[ok], C_hat[ok])),
            "MAE": mean_absolute_error(self.C_outer[ok], C_hat[ok]),
            "R2": r2_score(self.C_outer[ok], C_hat[ok]),
            "Spearman_vs_reference": spearmanr(C_hat[ok], self.C_outer[ok])[0],
            "Kendall_vs_reference": kendalltau(C_hat[ok], self.C_outer[ok])[0],
            "AUROC2_vs_correct": self._auc(C_hat[ok], self.correct[ok]),
            "Spearman_vs_correct": spearmanr(C_hat[ok], self.correct[ok])[0],
            "Spearman_vs_correct_p": spearmanr(C_hat[ok], self.correct[ok])[1]
        }

    def leakage_decomposition(self, C_disc, C_nest):
        print("\n=== Leakage decomposition ===")
        c2 = np.full(len(self.y), np.nan)
        for unseen in self.folds():
            tr = ~np.isin(self.subjects, unseen); te = np.isin(self.subjects, unseen)
            sel = self._select(self.R[tr], self.C_outer[tr]); c2[te] = self._fit_apply(sel, self.C_outer, tr, te)
        a1, a2, a3 = self._auc(C_disc, self.correct), self._auc(c2, self.correct), self._auc(C_nest, self.correct)
        df = pd.DataFrame([
            {"Condition": "1. selection leaky + target leaky", "AUROC2": a1},
            {"Condition": "2. selection fixed + target leaky", "AUROC2": a2},
            {"Condition": "3. selection fixed + target fixed (nested)", "AUROC2": a3},
            {"Condition": "-> optimism from SELECTION leakage", "AUROC2": a1 - a2},
            {"Condition": "-> optimism from TARGET leakage", "AUROC2": a2 - a3}
        ])
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        df.to_csv("Results/leakage_decomposition.csv", index=False)
        print("\n  Note: target-leakage component also contains a genuine difficulty penalty.")
        print("  It is therefore an UPPER bound on true information leakage, not a pure estimate.")
        return df

    def plot_comparison(self, C_disc, C_nest, n=400):
        plt.figure(figsize=(10, 5.6))
        plt.plot(self.C_outer[:n], label="Reference confidence", linewidth=1.6)
        plt.plot(C_nest[:n], "--", label="Estimated (leakage-free)", linewidth=1.6)
        plt.xlabel("Trial"); plt.ylabel("Confidence"); plt.title("Reference vs Estimated Confidence (unseen subjects)"); plt.legend()
        save_figure("Results/confidence_timeseries.png")

        plt.figure(figsize=(8, 6))
        plt.scatter(self.C_outer, C_nest, s=9, alpha=0.25, edgecolor="none")
        plt.plot([0, 1], [0, 1], "r--", linewidth=2, label="perfect agreement")
        plt.xlim(0, 1); plt.ylim(0, 1); plt.xlabel("Reference confidence"); plt.ylabel("Estimated confidence")
        plt.title("Estimated vs Reference (leakage-free)"); plt.legend()
        save_figure("Results/confidence_scatter.png")

        a_d, a_n = self._auc(C_disc, self.correct), self._auc(C_nest, self.correct)
        plt.figure(figsize=(7.5, 5.6))
        bars = plt.bar(["Discovery\n(whole-dataset\nselection)", "Validation\n(leakage-free\nnested)"], [a_d, a_n], color=["lightsteelblue", "seagreen"])
        plt.axhline(0.5, color="grey", linestyle=":", label="chance"); plt.ylim(0.45, 0.75)
        plt.ylabel("AUROC2 vs correctness"); plt.title(f"Optimism of the discovery protocol : {a_d-a_n:+.3f}")
        for b, v in zip(bars, [a_d, a_n]): plt.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=13)
        plt.legend(); save_figure("Results/confidence_optimism.png")

        counts = np.zeros(self.R.shape[1])
        for sel in self.fold_selections: counts[sel] += 1
        names = np.load("Data/feature_names.npy", allow_pickle=True); stable = np.argsort(-counts)[:12]
        plt.figure(figsize=(9, 6)); plt.barh(range(len(stable)), counts[stable][::-1], color="steelblue")
        plt.yticks(range(len(stable)), [names[i] for i in stable][::-1], fontsize=11)
        plt.xlabel(f"Selected in how many of {len(self.fold_selections)} folds")
        plt.title("Selection Stability\n(variables re-found independently in each fold)")
        save_figure("Results/confidence_selection_stability.png")

    def plot_calibration(self, C_hat):
        correct = self.correct; q = pd.qcut(C_hat, 10, labels=False)
        acc = np.array([correct[q == i].mean() for i in range(10)])
        se = np.array([correct[q == i].std()/np.sqrt((q == i).sum()) for i in range(10)])
        plt.figure(figsize=(9, 5.6)); plt.errorbar(range(1, 11), acc, yerr=se, marker="o", color="steelblue")
        plt.axhline(correct.mean(), color="black", linestyle="--", label=f"overall {correct.mean()*100:.1f}%")
        plt.xlabel("Estimated-confidence decile"); plt.ylabel("Accuracy"); plt.title("Calibration on UNSEEN subjects (leakage-free)")
        plt.legend(); save_figure("Results/confidence_calibration.png")

        cov = np.linspace(0.1, 1.0, 19); cc = correct[np.argsort(-C_hat)]
        plt.figure(figsize=(9, 5.6)); plt.plot(cov, [cc[:max(1, int(f*len(cc)))].mean() for f in cov], marker="o", markersize=5, color="seagreen")
        plt.axhline(correct.mean(), color="black", linestyle="--", label="keep all trials")
        plt.xlabel("Coverage (fraction of most-confident trials kept)"); plt.ylabel("Accuracy")
        plt.title("Selective Decoding (leakage-free)"); plt.legend(); save_figure("Results/selective_decoding.png")


if __name__ == "__main__":
    apply_style(); est = ConfidenceEstimator(); est.load_data()

    print("\n--- DISCOVERY protocol (whole-dataset selection, OPTIMISTIC) ---")
    C_disc = est.run_discovery(); m_disc = est.evaluate(C_disc, "Discovery (whole-dataset selection)")
    print(f"  AUROC2 vs correctness : {m_disc['AUROC2_vs_correct']:.4f}   <- descriptive only, do NOT quote as predictive")

    print("\n--- VALIDATION protocol (leakage-free nested) ---")
    C_nest = est.run_nested(inner="loso"); m_nest = est.evaluate(C_nest, "Validation (leakage-free nested)")

    print("\nLeakage-free performance on UNSEEN subjects")
    print(f"  RMSE                     : {m_nest['RMSE']:.6f}")
    print(f"  R2                       : {m_nest['R2']:.6f}")
    print(f"  Spearman vs reference    : {m_nest['Spearman_vs_reference']:.4f}")
    print(f"  Kendall vs reference     : {m_nest['Kendall_vs_reference']:.4f}")
    print(f"  AUROC2 vs correctness    : {m_nest['AUROC2_vs_correct']:.4f}   <- REPORT THIS")
    print(f"  Spearman vs correctness  : {m_nest['Spearman_vs_correct']:.4f} (p={m_nest['Spearman_vs_correct_p']:.2e})")

    # UPDATED: fold-level AUROC2, top-10% and top-25% accuracy
    fold_auc, fold_a10, fold_a25 = [], [], []
    for unseen in est.folds():
        te = np.isin(est.subjects, unseen); c, y = C_nest[te], est.correct[te]; s = np.argsort(-c)
        fold_auc.append(est._auc(c, y))
        fold_a10.append(y[s[:max(1, int(.10*len(y)))]].mean()*100)
        fold_a25.append(y[s[:max(1, int(.25*len(y)))]].mean()*100)

    print(f"\n  AUROC2 across folds : {np.mean(fold_auc):.4f} ± {np.std(fold_auc,ddof=1):.4f}")
    print(f"  Accuracy @10%       : {np.mean(fold_a10):.1f}% ± {np.std(fold_a10,ddof=1):.2f}%")
    print(f"  Accuracy @25%       : {np.mean(fold_a25):.1f}% ± {np.std(fold_a25,ddof=1):.2f}%")

    est.leakage_decomposition(C_disc, C_nest)
    pd.DataFrame([m_disc, m_nest]).to_csv("Results/confidence_metrics.csv", index=False)
    pd.DataFrame({"trial": np.arange(len(est.y)), "subject": est.subjects, "reference_confidence": est.C_outer,
                  "estimated_discovery": C_disc, "estimated_leakage_free": C_nest, "correct": est.correct}).to_csv("Results/confidence_prediction.csv", index=False)
    np.save("Data/estimated_confidence.npy", C_nest); np.save("Data/estimated_confidence_discovery.npy", C_disc)

    with open("Results/confidence_model.pkl", "wb") as f: pickle.dump(est.model, f)

    print("\nSaved confidence_metrics.csv, confidence_prediction.csv, confidence_model.pkl")
    est.plot_comparison(C_disc, C_nest); est.plot_calibration(C_nest)
    print("\nModule 7 Completed Successfully.")