"""

Module 10 : Module_10_variable_activity.py

WHAT ARE THE SELECTED VARIABLES, AND DO THEY COMPUTE CONFIDENCE?



        mean_t [ C4(t-3) * C4(t-6) ]  =  autocovariance of C4 at lag 3 samples
        mean_t [ C4(t-0) * C4(t-0) ]  =  variance of C4  =  band power

    At 250 Hz, 3 samples is 12 ms and 6 samples is 24 ms. For a narrowband
    rhythm at frequency f the autocovariance at lag tau is approximately
    power * cos(2 pi f tau), so:

        lag-0 terms          carry BAND POWER
        lag-3 / lag-0 ratio  carries the DOMINANT FREQUENCY

    
"""

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from Module_1_eeg_data import apply_style, save_figure, EEGData

FS = 250.0
MI_START, MI_END = 125, 1000
K, S = 3, 3
CH = EEGData.CHANNELS                       # ("C3", "C4") in this study


def parse_term(name, k=K, s=S):
    """'C4(t-3)*C4(t-6)' -> (channel_a, lag_a, channel_b, lag_b), or None."""
    nm = name.replace("^2", "")
    parts = nm.split("*") if "*" in nm else [nm, nm]
    out = []
    for p in parts:
        p = p.strip()
        if "(" not in p:
            return None
        ch = p.split("(")[0]
        lag = int(p.split("t-")[1].rstrip(")"))
        if ch not in CH:
            return None
        out.append((CH.index(ch), lag))
    return out[0][0], out[0][1], out[1][0], out[1][1]


def term_value(X, ch_a, lag_a, ch_b, lag_b):
    """The raw, uncompressed reservoir term for every trial."""
    start = max(MI_START, (K - 1) * S)
    a = X[:, ch_a, start - lag_a: MI_END - lag_a].astype(np.float64)
    b = X[:, ch_b, start - lag_b: MI_END - lag_b].astype(np.float64)
    return (a * b).mean(-1)


def term_running(X, ch_a, lag_a, ch_b, lag_b, step=10):
    """The same term as a causal running mean, giving it a timecourse."""
    start = max(MI_START, (K - 1) * S)
    a = X[:, ch_a, start - lag_a: MI_END - lag_a].astype(np.float64)
    b = X[:, ch_b, start - lag_b: MI_END - lag_b].astype(np.float64)
    prod = a * b
    grid = np.arange(step - 1, prod.shape[1], step)
    return np.cumsum(prod, axis=1)[:, grid] / (grid + 1), (start + grid) / FS


def safe_name(nm):
    return (nm.replace("*", "x").replace("(", "").replace(")", "")
              .replace("-", "").replace("^", "p"))


if __name__ == "__main__":
    apply_style()
    print("=" * 76)
    print("MODULE 10 : SELECTED-VARIABLE ACTIVITY AND CONFIDENCE COMPUTATION")
    print("=" * 76)

    X = np.load("Data/eeg_preprocessed.npy")
    y = np.load("Data/eeg_labels.npy")
    conf = np.load("Data/reference_confidence.npy")
    correct = np.load("Data/correct.npy")
    split = pd.read_csv("Results/confidence_vs_uncertainty_variables.csv")
    print(f"  {X.shape[0]} trials, {X.shape[1]} channels ({', '.join(CH)})")
    print(f"  {(split.Encoding=='confidence').sum()} confidence-encoding, "
          f"{(split.Encoding=='uncertainty').sum()} uncertainty-encoding variables\n")

    # ---- interpretable physiological quantities, one value per trial
    seg = X[:, :, MI_START:MI_END].astype(np.float64)
    power = {c: (seg[:, i] ** 2).mean(-1) for i, c in enumerate(CH)}
    f, P = welch(seg, fs=FS, nperseg=256, axis=-1)
    band = (f >= 8) & (f <= 30)
    peak_f = {c: f[band][P[:, i][:, band].argmax(1)] for i, c in enumerate(CH)}
    coupling = (seg[:, 0] * seg[:, -1]).mean(-1)
    lat = np.log(power[CH[0]] + 1e-12) - np.log(power[CH[-1]] + 1e-12)

    rows = []
    for _, r in split.iterrows():
        pt = parse_term(r.Feature)
        if pt is None:
            continue
        ca, la, cb, lb = pt
        v = term_value(X, ca, la, cb, lb)
        rows.append(dict(
            Feature=r.Feature, Encoding=r.Encoding, Spearman=r.Spearman,
            kind=("within-channel" if ca == cb else "cross-channel"),
            lag_ms=abs(la - lb) / FS * 1000,
            rho_C3_power=spearmanr(v, power[CH[0]])[0],
            rho_C4_power=spearmanr(v, power[CH[-1]])[0],
            rho_peak_freq=spearmanr(v, peak_f[CH[-1]])[0],
            rho_coupling=spearmanr(v, coupling)[0],
            rho_lateralisation=spearmanr(v, lat)[0],
            rho_confidence=spearmanr(v, conf)[0]))
    tab = pd.DataFrame(rows)
    tab.to_csv("Results/variable_activity_interpretation.csv", index=False)
    print("What each selected variable actually measures (Spearman):")
    print(tab[["Feature", "Encoding", "kind", "lag_ms", "rho_C4_power",
               "rho_peak_freq", "rho_coupling", "rho_confidence"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    ac0 = term_value(X, CH.index(CH[-1]), 0, CH.index(CH[-1]), 0)
    ac3 = term_value(X, CH.index(CH[-1]), 0, CH.index(CH[-1]), 3)
    print(f"\n  identity checks:")
    print(f"    lag-0 term vs {CH[-1]} band power            : "
          f"rho = {spearmanr(ac0, power[CH[-1]])[0]:+.4f}")
    print(f"    lag-12ms / lag-0 ratio vs peak frequency : "
          f"rho = {spearmanr(ac3/(ac0+1e-12), peak_f[CH[-1]])[0]:+.4f}")

    conf_rows = tab[tab.Encoding == "confidence"]
    unc_rows = tab[tab.Encoding == "uncertainty"]
    print(f"\n  confidence-encoding variables are "
          f"{(conf_rows.kind == 'within-channel').sum()}/{len(conf_rows)} within-channel")
    print(f"  uncertainty-encoding variables are "
          f"{(unc_rows.kind == 'cross-channel').sum()}/{len(unc_rows)} cross-channel")

    # -------------------------------------------------------------- figure 1
    x = np.arange(len(tab)); w = 0.2
    plt.figure(figsize=(13, 6))
    for off, col, lab in [(-1.5*w, "rho_C4_power", f"{CH[-1]} band power"),
                          (-0.5*w, "rho_peak_freq", f"{CH[-1]} peak frequency"),
                          (0.5*w, "rho_coupling", f"{CH[0]}-{CH[-1]} coupling"),
                          (1.5*w, "rho_confidence", "confidence")]:
        plt.bar(x + off, tab[col], w, label=lab)
    plt.axhline(0, color="black", lw=1.2)
    plt.xticks(x, tab.Feature, rotation=45, ha="right", fontsize=10)
    plt.ylabel(r"Spearman $\rho$")
    plt.title("What the selected variables measure\n"
              "power and spectral shape, or inter-electrode coupling")
    plt.legend(fontsize=12)
    save_figure("Results/variable_meaning.png")

    # -------------------------------------------------------------- figure 2
    plt.figure(figsize=(10, 6))
    mk = {"within-channel": "o", "cross-channel": "s"}
    for kind in tab["kind"].unique():
        m = tab["kind"] == kind
        plt.scatter(tab.loc[m, "rho_coupling"], tab.loc[m, "Spearman"],
                    marker=mk[kind], s=90, label=kind,
                    c=["#1b7f5a" if v > 0 else "#b3202c"
                       for v in tab.loc[m, "Spearman"]])
    plt.axhline(0, color="black", lw=1.2)
    plt.axvline(0, color="#7a7a7a", ls=":", lw=2)
    plt.xlabel(rf"$\rho$ with {CH[0]}-{CH[-1]} coupling")
    plt.ylabel(r"$\rho$ with confidence  (sign retained)")
    plt.title("Uncertainty-encoding variables are the coupling terms\n"
              "green above zero = confidence, red below zero = uncertainty")
    plt.legend(fontsize=12)
    save_figure("Results/variable_coupling_vs_confidence.png")

    # ------------------------------------------------------- figures 3 and 4
    hi = conf >= np.percentile(conf, 70)
    lo = conf <= np.percentile(conf, 30)
    picks = []
    if len(conf_rows):
        picks.append((conf_rows.iloc[0].Feature, "confidence-encoding"))
    if len(unc_rows):
        picks.append((unc_rows.iloc[-1].Feature, "uncertainty-encoding"))

    for nm, kind in picks:
        ca, la, cb, lb = parse_term(nm)
        rf, t = term_running(X, ca, la, cb, lb)
        plt.figure(figsize=(10, 5.6))
        for m, colour, lab in [(hi, "#1b7f5a", "high confidence (top 30%)"),
                               (lo, "#b3202c", "low confidence (bottom 30%)")]:
            mu = rf[m].mean(0); se = rf[m].std(0) / np.sqrt(m.sum())
            plt.plot(t, mu, color=colour, label=lab)
            plt.fill_between(t, mu - se, mu + se, color=colour, alpha=0.25, lw=0)
        plt.xlabel("Time within trial (s)")
        plt.ylabel(f"running mean of {nm}")
        plt.title(f"{nm}  ({kind})\nactivity for high- against low-confidence trials")
        plt.legend(fontsize=12)
        save_figure(f"Results/variable_timecourse_{safe_name(nm)}.png")

        sep = np.array([(rf[hi, j].mean() - rf[lo, j].mean()) /
                        np.sqrt((rf[hi, j].var(ddof=1) + rf[lo, j].var(ddof=1)) / 2)
                        for j in range(rf.shape[1])])
        print(f"\n  {nm} ({kind}):")
        print(f"    separation d at 0.6 s = {sep[0]:+.3f}, "
              f"at 2.0 s = {sep[len(sep)//2]:+.3f}, at 4.0 s = {sep[-1]:+.3f}")

    plt.figure(figsize=(10, 5.6))
    for nm, kind in picks:
        ca, la, cb, lb = parse_term(nm)
        rf, t = term_running(X, ca, la, cb, lb)
        sep = np.array([(rf[hi, j].mean() - rf[lo, j].mean()) /
                        np.sqrt((rf[hi, j].var(ddof=1) + rf[lo, j].var(ddof=1)) / 2)
                        for j in range(rf.shape[1])])
        plt.plot(t, sep, label=f"{nm} ({kind})",
                 color="#1b7f5a" if kind.startswith("confidence") else "#b3202c")
    plt.axhline(0, color="black", lw=1.2)
    plt.xlabel("Time within trial (s)")
    plt.ylabel("Cohen's d  (high vs low confidence)")
    plt.title("When do the selected variables start to separate confidence?")
    plt.legend(fontsize=11)
    save_figure("Results/variable_separation_over_time.png")

    print("\nSaved Results/variable_activity_interpretation.csv and four figures")
    print("\nModule 10 Completed Successfully.")
