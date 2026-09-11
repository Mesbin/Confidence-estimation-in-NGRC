"""
=====================================================================
Study 2, Module 1 : S2_Module_1_realtime_reservoir.py
Real-Time Confidence-Aware NGRC for Early Motor-Imagery Decision Making

Study 1 asked "can confidence be estimated from the reservoir?".
Study 2 asks "can it be monitored early enough to commit a command before the
trial ends?".

The reservoir state is kept as a function of time instead of being collapsed
at the end of the trial:

    X_bar(t) = (1 / (t - t0)) * sum_{tau <= t} X(tau)

This is causal because only current and past samples are used. At t = T it
reproduces the trial-level representation.

Preprocessing:
  - Subject 4 removed
  - C3, Cz and C4 retained
  - band-pass 8-30 Hz
  - Euclidean Alignment

Output:
  Data/rt_state.npy
  Data/rt_time.npy
  Data/labels.npy
  Data/subjects.npy
=====================================================================
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


FS = 250.0
BAND = (8, 30)
MI_START, MI_END = 125, 1000

# Updated Study 2 channel configuration
CHANNELS = ("C3", "Cz", "C4")

NGRC_K, NGRC_S = 3, 3
GRID_STEP = 10
N_FOLDS, SEED = 3, 0

BLUE, ORANGE, GREEN, GREY, RED = (
    "#1f5fa9",
    "#d1571f",
    "#1b7f5a",
    "#7a7a7a",
    "#b3202c"
)


def apply_style():
    """400 dpi, base font 15 pt, line width 2.8, no grid, no box."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.6,
        "axes.labelpad": 8,
        "axes.titlepad": 12,
        "lines.linewidth": 2.8,
        "lines.markersize": 8,
        "lines.markeredgewidth": 0,
        "patch.linewidth": 0,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "errorbar.capsize": 4,
    })


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"  figure -> {filename}")


def subject_folds(subjects, n_folds=N_FOLDS, seed=SEED):
    ids = np.unique(subjects)
    return np.array_split(
        np.random.default_rng(seed).permutation(ids),
        n_folds
    )


# --------------------------------------------------------------- preprocessing

def load_raw(folder="Data"):
    X = np.load(
        f"{folder}/X_epochs.npy"
    ).astype(np.float32) * 1e6

    y = np.load(
        f"{folder}/y_labels.npy"
    )

    subjects = pd.read_csv(
        f"{folder}/epoch_labels.csv"
    )["Subject"].values

    # Remove Subject 4
    keep = subjects != 4
    X, y, subjects = X[keep], y[keep], subjects[keep]

    # Keep C3, Cz and C4
    X = X[:, [0, 1, 2], :]

    print(
        f"  Subject 4 removed | "
        f"C3+Cz+C4 retained | "
        f"trials {X.shape[0]} | "
        f"channels {X.shape[1]} | "
        f"subjects {len(np.unique(subjects))}"
    )

    return X, y, subjects


def bandpass(X, band=BAND):
    sos = butter(
        4,
        list(band),
        btype="band",
        fs=FS,
        output="sos"
    )

    Xf = sosfiltfilt(
        sos,
        X,
        axis=-1
    ).astype(np.float32)

    return Xf / np.sqrt((Xf ** 2).mean())


def euclidean_align(X, subjects):
    Xa = np.empty_like(X)

    for s in np.unique(subjects):

        m = subjects == s
        Xs = X[m].astype(np.float64)

        R = (
            np.einsum(
                "nct,ndt->cd",
                Xs,
                Xs
            )
            / (Xs.shape[0] * Xs.shape[2])
        )

        w, V = np.linalg.eigh(R)

        Xa[m] = np.einsum(
            "cd,ndt->nct",
            V
            @ np.diag(
                1 / np.sqrt(
                    np.maximum(w, 1e-12)
                )
            )
            @ V.T,
            Xs
        ).astype(X.dtype)

    return Xa


# ------------------------------------------------- causal running-mean state

def running_state(
    X,
    k=NGRC_K,
    s=NGRC_S,
    step=GRID_STEP,
    chunk=250
):
    """
    X_bar(t) at every decision opportunity.
    Returns (state, grid_samples).
    """

    n, nch, _ = X.shape

    kc = k * nch

    iu = np.triu_indices(kc)

    start = max(
        MI_START,
        (k - 1) * s
    )

    grid = np.arange(
        start + step - 1,
        MI_END - 1,
        step
    )

    grid = np.append(
        grid,
        MI_END - 1
    )

    grid = np.unique(grid)

    gi = grid - start

    cnt = (
        gi + 1
    ).astype(np.float64)

    out = np.zeros(
        (
            n,
            len(grid),
            1 + kc + len(iu[0])
        ),
        np.float32
    )

    for a in range(0, n, chunk):

        b = min(
            a + chunk,
            n
        )

        D = np.stack(
            [
                X[
                    a:b,
                    :,
                    start - j * s:
                    MI_END - j * s
                ]
                for j in range(k)
            ],
            1
        )

        D = D.reshape(
            b - a,
            kc,
            -1
        ).astype(np.float64)

        prod = (
            D[:, iu[0], :]
            * D[:, iu[1], :]
        )

        lin = (
            np.cumsum(D, axis=2)[:, :, gi]
            / cnt
        )

        qua = (
            np.cumsum(prod, axis=2)[:, :, gi]
            / cnt
        )

        out[
            a:b,
            :,
            0
        ] = 1.0

        out[
            a:b,
            :,
            1:1 + kc
        ] = lin.transpose(
            0, 2, 1
        )

        out[
            a:b,
            :,
            1 + kc:
        ] = qua.transpose(
            0, 2, 1
        )

    return out, grid


def signed_log_fixed(state):
    """Signed-log compression using the FULL-TRIAL scale."""

    scale = np.median(
        np.abs(
            state[
                :,
                -1,
                :
            ].astype(np.float64)
        ),
        axis=0,
        keepdims=True
    ) + 1e-12

    out = (
        np.sign(state)
        * np.log1p(
            np.abs(state) / scale
        )
    )

    out[:, :, 0] = 1.0

    return out.astype(np.float32)


def feature_names(
    k=NGRC_K,
    s=NGRC_S
):
    base = [
        f"{c}(t-{j*s})"
        for j in range(k)
        for c in CHANNELS
    ]

    iu = np.triu_indices(
        len(base)
    )

    return (
        ["const"]
        + base
        + [
            f"{base[i]}^2"
            if i == j
            else f"{base[i]}*{base[j]}"
            for i, j in zip(*iu)
        ]
    )


if __name__ == "__main__":

    os.makedirs(
        "Data",
        exist_ok=True
    )

    os.makedirs(
        "Results",
        exist_ok=True
    )

    apply_style()

    print("=" * 70)
    print(
        "STUDY 2, MODULE 1 : "
        "causal running-mean NGRC reservoir"
    )
    print("=" * 70)

    X, y, subjects = load_raw()

    print(
        f"  trials {X.shape[0]} | "
        f"channels {X.shape[1]} | "
        f"{X.shape[2]/FS:.1f} s at {FS:.0f} Hz | "
        f"subjects {len(np.unique(subjects))}"
    )

    Xa = euclidean_align(
        bandpass(X),
        subjects
    )

    print(
        f"  band-pass {BAND[0]}-{BAND[1]} Hz + "
        f"Euclidean Alignment"
    )

    state, grid = running_state(Xa)

    state = signed_log_fixed(state)

    t = grid / FS

    print(
        f"  running-mean state : {state.shape} -> "
        f"{len(grid)} decision opportunities from "
        f"{t[0]:.2f} s to {t[-1]:.2f} s every "
        f"{GRID_STEP/FS*1000:.0f} ms"
    )

    np.save(
        "Data/rt_state.npy",
        state
    )

    np.save(
        "Data/rt_time.npy",
        t
    )

    np.save(
        "Data/labels.npy",
        y
    )

    np.save(
        "Data/subjects.npy",
        subjects
    )

    np.save(
        "Data/feature_names.npy",
        np.array(
            feature_names(),
            dtype=object
        )
    )

    # --- figure 1a : the state of one trial, updated on-line

    plt.figure(
        figsize=(9, 5.6)
    )

    for i in [1, 2, 3]:
        plt.plot(
            t,
            state[0, :, i],
            label=f"linear dim {i}"
        )

    plt.plot(
        t,
        state[0, :, 20],
        ls="--",
        color=GREY,
        label="quadratic dim"
    )

    plt.xlabel(
        "Time within trial (s)"
    )

    plt.ylabel(
        "State value"
    )

    plt.title(
        "Running-mean reservoir state, one trial\n"
        "(updated on-line, no future samples)"
    )

    plt.legend(
        fontsize=12
    )

    save_figure(
        "Results/S2_fig01_state_one_trial.png"
    )

    # --- figure 1b : the representation settles as evidence accumulates

    drift = np.abs(
        np.diff(
            state,
            axis=1
        )
    ).mean(
        axis=(0, 2)
    )

    plt.figure(
        figsize=(9, 5.6)
    )

    plt.plot(
        t[1:],
        drift,
        color=BLUE
    )

    plt.xlabel(
        "Time within trial (s)"
    )

    plt.ylabel(
        "mean |change| per update"
    )

    plt.title(
        "The representation settles as evidence accumulates"
    )

    save_figure(
        "Results/S2_fig02_state_settling.png"
    )

    print(
        "\nStudy 2, Module 1 Completed Successfully."
    )