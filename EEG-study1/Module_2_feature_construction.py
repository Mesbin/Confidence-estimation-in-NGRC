"""
Module 2 : Module_2_feature_construction.py

Builds the NGRC reservoir from the preprocessed EEG, using exactly the
Gauthier et al. (2021) construction used in the Lorenz-63 pipeline:

    d(t) = [u(t), u(t-s), ..., u(t-(k-1)s)]     delay embedding
    X(t) = [1, d(t), d(t) (x) d(t)]             polynomial expansion

Difference from Lorenz-63: EEG is a set of independent TRIALS rather than one
continuous trajectory, so the reservoir state is time-averaged over the
motor-imagery window to give ONE feature vector per trial. The time-averaged
quadratic block equals the lagged covariance D D^T / T, which is what makes
this cheap to compute.

"""

import numpy as np
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure


class NGRCFeatureConstructor:

    MI_START, MI_END = 125, 1000       # motor-imagery window (0.5 s - 4.0 s)
    CHANNELS = ("C3", "Cz", "C4")

    def __init__(self, delay=3, skip=3):
        self.k = delay      # number of delayed copies
        self.s = skip       # spacing between them, in samples

    # ==========================================================
    def load_dataset(self, folder="Data"):
        X = np.load(f"{folder}/eeg_preprocessed.npy")
        print("Preprocessed EEG Loaded. Shape :", X.shape, "(trials, channels, samples)")
        return X

    # ==========================================================
    def delay_embedding(self, X):
        """(n_trials, n_ch, n_samples) -> (n_trials, k*n_ch, n_window)."""
        n, c, T = X.shape
        start = max(self.MI_START, (self.k - 1) * self.s)
        D = np.stack([X[:, :, start - j*self.s: self.MI_END - j*self.s]
                      for j in range(self.k)], axis=1)
        D = D.reshape(n, self.k * c, -1)
        print("Delay-Embedded Shape :", D.shape, f"(k={self.k}, skip={self.s})")
        return D

    # ==========================================================
    def linear_features(self, D):
        """Time-average of the linear block -> one value per delayed channel."""
        lin = D.mean(axis=-1)
        print("Linear Feature Shape :", lin.shape)
        return lin

    # ==========================================================
    def quadratic_features(self, D):
        """Time-average of all pairwise products = lagged covariance."""
        n, kc, T = D.shape
        Dd = D.astype(np.float64)
        q = np.einsum("nit,njt->nij", Dd, Dd) / T
        idx = list(combinations_with_replacement(range(kc), 2))
        i_arr = np.array([i for i, j in idx]); j_arr = np.array([j for i, j in idx])
        quad = q[:, i_arr, j_arr]
        print("Quadratic Feature Shape :", quad.shape)
        return quad, idx

    # ==========================================================
    def total_features(self, linear, quadratic):
        constant = np.ones((linear.shape[0], 1))
        reservoir = np.hstack((constant, linear, quadratic))
        print("Total Reservoir Shape :", reservoir.shape)
        return reservoir

    # ==========================================================
    def signed_log(self, R):
        """Compress the dynamic range of the quadratic block, keep the sign.
        Without this the quadratic terms dominate the linear ones by orders of
        magnitude and the ridge readout is badly conditioned."""
        scale = np.median(np.abs(R), axis=0, keepdims=True) + 1e-12
        out = np.sign(R) * np.log1p(np.abs(R) / scale)
        out[:, 0] = 1.0                      # keep the bias column as a constant
        print("Applied signed-log compression to the reservoir")
        return out

    # ==========================================================
    def feature_names(self, quad_indices):
        base = [f"{c}(t-{j*self.s})" for j in range(self.k) for c in self.CHANNELS]
        names = ["const"] + base
        names += [f"{base[i]}^2" if i == j else f"{base[i]}*{base[j]}" for i, j in quad_indices]
        return names

    # ==========================================================
    def save(self, reservoir, names, folder="Data"):
        np.save(f"{folder}/ngrc_reservoir.npy", reservoir)
        np.save(f"{folder}/feature_names.npy", np.array(names, dtype=object),
                allow_pickle=True)
        print(f"Reservoir Saved to {folder}/ngrc_reservoir.npy")

    # ==========================================================

if __name__ == "__main__":
    apply_style()
    fc = NGRCFeatureConstructor(delay=3, skip=3)
    X = fc.load_dataset()

    D = fc.delay_embedding(X)
    linear = fc.linear_features(D)
    quadratic, quad_idx = fc.quadratic_features(D)
    reservoir = fc.total_features(linear, quadratic)
    reservoir = fc.signed_log(reservoir)

    names = fc.feature_names(quad_idx)
    print("Example feature names :", names[:4], "...", names[-2:])

    fc.save(reservoir, names)
    print("\nModule 2 Completed Successfully.")
