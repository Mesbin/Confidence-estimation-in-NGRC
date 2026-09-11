"""

Module 1 : Module_1_eeg_data.py

Loads the BCI Competition IV-2b motor-imagery dataset and applies the two
preprocessing steps that precede the NGRC reservoir:

    1. Band-pass 8-30 Hz   (mu + beta sensorimotor rhythms, zero-phase)
    2. Euclidean Alignment (per-subject covariance whitening, label-free)

Equivalent role to Module 1 in the Lorenz-63 pipeline: it produces the raw
state time series that the reservoir is built from.

"""

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch


# --------------------------------------------------------------------------
# Publication figure style, used by every module (imported from here).
# 400 dpi, base font 15 pt, line width 2.8, no grid lines, no box.
# --------------------------------------------------------------------------
def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 400,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.titlesize": 16, "axes.labelsize": 15,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.6, "axes.labelpad": 8, "axes.titlepad": 12,
        "lines.linewidth": 2.8, "lines.markersize": 8, "lines.markeredgewidth": 0,
        "patch.linewidth": 0,
        "xtick.major.width": 1.6, "ytick.major.width": 1.6,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "errorbar.capsize": 4,
    })


def save_figure(filename):
    """Save the current figure at 400 dpi, display it, and report the path."""
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"Saved {filename}")



class EEGData:

    FS = 250.0                 # sampling rate (Hz)
    BAND = (8, 30)             # mu + beta
    MI_START, MI_END = 125, 1000     # motor-imagery window: 0.5 s - 4.0 s

    # ---- THE CHANGE THAT DEFINES THIS STUDY ------------------------------
    CHANNELS = ("C3", "Cz", "C4")    # all three channels KEPT, Cz included
    KEEP_CHANNELS = [0, 1, 2]
    DROP_SUBJECT = 4                 # S4 decodes ~89%, far above the rest, and
                                     # skews every aggregate in the study
    # ----------------------------------------------------------------------

    def __init__(self, band=(8, 30)):
        self.band = band
        self.X = None
        self.y = None
        self.subjects = None

    # ==========================================================
    def load(self, folder="Data"):
        self.X = np.load(f"{folder}/X_epochs.npy").astype(np.float32) * 1e6   # -> microvolts
        self.y = np.load(f"{folder}/y_labels.npy")                            # 0 = LH, 1 = RH
        df = pd.read_csv(f"{folder}/epoch_labels.csv")
        self.subjects = df["Subject"].values

        n_before, s_before = self.X.shape[0], len(np.unique(self.subjects))
        self.X = self.X[:, self.KEEP_CHANNELS, :]          # drop Cz
        keep = self.subjects != self.DROP_SUBJECT          # drop Subject 4
        self.X, self.y, self.subjects = self.X[keep], self.y[keep], self.subjects[keep]

        print("Dataset Loaded.")
        print(f"  Cz KEPT      : {self.X.shape[1]} channels "
              f"({', '.join(self.CHANNELS)})")
        print(f"  S{self.DROP_SUBJECT} REMOVED   : {s_before} -> "
              f"{len(np.unique(self.subjects))} subjects, "
              f"{n_before} -> {self.X.shape[0]} trials")
        print("  Trials      :", self.X.shape[0])
        print("  Channels    :", self.X.shape[1], f"({', '.join(self.CHANNELS)})")
        print("  Samples     :", self.X.shape[2], f"({self.X.shape[2]/self.FS:.1f} s at {self.FS:.0f} Hz)")
        print("  Class counts:", f"LH={np.sum(self.y==0)}  RH={np.sum(self.y==1)}")
        print("  Subjects    :", len(np.unique(self.subjects)))
        return self.X, self.y, self.subjects

    # ==========================================================
    def bandpass(self, X):
        """Zero-phase Butterworth. Zero-phase matters: the NGRC delay embedding
        depends on preserved temporal relationships between samples."""
        sos = butter(4, list(self.band), btype="band", fs=self.FS, output="sos")#Create a stable 4th-order Butterworth filter that keeps 8–30 Hz for EEG sampled at 250 Hz.
        Xf = sosfiltfilt(sos, X, axis=-1).astype(np.float32)#shape/timing of the EEG events is preserved much better.
        Xf /= np.sqrt((Xf ** 2).mean())#EEG into a more manageable numerical scale for later processing.
        print(f"Band-pass applied : {self.band[0]}-{self.band[1]} Hz (4th order, zero-phase)")
        return Xf

    # ==========================================================
    def euclidean_align(self, X, subjects):
        """He & Wu (2020). Per subject: R = mean trial covariance, X <- R^-1/2 X.
        Uses NO labels, so it is legitimate on an unseen subject."""
        Xa = np.empty_like(X)
        covs = []
        for s in np.unique(subjects):
            m = subjects == s
            Xs = X[m].astype(np.float64)
            R = np.einsum("nct,ndt->cd", Xs, Xs) / (Xs.shape[0] * Xs.shape[2])
            w, V = np.linalg.eigh(R)
            Rm12 = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
            Xa[m] = np.einsum("cd,ndt->nct", Rm12, Xs).astype(X.dtype)
            covs.append(R)
        print("Euclidean Alignment applied to", len(covs), "subjects")
        return Xa, np.stack(covs)

    # ==========================================================
    def dispersion(self, X, subjects):
        """How far each subject's mean covariance sits from the group centre."""
        cov = np.stack([
            (np.einsum("nct,ndt->cd", X[subjects == s].astype(np.float64),
                       X[subjects == s].astype(np.float64))
             / ((subjects == s).sum() * X.shape[2]))
            for s in np.unique(subjects)])
        grand = cov.mean(0)
        d = np.array([np.linalg.norm(c - grand, "fro") / np.linalg.norm(grand, "fro") for c in cov])
        return cov, d

    # ==========================================================
    def save(self, Xa, folder="Data"):
        np.save(f"{folder}/eeg_preprocessed.npy", Xa.astype(np.float32))
        np.save(f"{folder}/eeg_labels.npy", self.y)
        np.save(f"{folder}/eeg_subjects.npy", self.subjects)
        print(f"Preprocessed data saved to {folder}/eeg_preprocessed.npy")

    # ==========================================================
    def plot_spectrum(self, X, filename="Results/eeg_spectrum.png"):
        f, P = welch(X, fs=self.FS, nperseg=256, axis=-1)
        Pm = P.mean(axis=0)
        plt.figure(figsize=(11, 5.5))
        for ci, nm in enumerate(self.CHANNELS):
            plt.semilogy(f, Pm[ci], label=nm, linewidth=1.2)
        plt.axvspan(self.band[0], self.band[1], color="seagreen", alpha=0.15,
                    label=f"kept band {self.band[0]}-{self.band[1]} Hz")
        plt.xlim(0, 60)
        plt.xlabel("Frequency (Hz)"); plt.ylabel("Power spectral density")
        plt.title("Raw EEG Spectrum — mu peak near 10 Hz motivates the 8-30 Hz band")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        print(f"Saved {filename}")

    # ==========================================================
    def plot_alignment(self, disp_before, disp_after, cov_before, cov_after):
        """Three separate figures, one per panel."""
        subs = np.unique(self.subjects)

        plt.figure(figsize=(9, 5.6))
        w = 0.38; xs = np.arange(len(subs))
        plt.bar(xs - w/2, disp_before, w, label="band-pass only", color="lightgray")
        plt.bar(xs + w/2, disp_after, w, label="+ Euclidean Alignment", color="steelblue")
        plt.xticks(xs, subs)
        plt.xlabel("Subject"); plt.ylabel("Distance to group mean covariance")
        plt.title("Between-Subject Dispersion\n(lower = better aligned)")
        plt.legend()
        save_figure("Results/eeg_alignment_dispersion.png")

        for cv, ttl, fn in [(cov_before, "Before alignment", "before"),
                            (cov_after, "After alignment", "after")]:
            plt.figure(figsize=(8, 5.6))
            im = plt.imshow(cv.reshape(len(cv), -1), aspect="auto", cmap="coolwarm")
            plt.colorbar(im, fraction=0.046)
            plt.title(f"Mean Covariance per Subject\n{ttl}")
            plt.xlabel(f"Covariance entry ({len(self.CHANNELS)}x{len(self.CHANNELS)} flattened)"); plt.ylabel("Subject")
            plt.yticks(range(len(subs)), subs)
            save_figure(f"Results/eeg_alignment_{fn}.png")


if __name__ == "__main__":
    apply_style()
    eeg = EEGData(band=(8, 30))
    X, y, subjects = eeg.load()

    eeg.plot_spectrum(X)

    Xf = eeg.bandpass(X)

    cov_b, disp_b = eeg.dispersion(Xf, subjects)
    Xa, cov_a_raw = eeg.euclidean_align(Xf, subjects)
    cov_a, disp_a = eeg.dispersion(Xa, subjects)

    print(f"\nCovariance dispersion : {disp_b.mean():.4f} -> {disp_a.mean():.4f}")
    eeg.plot_alignment(disp_b, disp_a, cov_b, cov_a)

    pd.DataFrame({"subject": np.unique(subjects),
                  "dispersion_before": disp_b,
                  "dispersion_after": disp_a}).to_csv("Results/preprocessing_summary.csv",
                                                      index=False)
    print("Saved Results/preprocessing_summary.csv")

    eeg.save(Xa)
    print("\nModule 1 Completed Successfully.")
