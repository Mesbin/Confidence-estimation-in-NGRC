"""
Module 3 : Module_3_classification_readout.py

The Lorenz-63 pipeline trains a RIDGE readout to predict the next continuous
state. Here the task is CLASSIFICATION, so the readout produces two activities

    o_LH = p(LH | x)      o_RH = p(RH | x) = 1 - o_LH

and the DECISION READOUT SEPARATION is

    Delta = o_LH - o_RH   in [-1, 1] ,   decision = sign(Delta)

Delta is the bridge to the confidence definition in Module 5: when the two
readout activities are close, |Delta| is small and confidence is low.

VALIDATION PROTOCOL used from here on:
    subject-level 3-fold cross-validation. Each fold holds out 3 UNSEEN
    subjects; every subject is unseen exactly once. The readout never sees the
    held-out subjects during training.

"""

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from Module_1_eeg_data import apply_style, save_figure
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ClassificationReadout:

    def __init__(self, C=0.05, n_folds=3, seed=0):
        self.C = C
        self.n_folds = n_folds
        self.seed = seed
        self.coefs = None

    # ==========================================================
    def load(self, folder="Data"):
        R = np.load(f"{folder}/ngrc_reservoir.npy")
        y = np.load(f"{folder}/eeg_labels.npy")
        subjects = np.load(f"{folder}/eeg_subjects.npy")
        print("Reservoir Loaded. Shape :", R.shape)
        print("Labels Loaded    . Shape :", y.shape)
        return R, y, subjects

    # ==========================================================
    def subject_folds(self, subjects):
        ids = np.unique(subjects)
        folds = np.array_split(np.random.default_rng(self.seed).permutation(ids), self.n_folds)
        print("\nSubject folds (each held out once as UNSEEN):")
        for i, f in enumerate(folds):
            print(f"  fold {i} unseen subjects : {sorted(f.tolist())}")
        return folds

    # ==========================================================
    def train_and_apply(self, R, y, subjects, folds):
        """Train on the training subjects of each fold, apply to the unseen ones."""
        n = len(y)
        o_LH = np.full(n, np.nan)
        fold_id = np.full(n, -1)
        coefs = []

        for fi, unseen in enumerate(folds):
            tr = ~np.isin(subjects, unseen)
            te = np.isin(subjects, unseen)
            fold_id[te] = fi

            scaler = StandardScaler().fit(R[tr])
            clf = LogisticRegression(C=self.C, max_iter=3000)
            clf.fit(scaler.transform(R[tr]), y[tr])

            lh_col = list(clf.classes_).index(0)
            o_LH[te] = clf.predict_proba(scaler.transform(R[te]))[:, lh_col]
            coefs.append(clf.coef_[0])

            acc = ((o_LH[te] > 0.5).astype(int) == (y[te] == 0).astype(int)).mean()
            print(f"  fold {fi} : trained on {tr.sum()} trials, "
                  f"applied to {te.sum()} unseen trials, accuracy {acc:.4f}")

        self.coefs = np.stack(coefs)
        o_RH = 1.0 - o_LH
        delta = o_LH - o_RH

        correct = ((delta < 0).astype(int) == y).astype(int)
        fold_acc = np.array([correct[fold_id == f].mean() for f in range(len(folds))])
        print(f"\n  accuracy across folds : {fold_acc.mean()*100:.2f}% "
              f"+/- {fold_acc.std(ddof=1)*100:.2f} (SD)")
        return o_LH, o_RH, delta, fold_id

    # ==========================================================
    def save(self, o_LH, o_RH, delta, fold_id, folder="Data"):
        np.save(f"{folder}/readout_o_LH.npy", o_LH)
        np.save(f"{folder}/readout_o_RH.npy", o_RH)
        np.save(f"{folder}/readout_delta.npy", delta)
        np.save(f"{folder}/fold_id.npy", fold_id)
        np.save(f"{folder}/readout_weights.npy", self.coefs)
        print(f"\nReadout activities saved to {folder}/readout_delta.npy")

    # ==========================================================

if __name__ == "__main__":
    apply_style()
    model = ClassificationReadout(C=0.05, n_folds=3, seed=0)
    R, y, subjects = model.load()
    names = np.load("Data/feature_names.npy", allow_pickle=True)

    folds = model.subject_folds(subjects)
    print("\nTraining the classification readout :")
    o_LH, o_RH, delta, fold_id = model.train_and_apply(R, y, subjects, folds)

    model.save(o_LH, o_RH, delta, fold_id)
    print("\nModule 3 Completed Successfully.")
