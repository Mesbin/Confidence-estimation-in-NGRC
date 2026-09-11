"""
=====================================================================
Study 2, Module 4 : S2_Module_4_early_decision.py

THE DECISION RULE

For each trial, find the earliest reliable time at which confidence is
sufficient:

    commit at the first t where, over the preceding window of D seconds,
        C_hat has been continuously >= C_th
        and the predicted class has not changed
    otherwise                                    ->  NO COMMAND

Three outcomes: LEFT / RIGHT / NO COMMAND. For a wheelchair, issuing no command
is preferable to issuing the wrong one.

WHY A SUSTAINED WINDOW RATHER THAN A SINGLE CROSSING

Committing at the first instant C_hat crosses the threshold does not work on
this data. Early in the trial the running mean is formed from few samples, so
the readout probabilities are noise-inflated and confidence is high for the
wrong reason -- C_ref(t) falls over the trial rather than rising (Module 2).
Measured consequences of the single-crossing rule:

    first crossing of 0.52 : 78% of trials commit, median 0.62 s, 66% accurate
                             (BELOW the 69.9% full-trial baseline)
    accuracy by crossing time : 0.5-0.8 s -> 63.5%,  1.2-1.8 s -> 80.7%
                             (crossing EARLIER predicts a WORSE decision)

A correct trial also dwells above the threshold for 42% of the trial against
30% for a wrong one, so the dwell requirement uses a real signal rather than
merely delaying the answer.

PARAMETER SELECTION IS NESTED

D and C_th are chosen inside the training subjects, on an inner
leave-one-subject-out split, and frozen before the unseen subjects are touched.
The criterion follows the aim of the study: among operating points meeting a
stated reliability target, take the one that decides EARLIEST.

Output : Results/S2_operating_point.csv, Results/S2_threshold_sweep.csv,
         Results/S2_per_trial_decisions.csv, Results/S2_inner_selection.csv
=====================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from S2_Module_1_realtime_reservoir import subject_folds
from S2_Module_2_realtime_readout import fit_readout
from S2_Module_3_confidence_over_time import (quad_expand, select_confidence_variables,
                                              SUBSAMPLE, RIDGE_ALPHA)

TARGET_ACCURACY = 0.75      # reliability a command must meet, set by the application
MIN_COVERAGE = 0.20         # below this an operating point is not useful
DWELLS = np.array([0.20, 0.40, 0.60, 0.80, 1.00])          # candidate D, seconds
THRESHOLDS = np.round(np.arange(0.35, 0.76, 0.05), 2)      # candidate C_th


def sustained_commit(conf, pred, t, thr, dwell):
    """First index at which conf has been >= thr for `dwell` seconds with the
    predicted class unchanged. Returns (index, decision); index -1 = no command."""
    n, T = conf.shape
    step = t[1] - t[0]
    k = max(1, int(round(dwell / step)))
    ok = conf >= thr
    run_ok = np.ones((n, T), bool)
    run_cls = np.ones((n, T), bool)
    for j in range(k):
        run_ok[:, k - 1:] &= ok[:, k - 1 - j: T - j]
        if j > 0:
            run_cls[:, k - 1:] &= (pred[:, k - 1:] == pred[:, k - 1 - j: T - j])
    fire = run_ok & run_cls
    fire[:, :k - 1] = False
    idx = np.where(fire.any(1), fire.argmax(1), -1)
    dec = np.where(idx >= 0, pred[np.arange(n), np.clip(idx, 0, None)], -1)
    return idx, dec


def evaluate(idx, dec, y, t, mask):
    committed = mask & (idx >= 0)
    n_mask = int(mask.sum())
    if committed.sum() == 0:
        return dict(coverage=0.0, accuracy=np.nan, wrong_rate=0.0, rejection=1.0,
                    median_T=np.nan, mean_T=np.nan, time_saved=np.nan)
    ok = dec[committed] == y[committed]
    tt = t[idx[committed]]
    return dict(coverage=committed.sum() / n_mask, accuracy=float(ok.mean()),
                wrong_rate=(~ok).sum() / n_mask,
                rejection=1 - committed.sum() / n_mask,
                median_T=float(np.median(tt)), mean_T=float(tt.mean()),
                time_saved=float(t[-1] - np.median(tt)))


def inner_confidence(state, y, subjects, train_mask):
    """C_hat(t) and predictions for the TRAINING subjects, by leave-one-subject-
    out inside them, so nothing derived from the unseen subjects is used."""
    n, nT, P = state.shape
    chat = np.full((n, nT), np.nan, np.float32)
    pred = np.zeros((n, nT), int)
    for hs in np.unique(subjects[train_mask]):
        held = subjects == hs
        fit = train_mask & ~held
        pred[held] = ((2 * fit_readout(state, y, fit, held) - 1) < 0).astype(int)
        cref = np.full((n, nT), np.nan, np.float32)
        for hs2 in np.unique(subjects[fit]):
            h2 = subjects == hs2
            cref[h2] = np.abs(2 * fit_readout(state, y, fit & ~h2, h2) - 1)
        Zf = state[fit].reshape(-1, P); cf = cref[fit].reshape(-1)
        sel = select_confidence_variables(Zf[::SUBSAMPLE], cf[::SUBSAMPLE])
        s1 = StandardScaler().fit(Zf[:, sel])
        Q = quad_expand(s1.transform(Zf[:, sel]))
        s2 = StandardScaler().fit(Q)
        g = Ridge(alpha=RIDGE_ALPHA).fit(s2.transform(Q), cf)
        Zh = state[held].reshape(-1, P)
        chat[held] = g.predict(s2.transform(quad_expand(s1.transform(Zh[:, sel])))).reshape(
            int(held.sum()), nT)
    return chat, pred


def choose_operating_point(chat_in, pred_in, y, t, train_mask):
    """Earliest operating point meeting the reliability target, on training data."""
    best, rows = None, []
    for D in DWELLS:
        for thr in THRESHOLDS:
            i, d = sustained_commit(chat_in, pred_in, t, thr, D)
            m = evaluate(i, d, y, t, train_mask)
            rows.append(dict(dwell=D, threshold=thr, **m))
            if (m["coverage"] >= MIN_COVERAGE and not np.isnan(m["accuracy"])
                    and m["accuracy"] >= TARGET_ACCURACY):
                key = (m["median_T"], -m["coverage"])
                if best is None or key < best[0]:
                    best = (key, D, thr, m)
    if best is None:                       # target unreachable: take the most accurate
        r = max([r for r in rows if r["coverage"] >= MIN_COVERAGE],
                key=lambda r: (r["accuracy"] if not np.isnan(r["accuracy"]) else -1))
        best = ((r["median_T"], -r["coverage"]), r["dwell"], r["threshold"], r)
    return best[1], best[2], best[3], pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 70)
    print("STUDY 2, MODULE 4 : sustained-crossing early decision")
    print("=" * 70)

    state = np.load("Data/rt_state.npy")
    t = np.load("Data/rt_time.npy")
    y = np.load("Data/labels.npy")
    subjects = np.load("Data/subjects.npy")
    chat = np.load("Data/rt_chat.npy")
    delta = np.load("Data/rt_delta.npy")
    correct = np.load("Data/rt_correct.npy")
    pred = (delta < 0).astype(int)
    n = len(y)

    print(f"  reliability target {TARGET_ACCURACY:.0%}, minimum coverage {MIN_COVERAGE:.0%}")
    print(f"  searching dwell D in {list(DWELLS)} s and C_th in "
          f"[{THRESHOLDS[0]:.2f}, {THRESHOLDS[-1]:.2f}], on training subjects only\n")

    idx_all = np.full(n, -1); dec_all = np.full(n, -1)
    chosen, inner_tables = {}, []
    for fold, unseen in enumerate(subject_folds(subjects)):
        te = np.isin(subjects, unseen); tr = ~te
        chat_in, pred_in = inner_confidence(state, y, subjects, tr)
        D, thr, m, tbl = choose_operating_point(chat_in, pred_in, y, t, tr)
        chosen[fold] = (D, thr)
        inner_tables.append(tbl.assign(fold=fold))
        i, d = sustained_commit(chat, pred, t, thr, D)
        idx_all[te] = i[te]; dec_all[te] = d[te]
        print(f"  fold {fold}: chosen on training subjects -> D = {D:.2f} s, "
              f"C_th = {thr:.2f}   (inner: {m['coverage']*100:.0f}% coverage, "
              f"{m['accuracy']*100:.0f}% accurate, median {m['median_T']:.2f} s)")
    pd.concat(inner_tables).to_csv("Results/S2_inner_selection.csv", index=False)

    res = evaluate(idx_all, dec_all, y, t, np.ones(n, bool))
    full_acc = float(correct[:, -1].mean())
    Td = np.where(idx_all >= 0, t[np.clip(idx_all, 0, None)], np.nan)

    print("\n" + "=" * 70)
    print("EARLY DECISION ON UNSEEN SUBJECTS (parameters frozen from training)")
    print("=" * 70)
    print(f"  coverage (commands issued) : {res['coverage']*100:.1f}%")
    print(f"  accepted-trial accuracy    : {res['accuracy']*100:.1f}%")
    print(f"  wrong-command rate         : {res['wrong_rate']*100:.1f}% of all trials")
    print(f"  no command                 : {res['rejection']*100:.1f}% of all trials")
    print(f"  median decision time       : {res['median_T']:.2f} s")
    print(f"  mean decision time         : {res['mean_T']:.2f} s")
    print(f"  spread of decision times   : SD {np.nanstd(Td):.2f} s, "
          f"IQR {np.nanpercentile(Td,25):.2f}-{np.nanpercentile(Td,75):.2f} s")
    print(f"  time saved vs full trial   : {res['time_saved']:.2f} s")
    print(f"\n  full-trial baseline        : 100.0% coverage, {full_acc*100:.1f}% accuracy, "
          f"{t[-1]:.2f} s, wrong-command rate {(1-full_acc)*100:.1f}%")

    D0 = chosen[0][0]
    rows = []
    for thr in THRESHOLDS:
        i, d = sustained_commit(chat, pred, t, thr, D0)
        rows.append(dict(dwell=D0, threshold=thr, **evaluate(i, d, y, t, np.ones(n, bool))))
    sweep = pd.DataFrame(rows)
    sweep.to_csv("Results/S2_threshold_sweep.csv", index=False)
    print(f"\n  Threshold sweep on unseen subjects at D = {D0:.2f} s:")
    print(sweep[["threshold", "coverage", "accuracy", "wrong_rate", "median_T"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    pd.DataFrame({"trial": np.arange(n), "subject": subjects,
                  "true_class": np.where(y == 0, "LH", "RH"),
                  "committed": idx_all >= 0,
                  "decision": np.where(idx_all >= 0,
                                       np.where(dec_all == 0, "LEFT", "RIGHT"), "NO COMMAND"),
                  "T_decision_s": Td,
                  "correct": np.where(idx_all >= 0, dec_all == y, np.nan),
                  "full_trial_correct": correct[:, -1]}).to_csv(
        "Results/S2_per_trial_decisions.csv", index=False)
    pd.DataFrame([{"dwell_fold0": chosen[0][0], "threshold_fold0": chosen[0][1],
                   "dwell_fold1": chosen[1][0], "threshold_fold1": chosen[1][1],
                   "dwell_fold2": chosen[2][0], "threshold_fold2": chosen[2][1],
                   **res, "full_trial_accuracy": full_acc,
                   "full_trial_wrong_rate": 1 - full_acc,
                   "trial_length_s": float(t[-1]),
                   "decision_time_sd": float(np.nanstd(Td))}]).to_csv(
        "Results/S2_operating_point.csv", index=False)
    np.save("Data/rt_decision_idx.npy", idx_all)
    np.save("Data/rt_decision.npy", dec_all)

    print("\nStudy 2, Module 4 Completed Successfully.")
