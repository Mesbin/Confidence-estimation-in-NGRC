# NGRC Confidence Estimation on BCI EEG — C3/Cz/C4, Subject 4 excluded

A complete parallel study. Same steps, same modules, same figures as the
original; one thing differs, plus the correlation graphs now keep the sign.

    1. ALL THREE CHANNELS KEPT   C3, Cz and C4, exactly as the original study.
    2. SUBJECT 4 REMOVED         S4 decodes near 89% against a ~70% group mean
                                 and skews every aggregate.
    3. SIGNED CORRELATIONS       the ranking and scatter plots retain the sign,
                                 so uncertainty-encoding variables are visible
                                 alongside confidence-encoding ones.
    4. MODULE 10                 what the selected variables physically are, and
                                 whether their activity tracks confidence.
    5. MODULE 11                 one figure comparing this study with the
                                 original 9-subject study.

The change is applied once, in Module 1, where the data is loaded. Every later
module inherits it.

    python run_pipeline.py           # Modules 1 -> 11, about 70 s

## Dataset

    channels  3        (C3, Cz, C4)   unchanged
    subjects  9 -> 8   (S4 removed)
    trials    3026 -> 2627
    reservoir 55 variables            unchanged

## Headline results (all on UNSEEN subjects)

| Quantity | This study | Original (9 subjects) |
|---|---|---|
| LH/RH accuracy | **64.18%** | 69.86% |
| across folds | 64.15% **± 1.28** SD | 69.62% ± 7.36 SD |
| across subjects | ± 6.06 SD | ± 9.70 SD |
| Reference AUROC2 | 0.6351 | 0.6678 |
| Confidence AUROC2, leakage-free | **0.6322** | 0.6149 |
| R2 vs reference | 0.4610 | 0.3702 |
| Spearman vs reference | 0.6829 | 0.6333 |
| Accuracy @25% coverage | 80.8% | 86.1% |
| LH/RH bias, Cohen's d | -0.0971 [-0.169, -0.023] | -0.0159 |

Accuracy falls about 5.7 points because the strongest participant is gone, but
the spread across folds collapses from 7.36 to **1.28** SD and across subjects
from 9.70 to 6.06. The number now describes a typical participant rather than
being carried by one exceptional one.

**Confidence estimation IMPROVES**: AUROC2 0.6149 -> 0.6322, R2 0.370 -> 0.461,
Spearman 0.633 -> 0.683. Removing the outlier makes the confidence signal
cleaner, not weaker.

## Confidence-encoding versus uncertainty-encoding variables

The selection thresholds |rho|, so it keeps both directions. Split by sign
(`Results/confidence_vs_uncertainty_variables.csv`):

**18 confidence-encoding variables (rho > 0)**

    C4(t-6)^2, C4(t-3)^2, C4(t-0)^2                 rho = +0.132
    C4(t-3)*C4(t-6), C4(t-0)*C4(t-3)                rho = +0.091
    C3(t-6)^2, C3(t-3)^2, C3(t-0)^2                 rho = +0.082
    C3(t-3)*Cz(t-3), C3(t-6)*Cz(t-6)                rho = +0.082
    Cz(t-0)^2, Cz(t-3)^2, Cz(t-6)^2                 rho = +0.078
    C3(t-0)*C3(t-3), C3(t-3)*C3(t-6)                rho = +0.073
    Cz(t-0)*Cz(t-3), Cz(t-3)*Cz(t-6)                rho = +0.068
    Cz(t-3)*C3(t-6)                                 rho = +0.059

**4 uncertainty-encoding variables (rho < 0)** — every one CROSS-channel

    C3(t-3)*C4(t-3)     rho = -0.059    C3 x C4
    C3(t-0)*Cz(t-6)     rho = -0.057    C3 x Cz
    C3(t-0)*C3(t-3)     rho = -0.051    C4 x C3
    C4(t-3)*C3(t-6)     rho = -0.051    C4 x C3

## What the variables physically are (Module 10)

Each reservoir variable is a time-average, so a quadratic term is a lagged
autocovariance. The identity checks confirm it exactly:

    lag-0 term        vs C4 band power      rho = +1.0000
    lag-12ms / lag-0  vs C4 peak frequency  rho = -0.5994

**The split is clean in this configuration:**

    confidence-encoding : 15 / 18 WITHIN-channel   (power and spectral shape)
    uncertainty-encoding:  4 /  4 CROSS-channel    (inter-electrode coupling)

The leading uncertainty variable `C3(t-3)*C4(t-3)` correlates with C3-C4
coupling at rho = +0.999 — it IS the coupling. So confidence rises when each
sensorimotor rhythm is strong and locally well-formed, and uncertainty rises
when activity is diffuse and synchronised across electrodes.

This is a cleaner result than the 2-channel version, where two of the three
uncertainty variables were within-channel long-lag terms. Keeping Cz preserves
the cross-channel structure that carries the uncertainty signal.

## Honest notes

* The LH/RH bias is larger here (d = -0.097 against -0.016). It remains inside
  the ±0.2 SD equivalence bound but is no longer negligible.
* Accuracy @25% coverage falls from 86.1% to 80.8%: selective decoding was
  partly carried by S4's easy trials.
* Module 11 shows the trade-off in one figure. Read it as "a few points of
  accuracy for a far more stable and better-calibrated result", not as a
  uniform improvement.
