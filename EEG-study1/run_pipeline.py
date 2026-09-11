"""
=====================================================================
run_pipeline.py

Runs Module 1 -> Module 11 end to end on the BCI EEG data.

THIS STUDY: all three channels (C3, Cz, C4) with Subject 4 removed. The structure,
the modules and the figures are identical to the original study; only the
channel set and the subject set differ, plus the correlation graphs now retain
the SIGN so that uncertainty-encoding variables are visible alongside
confidence-encoding ones.

Each module also runs standalone, in order, e.g.
    python Module_5_confidence_analysis.py

Figures are displayed as each module runs and are also saved to Results/.
=====================================================================
"""

import os
import subprocess
import sys
import time

os.makedirs("Data", exist_ok=True)
os.makedirs("Results", exist_ok=True)

modules = [
    ("Module_1_eeg_data.py", "EEG loading + preprocessing (band-pass, Euclidean Alignment)"),
    ("Module_2_feature_construction.py", "NGRC reservoir construction"),
    ("Module_3_classification_readout.py", "LH/RH classification readout"),
    ("Module_4_baseline_experiment.py", "Baseline decoding on unseen subjects"),
    ("Module_5_confidence_analysis.py", "Confidence target + correlation analysis"),
    ("Module_6_feature_selection.py", "Confidence-variable selection"),
    ("Module_7_confidence_estimator.py", "Confidence estimator (quadratic readout)"),
    ("Module_8_final_validation.py", "Final validation + LH/RH agnosticism"),
    ("Module_9_preprocessing_ablation.py", "Preprocessing ablation (is it necessary?)"),
    ("Module_10_variable_activity.py", "Selected-variable activity and confidence computation"),
    ("Module_11_study_comparison.py", "Comparison with the original 9-subject study"),
]

t0 = time.time()
for i, (m, desc) in enumerate(modules, 1):
    print("\n" + "=" * 70)
    print(f"RUNNING [{i}/{len(modules)}] {m}")
    print(f"         {desc}")
    print("=" * 70)
    result = subprocess.run([sys.executable, m])
    if result.returncode != 0:
        print(f"\n!! {m} FAILED - stopping pipeline !!")
        sys.exit(1)

print("\n" + "=" * 70)
print(f"FULL PIPELINE COMPLETE (Modules 1-11) in {time.time()-t0:.0f} s")
print("Figures and tables are in Results/ ; arrays are in Data/")
print("=" * 70)
