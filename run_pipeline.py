"""
=====================================================================
run_pipeline.py

Runs Module 1 -> Module 8 end to end.
=====================================================================
"""

import os
import subprocess
import sys

os.makedirs("Data", exist_ok=True)
os.makedirs("Results", exist_ok=True)

modules = [
    "Module_1_lorenz63.py",
    "Module_2_feature_construction.py",
    "Module_3_ridge_readout.py",
    "Module_4_baseline_experiment.py",
    "Module_5_confidence_analysis.py",
    "Module_6_feature_selection.py",
    "Module_7_confidence_estimator.py",
    "Module_8_final_validation.py",
]

for m in modules:
    print("\n" + "=" * 70)
    print(f"RUNNING {m}")
    print("=" * 70)
    result = subprocess.run([sys.executable, m])
    if result.returncode != 0:
        print(f"\n!! {m} FAILED - stopping pipeline !!")
        sys.exit(1)

print("\n" + "=" * 70)
print("FULL PIPELINE COMPLETE (Modules 1-8)")
print("=" * 70)
