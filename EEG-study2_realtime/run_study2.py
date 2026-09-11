"""
=====================================================================
run_study2.py

Study 2 : Real-Time Confidence-Aware NGRC for Early Motor-Imagery
          Decision Making

Runs Module 1 -> Module 5. Each module also runs standalone, in order,
since every stage saves its output to Data/.
=====================================================================
"""
import os, subprocess, sys, time

os.makedirs("Data", exist_ok=True); os.makedirs("Results", exist_ok=True)

MODULES = [
    ("S2_Module_1_realtime_reservoir.py", "Causal running-mean NGRC state X_bar(t)"),
    ("S2_Module_2_realtime_readout.py",   "Readout at every time point, o_LH(t), C_ref(t)"),
    ("S2_Module_3_confidence_over_time.py", "Confidence estimator C_hat(t), leakage-free"),
    ("S2_Module_4_early_decision.py",     "Threshold + T_min chosen nested, decision rule"),
    ("S2_Module_5_results.py",            "Six result figures and the summary"),
]

started = time.time(); timings = []
for i, (script, desc) in enumerate(MODULES, 1):
    print("\n" + "=" * 70)
    print(f"RUNNING [{i}/{len(MODULES)}]  {script}")
    print(f"          {desc}")
    print("=" * 70)
    t0 = time.time()
    if subprocess.run([sys.executable, script]).returncode != 0:
        print(f"\n{script} FAILED - stopping."); sys.exit(1)
    timings.append((script, time.time() - t0))

print("\n" + "=" * 70)
print(f"STUDY 2 COMPLETE in {time.time()-started:.0f} s")
print("=" * 70)
for s, sec in timings:
    print(f"  {sec:6.1f} s  {s}")
print("\n  Figures (400 dpi) and tables : Results/")
