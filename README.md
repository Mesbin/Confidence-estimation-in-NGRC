# Lorenz-63 NGRC Confidence Pipeline

## Run

    python run_pipeline.py            # Modules 1 -> 8

## Figures

Eleven figures, one panel per file, all written at **400 dpi** with base font
15 pt, line width 2.8, no grid lines and no box. The style is defined once by
`apply_style()` in `Module_1_lorenz63.py` and imported by every other module.

| File | Module | Was |
|---|---|---|
| `lorenz_attractor.png` | 1 | as before |
| `prediction_vs_true_x.png` | 4 | `prediction_vs_true` panel x |
| `prediction_vs_true_y.png` | 4 | `prediction_vs_true` panel y |
| `prediction_vs_true_z.png` | 4 | `prediction_vs_true` panel z |
| `correlation_heatmap.png` | 5 | as before |
| `feature_ranking_barplot.png` | 5 | as before |
| `confidence_distribution.png` | 5 | as before |
| `selected_features_plot.png` | 6 | as before |
| `confidence_timeseries.png` | 7 | `confidence_comparison` panel 1 |
| `confidence_scatter.png` | 7 | `confidence_comparison` panel 2 |
| `sensitivity_plot.png` | 8 | as before |

Removed from the plotting code only (none appear in the report): Lorenz time
series, reservoir feature snapshot, readout weight heatmap, baseline error time
series, feature reduction pie chart, confidence definitions plot, ablation plot.
Every analysis and every CSV table is unchanged.

## Results (unchanged)

| Quantity | Value |
|---|---|
| Confidence estimator RMSE | 0.002629 |
| R2 | 0.99946 |
| Pearson r | 0.99974 |
| Selected features | 8 of 28 |
| Ablation, full reservoir | R2 = 1.0000 |
| Ablation, reduced reservoir | R2 = 0.9995 |
