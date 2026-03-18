"""
oxrl.sweep -- Experiment sweep infrastructure for NeurIPS 2026.

Generates, launches, and aggregates large-scale experiment grids across
(algorithm x model x task x seed) combinations.

Modules:
    sweep:    Generate the experiment grid and config files.
    launcher: Launch experiments sequentially or via SLURM.
    results:  Aggregate per-run JSON results into CSV tables and LaTeX.
"""
