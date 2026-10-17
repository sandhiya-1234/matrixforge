"""Plot GFLOP/s per kernel vs matrix size from results/benchmark.csv.

Usage:  python scripts/plot_results.py [path/to/benchmark.csv]
Runs on any machine (no GPU needed).  Requires: pandas, matplotlib.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

csv_path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/benchmark.csv")
df = pd.read_csv(csv_path)

# Only square shapes go on the line plot; odd shapes are correctness tests.
sq = df[(df.M == df.N) & (df.N == df.K)].copy()
gpu = sq.gpu.iloc[0]

fig, ax = plt.subplots(figsize=(8, 5))
for name, g in sq.groupby("kernel"):
    g = g.sort_values("N")
    ax.plot(g.N, g.gflops, marker="o", label=name)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Matrix size N (N x N x N, FP32)")
ax.set_ylabel("Throughput (GFLOP/s, log scale)")
ax.set_title(f"Matrix multiplication throughput - {gpu}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()

out = csv_path.parent / "gflops.png"
fig.savefig(out, dpi=150)
print(f"Saved {out}")

bad = df[df.correct == 0]
if len(bad):
    print("\nWARNING: incorrect results found:")
    print(bad[["M", "N", "K", "kernel", "max_rel_err"]].to_string(index=False))
