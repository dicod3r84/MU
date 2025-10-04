"""
MU Crossover and Acceleration Visualization
-------------------------------------------
Purpose:
  To visualize the dominance crossover between slow and fast paths
  across slope values for multiple β (beta) levels, and to compute
  the local acceleration (2nd derivative) of dominance ratio.

Expected Inputs:
  mu_phase_beta_dq.csv  ← data file with columns:
     beta, slope, DeltaQ, ratio_wfast_wslow, log10_ratio

Outputs:
  mu_crossover_plot.png     ← crossover of dominance ratio vs slope
  mu_crossover_accel.png    ← acceleration (curvature) map
  console summary with crossover slopes for each beta
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load Data ===
df = pd.read_csv("mu_phase_data.csv")

# === Compute Additional Metrics ===
df["ratio"] = df["ratio_wfast_wslow"]
df["log_ratio"] = np.log10(df["ratio"].replace(0, np.nan))
df["abs_dq"] = np.abs(df["DeltaQ"])

# === Identify Crossover Point ===
crossover_points = []
for beta, group in df.groupby("beta"):
    group_sorted = group.sort_values("slope")
    sign_change = np.sign(group_sorted["log_ratio"]).diff().fillna(0) != 0
    crossing = group_sorted[sign_change]
    if not crossing.empty:
        slope_cross = crossing["slope"].iloc[0]
        crossover_points.append((beta, slope_cross))

# Print results summary
print("\n--- MU Crossover Summary ---")
for beta, slope in crossover_points:
    print(f"β = {beta:.2f} → crossover near slope = {slope:.3f}")
print("-----------------------------\n")

# === Plot 1: Ratio vs Slope (Crossover Plot) ===
plt.figure(figsize=(10, 6))
for beta, group in df.groupby("beta"):
    plt.plot(group["slope"], group["log_ratio"], label=f"β={beta:.2f}")
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel("Slope")
plt.ylabel("log₁₀(ratio w_fast/w_slow)")
plt.title("Dominance Crossover between Fast and Slow Paths")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mu_crossover_plot.png", dpi=300)
plt.show()

# === Plot 2: Acceleration (Curvature) Map ===
pivot = df.pivot_table(index="slope", columns="beta", values="log_ratio")
accel = np.gradient(np.gradient(pivot, axis=0), axis=0)

plt.figure(figsize=(10, 6))
plt.imshow(accel, aspect='auto', origin='lower',
           extent=[df["beta"].min(), df["beta"].max(),
                   df["slope"].min(), df["slope"].max()],
           cmap="coolwarm")
plt.colorbar(label="Acceleration of Dominance (2nd derivative)")
plt.xlabel("β (beta)")
plt.ylabel("Slope")
plt.title("Acceleration Map: Fast–Slow Dominance Transition")
plt.tight_layout()
plt.savefig("mu_crossover_accel.png", dpi=300)
plt.show()

print("✅ Plots saved as mu_crossover_plot.png and mu_crossover_accel.png")

