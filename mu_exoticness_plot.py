import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("=== MU Exoticness Visualization ===")

# Load your file (adjust if filename differs)
df = pd.read_csv("mu_phase_data.csv")

# Normalize header names (strip spaces, unify symbols)
df.columns = [c.strip().replace("Δ", "Delta").replace("−", "-").lower() for c in df.columns]
print("\nDetected headers:", df.columns.tolist())

# Identify likely fast/slow weight columns
possible_fast = [c for c in df.columns if "fast" in c]
possible_slow = [c for c in df.columns if "slow" in c]
if not possible_fast or not possible_slow:
    raise ValueError("Could not identify 'fast' and 'slow' columns automatically.")

fast_col = possible_fast[0]
slow_col = possible_slow[0]
print(f"\nUsing columns: fast = '{fast_col}', slow = '{slow_col}'")

# Compute exoticness ratio
df["ratio_fast_slow"] = df[fast_col] / df[slow_col]

# Aggregate exoticness by beta
agg = df.groupby("beta")["ratio_fast_slow"].mean().reset_index()
agg.rename(columns={"ratio_fast_slow": "exoticness"}, inplace=True)

print("\n--- Aggregated Exoticness ---")
print(agg)

# Fit log trend
z = np.polyfit(agg["beta"], np.log10(agg["exoticness"] + 1e-12), 1)
trend = z[0]
print(f"\nTrend slope (log-scale): {trend:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(agg["beta"], agg["exoticness"], "o-", lw=2, label="Mean Exoticness (fast/slow ratio)")
plt.xlabel("β (Beta)")
plt.ylabel("Exoticness Ratio (Fast/Slow)")
plt.title("MU Exoticness vs β — Dimensional Acceleration Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mu_exoticness_plot.png", dpi=300)
print("\n✅ Saved as mu_exoticness_plot.png")

if trend > 0:
    print("\n🌀 Interpretation: Exoticness increases with β — higher β promotes multi-dimensional unfolding.")
else:
    print("\n🌀 Interpretation: Exoticness decreases with β — higher β suppresses branching and fast-path expression.")

