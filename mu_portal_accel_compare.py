import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === MU Portal Crossover Visualization ===
print("=== Generating MU portal crossover plot ===")

# Load your CSV
df = pd.read_csv("mu_phase_data.csv")

# Ensure numeric and sorted
df = df.sort_values(["beta", "slope"])

# Function to find portal entry (ΔQ crosses 0)
def find_portal_slope(sub):
    dq = sub["DeltaQ"].values
    slopes = sub["slope"].values
    for i in range(len(dq) - 1):
        if dq[i] * dq[i+1] < 0:  # sign change
            # Linear interpolation between points
            frac = abs(dq[i]) / (abs(dq[i]) + abs(dq[i+1]))
            return slopes[i] + frac * (slopes[i+1] - slopes[i])
    return np.nan

# Get unique β and portal slopes
portal_points = []
for beta in sorted(df["beta"].unique()):
    sub = df[df["beta"] == beta]
    slope0 = find_portal_slope(sub)
    portal_points.append((beta, slope0))

portal_df = pd.DataFrame(portal_points, columns=["beta", "slope_portal"])
print("\n=== Portal entry points (ΔQ ≈ 0) ===")
print(portal_df)

# === Plot: log10(ratio) vs slope for each beta ===
plt.figure(figsize=(10,6))
for beta, sub in df.groupby("beta"):
    plt.plot(sub["slope"], sub["log10_ratio"], label=f"β={beta}")
    slope0 = portal_df.loc[portal_df["beta"] == beta, "slope_portal"].values[0]
    if not np.isnan(slope0):
        plt.axvline(slope0, color='gray', linestyle='--', alpha=0.6)
        plt.text(slope0, sub["log10_ratio"].min(), f"{beta}", fontsize=8, ha='center', color='black')

plt.axhline(0, color='black', linewidth=0.8)
plt.title("MU Portal Entry (ΔQ ≈ 0) — log10(Fast/Slow) vs Slope")
plt.xlabel("Slope")
plt.ylabel("log10(Fast/Slow)")
plt.legend(title="β Levels")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mu_crossover_portal_plot.png", dpi=300)
plt.show(block=False)

print("\n✅ Saved plot as mu_crossover_portal_plot.png")

