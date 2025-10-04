import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== MU Portal Exoticness Acceleration Analysis ===")

df = pd.read_csv("mu_full_portal_scan.csv")

# Compute slope of log10_ratio after the portal (slope > 0)
results = []
for beta in sorted(df["beta"].unique()):
    sub = df[(df["beta"] == beta) & (df["slope"] > 0)]
    if len(sub) > 3:
        # linear fit of log10_ratio vs slope to estimate "acceleration" strength
        coeffs = np.polyfit(sub["slope"], sub["log10_ratio"], 1)
        accel = coeffs[0]  # slope of fit = acceleration factor
        results.append([beta, accel])

accel_df = pd.DataFrame(results, columns=["beta", "acceleration"])
accel_df.to_csv("mu_portal_exoticness.csv", index=False)

# Plot exoticness vs beta
plt.figure(figsize=(8,5))
plt.plot(accel_df["beta"], accel_df["acceleration"], marker="o")
plt.xlabel("β (Coherence)")
plt.ylabel("Acceleration of log₁₀(w_fast/w_slow) after portal")
plt.title("MU Exoticness Growth Rate vs β")
plt.grid(True)
plt.tight_layout()
plt.savefig("mu_portal_exoticness.png")
plt.show(block=False)

print("✅ Saved results as mu_portal_exoticness.csv and mu_portal_exoticness.png")

