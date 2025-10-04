import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Test Overview ===
print("=== Maximization Universe: Full Portal Scan ===")
print("Sweeping beta=8→10, slope=-1.2→1.2")

# === Simulation setup ===
betas = np.arange(8.0, 10.25, 0.25)
slopes = np.arange(-1.2, 1.21, 0.02)

records = []

for beta in betas:
    for slope in slopes:
        Q_slow = 0.03125
        Q_fast = slope * 0.65  # fast path coupling proportional to slope
        deltaQ = Q_fast - Q_slow

        # model weights
        w_slow = np.exp(beta * Q_slow)
        w_fast = np.exp(beta * Q_fast)
        ratio = w_fast / w_slow
        log10_ratio = np.log10(ratio)

        # classification
        if abs(deltaQ) < 0.01:
            phase = "COEXISTENCE"
        elif ratio > 1:
            phase = "FAST path dominant"
        else:
            phase = "SLOW path dominant"

        records.append([beta, slope, Q_slow, Q_fast, deltaQ, w_slow, w_fast, ratio, log10_ratio, phase])

df = pd.DataFrame(records, columns=[
    "beta", "slope", "Q_slow", "Q_fast", "DeltaQ",
    "w_slow", "w_fast", "ratio_wfast_wslow", "log10_ratio", "phase"
])

# === Save to CSV ===
df.to_csv("mu_full_portal_scan.csv", index=False)
print("✅ Saved data to mu_full_portal_scan.csv")

# === Plotting ===
plt.figure(figsize=(10,6))
for beta in betas:
    sub = df[df["beta"] == beta]
    plt.plot(sub["slope"], sub["log10_ratio"], label=f"β={beta:.2f}")
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Slope")
plt.ylabel("log₁₀(w_fast / w_slow)")
plt.title("MU Portal Crossover: Fast vs Slow Path Dominance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mu_full_portal_plot.png")
print("✅ Saved crossover plot as mu_full_portal_plot.png")
plt.show(block=False)

