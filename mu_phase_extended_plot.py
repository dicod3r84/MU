import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================================================================
# Plot: MU Domain Crossover (Multi-β)
# ================================================================
print("=== Generating MU crossover plot ===")

# Load data
df = pd.read_csv("mu_phase_extended.csv")

# Prepare figure
plt.figure(figsize=(9, 6))
betas = sorted(df['beta'].unique())

for beta in betas:
    sub = df[df['beta'] == beta]
    plt.plot(sub['slope'], sub['log10_ratio'], label=f"β={beta:.2f}")

# Draw zero line (the event-horizon)
plt.axhline(0, color='k', linestyle='--', linewidth=1.2, alpha=0.8)
plt.title("MU Phase Crossover — log₁₀(w_fast / w_slow) vs Slope")
plt.xlabel("Slope (∂Q/∂T)")
plt.ylabel("log₁₀ Ratio (Fast/Slow)")
plt.legend(title="β values", fontsize=8)
plt.grid(True, alpha=0.25)

# Save + show
plt.tight_layout()
plt.savefig("mu_crossover_extended_plot.png", dpi=300)
plt.show()

print("Saved as mu_crossover_extended_plot.png")

