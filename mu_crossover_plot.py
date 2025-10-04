import pandas as pd
import matplotlib.pyplot as plt

# --- Load data ---
df = pd.read_csv("mu_phase_data.csv")

# --- Set up plot ---
plt.figure(figsize=(8, 6))
betas = sorted(df['beta'].unique())

# --- Plot log10_ratio vs slope for each beta ---
for b in betas:
    sub = df[df['beta'] == b]
    plt.plot(sub['slope'], sub['log10_ratio'], label=f"β={b:.2f}")

# --- Mark the event horizon (where log10_ratio=0) ---
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Crossover Behavior (μ Domain)")
plt.xlabel("slope (approach → horizon → acceleration)")
plt.ylabel("log₁₀(weight ratio w_fast / w_slow)")
plt.legend(title="β")
plt.grid(True, alpha=0.3)

# --- Save and show non-blocking ---
plt.tight_layout()
plt.savefig("mu_crossover_plot.png", dpi=300)
plt.show(block=False)
print("\n✅ Saved plot as mu_crossover_plot.png\n")

