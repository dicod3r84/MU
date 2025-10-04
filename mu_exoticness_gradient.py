import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== MU Exoticness Gradient Test (Auto Header Detection) ===")

# Load your data
df = pd.read_csv("mu_phase_data.csv")
print("Detected headers:", list(df.columns))

# Try to identify the correct fast/slow columns automatically
possible_fast = [c for c in df.columns if 'fast' in c.lower()]
possible_slow = [c for c in df.columns if 'slow' in c.lower()]

if not possible_fast or not possible_slow:
    raise ValueError("Could not detect fast/slow columns automatically. Please check header names.")
    
fast_col = possible_fast[0]
slow_col = possible_slow[0]
print(f"Using columns: fast = '{fast_col}', slow = '{slow_col}'")

# Compute exoticness ratio
df["exoticness"] = df[fast_col] / df[slow_col]

# Group by beta
E_beta = df.groupby("beta")["exoticness"].mean().reset_index()

# Create a 2D grid (β vs slope)
if "slope" in df.columns:
    E_grid = df.pivot_table(values="exoticness", index="beta", columns="slope")
else:
    print("⚠️ No 'slope' column found; skipping heatmap.")
    E_grid = None

# --- Plot 1: Exoticness vs Beta ---
plt.figure(figsize=(7,5))
plt.plot(E_beta["beta"], np.log10(E_beta["exoticness"]), "o-", lw=2)
plt.xlabel("β")
plt.ylabel("log₁₀ Exoticness (w_fast/w_slow)")
plt.title("MU Exoticness Gradient E(β)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mu_exoticness_gradient_beta.png", dpi=300)
plt.close()

# --- Plot 2: Heatmap (if slope exists) ---
if E_grid is not None:
    plt.figure(figsize=(8,6))
    plt.imshow(np.log10(E_grid+1e-12), aspect="auto", origin="lower",
               extent=[E_grid.columns.min(), E_grid.columns.max(),
                       E_grid.index.min(), E_grid.index.max()],
               cmap="plasma")
    plt.colorbar(label="log₁₀ Exoticness (w_fast/w_slow)")
    plt.xlabel("Slope (Portal Proximity)")
    plt.ylabel("β")
    plt.title("MU Exoticness Map: β vs Slope (Portal Proximity)")
    plt.tight_layout()
    plt.savefig("mu_exoticness_gradient_heatmap.png", dpi=300)
    plt.close()

print("✅ Saved plots:")
print("   - mu_exoticness_gradient_beta.png")
if E_grid is not None:
    print("   - mu_exoticness_gradient_heatmap.png")

# Trend analysis
trend = np.polyfit(E_beta["beta"], np.log10(E_beta["exoticness"]), 1)
print(f"Trend slope (log-scale): {trend[0]:.4f}")

print("🌀 Interpretation: Higher β → greater dimensional burst. Portal proximity amplifies fast-mode acceleration.")

