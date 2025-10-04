import numpy as np
import matplotlib.pyplot as plt

# =======================================================
# Test 13 – Negative Slope Sweep
# =======================================================
# Goal: explore symmetry/asymmetry when slope is reversed.
# Parameters: beta = 8 → 10 (step 0.25), slope = -0.26 → -1.20
# Gamma fixed = 1.0
# =======================================================

# Constants
gamma = 1.0
betas = np.arange(8.0, 10.01, 0.25)   # Beta sweep
slopes = np.linspace(-0.26, -1.20, 25)  # Negative slope sweep
hbar = 0.1

def Q(v):
    """Toy function for base Q value"""
    return v**2 / (2.0 * gamma)

def Q_T(v, beta):
    """Selector-modified Q using beta"""
    return np.tanh(beta) * v

# Storage arrays
results = []

print("\n=== Test 13: Negative Slope Sweep ===")
for beta in betas:
    for slope in slopes:
        v_slow = 0.25
        v_fast = v_slow + abs(slope)  # Use |slope| so Q_fast > Q_slow

        # Compute Qs
        Q_slow = Q(v_slow)
        Q_fast = Q(v_fast)
        QT_slow = Q_T(v_slow, beta)
        QT_fast = Q_T(v_fast, beta)

        # Weights (Boltzmann-like)
        w_slow = np.exp(beta * QT_slow / hbar)
        w_fast = np.exp(beta * QT_fast / hbar)

        deltaQ = Q_fast - Q_slow
        ratio = w_fast / w_slow if w_slow != 0 else np.nan

        results.append((beta, slope, deltaQ, ratio, np.log10(ratio) if ratio > 0 else -np.inf))

        # Print console summary
        print(f"--- beta={beta:.2f}, slope={slope:.2f} ---")
        print(f"Q_slow={Q_slow:.6f}, Q_fast={Q_fast:.6f}, ΔQ={deltaQ:.6f}")
        print(f"w_slow={w_slow:.4e}, w_fast={w_fast:.4e}, ratio={ratio:.4e}")
        if ratio < 1e-12:
            print("**Fast path suppressed**")
        elif ratio > 1e+6:
            print("**Fast path dominant**")
        else:
            print("**Coexistence regime**")

# Convert results for plotting
results = np.array(results, dtype=object)
betas_unique = np.unique(results[:,0])

# =======================================================
# Plot 1: ΔQ vs slope
# =======================================================
plt.figure(figsize=(8,6))
for beta in betas_unique:
    mask = results[:,0] == beta
    plt.plot(results[mask,1].astype(float), results[mask,2].astype(float),
             label=f"β={beta:.2f}")
plt.xlabel("Slope (negative)")
plt.ylabel("ΔQ = Q_fast - Q_slow")
plt.title("Test 13: ΔQ vs Negative Slopes")
plt.legend()
plt.grid(True)
plt.savefig("test13_deltaQ_vs_slope.png", dpi=200)

# =======================================================
# Plot 2: log10(weight ratio) vs slope
# =======================================================
plt.figure(figsize=(8,6))
for beta in betas_unique:
    mask = results[:,0] == beta
    plt.plot(results[mask,1].astype(float), results[mask,4].astype(float),
             label=f"β={beta:.2f}")
plt.xlabel("Slope (negative)")
plt.ylabel("log10(w_fast / w_slow)")
plt.title("Test 13: Weight Ratio vs Negative Slopes")
plt.legend()
plt.grid(True)
plt.savefig("test13_logratio_vs_slope.png", dpi=200)

print("\n=== Test 13 Complete ===")
print("Plots saved as test13_deltaQ_vs_slope.png and test13_logratio_vs_slope.png")

