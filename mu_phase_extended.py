import numpy as np
import pandas as pd

# ================================================================
# Test 14: Extended Slope Sweep (Crossover Domain)
# ================================================================
print("=== Test 14: Extended Slope Sweep (Crossover Domain) ===")

betas = np.arange(8.0, 10.25, 0.25)
slopes = np.linspace(-1.2, 3.0, 120)

data = []

for beta in betas:
    for slope in slopes:
        # Core relationships (simplified energy mapping)
        Q_slow = 0.03125
        Q_fast = slope * 0.65 / beta  # slower rise for large beta
        delta_Q = Q_fast - Q_slow

        # "Weight" terms (energy pathway intensity)
        w_slow = beta ** 0.5
        w_fast = np.exp(beta * Q_fast)

        ratio = w_fast / w_slow
        log_ratio = np.log10(ratio)

        # Phase dominance
        dominance = "FAST" if ratio > 1 else "SLOW"

        data.append({
            "beta": beta,
            "slope": slope,
            "Q_slow": Q_slow,
            "Q_fast": Q_fast,
            "delta_Q": delta_Q,
            "w_slow": w_slow,
            "w_fast": w_fast,
            "ratio": ratio,
            "log10_ratio": log_ratio,
            "dominance": dominance
        })

        print(f"β={beta:.2f}, slope={slope:.2f}, ΔQ={delta_Q:.3f}, log10(ratio)={log_ratio:.3f} -> {dominance}")

# Save results
df = pd.DataFrame(data)
df.to_csv("mu_phase_extended.csv", index=False)

print("\n=== Test 14 Complete ===")
print("Saved as mu_phase_extended.csv")

