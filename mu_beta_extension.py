import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== MU β-Extension Sweep ===")

# Parameter ranges
betas = np.arange(8.0, 14.25, 0.25)
slopes = np.arange(-1.0, 1.05, 0.05)

def Q(v, beta): return np.exp(-beta * v**2)
def Q_T(v, beta): return np.tanh(beta * v)

records = []

for beta in betas:
    for slope in slopes:
        v_slow = 0.03125
        v_fast = slope * v_slow * 10
        Qs, Qf = Q(v_slow, beta), Q(v_fast, beta)
        QT_s, QT_f = Q_T(v_slow, beta), Q_T(v_fast, beta)

        ΔQ = QT_f - QT_s
        w_slow = np.exp(-beta * abs(Qs - QT_s))
        w_fast = np.exp(-beta * abs(Qf - QT_f))
        ratio = w_fast / w_slow

        phase = "FAST path dominant" if ratio > 1 else "SLOW path dominant"
        records.append([beta, slope, ΔQ, ratio, np.log10(ratio+1e-12), phase])

df = pd.DataFrame(records, columns=["beta", "slope", "ΔQ", "ratio_wfast_wslow", "log10_ratio", "phase"])
df.to_csv("mu_phase_beta_extended.csv", index=False)

# Plot log10 ratio vs slope for each beta
plt.figure(figsize=(9,6))
for beta in betas:
    sub = df[df["beta"] == beta]
    plt.plot(sub["slope"], sub["log10_ratio"], label=f"β={beta:.2f}")

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("slope")
plt.ylabel("log₁₀(w_fast / w_slow)")
plt.title("MU β-Extended Exoticness Map")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("mu_exoticness_extended.png", dpi=300)
print("✅ Saved results: mu_phase_beta_extended.csv, mu_exoticness_extended.png")

