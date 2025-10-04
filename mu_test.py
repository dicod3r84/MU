import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Parameters
# ------------------------
hbar = 0.1
gamma = 1.0
betas = np.arange(8.0, 10.01, 0.25)

# Slow path (fixed)
def r_slow(t): return 0.25 * t

# Fast family: vary slope to span a wide ΔQ range
offset = 0.6
slopes = np.linspace(0.26, 1.20, 25)  # ~small to large ΔQ
def make_fast(slope): return (lambda t: offset + slope * t)

# Selector
r_c = 0.25
def T_r(r): return 1.0 / (1.0 + np.abs(r - r_c))

# Time grid
tgrid = np.linspace(0, 1.0, 600)

# Integrals
def compute_Q(rpath, gamma):
    drdt = np.gradient(rpath(tgrid), tgrid)
    integrand = 0.5 * gamma * (drdt**2)
    return np.trapz(integrand, tgrid)

def compute_QT(rpath):
    return np.trapz(T_r(rpath(tgrid)), tgrid)

# Precompute slow branch Q, QT (constant across grid)
Q_slow  = compute_Q(r_slow, gamma)
QT_slow = compute_QT(r_slow)

# Allocate arrays
R = np.zeros((len(betas), len(slopes)))    # ratio = w_fast / w_slow
DQ = np.zeros(len(slopes))                 # ΔQ for each slope

# Sweep
for j, slope in enumerate(slopes):
    r_fast = make_fast(slope)
    Q_fast  = compute_Q(r_fast, gamma)
    QT_fast = compute_QT(r_fast)
    DQ[j] = Q_fast - Q_slow

    for i, beta in enumerate(betas):
        w_slow = np.exp(-(Q_slow/hbar) + (beta*QT_slow)/hbar)
        w_fast = np.exp(-(Q_fast/hbar) + (beta*QT_fast)/hbar)
        R[i, j] = w_fast / w_slow

# Convert to log10 ratio; clip very small to avoid -inf for plotting
logR = np.log10(np.clip(R, 1e-20, None))

# ------------------------
# Plots
# ------------------------

# Heatmap of log10 ratio vs (beta, ΔQ)
plt.figure(figsize=(9,6))
extent = [DQ.min(), DQ.max(), betas.min(), betas.max()]
plt.imshow(logR, aspect='auto', origin='lower', extent=extent, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label('log10(w_fast / w_slow)')

# Regime contours
levels = [-6, -4, -1, 0]  # lines for collapse, strong suppression, coexistence
CS = plt.contour(DQ, betas, logR, levels=levels, colors='w', linewidths=1.2)
plt.clabel(CS, inline=True, fmt=lambda v: f"{v:.0f}", fontsize=9)

plt.xlabel("ΔQ = Q_fast − Q_slow")
plt.ylabel("beta")
plt.title("MU Phase Diagram: log10(weight ratio) over (beta, ΔQ)")
plt.tight_layout()
plt.savefig("mu_phase_beta_dQ.png")
plt.close()

# Optional: 1D slices — ratio vs ΔQ at a few betas
for beta_pick in [8.0, 9.0, 10.0]:
    idx = np.where(np.isclose(betas, beta_pick))[0][0]
    plt.semilogy(DQ, R[idx,:], label=f"beta={beta_pick}")
plt.xlabel("ΔQ")
plt.ylabel("w_fast / w_slow")
plt.title("Suppression curves at selected betas")
plt.legend()
plt.tight_layout()
plt.savefig("mu_ratio_vs_dQ_slices.png")
plt.close()

# Optional: save CSV for analysis
import csv
with open("mu_phase_data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["beta","slope","DeltaQ","ratio_wfast_wslow","log10_ratio"])
    for i,b in enumerate(betas):
        for j,s in enumerate(slopes):
            w.writerow([float(b), float(s), float(DQ[j]), float(R[i,j]), float(logR[i,j])])

print("Saved: mu_phase_beta_dQ.png, mu_ratio_vs_dQ_slices.png, mu_phase_data.csv")

