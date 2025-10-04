# mu_degeneracy_scan.py
# Measures MU branch degeneracy (effective number of significant paths) just after the portal.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Model knobs (same conventions as before) ----------
HBAR = 0.1
T0   = 1.0
# Weight: w ~ exp(+beta*(T - Q)/HBAR)
def weight(Q, T, beta):
    return np.exp((beta*(T - Q))/HBAR)

# Slow/fast toy function forms (as in prior tests)
def Q_slow_from_slope(s):
    # near portal the slow action cost ~ small constant (your earlier windows used 0.03125 etc.)
    return 0.03125

def Q_fast_from_slope(s, a=0.75, b=0.05):
    # simple convex rise for s>0, tuned so ΔQ becomes positive after portal:
    # Q_fast(s) = b + a*s + (s**2)/4  (feel free to tweak; stays monotone for s>0)
    return b + a*s + 0.25*s*s

def T_slow_from_slope(s):
    # T smaller on slow leg, modest monotone increase after portal
    return 0.89 + 0.05*s

def T_fast_from_slope(s):
    # T larger on fast leg, stronger monotone increase after portal
    return 0.97 + 0.15*s

# ---------- Degeneracy metrics ----------
def effective_branch_count(weights):
    W = np.array(weights, dtype=float)
    W = np.clip(W, 1e-300, np.inf)
    p = W / W.sum()
    H = -(p*np.log(p)).sum()         # natural log
    Neff = float(np.exp(H))
    return Neff, H, p

def threshold_count(prob, tau=0.01):
    return int(np.sum(prob >= tau))

# ---------- Scan parameters ----------
betas = np.arange(8.0, 10.0001, 0.25)   # 8.0 → 10.0
s0_list = [0.005, 0.01, 0.02, 0.04, 0.08]  # how far after portal we evaluate
N_paths = 101                              # micro-paths in [s0, s0+Δ]
window  = 0.06                             # Δ range width after s0
rng = np.random.default_rng(1)             # reproducible jitter

rows = []
print("=== MU Degeneracy Scan (post-portal) ===")
for beta in betas:
    for s0 in s0_list:
        # sample micro-slopes just after portal
        slopes = np.linspace(s0, s0+window, N_paths)
        slopes += rng.normal(0.0, 0.002, size=N_paths)  # tiny jitter
        slopes = np.clip(slopes, s0, s0+window)

        w_list = []
        wf_sum = 0.0
        ws_sum = 0.0
        for s in slopes:
            Qs = Q_slow_from_slope(s)
            Qf = Q_fast_from_slope(s)
            Ts = T_slow_from_slope(s)
            Tf = T_fast_from_slope(s)
            ws = weight(Qs, Ts, beta)
            wf = weight(Qf, Tf, beta)
            # Total path weight = combine slow+fast branches (two-channel toy)
            w_tot = ws + wf
            w_list.append(w_tot)
            ws_sum += ws
            wf_sum += wf

        Neff, H, p = effective_branch_count(w_list)
        D01 = threshold_count(p, tau=0.01)

        rows.append({
            "beta": float(beta),
            "s0": float(s0),
            "Neff": Neff,
            "H_natlog": H,
            "D_tau1pct": D01,
            "sum_fast": wf_sum,
            "sum_slow": ws_sum
        })
        print(f"beta={beta:.2f}, s0={s0:.3f} | Neff={Neff:.2f}, D_1%={D01}, sum_fast/sum_slow={(wf_sum/ws_sum):.3f}")

df = pd.DataFrame(rows)
df.to_csv("mu_degeneracy_scan.csv", index=False)
print("\nSaved CSV: mu_degeneracy_scan.csv")

# ---------- Plots (non-blocking) ----------
# Heatmap: pivot Neff(β, s0)
pivot = df.pivot(index="beta", columns="s0", values="Neff")
plt.figure(figsize=(7.5, 5.5))
im = plt.imshow(pivot.values, aspect='auto', origin='lower',
                extent=[pivot.columns.min(), pivot.columns.max(),
                        pivot.index.min(), pivot.index.max()])
plt.colorbar(im, label="Neff (effective branches)")
plt.xlabel("s0 (portal offset)")
plt.ylabel("beta")
plt.title("MU Degeneracy Heatmap: Neff vs (β, portal offset)")
plt.tight_layout()
plt.savefig("mu_degeneracy_heatmap.png", dpi=150)
plt.close()

# Slices: Neff vs s0 for several β
plt.figure(figsize=(7.5, 5.5))
for beta in [8.0, 8.5, 9.0, 9.5, 10.0]:
    sub = df[df["beta"]==beta].sort_values("s0")
    plt.plot(sub["s0"], sub["Neff"], marker='o', label=f"β={beta:.1f}")
plt.xlabel("s0 (portal offset)")
plt.ylabel("Neff")
plt.title("MU Degeneracy Slices: Neff vs s0")
plt.legend()
plt.tight_layout()
plt.savefig("mu_degeneracy_slices.png", dpi=150)
plt.close()

print("Saved plots: mu_degeneracy_heatmap.png, mu_degeneracy_slices.png")

