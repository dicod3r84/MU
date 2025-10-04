import numpy as np
import matplotlib.pyplot as plt
import csv

# -----------------------------
# Configurable parameters
# -----------------------------
betas = np.arange(8.0, 10.01, 0.25)   # Beta sweep
slopes = np.linspace(-1.2, 1.2, 100)  # finer slope resolution
output_file = "phase_diagram.csv"

# -----------------------------
# Placeholder functions
# -----------------------------
def Q_slow_val(beta, slope):
    return 0.03125  # fixed slow channel

def Q_fast_val(beta, slope):
    return 0.1 + slope * 0.65  # slope-dependent fast channel

def weights(beta, Q):
    return np.exp(beta * Q)

# -----------------------------
# Helper: find zero crossing
# -----------------------------
def find_crossing(x, y, target=0.0):
    """Find approximate x where y crosses target"""
    y = np.array(y)
    idx = np.where(np.diff(np.sign(y - target)) != 0)[0]
    if len(idx) == 0:
        return None
    # Linear interpolation for more accuracy
    i = idx[0]
    x0, x1 = x[i], x[i+1]
    y0, y1 = y[i], y[i+1]
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)

# -----------------------------
# Run sweep and log results
# -----------------------------
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["beta", "slope", "Q_slow", "Q_fast", "ΔQ", 
                     "w_slow", "w_fast", "ratio", "dominant", 
                     "crit_slope_dQ0", "crit_slope_ratio05"])
    
    for beta in betas:
        dQ_vals = []
        ratio_vals = []
        
        for slope in slopes:
            Qs = Q_slow_val(beta, slope)
            Qf = Q_fast_val(beta, slope)
            dQ = Qf - Qs
            dQ_vals.append(dQ)

            ws = weights(beta, Qs)
            wf = weights(beta, Qf)
            ratio = wf / (wf + ws)
            ratio_vals.append(ratio)

            dominant = "fast" if ratio > 0.5 else "slow"

            # Console output
            print(f"--- beta={beta:.2f}, slope={slope:.2f} ---")
            print(f"Q_slow={Qs:.6f}, Q_fast={Qf:.6f}, ΔQ={dQ:.6f}")
            print(f"w_slow={ws:.4e}, w_fast={wf:.4e}, ratio={ratio:.4e}")
            print(f"**{dominant.upper()} path dominant**\n")

            # Write one line (critical slopes filled later)
            writer.writerow([beta, slope, Qs, Qf, dQ, ws, wf, ratio, dominant, "", ""])

        # After sweeping slopes → extract critical points
        crit_dQ = find_crossing(slopes, dQ_vals, target=0.0)
        crit_ratio = find_crossing(slopes, ratio_vals, target=0.5)

        # Add a summary row for this beta
        writer.writerow([beta, "summary", "", "", "", "", "", "", "",
                         crit_dQ if crit_dQ is not None else "NA",
                         crit_ratio if crit_ratio is not None else "NA"])

print(f"Results with critical slopes written to {output_file}")

# -----------------------------
# Combined plot: ΔQ and Ratio
# -----------------------------
for beta in betas:
    dQ_vals = [Q_fast_val(beta, s) - Q_slow_val(beta, s) for s in slopes]
    ratio_vals = []
    for slope in slopes:
        Qs = Q_slow_val(beta, slope)
        Qf = Q_fast_val(beta, slope)
        ws = weights(beta, Qs)
        wf = weights(beta, Qf)
        ratio_vals.append(wf / (wf + ws))

    fig, ax1 = plt.subplots(figsize=(10,6))

    # ΔQ
    ax1.plot(slopes, dQ_vals, color="tab:blue", label="ΔQ")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Slope")
    ax1.set_ylabel("ΔQ = Q_fast - Q_slow", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Ratio
    ax2 = ax1.twinx()
    ax2.plot(slopes, ratio_vals, color="tab:red", linestyle="--", label="Ratio")
    ax2.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Weight Ratio (fast / total)", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    fig.suptitle(f"ΔQ and Ratio Overlay (β={beta:.2f})")
    fig.tight_layout()
    plt.show()

