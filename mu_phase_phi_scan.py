# ~/Desktop/mu_phase_phi_scan.py
# Non-blocking analysis of MU burst/harmonization and possible π/2 phase structure
# Saves: mu_phi_vs_slope.png, mu_phi_hist.png, mu_pi2_hits_vs_beta.png,
#        mu_burst_amplitude_vs_beta.png, mu_harmonization_width_vs_beta.png
# and prints a summary report.

import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

CSV_PATH = Path("~/Desktop/mu_phase_data.csv").expanduser()

print("=== MU π/2 Phase & Burst/Harmonization Analysis ===")
print(f"Loading: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# --------- Column auto-detection (robust to header variants) ----------
cols = {c.lower().strip(): c for c in df.columns}

def pick(*cands, required=True):
    for c in cands:
        lc = c.lower()
        if lc in cols:
            return cols[lc]
        # try exact unicode delta and ascii fallback
        for k in cols:
            if k.replace("Δ","delta") == lc or k.replace("δ","delta") == lc:
                return cols[k]
    if required:
        raise KeyError(f"Missing required column. Tried: {cands}")
    return None

col_beta   = pick("beta")
col_slope  = pick("slope")
col_Qs     = "Q_slow"
col_Qf     = "Q_fast"
col_dq     = pick("ΔQ","DeltaQ","deltaq")
# ratio can be either a direct 'ratio' or 'ratio_wfast_wslow', else compute from w_fast/w_slow
col_ratio  = "ratio_wfast_wslow"
col_wfast  = pick("w_fast", required=False)
col_wslow  = pick("w_slow", required=False)
col_domin  = pick("dominant","phase", required=False)

# Build a clean working frame
work = df[[col_beta, col_slope, col_Qs, col_Qf, col_dq]].copy()
work.columns = ["beta","slope","Q_slow","Q_fast","DeltaQ"]

# Ratio (fast/slow) as our primary “dominance” observable
if col_ratio is not None:
    work["ratio"] = df[col_ratio].astype(float)
elif (col_wfast is not None) and (col_wslow is not None):
    w_fast = df[col_wfast].astype(float).replace(0, np.nan)
    w_slow = df[col_wslow].astype(float).replace(0, np.nan)
    work["ratio"] = w_fast / w_slow
else:
    # If nothing else, approximate ratio from |Q_fast|/|Q_slow|
    # (fallback; should rarely be needed)
    work["ratio"] = (df[col_Qf].abs() / (df[col_Qs].abs() + 1e-12)).astype(float)

# Log dominance
work["log10_ratio"] = np.log10(np.clip(work["ratio"], 1e-300, 1e300))

# Optionally keep the text phase label
if col_domin is not None:
    work["phase_label"] = df[col_domin].astype(str)
else:
    work["phase_label"] = np.where(work["log10_ratio"]>0, "FAST path dominant", "SLOW path dominant")

# --------- Phase angle for (slow, fast) “vector” ----------
# Interpret the pair (w_slow, w_fast) or (|Q_slow|, |Q_fast|) as a 2D vector and compute angle.
if (col_wfast is not None) and (col_wslow is not None):
    vx = df[col_wslow].astype(float).values
    vy = df[col_wfast].astype(float).values
else:
    # fallback to |Q| as proxy
    vx = work["Q_slow"].abs().values
    vy = work["Q_fast"].abs().values

# θ = arctan2(vy, vx) ∈ (−π, π], where vx≈“slow”, vy≈“fast”
theta = np.arctan2(vy, vx)
work["theta"] = theta

# --------- Detect portal (zero-crossing of log10_ratio) per β ----------
def first_zero_cross(slopes, logs):
    # find first index where it crosses from <=0 to >0
    for i in range(1, len(logs)):
        if (logs[i-1] <= 0) and (logs[i] > 0):
            # linear interpolate slope location
            x0, x1 = slopes[i-1], slopes[i]
            y0, y1 = logs[i-1], logs[i]
            t = -y0 / (y1 - y0 + 1e-18)
            return x0 + t*(x1 - x0)
    return np.nan

portal = []
for b, sub in work.groupby("beta"):
    sub = sub.sort_values("slope")
    s    = sub["slope"].values
    lgr  = sub["log10_ratio"].values
    portal_slope = first_zero_cross(s, lgr)
    portal.append((b, portal_slope))
portal_df = pd.DataFrame(portal, columns=["beta","slope_portal"]).sort_values("beta")

print("\nPortal (zero-crossing) per β (first 10):")
print(portal_df.head(10).to_string(index=False))

# --------- Burst amplitude & harmonization width ----------
# Burst amplitude: max log10_ratio in a window after portal (e.g., slope ≥ portal)
# Harmonization width: range in slope until log10_ratio drops back near 0 (or below)
burst = []
for b, sub in work.groupby("beta"):
    sub = sub.sort_values("slope")
    ps = portal_df.loc[portal_df["beta"]==b, "slope_portal"].values
    if len(ps)==0 or np.isnan(ps[0]):
        burst.append((b, np.nan, np.nan))
        continue
    p = ps[0]
    after = sub[sub["slope"]>=p].copy()
    if after.empty:
        burst.append((b, np.nan, np.nan))
        continue
    peak = after["log10_ratio"].max()
    # harmonization: first time it comes back within |log10_ratio| <= 0.05 after the peak index
    peak_idx = after["log10_ratio"].idxmax()
    tail = after.loc[peak_idx:]
    near0 = tail[np.abs(tail["log10_ratio"])<=0.05]
    if near0.empty:
        width = np.nan
    else:
        width = float(near0["slope"].iloc[0] - p)
    burst.append((b, peak, width))

metrics = pd.DataFrame(burst, columns=["beta","burst_amplitude","harmonization_width"]).sort_values("beta")
print("\nMetrics (first 10):")
print(metrics.head(10).to_string(index=False))

# --------- π/2 cadence check ----------
# We’ll count points “near” multiples of π/2. Use tolerance eps (radians).
eps = 0.05  # ~2.9 degrees
# angles to test in [-π, π]
pi2_marks = np.array([ -math.pi, -math.pi*0.5, 0.0, math.pi*0.5, math.pi ])
def nearest_pi2_hits(th):
    hits = []
    for t in th:
        d = np.min(np.abs(((t - pi2_marks + math.pi) % (2*math.pi)) - math.pi))
        hits.append(d <= eps)
    return np.array(hits, dtype=bool)

work["pi2_hit"] = nearest_pi2_hits(work["theta"].values)

# Pi/2 hits per β (fraction of samples for that β)
pi2_summary = work.groupby("beta")["pi2_hit"].mean().reset_index(name="pi2_hit_rate")
# Correlate hit-rate with β (simple trend)
z = np.polyfit(pi2_summary["beta"], pi2_summary["pi2_hit_rate"], 1)
trend_slope = z[0]

# --------- Plots ----------
outdir = Path("~/Desktop").expanduser()

# θ vs slope (color by β)
plt.figure(figsize=(9,5))
for b, sub in work.groupby("beta"):
    sub = sub.sort_values("slope")
    plt.plot(sub["slope"], sub["theta"], alpha=0.6, label=f"β={b:.2f}")
plt.axhline(0, ls="--", lw=1)
for m in pi2_marks:
    plt.axhline(m, ls=":", lw=0.8)
plt.xlabel("slope")
plt.ylabel("theta = arctan2(fast, slow)")
plt.title("Phase angle (θ) across slope (π/2 gridlines)")
plt.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(outdir/"mu_phi_vs_slope.png", dpi=150)

# θ histogram
plt.figure(figsize=(6,4))
plt.hist(work["theta"], bins=72)
for m in pi2_marks:
    plt.axvline(m, ls=":", lw=0.8)
plt.xlabel("theta")
plt.ylabel("count")
plt.title("Distribution of phase angle θ")
plt.tight_layout()
plt.savefig(outdir/"mu_phi_hist.png", dpi=150)

# Pi/2 hit rate vs β
plt.figure(figsize=(7,4))
plt.plot(pi2_summary["beta"], pi2_summary["pi2_hit_rate"], marker="o")
plt.xlabel("beta")
plt.ylabel("π/2 hit rate")
plt.title("Closeness to π/2 grid vs β")
plt.tight_layout()
plt.savefig(outdir/"mu_pi2_hits_vs_beta.png", dpi=150)

# Burst metrics vs β
plt.figure(figsize=(7,4))
plt.plot(metrics["beta"], metrics["burst_amplitude"], marker="o")
plt.xlabel("beta"); plt.ylabel("peak log10(ratio)")
plt.title("Burst amplitude vs β")
plt.tight_layout()
plt.savefig(outdir/"mu_burst_amplitude_vs_beta.png", dpi=150)

plt.figure(figsize=(7,4))
plt.plot(metrics["beta"], metrics["harmonization_width"], marker="o")
plt.xlabel("beta"); plt.ylabel("Δslope to near-balance")
plt.title("Harmonization width vs β")
plt.tight_layout()
plt.savefig(outdir/"mu_harmonization_width_vs_beta.png", dpi=150)

# --------- Text Report ----------
print("\n=== Report ===")
print(f"π/2 tolerance (radians): ±{eps}")
print(f"π/2 hit-rate trend vs β (slope): {trend_slope:.4f}  "
      f"({'increasing' if trend_slope>0 else 'decreasing' if trend_slope<0 else 'flat'})")

n_hits = int(work["pi2_hit"].sum())
print(f"Total π/2-near hits: {n_hits} of {len(work)} samples")

# crude check: do bursts shrink with β?
m = metrics.dropna()
if len(m)>=2:
    z_burst = np.polyfit(m["beta"], m["burst_amplitude"], 1)
    z_width = np.polyfit(m["beta"], m["harmonization_width"], 1)
    print(f"Burst amplitude trend vs β: slope={z_burst[0]:.4f} "
          f"({'down' if z_burst[0]<0 else 'up' if z_burst[0]>0 else 'flat'})")
    print(f"Harmonization width trend vs β: slope={z_width[0]:.4f} "
          f"({'down' if z_width[0]<0 else 'up' if z_width[0]>0 else 'flat'})")
else:
    print("Not enough portal detections yet for burst metrics — check zero-crossings in your data.")

print("\nSaved:")
print(" - mu_phi_vs_slope.png")
print(" - mu_phi_hist.png")
print(" - mu_pi2_hits_vs_beta.png")
print(" - mu_burst_amplitude_vs_beta.png")
print(" - mu_harmonization_width_vs_beta.png")
print("\nDone.")

