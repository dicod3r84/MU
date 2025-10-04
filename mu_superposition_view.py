# mu_superposition_view.py
# Visualizes burst→harmonization→superposition as β increases.
# Saves: mu_superposition_overlay.png, mu_burst_amplitude_vs_beta.png, mu_harmonization_width_vs_beta.png

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-blocking, save-to-file only
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 1) Load data (robust filename & header detection)
# -----------------------------
CANDIDATE_FILES = [
    "mu_phase_data.csv",
    "mu_phase_beta_dq.csv",
    "phase_diagram.csv",
    "mu_portal_scan.csv",
    "mu_phase_long.csv",
    "mu_phase.csv"
]

fname = None
for f in CANDIDATE_FILES:
    if Path(f).exists():
        fname = f
        break

if fname is None:
    raise FileNotFoundError(
        "No MU CSV found. Expected one of: "
        + ", ".join(CANDIDATE_FILES)
        + "\nPlace your CSV on Desktop and rerun."
    )

df = pd.read_csv(fname)
cols = {c.lower().strip(): c for c in df.columns}

def pick(*names):
    for n in names:
        key = n.lower()
        if key in cols:
            return cols[key]
    return None

col_beta   = pick("beta","b")
col_slope  = pick("slope")
col_logr   = pick("log10_ratio","log_ratio","log10(w_fast/w_slow)")
col_ratio  = pick("ratio","ratio_wfast_wslow")
col_wf     = pick("w_fast","wfast","fast_weight")
col_ws     = pick("w_slow","wslow","slow_weight")
col_dq     = pick("DeltaQ","deltaq","ΔQ","dq")

if col_beta is None or col_slope is None:
    raise ValueError("Could not find required columns 'beta' and 'slope' in CSV.")

# Build log10_ratio if needed
if col_logr is None:
    if col_ratio is not None:
        df["__ratio__"] = df[col_ratio].astype(float)
        # Guard against non-positive values:
        df["__ratio__"] = df["__ratio__"].where(df["__ratio__"]>0, np.nan)
        df["log10_ratio"] = np.log10(df["__ratio__"])
    elif col_wf is not None and col_ws is not None:
        wf = df[col_wf].astype(float).replace(0, np.nan)
        ws = df[col_ws].astype(float).replace(0, np.nan)
        df["log10_ratio"] = np.log10(wf/ws)
    else:
        raise ValueError("Need either 'log10_ratio' or a way to compute it (ratio or w_fast/w_slow).")
    col_logr = "log10_ratio"

# Keep only necessary columns and sort
df = df[[col_beta, col_slope, col_logr] + ([col_dq] if col_dq else [])].copy()
df.rename(columns={col_beta:"beta", col_slope:"slope", col_logr:"log10_ratio"}, inplace=True)
if col_dq: df.rename(columns={col_dq:"DeltaQ"}, inplace=True)
df = df.dropna(subset=["beta","slope","log10_ratio"]).copy()
df["beta"] = df["beta"].astype(float)
df["slope"] = df["slope"].astype(float)
df["log10_ratio"] = df["log10_ratio"].astype(float)

# -----------------------------
# 2) Find portal slope per β (zero-crossing)
# Prefer DeltaQ zero; else log10_ratio zero.
# -----------------------------
def zero_crossing_x(x, y):
    """Return x* where y crosses 0 between consecutive points (linear interp). Use the crossing nearest to slope=0 if multiple."""
    x = np.asarray(x)
    y = np.asarray(y)
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return np.nan
    # choose crossing nearest to slope=0
    candidates = []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        y0, y1 = y[i], y[i+1]
        if (y1 - y0) == 0: 
            continue
        xc = x0 - y0*(x1-x0)/(y1-y0)
        candidates.append(xc)
    if not candidates:
        return np.nan
    candidates = np.array(candidates)
    return candidates[np.argmin(np.abs(candidates))]

beta_vals = sorted(df["beta"].unique())
portal = []
for b in beta_vals:
    sub = df[df["beta"]==b].sort_values("slope")
    if "DeltaQ" in sub.columns and sub["DeltaQ"].notna().any():
        xc = zero_crossing_x(sub["slope"], sub["DeltaQ"])
    else:
        xc = zero_crossing_x(sub["slope"], sub["log10_ratio"])
    portal.append((b, xc))

portal_df = pd.DataFrame(portal, columns=["beta","slope_portal"])

# -----------------------------
# 3) Burst amplitude & harmonization width
# Burst amplitude: max(log10_ratio) - min(log10_ratio) for slope in [portal, portal+Δ] (Δ default 0.6)
# Harmonization width: width around portal where |log10_ratio| <= eps (eps default 0.05)
# -----------------------------
POST_WINDOW = 0.60
EPS = 0.05

records = []
for b, sp in portal:
    sub = df[df["beta"]==b].copy()
    sub = sub.sort_values("slope")
    if not np.isfinite(sp):
        records.append((b, np.nan, np.nan))
        continue
    # Burst window
    w = sub[(sub["slope"] >= sp) & (sub["slope"] <= sp + POST_WINDOW)]
    if len(w)==0:
        burst = np.nan
    else:
        burst = float(w["log10_ratio"].max() - w["log10_ratio"].min())
    # Harmonization band (near portal)
    h = sub[(sub["slope"] >= sp - POST_WINDOW/2) & (sub["slope"] <= sp + POST_WINDOW/2)].copy()
    if len(h)==0:
        hwidth = np.nan
    else:
        # Find contiguous region around portal where |log10_ratio| <= EPS
        h["ok"] = (h["log10_ratio"].abs() <= EPS).astype(int)
        # Convert to width by slope span of ok points near sp
        ok = h[h["ok"]==1]
        if len(ok)==0:
            hwidth = 0.0
        else:
            # take the largest contiguous block that contains the slope closest to portal
            idx_center = (ok["slope"]-sp).abs().idxmin()
            sc = ok.loc[idx_center, "slope"]
            # expand left
            left = ok[ok["slope"]<=sc].sort_values("slope")
            right = ok[ok["slope"]>=sc].sort_values("slope")
            if len(left)>0:
                sL = left["slope"].iloc[0]
            else:
                sL = sc
            if len(right)>0:
                sR = right["slope"].iloc[-1]
            else:
                sR = sc
            hwidth = float(sR - sL)
    records.append((b, burst, hwidth))

metrics = pd.DataFrame(records, columns=["beta","burst_amplitude","harmonization_width"]).sort_values("beta")

# -----------------------------
# 4) Overlay of curves for highest betas (to see damping → superposition)
# -----------------------------
# Choose the top N betas
N_OVERLAY = min(6, len(beta_vals))
overlay_betas = sorted(beta_vals)[-N_OVERLAY:]

plt.figure(figsize=(10,6))
for b in overlay_betas:
    sub = df[df["beta"]==b].sort_values("slope")
    sp = portal_df.loc[portal_df["beta"]==b, "slope_portal"].values[0]
    # Shade harmonization band
    if np.isfinite(sp):
        plt.axvspan(sp-EPS*1.2, sp+EPS*1.2, alpha=0.07)
        plt.axvline(sp, linestyle="--", linewidth=1)

    plt.plot(sub["slope"], sub["log10_ratio"], label=f"β={b:.2f}")

plt.axhline(0, linestyle=":", linewidth=1)
plt.xlabel("slope")
plt.ylabel("log10(w_fast / w_slow)")
plt.title("Burst → Harmonization → Superposition (high-β overlay)")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.savefig("mu_superposition_overlay.png", dpi=180)

# -----------------------------
# 5) Burst amplitude vs β (with gentle extrapolation)
# -----------------------------
x = metrics["beta"].values
y = metrics["burst_amplitude"].values
mask = np.isfinite(y)
x_fit, y_fit = x[mask], y[mask]

plt.figure(figsize=(9,5))
plt.scatter(x_fit, y_fit, s=22, label="measured")

# Fit a smooth decay (exponential in β): y ≈ a*exp(-c*(β-β0)) + d
if len(x_fit) >= 3:
    # simple least squares on transformed form using non-linear fit via grid
    # Coarse grid for stability
    b0 = x_fit.min()
    best = None
    a_grid = np.linspace(max(y_fit)*0.5, max(y_fit)*1.5, 25)
    c_grid = np.linspace(0.05, 0.8, 25)
    d_grid = np.linspace(0.0, max(0.2, np.nanmin(y_fit)), 25)
    for a in a_grid:
        for c in c_grid:
            for d in d_grid:
                pred = a*np.exp(-c*(x_fit-b0)) + d
                err = np.nanmean((pred - y_fit)**2)
                if (best is None) or (err < best[0]):
                    best = (err, a, c, d, b0)
    _, a, c, d, b0 = best
    x_ext = np.linspace(x_fit.min(), x_fit.max()+2.0, 200)
    y_ext = a*np.exp(-c*(x_ext-b0)) + d
    plt.plot(x_ext, y_ext, linewidth=2, label="exp-fit")
    # Extrapolation region shading
    xmax_measured = x_fit.max()
    plt.axvspan(xmax_measured, x_ext.max(), alpha=0.06)
else:
    a=c=d=b0=None

plt.axhline(0, linestyle=":", linewidth=1)
plt.xlabel("β")
plt.ylabel("Burst amplitude (Δ log10 ratio) in post-portal window")
plt.title("Burst amplitude vs β (damping trend)")
plt.legend()
plt.tight_layout()
plt.savefig("mu_burst_amplitude_vs_beta.png", dpi=180)

# -----------------------------
# 6) Harmonization width vs β
# -----------------------------
xw = metrics["beta"].values
yw = metrics["harmonization_width"].values
maskw = np.isfinite(yw)
xw_fit, yw_fit = xw[maskw], yw[maskw]

plt.figure(figsize=(9,5))
plt.scatter(xw_fit, yw_fit, s=22, label="measured")

# Fit simple saturating curve: y ≈ L*(1 - exp(-k*(β-β0)))
if len(xw_fit) >= 3:
    b0 = xw_fit.min()
    best = None
    L_grid = np.linspace(max(yw_fit)*0.6, max(yw_fit)*1.8, 25)
    k_grid = np.linspace(0.05, 0.8, 25)
    for L in L_grid:
        for k in k_grid:
            pred = L*(1.0 - np.exp(-k*(xw_fit-b0)))
            err = np.nanmean((pred - yw_fit)**2)
            if (best is None) or (err < best[0]):
                best = (err, L, k, b0)
    _, L, k, b0 = best
    xw_ext = np.linspace(xw_fit.min(), xw_fit.max()+2.0, 200)
    yw_ext = L*(1.0 - np.exp(-k*(xw_ext-b0)))
    plt.plot(xw_ext, yw_ext, linewidth=2, label="sat-fit")
    plt.axvspan(xw_fit.max(), xw_ext.max(), alpha=0.06)

plt.axhline(0, linestyle=":", linewidth=1)
plt.xlabel("β")
plt.ylabel(f"Harmonization width (|log10 ratio| ≤ {EPS})")
plt.title("Harmonization (superposition) width vs β")
plt.legend()
plt.tight_layout()
plt.savefig("mu_harmonization_width_vs_beta.png", dpi=180)

# -----------------------------
# 7) Console summary
# -----------------------------
print("=== MU Superposition View ===")
print(f"Loaded: {fname}")
print("\nPortal (zero-crossing) per β (first 10):")
print(portal_df.head(10).to_string(index=False))

print("\nMetrics (first 10):")
print(metrics.head(10).to_string(index=False))

print("\nSaved figures:")
print(" - mu_superposition_overlay.png")
print(" - mu_burst_amplitude_vs_beta.png")
print(" - mu_harmonization_width_vs_beta.png")

