
from pathlib import Path
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================ PATHS
BASE = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
GAP_DIR    = BASE / r"models\04_gap\results"
SUPPLY_DIR = BASE / r"models\03_supply\results"
RES = BASE / r"models\05_uncertainty\results"
IMG = BASE / r"models\05_uncertainty\images"

if os.environ.get("UNC_TEST") == "1":
    GAP_DIR = Path("."); SUPPLY_DIR = Path(".")
    RES = Path("unc_out"); IMG = Path("unc_out")
RES.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


# ============================================================ CONFIG
HORIZON = (2025, 2040)
N_DRAWS = 4000
SEED = 20260825

# [UNC-6] rho governs the NEED axis only.
RHO = 0.6
RHO_GRID = [0.0, 0.3, 0.6, 0.9, 1.0]
RUN_RHO_SENSITIVITY = True

SIGMA_CAP = 0.20
SIGMA_DEFAULT = 0.18
MIN_MC_LEVEL = 1.0
MAX_P90_P10 = 25.0
DRAW_CLIP = 4.0

ANCHOR_TOL = 0.04
ANCHOR_TOL_CUM = 0.02
RECONSTRUCTION_TOL = 0.05

CV_GRAD = 0.06
RATE_FLOOR = 0.005

# [UNC-11] dispersion of the LATERAL reserve's depletion rate.
#   The previous revision carried feasible_lateral as a CONSTANT replicated
#   across every draw, so the only source of variation in the lateral gap was
#   the need shock. The block therefore identified the lateral channel as the
#   binding one and then modelled it deterministically on the supply side,
#   exactly where it mattered.
#   The CV is estimated from 04's lateral_channel_history.csv using the same
#   detrended estimator as [UNC-10], and falls back to this default when that
#   file is absent or too short.
LATERAL_CV_DEFAULT = 0.15
LATERAL_CV_CAP = 0.50
LATERAL_SHOCK_CLIP = (0.45, 1.75)

# [UNC-10] how to turn the observed rate history into a dispersion.
#   "detrended" -- residuals around a fitted linear trend in logs. Measures
#                  uncertainty around the CURRENT level, which is what the
#                  projection needs.
#   "raw"       -- CV of the raw levels. On a trending series this returns the
#                  amplitude of the trend and overstates the dispersion.
RATE_DISPERSION_MODE = "detrended"
RATE_CV_CAP = 0.60

# [UNC-7] which need series the deficit is measured against.
#   "channel" -- need_young vs pool x rate, plus the lateral channel from 04.
#                The only perimeter-consistent option.
#   "total"   -- the v1 behaviour, kept to reproduce the error. Not reportable.
NEED_BASIS = "channel"

FALLBACK_COHORT_CEILING = 0.85
ROLL_GRAD_YEARS_FALLBACK = 3
CEILING_WARN_SHARE = 0.02

DEGENERATE_TOL = 1e-9

DEMO_WEIGHTS = {"central": 0.55, "high": 0.225, "low": 0.225}
DEMO_EXCLUDE = {"no_migration"}

INK = "#1a1a1a"; SHORT = "#b5171e"; MID = "#3b4a5a"; BAND = "#6c8ebf"
LATERAL = "#e08a2c"; RECRUIT = "#128a4b"
DEMO_COLORS = {"central": MID, "high": "#b5171e", "low": "#7aa6c2"}
Z90 = 1.2815515655446004


# ============================================================ UTILS
def find_col(cols, *keys):
    low = {c: str(c).lower() for c in cols}
    for k in keys:
        for c in cols:
            if k in low[c]:
                return c
    return None


def find_col_exact(cols, *names):
    """Exact match first: `recruitment` is a prefix of `recruitment_young`."""
    lower = {str(c).lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def detect(search_dir, must_have, prefer=None):
    for name in (prefer or []):
        p = Path(search_dir) / name
        if p.exists():
            return p
    for g in glob.glob(str(Path(search_dir) / "*.csv")):
        head = read_csv_safe(g)
        if head is None:
            continue
        cols = [str(c).lower() for c in head.columns]
        if all(any(m in c for c in cols) for m in must_have):
            return Path(g)
    return None


def spread(arr):
    a = np.asarray(arr, dtype=float).ravel()
    a = a[np.isfinite(a)]
    return 0.0 if a.size == 0 else float(a.max() - a.min())


# ============================================================ INPUTS
def load_recruitment_central():
    """Central need by year, plus the replacement/expansion split."""
    d = read_csv_safe(Path(GAP_DIR) / "gap_national.csv")
    if d is None:
        print("[error] gap_national.csv not found -> run 04_gap first")
        return None
    yc = find_col(d.columns, "year", "ano")
    rc = find_col(d.columns, "recruitment_need", "recruit")
    rep = find_col(d.columns, "replacement", "reposicao")
    exq = find_col(d.columns, "exits", "saida")
    expc = find_col(d.columns, "expansion", "expansao")
    out = pd.DataFrame({"year": pd.to_numeric(d[yc], errors="coerce").astype(int),
                        "need": pd.to_numeric(d[rc], errors="coerce")})
    if rep:
        out["replacement"] = pd.to_numeric(d[rep], errors="coerce")
    elif exq:
        out["replacement"] = pd.to_numeric(d[exq], errors="coerce")
    else:
        out["replacement"] = out["need"]
    out["expansion"] = (pd.to_numeric(d[expc], errors="coerce") if expc
                        else out["need"] - out["replacement"])
    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])].sort_values("year")
    out = out.reset_index(drop=True)
    share = np.where(out["need"] > 0, out["replacement"] / out["need"], 1.0)
    print(f"[input] gap_national ({len(out)} years; replacement share = {share.mean():.1%})")
    return out


def load_channel_need():
    """[UNC-7] The CHANNEL split of the need, and the lateral supply, from 04.

    This is the series the graduate pool actually constrains. Without it the
    block sets the total need against one channel's supply.

    Returns a frame with need_young, need_lateral, feasible_lateral, or None.
    """
    d = read_csv_safe(Path(GAP_DIR) / "pipeline_gap.csv")
    if d is None:
        print("[UNC-7] pipeline_gap.csv not found -> channel split unavailable")
        return None
    sc = find_col(d.columns, "conversion_scn", "scenario")
    if sc is not None:
        d = d[d[sc].astype(str).str.lower() == "central"]
    yc = find_col(d.columns, "year", "ano")
    ny = find_col_exact(d.columns, "need_young")
    nl = find_col_exact(d.columns, "need_lateral")
    fl = find_col_exact(d.columns, "feasible_lateral")
    ft = find_col_exact(d.columns, "recruitment_need")
    if not (yc and ny and nl):
        print("[UNC-7] pipeline_gap.csv lacks need_young/need_lateral -> unavailable")
        return None
    out = pd.DataFrame({
        "year": pd.to_numeric(d[yc], errors="coerce").astype(int),
        "need_young": pd.to_numeric(d[ny], errors="coerce"),
        "need_lateral": pd.to_numeric(d[nl], errors="coerce"),
    })
    out["feasible_lateral"] = (pd.to_numeric(d[fl], errors="coerce") if fl
                               else np.nan)
    if ft:
        out["need_total_04"] = pd.to_numeric(d[ft], errors="coerce")
    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])]
    out = out.sort_values("year").reset_index(drop=True)
    if out.empty:
        return None
    ysh = float(out["need_young"].sum() /
                max(out["need_young"].sum() + out["need_lateral"].sum(), 1e-9))
    print(f"[UNC-7] channel split read from pipeline_gap.csv: young share of need "
          f"= {ysh:.1%}")
    print(f"[UNC-7]   need_young       {out['need_young'].mean():>8,.0f}/yr mean")
    print(f"[UNC-7]   need_lateral     {out['need_lateral'].mean():>8,.0f}/yr mean")
    if out["feasible_lateral"].notna().any():
        print(f"[UNC-7]   feasible_lateral {out['feasible_lateral'].mean():>8,.0f}/yr "
              f"mean (from 04's reserve model)")
    else:
        print(f"[UNC-7]   feasible_lateral MISSING -> the lateral channel will be")
        print(f"[UNC-7]   assumed to deliver its own need exactly")
    return out


def load_anchor():
    """[UNC-9] The external anchor factor 04 applied to entries.

    The rates in conversion_history.csv are already multiplied by this, so the
    whole distribution 05 draws from is conditional on it.
    """
    d = read_csv_safe(Path(GAP_DIR) / "anchor_robustness.csv")
    if d is None:
        return None
    af = find_col_exact(d.columns, "anchor_factor")
    vr = find_col_exact(d.columns, "verdict_anchored")
    vu = find_col_exact(d.columns, "verdict_unanchored_raw")
    if af is None:
        return None
    v = pd.to_numeric(pd.Series([d[af].iloc[0]]), errors="coerce").iloc[0]
    if not np.isfinite(v) or v <= 0:
        return None
    info = {"factor": float(v),
            "verdict_anchored": str(d[vr].iloc[0]) if vr else None,
            "verdict_unanchored": str(d[vu].iloc[0]) if vu else None}
    print(f"\n[UNC-9] the rates read here are ANCHORED: 04 scaled observed entries")
    print(f"[UNC-9] by {info['factor']:.3f} before forming the ratio, so every draw in")
    print(f"[UNC-9] this block is conditional on that external level.")
    if info["verdict_anchored"] and info["verdict_unanchored"]:
        if info["verdict_anchored"] != info["verdict_unanchored"]:
            print(f"[UNC-9]   04 found the verdict MOVES without it: "
                  f"'{info['verdict_anchored']}' anchored against")
            print(f"[UNC-9]   '{info['verdict_unanchored']}' unanchored. The anchor is "
                  f"deciding the sign, not")
            print(f"[UNC-9]   the data. This block inherits that dependence in full.")
        else:
            print(f"[UNC-9]   04 found the verdict stable without it "
                  f"('{info['verdict_anchored']}' both ways).")
    return info


def load_sigma(years):
    """Relative log-sd per year from the 03 Monte Carlo, with degeneracy guards."""
    p = detect(SUPPLY_DIR, must_have=["year"], prefer=["recruitment_montecarlo_v3.csv"])
    sig = np.full(len(years), SIGMA_DEFAULT, dtype=float)
    bad = []
    if p is None:
        print(f"[input] need uncertainty: no Monte Carlo found -> default sigma "
              f"{SIGMA_DEFAULT}")
        return sig, bad, False
    d = read_csv_safe(p)
    if d is None:
        return sig, bad, False
    yc = find_col(d.columns, "year", "ano")
    c10 = find_col(d.columns, "p10", "q10")
    c50 = find_col(d.columns, "p50", "median")
    c90 = find_col(d.columns, "p90", "q90")
    if not (yc and c10 and c90):
        print(f"[input] Monte Carlo lacks P10/P90 -> default sigma {SIGMA_DEFAULT}")
        return sig, bad, False
    d = d.copy()
    d[yc] = pd.to_numeric(d[yc], errors="coerce")
    d = d.dropna(subset=[yc])
    d = d.set_index(d[yc].astype(int))
    raw = []
    for j, y in enumerate(years):
        if y not in d.index:
            bad.append(int(y)); continue
        a = float(pd.to_numeric(pd.Series([d.loc[y, c10]]), errors="coerce").iloc[0])
        b = float(pd.to_numeric(pd.Series([d.loc[y, c90]]), errors="coerce").iloc[0])
        m = (float(pd.to_numeric(pd.Series([d.loc[y, c50]]), errors="coerce").iloc[0])
             if c50 else np.sqrt(max(a, 1e-9) * max(b, 1e-9)))
        if (not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= a
                or (np.isfinite(m) and m < MIN_MC_LEVEL) or (b / a) > MAX_P90_P10):
            bad.append(int(y)); continue
        s = (np.log(b) - np.log(a)) / (2 * Z90)
        if np.isfinite(s) and s > 0:
            raw.append(s)
            sig[j] = min(s, SIGMA_CAP)
        else:
            bad.append(int(y))
    if bad and len(bad) < len(years):
        good = np.array([y not in bad for y in years])
        sig[~good] = np.interp(np.flatnonzero(~good), np.flatnonzero(good), sig[good])
        print(f"[guard] degenerate Monte Carlo year(s) dropped and interpolated: {bad}")
    elif bad and len(bad) == len(years):
        print(f"[guard] ALL Monte Carlo years degenerate -> default sigma {SIGMA_DEFAULT}")
    raw_med = float(np.median(raw)) if raw else float("nan")
    print(f"[input] annual log-sd of the need: min={sig.min():.3f} "
          f"median={np.median(sig):.3f} max={sig.max():.3f} "
          f"(raw median={raw_med:.3f}, cap={SIGMA_CAP})")
    return sig, bad, True


def load_conversion():
    """Official projected rate path, the observed history, and 04's ceiling."""
    d = read_csv_safe(Path(GAP_DIR) / "conversion_rate_paths.csv")
    path, mode = None, None
    rate_ceiling, cohort_ceiling, roll = None, None, None
    if d is not None:
        yc = find_col(d.columns, "year", "ano")
        mo = find_col(d.columns, "rate_mode_official", "rate_mode_oficial")
        mode = str(d[mo].iloc[0]) if mo else "last"
        col = find_col_exact(d.columns, f"rate_{mode}") or \
              find_col(d.columns, "rate_last", "rate_damped", "rate_median")
        if yc and col:
            path = {int(y): float(v) for y, v in zip(d[yc], d[col]) if np.isfinite(v)}
        rc = find_col_exact(d.columns, "rate_ceiling")
        cc = find_col_exact(d.columns, "cohort_conv_ceiling")
        if rc is not None:
            v = pd.to_numeric(pd.Series([d[rc].iloc[0]]), errors="coerce").iloc[0]
            if np.isfinite(v) and v > 0:
                rate_ceiling = float(v)
        if cc is not None:
            v = pd.to_numeric(pd.Series([d[cc].iloc[0]]), errors="coerce").iloc[0]
            if np.isfinite(v) and v > 0:
                cohort_ceiling = float(v)
        if rate_ceiling and cohort_ceiling and rate_ceiling > 0:
            roll = int(round(cohort_ceiling / rate_ceiling))

    h = read_csv_safe(Path(GAP_DIR) / "conversion_history.csv")
    hist = None; hist_years = None
    if h is not None:
        rc2 = find_col_exact(h.columns, "rate") or find_col(h.columns, "rate", "taxa")
        yc2 = find_col(h.columns, "year", "ano")
        if rc2:
            tmp = pd.DataFrame({"rate": pd.to_numeric(h[rc2], errors="coerce")})
            if yc2:
                tmp["year"] = pd.to_numeric(h[yc2], errors="coerce")
            tmp = tmp.dropna(subset=["rate"])
            tmp = tmp[(tmp["rate"] > 0) & (tmp["rate"] < 3)]
            if len(tmp) >= 3:
                hist = tmp["rate"].values
                hist_years = (tmp["year"].values if "year" in tmp.columns
                              else np.arange(len(tmp), dtype=float))
                print(f"[input] conversion_history: {len(tmp)} rates "
                      f"(median={np.median(hist):.1%}, min={hist.min():.1%}, "
                      f"max={hist.max():.1%})")

    if path:
        print(f"[input] conversion-rate path (official mode='{mode}'): "
              f"{path.get(HORIZON[0], float('nan')):.1%} -> "
              f"{path.get(HORIZON[1], float('nan')):.1%}")

    if rate_ceiling is None:
        roll = ROLL_GRAD_YEARS_FALLBACK
        cohort_ceiling = FALLBACK_COHORT_CEILING
        rate_ceiling = cohort_ceiling / roll
        print(f"[ceiling] conversion_rate_paths.csv carries no ceiling columns -> "
              f"falling back to {cohort_ceiling:.0%} cohort conversion "
              f"(rate {rate_ceiling:.2%})")
    else:
        print(f"[ceiling] rate ceiling INHERITED from 04: {rate_ceiling:.2%} "
              f"(= {cohort_ceiling:.0%} cohort conversion over {roll} cohorts)")
    return path, hist, hist_years, mode, rate_ceiling, cohort_ceiling, roll


def rate_dispersion(hist, hist_years):
    """[UNC-10] Dispersion of the conversion rate around its CURRENT level.

    The raw CV of a trending series measures the amplitude of the trend. On the
    real history the nine rates climb almost monotonically from 2.9% to 19.6%,
    which returns a CV near 49% and hands the conversion rate most of the
    variance. What the projection needs is the scatter around the level, so the
    default fits a linear trend in logs and takes the residual dispersion.

    Returns (cv_used, meta).
    """
    if hist is None or len(hist) < 3:
        return 0.20, {"mode": "default", "cv_raw": np.nan, "cv_detrended": np.nan,
                      "slope_pp_per_year": np.nan, "n": 0}
    x = np.asarray(hist_years, float)
    y = np.asarray(hist, float)
    cv_raw = float(np.std(y, ddof=1) / max(np.mean(y), 1e-9))

    cv_det = np.nan; slope_pp = np.nan
    if len(y) >= 3 and np.ptp(x) > 0:
        ly = np.log(np.clip(y, 1e-9, None))
        b, a = np.polyfit(x - x.mean(), ly, 1)
        resid = ly - (a + b * (x - x.mean()))
        # residual sd in logs is approximately the relative dispersion
        cv_det = float(np.std(resid, ddof=1))
        # slope in percentage points per year, at the mean level
        slope_pp = float(b * np.mean(y) * 100.0)

    use_det = (RATE_DISPERSION_MODE == "detrended" and np.isfinite(cv_det)
               and cv_det > DEGENERATE_TOL)
    cv = cv_det if use_det else cv_raw
    cv = float(np.clip(cv, 0.0, RATE_CV_CAP))

    print(f"\n[UNC-10] conversion-rate dispersion")
    print(f"[UNC-10]   raw CV of the levels        = {cv_raw:.1%}")
    if np.isfinite(cv_det):
        print(f"[UNC-10]   residual sd around a trend  = {cv_det:.1%}  "
              f"(trend {slope_pp:+.2f} pp/yr)")
    if np.isfinite(cv_det) and cv_raw > 1.5 * cv_det:
        print(f"[UNC-10]   the raw figure is {cv_raw/max(cv_det,1e-9):.1f}x the "
              f"detrended one: the series TRENDS,")
        print(f"[UNC-10]   so resampling its levels measures the climb, not the "
              f"scatter around")
        print(f"[UNC-10]   today's level. Using '{RATE_DISPERSION_MODE}' -> "
              f"CV = {cv:.1%}")
    else:
        print(f"[UNC-10]   using '{RATE_DISPERSION_MODE}' -> CV = {cv:.1%}")
    return cv, {"mode": RATE_DISPERSION_MODE, "cv_raw": cv_raw,
                "cv_detrended": cv_det, "slope_pp_per_year": slope_pp,
                "n": int(len(y)), "cv_used": cv}


def lateral_dispersion():
    """[UNC-11] Dispersion of the lateral reserve's depletion rate.

    04 estimates the lateral rate as the last-5-year mean of
    lateral entries / previous-year stock, and publishes the year-by-year series
    in lateral_channel_history.csv. That series is the natural evidence for how
    uncertain the level is, so the CV is estimated from it rather than assumed.

    The same detrended estimator as [UNC-10] is used: a trending series would
    otherwise have the amplitude of its climb read as scatter. Falls back to
    LATERAL_CV_DEFAULT when the file is absent or shorter than three years.

    Returns (cv_used, meta).
    """
    meta = {"mode": "default", "cv_raw": np.nan, "cv_detrended": np.nan,
            "slope_pp_per_year": np.nan, "n": 0, "source": "default",
            "cv_used": LATERAL_CV_DEFAULT}
    d = read_csv_safe(Path(GAP_DIR) / "lateral_channel_history.csv")
    if d is None:
        print(f"\n[UNC-11] lateral reserve dispersion")
        print(f"[UNC-11]   lateral_channel_history.csv not found -> default "
              f"CV = {LATERAL_CV_DEFAULT:.0%}")
        print(f"[UNC-11]   that is an ASSUMPTION, not an estimate. Run 04_gap to "
              f"replace it.")
        return LATERAL_CV_DEFAULT, meta

    rc = find_col_exact(d.columns, "lateral_rate") or \
        find_col(d.columns, "lateral_rate", "rate")
    yc = find_col(d.columns, "year", "ano")
    if rc is None:
        print(f"\n[UNC-11] lateral_channel_history.csv lacks a rate column -> "
              f"default CV = {LATERAL_CV_DEFAULT:.0%}")
        return LATERAL_CV_DEFAULT, meta

    tmp = pd.DataFrame({"rate": pd.to_numeric(d[rc], errors="coerce")})
    if yc:
        tmp["year"] = pd.to_numeric(d[yc], errors="coerce")
    tmp = tmp.dropna(subset=["rate"])
    tmp = tmp[(tmp["rate"] > 0) & (tmp["rate"] < 1)]
    if len(tmp) < 3:
        print(f"\n[UNC-11] lateral history has {len(tmp)} usable year(s) -> default "
              f"CV = {LATERAL_CV_DEFAULT:.0%}")
        return LATERAL_CV_DEFAULT, meta

    y = tmp["rate"].values.astype(float)
    x = (tmp["year"].values.astype(float) if "year" in tmp.columns
         else np.arange(len(y), dtype=float))
    cv_raw = float(np.std(y, ddof=1) / max(np.mean(y), 1e-9))

    cv_det = np.nan; slope_pp = np.nan
    if np.ptp(x) > 0:
        ly = np.log(np.clip(y, 1e-9, None))
        b, a = np.polyfit(x - x.mean(), ly, 1)
        resid = ly - (a + b * (x - x.mean()))
        cv_det = float(np.std(resid, ddof=1))
        slope_pp = float(b * np.mean(y) * 100.0)

    use_det = (RATE_DISPERSION_MODE == "detrended" and np.isfinite(cv_det)
               and cv_det > DEGENERATE_TOL)
    cv = cv_det if use_det else cv_raw
    cv = float(np.clip(cv, 0.0, LATERAL_CV_CAP))

    print(f"\n[UNC-11] lateral reserve dispersion, estimated from "
          f"lateral_channel_history.csv")
    print(f"[UNC-11]   {len(y)} observed year(s) of lateral entries / prev-year stock")
    print(f"[UNC-11]   observed rate {y.min():.3%} to {y.max():.3%} "
          f"(mean {y.mean():.3%})")
    print(f"[UNC-11]   raw CV of the levels        = {cv_raw:.1%}")
    if np.isfinite(cv_det):
        print(f"[UNC-11]   residual sd around a trend  = {cv_det:.1%}  "
              f"(trend {slope_pp:+.3f} pp/yr)")
    print(f"[UNC-11]   using '{RATE_DISPERSION_MODE}' -> CV = {cv:.1%}")
    print(f"[UNC-11]   the reserve is the channel that binds, so leaving it")
    print(f"[UNC-11]   deterministic understated the uncertainty where it matters")
    meta = {"mode": RATE_DISPERSION_MODE, "cv_raw": cv_raw, "cv_detrended": cv_det,
            "slope_pp_per_year": slope_pp, "n": int(len(y)),
            "source": "lateral_channel_history.csv", "cv_used": cv,
            "rate_min": float(y.min()), "rate_max": float(y.max()),
            "rate_mean": float(y.mean())}
    return cv, meta


def load_pool():
    d = read_csv_safe(Path(GAP_DIR) / "graduates_projection.csv")
    if d is None:
        return None
    yc = find_col(d.columns, "year", "ano")
    gc = find_col_exact(d.columns, "graduate_pool") or \
         find_col(d.columns, "graduate_pool", "diplomados_pool") or \
         find_col(d.columns, "graduates", "diplomados")
    if not (yc and gc):
        return None
    out = pd.DataFrame({"year": pd.to_numeric(d[yc], errors="coerce").astype(int),
                        "pool": pd.to_numeric(d[gc], errors="coerce")})
    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])].dropna()
    out = out.reset_index(drop=True)
    print(f"[input] graduate pool ({len(out)} years, column='{gc}', "
          f"{out['pool'].iloc[0]:,.0f} -> {out['pool'].iloc[-1]:,.0f})")
    return out


def load_demand_levels():
    d = read_csv_safe(Path(GAP_DIR) / "gap_stock.csv")
    if d is None:
        print("[input] gap_stock.csv not found -> demographic axis disabled")
        return None
    sc = find_col(d.columns, "scenario", "cenario")
    yc = find_col(d.columns, "year", "ano")
    vc = find_col_exact(d.columns, "demand") or find_col(d.columns, "demand", "procura")
    if not (sc and yc and vc):
        print("[input] gap_stock without scenario/year/demand -> axis disabled")
        return None
    d = d.copy()
    d["scenario"] = d[sc].astype(str).str.lower()
    d = d[~d["scenario"].isin(DEMO_EXCLUDE)]
    d["year"] = pd.to_numeric(d[yc], errors="coerce").astype(int)
    d["demand"] = pd.to_numeric(d[vc], errors="coerce")
    piv = d.pivot_table(index="year", columns="scenario", values="demand").sort_index()
    keep = [c for c in piv.columns if c in DEMO_WEIGHTS]
    if not keep or "central" not in keep:
        print("[input] gap_stock lacks a central scenario -> axis disabled")
        return None
    piv = piv[keep]
    sp = float(piv.iloc[-1].max() - piv.iloc[-1].min())
    print(f"[input] demand levels by scenario: {keep} "
          f"(2040 high-low spread = {sp:,.0f} teachers)")
    return piv


# ============================================================ SCENARIO NEEDS
def build_scenario_needs(rec, demand_piv, channels):
    """need_s(t) = replacement(t) + [D_s(t) - D_s(t-1)], split by channel.

    [UNC-7] The scenario shifts the TOTAL need; the young/lateral split from 04
    is applied proportionally, so both channels move together with demography
    and the perimeter stays consistent.

    [UNC-1] Negative results are floored at zero and counted.
    """
    years = rec["year"].tolist()
    rep = rec.set_index("year")["replacement"].reindex(years).values.astype(float)

    if channels is not None:
        ch = channels.set_index("year").reindex(years)
        ny = ch["need_young"].values.astype(float)
        nl = ch["need_lateral"].values.astype(float)
        tot_ch = np.where((ny + nl) > 0, ny + nl, np.nan)
        young_share = np.where(np.isfinite(tot_ch), ny / tot_ch, np.nan)
        young_share = pd.Series(young_share).ffill().bfill().values
    else:
        young_share = np.ones(len(years))

    needs, needs_y, needs_l = {}, {}, {}
    if demand_piv is None or demand_piv.empty:
        base = rec["need"].values.astype(float)
        needs["central"] = base
        needs_y["central"] = base * young_share
        needs_l["central"] = base * (1.0 - young_share)
        print("[scenario needs] demographic axis disabled -> central path only")
        return needs, needs_y, needs_l, ["central"], 0, young_share

    n_floored = 0
    for scn in demand_piv.columns:
        dser = demand_piv[scn].reindex(years)
        exp = dser.diff().values.astype(float)
        exp[0] = float(rec["expansion"].iloc[0]) if "expansion" in rec else 0.0
        raw = rep + exp
        n_floored += int((raw < 0).sum())
        tot = np.maximum(raw, 0.0)
        needs[scn] = tot
        needs_y[scn] = tot * young_share
        needs_l[scn] = tot * (1.0 - young_share)

    scns = list(needs)
    tot_s = {s: float(np.nansum(needs[s])) for s in scns}
    print("[scenario needs] cumulative 2025-2040: " +
          " | ".join(f"{s}={tot_s[s]:,.0f}" for s in scns))
    print(f"[scenario needs] high-low spread over the horizon = "
          f"{max(tot_s.values()) - min(tot_s.values()):,.0f} teachers")
    if channels is not None:
        print(f"[scenario needs] split by channel using 04's young share "
              f"({np.mean(young_share):.1%} mean)  [UNC-7]")
    if n_floored:
        print(f"[UNC-1] {n_floored} (scenario, year) cell(s) had a NEGATIVE need and were")
        print(f"[UNC-1] floored at zero.")
    return needs, needs_y, needs_l, scns, n_floored, young_share


def check_reconstruction(rec, needs):
    """[UNC-5] Central rebuilt vs central published, like for like."""
    if "central" not in needs:
        return None
    pub = rec["need"].values.astype(float)
    rebuilt = np.asarray(needs["central"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(pub != 0, rebuilt / pub - 1.0, np.nan)
    worst = float(np.nanmax(np.abs(dev))) if np.isfinite(dev).any() else np.nan
    iworst = int(np.nanargmax(np.abs(dev))) if np.isfinite(dev).any() else 0
    cum_dev = (rebuilt.sum() / pub.sum() - 1.0) if pub.sum() else np.nan
    ok = np.isfinite(worst) and worst <= RECONSTRUCTION_TOL

    print(f"\n[UNC-5] reconstruction check: central need rebuilt as "
          f"replacement + d(demand)")
    print(f"        vs the need 04 actually published")
    print(f"        worst year deviation = {worst:.1%} "
          f"({int(rec['year'].iloc[iworst])}), cumulative {cum_dev:+.1%} "
          f"-> {'PASS' if ok else 'WARN'}")
    if not ok:
        print(f"        >>> the two do NOT agree within {RECONSTRUCTION_TOL:.0%}.")
    return pd.DataFrame({"year": rec["year"].values, "need_published_04": pub,
                         "need_rebuilt_central": rebuilt, "rel_dev": dev})


# ============================================================ DIAGNOSTICS
def diagnose_inputs(sig, cv_rate, needs, scns, channels,
                    cv_lateral=LATERAL_CV_DEFAULT, lat_meta=None):
    print("\n[DIAGNOSTIC] can each axis vary?")
    issues = []

    s_sig = float(np.max(sig)) if len(sig) else 0.0
    ok_need = s_sig > DEGENERATE_TOL
    print(f"    need shock       : sigma max = {s_sig:.4f}   "
          f"{'OK' if ok_need else 'DEGENERATE (no spread)'}")
    if not ok_need:
        issues.append("need_shock")

    ok_rate = cv_rate > DEGENERATE_TOL
    print(f"    conversion rate  : CV = {cv_rate:.1%}   "
          f"{'OK' if ok_rate else 'DEGENERATE'}")
    if not ok_rate:
        issues.append("conversion_rate")

    ok_pool = CV_GRAD > DEGENERATE_TOL
    print(f"    graduate pool    : CV = {CV_GRAD:.1%}   {'OK' if ok_pool else 'DEGENERATE'}")
    if not ok_pool:
        issues.append("graduate_pool")

    # [UNC-11] the reserve is the binding channel; say whether it can vary
    ok_lat = cv_lateral > DEGENERATE_TOL
    src = (lat_meta or {}).get("source", "default")
    tag = "OK" if ok_lat else "DEGENERATE (deterministic reserve)"
    print(f"    lateral reserve  : CV = {cv_lateral:.1%} ({src})   {tag}")
    if not ok_lat:
        issues.append("lateral_reserve")

    if len(scns) > 1:
        tot = {s: float(np.nansum(needs[s])) for s in scns}
        sp = max(tot.values()) - min(tot.values())
        ok_demo = sp > 1.0
        print(f"    demographic scn  : high-low spread = {sp:,.0f}   "
              f"{'OK' if ok_demo else 'DEGENERATE'}")
    else:
        ok_demo = False
        print("    demographic scn  : single scenario   DEGENERATE")
    if not ok_demo:
        issues.append("demographic_scenario")

    # [UNC-7] a perimeter check is a diagnostic, not a footnote
    print(f"    perimeter        : basis='{NEED_BASIS}'   "
          f"{'OK (channel-consistent)' if (NEED_BASIS == 'channel' and channels is not None) else 'MISMATCHED'}")
    if NEED_BASIS != "channel" or channels is None:
        issues.append("perimeter")

    if issues:
        print(f"[DEGENERATE] {len(issues)} axis/axes flagged: {', '.join(issues)}")
    else:
        print("[DIAGNOSTIC] all axes can vary and the perimeter is consistent.")
    return issues


# ============================================================ SIMULATION
def simulate(rho, rec, needs_y, needs_l, feas_l, scns, pool, sig, rate_path,
             cv_rate, years, rate_ceiling, cv_lateral=LATERAL_CV_DEFAULT,
             seed=SEED):
    """[UNC-7] Two channels. The graduate pool constrains the young channel only.

    [UNC-8] Both the clipped and the unclipped rate are carried, so the effect
    of the ceiling's one-sided truncation can be measured.

    [UNC-11] The lateral reserve now carries its own dispersion.
    """
    rng = np.random.default_rng(seed)
    n, T = N_DRAWS, len(years)

    w = np.array([DEMO_WEIGHTS.get(s, 0.0) for s in scns], float)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(scns)) / len(scns)
    picked_idx = rng.choice(len(scns), size=n, p=w)

    base_y = np.array([needs_y[s] for s in scns], float)
    base_l = np.array([needs_l[s] for s in scns], float)
    need_y_central = base_y[picked_idx, :]
    need_l_central = base_l[picked_idx, :]

    # need shock: common to both channels, they move with the same demography
    Zc = rng.normal(0.0, 1.0, size=(n, 1))
    Zt = rng.normal(0.0, 1.0, size=(n, T))
    eps = np.sqrt(rho) * Zc + np.sqrt(max(1.0 - rho, 0.0)) * Zt
    mult = np.clip(np.exp(sig[None, :] * eps), 1.0 / DRAW_CLIP, DRAW_CLIP)
    need_y = need_y_central * mult
    need_l = need_l_central * mult

    # conversion rate
    rate_base = np.array([rate_path.get(int(y), np.nan) for y in years], float)
    rate_shock = np.clip(rng.normal(1.0, cv_rate, size=(n, 1)), 0.35, 1.9)
    rate_unclipped = np.clip(rate_base[None, :] * rate_shock, RATE_FLOOR, None)
    rate_draws = np.clip(rate_unclipped, RATE_FLOOR, rate_ceiling)
    share_clipped = float(np.mean(rate_unclipped > rate_ceiling))

    # graduate pool
    pool_base = pool.set_index("year")["pool"].reindex(years).values.astype(float)
    pool_shock = np.clip(rng.normal(1.0, CV_GRAD, size=(n, 1)), 0.7, 1.3)
    pool_draws = pool_base[None, :] * pool_shock

    ent_y = pool_draws * rate_draws
    ent_y_unclipped = pool_draws * rate_unclipped

    # [UNC-7][UNC-11] lateral supply from 04's reserve model, WITH its own shock.
    # The previous revision replicated feasible_lateral across every draw, so the
    # channel the block identifies as binding had no supply uncertainty at all.
    # The shock is drawn once per path and held across years, matching how the
    # rate and pool shocks are treated: the depletion rate is a level assumption,
    # not a year-by-year process.
    lat_shock = np.clip(rng.normal(1.0, cv_lateral, size=(n, 1)),
                        LATERAL_SHOCK_CLIP[0], LATERAL_SHOCK_CLIP[1])
    if feas_l is not None and np.isfinite(feas_l).all():
        ent_l_central = np.tile(feas_l[None, :], (n, 1))
        ent_l = ent_l_central * lat_shock
        lateral_modelled = True
    else:
        ent_l_central = need_l          # assume the reserve delivers exactly
        ent_l = ent_l_central
        lateral_modelled = False

    gap_y = need_y - ent_y
    gap_l = need_l - ent_l
    gap_y_unclipped = need_y - ent_y_unclipped
    # [GAP-29] no cross-channel credit
    gap_no_offset = np.maximum(gap_y, 0.0) + np.maximum(gap_l, 0.0)
    gap_netted = (need_y + need_l) - (ent_y + ent_l)

    return {"need_y": need_y, "need_l": need_l,
            "need_y_central": need_y_central, "need_l_central": need_l_central,
            "ent_y": ent_y, "ent_l": ent_l, "ent_y_unclipped": ent_y_unclipped,
            "gap_y": gap_y, "gap_l": gap_l, "gap_y_unclipped": gap_y_unclipped,
            "gap_no_offset": gap_no_offset, "gap_netted": gap_netted,
            "rate": rate_draws, "rate_unclipped": rate_unclipped, "pool": pool_draws,
            "ent_l_central": ent_l_central, "lat_shock": lat_shock,
            "lateral_modelled": bool(lateral_modelled),
            "picked": np.array([scns[i] for i in picked_idx]),
            "cv_rate": cv_rate, "cv_lateral": cv_lateral,
            "base_y": base_y, "base_l": base_l,
            "scns": list(scns), "share_clipped": share_clipped}


def summarise(sim, rec, years):
    """[UNC-7] The headline is the GRADUATE-channel gap, the binding one."""
    d = sim["gap_y"]
    fan = pd.DataFrame({
        "year": years,
        "p10": np.percentile(d, 10, axis=0), "p25": np.percentile(d, 25, axis=0),
        "p50": np.percentile(d, 50, axis=0), "p75": np.percentile(d, 75, axis=0),
        "p90": np.percentile(d, 90, axis=0),
        "prob_shortage": (d > 0).mean(axis=0),
        "prob_shortage_unclipped": (sim["gap_y_unclipped"] > 0).mean(axis=0),
        "median_need_young": np.median(sim["need_y"], axis=0),
        "median_entrants_young": np.median(sim["ent_y"], axis=0),
        "median_gap_lateral": np.median(sim["gap_l"], axis=0),
        "median_gap_no_offset": np.median(sim["gap_no_offset"], axis=0),
        "median_gap_netted": np.median(sim["gap_netted"], axis=0),
        # [UNC-11] the lateral channel now has a distribution of its own
        "lat_p10": np.percentile(sim["gap_l"], 10, axis=0),
        "lat_p90": np.percentile(sim["gap_l"], 90, axis=0),
        "prob_shortage_lateral": (sim["gap_l"] > 0).mean(axis=0),
        "median_entrants_lateral": np.median(sim["ent_l"], axis=0),
    })
    fan["lat_sign_determined"] = (fan.lat_p10 > 0) | (fan.lat_p90 < 0)
    fan["sign_determined"] = (fan.p10 > 0) | (fan.p90 < 0)
    fan["zero_in_p10_p90"] = ~fan["sign_determined"]
    fan["zero_in_p25_p75"] = (fan.p25 < 0) & (fan.p75 > 0)
    fan["band_width"] = fan.p90 - fan.p10

    cum = (sim["need_y"] + sim["need_l"]).sum(axis=1)
    cum_stats = {"p10": float(np.percentile(cum, 10)), "p50": float(np.percentile(cum, 50)),
                 "p90": float(np.percentile(cum, 90)), "mean": float(cum.mean()),
                 "central_path_total": float(rec["need"].sum())}
    return fan, cum, cum_stats


def variance_decomposition(sim, years, target="graduate"):
    """[UNC-2][UNC-7][UNC-11] Freeze one axis at a time.

    `target` selects the quantity being decomposed:
      "graduate" -- the graduate-channel gap, need_y - ent_y. The lateral
                    reserve does not enter this quantity at all, so it is
                    correctly ABSENT rather than reported as a zero bar.
      "combined" -- the no-cross-credit gap, max(gap_y,0) + max(gap_l,0), which
                    is the [GAP-29] headline. The lateral reserve appears here,
                    and on a run where the lateral channel is the binding one
                    this is the decomposition that matters.
    """
    need_y, ent_y = sim["need_y"], sim["ent_y"]
    need_l, ent_l = sim["need_l"], sim["ent_l"]
    rate_med = np.median(sim["rate"], axis=0)[None, :]
    pool_med = np.median(sim["pool"], axis=0)[None, :]
    lat_med = np.median(sim["lat_shock"], axis=0)[None, :]

    scns = sim["scns"]
    if "central" in scns:
        den_y = np.where(sim["need_y_central"] == 0, np.nan, sim["need_y_central"])
        shock_y = np.where(np.isfinite(den_y), need_y / den_y, 1.0)
        cen_y = sim["base_y"][scns.index("central")][None, :]
        need_y_fix_scn = cen_y * shock_y
        den_l = np.where(sim["need_l_central"] == 0, np.nan, sim["need_l_central"])
        shock_l = np.where(np.isfinite(den_l), need_l / den_l, 1.0)
        cen_l = sim["base_l"][scns.index("central")][None, :]
        need_l_fix_scn = cen_l * shock_l
    else:
        need_y_fix_scn, need_l_fix_scn = need_y, need_l

    ent_l_fix = sim["ent_l_central"] * lat_med     # lateral shock frozen

    def _combine(gy, gl):
        return (np.maximum(gy, 0.0) + np.maximum(gl, 0.0)).sum(axis=1)

    if target == "combined":
        def _v(gy, gl):
            return float(np.var(_combine(gy, gl)))
        v_all = _v(need_y - ent_y, need_l - ent_l)
        v_no_rate = _v(need_y - sim["pool"] * rate_med, need_l - ent_l)
        v_no_shock = _v(sim["need_y_central"] - ent_y,
                        sim["need_l_central"] - ent_l)
        v_no_pool = _v(need_y - pool_med * sim["rate"], need_l - ent_l)
        v_no_scn = _v(need_y_fix_scn - ent_y, need_l_fix_scn - ent_l)
        v_no_lat = _v(need_y - ent_y, need_l - ent_l_fix)
        axes = ["conversion_rate", "need_shock", "graduate_pool",
                "demographic_scenario", "lateral_reserve"]
        drops = [max(v_all - v_no_rate, 0.0), max(v_all - v_no_shock, 0.0),
                 max(v_all - v_no_pool, 0.0), max(v_all - v_no_scn, 0.0),
                 max(v_all - v_no_lat, 0.0)]
    else:
        def _v(gy):
            return float(np.var(gy.sum(axis=1)))
        v_all = _v(need_y - ent_y)
        v_no_rate = _v(need_y - sim["pool"] * rate_med)
        v_no_shock = _v(sim["need_y_central"] - ent_y)
        v_no_pool = _v(need_y - pool_med * sim["rate"])
        v_no_scn = _v(need_y_fix_scn - ent_y)
        axes = ["conversion_rate", "need_shock", "graduate_pool",
                "demographic_scenario"]
        drops = [max(v_all - v_no_rate, 0.0), max(v_all - v_no_shock, 0.0),
                 max(v_all - v_no_pool, 0.0), max(v_all - v_no_scn, 0.0)]

    contrib = pd.DataFrame({"axis": axes, "variance_reduction_if_fixed": drops})
    s = contrib["variance_reduction_if_fixed"].sum()
    contrib["contribution_pct"] = np.where(s > 0,
                                           contrib["variance_reduction_if_fixed"] / s, np.nan)
    contrib["total_variance"] = v_all
    contrib["target"] = target
    return contrib.sort_values("contribution_pct", ascending=False).reset_index(drop=True)


def sanity_checks(fan, cum_stats, rec):
    """[UNC-5] Mixture median vs 04 central: weaker than the reconstruction test."""
    rows = []
    cen = rec.set_index("year")["need"]
    worst = 0.0; worst_y = None
    for _, r in fan.iterrows():
        y = int(r["year"]); c = float(cen.get(y, np.nan))
        tot = r["median_need_young"] + (r["median_need_young"] * 0)  # placeholder
        rows.append({"year": y, "median_need_young": r["median_need_young"],
                     "central_need_total": c})
    cum_dev = cum_stats["p50"] / cum_stats["central_path_total"] - 1.0
    ok_cum = abs(cum_dev) <= ANCHOR_TOL_CUM
    print(f"[check] cumulative TOTAL median = {cum_stats['p50']:,.0f} vs 04 central "
          f"{cum_stats['central_path_total']:,.0f} ({cum_dev:+.1%}) -> "
          f"{'PASS' if ok_cum else 'WARN'}")
    print(f"[check]   this compares the TOTAL need across both channels, which is")
    print(f"[check]   the quantity 04 publishes. The deficit headline below is the")
    print(f"[check]   GRADUATE channel only  [UNC-7]")
    return pd.DataFrame(rows)


# ============================================================ FIGURES
def fig_fan(fan, degenerate):
    fig, ax = plt.subplots(figsize=(11, 6))
    max_w = float(fan["band_width"].max())
    if degenerate or max_w <= DEGENERATE_TOL:
        ax.plot(fan.year, fan.p50, "-o", color=INK, lw=2.4, ms=6,
                label="deficit (no dispersion -- see the diagnostic)")
        ax.text(0.5, 0.5,
                "DEGENERATE SIMULATION\nP10 = P50 = P90 in every year.",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color=SHORT, bbox=dict(boxstyle="round,pad=0.6", fc="#fdecec",
                                       ec=SHORT, alpha=0.95))
    else:
        ax.fill_between(fan.year, fan.p10, fan.p90, color=BAND, alpha=0.20, label="P10-P90")
        ax.fill_between(fan.year, fan.p25, fan.p75, color=BAND, alpha=0.38, label="P25-P75")
        ax.plot(fan.year, fan.p50, "-o", color=INK, lw=2.2, ms=5, label="median")
        n_und = int(fan["zero_in_p10_p90"].sum())
        ax.text(0.99, 0.02, f"zero lies inside P10-P90 in {n_und}/{len(fan)} years",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color="0.35")
    ax.axhline(0, color=SHORT, lw=1.3, ls="--", label="balance (0)")
    ax.set_title("GRADUATE-channel deficit: distribution by year\n"
                 "(need_young - pool x rate)", fontweight="bold", fontsize=11)
    ax.set_xlabel("year"); ax.set_ylabel("teachers   >0 = shortage / <0 = surplus")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.text(0.5, 0.005,
             "This is the channel the graduate pool constrains. The lateral channel is "
             "modelled separately [UNC-7].",
             ha="center", fontsize=8, color="0.45")
    fig.tight_layout(rect=[0, 0.03, 1, 1]); out = IMG / "deficit_fan.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_channels(fan, cv_lateral=np.nan, lat_meta=None):
    """[UNC-7][UNC-11] The two channels side by side, and what netting conceals.

    [UNC-11] Both channels now carry a P10-P90 band. Previously the lateral line
    was a single deterministic path, which read as certainty about precisely the
    channel the block identifies as binding.
    """
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.fill_between(fan.year, fan.p10, fan.p90, color=RECRUIT, alpha=0.13,
                    label="graduate P10-P90")
    if {"lat_p10", "lat_p90"} <= set(fan.columns):
        ax.fill_between(fan.year, fan.lat_p10, fan.lat_p90, color=LATERAL,
                        alpha=0.15, label="lateral P10-P90 [UNC-11]")
    ax.plot(fan.year, fan.p50, "-o", color=RECRUIT, lw=2.4, ms=5,
            label="graduate channel (median)")
    ax.plot(fan.year, fan.median_gap_lateral, "-s", color=LATERAL, lw=2.4, ms=5,
            label="lateral channel (median)")
    ax.plot(fan.year, fan.median_gap_no_offset, "-^", color=SHORT, lw=2.0, ms=5,
            label="sum per channel, no cross-credit [GAP-29]")
    ax.plot(fan.year, fan.median_gap_netted, ":", color=INK, lw=1.8,
            label="netted (what offsetting reports)")
    ax.axhline(0, color="0.4", lw=1.2, ls="--")
    ax.set_title("Deficit by channel: the graduate pool is not the binding constraint",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers   >0 = shortage / <0 = surplus")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    if (fan.p50 < 0).any() and (fan.median_gap_lateral > 0).any():
        ax.text(0.02, 0.03,
                "the two channels have opposite signs:\n"
                "netting credits one against the other [GAP-29]",
                transform=ax.transAxes, fontsize=8.5, color=SHORT, va="bottom")
    if np.isfinite(cv_lateral):
        src = (lat_meta or {}).get("source", "default")
        if src == "default":
            foot = (f"The lateral band uses an ASSUMED CV of {cv_lateral:.0%}: "
                    f"04's lateral_channel_history.csv was not found [UNC-11].")
        else:
            foot = (f"The lateral band uses a CV of {cv_lateral:.0%}, estimated from "
                    f"{(lat_meta or {}).get('n', 0)} observed years of the reserve's "
                    f"depletion rate [UNC-11].")
        fig.text(0.5, 0.005, foot, ha="center", fontsize=8, color="0.45")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    out = IMG / "deficit_by_channel.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_probability(fan, share_clipped):
    """[UNC-8] Both the clipped and the unclipped probability."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pct = 100 * fan.prob_shortage
    ax.plot(fan.year, pct, "-o", color=SHORT, lw=2.4, ms=6, label="with the 04 ceiling")
    ax.fill_between(fan.year, 0, pct, color=SHORT, alpha=0.08)
    if "prob_shortage_unclipped" in fan.columns:
        pu = 100 * fan.prob_shortage_unclipped
        if float((pct - pu).abs().max()) > 0.5:
            ax.plot(fan.year, pu, "--s", color=MID, lw=2.0, ms=5,
                    label="without the ceiling [UNC-8]")
    ax.axhline(50, color="0.5", lw=1, ls=":")
    allv = pd.concat([pct] + ([100 * fan.prob_shortage_unclipped]
                              if "prob_shortage_unclipped" in fan.columns else []))
    lo = max(0, float(allv.min()) - 12); hi = min(100, float(allv.max()) + 12)
    ax.set_ylim(lo, hi)
    if lo <= 50 <= hi:
        ax.text(fan.year.min(), 50, " coin flip", fontsize=8.5, color="0.45", va="bottom")
    ax.set_title("Probability of a GRADUATE-channel shortage (deficit > 0)",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("probability (%)")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    if share_clipped > CEILING_WARN_SHARE:
        fig.text(0.5, 0.005,
                 f"{share_clipped:.1%} of sampled rates were clipped at the ceiling. "
                 f"Truncating entries from above only\nshifts the deficit rightwards, so "
                 f"part of the solid line is construction [UNC-8].",
                 ha="center", fontsize=8, color="0.45")
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.tight_layout()
    out = IMG / "deficit_probability.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_cumulative(cum, stats):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng_ = float(np.max(cum) - np.min(cum))
    if rng_ <= DEGENERATE_TOL:
        v = float(np.median(cum))
        ax.axvline(v, color=INK, lw=3)
        ax.set_xlim(v * 0.9, v * 1.1); ax.set_ylim(0, 1)
        ax.text(v, 0.55, f"  all {len(cum):,} draws = {v:,.0f}", ha="left", va="center",
                fontsize=11, color=INK)
        ax.set_yticks([])
    else:
        lo, hi = np.percentile(cum, [0.5, 99.5])
        ax.hist(cum[(cum >= lo) & (cum <= hi)], bins=45, color=BAND, alpha=0.75,
                edgecolor="white")
        ax.set_xlim(lo, hi); top = ax.get_ylim()[1]
        for q, lab, col in [(stats["p10"], "P10", "0.4"), (stats["p50"], "P50", INK),
                            (stats["p90"], "P90", "0.4")]:
            ax.axvline(q, color=col, lw=1.8, ls="--")
            ax.text(q, top * 0.95, f" {lab}={q:,.0f}", rotation=90, va="top", ha="right",
                    fontsize=8, color=col)
        cp = stats.get("central_path_total")
        if cp:
            ax.axvline(cp, color=SHORT, lw=1.8)
            ax.text(cp, top * 0.55, f" 04_gap central = {cp:,.0f}", rotation=90,
                    va="center", ha="right", fontsize=8, color=SHORT)
    ax.set_title("Cumulative recruitment need 2025-2040 -- distribution (both channels)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("new teachers (horizon total)"); ax.set_ylabel("frequency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); out = IMG / "cumulative_recruitment.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def _one_variance_panel(ax, contrib, xlabel, highlight=None):
    c = contrib.dropna(subset=["contribution_pct"]).sort_values("contribution_pct")
    if c.empty or float(contrib["total_variance"].iloc[0]) <= DEGENERATE_TOL:
        ax.text(0.5, 0.5, "Total variance is zero.\nNo decomposition is possible.",
                transform=ax.transAxes, ha="center", va="center", fontsize=11,
                color=SHORT)
        ax.set_xticks([]); ax.set_yticks([])
        return
    colours = []
    for axis_name, v in zip(c["axis"], c["contribution_pct"]):
        if highlight and axis_name == highlight:
            colours.append(LATERAL)
        elif v >= 0.4:
            colours.append(SHORT)
        else:
            colours.append(MID)
    ax.barh(c["axis"], 100 * c["contribution_pct"], color=colours, alpha=0.9)
    for i, v in enumerate(c["contribution_pct"]):
        ax.text(100 * v, i, f" {100*v:.0f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.2, axis="x")
    ax.spines[["top", "right"]].set_visible(False)


def fig_variance(contrib, rate_meta, contrib_comb=None, lat_meta=None):
    """[UNC-11] Two panels: the graduate gap, and the combined gap where the
    lateral reserve actually lives."""
    if contrib_comb is None:
        fig, ax1 = plt.subplots(figsize=(10, 5)); ax2 = None
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.4))

    _one_variance_panel(ax1, contrib,
                        "share of the GRADUATE-channel deficit's variance (%)")
    ax1.set_title("Graduate channel", fontweight="bold", fontsize=11)
    note = ("one-at-a-time; shares are normalised and do not\n"
            "form an exact variance partition [UNC-2]")
    if rate_meta and np.isfinite(rate_meta.get("cv_raw", np.nan)):
        note += (f"\nrate CV = {rate_meta['cv_used']:.0%} "
                 f"('{rate_meta['mode']}'); the raw trending series "
                 f"gives {rate_meta['cv_raw']:.0%} [UNC-10]")
    note += "\nthe lateral reserve does not enter this gap [UNC-11]"
    ax1.text(0.99, 0.03, note, transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=8, color="0.45")

    if ax2 is not None:
        _one_variance_panel(ax2, contrib_comb,
                            "share of the COMBINED gap's variance (%)",
                            highlight="lateral_reserve")
        ax2.set_title("Both channels, no cross-credit [GAP-29]",
                      fontweight="bold", fontsize=11)
        n2 = "the lateral reserve enters here [UNC-11]"
        if lat_meta:
            if lat_meta.get("source") == "default":
                n2 += f"\nits CV is ASSUMED at {lat_meta['cv_used']:.0%}"
            else:
                n2 += (f"\nits CV = {lat_meta['cv_used']:.0%}, estimated from "
                       f"{lat_meta.get('n', 0)} observed years")
        ax2.text(0.99, 0.03, n2, transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=8, color="0.45")

    fig.suptitle("Where the uncertainty comes from (variance decomposition)",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); out = IMG / "variance_decomposition.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_demo(sim, years):
    picked = sim["picked"]; d = sim["gap_y"]
    scns = [s for s in ["central", "high", "low"] if s in set(picked)]
    if len(scns) < 2:
        print("[fig] demographic scenarios: fewer than two -> skipped"); return None
    rows = []
    for s in scns:
        sel = (picked == s)
        if sel.sum() < 20:
            continue
        med = np.median(d[sel, :], axis=0)
        rows += [{"scenario": s, "year": int(y), "median_deficit": float(med[j])}
                 for j, y in enumerate(years)]
    tab = pd.DataFrame(rows)
    if tab.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 6))
    for s in tab.scenario.unique():
        g = tab[tab.scenario == s].sort_values("year")
        ax.plot(g.year, g.median_deficit, "-o", lw=2.2, ms=5,
                color=DEMO_COLORS.get(s, INK), label=f"{s} scenario")
    ax.axhline(0, color=SHORT, lw=1.3, ls="--")
    ax.set_title("Median GRADUATE-channel deficit by demographic scenario",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers   >0 = shortage / <0 = surplus")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); out = IMG / "deficit_by_demographic_scenario.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)
    return tab


def fig_rho(tab, degenerate, rate_share_baseline=np.nan):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.4))
    fig.subplots_adjust(wspace=0.30)
    flat = float(tab.rel_width.max() - tab.rel_width.min()) <= DEGENERATE_TOL

    ax1.plot(tab.rho, tab.p50, "-o", color=INK, lw=2.2, ms=6, label="P50")
    if not flat:
        ax1.fill_between(tab.rho, tab.p10, tab.p90, color=BAND, alpha=0.25, label="P10-P90")
    ax1.set_title("Cumulative need vs rho", fontweight="bold")
    ax1.set_xlabel("rho"); ax1.set_ylabel("new teachers 2025-2040")
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False)

    ax2.plot(tab.rho, tab.rel_width, "-o", color=MID, lw=2.2, ms=6)
    ax2.set_title("Relative width of the cumulative band", fontweight="bold")
    ax2.set_xlabel("rho"); ax2.set_ylabel("(P90-P10)/P50")
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    if flat:
        ax2.set_ylim(-0.05, 0.55)
        ax2.text(0.5, 0.5, "rho cannot matter:\nthe need shock has zero dispersion",
                 transform=ax2.transAxes, ha="center", va="center",
                 fontsize=10.5, color=SHORT,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#fdecec", ec=SHORT, alpha=0.9))

    ax3.plot(tab.rho, 100 * tab.rate_share, "-o", color=SHORT, lw=2.2, ms=6)
    ax3.set_ylim(0, 100)
    ax3.set_title("Conversion-rate share of variance", fontweight="bold")
    ax3.set_xlabel("rho"); ax3.set_ylabel("share (%)")
    ax3.grid(alpha=0.25); ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Rho sensitivity -- rho is an assumption, not an estimate",
                 fontweight="bold", y=1.02)
    note = ("rho splits the NEED shock into common and idiosyncratic parts. The rate and "
            "pool shocks are drawn once per path and held across years,\nwhich is rho = 1 "
            "for those axes by construction, so rho can only move the need share of the "
            "variance [UNC-6].")
    if np.isfinite(rate_share_baseline):
        note += (f" At the baseline rho the conversion rate alone carries "
                 f"{100*rate_share_baseline:.0f}% of it.")
    fig.text(0.5, -0.04, note, ha="center", fontsize=8.5, color="0.45")
    out = IMG / "rho_sensitivity.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


# ============================================================ MAIN
def main():
    print("=" * 78)
    print("05_uncertainty -- Monte Carlo around the 04_gap pipeline deficit")
    print(f"                 need basis: '{NEED_BASIS}' | rate dispersion: "
          f"'{RATE_DISPERSION_MODE}'")
    print("=" * 78)

    rec = load_recruitment_central()
    if rec is None:
        return
    channels = load_channel_need()                    # [UNC-7]
    if channels is None and NEED_BASIS == "channel":
        print("\n[UNC-7] >>> NO CHANNEL SPLIT AVAILABLE. Running on the TOTAL need would")
        print("[UNC-7] >>> set it against graduate-only supply, which is the perimeter")
        print("[UNC-7] >>> error this revision removes. Run 04_gap first.")
        return

    pool = load_pool()
    (rate_path, conv_hist, hist_years, rate_mode,
     rate_ceiling, cohort_ceiling, roll) = load_conversion()
    if pool is None or rate_path is None:
        print("[error] missing graduate pool or conversion-rate path -> cannot run")
        return
    anchor = load_anchor()                            # [UNC-9]
    cv_rate, rate_meta = rate_dispersion(conv_hist, hist_years)   # [UNC-10]
    cv_lateral, lat_meta = lateral_dispersion()                   # [UNC-11]
    demand_piv = load_demand_levels()

    years = sorted(set(rec["year"]).intersection(set(pool["year"])))
    if channels is not None:
        years = sorted(set(years).intersection(set(channels["year"])))
    rec = rec[rec.year.isin(years)].sort_values("year").reset_index(drop=True)
    pool = pool[pool.year.isin(years)].sort_values("year").reset_index(drop=True)
    if channels is not None:
        channels = channels[channels.year.isin(years)].sort_values("year")
        channels = channels.reset_index(drop=True)
    sig, bad, used_mc = load_sigma(years)

    needs, needs_y, needs_l, scns, n_floored, yshare = build_scenario_needs(
        rec, demand_piv, channels)
    pd.DataFrame({"year": years,
                  **{f"total_{s}": needs[s] for s in scns},
                  **{f"young_{s}": needs_y[s] for s in scns},
                  **{f"lateral_{s}": needs_l[s] for s in scns}}
                 ).to_csv(RES / "scenario_needs.csv", index=False)

    recon = check_reconstruction(rec, needs)
    if recon is not None:
        recon.to_csv(RES / "reconstruction_check.csv", index=False)

    issues = diagnose_inputs(sig, cv_rate, needs, scns, channels,
                             cv_lateral=cv_lateral, lat_meta=lat_meta)

    feas_l = (channels["feasible_lateral"].values.astype(float)
              if (channels is not None and channels["feasible_lateral"].notna().all())
              else None)

    sim = simulate(RHO, rec, needs_y, needs_l, feas_l, scns, pool, sig, rate_path,
                   cv_rate, years, rate_ceiling, cv_lateral=cv_lateral)
    fan, cum, cum_stats = summarise(sim, rec, years)
    checks = sanity_checks(fan, cum_stats, rec)
    contrib = variance_decomposition(sim, years, target="graduate")
    # [UNC-11] the lateral reserve only enters the COMBINED gap, so it needs its
    # own decomposition. On a run where the lateral channel binds, this is the
    # one that answers "where should uncertainty be reduced".
    contrib_comb = variance_decomposition(sim, years, target="combined")

    degenerate = float(fan["band_width"].max()) <= DEGENERATE_TOL

    print(f"\n[setup] draws={N_DRAWS} | rate CV={sim['cv_rate']:.1%} "
          f"('{rate_meta['mode']}') | rate mode='{rate_mode}'")
    print(f"[setup] need shock: rho={RHO}, sigma capped at {SIGMA_CAP}")
    print(f"[setup] rate ceiling = {rate_ceiling:.2%} "
          f"({cohort_ceiling:.0%} cohort conversion over {roll} cohorts)")
    print(f"[setup] rho applies to the NEED axis only; rate and pool shocks are")
    print(f"[setup] drawn once per path and held across years (rho = 1) [UNC-6]")

    # [UNC-8] price the one-sided truncation
    sc = sim["share_clipped"]
    if sc > 0:
        dmax = float((fan["prob_shortage"] - fan["prob_shortage_unclipped"]).abs().max())
        print(f"\n[UNC-8] {sc:.1%} of sampled rates exceeded the ceiling and were clipped.")
        print(f"[UNC-8] The ceiling truncates entries from ABOVE only, so the deficit")
        print(f"[UNC-8] distribution is pushed rightwards by construction.")
        print(f"[UNC-8]   P(shortage) with the ceiling    : "
              f"{fan['prob_shortage'].min():.1%} to {fan['prob_shortage'].max():.1%}")
        print(f"[UNC-8]   P(shortage) without the ceiling : "
              f"{fan['prob_shortage_unclipped'].min():.1%} to "
              f"{fan['prob_shortage_unclipped'].max():.1%}")
        print(f"[UNC-8]   worst-year difference           : {dmax:.1%}")
        if dmax > 0.02:
            print(f"[UNC-8]   >>> the ceiling is doing visible work. Report both.")

    print(f"\n[result] cumulative need 2025-2040 (BOTH channels): "
          f"P10={cum_stats['p10']:,.0f} | P50={cum_stats['p50']:,.0f} | "
          f"P90={cum_stats['p90']:,.0f}")
    print(f"         (04_gap central path = {cum_stats['central_path_total']:,.0f})")

    if degenerate:
        print("\n[DEGENERATE] the deficit distribution has ZERO width in every year.")
    else:
        n_und = int(fan["zero_in_p10_p90"].sum())
        print(f"\n[VERDICT] GRADUATE channel: zero inside P10-P90 in {n_und}/{len(fan)} years")
        if n_und == len(fan):
            print("[VERDICT] the sign of the graduate gap is NOT determined in any year.")
        elif n_und == 0:
            side = "SHORTAGE" if float(fan["p50"].median()) > 0 else "SURPLUS"
            print(f"[VERDICT] the sign is determined in every year: {side}.")
        else:
            det = fan.loc[fan["sign_determined"], "year"].tolist()
            print(f"[VERDICT] sign determined only in: {det}")

    p = fan["prob_shortage"]
    print(f"[result] P(graduate shortage): min={p.min():.1%} "
          f"({int(fan.loc[p.idxmin(),'year'])}) | max={p.max():.1%} "
          f"({int(fan.loc[p.idxmax(),'year'])}) | years above 50% = "
          f"{int((p > 0.5).sum())}/{len(p)}")

    # [UNC-7] the channel the constraint actually binds on
    med_y = float(np.median(sim["gap_y"]))
    med_l = float(np.median(sim["gap_l"]))
    med_no = float(np.median(sim["gap_no_offset"].sum(axis=1)))
    med_net = float(np.median(sim["gap_netted"].sum(axis=1)))
    print(f"\n[UNC-7] deficit by channel, median across draws:")
    print(f"[UNC-7]   graduate : {med_y:+,.0f}/yr")
    print(f"[UNC-7]   lateral  : {med_l:+,.0f}/yr")
    print(f"[UNC-7]   cumulative, no cross-credit : {med_no:+,.0f}  [GAP-29]")
    print(f"[UNC-7]   cumulative, netted          : {med_net:+,.0f}")
    if med_y < 0 and med_l > 0:
        print(f"[UNC-7]   >>> the binding channel is LATERAL, not the graduate pool.")
        print(f"[UNC-7]   >>> That channel is drawn from the contracted reserve, which")
        print(f"[UNC-7]   >>> 04 flags as unmeasured anywhere in this chain [GAP-8].")

    # [UNC-11] the reserve now has a distribution, so report it like the other one
    if "prob_shortage_lateral" in fan.columns:
        pl = fan["prob_shortage_lateral"]
        n_det_l = int(fan["lat_sign_determined"].sum())
        print(f"\n[UNC-11] LATERAL channel, with the reserve's own dispersion "
              f"(CV = {cv_lateral:.1%}):")
        print(f"[UNC-11]   P(lateral shortage): min={pl.min():.1%} "
              f"({int(fan.loc[pl.idxmin(),'year'])}) | max={pl.max():.1%} "
              f"({int(fan.loc[pl.idxmax(),'year'])})")
        print(f"[UNC-11]   years above 50%    : {int((pl > 0.5).sum())}/{len(pl)}")
        print(f"[UNC-11]   sign determined in : {n_det_l}/{len(fan)} years")
        if (lat_meta or {}).get("source") == "default":
            print(f"[UNC-11]   >>> the CV is ASSUMED, not estimated. Treat this")
            print(f"[UNC-11]   >>> probability as illustrative until 04 publishes")
            print(f"[UNC-11]   >>> lateral_channel_history.csv.")

    print("\n[result] variance decomposition (one-at-a-time, normalised):")
    tv = float(contrib["total_variance"].iloc[0])
    if tv <= DEGENERATE_TOL:
        print("    total variance is ZERO -> the decomposition is meaningless.")
    else:
        for _, row in contrib.iterrows():
            pct = row["contribution_pct"]
            bar = "#" * int(round(30 * pct)) if np.isfinite(pct) else ""
            print(f"    {row['axis']:<24} {pct:6.1%}  {bar}")
        print(f"         -> priority for reducing uncertainty: {contrib.iloc[0]['axis']}")
        print(f"         note: the axes are not independent, so these shares are a")
        print(f"         ranking, not an exact partition of the variance  [UNC-2]")
        print(f"         note: the lateral reserve does not enter the GRADUATE gap,")
        print(f"         so it is absent here by construction, not by omission [UNC-11]")

    # [UNC-11] the combined gap is where the lateral reserve lives
    print("\n[result] variance decomposition of the COMBINED gap "
          "(no cross-credit, [GAP-29]):")
    tvc = float(contrib_comb["total_variance"].iloc[0])
    if tvc <= DEGENERATE_TOL:
        print("    total variance is ZERO -> the decomposition is meaningless.")
    else:
        for _, row in contrib_comb.iterrows():
            pct = row["contribution_pct"]
            bar = "#" * int(round(30 * pct)) if np.isfinite(pct) else ""
            print(f"    {row['axis']:<24} {pct:6.1%}  {bar}")
        top = contrib_comb.iloc[0]["axis"]
        print(f"         -> priority for reducing uncertainty: {top}")
        lat_share = contrib_comb.loc[contrib_comb.axis == "lateral_reserve",
                                     "contribution_pct"]
        if not lat_share.empty and np.isfinite(lat_share.iloc[0]):
            print(f"         the lateral reserve carries {lat_share.iloc[0]:.1%} of it, "
                  f"at CV = {cv_lateral:.1%}")
            if lat_meta.get("source") == "default":
                print(f"         >>> that CV is an ASSUMPTION "
                      f"({LATERAL_CV_DEFAULT:.0%}), not an estimate [UNC-11]")

    if len(scns) > 1:
        tot = {s: float(np.nansum(needs[s])) for s in scns}
        share_demo = contrib.loc[contrib.axis == "demographic_scenario",
                                 "contribution_pct"].iloc[0]
        print(f"\n[demographic axis] cumulative need by scenario: " +
              " | ".join(f"{s}={tot[s]:,.0f}" for s in scns))
        print(f"[demographic axis] high-low spread = "
              f"{max(tot.values())-min(tot.values()):,.0f} teachers | "
              f"variance share = {share_demo:.1%}")

    fan.to_csv(RES / "deficit_fan.csv", index=False)
    fan[["year", "prob_shortage", "prob_shortage_unclipped"]].to_csv(
        RES / "deficit_probability.csv", index=False)
    pd.DataFrame([cum_stats]).to_csv(RES / "cumulative_recruitment.csv", index=False)
    pd.concat([contrib, contrib_comb], ignore_index=True).to_csv(
        RES / "variance_decomposition.csv", index=False)
    pd.DataFrame([lat_meta]).to_csv(RES / "lateral_dispersion.csv", index=False)
    checks.to_csv(RES / "sanity_checks.csv", index=False)
    pd.DataFrame([rate_meta]).to_csv(RES / "rate_dispersion.csv", index=False)
    pd.DataFrame([{"rate_ceiling": rate_ceiling, "cohort_conv_ceiling": cohort_ceiling,
                   "roll_grad_years": roll, "share_draws_clipped": sim["share_clipped"],
                   "need_basis": NEED_BASIS,
                   "rate_dispersion_mode": RATE_DISPERSION_MODE,
                   "rate_cv_used": cv_rate, "rate_cv_raw": rate_meta.get("cv_raw"),
                   "lateral_cv_used": cv_lateral,
                   "lateral_cv_raw": lat_meta.get("cv_raw"),
                   "lateral_cv_source": lat_meta.get("source"),
                   "lateral_cv_n_years": lat_meta.get("n"),
                   "lateral_modelled": sim.get("lateral_modelled"),
                   "anchor_factor": (anchor or {}).get("factor"),
                   "anchor_verdict_anchored": (anchor or {}).get("verdict_anchored"),
                   "anchor_verdict_unanchored": (anchor or {}).get("verdict_unanchored"),
                   "degenerate_axes": ",".join(issues) if issues else "",
                   "simulation_degenerate": bool(degenerate),
                   "negative_need_cells_floored": int(n_floored),
                   "rho_applies_to": "need_shock_only"}]
                 ).to_csv(RES / "ceiling_and_diagnostics.csv", index=False)

    fig_fan(fan, degenerate)
    fig_channels(fan, cv_lateral=cv_lateral, lat_meta=lat_meta)
    fig_probability(fan, sim["share_clipped"])
    fig_cumulative(cum, cum_stats)
    fig_variance(contrib, rate_meta, contrib_comb=contrib_comb, lat_meta=lat_meta)
    demo_tab = fig_demo(sim, years)
    if demo_tab is not None:
        demo_tab.to_csv(RES / "deficit_by_demographic_scenario.csv", index=False)

    if RUN_RHO_SENSITIVITY:
        print("\n" + "=" * 78)
        print("RHO SENSITIVITY -- rho is an assumption, not an estimate")
        print("=" * 78)
        rows = []
        for r in RHO_GRID:
            s_r = simulate(r, rec, needs_y, needs_l, feas_l, scns, pool, sig,
                           rate_path, cv_rate, years, rate_ceiling,
                           cv_lateral=cv_lateral)
            f_r, c_r, cs_r = summarise(s_r, rec, years)
            k_r = variance_decomposition(s_r, years, target="graduate")
            rate_share = float(k_r.loc[k_r.axis == "conversion_rate",
                                       "contribution_pct"].iloc[0])
            rows.append({"rho": r, "p10": cs_r["p10"], "p50": cs_r["p50"],
                         "p90": cs_r["p90"],
                         "rel_width": (cs_r["p90"] - cs_r["p10"]) / max(cs_r["p50"], 1.0),
                         "prob_min": f_r["prob_shortage"].min(),
                         "prob_max": f_r["prob_shortage"].max(),
                         "years_above_50": int((f_r["prob_shortage"] > 0.5).sum()),
                         "sign_determined_years": int(f_r["sign_determined"].sum()),
                         "rate_share": rate_share if np.isfinite(rate_share) else 0.0})
        tab = pd.DataFrame(rows)
        tab.to_csv(RES / "rho_sensitivity.csv", index=False)

        print(f"{'rho':>7}{'P10':>12}{'P50':>12}{'P90':>12}{'rel.width':>12}")
        for _, t in tab.iterrows():
            mark = "  <- baseline" if abs(t.rho - RHO) < 1e-9 else ""
            print(f"{t.rho:>7.1f}{t.p10:>12,.0f}{t.p50:>12,.0f}{t.p90:>12,.0f}"
                  f"{t.rel_width:>12.3f}{mark}")

        flat = float(tab.rel_width.max() - tab.rel_width.min()) <= DEGENERATE_TOL
        print("\n[rho verdict]")
        if flat:
            print("    rho has NO effect: the need shock carries no dispersion.")
        elif (tab.sign_determined_years == len(fan)).all():
            print("    the sign is determined for every rho in the grid.")
        elif (tab.sign_determined_years == 0).all():
            print("    the sign is undetermined for every rho -> robust conclusion.")
        else:
            print("    the verdict changes with rho; state the assumed value prominently.")
        base_rate_share = float(contrib.loc[contrib.axis == "conversion_rate",
                                            "contribution_pct"].iloc[0])
        print(f"    rho reaches the NEED axis only; at rho={RHO} the conversion rate "
              f"alone carries")
        print(f"    {base_rate_share:.0%} of the variance, which rho cannot touch [UNC-6]")
        fig_rho(tab, flat, rate_share_baseline=base_rate_share)

    # ---------------------------------------------------- caveats
    print("\n" + "=" * 78)
    print("CAVEATS")
    print("=" * 78)
    print(f"  1. [UNC-7] The headline deficit is the GRADUATE channel: need_young")
    print(f"     against pool x rate. The lateral channel is reported beside it and")
    print(f"     the two are NOT netted [GAP-29].")
    if NEED_BASIS != "channel":
        print(f"     >>> NEED_BASIS is '{NEED_BASIS}'. The total need is being set")
        print(f"     >>> against graduate-only supply. NOT reportable.")
    if anchor:
        print(f"  2. [UNC-9] Every rate here is ANCHORED: 04 scaled entries by "
              f"{anchor['factor']:.3f}")
        print(f"     before forming the ratio. The whole distribution is conditional on")
        print(f"     that external level.")
        if (anchor.get("verdict_anchored") and anchor.get("verdict_unanchored")
                and anchor["verdict_anchored"] != anchor["verdict_unanchored"]):
            print(f"     >>> 04 found the verdict MOVES without the anchor "
                  f"('{anchor['verdict_anchored']}' vs")
            print(f"     >>> '{anchor['verdict_unanchored']}'). Report that alongside "
                  f"any probability here.")
    else:
        print("  2. [UNC-9] anchor_robustness.csv not found; the anchor's effect on")
        print("     these draws is UNTESTED.")
    print(f"  3. [UNC-8] {sim['share_clipped']:.1%} of sampled rates were clipped at the "
          f"{rate_ceiling:.2%}")
    print(f"     ceiling, from above only. Both probabilities are in "
          f"deficit_probability.csv.")
    print(f"  4. [UNC-10] The rate dispersion is the residual around a fitted trend "
          f"({cv_rate:.0%}),")
    if np.isfinite(rate_meta.get("cv_raw", np.nan)):
        print(f"     not the raw CV of a trending series ({rate_meta['cv_raw']:.0%}), "
              f"which would")
        print(f"     measure the climb rather than the scatter around today's level.")
    src = lat_meta.get("source", "default")
    if src == "default":
        print(f"  5. [UNC-11] The lateral reserve's CV is ASSUMED at {cv_lateral:.0%}: "
              f"04's")
        print(f"     lateral_channel_history.csv was not found. The reserve is the")
        print(f"     binding channel here, so that assumption is load-bearing.")
    else:
        print(f"  5. [UNC-11] The lateral reserve carries a CV of {cv_lateral:.0%}, "
              f"estimated from")
        print(f"     {lat_meta.get('n', 0)} observed years of its depletion rate. It is a "
              f"depletion")
        print(f"     assumption, not a measure of reserve capacity: 04 flags the reserve")
        print(f"     as unmeasured anywhere in the chain [GAP-8].")
    print("  6. This uncertainty is conditional on the model structure; it does not")
    print("     cover specification error (the observed-versus-normative ratio in 02,")
    print("     the perimeter gap against DGEEC, or the unmeasured substitution reserve).")
    if recon is not None:
        w = float(np.nanmax(np.abs(recon["rel_dev"])))
        if np.isfinite(w) and w > RECONSTRUCTION_TOL:
            print(f"  7. [UNC-5] The central need rebuilt here drifts up to {w:.0%} from")
            print(f"     the path 04 published.")
    print("=" * 78)
    print("Done. Results in:", RES, "| figures in:", IMG)
    print("=" * 78)


if __name__ == "__main__":
    main()
