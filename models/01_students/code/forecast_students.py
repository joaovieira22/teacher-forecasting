
from __future__ import annotations

from pathlib import Path
import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================ PATHS
BASE = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")


def _find_master(base: Path) -> Path:
    cands = [
        base / r"data\final\master_panel_nuts3.xlsx",
        base / r"data\master_panel_nuts3.xlsx",
        base / r"data\final\result\master_panel_nuts3.xlsx",
    ]
    for c in cands:
        if c.exists():
            return c
    data_dir = base / "data"
    if data_dir.exists():
        for c in data_dir.rglob("master_panel_nuts3.xlsx"):
            return c
    return cands[0]


MASTER = _find_master(BASE)
OUT_DIR = BASE / r"models\01_students\results"
IMG_DIR = BASE / r"models\01_students\images"

if os.environ.get("STUDENTS_TEST") == "1":
    MASTER = Path("master_panel_nuts3.xlsx")
    OUT_DIR = Path("stu_out")
    IMG_DIR = Path("stu_out")

OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "students_forecast_nuts3.xlsx"

# ============================================================ CONFIG
CYCLE_TO_BAND = {
    "students_pre_school":     "pop_0_4",
    "students_basic_1":        "pop_5_9",
    "students_basic_2":        "pop_10_14",
    "students_basic_3":        "pop_10_14",
    "students_secondary":      "pop_15_19",
    "students_post_secondary": "pop_15_19",
}
CYCLES = list(CYCLE_TO_BAND)
BANDS = sorted(set(CYCLE_TO_BAND.values()))

# "no_migration" was dropped: it is a counterfactual closure, not a plausible path, and
# it was widening the reported scenario range without informing any decision.
SCENARIOS = ["central", "high", "low"]
DROPPED_SCENARIOS = ["no_migration"]

DAMPING = 0.85
SHRINK_K = 3.0
LOOKBACK_SLOPE = 6

N_BOOT = 400
RANDOM_SEED = 20260818
RNG = np.random.default_rng(RANDOM_SEED)

SD_FLOOR = 0.01
SD_CAP = 0.20
SD_H_CAP = 0.45
DRAW_CLIP = 3.0
EPS_TRUNC = 2.0

# [FIX-3] share of the shock that is common across regions within a cycle-year.
# 0 reproduces the v1 behaviour (independent draws, collapsing aggregate band);
# 1 makes every region move together. 0.65 is a judgement call and is written to
# the summary sheet so it can be challenged.
COMMON_SHOCK_RHO = 0.65

# bounds on the behavioural drift in coverage; binding cases are reported.
COV_RATIO_LO, COV_RATIO_HI = 0.70, 1.40

BASE_OBS_YEAR = 2024   # last observed student year = level anchor

# [FIX-7] How to reconcile the historical population series with the forecast one.
#   "auto"           : splice if the two series overlap, otherwise step-estimate.
#   "spliced"        : force the splice (fails loudly if there is no overlap).
#   "hist_base"      : v2 behaviour, hist(base) denominator, carries any series break.
#   "forecast_base"  : v1 behaviour, first forecast year denominator, drops the
#                      base-year -> first-forecast-year demographic step entirely.
POP_LINK_MODE = "auto"

# [FIX-8] How to estimate the base_year -> first_forecast_year step when the two
# population series do not overlap. This step is NOT identified from the data; it is an
# assumption, and the choice matters.
#   "forecast_first_delta" : use the forecast series' own first year-on-year rate. One
#                            single migration assumption governs the whole path. Internally
#                            consistent, and scenario-specific.
#   "hist_trend"           : extrapolate the historical series' recent growth for one year,
#                            then switch to the forecast shape. Assumes the recent (in
#                            Portugal, immigration-driven) upswing runs one more year and
#                            then stops abruptly. This is the v3 behaviour.
#   "none"                 : step = 1.0. No demographic movement in the gap year.
STEP_MODE = "forecast_first_delta"

# years used by STEP_MODE="hist_trend"
STEP_LOOKBACK = 3

# a measured break beyond this is reported as a hard warning
BREAK_WARN = 0.01


# ============================================================ IO
def num(s):
    return pd.to_numeric(s, errors="coerce")


def load_master(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path, engine="openpyxl")
    sh = "master_all" if "master_all" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sh, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # [FIX-5] do not assume students_total exists
    if "data_scope" not in df.columns:
        marker = None
        for cand in ["students_total"] + CYCLES:
            if cand in df.columns:
                marker = cand
                break
        if marker is None:
            raise KeyError(
                "data_scope is absent and no student column was found to infer it from."
            )
        print(f"[io] data_scope missing; inferred from '{marker}'.")
        df["data_scope"] = np.where(df[marker].notna(), "historical", "forecast")

    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce").astype("Int64")
    for c in ["nuts2_code", "nuts3_code"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df


def _pop_of(row, band):
    """Population for a band, preferring the historical column when it is populated."""
    v = row.get("hist_" + band, np.nan)
    if pd.isna(v):
        v = row.get(band, np.nan)
    return v


# ============================================================ COVERAGE
def build_coverage(hist: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, r in hist.iterrows():
        for cyc, band in CYCLE_TO_BAND.items():
            pop = _pop_of(r, band)
            st = r.get(cyc, np.nan)
            if pd.isna(pop) or float(pop) <= 0 or pd.isna(st):
                continue
            recs.append({
                "nuts3_code": r["nuts3_code"],
                "nuts3_nome": r.get("nuts3_nome", ""),
                "nuts2_code": r.get("nuts2_code", ""),
                "year_start": int(r["year_start"]),
                "cycle": cyc,
                "students": float(st),
                "pop": float(pop),
                "coverage": float(st) / float(pop),
            })
    cov = pd.DataFrame(recs)
    if cov.empty:
        raise RuntimeError("Coverage panel is empty; check population bands and student columns.")
    cov["log_cov"] = np.log(cov["coverage"].clip(lower=1e-6))
    return cov


# ============================================================ PROJECTION CORE
def weighted_recent_slope(years, y, lookback):
    """Exponentially weighted slope over the last `lookback` points.

    Returns (slope, level_at_last_year, intercept_at_last_year_from_the_fit).
    The third element is new: it is the fitted value at x=0 rather than the raw
    last observation, and it is what [FIX-2] needs for honest residuals.
    """
    years = np.asarray(years, float)
    y = np.asarray(y, float)
    ok = np.isfinite(years) & np.isfinite(y)
    years, y = years[ok], y[ok]
    if len(years) == 0:
        return 0.0, np.nan, np.nan
    if len(years) < 3:
        return 0.0, float(y[-1]), float(y[-1])

    yy = years[-lookback:]
    vv = y[-lookback:]
    x = yy - yy.max()
    w = np.exp(0.25 * x)
    w = w / w.sum()
    xb = (w * x).sum()
    yb = (w * vv).sum()
    den = (w * (x - xb) ** 2).sum()
    slope = 0.0 if den <= 0 else float((w * (x - xb) * (vv - yb)).sum() / den)
    intercept = float(yb - slope * xb)          # fitted value at x = 0
    return slope, float(vv[-1]), intercept


def level_project(level, slope, future_years, base_year, phi):
    out = []
    for fy in future_years:
        h = fy - base_year
        cum = sum((phi ** kk) * slope for kk in range(1, h + 1))
        out.append(level + cum)
    return np.array(out)


def project_series(level, slope, horizon, phi):
    out, cum = [], 0.0
    for k in range(1, horizon + 1):
        cum += (phi ** k) * slope
        out.append(level + cum)
    return np.array(out)


def fit_cycle_projection(cov_cycle, future_years, method="damped",
                         phi=DAMPING, k=SHRINK_K, lookback=LOOKBACK_SLOPE):
    regions = cov_cycle["nuts3_code"].unique()
    slopes, levels, nobs, last_year = {}, {}, {}, {}

    for rc in regions:
        d = cov_cycle[cov_cycle["nuts3_code"] == rc].sort_values("year_start")
        y = d["log_cov"].values
        s, lv, _ = weighted_recent_slope(d["year_start"].values, y, lookback)
        if method == "median3":
            lv = float(np.median(y[-3:]))
        elif method in ("last", "rw"):
            lv = float(y[-1])
        slopes[rc] = s
        levels[rc] = lv
        nobs[rc] = len(d)
        last_year[rc] = int(d["year_start"].max())

    cyc_slope = float(np.median(list(slopes.values()))) if slopes else 0.0
    cyc_level = float(np.median(list(levels.values()))) if levels else 0.0

    proj = {}
    for rc in regions:
        n = nobs[rc]
        w = n / (n + k)
        base = last_year[rc]
        if method == "damped":
            s_sh = w * slopes[rc] + (1 - w) * cyc_slope
            vals = level_project(levels[rc], s_sh, future_years, base, phi)
        else:
            lv_sh = w * levels[rc] + (1 - w) * cyc_level
            vals = np.repeat(lv_sh, len(future_years))
        proj[rc] = dict(zip(future_years, vals))
    return proj, cyc_slope


# ============================================================ BACKTEST
def mase(yt, yp, ytr):
    yt = np.asarray(yt, float)
    yp = np.asarray(yp, float)
    ytr = np.asarray(ytr, float)
    d = np.mean(np.abs(np.diff(ytr))) if len(ytr) >= 2 else np.nan
    if not np.isfinite(d) or d <= 0:
        return np.nan
    return float(np.mean(np.abs(yt - yp)) / d)


def bench(kind, v, h):
    v = np.asarray(v, float)
    if kind in ("last", "rw"):
        return np.repeat(v[-1], h)
    if kind == "median3":
        return np.repeat(np.median(v[-3:]), h)
    if kind == "damped":
        s, lv, _ = weighted_recent_slope(np.arange(len(v)), v, LOOKBACK_SLOPE)
        return project_series(lv, s, h, DAMPING)
    return np.repeat(v[-1], h)


def rolling_origin(cov, min_train=4):
    # [FIX-6] "rw" was an alias of "last"; scoring both double-counted the baseline.
    methods = ["last", "median3", "damped"]
    rows = []
    for cyc in CYCLES:
        cc = cov[cov.cycle == cyc]
        for rc in cc.nuts3_code.unique():
            d = cc[cc.nuts3_code == rc].sort_values("year_start")
            y = d.log_cov.values
            if len(y) < min_train + 1:
                continue
            for o in range(min_train, len(y)):
                tr, te = y[:o], y[o:]
                for m in methods:
                    sc = mase(te, bench(m, tr, len(te)), tr)
                    if np.isfinite(sc):
                        rows.append({"cycle": cyc, "method": m, "mase": sc})
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev, {}
    tab = ev.groupby(["cycle", "method"], as_index=False)["mase"].mean()
    best = tab.sort_values(["cycle", "mase"]).groupby("cycle").first()["method"].to_dict()
    return tab, best


def region_residual_std(d):
    """[FIX-2] residuals against the fitted line, not against the last observation."""
    d = d.sort_values("year_start")
    y = d.log_cov.values
    if len(y) < 4:
        return 0.05
    s, _lv, intercept = weighted_recent_slope(d.year_start.values, y, LOOKBACK_SLOPE)
    x = d.year_start.values - d.year_start.values.max()
    res = y - (intercept + s * x)
    sd = np.std(res, ddof=1) if len(res) > 2 else 0.05
    if not np.isfinite(sd):
        sd = 0.05
    return float(min(max(sd, SD_FLOOR), 0.25))


# ============================================================ NUTS2 GROWTH FACTORS
def _nuts2_band_table(frame, bands, by_scenario):
    """NUTS2 population by band and year, summed from NUTS3."""
    keys = (["scenario"] if by_scenario else []) + ["nuts2_code", "year_start"]
    rows = []
    for _, r in frame.iterrows():
        rec = {k: r.get(k) for k in keys}
        for b in bands:
            rec[b] = _pop_of(r, b) if not by_scenario else r.get(b, np.nan)
        rows.append(rec)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t["year_start"] = pd.to_numeric(t["year_start"], errors="coerce").astype("Int64")
    t = t.dropna(subset=["year_start"])
    t["year_start"] = t["year_start"].astype(int)
    return t.groupby(keys, as_index=False)[bands].sum(min_count=1)


def nuts2_growth_factors(fut, hist, base_year, mode=POP_LINK_MODE):
    """[FIX-7] Demographic growth on a single, linked population level.

    The target is  growth(year) = pop(year) / pop(base_year)  with numerator and
    denominator on the SAME level. The forecast frame and the historical frame are
    different vintages, so their levels need not agree. Three regimes:

      overlap present  -> splice. lambda = pop_hist(t*) / pop_fc(t*) at the latest
                          overlapping year, applied to the whole forecast path, so the
                          ratio is taken within one level.
      no overlap       -> the base_year -> first_forecast_year step is not identified
                          from the data. It is estimated from the historical series'
                          own recent growth and the forecast series contributes only
                          its internal shape.
      hist_base        -> v2. Kept so the size of the break can be reproduced.
      forecast_base    -> v1. Kept for the same reason.

    Returns (growth, first_fy, diagnostics).
    """
    present = [b for b in BANDS if b in fut.columns]
    missing = [b for b in BANDS if b not in fut.columns]
    if missing:
        print(f"[growth] WARNING population bands absent from the forecast frame: {missing}")
    if not present:
        raise RuntimeError("No population bands available in the forecast frame.")

    fc = _nuts2_band_table(fut, present, by_scenario=True)
    hs = _nuts2_band_table(hist, present, by_scenario=False)
    if fc.empty or hs.empty:
        raise RuntimeError("Could not build NUTS2 population tables.")

    first_fy = int(fc["year_start"].min())
    fc_years = set(fc["year_start"].unique())
    hs_years = set(hs["year_start"].unique())
    overlap = sorted(fc_years & hs_years)

    hs_i = hs.set_index(["nuts2_code", "year_start"])
    fc_i = fc.set_index(["scenario", "nuts2_code", "year_start"])

    # ---- measure the raw break the v2 denominator would have imposed
    break_rows = []
    for n2 in hs["nuts2_code"].unique():
        for b in present:
            hb = hs_i.at[(n2, base_year), b] if (n2, base_year) in hs_i.index else np.nan
            fb = fc_i.at[("central", n2, first_fy), b] if ("central", n2, first_fy) in fc_i.index else np.nan
            if pd.notna(hb) and pd.notna(fb) and hb > 0:
                break_rows.append({"nuts2_code": n2, "band": b,
                                   "hist_base": float(hb), "fc_first": float(fb),
                                   "ratio_fc_over_hist": float(fb) / float(hb)})
    brk = pd.DataFrame(break_rows)

    # ---- choose the regime
    chosen = mode
    if mode == "auto":
        chosen = "spliced" if overlap else "step_estimated"
    elif mode == "spliced" and not overlap:
        raise RuntimeError("POP_LINK_MODE='spliced' but the two population series do not overlap.")

    print(f"[growth] historical years {min(hs_years)}-{max(hs_years)} | "
          f"forecast years {first_fy}-{max(fc_years)} | overlap {overlap if overlap else 'none'}")
    if not brk.empty:
        med = brk.groupby("band")["ratio_fc_over_hist"].median()
        print(f"[growth] forecast({first_fy}) / historical({base_year}) by band, median across NUTS2:")
        for b, v in med.items():
            flag = "  <-- break" if abs(v - 1.0) > BREAK_WARN else ""
            print(f"           {b:<12} {v:.4f}{flag}")
        worst = float((med - 1.0).abs().max())
        if worst > BREAK_WARN:
            print(f"[growth] WARNING the two population vintages differ in level by up to "
                  f"{100 * worst:.2f}%. Under mode 'hist_base' that difference would enter the")
            print(f"[growth]         projection as if it were demography. Mode in use: '{chosen}'.")
    print(f"[growth] linking mode: {chosen}")

    # ---- splice factors
    lam = {}
    t_star = max(overlap) if overlap else None
    if chosen == "spliced":
        for (scen, n2, yr) in fc_i.index:
            if yr != t_star:
                continue
            for b in present:
                hv = hs_i.at[(n2, t_star), b] if (n2, t_star) in hs_i.index else np.nan
                fv = fc_i.at[(scen, n2, t_star), b]
                lam[(scen, n2, b)] = (float(hv) / float(fv)) if (pd.notna(hv) and pd.notna(fv)
                                                                 and fv > 0) else np.nan

    # ---- step factors when there is no overlap
    step = {}
    if chosen == "step_estimated":
        gap = max(0, first_fy - base_year)
        scen_list = sorted(fc["scenario"].dropna().unique())
        for n2 in hs["nuts2_code"].unique():
            sub = hs[hs.nuts2_code == n2].sort_values("year_start")
            for b in present:
                if STEP_MODE == "none":
                    g = 1.0
                    for sc_ in scen_list:
                        step[(sc_, n2, b)] = 1.0
                    continue
                if STEP_MODE == "hist_trend":
                    v = pd.to_numeric(sub[b], errors="coerce").dropna().values
                    if len(v) >= 2:
                        k = min(STEP_LOOKBACK, len(v) - 1)
                        g = (v[-1] / v[-1 - k]) ** (1.0 / k) if v[-1 - k] > 0 else 1.0
                    else:
                        g = 1.0
                    for sc_ in scen_list:
                        step[(sc_, n2, b)] = float(g) ** gap
                    continue
                # forecast_first_delta: scenario-specific, uses the forecast's own rate
                for sc_ in scen_list:
                    k0 = (sc_, n2, first_fy)
                    k1 = (sc_, n2, first_fy + 1)
                    if k0 in fc_i.index and k1 in fc_i.index:
                        v0, v1 = fc_i.at[k0, b], fc_i.at[k1, b]
                        g = (float(v1) / float(v0)) if (pd.notna(v0) and pd.notna(v1)
                                                        and v0 > 0) else 1.0
                    else:
                        g = 1.0
                    step[(sc_, n2, b)] = float(g) ** gap
        if chosen == "step_estimated":
            vals = [v for v in step.values() if np.isfinite(v)]
            if vals:
                print(f"[growth] STEP_MODE='{STEP_MODE}' | base-year step, median across "
                      f"NUTS2/bands: {np.median(vals):.4f} "
                      f"(range {np.min(vals):.4f}-{np.max(vals):.4f})")

    # ---- assemble
    growth, unresolved = {}, 0
    for (scen, n2, yr), row in fc_i.iterrows():
        for b in present:
            num_v = row[b]
            if pd.isna(num_v):
                continue
            g = np.nan
            if chosen == "spliced":
                l = lam.get((scen, n2, b), np.nan)
                hb = hs_i.at[(n2, base_year), b] if (n2, base_year) in hs_i.index else np.nan
                if pd.notna(l) and pd.notna(hb) and hb > 0:
                    g = (l * float(num_v)) / float(hb)
            elif chosen == "step_estimated":
                fb = fc_i.at[(scen, n2, first_fy), b] if (scen, n2, first_fy) in fc_i.index else np.nan
                if pd.notna(fb) and fb > 0:
                    g = step.get((scen, n2, b), 1.0) * float(num_v) / float(fb)
            elif chosen == "hist_base":
                hb = hs_i.at[(n2, base_year), b] if (n2, base_year) in hs_i.index else np.nan
                if pd.notna(hb) and hb > 0:
                    g = float(num_v) / float(hb)
            elif chosen == "forecast_base":
                fb = fc_i.at[(scen, n2, first_fy), b] if (scen, n2, first_fy) in fc_i.index else np.nan
                if pd.notna(fb) and fb > 0:
                    g = float(num_v) / float(fb)
            if pd.isna(g):
                unresolved += 1
                continue
            growth[(scen, n2, b, int(yr))] = g

    if unresolved:
        print(f"[growth] WARNING {unresolved} (scenario, NUTS2, band, year) cells had no "
              f"usable denominator and were dropped.")

    diag = {
        "base_year": base_year,
        "first_forecast_year": first_fy,
        "mode_requested": mode,
        "mode_used": chosen,
        "step_mode": (STEP_MODE if chosen == "step_estimated" else None),
        "overlap_years": overlap,
        "splice_year": t_star,
        "bands_used": present,
        "bands_missing": missing,
        "median_break_by_band": ({} if brk.empty else
                                 brk.groupby("band")["ratio_fc_over_hist"].median().round(5).to_dict()),
        "unresolved_cells": unresolved,
    }
    return growth, first_fy, diag, brk


# ============================================================ MAIN
def main():
    print("=" * 78)
    print("01_students -- anchored NUTS III projection")
    print("=" * 78)
    print("Reading master:", MASTER)

    df = load_master(MASTER)
    hist = df[df.data_scope == "historical"].copy()
    fut = df[df.data_scope == "forecast"].copy()

    if "scenario" in fut.columns:
        present = set(fut["scenario"].dropna().unique())
        dropped = sorted(present & set(DROPPED_SCENARIOS))
        if dropped:
            n0 = len(fut)
            fut = fut[~fut["scenario"].isin(DROPPED_SCENARIOS)].copy()
            print(f"[scenarios] dropped {dropped} ({n0 - len(fut)} forecast rows).")
        unknown = sorted(present - set(SCENARIOS) - set(DROPPED_SCENARIOS))
        if unknown:
            print(f"[scenarios] WARNING present in the data but not in SCENARIOS: {unknown}")
        absent = sorted(set(SCENARIOS) - present)
        if absent:
            print(f"[scenarios] WARNING requested but absent from the data: {absent}")
        print(f"[scenarios] in use: {SCENARIOS}")

    for c in CYCLES + ["students_total"]:
        if c in hist.columns:
            hist[c] = num(hist[c])
    for b in BANDS:
        if b in fut.columns:
            fut[b] = num(fut[b])
        if b in hist.columns:
            hist[b] = num(hist[b])
        if "hist_" + b in hist.columns:
            hist["hist_" + b] = num(hist["hist_" + b])

    cov = build_coverage(hist)
    print(f"Coverage: {len(cov)} obs | cycles {cov.cycle.nunique()} | "
          f"NUTS3 {cov.nuts3_code.nunique()} | {cov.year_start.min()}-{cov.year_start.max()}")

    # ---------------- observed students in the base year (level anchor)
    base_obs = hist[hist.year_start == BASE_OBS_YEAR].copy()
    if base_obs.empty:
        fallback_year = int(hist.year_start.max())
        base_obs = hist[hist.year_start == fallback_year].copy()
        print(f"[anchor] WARNING {BASE_OBS_YEAR} absent; anchored on {fallback_year} instead.")
        anchor_year = fallback_year
    else:
        anchor_year = BASE_OBS_YEAR

    obs_students, obs_cov = {}, {}
    for _, r in base_obs.iterrows():
        for cyc, band in CYCLE_TO_BAND.items():
            st = r.get(cyc, np.nan)
            if pd.isna(st):
                continue
            obs_students[(r["nuts3_code"], cyc)] = float(st)
            pop = _pop_of(r, band)
            if pd.notna(pop) and float(pop) > 0:
                obs_cov[(r["nuts3_code"], cyc)] = float(st) / float(pop)
    print(f"[anchor] {len(obs_students)} region-cycle cells anchored on {anchor_year}; "
          f"{len(obs_cov)} have an observed coverage.")

    # ---------------- backtest and coverage projection (behavioural drift only)
    mtab, best = rolling_origin(cov)
    if not mtab.empty:
        print("\nMASE by cycle/method (coverage, log scale):")
        print(mtab.pivot(index="cycle", columns="method", values="mase").round(3).to_string())
        print("Selected method by cycle:", best)

    fy = sorted(fut.year_start.dropna().astype(int).unique().tolist())
    proj, rsd = {}, {}
    for cyc in CYCLES:
        cc = cov[cov.cycle == cyc]
        if cc.empty:
            continue
        m = best.get(cyc, "damped") if best else "damped"
        p, _ = fit_cycle_projection(cc, fy, method=m)
        proj[cyc] = p
        for rc in cc.nuts3_code.unique():
            rsd[(cyc, rc)] = region_residual_std(cc[cc.nuts3_code == rc])

    # ---------------- NUTS2 demographic growth, anchored on the observed base year
    growth, first_fy, gdiag, brk_tab = nuts2_growth_factors(fut, hist, anchor_year, POP_LINK_MODE)

    # ---------------- maps
    n2_nome_map = fut.drop_duplicates("nuts2_code").set_index("nuts2_code")["nuts2_nome"].to_dict() \
        if "nuts2_nome" in fut.columns else {}
    n3_nome_map = fut.drop_duplicates("nuts3_code").set_index("nuts3_code")["nuts3_nome"].to_dict() \
        if "nuts3_nome" in fut.columns else {}
    n3_to_n2 = fut.drop_duplicates("nuts3_code").set_index("nuts3_code")["nuts2_code"].to_dict()

    # ---------------- [FIX-3] correlated shock structure
    rho = float(np.clip(COMMON_SHOCK_RHO, 0.0, 1.0))
    w_common = np.sqrt(rho)
    w_idio = np.sqrt(1.0 - rho)
    common_eps = {}
    for scen in SCENARIOS:
        for cyc in CYCLES:
            for yr in fy:
                common_eps[(scen, cyc, yr)] = RNG.normal(0.0, 1.0, N_BOOT)

    # aggregate draw accumulators (summing draws, not percentiles)
    acc_nat, acc_cycle, acc_n2 = {}, {}, {}

    def _acc(d, key, arr):
        if key in d:
            d[key] += arr
        else:
            d[key] = arr.copy()

    rows = []
    n_clipped = 0
    n_cells = 0
    missing_n2 = set()

    for (rc, cyc), st0 in obs_students.items():
        band = CYCLE_TO_BAND[cyc]
        n2 = n3_to_n2.get(rc)
        if n2 is None:
            missing_n2.add(rc)
            continue
        cov0 = obs_cov.get((rc, cyc), np.nan)
        sd = min(rsd.get((cyc, rc), 0.05), SD_CAP)

        for scen in SCENARIOS:
            for yr in fy:
                g = growth.get((scen, n2, band, yr), np.nan)
                if pd.isna(g):
                    continue

                # behavioural drift: projected coverage relative to base-year coverage
                lc = proj.get(cyc, {}).get(rc, {}).get(yr, np.nan)
                if pd.notna(lc) and pd.notna(cov0) and cov0 > 0:
                    cov_ratio_raw = float(np.exp(lc)) / cov0
                else:
                    cov_ratio_raw = 1.0
                cov_ratio = float(np.clip(cov_ratio_raw, COV_RATIO_LO, COV_RATIO_HI))
                if abs(cov_ratio - cov_ratio_raw) > 1e-9:
                    n_clipped += 1
                n_cells += 1

                central = st0 * g * cov_ratio

                # horizon measured from the anchor year, not the first forecast year
                h = max(1, yr - anchor_year)
                sd_h = min(sd * np.sqrt(min(h, 8)), SD_H_CAP)

                z = (w_common * common_eps[(scen, cyc, yr)]
                     + w_idio * RNG.normal(0.0, 1.0, N_BOOT))
                eps = np.clip(z * sd_h, -EPS_TRUNC * sd_h, EPS_TRUNC * sd_h)
                draws = np.clip(np.exp(eps) * central, central / DRAW_CLIP, central * DRAW_CLIP)

                _acc(acc_nat, (scen, yr), draws)
                _acc(acc_cycle, (scen, yr, cyc), draws)
                _acc(acc_n2, (scen, yr, n2), draws)

                p10, p50, p90 = np.percentile(draws, [10, 50, 90])
                rows.append({
                    "year_start": yr, "school_year": f"{yr}/{yr + 1}", "scenario": scen,
                    "nuts2_code": n2, "nuts2_nome": n2_nome_map.get(n2, ""),
                    "nuts3_code": rc, "nuts3_nome": n3_nome_map.get(rc, ""),
                    "cycle": cyc,
                    "growth_factor": float(g),
                    "cov_ratio_raw": float(cov_ratio_raw),
                    "cov_ratio": cov_ratio,
                    "cov_ratio_clipped": bool(abs(cov_ratio - cov_ratio_raw) > 1e-9),
                    "students_central": float(central),
                    "students_p10": float(p10),
                    "students_p50": float(p50),
                    "students_p90": float(p90),
                })

    if missing_n2:
        print(f"[map] WARNING {len(missing_n2)} NUTS3 codes present in history are absent "
              f"from the forecast frame and were dropped: {sorted(missing_n2)[:8]}")

    long = pd.DataFrame(rows)
    if long.empty:
        raise RuntimeError("No projections; check observed base-year students and population bands.")

    for c in ["students_central", "students_p10", "students_p50", "students_p90"]:
        long[c] = long[c].round().astype(int)

    if n_cells:
        print(f"[drift] coverage ratio clipped to [{COV_RATIO_LO}, {COV_RATIO_HI}] in "
              f"{n_clipped}/{n_cells} cells ({100 * n_clipped / n_cells:.1f}%).")

    # ---------------- wide / aggregates
    wide = long.pivot_table(
        index=["year_start", "school_year", "scenario", "nuts2_code", "nuts2_nome",
               "nuts3_code", "nuts3_nome"],
        columns="cycle", values="students_central", aggfunc="first").reset_index()
    wide.columns.name = None
    cc_cols = [c for c in CYCLES if c in wide.columns]
    wide["students_total"] = wide[cc_cols].sum(axis=1)

    nuts2 = wide.groupby(["year_start", "school_year", "scenario", "nuts2_code", "nuts2_nome"],
                         as_index=False)[cc_cols + ["students_total"]].sum()
    nac = wide.groupby(["year_start", "school_year", "scenario"],
                       as_index=False)[cc_cols + ["students_total"]].sum()

    # [FIX-3] aggregate bands from summed draws
    def _bands(acc, keycols):
        out = []
        for key, arr in acc.items():
            p10, p50, p90 = np.percentile(arr, [10, 50, 90])
            rec = dict(zip(keycols, key))
            rec.update({"students_p10": float(p10), "students_p50": float(p50),
                        "students_p90": float(p90)})
            out.append(rec)
        return pd.DataFrame(out)

    nat_bands = _bands(acc_nat, ["scenario", "year_start"])
    cyc_bands = _bands(acc_cycle, ["scenario", "year_start", "cycle"])
    n2_bands = _bands(acc_n2, ["scenario", "year_start", "nuts2_code"])

    nac = nac.merge(nat_bands, on=["scenario", "year_start"], how="left")
    nuts2 = nuts2.merge(n2_bands, on=["scenario", "year_start", "nuts2_code"], how="left")
    for c in ["students_p10", "students_p50", "students_p90"]:
        nac[c] = nac[c].round().astype("Int64")
        nuts2[c] = nuts2[c].round().astype("Int64")

    # ---------------- anchor consistency check
    print("\n[check] anchor consistency, central scenario:")
    obs_total = float(sum(obs_students.values()))
    first_row = nac[(nac.scenario == "central") & (nac.year_start == min(fy))]
    if not first_row.empty:
        proj_first = float(first_row["students_total"].iloc[0])
        print(f"    observed {anchor_year}: {obs_total:,.0f}")
        print(f"    projected {min(fy)}:   {proj_first:,.0f}  "
              f"({100 * (proj_first / obs_total - 1):+.2f}%)")
        gap = 100 * (proj_first / obs_total - 1)
        _f = long[(long.scenario == "central") & (long.year_start == min(fy))]
        _demo = float((_f.students_central / _f.cov_ratio.clip(lower=1e-9)).sum())
        print(f"    of which demographic step: {100 * (_demo / obs_total - 1):+.2f}%")
        print(f"           coverage drift:     {100 * (proj_first / _demo - 1):+.2f}%")
        if abs(gap) > 2.5:
            print("    >>> A one-year move of this size is not a demographic path. Inspect the")
            print("    >>> 'population_break' sheet: it is almost certainly a level difference")
            print("    >>> between the historical and the forecast population vintages.")

    cen = nac[nac.scenario == "central"].sort_values("year_start")
    if len(cen) >= 2:
        a, b = cen.iloc[0], cen.iloc[-1]
        print(f"[check] central {int(a.year_start)}->{int(b.year_start)}: "
              f"{a.students_total:,.0f} -> {b.students_total:,.0f} "
              f"({100 * (b.students_total / a.students_total - 1):+.1f}%)")
        w10 = 100 * (b.students_p90 - b.students_p10) / b.students_total
        print(f"[check] national 80% band width in {int(b.year_start)}: {w10:.1f}% of the level "
              f"(rho = {rho:.2f}).")

    # ---------------- write
    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as w:
        long.sort_values(["scenario", "year_start", "nuts3_code", "cycle"]).to_excel(
            w, sheet_name="students_long", index=False)
        wide.sort_values(["scenario", "year_start", "nuts3_code"]).to_excel(
            w, sheet_name="students_nuts3_wide", index=False)
        nuts2.sort_values(["scenario", "year_start", "nuts2_code"]).to_excel(
            w, sheet_name="students_nuts2", index=False)
        nac.sort_values(["scenario", "year_start"]).to_excel(
            w, sheet_name="students_continente", index=False)
        cyc_bands.sort_values(["scenario", "year_start", "cycle"]).to_excel(
            w, sheet_name="bands_by_cycle", index=False)
        if not mtab.empty:
            mtab.to_excel(w, sheet_name="backtest_mase", index=False)
        cov.to_excel(w, sheet_name="coverage_historica", index=False)
        pd.DataFrame({
            "metric": ["n_nuts3", "n_cycles", "first_fc", "last_fc", "scenarios",
                       "scenarios_dropped", "anchor_year", "growth_denominator", "pop_link_mode", "step_mode", "common_shock_rho",
                       "cov_ratio_bounds", "cov_ratio_clipped_share", "n_boot",
                       "random_seed", "method"],
            "value": [wide.nuts3_code.nunique(), len(cc_cols), int(long.year_start.min()),
                      int(long.year_start.max()), ", ".join(SCENARIOS),
                      ", ".join(DROPPED_SCENARIOS) or "none", anchor_year,
                      f"NUTS2 population in {anchor_year}, linked", gdiag["mode_used"],
                      gdiag.get("step_mode"), rho,
                      f"[{COV_RATIO_LO}, {COV_RATIO_HI}]",
                      round(n_clipped / n_cells, 4) if n_cells else np.nan,
                      N_BOOT, RANDOM_SEED,
                      "anchored_base_year x NUTS2_growth x bounded_coverage_drift"],
        }).to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame({"key": list(gdiag.keys()),
                      "value": [json.dumps(v, default=str) for v in gdiag.values()]}).to_excel(
            w, sheet_name="growth_diagnostics", index=False)
        if brk_tab is not None and not brk_tab.empty:
            brk_tab.sort_values(["band", "nuts2_code"]).to_excel(
                w, sheet_name="population_break", index=False)

    # ---------------- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # [FIX-4] mask years with partial regional coverage instead of summing NaN as zero
        n_regions = hist.nuts3_code.nunique()
        hn = hist.groupby("year_start", as_index=False)[
            [c for c in CYCLES if c in hist.columns]].sum(min_count=1)
        counts = hist.groupby("year_start")[
            [c for c in CYCLES if c in hist.columns]].count()
        for c in counts.columns:
            bad = counts.index[counts[c] < n_regions]
            hn.loc[hn.year_start.isin(bad), c] = np.nan

        cyc_plot = cyc_bands[cyc_bands.scenario == "central"].merge(
            long[long.scenario == "central"].groupby(
                ["year_start", "cycle"], as_index=False)["students_central"].sum(),
            on=["year_start", "cycle"], how="left").sort_values("year_start")

        cols = plt.cm.tab10.colors
        fig, ax = plt.subplots(figsize=(11, 6))
        for i, cyc in enumerate(CYCLES):
            d = cyc_plot[cyc_plot.cycle == cyc]
            if d.empty:
                continue
            ax.fill_between(d.year_start, d.students_p10, d.students_p90,
                            alpha=0.15, color=cols[i % 10])
            ax.plot(d.year_start, d.students_central, color=cols[i % 10], lw=2,
                    label=cyc.replace("students_", ""))
            if cyc in hn.columns:
                ax.plot(hn.year_start, hn[cyc], color=cols[i % 10], lw=1.2, ls="--", alpha=0.7)
        ax.set_ylim(0, None)
        ax.set_xlim(2014, None)
        ax.set_title("Students by cycle, Mainland (history dashed, central + 80% band)",
                     fontweight="bold")
        ax.set_xlabel("year")
        ax.set_ylabel("students")
        ax.legend(ncol=3, fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(IMG_DIR / "fanchart_nacional_ciclos.png", dpi=140)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for scen in SCENARIOS:
            d = nac[nac.scenario == scen].sort_values("year_start")
            if d.empty:
                continue
            ax2.plot(d.year_start, d.students_total, lw=2, label=scen)
        dc = nac[nac.scenario == "central"].sort_values("year_start")
        if not dc.empty:
            ax2.fill_between(dc.year_start, dc.students_p10.astype(float),
                             dc.students_p90.astype(float), alpha=0.12, color="grey",
                             label="central 80% band")
        if "students_total" in hist.columns:
            ht = hist.groupby("year_start", as_index=False)["students_total"].sum(min_count=1)
            ct = hist.groupby("year_start")["students_total"].count()
            ht.loc[ht.year_start.isin(ct.index[ct < n_regions]), "students_total"] = np.nan
            ax2.plot(ht.year_start, ht.students_total, color="black", lw=1.5, ls="--",
                     label="history")
        ax2.set_xlim(2014, None)
        ax2.set_ylim(0, None)
        ax2.set_title("Total students, Mainland by demographic scenario", fontweight="bold")
        ax2.set_xlabel("year")
        ax2.set_ylabel("students")
        ax2.legend(frameon=False)
        fig2.tight_layout()
        fig2.savefig(IMG_DIR / "total_nacional_cenarios.png", dpi=140)
        plt.close(fig2)
    except Exception as e:
        print("figure warning:", e)

    print("\nOK ->", OUT_FILE)
    print(nac[nac.scenario == "central"][
        ["school_year", "students_total", "students_p10", "students_p90"]].head(4).to_string(
        index=False))


if __name__ == "__main__":
    main()
