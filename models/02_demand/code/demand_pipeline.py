

from __future__ import annotations

import os
import re
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# ============================================================ PATHS
BASE = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
MODELS_DIR = BASE / "models"
DATA_FILE = MODELS_DIR / "00_data" / "master_panel_nuts3_with_age.xlsx"
STUDENTS_FILE = MODELS_DIR / "01_students" / "results" / "students_forecast_nuts3.xlsx"
STUDENTS_SHEET = "students_nuts3_wide"
DEMAND_DIR = MODELS_DIR / "02_demand" / "results"
IMG_DIR = MODELS_DIR / "02_demand" / "images"
SUPPLY_DIR = MODELS_DIR / "03_supply" / "results"
GAP_DIR = MODELS_DIR / "04_gap" / "results"

if os.environ.get("DEMAND_TEST") == "1":
    DATA_FILE = Path("master_panel_nuts3_with_age.xlsx")
    STUDENTS_FILE = Path("students_forecast_nuts3.xlsx")
    DEMAND_DIR = Path("dem_out"); IMG_DIR = Path("dem_out")
    SUPPLY_DIR = Path("."); GAP_DIR = Path(".")

DEMAND_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR = DEMAND_DIR

# ============================================================ CONFIG
HORIZON = (2025, 2040)
FORECAST_YEARS = list(range(HORIZON[0], HORIZON[1] + 1))
FIRST_RATIO_YEAR = 2014
LAST_OBS_YEAR = 2024
EXCLUDE_YEARS = [2020]                 # COVID excluded by default (pre-registered)
ROLLING_ORIGINS = [2018, 2019, 2021, 2022]
DRIFT_WINDOW = 4
DAMPING_PHI = 0.90
LONG_DIFF_H = 3
ECM_LAMBDA = 0.5
AR1_RHO = 0.6
RANDOM_SEED = 20260818

# [FIX-24] scenarios that must never reach the reporting tables. Block 01 no longer
# produces no_migration; the entry is kept so an older vintage of the input still
# gets filtered, and what was actually found is printed at run time.
DEMO_EXCLUDE = {"no_migration"}
REPORTING_SCENARIOS = ["central", "high", "low"]

ELASTICITY_TWO_WAY_FE = True
ELASTICITY_BAND = (0.30, 1.20)
ELASTICITY_T_MIN = 2.0
ELASTICITY_H_GRID = (2, 3, 4, 5)

WILD_BOOT_REPS = 999
WILD_BOOT_ALPHA = 0.05

# [FIX-22] the deployment rule tests departure from proportionality, not from zero.
# Set to False to fall back to the old H0: beta = 0 rule.
REQUIRE_REJECT_PROPORTIONALITY = True

# [FIX-21] admissibility gate applied inside the bake-off, per origin. The wild
# bootstrap is too expensive to run at every origin, so the cheap band + asymptotic
# t rule is used there and the discrepancy with the full rule is reported.
GATE_BAKEOFF_BETA = True

# [FIX-23] beta variants carried into the contract as explicit scenario columns.
BETA_SCENARIOS = {
    "beta_regfe":  "beta_region_fe_only",       # region FE only, uses the national trend
    "beta_twoway": "beta_region_and_year_fe",   # selected specification
    "beta_long":   "beta_h5",                   # longest horizon available
}

# [FIX-30] heatmap settings.
#   "naive"    -- colour by MASE relative to naive_last in the SAME cell (default).
#   "absolute" -- colour by raw MASE with 1.0 as the centre. Available, but see the
#                 module docstring: at these horizons it paints almost everything red.
HEATMAP_REFERENCE = "naive"
HEATMAP_RATIO_CLIP = 2.0     # colour saturates at 0.5x and 2.0x
HEATMAP_MARK_MASE_LT1 = True  # star + heavy outline on any cell with absolute MASE < 1

INCLUDE_POST_SECONDARY = False

CELLS = ["pre_school", "basic_1", "basic_2", "basic_3_secondary"]


def _b3s_student_cols(include_ps=None):
    inc = INCLUDE_POST_SECONDARY if include_ps is None else include_ps
    cols = ["students_basic_3", "students_secondary"]
    if inc:
        cols.append("students_post_secondary")
    return cols


def _build_demand_cells(include_ps=None):
    return {
        "pre_school":        ("students_pre_school", "teachers_pre_school"),
        "basic_1":           ("students_basic_1", "teachers_basic_1"),
        "basic_2":           ("students_basic_2", "teachers_basic_2"),
        "basic_3_secondary": (_b3s_student_cols(include_ps),
                              "teachers_basic_3_secondary"),
    }


DEMAND_CELLS: dict = _build_demand_cells()

TIE_PREFERENCE = ["naive_last", "median3", "hybrid_robust", "ar1_meanrev",
                  "ecm", "theta_like", "growth_ratio", "hybrid_ecm_mr",
                  "mean_all", "damped_trend"]

INK = "#1a1a1a"; MID = "#3b4a5a"
DEMO_COLORS = {"central": MID, "high": "#b5171e", "low": "#7aa6c2"}

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


# ============================================================ HELPERS
def code(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\.0$", "", str(x).strip()).replace(" ", "")


def as_num(s):
    return pd.to_numeric(s, errors="coerce")


def year_start(sy):
    m = re.search(r"(\d{4})", str(sy)) if not pd.isna(sy) else None
    return int(m.group(1)) if m else np.nan


def read_sheet(path, preferred):
    xls = pd.ExcelFile(path, engine="openpyxl")
    sh = next((s for s in preferred if s in xls.sheet_names), xls.sheet_names[0])
    df = pd.read_excel(path, sheet_name=sh, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def students_of_cell(df, cell, cells_map=None):
    cmap = cells_map or DEMAND_CELLS
    scol = cmap[cell][0]
    if isinstance(scol, (list, tuple)):
        cols = [c for c in scol if c in df.columns]
        return df[cols].apply(as_num).sum(axis=1, min_count=1)
    return as_num(df[scol])


def _reg_slope(years, vals):
    ok = np.isfinite(years) & np.isfinite(vals)
    yy, vv = years[ok].astype(float), vals[ok].astype(float)
    if len(yy) < 2 or np.var(yy) == 0:
        return 0.0
    return float(np.polyfit(yy, vv, 1)[0])


def _geom_damped_sum(drift, h, phi):
    if phi >= 1:
        return drift * h
    return drift * phi * (1 - phi ** h) / (1 - phi)


def find_col(cols, *keys):
    low = {c: str(c).lower() for c in cols}
    for k in keys:
        for c in cols:
            if k in low[c]:
                return c
    return None


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_file(search_dirs, filenames=(), must_have_cols=(), recurse=True):
    """Return (path, tried) -- first match by exact name, then by columns."""
    tried = []
    dirs = [Path(d) for d in search_dirs if d is not None]
    for d in dirs:
        for nm in filenames:
            p = d / nm
            tried.append(str(p))
            if p.exists():
                return p, tried
    if recurse:
        for d in dirs:
            for nm in filenames:
                for hit in d.rglob(nm):
                    tried.append(str(hit))
                    return hit, tried
    if must_have_cols:
        for d in dirs:
            pattern = "**/*.csv" if recurse else "*.csv"
            for g in d.glob(pattern):
                head = read_csv_safe(g)
                if head is None:
                    continue
                cols = [str(c).lower() for c in head.columns]
                if all(any(m in c for c in cols) for m in must_have_cols):
                    tried.append(str(g))
                    return g, tried
    return None, tried


# ============================================================ RATIO PANEL
def _load_hist():
    hist = read_sheet(DATA_FILE, ["historical"])
    hist["nuts3_code"] = hist["nuts3_code"].map(code)
    if "year_start" not in hist.columns and "school_year" in hist.columns:
        hist["year_start"] = hist["school_year"].map(year_start)
    hist["year_start"] = as_num(hist["year_start"]).astype("Int64")
    return hist


def load_ratio_panel(cells_map=None):
    cmap = cells_map or DEMAND_CELLS
    hist = _load_hist()
    rows = []
    for cell, (_, tcol) in cmap.items():
        if tcol not in hist.columns:
            print(f"[panel] WARNING teacher column '{tcol}' absent; cell '{cell}' skipped.")
            continue
        stu = students_of_cell(hist, cell, cmap)
        tea = as_num(hist[tcol])
        r = stu / tea.replace(0, np.nan)
        rows.append(pd.DataFrame({
            "year_start": hist["year_start"].astype("Int64").values,
            "nuts3_code": hist["nuts3_code"].values, "cell": cell,
            "students": stu.values, "teachers": tea.values, "ratio": r.values}))
    if not rows:
        raise RuntimeError("No cells could be built; check the teacher columns.")
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["ratio"].notna() & np.isfinite(panel["ratio"])].copy()
    panel = panel[panel["year_start"] >= FIRST_RATIO_YEAR].copy()
    if EXCLUDE_YEARS:
        panel = panel[~panel["year_start"].isin(EXCLUDE_YEARS)].copy()

    # [FIX-25] duplicated cell-region-year entries would turn scalar lookups into Series
    dup = panel.duplicated(subset=["cell", "nuts3_code", "year_start"], keep=False)
    if dup.any():
        n = int(dup.sum())
        print(f"[panel] WARNING {n} duplicated (cell, region, year) rows; keeping the last.")
        panel = panel.drop_duplicates(subset=["cell", "nuts3_code", "year_start"], keep="last")
    return panel


def post_secondary_evidence():
    """Quantify the post-secondary decision instead of leaving it to taste."""
    hist = _load_hist()
    h = hist[hist["year_start"] == LAST_OBS_YEAR]
    tcol = "teachers_basic_3_secondary"
    if h.empty or tcol not in h.columns or "students_post_secondary" not in h.columns:
        print("[post-sec] columns unavailable -- skipped")
        return None

    tea = as_num(h[tcol]).sum()
    s_wo = sum(as_num(h[c]).sum() for c in ["students_basic_3", "students_secondary"]
               if c in h.columns)
    s_ps = as_num(h["students_post_secondary"]).sum()
    if tea <= 0:
        return None
    r_wo, r_w = s_wo / tea, (s_wo + s_ps) / tea

    print("\n" + "=" * 78)
    print("POST-SECONDARY: EVIDENCE FOR THE INCLUSION DECISION")
    print("=" * 78)
    print(f"  1) {LAST_OBS_YEAR} national ratio  excl={r_wo:.3f}  incl={r_w:.3f}  "
          f"({s_ps:,.0f} students, {100*(r_w/r_wo-1):+.2f}% on the ratio)")

    d = pd.DataFrame({
        "reg": h["nuts3_code"].values,
        "t": as_num(h[tcol]).values,
        "s_core": sum(as_num(h[c]) for c in ["students_basic_3", "students_secondary"]
                      if c in h.columns).values,
        "s_ps": as_num(h["students_post_secondary"]).values,
    }).dropna()
    d = d[(d["t"] > 0) & (d["s_core"] > 0)]
    slope = np.nan; corr = np.nan
    if len(d) >= 8:
        d["ratio_excl"] = d["s_core"] / d["t"]
        d["ps_share"] = d["s_ps"] / (d["s_core"] + d["s_ps"])
        x = d["ps_share"].to_numpy(float)
        y = np.log(d["ratio_excl"].to_numpy(float))
        if np.var(x) > 0:
            slope = float(np.polyfit(x, y, 1)[0])
            corr = float(np.corrcoef(x, y)[0, 1])
        print(f"  2) cross-section: d log(ratio_excl) / d (post-sec share) = {slope:+.3f} "
              f"(corr {corr:+.2f}, n={len(d)})")
        if np.isfinite(corr) and abs(corr) < 0.25:
            print("     -> no association. The teacher column does not look like it")
            print("        carries post-secondary staff; EXCLUDING is the right call.")
        elif np.isfinite(corr) and corr > 0.25:
            print("     -> positive association. Consider INCLUDE_POST_SECONDARY = True.")
        else:
            print("     -> negative association; inconsistent with inclusion. Keep excluding.")

    eff = (r_w / r_wo - 1)
    print(f"  3) mechanical effect on that cell's projected demand: {-100*eff/(1+eff):+.2f}% "
          f"(a higher ratio divides into fewer teachers)")
    print(f"  Current setting: INCLUDE_POST_SECONDARY = {INCLUDE_POST_SECONDARY}")
    return {"ratio_excl": r_wo, "ratio_incl": r_w, "post_sec_students": s_ps,
            "xsec_slope": slope, "xsec_corr": corr}


# ============================================================ RATIO PROJECTORS
def _fit_region_series(train):
    out = {}
    for r, g in train.groupby("nuts3_code"):
        g = g.sort_values("year_start")
        out[r] = (g["year_start"].to_numpy(), g["ratio"].to_numpy())
    return out


def proj_naive_last(train, targets, last_year):
    S = _fit_region_series(train); out = {}
    for r, (yrs, R) in S.items():
        for y in targets:
            out[(r, y)] = float(R[-1])
    return out


def proj_median3(train, targets, last_year):
    S = _fit_region_series(train); out = {}
    for r, (yrs, R) in S.items():
        m = float(np.median(R[-3:] if len(R) >= 3 else R))
        for y in targets:
            out[(r, y)] = m
    return out


def proj_mean_all(train, targets, last_year):
    S = _fit_region_series(train); out = {}
    for r, (yrs, R) in S.items():
        m = float(np.mean(R))
        for y in targets:
            out[(r, y)] = m
    return out


def proj_damped_trend(train, targets, last_year):
    S = _fit_region_series(train)
    pooled = []
    for r, (yrs, R) in S.items():
        lr = np.log(R); tail = yrs >= (yrs.max() - DRIFT_WINDOW + 1)
        pooled.append(_reg_slope(yrs[tail], lr[tail]))
    pooled_drift = float(np.nanmedian(pooled)) if pooled else 0.0
    out = {}
    for r, (yrs, R) in S.items():
        lr = np.log(R); tail = yrs >= (yrs.max() - DRIFT_WINDOW + 1)
        d = 0.5 * _reg_slope(yrs[tail], lr[tail]) + 0.5 * pooled_drift
        L = float(lr[-1])
        for y in targets:
            out[(r, y)] = float(np.exp(L + _geom_damped_sum(d, y - last_year, DAMPING_PHI)))
    return out


def proj_ar1_meanrev(train, targets, last_year):
    """AR(1) on log-ratio: L_t = mu + rho*(L_{t-1}-mu). Reverts to region mean."""
    S = _fit_region_series(train); out = {}
    rhos = []
    for r, (yrs, R) in S.items():
        L = np.log(R)
        if len(L) >= 4:
            x, y = L[:-1] - L[:-1].mean(), L[1:] - L[1:].mean()
            denom = (x * x).sum()
            if denom > 0:
                rhos.append(float((x * y).sum() / denom))
    rho = float(np.clip(np.median(rhos), 0.0, 0.95)) if rhos else AR1_RHO
    for r, (yrs, R) in S.items():
        L = np.log(R); mu = float(L.mean()); Lt = float(L[-1])
        for y in targets:
            h = y - last_year
            out[(r, y)] = float(np.exp(mu + (rho ** h) * (Lt - mu)))
    return out


def proj_ecm(train, targets, last_year):
    """Error-correction toward a long-run ratio R* (region mean of log-ratio)."""
    S = _fit_region_series(train); out = {}
    for r, (yrs, R) in S.items():
        L = np.log(R); Rstar = float(L.mean()); Lt = float(L[-1])
        cur = Lt; vals = {}; y = last_year
        while y < max(targets):
            y += 1
            cur = cur + ECM_LAMBDA * (Rstar - cur)
            vals[y] = cur
        for yy in targets:
            out[(r, yy)] = float(np.exp(vals.get(yy, cur)))
    return out


def proj_theta_like(train, targets, last_year):
    """Blend last level with a very light global drift (theta-style)."""
    S = _fit_region_series(train)
    glob = [_reg_slope(yrs, np.log(R)) for r, (yrs, R) in S.items()]
    gdrift = 0.3 * (float(np.nanmedian(glob)) if glob else 0.0)
    out = {}
    for r, (yrs, R) in S.items():
        L = float(np.log(R[-1]))
        for y in targets:
            out[(r, y)] = float(np.exp(L + _geom_damped_sum(gdrift, y - last_year, 0.8)))
    return out


def proj_hybrid_robust(train, targets, last_year):
    a = proj_naive_last(train, targets, last_year)
    b = proj_median3(train, targets, last_year)
    c = proj_ar1_meanrev(train, targets, last_year)
    return {k: float(np.median([a[k], b[k], c[k]])) for k in a}


def proj_hybrid_ecm_mr(train, targets, last_year):
    a = proj_ecm(train, targets, last_year)
    b = proj_ar1_meanrev(train, targets, last_year)
    return {k: 0.5 * a[k] + 0.5 * b[k] for k in a}


RATIO_METHODS = {
    "naive_last": proj_naive_last,
    "median3": proj_median3,
    "mean_all": proj_mean_all,
    "damped_trend": proj_damped_trend,
    "ar1_meanrev": proj_ar1_meanrev,
    "ecm": proj_ecm,
    "theta_like": proj_theta_like,
    "hybrid_robust": proj_hybrid_robust,
    "hybrid_ecm_mr": proj_hybrid_ecm_mr,
}


# ============================================================ ELASTICITY
def _two_way_demean(values, g1, g2, iters=200, tol=1e-11):
    """Alternating-projection within transformation on two grouping vectors."""
    v = np.asarray(values, dtype=float).copy()
    s1 = pd.Series(g1); s2 = pd.Series(g2)
    for _ in range(iters):
        v0 = v.copy()
        v = v - pd.Series(v).groupby(s1).transform("mean").to_numpy()
        v = v - pd.Series(v).groupby(s2).transform("mean").to_numpy()
        if np.max(np.abs(v - v0)) < tol:
            break
    return v


def _long_diff_frame(train, h):
    recs = []
    for r, gr in train.groupby("nuts3_code"):
        gr = gr.sort_values("year_start")
        yr = gr["year_start"].to_numpy()
        lt = dict(zip(yr, np.log(gr["teachers"].to_numpy())))
        ls = dict(zip(yr, np.log(gr["students"].to_numpy())))
        for y0 in yr:
            y1 = y0 + h
            if y1 in lt and y1 in ls:
                recs.append((r, int(y1), ls[y1] - ls[y0], lt[y1] - lt[y0]))
    if not recs:
        return pd.DataFrame(columns=["r", "t", "x", "y"])
    d = pd.DataFrame(recs, columns=["r", "t", "x", "y"])
    return d.replace([np.inf, -np.inf], np.nan).dropna()


def _ols_within(d, two_way):
    if two_way and d["t"].nunique() >= 2:
        x = _two_way_demean(d["x"].to_numpy(), d["r"].to_numpy(), d["t"].to_numpy())
        y = _two_way_demean(d["y"].to_numpy(), d["r"].to_numpy(), d["t"].to_numpy())
    else:
        x = (d["x"] - d.groupby("r")["x"].transform("mean")).to_numpy()
        y = (d["y"] - d.groupby("r")["y"].transform("mean")).to_numpy()
    sxx = float((x * x).sum())
    if sxx <= 0:
        return np.nan, np.nan, x, y
    beta = float((x * y).sum() / sxx)
    resid = y - beta * x
    meat = 0.0
    for _, idx in d.groupby("r").indices.items():
        meat += float((x[idx] * resid[idx]).sum()) ** 2
    n_g = d["r"].nunique(); n = len(d)
    dof = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - 2, 1))
    se = float(np.sqrt(dof * meat / (sxx ** 2))) if meat >= 0 else np.nan
    return beta, se, x, y


def wild_cluster_bootstrap(d, two_way=True, reps=WILD_BOOT_REPS,
                           seed=RANDOM_SEED, beta0=0.0):
    """Wild cluster bootstrap-t, null imposed, Rademacher weights.

    With 24 clusters the asymptotic cluster-robust t over-rejects badly. The
    restricted (null-imposed) wild bootstrap is the standard remedy: impose
    beta = beta0, resample the residuals with a sign flip drawn once per
    cluster, and compare the observed t against the bootstrap distribution.

    [FIX-22] beta0 is now a genuine argument of the decision. Testing beta0 = 0
    only establishes that some elasticity exists. Testing beta0 = 1 asks whether
    the data reject strict proportionality, which is the question that justifies
    departing from the DGEEC convention.

    Returns (p_value, t_obs, t_crit_lo, t_crit_hi).
    """
    beta, se, x, y = _ols_within(d, two_way)
    if not (np.isfinite(beta) and np.isfinite(se)) or se <= 0:
        return np.nan, np.nan, np.nan, np.nan
    t_obs = (beta - beta0) / se

    resid0 = y - beta0 * x
    groups = list(d.groupby("r").indices.items())
    rng = np.random.default_rng(seed)
    sxx = float((x * x).sum())

    t_star = np.empty(reps)
    for b in range(reps):
        y_star = np.empty_like(y)
        for _, idx in groups:
            w = 1.0 if rng.random() < 0.5 else -1.0     # Rademacher, per cluster
            y_star[idx] = beta0 * x[idx] + w * resid0[idx]
        bb = float((x * y_star).sum() / sxx)
        rr = y_star - bb * x
        meat = 0.0
        for _, idx in groups:
            meat += float((x[idx] * rr[idx]).sum()) ** 2
        n_g = len(groups); n = len(d)
        dof = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - 2, 1))
        se_b = np.sqrt(dof * meat / (sxx ** 2)) if meat >= 0 else np.nan
        t_star[b] = (bb - beta0) / se_b if (np.isfinite(se_b) and se_b > 0) else np.nan

    t_star = t_star[np.isfinite(t_star)]
    if t_star.size == 0:
        return np.nan, t_obs, np.nan, np.nan
    p = float((np.sum(np.abs(t_star) >= abs(t_obs)) + 1) / (t_star.size + 1))
    lo = float(np.quantile(t_star, WILD_BOOT_ALPHA / 2))
    hi = float(np.quantile(t_star, 1 - WILD_BOOT_ALPHA / 2))
    return p, float(t_obs), lo, hi


def fe_beta_full(train, h=LONG_DIFF_H, two_way=ELASTICITY_TWO_WAY_FE,
                 bootstrap=False):
    """Long-difference elasticity of log teachers on log students."""
    empty = {"beta": np.nan, "se": np.nan, "n": 0, "n_regions": 0,
             "two_way": two_way, "h": h, "boot_p0": np.nan, "boot_p1": np.nan,
             "t_obs": np.nan}
    d = _long_diff_frame(train, h)
    if len(d) < 8 or d["r"].nunique() < 3:
        return empty
    beta, se, _, _ = _ols_within(d, two_way)
    if not np.isfinite(beta):
        return empty
    out = {"beta": beta, "se": se, "n": len(d), "n_regions": d["r"].nunique(),
           "two_way": two_way, "h": h, "boot_p0": np.nan, "boot_p1": np.nan,
           "t_obs": np.nan}
    if bootstrap:
        p0, t_obs, _, _ = wild_cluster_bootstrap(d, two_way=two_way, beta0=0.0)
        p1, _, _, _ = wild_cluster_bootstrap(d, two_way=two_way, beta0=1.0)
        out["boot_p0"] = p0
        out["boot_p1"] = p1
        out["t_obs"] = t_obs
    return out


def beta_is_plausible(b):
    """Point estimate inside the admissible band. Necessary, never sufficient."""
    lo, hi = ELASTICITY_BAND
    return bool(np.isfinite(b) and lo <= b <= hi)


def beta_is_usable(b, se, t_min=ELASTICITY_T_MIN):
    """Asymptotic |t| and band. Cheap gate, used inside the bake-off."""
    if not (np.isfinite(b) and np.isfinite(se)) or se <= 0:
        return False, np.nan
    t = b / se
    lo, hi = ELASTICITY_BAND
    return bool(abs(t) >= t_min and lo <= b <= hi), float(t)


def beta_is_usable_boot(b, boot_p0, boot_p1, alpha=WILD_BOOT_ALPHA):
    """[FIX-22] Deployment rule.

    Three conditions, all necessary:
      (a) the point estimate sits inside the admissible band;
      (b) the wild bootstrap rejects H0: beta = 0, so an elasticity is detectable;
      (c) the wild bootstrap rejects H0: beta = 1, so the data actually contradict
          strict proportionality. Without (c) there is no evidential basis for
          departing from the DGEEC convention, whatever the point estimate says.
    """
    lo, hi = ELASTICITY_BAND
    if not np.isfinite(b) or not (lo <= b <= hi):
        return False
    if not np.isfinite(boot_p0) or boot_p0 > alpha:
        return False
    if REQUIRE_REJECT_PROPORTIONALITY:
        if not np.isfinite(boot_p1) or boot_p1 > alpha:
            return False
    return True


# ============================================================ BACKTEST
def _region_scales(train):
    """naive-1 scale on teachers, per region, from the training window."""
    sc = {}
    for r, gr in train.groupby("nuts3_code"):
        v = gr.sort_values("year_start")["teachers"].to_numpy(float)
        if len(v) >= 2:
            s = float(np.mean(np.abs(np.diff(v))))
            if np.isfinite(s) and s > 0:
                sc[r] = s
    return sc


def _score(errs_by_region, obs_by_region, scales):
    """Return (mase_per_region, mase_pooled, wape)."""
    scaled, pooled_obs, all_err = [], [], []
    for r, errs in errs_by_region.items():
        if not errs:
            continue
        all_err.extend(errs)
        pooled_obs.extend(obs_by_region.get(r, []))
        if r in scales:
            scaled.append(np.mean(errs) / scales[r])
    if not all_err:
        return np.nan, np.nan, np.nan
    mase = float(np.mean(scaled)) if scaled else np.nan
    sc_pool = float(np.mean(list(scales.values()))) if scales else np.nan
    mase_pooled = (float(np.mean(all_err)) / sc_pool) if sc_pool and sc_pool > 0 else np.nan
    wape = float(np.sum(all_err) / max(np.sum(np.abs(pooled_obs)), 1e-9))
    return mase, mase_pooled, wape


def backtest(panel=None):
    panel = load_ratio_panel() if panel is None else panel
    rows = []
    for cell, cg in panel.groupby("cell"):
        for origin in ROLLING_ORIGINS:
            train = cg[cg["year_start"] <= origin]
            test = cg[cg["year_start"] > origin]
            if train.empty or test.empty:
                continue
            targets = sorted(test["year_start"].unique())
            scales = _region_scales(train)

            # [FIX-25] dict lookups instead of .loc on a possibly duplicated index
            t_stu = {(r, y): float(s) for r, y, s in
                     zip(test["nuts3_code"], test["year_start"], test["students"])}
            t_tea = {(r, y): float(t) for r, y, t in
                     zip(test["nuts3_code"], test["year_start"], test["teachers"])}

            for name, fn in RATIO_METHODS.items():
                errs, obs = {}, {}
                proj = fn(train, targets, origin)
                for (r, y), Rhat in proj.items():
                    if (r, y) in t_tea and Rhat and np.isfinite(Rhat):
                        pred = t_stu[(r, y)] / Rhat
                        errs.setdefault(r, []).append(abs(t_tea[(r, y)] - pred))
                        obs.setdefault(r, []).append(abs(t_tea[(r, y)]))
                m, mp, w = _score(errs, obs, scales)
                if np.isfinite(w):
                    rows.append({"cell": cell, "origin": origin, "method": name,
                                 "mase": m, "mase_pooled": mp, "wape": w,
                                 "beta": np.nan, "beta_se": np.nan,
                                 "beta_admissible": True})

            # ---- growth_ratio, with the admissibility of its own beta recorded
            info = fe_beta_full(train)
            beta_raw = info["beta"]
            ok_beta, _t = beta_is_usable(beta_raw, info["se"])
            beta = beta_raw if np.isfinite(beta_raw) else 1.0

            base_T = train[train["year_start"] == origin].set_index("nuts3_code")["teachers"]
            base_S = train[train["year_start"] == origin].set_index("nuts3_code")["students"]
            base_T = base_T[~base_T.index.duplicated(keep="last")]
            base_S = base_S[~base_S.index.duplicated(keep="last")]
            errs, obs = {}, {}
            for r in base_T.index:
                if r not in base_S.index or base_S[r] <= 0:
                    continue
                for y in targets:
                    if (r, y) in t_tea:
                        pred = base_T[r] * (t_stu[(r, y)] / base_S[r]) ** beta
                        errs.setdefault(r, []).append(abs(t_tea[(r, y)] - pred))
                        obs.setdefault(r, []).append(abs(t_tea[(r, y)]))
            m, mp, w = _score(errs, obs, scales)
            if np.isfinite(w):
                rows.append({"cell": cell, "origin": origin, "method": "growth_ratio",
                             "mase": m, "mase_pooled": mp, "wape": w,
                             "beta": beta_raw, "beta_se": info["se"],
                             "beta_admissible": bool(ok_beta)})

    bt = pd.DataFrame(rows)
    if bt.empty:
        raise RuntimeError("Bake-off produced no rows; check ROLLING_ORIGINS against the panel.")

    agg = dict(mase=("mase", "mean"), mase_pooled=("mase_pooled", "mean"),
               wape=("wape", "mean"))
    summary = bt.groupby(["cell", "method"], as_index=False).agg(**agg)
    overall = (bt.groupby("method", as_index=False).agg(**agg).sort_values("mase"))

    return bt, summary, overall


_AGG = dict(mase=("mase", "mean"), mase_pooled=("mase_pooled", "mean"),
            wape=("wape", "mean"))


def build_gated(bt, blocked_cells=None):
    """[FIX-28] Build the comparable bake-off tables.

    Three distinct reasons can remove growth_ratio, and v3 conflated them:

      (a) the cell is BLOCKED by the elasticity audit. growth_ratio is then not a
          candidate at all, so it is dropped and the remaining methods keep every
          origin. v3 instead rebased the whole cell onto the single origin where
          growth_ratio happened to survive, discarding three quarters of the
          evidence for methods that were never in question. That is what flipped
          pre_school from hybrid_robust (4 origins) to median3 (1 origin).

      (b) the cell is eligible but growth_ratio's beta was inadmissible at some
          origins. Those origins are dropped for growth_ratio, and the remaining
          methods are rebased onto the surviving origins so the per-cell MASE is
          measured on the same test windows.

      (c) nothing is dropped: raw and gated coincide.

    The overall ranking is computed on a BALANCED set of (cell, origin) pairs, i.e.
    only pairs in which every method is present. v3 averaged growth_ratio over 2
    pairs and its competitors over 10, which made the headline gain meaningless.
    """
    blocked_cells = set(blocked_cells or [])

    keep, notes = [], []
    for cell, g in bt.groupby("cell"):
        others = g[g["method"] != "growth_ratio"]
        gr = g[g["method"] == "growth_ratio"]

        if cell in blocked_cells:
            keep.append(others)
            notes.append({"cell": cell, "regime": "growth_ratio blocked (elasticity)",
                          "origins_kept": others["origin"].nunique(),
                          "gr_present": False})
            continue

        gr_ok = gr[gr["beta_admissible"]]
        if gr_ok.empty:
            keep.append(others)
            notes.append({"cell": cell, "regime": "no admissible beta at any origin",
                          "origins_kept": others["origin"].nunique(),
                          "gr_present": False})
            continue

        ok_origins = set(gr_ok["origin"].unique())
        keep.append(pd.concat([gr_ok, others[others["origin"].isin(ok_origins)]],
                              ignore_index=True))
        notes.append({"cell": cell,
                      "regime": ("rebased to admissible origins"
                                 if len(ok_origins) < g["origin"].nunique()
                                 else "all origins admissible"),
                      "origins_kept": len(ok_origins), "gr_present": True})

    gated = pd.concat(keep, ignore_index=True) if keep else bt.iloc[0:0]
    gate_notes = pd.DataFrame(notes)

    summary_gated = (gated.groupby(["cell", "method"], as_index=False).agg(**_AGG)
                     .merge(gated.groupby(["cell", "method"], as_index=False)
                            .agg(n_origins=("origin", "nunique")),
                            on=["cell", "method"], how="left"))

    # balanced overall: only (cell, origin) pairs where every method is present
    n_methods = gated["method"].nunique()
    pair_counts = gated.groupby(["cell", "origin"])["method"].nunique()
    balanced_pairs = set(pair_counts[pair_counts == n_methods].index)
    if balanced_pairs:
        bal = gated[[(c, o) in balanced_pairs
                     for c, o in zip(gated["cell"], gated["origin"])]]
    else:
        bal = gated.iloc[0:0]

    overall_gated = (bal.groupby("method", as_index=False).agg(**_AGG)
                     .merge(bal.groupby("method", as_index=False)
                            .agg(n_pairs=("origin", "size")), on="method", how="left")
                     .sort_values("mase")) if len(bal) else pd.DataFrame()

    # unbalanced overall, kept for reference and explicitly labelled
    overall_unbal = (gated.groupby("method", as_index=False).agg(**_AGG)
                     .merge(gated.groupby("method", as_index=False)
                            .agg(n_pairs=("origin", "size")), on="method", how="left")
                     .sort_values("mase"))

    return summary_gated, overall_gated, overall_unbal, gate_notes, sorted(balanced_pairs)


def report_bakeoff_gate(bt):
    """[FIX-21] How often was growth_ratio scored on a beta the rule would reject?"""
    gr = bt[bt["method"] == "growth_ratio"]
    if gr.empty:
        return None
    tab = (gr.groupby("cell")
             .agg(origins=("origin", "size"),
                  admissible=("beta_admissible", "sum"),
                  beta_min=("beta", "min"), beta_max=("beta", "max"))
             .reset_index())
    tab["admissible"] = tab["admissible"].astype(int)
    print("\n" + "=" * 78)
    print("BAKE-OFF GATE: admissibility of the beta used at each origin")
    print("=" * 78)
    print(tab.round(3).to_string(index=False))
    bad = tab[tab["admissible"] < tab["origins"]]
    if not bad.empty:
        print("\n  >>> In these cells growth_ratio was scored using betas that the")
        print("  >>> admissibility rule rejects, so its raw MASE is not comparable:")
        for _, r in bad.iterrows():
            print(f"      {r['cell']}: {int(r['admissible'])}/{int(r['origins'])} origins "
                  f"admissible, beta range [{r['beta_min']:+.3f}, {r['beta_max']:+.3f}]")
        print("  >>> bakeoff_by_cell_gated.csv drops those origins. Selection uses the")
        print("  >>> gated table so a method is never chosen on coefficients it cannot use.")
    else:
        print("\n  >>> every origin produced an admissible beta; raw and gated agree.")
    return tab


# ============================================================ ELASTICITY DIAGNOSTICS
def elasticity_report(panel):
    """Estimate and audit beta per cell, with a wild cluster bootstrap."""
    rows = []
    for cell in CELLS:
        cg = panel[panel["cell"] == cell]
        if cg.empty:
            continue
        two = fe_beta_full(cg, two_way=True, bootstrap=True)
        one = fe_beta_full(cg, two_way=False)
        usable_v2, tstat = beta_is_usable(two["beta"], two["se"])
        usable_v3 = beta_is_usable_boot(two["beta"], two["boot_p0"], two["boot_p1"])
        rec = {
            "cell": cell,
            "beta_region_and_year_fe": two["beta"], "se_cluster_region": two["se"],
            "t_stat_asymptotic": tstat,
            "wild_boot_p_vs_0": two["boot_p0"],
            "wild_boot_p_vs_1": two["boot_p1"],
            "ci95_lo": two["beta"] - 1.96 * two["se"] if np.isfinite(two["se"]) else np.nan,
            "ci95_hi": two["beta"] + 1.96 * two["se"] if np.isfinite(two["se"]) else np.nan,
            "beta_region_fe_only": one["beta"], "se_region_fe_only": one["se"],
            "n_long_diffs": two["n"], "n_regions": two["n_regions"],
            "long_diff_h": two["h"],
            "band_lo": ELASTICITY_BAND[0], "band_hi": ELASTICITY_BAND[1],
            "t_min": ELASTICITY_T_MIN, "boot_reps": WILD_BOOT_REPS,
            "verdict_point_in_band": beta_is_plausible(two["beta"]),
            "verdict_asymptotic_t": usable_v2,
            "verdict_deployed": usable_v3,
        }
        for hh in ELASTICITY_H_GRID:
            r_h = fe_beta_full(cg, h=hh, two_way=True)
            rec[f"beta_h{hh}"] = r_h["beta"]
            rec[f"se_h{hh}"] = r_h["se"]
        rows.append(rec)

    rep = pd.DataFrame(rows)
    rep.to_csv(RES_DIR / "elasticity_diagnostics.csv", index=False)

    print("\n" + "=" * 96)
    print("ENROLMENT ELASTICITY OF TEACHER DEMAND (long differences, h=%d)" % LONG_DIFF_H)
    print("=" * 96)
    if rep.empty:
        print("  no estimates")
        return rep

    print(f"  {'cell':<20}{'beta':>8}{'se':>7}{'p(b=0)':>9}{'p(b=1)':>9}  "
          f"{'95% CI':<18}{'regFE':>8}  verdict")
    for _, r in rep.iterrows():
        v = "USE" if r["verdict_deployed"] else "BLOCK"
        ci = f"[{r['ci95_lo']:+.2f},{r['ci95_hi']:+.2f}]"
        print(f"  {r['cell']:<20}{r['beta_region_and_year_fe']:>+8.3f}"
              f"{r['se_cluster_region']:>7.3f}{r['wild_boot_p_vs_0']:>9.3f}"
              f"{r['wild_boot_p_vs_1']:>9.3f}  {ci:<18}"
              f"{r['beta_region_fe_only']:>+8.3f}  {v}")
    print(f"  Wild cluster bootstrap-t, {WILD_BOOT_REPS} reps, Rademacher, null imposed.")
    print("  p(b=0) asks whether any elasticity is detectable.")
    print("  p(b=1) asks whether the data REJECT strict proportionality. Only the second")
    print("  justifies departing from the DGEEC convention, so it now gates deployment.")

    # [FIX-23] identification warning: the two specifications disagree
    print("\n  Specification sensitivity (this is the central caveat of the block):")
    for _, r in rep.iterrows():
        b2, b1 = r["beta_region_and_year_fe"], r["beta_region_fe_only"]
        if np.isfinite(b2) and np.isfinite(b1):
            gapv = abs(b2 - b1)
            flag = "  <-- specifications disagree" if gapv > 0.15 else ""
            print(f"    {r['cell']:<20} region+year FE {b2:+.3f} | region FE {b1:+.3f}"
                  f" | gap {gapv:.3f}{flag}")
    print("    Year fixed effects absorb the common national enrolment decline, which is")
    print("    exactly the variation the projection extrapolates. The region-FE-only")
    print("    estimate keeps that variation and is carried into the contract as a")
    print("    scenario, not relegated to a diagnostic.")

    print("\n  Horizon sensitivity of beta (region+year FE):")
    hdr = "".join(f"{'h='+str(h):>10}" for h in ELASTICITY_H_GRID)
    print(f"  {'cell':<20}{hdr}")
    for _, r in rep.iterrows():
        vals = "".join(f"{r.get('beta_h'+str(h), np.nan):>+10.3f}" for h in ELASTICITY_H_GRID)
        print(f"  {r['cell']:<20}{vals}")
    print("  If beta rises with h, adjustment is incomplete at short horizons and the")
    print("  h=%d coefficient understates the 15-year elasticity." % LONG_DIFF_H)

    flip = rep[rep["verdict_asymptotic_t"] != rep["verdict_deployed"]]
    if not flip.empty:
        print(f"\n  >>> the full rule changes the verdict in: {flip['cell'].tolist()}")

    bad = rep[~rep["verdict_deployed"]]["cell"].tolist()
    if bad:
        print(f"\n  >>> growth_ratio BLOCKED in: {bad}")
        print(f"  >>> Rule: beta in {ELASTICITY_BAND}, bootstrap p(b=0) <= {WILD_BOOT_ALPHA}"
              + (f", and p(b=1) <= {WILD_BOOT_ALPHA}." if REQUIRE_REJECT_PROPORTIONALITY else "."))
    print("\n  A negative beta means teachers rise as enrolment falls. In levels that")
    print("  is the falling student/teacher ratio, not a behavioural response.")
    return rep


# ============================================================ WINNER SELECTION
def _rank_near_best(g, tol=0.02):
    g = g.dropna(subset=["mase"])
    if g.empty:
        return None, []
    best = g["mase"].min()
    near = g[g["mase"] <= best * (1 + tol)]["method"].tolist()
    ordered = g.sort_values("mase")["method"].tolist()
    pick = next((m for m in TIE_PREFERENCE if m in near), ordered[0])
    return pick, ordered


def _winners_from_summary(summary: pd.DataFrame, blocked=None) -> dict:
    """blocked: set of CELLS in which growth_ratio may not be selected."""
    blocked = set(blocked or [])
    winners = {}
    for cell, g in summary.groupby("cell"):
        gg = g[g["method"] != "growth_ratio"] if cell in blocked else g
        pick, ordered = _rank_near_best(gg)
        if pick is None:
            winners[cell] = ("naive_last", np.nan); continue
        winners[cell] = (pick, float(gg[gg["method"] == pick]["mase"].iloc[0]))
    return winners


def pick_winners(summary: pd.DataFrame, blocked_cells=None):
    blocked_cells = set(blocked_cells or [])
    winners = {}; detail = []
    for cell, g in summary.groupby("cell"):
        g = g.dropna(subset=["mase"])
        if g.empty:
            winners[cell] = "naive_last"; continue
        g_avail = g[g["method"] != "growth_ratio"] if cell in blocked_cells else g
        pick, ordered = _rank_near_best(g_avail)
        if pick is None:
            pick = "naive_last"
        winners[cell] = pick
        # [FIX-29] runner-up is the next method AFTER the pick, not ordered[1]:
        # when the tie-break moves the pick down the list, v3 reported the winner
        # itself as its own runner-up.
        rest = [m for m in ordered if m != pick]
        best_method = ordered[0] if ordered else ""
        detail.append({"cell": cell, "winner": pick,
                       "winner_mase": float(g_avail[g_avail["method"] == pick]["mase"].iloc[0]),
                       "best_mase": float(g["mase"].min()),
                       "best_method": best_method,
                       "naive_mase": float(g[g["method"] == "naive_last"]["mase"].iloc[0])
                       if "naive_last" in g["method"].values else np.nan,
                       "growth_ratio_blocked": cell in blocked_cells,
                       "tie_break_applied": bool(best_method and best_method != pick),
                       "runner_up": rest[0] if rest else ""})
    return winners, pd.DataFrame(detail)


# ============================================================ COVID SENSITIVITY
def run_setting(exclude_covid: bool, blocked_cells=None):
    global EXCLUDE_YEARS, ROLLING_ORIGINS
    orig_excl, orig_orig = EXCLUDE_YEARS, ROLLING_ORIGINS
    if exclude_covid:
        EXCLUDE_YEARS = [2020]; ROLLING_ORIGINS = [2018, 2019, 2021, 2022]
    else:
        EXCLUDE_YEARS = []; ROLLING_ORIGINS = [2018, 2019, 2020, 2021, 2022]
    try:
        bt, summary, overall = backtest()
        summary_gated, _og, _ou, _n, _p = build_gated(bt, blocked_cells)
    finally:
        EXCLUDE_YEARS, ROLLING_ORIGINS = orig_excl, orig_orig
    return summary_gated, overall


def covid_sensitivity(sum_excl, blocked_cells=None):
    blocked_cells = set(blocked_cells or [])
    w_excl = _winners_from_summary(sum_excl, blocked=blocked_cells)
    print("[covid] running bake-off WITH COVID included ...")
    sum_incl, _ = run_setting(exclude_covid=False, blocked_cells=blocked_cells)
    w_incl = _winners_from_summary(sum_incl, blocked=blocked_cells)

    cells = sorted(set(w_excl) | set(w_incl))
    rows = []
    for c in cells:
        we, me = w_excl.get(c, ("-", np.nan))
        wi, mi = w_incl.get(c, ("-", np.nan))
        gap = (abs(mi - me) / me * 100) if (me == me and mi == mi and me) else np.nan
        rows.append({"cell": c,
                     "winner_excl_covid": we, "mase_excl": round(me, 3) if me == me else np.nan,
                     "winner_incl_covid": wi, "mase_incl": round(mi, 3) if mi == mi else np.nan,
                     "mase_gap_pct": round(gap, 1) if gap == gap else np.nan,
                     "winner_changed": "YES" if we != wi else "no"})
    comp = pd.DataFrame(rows)
    comp.to_csv(RES_DIR / "covid_sensitivity_winners.csv", index=False)
    sum_excl.to_csv(RES_DIR / "bakeoff_by_cell_excl_covid.csv", index=False)
    sum_incl.to_csv(RES_DIR / "bakeoff_by_cell_incl_covid.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comp))
    ax.bar(x - 0.2, comp["mase_excl"], width=0.4, label="COVID excluded", color="#1A5276")
    ax.bar(x + 0.2, comp["mase_incl"], width=0.4, label="COVID included", color="#E67E22")
    ax.set_xticks(x); ax.set_xticklabels(comp["cell"], rotation=20, ha="right")
    ax.set_ylabel("winning-method MASE (multi-step; levels not comparable to 1)")
    for i, row in comp.iterrows():
        tag = f"{row['winner_excl_covid']}\nvs {row['winner_incl_covid']}"
        col = "#C0392B" if row["winner_changed"] == "YES" else "#555"
        yv = np.nanmax([row["mase_excl"], row["mase_incl"]]) * 1.02
        ax.text(i, yv, tag, ha="center", va="bottom", fontsize=7, color=col)
    ax.set_title("COVID sensitivity of the per-cell winning method", fontweight="bold")
    ax.legend()
    fig.tight_layout(); fig.savefig(IMG_DIR / "covid_sensitivity.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 74)
    print("COVID SENSITIVITY OF PER-CELL WINNERS")
    print("=" * 74)
    print(comp.to_string(index=False))
    changed = comp[comp["winner_changed"] == "YES"]
    if not changed.empty:
        print(f"\n>>> winner changes with COVID in: {changed['cell'].tolist()}")
        for _, r in changed.iterrows():
            print(f"    {r['cell']}: MASE {r['mase_excl']} vs {r['mase_incl']} "
                  f"({r['mase_gap_pct']}% apart)")
        print(">>> A small gap means the two methods are near-equivalent and the")
        print(">>> switch is a tie-break artefact, not evidence of fragility.")
    else:
        print("\n>>> ROBUST: the per-cell winners are IDENTICAL with and without COVID.")
    print("[ok]", IMG_DIR / "covid_sensitivity.png")
    return comp


# ============================================================ PROJECT SELECTED RATIO
def project_selected(winners, panel=None):
    panel = load_ratio_panel() if panel is None else panel
    rows = []
    for cell, cg in panel.groupby("cell"):
        method = winners.get(cell, "naive_last")
        fn = RATIO_METHODS.get(method, RATIO_METHODS["naive_last"])
        label = method if method != "growth_ratio" else "growth_ratio_elast"
        proj = fn(cg, FORECAST_YEARS, LAST_OBS_YEAR)
        for (r, y), Rhat in proj.items():
            rows.append({"cell": cell, "nuts3_code": r, "year_start": y,
                         "R_selected": float(Rhat), "method": label})
    return pd.DataFrame(rows), panel


# ============================================================ COMBINE WITH ENROLMENT
def load_students(cells_map=None):
    cmap = cells_map or DEMAND_CELLS
    xls = pd.ExcelFile(STUDENTS_FILE, engine="openpyxl")
    sh = STUDENTS_SHEET if STUDENTS_SHEET in xls.sheet_names else xls.sheet_names[0]
    w = pd.read_excel(STUDENTS_FILE, sheet_name=sh, engine="openpyxl")
    w.columns = [str(c).strip() for c in w.columns]
    w["nuts3_code"] = w["nuts3_code"].map(code)
    w["year_start"] = pd.to_numeric(w["year_start"], errors="coerce").astype("Int64")
    if "scenario" not in w.columns:
        w["scenario"] = "central"

    # [FIX-24] report what block 01 actually delivered instead of assuming
    found = sorted(w["scenario"].dropna().unique())
    dropped = sorted(set(found) & DEMO_EXCLUDE)
    if dropped:
        w = w[~w["scenario"].isin(DEMO_EXCLUDE)].copy()
        print(f"[students] dropped non-reporting scenarios: {dropped}")
    kept = sorted(w["scenario"].dropna().unique())
    absent = sorted(set(REPORTING_SCENARIOS) - set(kept))
    print(f"[students] scenarios in use: {kept}" + (f" | absent: {absent}" if absent else ""))

    rows = []
    for cell in cmap:
        s = students_of_cell(w, cell, cmap)
        rows.append(pd.DataFrame({"scenario": w["scenario"].values,
                                  "year_start": w["year_start"].values,
                                  "nuts3_code": w["nuts3_code"].values,
                                  "cell": cell, "students": s.values}))
    out = pd.concat(rows, ignore_index=True)
    return out[out["students"].notna()].copy()


def build_selected_demand(sel_ratio, winners, panel, rep):
    """[FIX-23] Build the selected path AND the beta scenario range."""
    students = load_students()
    df = students.merge(
        sel_ratio[["cell", "nuts3_code", "year_start", "R_selected", "method"]],
        on=["cell", "nuts3_code", "year_start"], how="left")
    df["demand_selected"] = df["students"] / df["R_selected"]

    last = panel[panel["year_start"] == LAST_OBS_YEAR]
    base = {}
    for c in CELLS:
        b = last[last["cell"] == c].set_index("nuts3_code")
        base[c] = b[~b.index.duplicated(keep="last")]

    # proportional benchmark, beta = 1
    prop = np.full(len(df), np.nan)
    for i, (cell, r, s) in enumerate(zip(df["cell"], df["nuts3_code"], df["students"])):
        b = base.get(cell)
        if b is None or r not in b.index:
            continue
        S0 = b.loc[r, "students"]; T0 = b.loc[r, "teachers"]
        if S0 and S0 > 0:
            prop[i] = T0 * (s / S0)
    df["demand_proportional"] = prop

    # betas per cell, from the diagnostics table
    def _beta_of(cell, col):
        if rep.empty or cell not in set(rep["cell"]):
            return np.nan
        v = rep.loc[rep["cell"] == cell, col]
        return float(v.iloc[0]) if len(v) and np.isfinite(v.iloc[0]) else np.nan

    betas_main = {c: _beta_of(c, "beta_region_and_year_fe") for c in CELLS}

    gr_cells = [c for c, m in winners.items() if m == "growth_ratio"]
    for cell in gr_cells:
        beta = betas_main.get(cell, 1.0)
        if not np.isfinite(beta):
            beta = 1.0
        b = base[cell]
        idx = df.index[df["cell"] == cell]
        for i in idx:
            r = df.at[i, "nuts3_code"]
            if r in b.index and b.loc[r, "students"] > 0:
                T0 = b.loc[r, "teachers"]; S0 = b.loc[r, "students"]
                df.at[i, "demand_selected"] = T0 * (df.at[i, "students"] / S0) ** beta

    # [FIX-23] one column per beta variant, so the range is explicit downstream
    for label, col in BETA_SCENARIOS.items():
        vals = df["demand_selected"].to_numpy(float).copy()
        for cell in gr_cells:
            beta = _beta_of(cell, col)
            if not np.isfinite(beta):
                continue
            b = base[cell]
            pos = np.where(df["cell"].to_numpy() == cell)[0]
            for p in pos:
                r = df["nuts3_code"].iat[p]
                if r in b.index and b.loc[r, "students"] > 0:
                    T0 = b.loc[r, "teachers"]; S0 = b.loc[r, "students"]
                    vals[p] = T0 * (df["students"].iat[p] / S0) ** beta
        df["demand_" + label] = vals

    df["R_selected"] = np.where(df["demand_selected"] > 0,
                                df["students"] / df["demand_selected"], np.nan)
    return df, betas_main


# ============================================================ CONTRACT + CHECKS
def implied_ratio_path(df_sel, panel):
    rows = []
    obs = panel[panel["year_start"] == LAST_OBS_YEAR]
    for cell in CELLS:
        o = obs[obs["cell"] == cell]
        if not o.empty and o["teachers"].sum() > 0:
            rows.append({"scenario": "observed", "year": LAST_OBS_YEAR, "cell": cell,
                         "students": float(o["students"].sum()),
                         "demand": float(o["teachers"].sum()),
                         "ratio": float(o["students"].sum() / o["teachers"].sum())})
    g = (df_sel.rename(columns={"year_start": "year"})
              .groupby(["scenario", "year", "cell"], as_index=False)
              [["students", "demand_selected"]].sum())
    for _, r in g.iterrows():
        if r["demand_selected"] > 0:
            rows.append({"scenario": r["scenario"], "year": int(r["year"]), "cell": r["cell"],
                         "students": r["students"], "demand": r["demand_selected"],
                         "ratio": r["students"] / r["demand_selected"]})
    out = pd.DataFrame(rows)
    out.to_csv(RES_DIR / "implied_ratio_path.csv", index=False)

    print("\n[ratio] implied national student/teacher ratio, central scenario:")
    base = {r["cell"]: r["ratio"] for r in rows if r["scenario"] == "observed"}
    cen = out[out["scenario"] == "central"]
    for cell in CELLS:
        c = cen[cen["cell"] == cell].sort_values("year")
        if c.empty or cell not in base:
            continue
        r0, r1 = base[cell], float(c["ratio"].iloc[-1])
        print(f"    {cell:<20} {LAST_OBS_YEAR}: {r0:6.2f}   "
              f"{int(c['year'].iloc[-1])}: {r1:6.2f}   ({100*(r1/r0-1):+.1f}%)")
    print("    DGEEC hold their staffing ratio constant; any movement here is a")
    print("    deliberate departure from the benchmark and should be defended.")
    return out


def build_totals(df_sel):
    df = df_sel.rename(columns={"year_start": "year"}) if "year" not in df_sel.columns else df_sel
    val_cols = ["demand_selected", "demand_proportional"] + \
               ["demand_" + k for k in BETA_SCENARIOS]
    val_cols = [c for c in val_cols if c in df.columns]
    reg = df.groupby(["scenario", "nuts3_code", "year"], as_index=False)[val_cols].sum()
    nat = reg.groupby(["scenario", "year"], as_index=False)[val_cols].sum()
    return reg, nat


def write_contract(reg, nat, dem_by_cell):
    out = reg.copy()
    out["reporting_scenario"] = ~out["scenario"].isin(DEMO_EXCLUDE)
    out.to_csv(DEMAND_DIR / "teacher_demand_selected.csv", index=False)
    dem_by_cell.to_csv(DEMAND_DIR / "teacher_demand_by_cell.csv", index=False)
    nat[~nat["scenario"].isin(DEMO_EXCLUDE)].to_csv(
        DEMAND_DIR / "teacher_demand_national.csv", index=False)
    extra = [c for c in nat.columns if c.startswith("demand_") and
             c not in ("demand_selected", "demand_proportional")]
    print(f"[csv] teacher_demand_selected.csv (04 contract, {reg.scenario.nunique()} scenarios)")
    print(f"      value columns: demand_selected, demand_proportional, {', '.join(extra)}")
    print("[csv] teacher_demand_by_cell.csv + teacher_demand_national.csv")


def dgeec_comparison(nat):
    """Compare the demand paths against the published DGEEC trajectory."""
    cen = nat[nat["scenario"] == "central"].sort_values("year")
    if cen.empty:
        return None
    y0, y1 = int(cen["year"].iloc[0]), 2034
    row1 = cen[cen["year"] == y1]
    if row1.empty:
        return None

    def pct(col):
        if col not in cen.columns:
            return np.nan
        a = float(cen[col].iloc[0]); b = float(row1[col].iloc[0])
        return 100 * (b / a - 1) if a else np.nan

    d_sel, d_pro = pct("demand_selected"), pct("demand_proportional")
    print("\n" + "=" * 78)
    print("BENCHMARK: our demand path vs DGEEC, %d-%d" % (y0, y1))
    print("=" * 78)
    print(f"  selected (per-cell winners)   {d_sel:+.2f}%")
    for label in BETA_SCENARIOS:
        v = pct("demand_" + label)
        if np.isfinite(v):
            print(f"  {label:<28}  {v:+.2f}%")
    print(f"  proportional (beta = 1)       {d_pro:+.2f}%   <- DGEEC convention")
    print("  DGEEC                          -5%      (enrolment -5%, ratio held constant)")
    print("  The proportional path IS the DGEEC convention. The spread across the beta")
    print("  variants is the elasticity assumption, and it should be reported as a range.")
    return {"selected_pct": d_sel, "proportional_pct": d_pro,
            **{label + "_pct": pct("demand_" + label) for label in BETA_SCENARIOS}}


def supply_stock_2025():
    p, tried = find_file(
        [SUPPLY_DIR, GAP_DIR, MODELS_DIR / "03_supply", MODELS_DIR / "04_gap"],
        filenames=("supply_projection.csv", "supply_universe_national.csv",
                   "supply_wave_v3.csv", "gap_national.csv", "supply_national.csv"),
        must_have_cols=("year", "stock"))
    if p is None:
        return None, None, tried
    d = read_csv_safe(p)
    if d is None:
        return None, None, tried
    yc = find_col(d.columns, "year", "ano"); sc = find_col(d.columns, "stock")
    if not (yc and sc):
        return None, None, tried
    d = d.copy(); d[yc] = pd.to_numeric(d[yc], errors="coerce")
    row = d[d[yc] == HORIZON[0]]
    if row.empty:
        return None, None, tried
    return float(pd.to_numeric(row[sc], errors="coerce").iloc[0]), Path(p).name, tried


def gap_nuts3_codes():
    p, tried = find_file(
        [GAP_DIR, MODELS_DIR / "04_gap"],
        filenames=("gap_recruitment_nuts3.csv", "gap_nuts3.csv",
                   "supply_universe_nuts3.csv", "gap_by_nuts3.csv"),
        must_have_cols=("nuts3",))
    if p is None:
        return None, None, tried
    d = read_csv_safe(p)
    if d is None:
        return None, None, tried
    rc = find_col(d.columns, "nuts3", "region")
    if rc is None:
        return None, None, tried
    return set(d[rc].astype(str).str.strip()), Path(p).name, tried


_RESULTS = {"pass": 0, "fail": 0, "skip": 0}


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    _RESULTS["pass" if ok else "fail"] += 1
    print(f"    [{tag}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def skip(name, reason, tried=None):
    _RESULTS["skip"] += 1
    print(f"    [SKIP] {name} -- {reason}")
    if tried:
        for t in tried[:6]:
            print(f"           tried: {t}")
    return None


def run_checks(dem_by_cell, reg, nat):
    print("[consistency] checks 02 <-> 01/03/04:")
    _RESULTS.update({"pass": 0, "fail": 0, "skip": 0})
    all_ok = True

    cells_found = set(dem_by_cell["cell"].unique())
    all_ok &= check("cells match block 04", cells_found == set(CELLS),
                    f"found={sorted(cells_found)}")

    yrs = set(int(y) for y in reg["year"].unique())
    full = set(range(HORIZON[0], HORIZON[1] + 1))
    all_ok &= check("horizon 2025-2040 complete", full.issubset(yrs),
                    f"missing={sorted(full - yrs) or 'none'}")

    scn = set(reg["scenario"].unique())
    all_ok &= check("central scenario present", "central" in scn, f"scenarios={sorted(scn)}")
    all_ok &= check("no excluded scenario leaked", not (scn & DEMO_EXCLUDE),
                    f"excluded={sorted(DEMO_EXCLUDE)}")

    codes_04, src04, tried04 = gap_nuts3_codes()
    if codes_04 is None:
        skip("NUTS III codes match block 04", "no block-04 geography file found", tried04)
    else:
        codes_02 = set(reg["nuts3_code"].astype(str).str.strip())
        inter = codes_02 & codes_04
        all_ok &= check("NUTS III codes match block 04",
                        len(inter) >= min(len(codes_04), 20),
                        f"common={len(inter)}/{len(codes_04)} via {src04}")

    stock25, src, tried = supply_stock_2025()
    d25 = nat[(nat.scenario == "central") & (nat.year == HORIZON[0])]["demand_selected"]
    if stock25 is None or d25.empty:
        skip("2025 base-year identity", "supply stock 2025 not found", tried)
    else:
        dd = float(d25.iloc[0]); diff = dd - stock25
        rel = abs(diff) / max(stock25, 1.0)
        print(f"    [INFO] 2025 base-year identity -- demand={dd:,.0f} vs "
              f"stock={stock25:,.0f} ({src}); residual={diff:+,.0f} ({rel:.2%})")
        print("           Not a validation: both are built from the same 2024 headcount,")
        print("           so they agree by construction up to the enrolment ratio.")

    need = {"scenario", "year", "nuts3_code", "demand_selected", "demand_proportional"}
    all_ok &= check("contract schema (04 reads unambiguously)",
                    need.issubset(set(reg.columns)), f"columns={list(reg.columns)}")

    ok_pos = (nat["demand_selected"] > 0).all() and (nat["demand_selected"] < 1e6).all()
    all_ok &= check("demand positive and non-exploding", ok_pos)

    print(f"[consistency] {_RESULTS['pass']} pass, {_RESULTS['fail']} fail, "
          f"{_RESULTS['skip']} skipped")
    if _RESULTS["skip"]:
        print("[consistency] NOTE: skipped checks verified nothing. Fix the paths above.")
    return all_ok


# ============================================================ FIGURES
def fig_overall(overall, gated=True):
    o = overall.sort_values("mase")
    base_v = o[o["method"] == "naive_last"]["mase"]
    base_v = float(base_v.iloc[0]) if len(base_v) else np.nan
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#27AE60" if (np.isfinite(base_v) and m < base_v) else "#C0392B"
              for m in o["mase"]]
    ax.barh(o["method"], o["mase"], color=colors)
    if np.isfinite(base_v):
        ax.axvline(base_v, color="k", lw=1.0, ls="--")
        ax.text(base_v, -0.6, " naive_last", fontsize=8, va="center", color="k")
    # [FIX-20] the reference is the naive baseline, not the value 1
    ax.set_xlabel("mean MASE across cells and origins "
                  "(multi-step; compare methods, not the level against 1)")
    ttl = ("Ratio-projection bake-off: overall MASE ranking"
           + ("\n(admissible betas only, common origins)" if gated else ""))
    ax.set_title(ttl, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(IMG_DIR / "bakeoff_overall.png", bbox_inches="tight")
    plt.close(fig)
    print("[ok]", IMG_DIR / "bakeoff_overall.png")


def fig_heat(summary, blocked_cells=None, reference=HEATMAP_REFERENCE):
    """[FIX-30] Bake-off heatmap by method x cell.

    Colour: MASE relative to naive_last measured in the SAME cell, diverging and
    centred at 1.00x. Green beats the benchmark in that column, red loses.

    Absolute MASE < 1 is marked separately, with a star and a heavy outline. That
    threshold is meaningful (the method beats the in-sample one-step naive even at
    multi-year horizons) but it cannot drive the colour scale: MASE here divides
    multi-step errors by the mean ONE-step change in the training window, so most
    admissible methods sit above 1 by construction and a red-above-1 rule would
    make the panel unreadable.

    Empty cells are labelled rather than left blank: "blocked" when the elasticity
    audit removed growth_ratio from that cell, "n/a" otherwise.
    """
    blocked_cells = set(blocked_cells or [])

    piv = summary.pivot_table(index="method", columns="cell", values="mase", aggfunc="mean")
    if piv.empty:
        print("[fig] heatmap: nothing to plot")
        return

    # [FIX-30a] keep every method as a row even if the gated table dropped it entirely
    for m in RATIO_METHODS:
        if m not in piv.index:
            piv.loc[m] = np.nan
    if "growth_ratio" not in piv.index:
        piv.loc["growth_ratio"] = np.nan

    if "n_origins" in summary.columns:
        piv_n = summary.pivot_table(index="method", columns="cell",
                                    values="n_origins", aggfunc="max")
    else:
        piv_n = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float)

    # ---- reference
    if reference == "naive" and "naive_last" in piv.index:
        base = piv.loc["naive_last"]
        ratio = piv.divide(base, axis=1)
        cbar_label = "MASE relative to naive_last in the same cell"
        centre, lo, hi = 1.0, 1.0 / HEATMAP_RATIO_CLIP, HEATMAP_RATIO_CLIP
        rel_mode = True
    else:
        ratio = piv.copy()
        cbar_label = "MASE (multi-step)"
        centre = 1.0
        lo = float(np.nanmin(piv.values)) if np.isfinite(piv.values).any() else 0.0
        hi = float(np.nanmax(piv.values)) if np.isfinite(piv.values).any() else 2.0
        lo = min(lo, 0.99); hi = max(hi, 1.01)
        rel_mode = False

    order = ratio.mean(axis=1, skipna=True).sort_values(na_position="last").index
    piv = piv.reindex(order); ratio = ratio.reindex(order)
    if not piv_n.empty:
        piv_n = piv_n.reindex(order)

    plot_vals = ratio.clip(lower=lo, upper=hi).to_numpy(float)
    norm = TwoSlopeNorm(vmin=lo, vcenter=centre, vmax=hi)

    fig, ax = plt.subplots(figsize=(11, 7.6))
    im = ax.imshow(plot_vals, aspect="auto", cmap="RdYlGn_r", norm=norm)

    # ---- axis labels, with the origin count per column
    col_labels = []
    for c in piv.columns:
        n = np.nan
        if not piv_n.empty and c in piv_n.columns and piv_n[c].notna().any():
            n = np.nanmax(piv_n[c].to_numpy(float))
        col_labels.append(f"{c}\n(n={int(n)} origins)" if np.isfinite(n) else str(c))
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)

    # ---- annotations
    any_lt1 = False
    for i, meth in enumerate(piv.index):
        for j, cell in enumerate(piv.columns):
            v = piv.values[i, j]
            if np.isfinite(v):
                rv = ratio.values[i, j]
                txt = f"{v:.2f}"
                if rel_mode and np.isfinite(rv):
                    txt += f"\n({rv:.2f}x)"
                if HEATMAP_MARK_MASE_LT1 and v < 1.0:
                    any_lt1 = True
                    txt = "* " + txt
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#12421f", lw=2.6, zorder=4))
                ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                        color="black", linespacing=1.15, zorder=5,
                        fontweight="bold" if (v < 1.0 and HEATMAP_MARK_MASE_LT1) else "normal")
            else:
                reason = ("blocked" if (meth == "growth_ratio" and cell in blocked_cells)
                          else "n/a")
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f4f4f4",
                                       edgecolor="#c2c2c2", hatch="///", lw=0.7, zorder=3))
                ax.text(j, i, reason, ha="center", va="center", fontsize=7.5,
                        color="#7a7a7a", style="italic", zorder=5)

    ax.set_xticks(np.arange(-0.5, len(piv.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(piv.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    cb = fig.colorbar(im, ax=ax, label=cbar_label, pad=0.02)
    if rel_mode:
        cb.ax.axhline(1.0, color="black", lw=1.2)
        cb.set_ticks([lo, 0.75, 1.0, 1.5, hi])
        cb.set_ticklabels([f"{lo:.2f}x", "0.75x", "1.00x\nnaive", "1.50x", f"{hi:.2f}x"])

    ax.set_title("Bake-off MASE by method x cell\n"
                 "colour = performance against naive_last in the SAME cell "
                 "(green beats it, red loses)",
                 fontweight="bold", fontsize=11)

    handles = []
    if any_lt1 and HEATMAP_MARK_MASE_LT1:
        handles.append(Patch(facecolor="none", edgecolor="#12421f", lw=2.6,
                             label="* absolute MASE < 1 (beats the one-step naive scale)"))
    if blocked_cells:
        handles.append(Patch(facecolor="#f4f4f4", edgecolor="#c2c2c2", hatch="///",
                             label="blocked: growth_ratio rejected by the elasticity audit"))
    if handles:
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
                  frameon=False, fontsize=8, ncol=1)

    fig.text(0.01, 0.005,
             "Cell text: absolute MASE, then the ratio to naive_last in brackets. "
             "MASE above 1 is the arithmetic norm at horizons of up to six years and "
             "is not a sign of failure,\nwhich is why colour is anchored on the naive "
             "benchmark rather than on 1. Columns rest on different numbers of rolling "
             "origins after gating and are not mutually comparable.",
             ha="left", va="bottom", fontsize=7.5, color="#555")
    fig.subplots_adjust(bottom=0.26)
    fig.savefig(IMG_DIR / "bakeoff_heatmap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", IMG_DIR / "bakeoff_heatmap.png")


def fig_selected(df_sel, winners):
    c = df_sel[df_sel["scenario"] == "central"]
    cols = ["demand_selected", "demand_proportional"] + \
           ["demand_" + k for k in BETA_SCENARIOS if "demand_" + k in c.columns]
    nat = c.groupby("year_start", as_index=False)[cols].sum()
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # [FIX-23] the beta range drawn as a band, so the elasticity choice is visible
    bcols = [x for x in cols if x.startswith("demand_beta")]
    if len(bcols) >= 2:
        lo_b = nat[bcols].min(axis=1); hi_b = nat[bcols].max(axis=1)
        ax.fill_between(nat["year_start"], lo_b, hi_b, color="#C0392B", alpha=0.13,
                        label="elasticity range (beta specifications)")
    ax.plot(nat["year_start"], nat["demand_selected"], "o-", color="#C0392B", lw=2.4, ms=6,
            label="model-selected (per-cell winner)")
    ax.plot(nat["year_start"], nat["demand_proportional"], "s--", color="#3b4a5a", lw=2.0, ms=5,
            label="proportional, beta = 1 (DGEEC convention)")
    vals = nat[cols].to_numpy(float)
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    pad = max((hi - lo) * 0.20, hi * 0.01)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(nat["year_start"].min() - 0.5, nat["year_start"].max() + 0.5)
    ax.set_title("Teacher demand: selected forecast vs strict proportionality (central, national)",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teaching posts (headcount)")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    txt = "   |   ".join(f"{k}: {v}" for k, v in winners.items())
    fig.text(0.5, 0.005, "per-cell winners  --  " + txt, ha="center", va="bottom",
             fontsize=8, color="#555")
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(IMG_DIR / "demand_selected.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", IMG_DIR / "demand_selected.png")


def fig_national(nat):
    nat = nat[~nat["scenario"].isin(DEMO_EXCLUDE)]
    if nat.empty:
        print("[fig] national: no data"); return
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for scn in [s for s in REPORTING_SCENARIOS if s in nat.scenario.unique()]:
        s = nat[nat.scenario == scn].sort_values("year")
        ax.plot(s.year, s.demand_selected, "-o", lw=2.4, ms=5,
                color=DEMO_COLORS.get(scn, INK), label=f"{scn} scenario")
    lo, hi = float(nat["demand_selected"].min()), float(nat["demand_selected"].max())
    pad = max((hi - lo) * 0.15, hi * 0.01)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(nat["year"].min() - 0.5, nat["year"].max() + 0.5)
    ax.set_title("Teacher demand (headcount), national -- by demographic scenario",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teaching posts (headcount)")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "demand_national_scenarios.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", IMG_DIR / "demand_national_scenarios.png")


# ============================================================ MAIN
def main():
    print("=" * 78)
    print("02_demand -- pipeline (bake-off -> elasticity audit -> selection -> demand)")
    print("=" * 78)

    ps_info = post_secondary_evidence()

    print("\n[1] Ratio bake-off (rolling-origin) ...")
    bt, summary, overall = backtest()
    bt.to_csv(RES_DIR / "bakeoff_detail.csv", index=False)
    summary.to_csv(RES_DIR / "bakeoff_by_cell.csv", index=False)
    overall.to_csv(RES_DIR / "bakeoff_overall.csv", index=False)

    print("\nRAW RANKING (all origins, no admissibility gate) -- reference only:")
    print(overall.round(3).to_string(index=False))
    if "growth_ratio" in set(overall["method"]):
        print(">>> Do not quote this table for growth_ratio: it includes origins whose")
        print(">>> beta the admissibility rule rejects. Use the gated ranking below.")

    max_h = (LAST_OBS_YEAR - min(ROLLING_ORIGINS)) if ROLLING_ORIGINS else 0
    print(f">>> NOTE on levels: MASE scales errors at horizons up to {max_h} years by the")
    print(">>> mean ONE-step change in the training window, so values above 1 are the")
    print(">>> arithmetic norm and are NOT evidence that the methods fail. Only the")
    print(">>> ranking between methods is informative.")

    report_bakeoff_gate(bt)

    # [FIX-28] the elasticity audit must run BEFORE the gate, because a blocked cell
    # removes growth_ratio from contention entirely and therefore must NOT cause the
    # other methods to be rebased onto a reduced set of origins.
    print("\n[2] Auditing the enrolment elasticity used by growth_ratio ...")
    panel_all = load_ratio_panel()
    rep = elasticity_report(panel_all)
    blocked = ([] if rep.empty else
               rep.loc[~rep["verdict_deployed"], "cell"].tolist())

    print("\n[2b] Building the comparable (gated) bake-off ...")
    summary_gated, overall_gated, overall_unbal, gate_notes, bal_pairs = \
        build_gated(bt, blocked_cells=blocked)
    summary_gated.to_csv(RES_DIR / "bakeoff_by_cell_gated.csv", index=False)
    if not overall_gated.empty:
        overall_gated.to_csv(RES_DIR / "bakeoff_overall_gated.csv", index=False)
    gate_notes.to_csv(RES_DIR / "bakeoff_gate_notes.csv", index=False)
    fig_overall(overall_gated if not overall_gated.empty else overall_unbal, gated=True)
    # [FIX-31] blocked cells passed in so the figure can label the gaps
    fig_heat(summary_gated, blocked_cells=blocked)

    print("\n  How each cell was treated:")
    print(gate_notes.to_string(index=False))

    if overall_gated.empty:
        print("\n  >>> No (cell, origin) pair carries every method, so a balanced overall")
        print("  >>> ranking does not exist. Per-cell tables are the only valid comparison.")
        print("\nUNBALANCED OVERALL (reference only, methods averaged over different sets):")
        print(overall_unbal.round(3).to_string(index=False))
    else:
        print(f"\nBALANCED OVERALL RANKING ({len(bal_pairs)} cell-origin pairs carrying "
              f"every method) -- headline:")
        print(overall_gated.round(3).to_string(index=False))
        print(f"  pairs: {bal_pairs}")
        best = overall_gated.iloc[0]
        nl = overall_gated[overall_gated["method"] == "naive_last"]["mase"]
        baseline = float(nl.iloc[0]) if len(nl) else np.nan
        if not np.isfinite(baseline):
            print(">>> naive_last absent; no baseline comparison.")
        elif best["mase"] >= baseline - 1e-9:
            print(">>> naive_last is the best method on the balanced set; the ratio is")
            print(">>> effectively flat and added complexity buys nothing.")
        else:
            gain = 100 * (baseline - best["mase"]) / baseline
            print(f">>> {best['method']} beats naive_last by {gain:.1f}% MASE "
                  f"({best['mase']:.3f} vs {baseline:.3f}) on the balanced set.")
        if len(bal_pairs) < 4:
            print(">>> CAUTION: the balanced set is small. Treat the overall ranking as")
            print(">>> indicative and rely on the per-cell tables for selection.")

        if not overall_unbal.empty:
            print("\nUNBALANCED OVERALL (reference only; n_pairs differs by method):")
            print(overall_unbal.round(3).to_string(index=False))
            print(">>> Methods here are averaged over different cell-origin sets, so the")
            print(">>> gaps between them are not interpretable. Use the balanced table.")

    print("\n[3] COVID sensitivity of winners ...")
    covid_sensitivity(summary_gated, blocked_cells=blocked)

    print("\n[4] Selecting per-cell winners and projecting the ratio 2025-2040 ...")
    # [FIX-21] selection runs on the gated table
    winners, detail = pick_winners(summary_gated, blocked_cells=blocked)
    detail.to_csv(RES_DIR / "cell_winner_selection.csv", index=False)
    print(detail.round(3).to_string(index=False))
    # [FIX-29] separate the two reasons the winner can differ from the best MASE.
    forced = detail[detail["growth_ratio_blocked"] &
                    (detail["best_method"] == "growth_ratio")]
    if not forced.empty:
        print("\n  NOTE: growth_ratio had the lowest MASE here but was rejected on")
        print("  identification grounds, so the selected method forecasts worse:")
        for _, r in forced.iterrows():
            print(f"    {r['cell']}: selected {r['winner']} MASE {r['winner_mase']:.3f} "
                  f"vs growth_ratio {r['best_mase']:.3f}")
        print("  This is deliberate and must be stated in the write-up.")

    tb = detail[detail["tie_break_applied"] & ~detail["growth_ratio_blocked"].isna()]
    tb = tb[tb["best_method"] != "growth_ratio"]
    if not tb.empty:
        print("\n  NOTE: tie-break applied (methods within 2% of the best MASE are")
        print("  treated as equivalent and the simpler one is preferred):")
        for _, r in tb.iterrows():
            gapp = 100 * (r["winner_mase"] / r["best_mase"] - 1) if r["best_mase"] else 0
            print(f"    {r['cell']}: picked {r['winner']} ({r['winner_mase']:.3f}) over "
                  f"{r['best_method']} ({r['best_mase']:.3f}), {gapp:+.1f}%")

    sel_ratio, panel = project_selected(winners, panel_all)

    print("\n[5] Combining with block-01 enrolment forecast -> demand ...")
    df_sel, betas = build_selected_demand(sel_ratio, winners, panel, rep)
    (df_sel[["cell", "nuts3_code", "year_start", "scenario", "R_selected", "method"]]
     .drop_duplicates(subset=["cell", "nuts3_code", "year_start", "scenario"])
     .to_csv(RES_DIR / "selected_ratio_projection.csv", index=False))
    implied_ratio_path(df_sel, panel)
    reg, nat = build_totals(df_sel)
    fig_selected(df_sel, winners)

    print("\n[6] Writing contract and running consistency checks ...")
    write_contract(reg, nat, df_sel.rename(columns={"year_start": "year"}))
    run_checks(df_sel.rename(columns={"year_start": "year"}), reg, nat)
    fig_national(nat)
    bench = dgeec_comparison(nat)

    cen = nat[nat.scenario == "central"].sort_values("year")
    print("\nSELECTED national demand (central):")
    print(cen.round(0).to_string(index=False))
    if len(cen) >= 2:
        a, b = cen.iloc[0], cen.iloc[-1]
        print(f"\nselected      {a['year']}->{b['year']}: "
              f"{a['demand_selected']:,.0f} -> {b['demand_selected']:,.0f} "
              f"({100*(b['demand_selected']/a['demand_selected']-1):+.1f}%)")
        print(f"proportional  {a['year']}->{b['year']}: "
              f"{a['demand_proportional']:,.0f} -> {b['demand_proportional']:,.0f} "
              f"({100*(b['demand_proportional']/a['demand_proportional']-1):+.1f}%)")
        gr_used = [c for c, m in winners.items() if m == "growth_ratio"]
        bc = [c for c in nat.columns if c.startswith("demand_beta")]
        endv = {c: float(b[c]) for c in bc if np.isfinite(b.get(c, np.nan))}
        if not gr_used:
            print("elasticity range: not applicable. No cell deploys growth_ratio, so the")
            print("  beta variants coincide with the selected path by construction. The")
            print("  elasticity assumption is inactive and the only live comparison is")
            print("  selected vs proportional above.")
        elif len(set(round(v) for v in endv.values())) <= 1:
            print(f"elasticity range in {int(b['year'])}: the beta variants agree to the")
            print(f"  nearest post. growth_ratio is deployed in {gr_used} but the")
            print("  specifications do not disagree materially.")
        else:
            lo_k = min(endv, key=endv.get); hi_k = max(endv, key=endv.get)
            spread = endv[hi_k] - endv[lo_k]
            print(f"elasticity range in {int(b['year'])} (growth_ratio in {gr_used}): "
                  f"{endv[lo_k]:,.0f} ({lo_k}) to {endv[hi_k]:,.0f} ({hi_k})")
            print(f"  spread {spread:,.0f} posts = "
                  f"{100*spread/float(b['demand_selected']):.1f}% of the selected level. "
                  f"Report this as a range.")
        print("The gap between the lines is the elasticity choice, not demography.")
    print("Winners per cell:", winners)

    with open(RES_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"horizon": HORIZON, "exclude_years": EXCLUDE_YEARS,
                   "rolling_origins": ROLLING_ORIGINS, "long_diff_h": LONG_DIFF_H,
                   "two_way_fe": ELASTICITY_TWO_WAY_FE,
                   "elasticity_band": ELASTICITY_BAND,
                   "elasticity_t_min": ELASTICITY_T_MIN,
                   "elasticity_h_grid": list(ELASTICITY_H_GRID),
                   "wild_boot_reps": WILD_BOOT_REPS,
                   "wild_boot_alpha": WILD_BOOT_ALPHA,
                   "require_reject_proportionality": REQUIRE_REJECT_PROPORTIONALITY,
                   "gate_bakeoff_beta": GATE_BAKEOFF_BETA,
                   "heatmap_reference": HEATMAP_REFERENCE,
                   "heatmap_ratio_clip": HEATMAP_RATIO_CLIP,
                   "beta_scenarios": BETA_SCENARIOS,
                   "betas_deployed": betas,
                   "reporting_scenarios": REPORTING_SCENARIOS,
                   "demo_exclude": sorted(DEMO_EXCLUDE),
                   "include_post_secondary": INCLUDE_POST_SECONDARY,
                   "post_secondary_evidence": ps_info,
                   "dgeec_comparison": bench,
                   "winners": winners, "blocked_cells": blocked,
                   "check_results": _RESULTS}, f, indent=2, default=str)
    print("=" * 78)
    print("Done. Results in:", DEMAND_DIR, "| figures in:", IMG_DIR)
    print("=" * 78)


if __name__ == "__main__":
    main()
