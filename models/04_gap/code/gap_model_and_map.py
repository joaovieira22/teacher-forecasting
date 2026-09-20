
from pathlib import Path
import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================ PATHS
BASE = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
DATA         = BASE / r"models\00_data"
STUDENTS_DIR = BASE / r"models\01_students\results"
DEMAND_DIR   = BASE / r"models\02_demand\results"
SUPPLY_DIR   = BASE / r"models\03_supply\results"
RES = BASE / r"models\04_gap\results"
IMG = BASE / r"models\04_gap\images"

MASTER_FILE       = DATA / "master_panel_nuts3_with_age.xlsx"
STUDENTS_FILE     = STUDENTS_DIR / "students_forecast_nuts3.xlsx"
GRAD_FILE         = DATA / "teacher_supply_panel.xlsx"
TEACHERS_AGE_FILE = DATA / "teachers_panel.xlsx"

if os.environ.get("GAP_TEST") == "1":
    DATA = Path("."); STUDENTS_DIR = Path("."); DEMAND_DIR = Path("."); SUPPLY_DIR = Path(".")
    RES = Path("gap_out"); IMG = Path("gap_out")
    MASTER_FILE = Path("master_panel_nuts3_with_age.xlsx")
    STUDENTS_FILE = Path("students_forecast_nuts3.xlsx")
    GRAD_FILE = Path("teacher_supply_panel.xlsx")
    TEACHERS_AGE_FILE = Path("teachers_panel.xlsx")
RES.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

# ============================================================ CONFIG
HORIZON = (2025, 2040)

RECONCILE_NUTS3 = True
RECONCILE_TOL   = 0.001

GRAD_POOL_COL   = "teacher_supply_core"
ROLL_GRAD_YEARS = 3
LAG_YEARS       = 1

GRAD_PROJECT_MODE  = "constant"   # "constant" | "damped"
GRAD_FREEZE_WINDOW = 5
GRAD_TREND_WINDOW  = 10
GRAD_DAMPING       = 0.85

SUPPLY_ENTRIES_FILE = "supply_entries_history.csv"
SUPPLY_CHANNEL_FILE = "supply_recruitment_by_channel.csv"
ENTRY_IS_STOCK   = True
ENTRY_BAND_YEARS = 6.0

# [GAP-1] which entry flow feeds the conversion ratio.
ENTRY_CHANNEL = "young"

# [GAP-4] the lateral channel comes from the contracted reserve.
MODEL_LATERAL_CHANNEL = True
LATERAL_RATE_WINDOW   = 5

ENTRY_ANCHOR        = 2000.0
ENTRY_ANCHOR_WINDOW = 3
ENTRY_ANCHOR_NOTE   = ("~2,000/yr: masters-in-teaching flow (1,668-2,069) and "
                       "first-time placements (2,651) -- both GRADUATE-channel")

CONV_WINDOW         = 3
RATE_SLOPE_LOOKBACK = 3
RATE_MODE           = "last"
RATE_DAMPING        = 0.85
RATE_FLOOR          = 0.005

COHORT_CONV_CEILING = 0.85
RATE_CEILING = COHORT_CONV_CEILING / ROLL_GRAD_YEARS

SMOOTH_RECRUITMENT = True
SMOOTH_WINDOW      = 3
# [GAP-35] tolerancia para o residuo que a suavizacao introduz no acumulado
SMOOTH_RESIDUAL_TOL = 0.001

EXTERNAL_STOCK_BENCHMARK      = 122000.0
EXTERNAL_STOCK_BENCHMARK_YEAR = 2024
EXTERNAL_STOCK_BENCHMARK_NOTE = ("DGEEC + Nova SBE 2025: ~122k active teachers, "
                                 "mainland, 2024/25")
UNIVERSE_TOL = 0.05

SUBSTITUTION_RESERVE_SHARE = 0.20
SUBSTITUTION_NOTE = ("EDULOG: contracted reserve ~20% of the corps; for 2031 it "
                     "projects ~8.7k permanent posts unfilled against ~15.7k "
                     "substitutions to cover")

STOCK_GAP        = True
RETIRE_OFFICIAL  = "base"
DEMO_EXCLUDE     = {"no_migration"}

UIS_REF_REPLACEMENT = 0.58
CONVENTION_TOL = 0.05

# [GAP-36] nome do ficheiro da figura da onda, para reconciliar com o texto
WAVE_FIGURE_NAME = "retirement_wave_vs_recruitment.png"

INK = "#1a1a1a"; RETIRE = "#b5171e"; RECRUIT = "#128a4b"; MID = "#3b4a5a"
LIGHT = "#7aa6c2"; LATERAL = "#e08a2c"

NUTS3_NAMES = {
    "111": "Alto Minho", "112": "Cavado", "119": "Ave", "11A": "AM Porto",
    "11B": "Alto Tamega e Barroso", "11C": "Tamega e Sousa", "11D": "Douro",
    "11E": "Terras de Tras-os-Montes",
    "150": "Algarve", "191": "Regiao de Aveiro", "192": "Regiao de Coimbra",
    "193": "Regiao de Leiria",
    "194": "Viseu Dao Lafoes", "195": "Beira Baixa", "196": "Beiras e Serra da Estrela",
    "1A0": "AM Lisboa", "1B0": "Peninsula de Setubal", "1C1": "Alentejo Litoral",
    "1C2": "Baixo Alentejo", "1C3": "Alto Alentejo", "1C4": "Alentejo Central",
    "1D1": "Oeste", "1D2": "Medio Tejo", "1D3": "Leziria do Tejo",
}

RATE_MODES = ["median", "last", "damped_trend"]
MODE_LABELS = {"median": "median of last {}y".format(CONV_WINDOW),
               "last": "last obs. (constant)",
               "damped_trend": "damped trend"}
MODE_COLORS = {"median": LIGHT, "last": RETIRE, "damped_trend": MID}

AGE_BANDS = ["lt25", "25_29", "30_34", "35_39", "40_44",
             "45_49", "50_54", "55_59", "60_plus"]

# [GAP-32] forma admissivel de um nome de coluna de escalao etario.
# Aceita teachers_lt25, teachers_25_29, teachers_60_plus, teachers_65p, etc.
# NAO aceita teachers_pre_school, teachers_basic_1, teachers_math, ...
AGE_BAND_PATTERN = re.compile(
    r"^teachers_(lt\d{2}|\d{2}_\d{2}|\d{2}_plus|\d{2}p|under\d{2}|over\d{2})$")
MIN_AGE_BANDS = 5   # abaixo disto nao ha perfil etario utilizavel

_MANIFEST = {}


# ============================================================ UTILS
def find_col(cols, *keys):
    low = {c: str(c).lower() for c in cols}
    for k in keys:
        for c in cols:
            if k in low[c]:
                return c
    return None


def find_col_exact(cols, *names):
    """[GAP-1] Exact match first: `new_entrants` is a prefix of
    `new_entrants_young`, so substring search picks the wrong column."""
    lower = {str(c).lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def pick_col(cols, exact_names, fuzzy_keys=(), label="", required=False):
    """[GAP-33][GAP-34] Exact first, substring only as a declared fallback.

    Returns (column, how) where `how` is 'exact', 'fuzzy' or None, so the caller
    can print which column it actually read. A fuzzy hit on a prefix is the
    failure mode that put one entry channel in place of the total need.
    """
    c = find_col_exact(cols, *exact_names)
    if c is not None:
        return c, "exact"
    if fuzzy_keys:
        c = find_col(cols, *fuzzy_keys)
        if c is not None:
            print(f"[read] WARNING: no exact match for {list(exact_names)}"
                  + (f" ({label})" if label else "")
                  + f"; falling back to '{c}' by substring. Verify it is the")
            print(f"[read] quantity you meant and not a channel or a component "
                  f"[GAP-33]")
            return c, "fuzzy"
    if required:
        print(f"[read] ERROR: none of {list(exact_names)} found"
              + (f" ({label})" if label else ""))
    return None, None


def detect_file(search_dir, must_have, prefer=None):
    for name in (prefer or []):
        p = Path(search_dir) / name
        if p.exists():
            return p
    for g in (glob.glob(str(Path(search_dir) / "*.csv"))
              + glob.glob(str(Path(search_dir) / "*.xlsx"))):
        try:
            head = pd.read_csv(g, nrows=1) if g.endswith(".csv") else pd.read_excel(g, nrows=1)
        except Exception:
            continue
        cols = [str(c).lower() for c in head.columns]
        if all(any(m in c for c in cols) for m in must_have):
            return Path(g)
    return None


def read_any(path, sheet=None):
    if str(path).endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)


def nice_region(code):
    return NUTS3_NAMES.get(str(code), str(code))


def safe_ratio(num, den):
    num = pd.to_numeric(pd.Series(num), errors="coerce")
    den = pd.to_numeric(pd.Series(den), errors="coerce")
    den = den.replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def sign_verdict(series):
    """[GAP-18] shortage / surplus / flips / none, not a boolean."""
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    valid = s.dropna()
    n_missing = int(s.isna().sum())
    if valid.empty:
        return "none", 0, n_missing
    if (valid > 0).all():
        return "shortage", int(len(valid)), n_missing
    if (valid < 0).all():
        return "surplus", int(len(valid)), n_missing
    return "flips", int(len(valid)), n_missing


VERDICT_TEXT = {
    "shortage": "shortage in all {n} years",
    "surplus": "SURPLUS in all {n} years",
    "flips": "sign flips across years",
    "none": "no valid years",
}


def resolve_recruitment(d, label):
    """[GAP-33] Total recruitment need, never one channel standing in for it.

    Order: exact `recruitment` / `recruitment_need`; failing that, the SUM of the
    two channel columns, stated as such; failing that, a declared substring
    fallback. The previous code went straight to substring and would return
    `recruitment_young` whenever `recruitment` was absent.
    """
    c, how = pick_col(d.columns, ("recruitment", "recruitment_need",
                                  "recruitment_total"), label=label)
    if c is not None:
        return pd.to_numeric(d[c], errors="coerce"), f"{c} ({how})"

    ry = find_col_exact(d.columns, "recruitment_young", "need_young")
    rl = find_col_exact(d.columns, "recruitment_lateral", "need_lateral")
    if ry and rl:
        print(f"[read] {label}: no total recruitment column; summing the two "
              f"channel columns")
        print(f"[read] '{ry}' + '{rl}'. This is a reconstruction, not a "
              f"published total  [GAP-33]")
        return (pd.to_numeric(d[ry], errors="coerce")
                + pd.to_numeric(d[rl], errors="coerce")), f"{ry}+{rl} (summed)"

    c, how = pick_col(d.columns, (), fuzzy_keys=("recruit", "entrada"),
                      label=label, required=True)
    if c is None:
        return None, None
    return pd.to_numeric(d[c], errors="coerce"), f"{c} ({how})"


# ============================================================ SUPPLY (03)
def load_supply_nuts3():
    path = detect_file(SUPPLY_DIR, must_have=["year", "recruit"],
                       prefer=["supply_universe_nuts3.csv", "supply_universe_nuts2.csv"])
    if path is None:
        print("[supply NUTS3] not found -> national only"); return None
    d = read_any(path)
    yc, _ = pick_col(d.columns, ("year", "ano"), fuzzy_keys=("year", "ano"))
    reg = find_col(d.columns, "region", "nuts3", "regiao")
    ret, _ = pick_col(d.columns, ("retirements_wave", "retirements"),
                      fuzzy_keys=("retire", "reforma", "wave"), label="NUTS3 wave")
    stk, _ = pick_col(d.columns, ("stock",), fuzzy_keys=("stock", "oferta"),
                      label="NUTS3 stock")
    rec, rec_src = resolve_recruitment(d, "supply NUTS3")
    if yc is None or rec is None:
        print("[supply NUTS3] year/recruit columns missing -> skipped"); return None
    out = pd.DataFrame({
        "year": pd.to_numeric(d[yc], errors="coerce").astype("Int64"),
        "region": d[reg].astype(str) if reg else "TOTAL",
        "recruitment_need": rec,
    })
    out["retirements_wave"] = pd.to_numeric(d[ret], errors="coerce") if ret else np.nan
    out["stock"] = pd.to_numeric(d[stk], errors="coerce") if stk else np.nan
    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])].dropna(subset=["year"])
    if out.empty:
        print("[supply NUTS3] no rows inside the horizon -> skipped"); return None
    out["year"] = out["year"].astype(int)
    print(f"[supply NUTS3] OK ({out.region.nunique()} regions, {len(out)} rows; "
          f"need from {rec_src})")
    return out


def load_supply_national():
    prefer = ["supply_projection.csv", "supply_universe_national.csv"]
    path = detect_file(SUPPLY_DIR, must_have=["year", "recruit"], prefer=prefer)
    if path is None:
        print("[supply national] not found"); return None
    d = read_any(path)
    yc, _ = pick_col(d.columns, ("year", "ano"), fuzzy_keys=("year", "ano"))
    ret, ret_how = pick_col(d.columns, ("retirements_wave", "retirements"),
                            fuzzy_keys=("retire", "reforma", "wave"),
                            label="national wave")
    stk, _ = pick_col(d.columns, ("stock",), fuzzy_keys=("stock",),
                      label="national stock")
    ex, ex_how = pick_col(d.columns, ("exits", "saidas"),
                          fuzzy_keys=("exit", "saida"), label="national exits")
    rec, rec_src = resolve_recruitment(d, "supply national")
    if yc is None or rec is None:
        print("[supply national] year/recruit columns missing"); return None

    out = pd.DataFrame({
        "year": pd.to_numeric(d[yc], errors="coerce").astype("Int64"),
        "recruitment_need": rec,
    })
    out["retirements_wave"] = pd.to_numeric(d[ret], errors="coerce") if ret else np.nan
    out["stock"] = pd.to_numeric(d[stk], errors="coerce") if stk else np.nan
    out["exits"] = pd.to_numeric(d[ex], errors="coerce") if ex else np.nan

    ry = find_col_exact(d.columns, "recruitment_young")
    rl = find_col_exact(d.columns, "recruitment_lateral")
    if ry and rl:
        out["need_young"] = pd.to_numeric(d[ry], errors="coerce")
        out["need_lateral"] = pd.to_numeric(d[rl], errors="coerce")

    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])].dropna(subset=["year"])
    if out.empty:
        print("[supply national] no rows inside the horizon"); return None
    out["year"] = out["year"].astype(int)

    # [GAP-34] say which columns were read, so a wrong pick is visible
    print(f"[supply national] need = {rec_src} | exits = "
          f"{ex or 'MISSING'} | wave = {ret or 'MISSING'}")
    _MANIFEST["columns_national"] = {"recruitment": rec_src, "exits": ex,
                                     "retirements_wave": ret, "stock": stk}

    if RETIRE_OFFICIAL == "seniority":
        wv = detect_file(SUPPLY_DIR, must_have=["year"], prefer=["supply_wave_v3.csv"])
        if wv is not None:
            w = read_any(wv)
            wy = find_col(w.columns, "year", "ano")
            wc = find_col(w.columns, "retire55", "wave", "reforma")
            if wy and wc:
                w = w[[wy, wc]].rename(columns={wy: "year", wc: "retirements_wave"})
                w["year"] = pd.to_numeric(w["year"], errors="coerce").astype("Int64")
                out = out.drop(columns=["retirements_wave"]).merge(w, on="year", how="left")
                print("[retirement wave] official = seniority (supply_wave_v3)")
    else:
        print("[retirement wave] official = base (retirements_wave)")

    out = out.sort_values("year").reset_index(drop=True)
    out["recruitment_need_unsmoothed"] = out["recruitment_need"]
    if SMOOTH_RECRUITMENT:
        base = out["exits"] if out["exits"].notna().any() else out["retirements_wave"]
        if base is not None and base.notna().any():
            expansion = out["recruitment_need"] - base
            w = max(3, SMOOTH_WINDOW | 1)
            exp_sm = expansion.rolling(window=w, center=True, min_periods=1).mean()
            out["recruitment_need"] = base + exp_sm
            d0 = out["recruitment_need_unsmoothed"].sum(); d1 = out["recruitment_need"].sum()
            rel = abs(d1 - d0) / max(abs(d0), 1.0)
            print(f"[smoothing] expansion with centred MA{w} "
                  f"(cumulative {d0:,.0f} -> {d1:,.0f}, diff {d1-d0:+.2f} = "
                  f"{rel:.4%})")
            # [GAP-35] min_periods=1 at the edges does not preserve the total
            if rel > SMOOTH_RESIDUAL_TOL:
                print(f"[smoothing] WARNING: the residual exceeds "
                      f"{SMOOTH_RESIDUAL_TOL:.1%}. Centred averaging with")
                print(f"[smoothing] min_periods=1 averages the first and last "
                      f"years over fewer observations,")
                print(f"[smoothing] so it redistributes AND changes the total. "
                      f"Report the unsmoothed cumulative  [GAP-35]")
            _MANIFEST["smoothing_residual_rel"] = float(rel)
    print(f"[supply national] OK ({len(out)} years) -- UNCONSTRAINED need from 03")
    return out


def load_channel_shares():
    """[GAP-4][GAP-30] Young/lateral split of the projected need."""
    p = Path(SUPPLY_DIR) / SUPPLY_CHANNEL_FILE
    if not p.exists():
        return None, np.nan
    try:
        d = read_any(p)
    except Exception as e:
        print(f"[channels] failed to read {SUPPLY_CHANNEL_FILE}: {e}")
        return None, np.nan
    yc = find_col(d.columns, "year", "ano")
    ry = find_col_exact(d.columns, "recruitment_young")
    rl = find_col_exact(d.columns, "recruitment_lateral")
    rt = find_col_exact(d.columns, "recruitment")
    if not (yc and ry and rl):
        return None, np.nan
    out = pd.DataFrame({
        "year": pd.to_numeric(d[yc], errors="coerce").astype("Int64"),
        "need_young": pd.to_numeric(d[ry], errors="coerce"),
        "need_lateral": pd.to_numeric(d[rl], errors="coerce"),
    })
    if rt:
        out["need_total_03"] = pd.to_numeric(d[rt], errors="coerce")
    out = out.dropna(subset=["year"])
    out["year"] = out["year"].astype(int)
    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])]
    if out.empty:
        return None, np.nan
    tot = out["need_young"] + out["need_lateral"]
    out["young_share"] = safe_ratio(out["need_young"], tot).values
    share = float(out["young_share"].mean())
    print(f"[channels] young/lateral split of the PROJECTED NEED, read from "
          f"{SUPPLY_CHANNEL_FILE}")
    print(f"[channels]   young share of projected need = {share:.1%} "
          f"(03's forward-looking split)")
    return out, share


def _age_band_columns(cols):
    """[GAP-32] Only names shaped like an age band.

    The old fallback took every `teachers_*` except `teachers_total`. On the real
    panel that sweeps in teachers_pre_school, teachers_basic_1, teachers_basic_2,
    teachers_basic_3_secondary, teachers_public, teachers_private and the
    subject columns, which are DIFFERENT partitions of the same people. Measured
    on the calibrated fixture: 139,098 becomes 250,376, a factor of 1.80, and the
    error surfaces downstream as a fabricated perimeter gap.
    """
    exact = [c for c in (f"teachers_{b}" for b in AGE_BANDS) if c in cols]
    if len(exact) >= MIN_AGE_BANDS:
        return exact, "declared AGE_BANDS"
    shaped = [c for c in cols if AGE_BAND_PATTERN.match(str(c).lower())]
    # teachers_55_plus overlaps 55_59 and 60_plus; drop open-ended duplicates
    if any(str(c).lower().endswith(("_59", "_64")) for c in shaped):
        shaped = [c for c in shaped
                  if not re.match(r"^teachers_(5[0-9]|6[0-9])_plus$", str(c).lower())
                  or str(c).lower() == "teachers_60_plus"]
    if len(shaped) >= MIN_AGE_BANDS:
        print(f"[stock] age bands matched by shape, not by the declared list: "
              f"{shaped}")
        return shaped, "pattern"
    return [], None


def historical_stock(basis="bands"):
    """[GAP-11][GAP-16][GAP-32] Teaching stock by year, on the AGE-BAND universe."""
    if not Path(MASTER_FILE).exists():
        return None, None
    try:
        xls = pd.ExcelFile(MASTER_FILE, engine="openpyxl")
        sh = "historical" if "historical" in xls.sheet_names else xls.sheet_names[0]
        h = pd.read_excel(MASTER_FILE, sheet_name=sh, engine="openpyxl")
    except Exception:
        return None, None
    h.columns = [str(c).strip() for c in h.columns]
    yc = find_col(h.columns, "year_start", "ano", "year")
    if yc is None:
        return None, None
    h[yc] = pd.to_numeric(h[yc], errors="coerce")

    band_cols, how = _age_band_columns(h.columns)
    tc = find_col_exact(h.columns, "teachers_total")

    if not band_cols:
        # [GAP-32] refuse rather than guess
        print("[stock] NO usable age-band columns found. The previous fallback")
        print("[stock] summed every teachers_* column except teachers_total,")
        print("[stock] which mixes cycle, sector and subject partitions of the")
        print("[stock] same people and inflates the stock (1.80x on the test")
        print("[stock] panel). Refusing to guess  [GAP-32]")
        if tc:
            h["_tot"] = pd.to_numeric(h[tc], errors="coerce")
            g = h.dropna(subset=[yc]).groupby(yc)["_tot"].sum()
            tot_by_year = {int(k): float(v) for k, v in g.items()
                           if np.isfinite(v) and v > 0}
            print("[stock] falling back to teachers_total, a DIFFERENT and larger")
            print("[stock] universe than the one 03 projects  [GAP-16]")
            return tot_by_year, {"basis": "teachers_total", "n_band_columns": 0,
                                 "has_teachers_total": True,
                                 "age_bands_unavailable": True}
        return None, None

    bands_by_year = tot_by_year = None
    h["_bands"] = h[band_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    g = h.dropna(subset=[yc]).groupby(yc)["_bands"].sum()
    bands_by_year = {int(k): float(v) for k, v in g.items()
                     if np.isfinite(v) and v > 0}
    if tc:
        h["_tot"] = pd.to_numeric(h[tc], errors="coerce")
        g = h.dropna(subset=[yc]).groupby(yc)["_tot"].sum()
        tot_by_year = {int(k): float(v) for k, v in g.items()
                       if np.isfinite(v) and v > 0}

    chosen = bands_by_year if (basis == "bands" and bands_by_year) else \
        (tot_by_year or bands_by_year)
    if not chosen:
        return None, None

    meta = {"basis": basis if (basis == "bands" and bands_by_year) else "teachers_total",
            "n_band_columns": len(band_cols), "band_match": how,
            "band_columns": list(band_cols),
            "has_teachers_total": bool(tc)}
    print(f"[stock] age-band columns used ({how}): {len(band_cols)}  [GAP-32]")
    if bands_by_year and tot_by_year:
        common = sorted(set(bands_by_year) & set(tot_by_year))
        if common:
            last = common[-1]
            b, t = bands_by_year[last], tot_by_year[last]
            meta.update({"last_year": last, "bands_sum": b, "teachers_total": t,
                         "coverage_factor": t / b if b else np.nan,
                         "unbanded": t - b,
                         "unbanded_share": (t - b) / t if t else np.nan})
            print(f"\n[stock] the panel carries TWO stock definitions at {last}:")
            print(f"[stock]   sum of age bands : {b:11,.0f}  <- what 03 projects")
            print(f"[stock]   teachers_total   : {t:11,.0f}")
            print(f"[stock]   difference       : {t-b:+11,.0f}  "
                  f"({100*(t-b)/t:.1f}% of the panel carry no age band)")
            print(f"[stock] using the AGE-BAND sum, so every downstream comparison is")
            print(f"[stock] on the same universe as 03's projection  [GAP-16]")
            if t < b:
                print(f"[stock] WARNING: teachers_total is SMALLER than the band sum.")
                print(f"[stock] One of the two is not what its name says  [GAP-32]")
    return chosen, meta


# ============================================================ UNIVERSE
def check_universe(nat, hist_stock=None, stock_meta=None):
    """[GAP-7][GAP-13][GAP-17][GAP-32] Perimeter check, like-for-like in time."""
    if nat is None or nat.empty or EXTERNAL_STOCK_BENCHMARK is None:
        return None
    s = nat.sort_values("year")
    proj_year = int(s["year"].iloc[0])
    proj_stock = (float(s["stock"].iloc[0])
                  if ("stock" in s.columns and s["stock"].notna().any()) else np.nan)

    bench_year = int(EXTERNAL_STOCK_BENCHMARK_YEAR)
    obs_stock = float(hist_stock.get(bench_year, np.nan)) if hist_stock else np.nan

    like_for_like = np.isfinite(obs_stock)
    base_stock = obs_stock if like_for_like else proj_stock
    base_year = bench_year if like_for_like else proj_year
    if not np.isfinite(base_stock):
        print("[universe] no usable stock series -> comparison skipped")
        return None

    gap = base_stock - EXTERNAL_STOCK_BENCHMARK
    rel = gap / EXTERNAL_STOCK_BENCHMARK

    print("\n" + "=" * 74)
    print("[GAP-7] UNIVERSE RECONCILIATION -- is this a perimeter gap or a real one?")
    print("=" * 74)
    if like_for_like:
        print(f"  chain stock, OBSERVED {base_year} (same year as the benchmark) : "
              f"{base_stock:11,.0f}")
    else:
        print(f"  chain stock, PROJECTED {base_year} (after that year's recruitment): "
              f"{base_stock:11,.0f}")
        print("  NOTE: the historical panel has no observation for the benchmark year,")
        print("        so this comparison is NOT like-for-like in time  [GAP-13]")
    print(f"  external benchmark {bench_year}                            : "
          f"{EXTERNAL_STOCK_BENCHMARK:11,.0f}")
    print(f"  source: {EXTERNAL_STOCK_BENCHMARK_NOTE}")
    print(f"  difference                                        : {gap:+11,.0f}  ({rel:+.1%})")

    # [GAP-32] a perimeter claim is only meaningful if the base was read correctly
    if stock_meta and stock_meta.get("age_bands_unavailable"):
        print("  >>> CAUTION: the age-band columns were not found, so this base is")
        print("  >>> teachers_total, a different universe from the one 03 projects.")
        print("  >>> Do not attribute this difference to private provision before")
        print("  >>> checking the column names  [GAP-32]")
    elif stock_meta and stock_meta.get("band_match") == "pattern":
        print("  >>> NOTE: age bands were matched by shape, not by the declared")
        print("  >>> list. Confirm the set before quoting this figure  [GAP-32]")

    flow_ok = None
    implied_exits = np.nan
    if like_for_like and np.isfinite(proj_stock):
        drop = base_stock - proj_stock
        rec0 = (float(s["recruitment_need"].iloc[0])
                if "recruitment_need" in s.columns else np.nan)
        ex0 = (float(s["exits"].iloc[0])
               if ("exits" in s.columns and s["exits"].notna().any()) else np.nan)
        print(f"  for reference, the PROJECTED {proj_year} base is {proj_stock:,.0f} "
              f"({proj_stock - base_stock:+,.0f} vs the observed {base_year} stock)")
        if np.isfinite(rec0):
            implied_exits = rec0 + drop
            print(f"  if that were one year of flows, exits in {proj_year} would have to be")
            print(f"  {implied_exits:,.0f} = {100*implied_exits/base_stock:.1f}% of the "
                  f"workforce", end="")
            if np.isfinite(ex0):
                print(f", against the {ex0:,.0f} the chain actually reports")
                flow_ok = abs(implied_exits - ex0) <= 0.25 * max(ex0, 1.0)
            else:
                print("")
                flow_ok = implied_exits <= 0.08 * base_stock
            if flow_ok:
                print(f"  >>> consistent: the {proj_year} base does follow from the "
                      f"{base_year} stock by flows  [GAP-17]")
            else:
                print(f"  >>> NOT consistent with flows. The two series are on DIFFERENT")
                print(f"  >>> DEFINITIONS, so part of the difference above is internal to")
                print(f"  >>> the panel and is not a perimeter question at all  [GAP-17]")

    if abs(rel) > UNIVERSE_TOL:
        print(f"  >>> The two bases differ by more than {UNIVERSE_TOL:.0%}. Until this is")
        print("  >>> explained, every level reported downstream carries the same relative")
        print("  >>> error. Candidate causes, in order of likelihood:")
        print("  >>>   - private establishments included in the panel, excluded externally")
        print("  >>>   - autonomous regions included in the panel, mainland-only externally")
        print("  >>>   - headcount vs FTE")
        print("  >>>   - 'active' defined differently (contracted reserve in or out)")
    else:
        print("  >>> The two bases agree within tolerance; the perimeter is not the issue.")
    return {"comparison_year": base_year, "chain_stock_at_comparison": base_stock,
            "like_for_like_in_time": bool(like_for_like),
            "projected_base_year": proj_year, "projected_base_stock": proj_stock,
            "external_benchmark": EXTERNAL_STOCK_BENCHMARK, "external_year": bench_year,
            "difference": gap, "relative_difference": rel,
            "perimeter_flag": bool(abs(rel) > UNIVERSE_TOL),
            "base_to_projection_is_flows": (bool(flow_ok) if flow_ok is not None else np.nan),
            "implied_exits_if_flows": float(implied_exits),
            "stock_basis": (stock_meta or {}).get("basis"),
            "n_band_columns": (stock_meta or {}).get("n_band_columns"),
            "note": EXTERNAL_STOCK_BENCHMARK_NOTE}


# ============================================================ RECONCILIATION
def reconcile_nuts3(sup, nat):
    """[GAP-5] Rake the regional need onto the national total, DIAGNOSING the cause."""
    if sup is None or sup.empty or nat is None or nat.empty:
        return sup, None
    s = sup.copy()
    s["recruitment_need_raw"] = s["recruitment_need"]
    if not RECONCILE_NUTS3:
        print("[reconcile] DISABLED -> regional series left unreconciled")
        return s, None

    nat_by_year = nat.set_index("year")["recruitment_need"].to_dict()
    unsm_by_year = (nat.set_index("year")["recruitment_need_unsmoothed"].to_dict()
                    if "recruitment_need_unsmoothed" in nat.columns else {})
    reg_by_year = s.groupby("year")["recruitment_need_raw"].sum().to_dict()

    rows, factors = [], {}
    for y, reg_tot in reg_by_year.items():
        nat_tot = float(nat_by_year.get(y, np.nan))
        unsm = float(unsm_by_year.get(y, np.nan))
        reg_tot = float(reg_tot)
        factors[y] = (nat_tot / reg_tot) if (np.isfinite(nat_tot) and reg_tot > 0) else 1.0
        rows.append({"year": int(y), "regional_sum_raw": reg_tot,
                     "national_smoothed": nat_tot, "national_unsmoothed": unsm,
                     "raking_factor": factors[y],
                     "excess_vs_smoothed": reg_tot - nat_tot if np.isfinite(nat_tot) else np.nan,
                     "excess_vs_unsmoothed": reg_tot - unsm if np.isfinite(unsm) else np.nan,
                     "excess_pct": (reg_tot / nat_tot - 1.0)
                                   if (np.isfinite(nat_tot) and nat_tot > 0) else np.nan})
    audit = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)

    s["raking_factor"] = s["year"].map(factors)
    s["recruitment_need"] = s["recruitment_need_raw"] * s["raking_factor"]

    tot_raw = float(audit["regional_sum_raw"].sum())
    tot_nat = float(audit["national_smoothed"].sum())
    tot_unsm = float(audit["national_unsmoothed"].sum())
    tot_new = float(s["recruitment_need"].sum())
    resid = abs(tot_new - tot_nat) / max(tot_nat, 1.0)
    excess_pct = (tot_raw / tot_nat - 1.0) if tot_nat > 0 else np.nan

    print(f"\n[reconcile] regional sum {tot_raw:,.0f} vs national (smoothed) "
          f"{tot_nat:,.0f} (excess {tot_raw-tot_nat:+,.0f} = {100*excess_pct:+.1f}%)")

    cause = "unknown"
    if np.isfinite(tot_unsm) and tot_unsm > 0:
        gap_unsm = abs(tot_raw - tot_unsm) / tot_unsm
        print(f"[reconcile] regional sum vs national BEFORE 04's smoothing: "
              f"{tot_raw:,.0f} vs {tot_unsm:,.0f} ({100*(tot_raw/tot_unsm-1):+.2f}%)")
        if gap_unsm < 1e-4:
            cause = "04_smoothing"
            print("[reconcile] DIAGNOSIS: the regional series already matches 03's own")
            print("            national total. The residual is created by the smoothing")
            print("            applied in THIS block, not by regional zero-flooring.")
        else:
            cause = "regional_floor_or_upstream"
            print("[reconcile] DIAGNOSIS: the regional sum differs from the national total")
            print("            even before 04 smooths. That points upstream: regions floor")
            print("            entry at zero and cannot absorb negative national expansion.")
    print(f"[reconcile] proportional raking applied per year "
          f"(factors {audit['raking_factor'].min():.4f}-{audit['raking_factor'].max():.4f})")
    print(f"[reconcile] after raking: {tot_new:,.0f} vs {tot_nat:,.0f} "
          f"-> residual {resid:.2%} {'PASS' if resid <= RECONCILE_TOL else 'FAIL'}")
    audit.attrs["cause"] = cause
    return s, audit


# ============================================================ DEMAND (02)
def load_demand_real():
    path = detect_file(DEMAND_DIR, must_have=["year"],
                       prefer=["teacher_demand_selected.csv", "teacher_demand.csv"])
    if path is None:
        return None
    d = read_any(path)
    yc  = find_col(d.columns, "year", "ano")
    vc  = find_col_exact(d.columns, "demand_selected") or \
          find_col(d.columns, "demand", "procura", "required")
    sc  = find_col(d.columns, "scenario", "cenario")
    reg = find_col(d.columns, "nuts3", "region", "regiao")
    if yc is None or vc is None:
        return None
    out = pd.DataFrame({
        "year": pd.to_numeric(d[yc], errors="coerce").astype("Int64"),
        "demand": pd.to_numeric(d[vc], errors="coerce"),
    })
    out["scenario"] = d[sc].astype(str).str.lower() if sc else "central"
    out["region"] = d[reg].astype(str) if reg else "TOTAL"
    out = out[(out.year >= HORIZON[0]) & (out.year <= HORIZON[1])].dropna(subset=["year"])
    if out.empty:
        return None
    out["year"] = out["year"].astype(int)
    out = out.groupby(["scenario", "region", "year"], as_index=False)["demand"].sum()
    print(f"[demand] 02_demand real OK (scenarios={sorted(out.scenario.unique())})")
    return out


def _observed_ratios():
    if not Path(MASTER_FILE).exists():
        return None
    try:
        xls = pd.ExcelFile(MASTER_FILE, engine="openpyxl")
        sh = "historical" if "historical" in xls.sheet_names else xls.sheet_names[0]
        h = pd.read_excel(MASTER_FILE, sheet_name=sh, engine="openpyxl")
    except Exception as e:
        print(f"[proxy] failed to read master: {e}"); return None
    h.columns = [str(c).strip() for c in h.columns]
    yc = find_col(h.columns, "year_start", "ano", "year")
    if yc is None:
        return None
    h[yc] = pd.to_numeric(h[yc], errors="coerce")
    if h[yc].dropna().empty:
        return None
    last = int(h[yc].max()); hb = h[h[yc] == last]
    def col(*ks): return find_col(h.columns, *ks)
    def ssum(c): return pd.to_numeric(hb[c], errors="coerce").sum() if c else np.nan
    ratios = {}
    try:
        ratios["pre_school"] = ssum(col("students_pre_school")) / ssum(col("teachers_pre_school"))
        ratios["basic_1"] = ssum(col("students_basic_1")) / ssum(col("teachers_basic_1"))
        ratios["basic_2"] = ssum(col("students_basic_2")) / ssum(col("teachers_basic_2"))
        s_sec = col("students_secondary")
        num = ssum(col("students_basic_3")) + (ssum(s_sec) if s_sec else 0.0)
        ratios["basic_3_secondary"] = num / ssum(col("teachers_basic_3_secondary"))
    except Exception:
        return None
    ratios = {k: v for k, v in ratios.items() if np.isfinite(v) and v > 0}
    return ratios or None


def build_demand_proxy():
    ratios = _observed_ratios()
    if ratios is None or not Path(STUDENTS_FILE).exists():
        print("[proxy] unavailable"); return None
    try:
        xls = pd.ExcelFile(STUDENTS_FILE, engine="openpyxl")
        sh = "students_long" if "students_long" in xls.sheet_names else xls.sheet_names[0]
        s = pd.read_excel(STUDENTS_FILE, sheet_name=sh, engine="openpyxl")
    except Exception as e:
        print(f"[proxy] failed to read students: {e}"); return None
    s.columns = [str(c).strip() for c in s.columns]
    yc  = find_col(s.columns, "year_start", "ano", "year")
    sc  = find_col(s.columns, "scenario", "cenario")
    reg = find_col(s.columns, "nuts3_code", "nuts3", "region")
    cyc = find_col(s.columns, "cycle", "ciclo")
    val = find_col(s.columns, "students_central", "students", "central", "value")
    if not all([yc, cyc, val]):
        return None
    s = s.copy()
    s["year"] = pd.to_numeric(s[yc], errors="coerce").astype("Int64")
    s["scenario"] = s[sc].astype(str).str.lower() if sc else "central"
    s["region"] = s[reg].astype(str) if reg else "TOTAL"
    s["students"] = pd.to_numeric(s[val], errors="coerce")
    def to_cell(c):
        c = str(c).lower()
        if "pre_school" in c or "pre-school" in c or "pre_escolar" in c: return "pre_school"
        if "basic_1" in c: return "basic_1"
        if "basic_2" in c: return "basic_2"
        if "basic_3" in c or "secondary" in c: return "basic_3_secondary"
        return None
    s["cell"] = s[cyc].map(to_cell)
    s = s[s["cell"].notna()].copy()
    if s.empty:
        return None
    s = s.groupby(["scenario", "region", "year", "cell"], as_index=False)["students"].sum()
    s["ratio"] = s["cell"].map(ratios)
    s = s.dropna(subset=["ratio"])
    s["demand_cell"] = safe_ratio(s["students"], s["ratio"]).values
    dem = (s.groupby(["scenario", "region", "year"], as_index=False)["demand_cell"].sum()
             .rename(columns={"demand_cell": "demand"}))
    dem = dem[(dem.year >= HORIZON[0]) & (dem.year <= HORIZON[1])]
    if dem.empty:
        return None
    dem["year"] = dem["year"].astype(int)
    print(f"[proxy] demand proxy built ({dem.region.nunique()} regions)")
    return dem


def get_demand():
    d = load_demand_real()
    if d is not None:
        d["source"] = "02_demand"; return d
    d = build_demand_proxy()
    if d is not None:
        d["source"] = "proxy_students_ratio"
    return d


# ============================================================ GRADUATES
def load_graduates_history():
    if not Path(GRAD_FILE).exists():
        print("[pipeline] teacher_supply_panel.xlsx not found"); return None
    try:
        g = pd.read_excel(GRAD_FILE)
    except Exception as e:
        print(f"[pipeline] failed to read the graduate panel: {e}"); return None
    g.columns = [str(c).strip() for c in g.columns]
    yc = find_col(g.columns, "year_start", "ano", "year")
    pool = GRAD_POOL_COL if GRAD_POOL_COL in g.columns else \
        find_col(g.columns, "supply_core", "supply_stem", "grad_education")
    if yc is None or pool is None:
        print("[pipeline] graduate columns not recognised"); return None
    d = g[[yc, pool]].copy(); d.columns = ["year", "grads"]
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["grads"] = pd.to_numeric(d["grads"], errors="coerce")
    d = d.dropna().sort_values("year")
    if d.empty:
        print("[pipeline] graduate panel has no usable rows"); return None
    print(f"[pipeline] graduate pool column = '{pool}': "
          f"{int(d.year.min())}-{int(d.year.max())} (last={d.grads.iloc[-1]:,.0f}/yr)")
    return d


def project_graduates(grad_h):
    """[GAP-31] Returns (projection dict, meta)."""
    if grad_h is None or len(grad_h) < 3:
        return None, None
    d = grad_h.sort_values("year")
    last_y = int(d["year"].max())
    last_obs = float(d["grads"].iloc[-1])
    proj = {int(y): float(g) for y, g in zip(d["year"], d["grads"])}

    if GRAD_PROJECT_MODE == "constant":
        lvl = float(d.tail(GRAD_FREEZE_WINDOW)["grads"].mean())
        for y in range(last_y + 1, HORIZON[1] + 1):
            proj[y] = lvl
        print(f"[pipeline] graduates projected CONSTANT at the last "
              f"{GRAD_FREEZE_WINDOW}-yr mean = {lvl:,.0f}/yr")
        meta = {"mode": "constant", "level_used": lvl, "last_observation": last_obs,
                "level_vs_last": (lvl / last_obs - 1.0) if last_obs else np.nan}
        return proj, meta

    lg = np.log(d["grads"].clip(lower=1.0).values)
    w = d.tail(GRAD_TREND_WINDOW)
    xw = w["year"].values.astype(float)
    lw = np.log(w["grads"].clip(lower=1.0).values)
    b = float(np.polyfit(xw - xw.max(), lw, 1)[0]) if (len(xw) >= 2 and np.ptp(xw) > 0) else 0.0
    last_l = float(lg[-1]); cum = 0.0
    for i, y in enumerate(range(last_y + 1, HORIZON[1] + 1), start=1):
        cum += (GRAD_DAMPING ** i) * b
        proj[y] = float(np.exp(last_l + cum))
    print(f"[pipeline] graduates projected with a DAMPED TREND "
          f"({proj[last_y]:,.0f} -> {proj[HORIZON[1]]:,.0f})")
    fut = [proj[y] for y in range(HORIZON[0], HORIZON[1] + 1) if y in proj]
    meta = {"mode": "damped", "level_used": float(np.mean(fut)) if fut else np.nan,
            "last_observation": last_obs}
    meta["level_vs_last"] = ((meta["level_used"] / last_obs - 1.0)
                             if (last_obs and np.isfinite(meta["level_used"])) else np.nan)
    return proj, meta


def graduate_pool(grad_dict, year):
    if not grad_dict:
        return np.nan
    yrs = [year - LAG_YEARS - k for k in range(ROLL_GRAD_YEARS)]
    vals = [grad_dict.get(y) for y in yrs]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return np.nan
    return float(sum(vals)) * (ROLL_GRAD_YEARS / len(vals))


# ============================================================ ENTRIES
def load_entries_history():
    """[GAP-1][GAP-30] Annual entry flow BY CHANNEL."""
    p = Path(SUPPLY_DIR) / SUPPLY_ENTRIES_FILE
    if p.exists():
        try:
            d = read_any(p)
            yc = find_col(d.columns, "year", "ano")
            c_young = find_col_exact(d.columns, "new_entrants_young")
            c_lat   = find_col_exact(d.columns, "new_entrants_lateral")
            c_tot   = find_col_exact(d.columns, "new_entrants")
            if yc and (c_young or c_tot):
                out = pd.DataFrame({"year": pd.to_numeric(d[yc], errors="coerce")})
                if c_young:
                    out["entries_young"] = pd.to_numeric(d[c_young], errors="coerce")
                if c_lat:
                    out["entries_lateral"] = pd.to_numeric(d[c_lat], errors="coerce")
                if c_tot:
                    out["entries_total"] = pd.to_numeric(d[c_tot], errors="coerce")

                if ENTRY_CHANNEL == "young" and c_young:
                    out["entries"] = out["entries_young"]
                    chan = "new_entrants_young"
                else:
                    if ENTRY_CHANNEL == "young":
                        print("[pipeline] WARNING: ENTRY_CHANNEL='young' but the young "
                              "column is absent; falling back to the TOTAL series.")
                    out["entries"] = out.get("entries_total", np.nan)
                    chan = c_tot or "new_entrants"

                out = out.dropna(subset=["year", "entries"])
                out = out.groupby("year", as_index=False).sum()
                out = out[out["entries"] > 0]
                if len(out) >= 4:
                    print(f"[pipeline] entries from 03_supply ({SUPPLY_ENTRIES_FILE}), "
                          f"channel = '{chan}'  [GAP-1]")
                    sh = np.nan
                    if c_young and c_tot:
                        sh = float(out["entries_young"].sum() / out["entries_total"].sum())
                        print(f"[pipeline]   young share of OBSERVED HISTORICAL ENTRIES "
                              f"= {sh:.1%}; the other {1-sh:.1%} is lateral")
                        print(f"[pipeline]   and does NOT come from the "
                              f"initial-teacher-education pool.")
                        print(f"[pipeline]   NOTE: this is NOT the same statistic as the "
                              f"young share of projected need")
                        print(f"[pipeline]   reported by [channels] above. Different "
                              f"numerator, different period  [GAP-21]")
                    if ENTRY_CHANNEL == "total":
                        print("[pipeline]   WARNING: using the TOTAL series puts lateral")
                        print("[pipeline]   entrants in a ratio whose denominator never")
                        print("[pipeline]   contained them. Reported for comparison only.")
                    return out, f"03_supply:{chan}", False, sh
        except Exception as e:
            print(f"[pipeline] failed to read {SUPPLY_ENTRIES_FILE}: {e}")

    cand = detect_file(SUPPLY_DIR, must_have=["year"],
                       prefer=["supply_projection.csv", "supply_hazards.csv"])
    if cand is not None:
        try:
            d = read_any(cand)
            yc = find_col(d.columns, "year", "ano")
            ec = find_col(d.columns, "new_entrant", "entrant", "inflow", "colocad")
            if yc and ec:
                out = d[[yc, ec]].copy(); out.columns = ["year", "entries"]
                out["year"] = pd.to_numeric(out["year"], errors="coerce")
                out["entries"] = pd.to_numeric(out["entries"], errors="coerce")
                out = out.dropna().groupby("year", as_index=False)["entries"].sum()
                out = out[out["entries"] > 0]
                if len(out) >= 4:
                    print(f"[pipeline] entries (flow) from {Path(cand).name}/{ec}")
                    return out, f"flow:{Path(cand).name}:{ec}", False, np.nan
        except Exception:
            pass

    if Path(TEACHERS_AGE_FILE).exists():
        try:
            xls = pd.ExcelFile(TEACHERS_AGE_FILE, engine="openpyxl")
            t = pd.read_excel(TEACHERS_AGE_FILE, sheet_name=xls.sheet_names[0],
                              engine="openpyxl")
            t.columns = [str(c).strip() for c in t.columns]
            yc = find_col(t.columns, "year_start", "ano", "year")
            lt25 = find_col_exact(t.columns, "teachers_lt25") or \
                find_col(t.columns, "teachers_lt25", "lt25")
            if yc and lt25:
                cols = [lt25]
                y2529 = find_col_exact(t.columns, "teachers_25_29")
                if y2529:
                    cols.append(y2529)
                use = t[[yc] + cols].copy()
                use[yc] = pd.to_numeric(use[yc], errors="coerce")
                for c in cols:
                    use[c] = pd.to_numeric(use[c], errors="coerce")
                use["entries"] = use[cols].sum(axis=1)
                out = use.dropna(subset=[yc]).groupby(yc, as_index=False)["entries"].sum()
                out.columns = ["year", "entries"]
                out = out[out["entries"] > 0]
                src = "teachers_lt25" + ("+25_29" if y2529 else "")
                print(f"[pipeline] FALLBACK entries (STOCK proxy) = {src}")
                return out, f"stock_proxy:{src}", True, np.nan
        except Exception as e:
            print(f"[pipeline] entries proxy failed: {e}")

    print("[pipeline] NO historical entries -> rate not estimable")
    return None, None, False, np.nan


def apply_stock_to_flow(entr_h, is_stock):
    if entr_h is None or not is_stock:
        return entr_h
    out = entr_h.copy()
    if ENTRY_IS_STOCK and ENTRY_BAND_YEARS and ENTRY_BAND_YEARS > 0:
        out["entries"] = out["entries"] / float(ENTRY_BAND_YEARS)
        print(f"[pipeline] STOCK->FLOW: entries / {ENTRY_BAND_YEARS:.0f} band years")
    return out


def apply_entry_anchor(entr_h):
    """[GAP-2][GAP-24][GAP-27] Rescale the GRADUATE entry flow to the external level."""
    if entr_h is None or entr_h.empty or ENTRY_ANCHOR is None:
        return entr_h, 1.0
    out = entr_h.sort_values("year").copy()
    recent = float(out.tail(ENTRY_ANCHOR_WINDOW)["entries"].mean())
    if (not np.isfinite(recent)) or recent <= 0:
        print("[anchor] recent mean not usable -> no calibration")
        return out, 1.0
    factor = float(ENTRY_ANCHOR) / recent

    out["entries_unanchored"] = out["entries"]
    out["entries"] = out["entries"] * factor
    if "entries_young" in out.columns and ENTRY_CHANNEL == "young":
        out["entries_young_unanchored"] = out["entries_young"]
        out["entries_young"] = out["entries_young"] * factor

    direction = "up" if factor > 1 else "down"
    print(f"[anchor] entries rescaled {direction}: last {ENTRY_ANCHOR_WINDOW}-yr mean "
          f"{recent:,.0f} -> {ENTRY_ANCHOR:,.0f}/yr (factor {factor:.3f})")
    print(f"[anchor] basis: {ENTRY_ANCHOR_NOTE}")
    if ENTRY_CHANNEL != "young":
        print("[anchor] WARNING: the anchor is a GRADUATE-channel quantity but the series")
        print("[anchor] being scaled is the TOTAL flow. That is not like-for-like.")
    elif abs(factor - 1.0) > 0.5:
        print(f"[anchor] NOTE: the factor is far from 1. The inferred series and the")
        print(f"[anchor] external source disagree by {abs(factor-1):.0%} on the level.")
    if {"entries_young", "entries_lateral", "entries_total"} <= set(out.columns):
        print(f"[anchor] the LATERAL column is deliberately left unscaled, so")
        print(f"[anchor] young + lateral no longer equals total in this frame. Use")
        print(f"[anchor] entries_*_unanchored for the historical identity  [GAP-27]")
    print(f"[anchor] CONSEQUENCE: the conversion rate is entries/pool and the pool is")
    print(f"[anchor] unchanged, so every estimated rate is multiplied by this same")
    print(f"[anchor] {factor:.3f}. The rate is the observed rate times the anchor, not an")
    print(f"[anchor] independent estimate  [GAP-24]")
    return out, factor


# ============================================================ LATERAL CHANNEL
def estimate_lateral_rate(entr_h, master_stock=None):
    """[GAP-4][GAP-10][GAP-16][GAP-27] Lateral entries over the PREVIOUS year's stock."""
    if entr_h is None or entr_h.empty or "entries_lateral" not in entr_h.columns:
        return None
    if master_stock is None or not master_stock:
        return None
    rows = []
    for _, r in entr_h.iterrows():
        y = int(r["year"])
        lat = float(r.get("entries_lateral", np.nan))
        stk = master_stock.get(y - 1, np.nan)
        if np.isfinite(lat) and np.isfinite(stk) and stk > 0:
            rows.append({"year": y, "entries_lateral": lat, "stock_prev_year": stk,
                         "lateral_rate": lat / stk})
    if not rows:
        return None
    m = pd.DataFrame(rows).sort_values("year")
    rate = float(m.tail(LATERAL_RATE_WINDOW)["lateral_rate"].mean())
    print(f"\n[lateral] lateral entries / PREVIOUS-year teaching stock, last "
          f"{LATERAL_RATE_WINDOW}y mean = {rate:.3%}")
    print(f"[lateral] this channel is drawn from the CONTRACTED RESERVE, not from")
    print(f"[lateral] initial teacher education, so it scales with the workforce.")
    print(f"[lateral] denominator = AGE-BAND stock(y-1), the same universe 03")
    print(f"[lateral] projects and the same year the rate is applied to "
          f"[GAP-10][GAP-16]")
    print(f"[lateral] NOT anchored: the external anchor is a graduate-channel")
    print(f"[lateral] quantity and does not apply here  [GAP-27]")
    return {"rate": rate, "hist": m, "basis": "stock_prev_year_agebands"}


# ============================================================ CONVERSION RATE
def _project_rate_paths(m):
    m = m.sort_values("year")
    recent = m.tail(CONV_WINDOW)
    med = float(recent["rate"].median())
    last_y = int(m["year"].max()); last_r = float(m["rate"].iloc[-1])

    sl = m.tail(RATE_SLOPE_LOOKBACK)
    if len(sl) >= 2 and np.ptp(sl["year"].values) > 0:
        slope = float(np.polyfit(sl["year"].values - sl["year"].values.max(),
                                 sl["rate"].values, 1)[0])
    else:
        slope = 0.0

    paths = {mode: {} for mode in RATE_MODES}
    for y in range(HORIZON[0], HORIZON[1] + 1):
        paths["median"][y] = float(np.clip(med, RATE_FLOOR, RATE_CEILING))
        paths["last"][y]   = float(np.clip(last_r, RATE_FLOOR, RATE_CEILING))
        h = max(y - last_y, 0)
        cum = sum((RATE_DAMPING ** k) * slope for k in range(1, h + 1))
        paths["damped_trend"][y] = float(np.clip(last_r + cum, RATE_FLOOR, RATE_CEILING))
    return paths, med, last_r, slope


def estimate_conversion(entr_h, grad_dict):
    """[GAP-1][GAP-26][GAP-31] rate(t) = GRADUATE entries(t) / graduate pool(t)."""
    if entr_h is None or entr_h.empty or not grad_dict:
        return None
    rows = []
    for _, r in entr_h.iterrows():
        t = int(r["year"]); pool = graduate_pool(grad_dict, t)
        ent = float(r["entries"])
        if np.isfinite(pool) and pool > 0 and np.isfinite(ent) and ent > 0:
            rate = ent / pool
            rows.append({"year": t, "entries": ent, "grad_pool": pool,
                         "rate": rate, "cohort_conversion": rate * ROLL_GRAD_YEARS})
    m = pd.DataFrame(rows)
    if m.empty:
        print("[pipeline] no entries/pool pairs -> rate not estimable"); return None

    paths, med, last_r, slope = _project_rate_paths(m)

    obs_min = float(m["cohort_conversion"].min())
    obs_max = float(m["cohort_conversion"].max())
    obs_med = float(m["cohort_conversion"].median())

    chan = "GRADUATE channel (<30)" if ENTRY_CHANNEL == "young" else "TOTAL entries"
    print(f"\n[RATIO] conversion rate = {chan} entries / graduate pool")
    print(f"[RATIO] pool = sum of {ROLL_GRAD_YEARS}y graduates, lag {LAG_YEARS}")
    print(f"[RATIO] estimated on the last {CONV_WINDOW} years: median={med:.1%} | "
          f"last obs={last_r:.1%} | slope={slope*100:+.2f} pp/yr")
    print(f"[RATIO] cohort conversion = rate x {ROLL_GRAD_YEARS}: "
          f"median={med*ROLL_GRAD_YEARS:.0%} | last={last_r*ROLL_GRAD_YEARS:.0%}")
    print(f"[RATIO] observed cohort conversion over the full history: "
          f"{obs_min:.0%}-{obs_max:.0%} (median {obs_med:.0%})")
    if obs_max > 1.0:
        print("[RATIO] WARNING: observed cohort conversion exceeds 100%. That is not a")
        print("[RATIO] behavioural finding, it means the numerator counts people the")
        print("[RATIO] denominator never contained. Check ENTRY_CHANNEL.  [GAP-1]")
    print(f"[RATIO] ceiling: cohort conversion capped at {COHORT_CONV_CEILING:.0%} "
          f"-> rate ceiling {RATE_CEILING:.1%}")
    if COHORT_CONV_CEILING >= 1.0:
        print("[RATIO] WARNING: a ceiling of 100% is not a constraint.")
    if CONV_WINDOW < 5:
        print(f"[RATIO] caution: the rate is estimated on only {CONV_WINDOW} observations.")

    hist_rates = m["rate"]
    official_base = {"median": med, "last": last_r}.get(RATE_MODE, last_r)
    pctile = float((hist_rates <= official_base).mean())
    print(f"[RATIO] the official mode sits at the {pctile:.0%} percentile of the "
          f"observed history")
    if pctile >= 0.95:
        print(f"[RATIO] that is at or near the MAXIMUM ever observed. Read alongside")
        print(f"[RATIO] the graduate-pool convention before calling it conservative "
              f"[GAP-31]")

    for mode in RATE_MODES:
        vals = np.array(list(paths[mode].values()))
        n_at = int(np.sum(vals >= RATE_CEILING - 1e-12))
        if n_at:
            print(f"[RATIO] mode '{mode}' is CAPPED at the ceiling in {n_at}/{len(vals)} "
                  f"years ({COHORT_CONV_CEILING:.0%} cohort conversion).")
    print(f"[RATIO] official mode = '{RATE_MODE}' ({MODE_LABELS.get(RATE_MODE, RATE_MODE)})")

    tail = m.tail(CONV_WINDOW)["rate"]
    q_lo, q_hi = float(tail.quantile(0.25)), float(tail.quantile(0.75))
    f_low = q_lo / official_base if official_base else 1.0
    f_high = q_hi / official_base if official_base else 1.0
    print(f"[RATIO] uncertainty band = the last {CONV_WINDOW}y interquartile RATIO "
          f"({f_low:.3f}x to {f_high:.3f}x)")
    print(f"[RATIO] applied to the '{RATE_MODE}' path. It is a proportional band "
          f"around that")
    print(f"[RATIO] path, not the raw 25-75 quantiles of the history  [GAP-26]")
    if f_low <= 1.0 and f_high <= 1.0:
        print(f"[RATIO] NOTE: both band factors are at or below 1, so the band sits")
        print(f"[RATIO] ENTIRELY BELOW the central path. That follows mechanically from")
        print(f"[RATIO] taking the maximum as the centre, and it means the projection")
        print(f"[RATIO] carries no representation of conversion rising further  [GAP-26]")

    return {"quantiles": {"low": q_lo, "central": med, "high": q_hi},
            "band_factors": {"low": f_low, "high": f_high},
            "official_base": official_base, "official_percentile": pctile,
            "hist": m, "paths": paths,
            "median": med, "last": last_r, "slope": slope,
            "obs_cohort_min": obs_min, "obs_cohort_max": obs_max}


# ============================================================ NEED SPLIT
def prepare_need_split(nat, channels=None, hist_stock=None):
    """[GAP-10][GAP-12][GAP-16] ONE need frame shared by every consumer."""
    n = nat.copy()
    if channels is not None:
        n = n.merge(channels[["year", "need_young", "need_lateral", "young_share"]],
                    on="year", how="left", suffixes=("", "_ch"))
        for c in ("need_young", "need_lateral"):
            if f"{c}_ch" in n.columns:
                n[c] = n[c].fillna(n[f"{c}_ch"])
                n = n.drop(columns=[f"{c}_ch"])

    if "need_young" not in n.columns or n["need_young"].isna().all():
        print("[split] WARNING: no channel split available. The graduate constraint")
        print("[split] would be applied to the TOTAL need, which is a perimeter error.")
        n["need_young"] = n["recruitment_need"]
        n["need_lateral"] = 0.0
        n["young_share"] = 1.0
        split_ok = False
    else:
        split_ok = True
        tot_ch = n["need_young"] + n["need_lateral"]
        scale = safe_ratio(n["recruitment_need"], tot_ch).fillna(1.0)
        n["need_young"] = n["need_young"] * scale
        n["need_lateral"] = n["need_lateral"] * scale
        if "young_share" not in n.columns:
            n["young_share"] = safe_ratio(n["need_young"],
                                          n["need_young"] + n["need_lateral"]).values
        print(f"[split] channel split rescaled to 04's smoothed total "
              f"(mean factor {float(scale.mean()):.4f}); ONE series now feeds the")
        print(f"[split] pipeline and the anchor test  [GAP-12]")

    n = n.sort_values("year").reset_index(drop=True)
    if "stock" in n.columns:
        prev = n["stock"].shift(1)
        first_year = int(n["year"].iloc[0])
        if hist_stock and (first_year - 1) in hist_stock:
            prev.iloc[0] = float(hist_stock[first_year - 1])
            nxt = float(n["stock"].iloc[0])
            jump = prev.iloc[0] / nxt - 1.0 if nxt else np.nan
            print(f"[split] stock({first_year-1}) from the historical panel, AGE-BAND "
                  f"basis = {prev.iloc[0]:,.0f}  [GAP-10][GAP-16]")
            if np.isfinite(jump) and abs(jump) > 0.05:
                print(f"[split]   WARNING: that is {jump:+.1%} against the projected "
                      f"{first_year} stock. The two")
                print(f"[split]   ends of this column may still be on different "
                      f"definitions; check [GAP-16].")
        else:
            prev.iloc[0] = n["stock"].iloc[0]
            print(f"[split] no historical stock for {first_year-1}; the first year "
                  f"falls back to its own stock  [GAP-10]")
        n["stock_prev_year"] = prev
    else:
        n["stock_prev_year"] = np.nan

    return n, split_ok


# ============================================================ ANCHOR ROBUSTNESS
def check_anchor_robustness(conv, grad_proj, need_frame, anchor_factor,
                            need_col="need_young"):
    """[GAP-3][GAP-12][GAP-24][GAP-25] Does the verdict depend on the anchor?"""
    if conv is None or not grad_proj or need_frame is None or need_frame.empty:
        return None
    if anchor_factor is None or not np.isfinite(anchor_factor) or anchor_factor <= 0:
        return None

    col = need_col if need_col in need_frame.columns else "recruitment_need"
    path = conv["paths"].get(RATE_MODE, conv["paths"]["last"])
    need_by_year = need_frame.set_index("year")[col].to_dict()

    gaps_a, gaps_u_capped, gaps_u_raw = [], [], []
    rates_a, rates_u = [], []
    any_capped = False
    for y in sorted(need_by_year):
        pool = graduate_pool(grad_proj, int(y))
        if not np.isfinite(pool):
            continue
        r_a = float(path.get(int(y), conv["last"]))
        r_u_raw = r_a / anchor_factor
        r_u = float(np.clip(r_u_raw, RATE_FLOOR, RATE_CEILING))
        any_capped = any_capped or (r_u_raw > RATE_CEILING)
        need = float(need_by_year[y])
        gaps_a.append(need - pool * r_a)
        gaps_u_capped.append(need - pool * r_u)
        gaps_u_raw.append(need - pool * r_u_raw)
        rates_a.append(r_a); rates_u.append(r_u_raw)
    if not gaps_a:
        return None

    v_a, _, _ = sign_verdict(gaps_a)
    v_u_cap, _, _ = sign_verdict(gaps_u_capped)
    v_u_raw, _, _ = sign_verdict(gaps_u_raw)
    ra, ru = float(np.mean(rates_a)), float(np.mean(rates_u))

    print("\n[GAP-3] anchor robustness -- does the VERDICT depend on the anchor?")
    print(f"    need series used: {col} (rescaled, same as the pipeline)  [GAP-12]")
    print(f"    rate path walked year by year, same as build_pipeline  [GAP-25]")
    print(f"    anchored, mean rate  = {ra:.2%}  (cohort {ra*ROLL_GRAD_YEARS:.0%})  "
          f"mean gap = {np.mean(gaps_a):+,.0f}/yr  [{v_a}]")
    print(f"    unanchored, uncapped = {ru:.2%}  (cohort {ru*ROLL_GRAD_YEARS:.0%})  "
          f"mean gap = {np.mean(gaps_u_raw):+,.0f}/yr  [{v_u_raw}]")
    if any_capped:
        print(f"    unanchored, capped   = mean gap "
              f"{np.mean(gaps_u_capped):+,.0f}/yr  [{v_u_cap}]")
    print(f"    the two rates differ by exactly the anchor factor "
          f"({ra:.4f} / {ru:.4f} = {anchor_factor:.3f}), so this is not")
    print(f"    two estimates but one estimate and one rescaling  [GAP-24]")

    if v_a == v_u_raw:
        print(f"    -> the VERDICT is robust: both give '{v_a}'. The anchor moves the")
        print("       magnitude only.")
    elif any_capped and v_a == v_u_cap and v_a != v_u_raw:
        print("    -> CAUTION: the verdict only survives BECAUSE the ceiling binds. That")
        print("       is an assumption rescuing a result, not evidence.")
    else:
        print(f"    -> the VERDICT DEPENDS on the anchor: '{v_a}' anchored against "
              f"'{v_u_raw}' unanchored.")
        print("       Report this prominently; the external level is driving the sign.")
    return pd.DataFrame([{
        "rate_mode": RATE_MODE, "need_series": col, "need_series_rescaled": True,
        "path_walked_year_by_year": True,
        "entry_channel": ENTRY_CHANNEL, "anchor_factor": anchor_factor,
        "mean_rate_anchored": ra, "mean_rate_unanchored_raw": ru,
        "cohort_conv_anchored": ra * ROLL_GRAD_YEARS,
        "cohort_conv_unanchored_raw": ru * ROLL_GRAD_YEARS,
        "unanchored_hits_ceiling": bool(any_capped),
        "mean_gap_anchored": float(np.mean(gaps_a)),
        "mean_gap_unanchored_raw": float(np.mean(gaps_u_raw)),
        "mean_gap_unanchored_capped": float(np.mean(gaps_u_capped)),
        "verdict_anchored": v_a, "verdict_unanchored_raw": v_u_raw,
        "verdict_unanchored_capped": v_u_cap,
        "verdict_robust_without_cap": bool(v_a == v_u_raw),
        "verdict_depends_on_cap": bool(any_capped and v_a == v_u_cap and v_a != v_u_raw),
    }])


# ============================================================ PIPELINE
def build_pipeline(need_frame, grad_proj, conv, lateral=None):
    """[GAP-4][GAP-10][GAP-26] The graduate constraint binds on the YOUNG channel."""
    if need_frame is None or need_frame.empty or not grad_proj or conv is None:
        return None
    paths = conv["paths"]
    if RATE_MODE not in paths:
        print(f"[pipeline] unknown RATE_MODE '{RATE_MODE}' -> falling back to 'last'")
    central_path = paths.get(RATE_MODE, paths["last"])

    f_low = conv["band_factors"]["low"]
    f_high = conv["band_factors"]["high"]

    n = need_frame
    lat_rate = lateral["rate"] if lateral else np.nan

    rows = []
    for _, r in n.iterrows():
        y = int(r["year"])
        need_tot = float(r["recruitment_need"])
        need_y = float(r["need_young"]); need_l = float(r["need_lateral"])
        pool = graduate_pool(grad_proj, y)
        stock_prev = float(r.get("stock_prev_year", np.nan))
        rate_c = central_path.get(y, np.nan)
        if not np.isfinite(rate_c):
            continue
        for scn, rate in [("central", rate_c), ("low", rate_c * f_low),
                          ("high", rate_c * f_high)]:
            rate = float(np.clip(rate, RATE_FLOOR, RATE_CEILING))
            ent_y = pool * rate if np.isfinite(pool) else np.nan
            if MODEL_LATERAL_CHANNEL and np.isfinite(lat_rate) and np.isfinite(stock_prev):
                ent_l = lat_rate * stock_prev
            else:
                ent_l = need_l
            ent_tot = (ent_y + ent_l) if (np.isfinite(ent_y) and np.isfinite(ent_l)) else np.nan
            rows.append({
                "year": y, "conversion_scn": scn,
                "recruitment_need": need_tot,
                "need_young": need_y, "need_lateral": need_l,
                "graduate_pool": pool, "stock_prev_year": stock_prev,
                "conversion_rate": rate, "cohort_conversion": rate * ROLL_GRAD_YEARS,
                "feasible_young": ent_y, "feasible_lateral": ent_l,
                "expected_entrants": ent_tot,
                "gap_young": need_y - ent_y,
                "gap_lateral": need_l - ent_l,
                "pipeline_gap": need_tot - ent_tot,
            })
    pg = pd.DataFrame(rows)
    if pg.empty:
        print("[pipeline] no rows produced"); return None
    pg = pg.sort_values(["conversion_scn", "year"]).reset_index(drop=True)

    need_y_by_year = n.set_index("year")["need_young"].to_dict()
    need_t_by_year = n.set_index("year")["recruitment_need"].to_dict()
    stock_prev_by_year = n.set_index("year")["stock_prev_year"].to_dict()

    comp = []
    for mode in RATE_MODES:
        for y in range(HORIZON[0], HORIZON[1] + 1):
            pool = graduate_pool(grad_proj, y)
            rate = paths[mode].get(y, np.nan)
            need_y = float(need_y_by_year.get(y, np.nan))
            need_t = float(need_t_by_year.get(y, np.nan))
            ent_y = pool * rate if (np.isfinite(pool) and np.isfinite(rate)) else np.nan
            stk = float(stock_prev_by_year.get(y, np.nan))
            if MODEL_LATERAL_CHANNEL and np.isfinite(lat_rate) and np.isfinite(stk):
                ent_l = lat_rate * stk
            else:
                ent_l = (need_t - need_y) if (np.isfinite(need_t)
                                              and np.isfinite(need_y)) else np.nan
            comp.append({"year": y, "rate_mode": mode, "conversion_rate": rate,
                         "cohort_conversion": rate * ROLL_GRAD_YEARS
                         if np.isfinite(rate) else np.nan,
                         "feasible_young": ent_y, "feasible_lateral": ent_l,
                         "expected_entrants": (ent_y + ent_l)
                         if (np.isfinite(ent_y) and np.isfinite(ent_l)) else np.nan,
                         "need_young": need_y, "recruitment_need": need_t,
                         "gap_young": (need_y - ent_y)
                         if (np.isfinite(need_y) and np.isfinite(ent_y)) else np.nan,
                         "pipeline_gap": (need_t - ent_y - ent_l)
                         if (np.isfinite(need_t) and np.isfinite(ent_y)
                             and np.isfinite(ent_l)) else np.nan})
    return pg, pd.DataFrame(comp)


def rate_mode_sweep_unanchored(need_frame, grad_proj, conv, anchor_factor):
    """[GAP-28] The same mode sweep with the anchor removed."""
    if (conv is None or not grad_proj or need_frame is None or need_frame.empty
            or anchor_factor is None or not np.isfinite(anchor_factor)
            or anchor_factor <= 0):
        return None
    need_y = need_frame.set_index("year")["need_young"].to_dict()
    rows = []
    for mode in RATE_MODES:
        path = conv["paths"][mode]
        for y in sorted(need_y):
            pool = graduate_pool(grad_proj, int(y))
            r_raw = float(path.get(int(y), np.nan)) / anchor_factor
            if not (np.isfinite(pool) and np.isfinite(r_raw)):
                continue
            r = float(np.clip(r_raw, RATE_FLOOR, RATE_CEILING))
            rows.append({"year": int(y), "rate_mode": mode,
                         "conversion_rate_unanchored": r,
                         "cohort_conversion_unanchored": r * ROLL_GRAD_YEARS,
                         "feasible_young_unanchored": pool * r,
                         "gap_young_unanchored": float(need_y[y]) - pool * r})
    return pd.DataFrame(rows) if rows else None


def build_constraint_summary(pg):
    """[GAP-29] Unmet need per channel, with NO cross-channel credit."""
    if pg is None or pg.empty:
        return None
    c = pg[pg.conversion_scn == "central"].sort_values("year").copy()
    if c.empty:
        return None
    c["unmet_young"] = (c["need_young"] - c["feasible_young"]).clip(lower=0)
    c["unmet_lateral"] = (c["need_lateral"] - c["feasible_lateral"]).clip(lower=0)
    c["unmet_no_offset"] = c["unmet_young"] + c["unmet_lateral"]
    c["unmet_netted"] = (c["recruitment_need"] - c["expected_entrants"]).clip(lower=0)
    c["surplus_young"] = (c["feasible_young"] - c["need_young"]).clip(lower=0)
    c["surplus_lateral"] = (c["feasible_lateral"] - c["need_lateral"]).clip(lower=0)
    c["coverage_young"] = safe_ratio(c["feasible_young"], c["need_young"]).values
    c["coverage_lateral"] = safe_ratio(c["feasible_lateral"], c["need_lateral"]).values
    c["coverage_total"] = safe_ratio(c["expected_entrants"], c["recruitment_need"]).values
    c["restriction_binds"] = c["unmet_young"] > 0

    out = c[["year", "recruitment_need", "need_young", "need_lateral",
             "feasible_young", "feasible_lateral", "expected_entrants",
             "coverage_young", "coverage_lateral", "coverage_total",
             "unmet_young", "unmet_lateral", "unmet_no_offset", "unmet_netted",
             "surplus_young", "surplus_lateral", "restriction_binds"]].reset_index(drop=True)

    t_uy = float(out["unmet_young"].sum()); t_ny = float(out["need_young"].sum())
    t_ul = float(out["unmet_lateral"].sum()); t_nl = float(out["need_lateral"].sum())
    t_no = float(out["unmet_no_offset"].sum()); t_net = float(out["unmet_netted"].sum())
    t_nt = float(out["recruitment_need"].sum())
    bind_years = out.loc[out["restriction_binds"], "year"].tolist()

    print(f"\n[constraint] GRADUATE channel: unmet {t_uy:,.0f} of {t_ny:,.0f} "
          f"({100*t_uy/max(t_ny,1):.0f}%)")
    print(f"[constraint] LATERAL  channel: unmet {t_ul:,.0f} of {t_nl:,.0f} "
          f"({100*t_ul/max(t_nl,1):.0f}%)")
    print(f"[constraint] SUM OF CHANNELS, no cross-credit: {t_no:,.0f} of "
          f"{t_nt:,.0f} ({100*t_no/max(t_nt,1):.0f}%)  [GAP-29]")
    print(f"[constraint] netted across channels          : {t_net:,.0f} "
          f"({100*t_net/max(t_nt,1):.0f}%)")
    if t_no > t_net + 1.0:
        print(f"[constraint] the two differ by {t_no - t_net:,.0f}. That is a surplus in")
        print(f"[constraint] one channel being credited against a shortfall in the other.")
        print(f"[constraint] It assumes the channels substitute one for one, which the")
        print(f"[constraint] age-profile split does not support. Report the no-offset")
        print(f"[constraint] figure  [GAP-29]")
    if bind_years:
        print(f"[constraint] the graduate restriction binds in {len(bind_years)}/{len(out)} "
              f"years ({bind_years[0]}-{bind_years[-1]})")
    else:
        print("[constraint] the graduate restriction never binds")
    cy = out["coverage_young"].dropna()
    cl = out["coverage_lateral"].dropna()
    if not cy.empty:
        print(f"[constraint] coverage, graduate: {cy.iloc[0]:.0%} -> {cy.iloc[-1]:.0%}")
    if not cl.empty:
        print(f"[constraint] coverage, lateral : {cl.iloc[0]:.0%} -> {cl.iloc[-1]:.0%}")
    return out


def uis_decomposition(nat):
    """[GAP-22] Replacement/expansion, with the international benchmark qualified."""
    if nat is None or nat.empty:
        return None
    d = nat.copy()
    d["replacement"] = (d["exits"].clip(lower=0) if d["exits"].notna().any()
                        else d["retirements_wave"])
    d["expansion"] = d["recruitment_need"] - d["replacement"]
    tot = float(d["recruitment_need"].sum()); rep = float(d["replacement"].sum())
    d["share_replacement"] = safe_ratio(d["replacement"], d["recruitment_need"]).values
    if tot > 0:
        share = rep / tot
        print(f"[UIS] replacement={share:6.1%} | expansion={1-share:6.1%} "
              f"(intl. ref. ~{UIS_REF_REPLACEMENT:.0%}/{1-UIS_REF_REPLACEMENT:.0%})")
        if share > 1.0:
            print("[UIS] NOTE: expansion is NEGATIVE, so replacement exceeds 100%. The")
            print("[UIS] system is shrinking: total need is smaller than the replacement")
            print("[UIS] of leavers. The international reference describes systems with")
            print("[UIS] GROWING enrolment and is not a meaningful comparator here "
                  "[GAP-22]")
    return d[["year", "recruitment_need", "replacement", "expansion", "share_replacement"]]


def substitution_reserve_note(nat):
    """[GAP-8][GAP-11] Size the component this chain does NOT measure."""
    if nat is None or nat.empty or SUBSTITUTION_RESERVE_SHARE is None:
        return None
    s = nat.sort_values("year")
    stock0 = float(s["stock"].iloc[0]) if s["stock"].notna().any() else np.nan
    if not np.isfinite(stock0):
        return None
    reserve = stock0 * SUBSTITUTION_RESERVE_SHARE
    print("\n" + "=" * 74)
    print("[GAP-8] WHAT THIS BLOCK DOES NOT MEASURE: the substitution reserve")
    print("=" * 74)
    print(f"  the need modelled here is PERMANENT POSTS only")
    print(f"  implied reserve at {SUBSTITUTION_RESERVE_SHARE:.0%} of the "
          f"{int(s['year'].iloc[0])} stock : {reserve:,.0f} teachers")
    print(f"  reference: {SUBSTITUTION_NOTE}")
    print("  >>> The missing component is of the same order as the measured one, and")
    print("  >>> on the EDULOG 2031 figures it is LARGER. The level reported here is")
    print("  >>> therefore not the effective requirement of the system in any year.")
    print("  >>> This IS an estimate and it is published in substitution_reserve.csv;")
    print("  >>> text elsewhere in the chain should cite it rather than say that no")
    print("  >>> estimate exists.")
    return {"stock_base_year": int(s["year"].iloc[0]), "stock_base": stock0,
            "reserve_share": SUBSTITUTION_RESERVE_SHARE,
            "implied_reserve": reserve, "note": SUBSTITUTION_NOTE}


def build_stock_gap(dem, nat, hist_stock=None):
    """[GAP-6][GAP-15][GAP-19] Demographic pressure, with an honest baseline."""
    if not STOCK_GAP or dem is None or dem.empty or nat is None or nat.empty:
        return None
    s = nat.sort_values("year")
    base_year = int(s["year"].iloc[0])
    proj_stock = float(s["stock"].iloc[0]) if s["stock"].notna().any() else np.nan

    dem = dem[~dem["scenario"].astype(str).str.lower().isin(DEMO_EXCLUDE)]
    if dem.empty:
        return None
    dnat = dem.groupby(["scenario", "year"], as_index=False)["demand"].sum()

    if not np.isfinite(proj_stock):
        base = dnat[dnat.year == HORIZON[0]].set_index("scenario")["demand"]
        dnat["stock_gap"] = dnat.apply(
            lambda r: r["demand"] - base.get(r["scenario"], np.nan), axis=1)
        dnat["baseline"] = f"{HORIZON[0]} demand"
        print(f"[stock gap] baseline = {HORIZON[0]} demand (no stock series available)")
        return dnat

    dnat["stock_gap"] = dnat["demand"] - proj_stock
    dnat["baseline"] = f"projected stock {base_year} (after that year's recruitment)"
    print(f"[stock gap] baseline = PROJECTED stock in {base_year} = {proj_stock:,.0f}")
    print(f"[stock gap]   this is the stock AFTER {base_year} recruitment, not a")
    print(f"[stock gap]   no-replacement counterfactual  [GAP-6]")
    print(f"[stock gap]   the baseline is a CONSTANT, so 'stock_gap' is the demand path")
    print(f"[stock gap]   shifted by {proj_stock:,.0f} and contains no supply dynamics.")
    print(f"[stock gap]   the series with supply content is 'gap_vs_no_replacement' "
          f"[GAP-19]")

    if s["exits"].notna().any():
        start = np.nan; src = ""
        if hist_stock and (base_year - 1) in hist_stock:
            start = float(hist_stock[base_year - 1])
            src = f"observed AGE-BAND stock {base_year-1}"
        elif "recruitment_need" in s.columns:
            rec0 = float(s["recruitment_need"].iloc[0])
            start = proj_stock - rec0 + float(s["exits"].iloc[0])
            src = f"projected {base_year} stock minus that year's net flows"
        if np.isfinite(start):
            surv = start
            no_repl = {}
            for _, r in s.iterrows():
                y = int(r["year"])
                ex = float(r["exits"]) if np.isfinite(r["exits"]) else 0.0
                surv = max(surv - ex, 0.0)
                no_repl[y] = surv
            dnat["stock_no_replacement"] = dnat["year"].map(no_repl)
            dnat["gap_vs_no_replacement"] = dnat["demand"] - dnat["stock_no_replacement"]
            print(f"[stock gap]   no-replacement path starts from {src} = "
                  f"{start:,.0f} and subtracts exits from {base_year} onward  [GAP-15]")
            print(f"[stock gap]   NOTE: with demand ~= stock, this series converges on")
            print(f"[stock gap]   cumulative recruitment by construction. It is a")
            print(f"[stock gap]   restatement, not corroboration  [GAP-19]")
    return dnat


# ============================================================ FIGURES
def fig_wave_vs_recruitment(nat):
    if nat is None or nat.empty or nat["retirements_wave"].isna().all():
        print("[fig] wave vs recruitment: no data"); return
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(nat.year, nat.retirements_wave, "-o", color=RETIRE, lw=2.4, ms=6,
            label="retirement wave (55-59 + 60+)")
    ax.plot(nat.year, nat.recruitment_need, "-s", color=RECRUIT, lw=2.4, ms=6,
            label="recruitment need")
    ax.fill_between(nat.year, nat.retirements_wave, nat.recruitment_need,
                    where=(nat.recruitment_need >= nat.retirements_wave),
                    color=RECRUIT, alpha=0.08)
    ax.set_title("Retirement wave vs. recruitment need, national", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers per year")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout(); out = IMG / WAVE_FIGURE_NAME
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    _MANIFEST.setdefault("figures", {})["wave"] = WAVE_FIGURE_NAME
    print("[ok]", out)


def fig_cum_recruit_nuts3(sup):
    if sup is None or sup.empty:
        print("[fig] NUTS3 cumulative: no data"); return
    cum = (sup.groupby("region", as_index=False)["recruitment_need"].sum()
             .sort_values("recruitment_need"))
    cum["name"] = cum.region.map(nice_region)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.34 * len(cum))))
    ax.barh(cum.name, cum.recruitment_need, color=MID, alpha=0.9)
    for y, v in enumerate(cum.recruitment_need):
        ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8, color=INK)
    suffix = " (reconciled to national)" if RECONCILE_NUTS3 else ""
    ax.set_title(f"Cumulative recruitment need 2025-2040, by NUTS III{suffix}",
                 fontweight="bold")
    ax.set_xlabel("new teachers (horizon sum)")
    ax.grid(alpha=0.2, axis="x"); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); out = IMG / "cumulative_recruitment_nuts3.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    _MANIFEST.setdefault("figures", {})["nuts3"] = "cumulative_recruitment_nuts3.png"
    print("[ok]", out)


def fig_reconciliation(sup, audit):
    if audit is None or audit.empty or sup is None or sup.empty:
        print("[fig] reconciliation: no data"); return
    if "recruitment_need_raw" not in sup.columns:
        print("[fig] reconciliation: raw column absent"); return
    cum = (sup.groupby("region", as_index=False)[["recruitment_need_raw", "recruitment_need"]]
             .sum().sort_values("recruitment_need", ascending=False))
    cum["name"] = cum.region.map(nice_region)
    top = cum.head(12).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    ypos = np.arange(len(top))
    ax1.barh(ypos + 0.19, top["recruitment_need_raw"], height=0.38,
             color=LIGHT, alpha=0.95, label="raw (as delivered by 03)")
    ax1.barh(ypos - 0.19, top["recruitment_need"], height=0.38,
             color=MID, alpha=0.95, label="reconciled (sums to 04's national)")
    ax1.set_yticks(ypos); ax1.set_yticklabels(top["name"], fontsize=9)
    ax1.set_title(f"Raw vs. reconciled need, top {len(top)} of "
                  f"{cum.region.nunique()} regions", fontweight="bold")
    ax1.set_xlabel("new teachers 2025-2040")
    ax1.grid(alpha=0.2, axis="x"); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=9, loc="lower right")

    ax2.plot(audit.year, 100 * audit.excess_pct, "-o", color=RETIRE, lw=2.4, ms=6)
    ax2.axhline(0, color="0.4", lw=1, ls="--")
    ax2.set_title("Excess of the regional sum over the national total", fontweight="bold")
    ax2.set_xlabel("year"); ax2.set_ylabel("excess (%)")
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    vals = 100 * audit.excess_pct.dropna()
    if not vals.empty:
        ax2.set_ylim(bottom=min(0, float(vals.min()) * 1.1))
    cause = audit.attrs.get("cause", "unknown")
    cap = {"04_smoothing": "the regional series already matches 03's national total;\n"
                           "this residual is created by 04's own smoothing",
           "regional_floor_or_upstream": "regions floor entry at zero and cannot\n"
                                         "absorb the negative national expansion",
           }.get(cause, "cause not determined")
    ax2.text(0.98, 0.04, cap, transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8.5, color="0.4")
    fig.suptitle("NUTS III reconciliation: proportional raking, year by year",
                 fontweight="bold", y=1.01)
    fig.tight_layout(); out = IMG / "nuts3_reconciliation.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_pipeline(pg, split_ok=True, band_label=None):
    """[GAP-4][GAP-14][GAP-20][GAP-26] Graduate channel, bands only when non-empty."""
    if pg is None or pg.empty:
        print("[fig] pipeline: no data"); return
    sub = pg[pg.conversion_scn == "central"].sort_values("year")
    if sub.empty:
        print("[fig] pipeline: no central scenario"); return
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.plot(sub.year, sub.need_young, "-o", color=RETIRE, lw=2.4, ms=6,
            label="graduate-channel need (<30), from 03")
    r_end = float(sub["conversion_rate"].iloc[-1])
    ax.plot(sub.year, sub.feasible_young, "-s", color=RECRUIT, lw=2.4, ms=6,
            label=f"feasible graduate entries (rate '{RATE_MODE}', {r_end:.0%} = "
                  f"{r_end*ROLL_GRAD_YEARS:.0%} cohort conversion)")
    piv = pg.pivot_table(index="year", columns="conversion_scn", values="feasible_young")
    if {"low", "high"}.issubset(piv.columns):
        ax.fill_between(piv.index, piv["low"], piv["high"], color=RECRUIT, alpha=0.15,
                        label=band_label or "rate uncertainty band")

    deficit = (sub["need_young"] >= sub["feasible_young"])
    surplus = (sub["feasible_young"] > sub["need_young"])
    if bool(deficit.any()):
        ax.fill_between(sub.year, sub.feasible_young, sub.need_young, where=deficit,
                        color=RETIRE, alpha=0.10, label="graduate-pipeline deficit")
    if bool(surplus.any()):
        ax.fill_between(sub.year, sub.need_young, sub.feasible_young, where=surplus,
                        color=RECRUIT, alpha=0.10,
                        label="graduate-pipeline surplus (restriction slack)")

    ax.plot(sub.year, sub.recruitment_need, ":", color="0.45", lw=1.8,
            label="TOTAL need (all channels, for scale)")
    ax.set_ylim(bottom=0)
    ax.set_title("Graduate pipeline: the channel the ITE pool actually constrains",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("new teachers per year")
    ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    # [GAP-36] keep the legend off the dotted total line
    ax.legend(frameon=False, loc="center left", fontsize=8.5)
    foot = ("The graduate pool constrains the <30 channel only. The lateral channel "
            "(30+) is drawn from the contracted reserve\nand is modelled separately "
            "[GAP-1][GAP-4].")
    if not bool(deficit.any()):
        foot += ("  The graduate restriction never binds here, so no deficit band is "
                 "drawn [GAP-20].")
    if not split_ok:
        foot += ("  WARNING: no channel split was available, so this is the TOTAL "
                 "need and the comparison is a perimeter error [GAP-14].")
    fig.text(0.5, 0.005, foot, ha="center", fontsize=8,
             color=(RETIRE if not split_ok else "0.45"))
    fig.tight_layout(rect=[0, 0.04, 1, 1]); out = IMG / "pipeline_gap_central.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_channels(pg):
    """[GAP-4][GAP-29] Need and feasible supply by channel, plus per-channel gaps."""
    if pg is None or pg.empty:
        print("[fig] channels: no data"); return
    c = pg[pg.conversion_scn == "central"].sort_values("year")
    if c.empty:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.8))
    ax1.stackplot(c.year, c.need_young, c.need_lateral,
                  labels=["graduate channel (<30)", "lateral / re-entry (30+)"],
                  colors=[RECRUIT, LATERAL], alpha=0.85)
    ax1.plot(c.year, c.recruitment_need, color=INK, lw=1.6, label="total need")
    ax1.set_title("Recruitment NEED by channel (from 03)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("year"); ax1.set_ylabel("teachers per year")
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax2.stackplot(c.year, c.feasible_young, c.feasible_lateral,
                  labels=["feasible from the graduate pool",
                          "feasible from the contracted reserve"],
                  colors=[RECRUIT, LATERAL], alpha=0.55)
    ax2.plot(c.year, c.recruitment_need, color=INK, lw=1.6, ls="--",
             label="total need (for comparison)")
    ax2.set_title("FEASIBLE entries by channel (04's restriction)",
                  fontweight="bold", fontsize=11)
    ax2.set_xlabel("year")
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax3.plot(c.year, c.gap_young, "-o", color=RECRUIT, lw=2.2, ms=5,
             label="graduate channel")
    ax3.plot(c.year, c.gap_lateral, "-s", color=LATERAL, lw=2.2, ms=5,
             label="lateral channel")
    ax3.plot(c.year, c.gap_young + c.gap_lateral, ":", color=INK, lw=1.8,
             label="netted (what offsetting reports)")
    ax3.axhline(0, color="0.4", lw=1, ls="--")
    ax3.set_title("GAP by channel: >0 = shortage", fontweight="bold", fontsize=11)
    ax3.set_xlabel("year"); ax3.set_ylabel("teachers per year")
    ax3.grid(alpha=0.25); ax3.spines[["top", "right"]].set_visible(False)
    ax3.legend(frameon=False, fontsize=8.5)
    if (c.gap_young < 0).any() and (c.gap_lateral > 0).any():
        ax3.text(0.02, 0.03,
                 "the two channels have opposite signs:\n"
                 "netting credits one against the other [GAP-29]",
                 transform=ax3.transAxes, fontsize=8, color=RETIRE, va="bottom")

    fig.suptitle("Only the green band is governed by initial teacher education; "
                 "the right panel is what netting conceals",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); out = IMG / "channel_split.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_rate_modes(pg_modes, unanchored=None):
    """[GAP-18][GAP-28] Mode sweep, anchored and unanchored side by side."""
    if pg_modes is None or pg_modes.empty:
        print("[fig] rate modes: no data"); return
    has_un = (unanchored is not None and not unanchored.empty)
    ncol = 3 if has_un else 2
    fig, axes = plt.subplots(1, ncol, figsize=(6.6 * ncol, 6))
    ax1, ax2 = axes[0], axes[1]
    ax3 = axes[2] if ncol == 3 else None

    for mode in RATE_MODES:
        s = pg_modes[pg_modes.rate_mode == mode].sort_values("year")
        if s.empty:
            continue
        ax1.plot(s.year, 100 * s.conversion_rate, "-o", ms=4, lw=2.2,
                 color=MODE_COLORS[mode], label=MODE_LABELS[mode])
    ymax1 = max(100 * RATE_CEILING * 1.20,
                float(np.nanmax(100 * pg_modes["conversion_rate"])) * 1.20,
                100.0 / ROLL_GRAD_YEARS * 1.05)
    ax1.set_ylim(0, ymax1)
    ax1.axhline(100 * RATE_CEILING, color=RETIRE, lw=1.3, ls="--", alpha=0.85)
    ax1.text(pg_modes.year.min(), 100 * RATE_CEILING + ymax1 * 0.015,
             f"ceiling: {COHORT_CONV_CEILING:.0%} cohort conversion",
             fontsize=8, color=RETIRE)
    full = 100.0 / ROLL_GRAD_YEARS
    if full <= ymax1:
        ax1.axhline(full, color="0.55", lw=1.0, ls=":")
        ax1.text(pg_modes.year.max(), full + ymax1 * 0.015, "100% cohort conversion",
                 fontsize=8, color="0.55", ha="right")
    ax1.set_title("Conversion-rate trajectory (anchored)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("year"); ax1.set_ylabel("rate (%)")
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, fontsize=8.5)

    for mode in RATE_MODES:
        s = pg_modes[pg_modes.rate_mode == mode].sort_values("year")
        if s.empty:
            continue
        ax2.plot(s.year, s.gap_young, "-o", ms=4, lw=2.2,
                 color=MODE_COLORS[mode], label=MODE_LABELS[mode])
    ax2.axhline(0, color="0.4", lw=1, ls="--")
    allg = pd.to_numeric(pg_modes["gap_young"], errors="coerce").dropna()
    if not allg.empty:
        side = ("every mode is in SURPLUS in every year" if (allg < 0).all()
                else "every mode is in shortage in every year" if (allg > 0).all()
                else "the sign varies across modes or years")
        ax2.text(0.02, 0.03, side, transform=ax2.transAxes, fontsize=8.5,
                 color="0.35", va="bottom")
    ax2.set_title("GRADUATE-channel gap, ANCHORED", fontweight="bold", fontsize=11)
    ax2.set_xlabel("year"); ax2.set_ylabel("teachers   >0 = shortage / <0 = surplus")
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8.5)

    if ax3 is not None:
        for mode in RATE_MODES:
            s = unanchored[unanchored.rate_mode == mode].sort_values("year")
            if s.empty:
                continue
            ax3.plot(s.year, s.gap_young_unanchored, "-o", ms=4, lw=2.2,
                     color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        ax3.axhline(0, color="0.4", lw=1, ls="--")
        au = pd.to_numeric(unanchored["gap_young_unanchored"], errors="coerce").dropna()
        if not au.empty:
            side = ("every mode is in SURPLUS" if (au < 0).all()
                    else "every mode is in shortage" if (au > 0).all()
                    else "THE MODES DISAGREE on sign")
            ax3.text(0.02, 0.03, side, transform=ax3.transAxes, fontsize=8.5,
                     color=(RETIRE if "DISAGREE" in side else "0.35"), va="bottom")
        ax3.set_title("GRADUATE-channel gap, UNANCHORED  [GAP-28]",
                      fontweight="bold", fontsize=11)
        ax3.set_xlabel("year")
        ax3.grid(alpha=0.25); ax3.spines[["top", "right"]].set_visible(False)
        ax3.legend(frameon=False, fontsize=8.5)

    fig.suptitle("The three modes share one anchor, so their agreement is not "
                 "independent evidence [GAP-28]", fontweight="bold", y=1.02)
    fig.tight_layout(); out = IMG / "pipeline_rate_sensitivity.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_conversion_ratio(conv):
    if conv is None:
        print("[fig] conversion ratio: no data"); return
    m = conv["hist"].sort_values("year")
    paths = conv["paths"]
    if m.empty:
        print("[fig] conversion ratio: empty history"); return

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    chan = "graduate channel (<30)" if ENTRY_CHANNEL == "young" else "TOTAL entries"
    ax.plot(m.year, 100 * m.rate, "-o", color=INK, lw=2.4, ms=7,
            label=f"observed rate ({chan} / pool)")
    fy = sorted(paths["last"].keys())
    for mode in RATE_MODES:
        vals = [100 * paths[mode][y] for y in fy]
        style = "-" if mode == RATE_MODE else "--"
        lab = MODE_LABELS[mode] + (" [official]" if mode == RATE_MODE else "")
        ax.plot(fy, vals, style, color=MODE_COLORS[mode], lw=2.2, label=f"projected: {lab}")

    top = max(100 * max(max(p.values()) for p in paths.values()),
              100 * float(m.rate.max()), 100.0 / ROLL_GRAD_YEARS) * 1.20
    ax.set_ylim(0, top)

    for _, r in m.iterrows():
        ax.annotate(f"{100*r.rate:.0f}", (r.year, 100 * r.rate),
                    textcoords="offset points", xytext=(0, 9), fontsize=7.5, ha="center")

    win = m.tail(CONV_WINDOW)
    ax.axvspan(win.year.min() - 0.4, win.year.max() + 0.4, color="#f5d76e", alpha=0.25)
    ax.text((win.year.min() + win.year.max()) / 2, top * 0.045,
            f"last {CONV_WINDOW}y\nestimation window", ha="center", va="bottom",
            fontsize=8, color="#8a6d00")
    ax.axhline(100 * RATE_CEILING, color=RETIRE, lw=1.3, ls="--", alpha=0.85)
    ax.text(m.year.min(), 100 * RATE_CEILING + top * 0.012,
            f"ceiling: {COHORT_CONV_CEILING:.0%} cohort conversion",
            fontsize=8, color=RETIRE)
    full = 100.0 / ROLL_GRAD_YEARS
    if full <= top:
        ax.axhline(full, color="0.55", lw=1.0, ls=":")
        ax.text(m.year.max(), full + top * 0.012, "100% cohort conversion",
                fontsize=8, color="0.55", ha="right")

    ax.set_title(f"Conversion ratio: share of the {ROLL_GRAD_YEARS}-year graduate pool "
                 f"entering teaching each year", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("rate (% of the pool, per year)")
    axr = ax.twinx()
    axr.set_ylim(0, top * ROLL_GRAD_YEARS)
    axr.set_ylabel(f"cohort conversion (%)   [rate x {ROLL_GRAD_YEARS}]", color="0.35")
    axr.tick_params(axis="y", colors="0.35")
    ax.grid(alpha=0.25); ax.spines[["top"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    pct = conv.get("official_percentile", np.nan)
    foot = f"Numerator = {chan}. "
    if ENTRY_ANCHOR is not None:
        foot += (f"Entries anchored at {ENTRY_ANCHOR:,.0f}/yr: every rate on this "
                 f"chart is scaled by that anchor [GAP-24]. ")
    if np.isfinite(pct):
        foot += f"The official mode sits at the {pct:.0%} percentile of the history."
    ax.text(0.99, -0.13, foot, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="0.45")
    fig.tight_layout(); out = IMG / "conversion_ratio.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_constraint_division(cs):
    """[GAP-20][GAP-23][GAP-29] Need vs feasible, coverage by channel."""
    if cs is None or cs.empty:
        print("[fig] constraint division: no data"); return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(cs.year, cs.need_young, "-o", color=RETIRE, lw=2.4, ms=6,
             label="03_supply: graduate-channel need")
    ax1.plot(cs.year, cs.feasible_young, "-s", color=RECRUIT, lw=2.4, ms=6,
             label="04_gap: feasible graduate entries (pool x rate)")

    unmet = (cs.need_young >= cs.feasible_young)
    surplus = (cs.feasible_young > cs.need_young)
    if bool(unmet.any()):
        ax1.fill_between(cs.year, cs.feasible_young, cs.need_young, where=unmet,
                         color=RETIRE, alpha=0.12, label="unmet (the restriction binds)")
    if bool(surplus.any()):
        ax1.fill_between(cs.year, cs.need_young, cs.feasible_young, where=surplus,
                         color=RECRUIT, alpha=0.12, label="surplus graduate entries")

    ax1.set_ylim(0, max(float(cs.need_young.max()),
                        float(cs.feasible_young.max())) * 1.32)
    ax1.set_title("Division of labour: 03 asks how many are needed, 04 asks how many "
                  "the ITE pool can supply", fontweight="bold")
    ax1.set_ylabel("teachers per year")
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")

    parts = []
    cy = cs["coverage_young"].dropna()
    if not cy.empty:
        ax2.plot(cs.year, 100 * cs.coverage_young, "-o", color=RECRUIT, lw=2.4, ms=6,
                 label="graduate channel")
        parts.append(100 * cy)
    if "coverage_lateral" in cs.columns:
        cl = cs["coverage_lateral"].dropna()
        if not cl.empty:
            ax2.plot(cs.year, 100 * cs.coverage_lateral, "-^", color=LATERAL,
                     lw=2.0, ms=5, label="lateral channel")
            parts.append(100 * cl)
    if "coverage_total" in cs.columns:
        ct = cs["coverage_total"].dropna()
        if not ct.empty:
            ax2.plot(cs.year, 100 * cs.coverage_total, "-s", color=MID,
                     lw=1.6, ms=4, alpha=0.8, label="all channels (netted)")
            parts.append(100 * ct)
    parts = [p for p in parts if not p.empty]
    if parts:
        allv = pd.concat(parts)
        lo = min(float(allv.min()) - 8, 92)
        hi = max(float(allv.max()) + 8, 108)
        ax2.set_ylim(lo, hi)
        ax2.axhline(100, color=RETIRE, lw=1.4, ls="--")
        ax2.text(cs.year.min(), 100 + (hi - lo) * 0.03, "full coverage",
                 fontsize=8.5, color=RETIRE)
        ax2.legend(frameon=False, fontsize=8.5, ncol=3)
    ax2.set_title("Coverage by channel: the netted line can sit above 100% while a "
                  "channel is short [GAP-29]", fontweight="bold", fontsize=10.5)
    ax2.set_xlabel("year"); ax2.set_ylabel("coverage (%)")
    ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); out = IMG / "constraint_division.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


def fig_stock_gap(sg, source=""):
    """[GAP-19] Demand against BOTH baselines, the flat one named as such."""
    if sg is None or sg.empty:
        print("[fig] stock gap: no data"); return
    colours = {"central": MID, "high": RETIRE, "low": LIGHT}
    has_nr = ("gap_vs_no_replacement" in sg.columns
              and sg["gap_vs_no_replacement"].notna().any())

    if has_nr:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(11, 6)); ax2 = None

    for scn in sg.scenario.unique():
        s = sg[sg.scenario == scn].sort_values("year")
        ax1.plot(s.year, s.stock_gap, "-o", lw=2.2, ms=5,
                 color=colours.get(scn, INK), label=f"scenario {scn}")
    ax1.axhline(0, color="0.4", lw=1)
    base_lbl = sg["baseline"].iloc[0] if "baseline" in sg.columns else "baseline"
    ax1.set_title(f"Demand against a FIXED baseline\n({base_lbl})",
                  fontweight="bold", fontsize=10.5)
    ax1.set_xlabel("year"); ax1.set_ylabel("teachers   >0 = shortage / <0 = surplus")
    ax1.grid(alpha=0.25); ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False)
    ax1.text(0.02, 0.03,
             "the baseline is a constant, so this is the\n"
             "demand path shifted: no supply dynamics [GAP-19]",
             transform=ax1.transAxes, fontsize=8, color="0.45", va="bottom")

    if ax2 is not None:
        for scn in sg.scenario.unique():
            s = sg[sg.scenario == scn].sort_values("year")
            ax2.plot(s.year, s.gap_vs_no_replacement, "-o", lw=2.2, ms=5,
                     color=colours.get(scn, INK), label=f"scenario {scn}")
        ax2.axhline(0, color="0.4", lw=1)
        ax2.set_title("Demand against the NO-REPLACEMENT path\n"
                      "(stock declining by exits, starting before the base year)",
                      fontweight="bold", fontsize=10.5)
        ax2.set_xlabel("year"); ax2.set_ylabel("teachers   >0 = shortage")
        ax2.grid(alpha=0.25); ax2.spines[["top", "right"]].set_visible(False)
        ax2.legend(frameon=False)
        ax2.text(0.02, 0.03,
                 "carries supply information, but converges on\n"
                 "cumulative recruitment by construction [GAP-15][GAP-19]",
                 transform=ax2.transAxes, fontsize=8, color="0.45", va="bottom")

    extra = f"  [demand: {source}]" if source else ""
    fig.suptitle("Demographic pressure: two baselines, only one of them informative"
                 + extra, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = IMG / "demographic_pressure_stock_gap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("[ok]", out)


# ============================================================ MAIN
def main():
    print("=" * 78)
    print("04_gap -- applies the pipeline RESTRICTION to the unconstrained need from 03")
    print(f"         entry channel: {ENTRY_CHANNEL} | graduates: {GRAD_PROJECT_MODE} | "
          f"rate mode: {RATE_MODE}")
    print(f"         cohort ceiling: {COHORT_CONV_CEILING:.0%} (rate {RATE_CEILING:.1%}) "
          f"| lateral modelled: {MODEL_LATERAL_CHANNEL} | reconcile: {RECONCILE_NUTS3}")
    print("=" * 78)

    sup = load_supply_nuts3()
    nat = load_supply_national()
    dem = get_demand()
    channels, share_need = load_channel_shares()
    hist_stock, stock_meta = historical_stock(basis="bands")   # [GAP-16][GAP-32]
    if stock_meta:
        _MANIFEST["stock_basis"] = stock_meta
    if nat is None:
        print("[ERROR] no national supply -> cannot build the gap."); return

    uni = check_universe(nat, hist_stock=hist_stock, stock_meta=stock_meta)
    if uni is not None:
        pd.DataFrame([uni]).to_csv(RES / "universe_reconciliation.csv", index=False)
        print("[csv] universe_reconciliation.csv")
        _MANIFEST["universe"] = uni

    audit = None
    if sup is not None:
        sup, audit = reconcile_nuts3(sup, nat)
        cum = (sup.groupby("region", as_index=False)[["recruitment_need",
                                                      "recruitment_need_raw"]].sum()
                 .sort_values("recruitment_need", ascending=False))
        cum["region_name"] = cum.region.map(nice_region)
        tot_rec = float(cum["recruitment_need"].sum())
        cum["share_of_national"] = (cum["recruitment_need"] / tot_rec) if tot_rec > 0 else np.nan
        cum = cum[["region", "region_name", "recruitment_need", "recruitment_need_raw",
                   "share_of_national"]]
        cum.to_csv(RES / "gap_recruitment_nuts3.csv", index=False)
        print(f"[csv] gap_recruitment_nuts3.csv ({len(cum)} regions; raw and reconciled)")
        unknown = [r for r in cum["region"] if str(r) not in NUTS3_NAMES]
        if unknown:
            print(f"[csv]   WARNING: {len(unknown)} region code(s) not in NUTS3_NAMES "
                  f"and will print as raw codes: {unknown}")
        if audit is not None:
            audit.to_csv(RES / "nuts3_reconciliation_audit.csv", index=False)
            print("[csv] nuts3_reconciliation_audit.csv")
            _MANIFEST["reconciliation_cause"] = audit.attrs.get("cause", "unknown")

    uis = uis_decomposition(nat)
    nat_out = nat.copy()
    nat_out["recruitment_cumulative"] = nat_out.sort_values("year")["recruitment_need"].cumsum()
    if uis is not None:
        nat_out = nat_out.merge(
            uis[["year", "replacement", "expansion", "share_replacement"]],
            on="year", how="left")
    nat_out.to_csv(RES / "gap_national.csv", index=False)
    print(f"[csv] gap_national.csv (cumulative 2025-2040 = "
          f"{nat_out['recruitment_need'].sum():,.0f})")

    grad_h = load_graduates_history()
    entr_h, entr_src, is_stock, share_hist = load_entries_history()
    entr_h = apply_stock_to_flow(entr_h, is_stock)
    entr_h, anchor_factor = apply_entry_anchor(entr_h)
    grad_proj, grad_meta = project_graduates(grad_h)
    conv = estimate_conversion(entr_h, grad_proj)

    if np.isfinite(share_need) and np.isfinite(share_hist):
        print(f"\n[GAP-21] two young-channel shares appear in this run and they are")
        print(f"[GAP-21] NOT the same statistic:")
        print(f"[GAP-21]   {share_need:.1%} = young share of PROJECTED NEED, 2025-2040")
        print(f"[GAP-21]   {share_hist:.1%} = young share of OBSERVED ENTRIES, history")
        print(f"[GAP-21]   ratio {share_need/share_hist:.2f}x. The conversion ratio uses")
        print(f"[GAP-21]   the SECOND; the channel split of the need uses the FIRST.")
        _MANIFEST["young_share_projected_need"] = float(share_need)
        _MANIFEST["young_share_observed_entries"] = float(share_hist)

    if conv is not None and grad_meta is not None:
        pct = conv.get("official_percentile", np.nan)
        lvl_vs_last = grad_meta.get("level_vs_last", np.nan)
        if np.isfinite(pct) and np.isfinite(lvl_vs_last):
            _MANIFEST["convention_rate_percentile"] = float(pct)
            _MANIFEST["convention_grads_vs_last"] = float(lvl_vs_last)
            mism = bool(pct >= 0.95 and lvl_vs_last < -CONVENTION_TOL)
            _MANIFEST["convention_mismatch"] = mism
            if mism:
                print(f"\n[GAP-31] MIXED CONVENTIONS. The rate is read at the "
                      f"{pct:.0%} percentile of its")
                print(f"[GAP-31] history, effectively its maximum, while the graduate "
                      f"pool is set")
                print(f"[GAP-31] {abs(lvl_vs_last):.0%} BELOW its latest observation "
                      f"({grad_meta['level_used']:,.0f} against")
                print(f"[GAP-31] {grad_meta['last_observation']:,.0f}).")
                print(f"[GAP-31] NOTE ON DIRECTION: a rate at its maximum RAISES feasible")
                print(f"[GAP-31] entries while a pool below its latest LOWERS them, so the")
                print(f"[GAP-31] two conventions work AGAINST each other. Compute both")
                print(f"[GAP-31] before describing them as pushing the same way.")

    lateral = (estimate_lateral_rate(entr_h, hist_stock)
               if MODEL_LATERAL_CHANNEL else None)

    need_frame, split_ok = prepare_need_split(nat, channels=channels,
                                              hist_stock=hist_stock)
    _MANIFEST["channel_split_ok"] = bool(split_ok)
    _MANIFEST["entry_channel"] = ENTRY_CHANNEL
    if lateral is not None:
        _MANIFEST["lateral_rate"] = float(lateral["rate"])
        _MANIFEST["lateral_rate_basis"] = lateral["basis"]

    pg = pg_modes = cs = unanch = None
    band_label = None
    if conv is not None and grad_proj:
        ch = conv["hist"][["year", "entries", "grad_pool", "rate",
                           "cohort_conversion"]].copy()
        ch["entries_source"] = entr_src
        ch["entry_channel"] = ENTRY_CHANNEL
        ch["anchor_factor"] = anchor_factor
        ch.to_csv(RES / "conversion_history.csv", index=False)

        rp = pd.DataFrame({"year": list(range(HORIZON[0], HORIZON[1] + 1))})
        for mode in RATE_MODES:
            rp[f"rate_{mode}"] = rp["year"].map(conv["paths"][mode])
            rp[f"cohort_conv_{mode}"] = rp[f"rate_{mode}"] * ROLL_GRAD_YEARS
        rp["rate_mode_official"] = RATE_MODE
        rp["entry_channel"] = ENTRY_CHANNEL
        rp["rate_ceiling"] = RATE_CEILING
        rp["cohort_conv_ceiling"] = COHORT_CONV_CEILING
        rp["estimation_window"] = CONV_WINDOW
        rp["anchor_factor"] = anchor_factor
        rp.to_csv(RES / "conversion_rate_paths.csv", index=False)

        gp = pd.DataFrame({"year": list(grad_proj.keys()),
                           "graduates": list(grad_proj.values())}).sort_values("year")
        gp["graduate_pool"] = gp["year"].map(lambda y: graduate_pool(grad_proj, int(y)))
        gp["projection_mode"] = GRAD_PROJECT_MODE
        gp.to_csv(RES / "graduates_projection.csv", index=False)

        if lateral is not None:
            lateral["hist"].to_csv(RES / "lateral_channel_history.csv", index=False)
            print("[csv] lateral_channel_history.csv")

        bf = conv["band_factors"]
        band_label = (f"rate band: IQR ratio {bf['low']:.2f}x-{bf['high']:.2f}x "
                      f"around the '{RATE_MODE}' path [GAP-26]")

        built = build_pipeline(need_frame, grad_proj, conv, lateral=lateral)
        if built is not None:
            pg, pg_modes = built
            pg["channel_split_ok"] = bool(split_ok)
            pg.to_csv(RES / "pipeline_gap.csv", index=False)

            unanch = rate_mode_sweep_unanchored(need_frame, grad_proj, conv,
                                                anchor_factor)
            if unanch is not None:
                pg_modes = pg_modes.merge(
                    unanch[["year", "rate_mode", "conversion_rate_unanchored",
                            "feasible_young_unanchored", "gap_young_unanchored"]],
                    on=["year", "rate_mode"], how="left")
            pg_modes.to_csv(RES / "pipeline_gap_by_rate_mode.csv", index=False)

            c = pg[pg.conversion_scn == "central"]
            print(f"\n[csv] pipeline_gap.csv (entries={entr_src}, mode='{RATE_MODE}')")
            print(f"      mean GRADUATE-channel gap = {c['gap_young'].mean():+,.0f}/yr")
            print(f"      mean LATERAL-channel  gap = {c['gap_lateral'].mean():+,.0f}/yr")
            print(f"      mean netted gap           = {c['pipeline_gap'].mean():+,.0f}/yr")
            if not split_ok:
                print("      >>> NO CHANNEL SPLIT: these figures compare the graduate pool")
                print("      >>> against the TOTAL need and are NOT reportable  [GAP-14]")

            print("\n[rate modes] GRADUATE-channel gap, ANCHORED and UNANCHORED  [GAP-28]:")
            v_anch, v_unanch = [], []
            for mode in RATE_MODES:
                s = pg_modes[pg_modes.rate_mode == mode]
                if s.empty:
                    continue
                va, n_ok, n_miss = sign_verdict(s["gap_young"])
                v_anch.append(va)
                line = (f"    {mode:<14} anch {s['conversion_rate'].iloc[0]:5.1%} "
                        f"gap {s['gap_young'].mean():+,.0f}/yr [{va}]")
                if ("gap_young_unanchored" in s.columns
                        and s["gap_young_unanchored"].notna().any()):
                    vu, _, _ = sign_verdict(s["gap_young_unanchored"])
                    v_unanch.append(vu)
                    line += (f" | unanch {s['conversion_rate_unanchored'].iloc[0]:5.1%} "
                             f"gap {s['gap_young_unanchored'].mean():+,.0f}/yr [{vu}]")
                n_cap = int((s["conversion_rate"] >= RATE_CEILING - 1e-12).sum())
                if n_cap:
                    line += f"  (capped {n_cap}y)"
                print(line)

            ua = set(v for v in v_anch if v != "none")
            uu = set(v for v in v_unanch if v != "none")
            _MANIFEST["rate_mode_verdicts_anchored"] = dict(zip(RATE_MODES, v_anch))
            if v_unanch:
                _MANIFEST["rate_mode_verdicts_unanchored"] = dict(zip(RATE_MODES, v_unanch))

            if not ua:
                print("    -> no mode could be evaluated.")
            elif not uu:
                print(f"    -> anchored, the modes give {sorted(ua)}. No unanchored")
                print(f"       comparison was available, so robustness is UNTESTED.")
            elif len(ua) == 1 and ua == uu:
                print(f"    -> all modes agree on '{list(ua)[0]}' BOTH anchored and")
                print(f"       unanchored. That is genuine robustness to the rate")
                print(f"       assumption.")
            elif len(ua) == 1 and ua != uu:
                print(f"    -> anchored the modes agree on '{list(ua)[0]}', but unanchored")
                print(f"       they give {sorted(uu)}. The three modes are median, last and")
                print(f"       damped trend of the SAME anchored series, so they all carry")
                print(f"       the identical {anchor_factor:.3f} factor. Their agreement is")
                print(f"       NOT independent evidence about the rate: it is one estimate")
                print(f"       seen three ways. Do not report 'robust under any rate")
                print(f"       assumption'  [GAP-28]")
            else:
                print(f"    -> the modes DISAGREE even anchored: {sorted(ua)}. Report a "
                      f"range.")

            cs = build_constraint_summary(pg)
            if cs is not None:
                cs.to_csv(RES / "constraint_summary.csv", index=False)
                print("[csv] constraint_summary.csv")
                _MANIFEST["unmet_no_offset"] = float(cs["unmet_no_offset"].sum())
                _MANIFEST["unmet_netted"] = float(cs["unmet_netted"].sum())

        rob = check_anchor_robustness(conv, grad_proj, need_frame, anchor_factor)
        if rob is not None:
            rob.to_csv(RES / "anchor_robustness.csv", index=False)
            print("[csv] anchor_robustness.csv")
            _MANIFEST["anchor"] = rob.iloc[0].to_dict()
    else:
        print("[warn] pipeline skipped: missing graduates or entries.")

    sg = build_stock_gap(dem, nat, hist_stock=hist_stock)
    src = dem["source"].iloc[0] if (dem is not None and "source" in dem.columns) else ""
    if sg is not None:
        sg.to_csv(RES / "gap_stock.csv", index=False)
        print(f"[csv] gap_stock.csv (demand: {src})")

    subs = substitution_reserve_note(nat)
    if subs is not None:
        pd.DataFrame([subs]).to_csv(RES / "substitution_reserve.csv", index=False)
        print("[csv] substitution_reserve.csv")
        _MANIFEST["substitution_reserve"] = subs

    print()
    fig_wave_vs_recruitment(nat)
    fig_cum_recruit_nuts3(sup)
    fig_reconciliation(sup, audit)
    fig_pipeline(pg, split_ok=split_ok, band_label=band_label)
    fig_channels(pg)
    fig_rate_modes(pg_modes, unanchored=unanch)
    fig_conversion_ratio(conv)
    fig_constraint_division(cs)
    fig_stock_gap(sg, source=src)

    with open(RES / "gap_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_MANIFEST, f, indent=2, default=str)
    print("[manifest] gap_run_manifest.json")

    # ---------------- caveats that must travel with the numbers
    print("\n" + "=" * 78)
    print("CAVEATS THAT MUST BE REPORTED WITH THESE NUMBERS")
    print("=" * 78)
    print("  1. [GAP-1] The conversion ratio uses the GRADUATE channel only, so its")
    print("     numerator and denominator describe the same population. The lateral")
    print("     channel is the LARGER of the two and is not governed by ITE.")
    if ENTRY_CHANNEL != "young":
        print("     >>> ENTRY_CHANNEL is NOT 'young'. Results are not reportable.")
    if not split_ok:
        print("     >>> NO CHANNEL SPLIT. The graduate constraint was applied to the")
        print("     >>> TOTAL need. Not reportable until "
              "supply_recruitment_by_channel.csv exists [GAP-14]")
    if uni is not None and uni.get("perimeter_flag"):
        print(f"  2. [GAP-7] At {uni['comparison_year']} the chain holds "
              f"{uni['chain_stock_at_comparison']:,.0f} against an external")
        print(f"     benchmark of {uni['external_benchmark']:,.0f}, a difference of "
              f"{uni['relative_difference']:+.1%}.")
        if not uni.get("like_for_like_in_time"):
            print("     WARNING: that comparison is not like-for-like in time [GAP-13].")
        print(f"     Stock basis = {uni.get('stock_basis')} on "
              f"{uni.get('n_band_columns')} age-band columns [GAP-32].")
        print("     Until the perimeter is explained, every level carries that error.")
    else:
        print("  2. [GAP-7] Universe checked against the external benchmark.")
    print("  3. [GAP-8] The need modelled here is PERMANENT POSTS. The substitution")
    print("     reserve is measured nowhere in this chain and is of comparable or")
    print("     larger size. An implied figure IS published in substitution_reserve.csv;")
    print("     cite it rather than stating that no estimate exists.")
    print("  4. [GAP-9] 02's student/teacher ratio is DESCRIPTIVE. If teachers were")
    print("     short in the observed period it is biased upwards, so projected demand")
    print("     is biased DOWNWARDS and the gap is a lower bound on that account.")
    if ENTRY_ANCHOR is not None:
        print(f"  5. [GAP-24][GAP-28] The entry LEVEL is anchored externally "
              f"({ENTRY_ANCHOR:,.0f}/yr).")
        print(f"     The pool is unchanged, so the rate is multiplied by exactly that")
        print(f"     factor. The three rate modes all carry it, so their agreement is")
        print(f"     NOT independent evidence. See pipeline_gap_by_rate_mode.csv for the")
        print(f"     unanchored sweep and anchor_robustness.csv for the verdict.")
    if RECONCILE_NUTS3 and audit is not None:
        print(f"  6. The regional series is raked onto 04's national total; the residual")
        print(f"     originates in {audit.attrs.get('cause', 'an undetermined source')}.")
    print(f"  7. The conversion rate is capped at {COHORT_CONV_CEILING:.0%} cohort "
          f"conversion (rate {RATE_CEILING:.1%}).")
    if lateral is not None:
        print(f"  8. [GAP-10] The lateral channel is projected at "
              f"{lateral['rate']:.3%} of the PREVIOUS")
        print(f"     year's age-band stock. It is a reserve-depletion assumption, not an")
        print(f"     estimate of reserve capacity, and it is NOT anchored [GAP-27].")
    if sg is not None and "stock_gap" in sg.columns:
        print("  9. [GAP-19] 'stock_gap' subtracts a CONSTANT from demand, so it carries")
        print("     no supply dynamics. Use 'gap_vs_no_replacement'.")
    if cs is not None:
        no = float(cs["unmet_no_offset"].sum()); net = float(cs["unmet_netted"].sum())
        if no > net + 1.0:
            print(f" 10. [GAP-29] Unmet need is {no:,.0f} summed per channel against "
                  f"{net:,.0f} netted.")
            print(f"     The difference is a surplus in one channel credited against a")
            print(f"     shortfall in the other. Report the per-channel figure.")
    if _MANIFEST.get("convention_mismatch"):
        print(" 11. [GAP-31] The rate is read at its historical maximum while the")
        print("     graduate pool is read below its latest observation. The two")
        print("     conventions work AGAINST each other; compute both before")
        print("     characterising their net effect.")
    if _MANIFEST.get("smoothing_residual_rel", 0) > SMOOTH_RESIDUAL_TOL:
        print(" 12. [GAP-35] The centred moving average does not preserve the cumulative")
        print("     at the edges of the horizon. Quote the unsmoothed total.")
    print(f" 13. [GAP-36] Figure names are recorded in the manifest. The wave figure is")
    print(f"     '{WAVE_FIGURE_NAME}' and the NUTS III figure is")
    print(f"     'cumulative_recruitment_nuts3.png'. Reconcile the .qmd labels against")
    print(f"     these before compiling; two blocks have used #fig-nuts3 for different")
    print(f"     files.")
    print("=" * 78)
    print("Done. Results in:", RES, "| figures in:", IMG)
    print("=" * 78)


if __name__ == "__main__":
    main()
