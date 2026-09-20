
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
DATA_DIR     = BASE / r"models\00_data"
STUDENTS_DIR = BASE / r"models\01_students\results"
DEMAND_DIR   = BASE / r"models\02_demand\results"
SUPPLY_DIR   = BASE / r"models\03_supply\results"
GAP_DIR      = BASE / r"models\04_gap\results"
UNC_DIR      = BASE / r"models\05_uncertainty\results"
RES = BASE / r"models\06_validation\results"
IMG = BASE / r"models\06_validation\images"

MASTER_FILE = DATA_DIR / "master_panel_nuts3_with_age.xlsx"

if os.environ.get("VAL_TEST") == "1":
    DATA_DIR = Path("."); STUDENTS_DIR = Path("."); DEMAND_DIR = Path(".")
    SUPPLY_DIR = Path("."); GAP_DIR = Path("."); UNC_DIR = Path(".")
    RES = Path("val_out"); IMG = Path("val_out")
    MASTER_FILE = Path("master_panel_nuts3_with_age.xlsx")
RES.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

# ============================================================ CONFIG
HORIZON = (2025, 2040)

TOL_IDENTITY   = 0.01
TOL_CROSSBLOCK = 0.02
TOL_SPATIAL    = 0.05
TOL_RECONCILED = 0.001
TOL_BENCHMARK  = 0.30

BOUND_HAZARD_60   = (0.02, 0.60)
BOUND_RATIO       = (5.0, 30.0)
BOUND_CONVERSION  = (0.01, 1.00)
BOUND_RECRUIT     = (1000, 15000)
BOUND_COHORT_CONV = (0.05, 1.00)

# --- family G: silent-failure guards ---
# [VAL-5] absolute AND relative. One teacher of band width on a ~5,700 median is a
# collapse dressed as a tight estimate, and the old absolute-only rule waved it through.
MIN_FAN_BAND = 1.0            # teachers, worst year
MIN_FAN_BAND_REL = 0.02       # and at least 2% of the median need

# [VAL-3] hindcast: how many age bands may lose to the naive benchmark
# [VAL-8] denominator guard for the per-band MASE: a naive error below this fraction of
# the band's own level is numerically zero, and the ratio is undefined rather than huge.
MASE_DENOM_REL = 1e-6
MASE_DENOM_ABS = 1e-6

HINDCAST_MAX_BANDS_LOST_WARN = 1
HINDCAST_MAX_BANDS_LOST_FAIL = 4

# --- external benchmark, current vintage ---
BENCH_LABEL          = "DGEEC/Nova SBE 2025-2034 (Nunes et al., 2025)"
BENCH_RECRUIT_TOTAL  = 38779
BENCH_YEARS          = 10
BENCH_WINDOW         = (2025, 2034)
BENCH_RETIRE_ANNUAL  = 4000.0

HINDCAST_HOLDOUT = 3
HINDCAST_MIN_TRAIN = 5

BANDS = ["lt25", "25_29", "30_34", "35_39", "40_44", "45_49", "50_54", "55_59", "60_plus"]
NICE  = ["<25", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"]
NB = len(BANDS)
G = 1.0 / 5.0
HCLIP = (0.0, 0.6)
FIRST_REAL_YEAR = 2014
HINDCAST_BAND0_HAZARD = 0.02   # was a bare magic number inside _estimate_hazards

OK = "#128a4b"; WARNC = "#e08a2c"; BAD = "#b5171e"; INK = "#1a1a1a"; MID = "#3b4a5a"
SKIPC = "#c8ccd0"; TAUTC = "#9aa4ad"

# [VAL-2] Checks that cannot fail because the producing block computes them by
# definition. Keyed by check name, valued by the reason. They still run: a corrupted file
# or a wrong column would trip them. They are simply not evidence.
TAUTOLOGICAL = {
    "stock[t]-stock[t-1] == entries-exits":
        "03 computes n = pre + E*profile and exits in one loop, and asserts this itself",
    "need == replacement + expansion":
        "04 defines expansion = need - replacement; this is the definition rearranged",
    "05 inherits the 04 rate ceiling":
        "05 reads the value out of 04's own CSV and writes it back",
    "03 stock tracks 02 demand (endogenous entry)":
        "endogenous entry sets E = max(demand - after, 0), so stock == demand whenever E>0",
    # [VAL-9] This label was WRONG in the previous revision and the run proved it: the
    # check was flagged "cannot fail" and then failed at 0.9562. It holds by construction
    # ONLY once the channel is matched; picking the wrong entry column is exactly the
    # failure mode it caught. Retained as by-construction with an honest reason.
    "04 entries == 03 exported flow (shape)":
        "holds once the CHANNEL is matched and the anchor removed; a wrong column or a "
        "channel mismatch still trips it, as it did before [VAL-9]",
    "sum(NUTS III) RECONCILED == national (closure)":
        "the raking factor is national/regional, so closure is arithmetic",
    "regional shares sum to 1":
        "shares are computed as x / sum(x)",
}


# ============================================================ RESULT COLLECTOR
class Report:
    def __init__(self):
        self.rows = []

    def add(self, family, name, status, value=np.nan, tol=np.nan, detail=""):
        taut = name in TAUTOLOGICAL
        self.rows.append({"family": family, "check": name, "status": status,
                          "value": value, "tolerance": tol, "detail": detail,
                          "by_construction": bool(taut),
                          "construction_reason": TAUTOLOGICAL.get(name, "")})
        val = "" if not np.isfinite(value) else f" [{value:,.4g}]"
        mark = "  (by construction)" if taut else ""
        print(f"    [{status}] {name}{val}" + (f" -- {detail}" if detail else "") + mark)

    def judge(self, family, name, rel_err, tol, detail="", warn_factor=2.0):
        if not np.isfinite(rel_err):
            return self.add(family, name, "SKIP", np.nan, tol, detail or "not computable")
        if abs(rel_err) <= tol:
            return self.add(family, name, "PASS", rel_err, tol, detail)
        if abs(rel_err) <= tol * warn_factor:
            return self.add(family, name, "WARN", rel_err, tol, detail)
        return self.add(family, name, "FAIL", rel_err, tol, detail)

    def frame(self):
        return pd.DataFrame(self.rows)


# ============================================================ UTILS
def find_col(cols, *keys):
    low = {c: str(c).lower() for c in cols}
    for k in keys:
        for c in cols:
            if k in low[c]:
                return c
    return None


def find_exact(cols, *names):
    low = {str(c).lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def read_csv_safe(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def detect(d, must_have, prefer=None):
    for name in (prefer or []):
        p = Path(d) / name
        if p.exists():
            return p
    for g in glob.glob(str(Path(d) / "*.csv")):
        h = read_csv_safe(g)
        if h is None:
            continue
        cols = [str(c).lower() for c in h.columns]
        if all(any(m in c for c in cols) for m in must_have):
            return Path(g)
    return None


def rel(a, b):
    try:
        a = float(a); b = float(b)
    except Exception:
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-9:
        return np.nan
    return a / b - 1.0


def year_filter(df, col, lo, hi):
    y = pd.to_numeric(df[col], errors="coerce")
    return df[(y >= lo) & (y <= hi)]


def as_bool(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def first_num(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(v.iloc[0]) if len(v) else np.nan


# ============================================================ A. ACCOUNTING
def check_accounting(rep):
    print("\n[A] Accounting identities")
    d = read_csv_safe(Path(SUPPLY_DIR) / "supply_projection.csv")
    if d is None:
        rep.add("A_accounting", "stock flow balance", "SKIP",
                detail="supply_projection.csv missing")
    else:
        yc = find_col(d.columns, "year"); sc = find_col(d.columns, "stock")
        rc = find_col(d.columns, "recruit"); ec = find_col(d.columns, "exit", "saida")
        if all([yc, sc, rc, ec]):
            x = d[[yc, sc, rc, ec]].copy()
            x.columns = ["year", "stock", "recruit", "exits"]
            for c in x.columns:
                x[c] = pd.to_numeric(x[c], errors="coerce")
            x = x.dropna().sort_values("year").reset_index(drop=True)
            if len(x) >= 2:
                dstock = x["stock"].diff().iloc[1:].values
                netflow = (x["recruit"] - x["exits"]).iloc[1:].values
                scale = np.maximum(np.abs(x["stock"].iloc[1:].values), 1.0)
                err = np.abs(dstock - netflow) / scale
                worst = float(np.nanmax(err))
                wy = int(x["year"].iloc[1:].values[int(np.nanargmax(err))])
                rep.judge("A_accounting", "stock[t]-stock[t-1] == entries-exits",
                          worst, TOL_IDENTITY, f"worst year {wy}")
            else:
                rep.add("A_accounting", "stock flow balance", "SKIP",
                        detail="fewer than two usable years")
        else:
            rep.add("A_accounting", "stock flow balance", "SKIP", detail="columns missing")

    g = read_csv_safe(Path(GAP_DIR) / "gap_national.csv")
    if g is None:
        rep.add("A_accounting", "need == replacement + expansion", "SKIP",
                detail="gap_national.csv missing")
        return
    nc = find_exact(g.columns, "recruitment_need") or find_col(g.columns, "recruit")
    rc = find_col(g.columns, "replacement", "reposicao")
    ec = find_col(g.columns, "expansion", "expansao")
    if all([nc, rc, ec]):
        need = pd.to_numeric(g[nc], errors="coerce")
        comp = pd.to_numeric(g[rc], errors="coerce") + pd.to_numeric(g[ec], errors="coerce")
        err = float(np.nanmax(np.abs(need - comp) / np.maximum(np.abs(need), 1.0)))
        rep.judge("A_accounting", "need == replacement + expansion", err, TOL_IDENTITY,
                  "UIS decomposition")
    else:
        rep.add("A_accounting", "need == replacement + expansion", "SKIP",
                detail="columns missing")


# ============================================================ B. CROSS-BLOCK
def check_crossblock(rep):
    print("\n[B] Cross-block consistency")

    g = read_csv_safe(Path(GAP_DIR) / "gap_national.csv")
    u = read_csv_safe(Path(UNC_DIR) / "cumulative_recruitment.csv")
    if g is not None and u is not None:
        nc = find_exact(g.columns, "recruitment_need") or find_col(g.columns, "recruit")
        central = float(pd.to_numeric(g[nc], errors="coerce").sum())
        p50c = find_col(u.columns, "p50")
        if p50c:
            med = first_num(u, p50c)
            rep.judge("B_crossblock", "05 median == 04 central (cumulative)",
                      rel(med, central), TOL_CROSSBLOCK,
                      f"median {med:,.0f} vs central {central:,.0f}")
        else:
            rep.add("B_crossblock", "05 median == 04 central", "SKIP",
                    detail="no p50 column")
    else:
        rep.add("B_crossblock", "05 median == 04 central", "SKIP", detail="inputs missing")

    rp = read_csv_safe(Path(GAP_DIR) / "conversion_rate_paths.csv")
    cd = read_csv_safe(Path(UNC_DIR) / "ceiling_and_diagnostics.csv")
    if rp is not None and cd is not None:
        c04 = find_exact(rp.columns, "rate_ceiling")
        c05 = find_exact(cd.columns, "rate_ceiling")
        if c04 and c05:
            v04 = first_num(rp, c04); v05 = first_num(cd, c05)
            rep.judge("B_crossblock", "05 inherits the 04 rate ceiling",
                      rel(v05, v04), 0.001, f"04 = {v04:.4f} | 05 = {v05:.4f}")
        else:
            rep.add("B_crossblock", "05 inherits the 04 rate ceiling", "SKIP",
                    detail="ceiling column absent in one of the files")
    else:
        rep.add("B_crossblock", "05 inherits the 04 rate ceiling", "SKIP",
                detail="conversion_rate_paths or ceiling_and_diagnostics missing")

    dm = detect(DEMAND_DIR, ["year"], prefer=["teacher_demand_selected.csv"])
    handled = False
    if dm is not None:
        d = read_csv_safe(dm)
        yc = find_col(d.columns, "year") if d is not None else None
        vc = find_col(d.columns, "demand_selected", "demand") if d is not None else None
        sc = find_col(d.columns, "scenario") if d is not None else None
        if d is not None and yc and vc:
            x = d.copy()
            if sc:
                x = x[x[sc].astype(str).str.lower() == "central"]
            x[yc] = pd.to_numeric(x[yc], errors="coerce")
            x[vc] = pd.to_numeric(x[vc], errors="coerce")
            nat02 = x.dropna(subset=[yc]).groupby(yc)[vc].sum()
            sp = read_csv_safe(Path(SUPPLY_DIR) / "supply_projection.csv")
            if sp is not None:
                yc2 = find_col(sp.columns, "year"); st = find_col(sp.columns, "stock")
                if yc2 and st:
                    s = sp[[yc2, st]].copy(); s.columns = ["year", "stock"]
                    s["year"] = pd.to_numeric(s["year"], errors="coerce")
                    s["stock"] = pd.to_numeric(s["stock"], errors="coerce")
                    common = sorted(set(nat02.index).intersection(set(s["year"].dropna())))
                    if common:
                        a = np.array([nat02.loc[y] for y in common], float)
                        b = np.array([float(s[s.year == y]["stock"].iloc[0]) for y in common])
                        err = float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))
                        rep.judge("B_crossblock",
                                  "03 stock tracks 02 demand (endogenous entry)",
                                  err, TOL_CROSSBLOCK,
                                  f"{len(common)} overlapping years")
                        handled = True
    if not handled:
        rep.add("B_crossblock", "03 stock tracks 02 demand", "SKIP",
                detail="demand or supply series unavailable")

    eh = read_csv_safe(Path(SUPPLY_DIR) / "supply_entries_history.csv")
    chist = read_csv_safe(Path(GAP_DIR) / "conversion_history.csv")
    if eh is not None and chist is not None:
        # [VAL-9] Match the CHANNEL, not just the scale. 04 feeds its conversion ratio
        # with the GRADUATE channel (new_entrants_young); find_col(.., "new_entrants")
        # returns the TOTAL column, because "new_entrants" is a prefix of every entry
        # column and the total one comes first. Dividing by the anchor factor then
        # compares young against total and the check fails by 0.80-0.98, which is simply
        # one minus the young share. 04 records which channel it used; read it.
        chan_col = find_exact(chist.columns, "entry_channel")
        channel = (str(chist[chan_col].iloc[0]).strip().lower()
                   if chan_col is not None else "")
        if channel == "young":
            ec = (find_exact(eh.columns, "new_entrants_young")
                  or find_col(eh.columns, "new_entrants", "entrant"))
        else:
            ec = find_exact(eh.columns, "new_entrants") or \
                 find_col(eh.columns, "new_entrants", "entrant")
        yc = find_col(eh.columns, "year")
        ce = find_col(chist.columns, "entries")
        cy = find_col(chist.columns, "year")
        af = find_exact(chist.columns, "anchor_factor")
        if all([ec, yc, ce, cy]):
            a = eh[[yc, ec]].copy(); a.columns = ["year", "e03"]
            b = chist[[cy, ce]].copy(); b.columns = ["year", "e04"]
            for df_ in (a, b):
                for c in df_.columns:
                    df_[c] = pd.to_numeric(df_[c], errors="coerce")
            m = a.merge(b, on="year", how="inner").dropna()
            if len(m) >= 3:
                k = 1.0
                if af is not None:
                    v = pd.to_numeric(chist[af], errors="coerce").dropna()
                    if len(v) and np.isfinite(v.iloc[0]) and v.iloc[0] > 0:
                        k = float(v.iloc[0])
                err = float(np.nanmax(np.abs(m.e04 / k - m.e03) /
                                      np.maximum(np.abs(m.e03), 1.0)))
                note = f"{len(m)} overlapping years; channel '{channel or 'unknown'}' "
                note += f"matched to 03 column '{ec}'"
                if abs(k - 1.0) > 1e-9:
                    note += f"; 04 anchor factor {k:.3f} removed before comparing"
                rep.judge("B_crossblock", "04 entries == 03 exported flow (shape)",
                          err, TOL_CROSSBLOCK, note)
            else:
                rep.add("B_crossblock", "04 entries == 03 exported flow", "SKIP",
                        detail="insufficient overlap")
        else:
            rep.add("B_crossblock", "04 entries == 03 exported flow", "SKIP",
                    detail="columns missing")
    else:
        rep.add("B_crossblock", "04 entries == 03 exported flow", "SKIP",
                detail="03 flow file or 04 conversion history missing")


# ============================================================ C. SPATIAL
def check_spatial(rep):
    print("\n[C] Spatial reconciliation (NUTS III vs national)")
    reg = read_csv_safe(Path(GAP_DIR) / "gap_recruitment_nuts3.csv")
    nat = read_csv_safe(Path(GAP_DIR) / "gap_national.csv")
    if reg is None or nat is None:
        rep.add("C_spatial", "sum(NUTS III) == national", "SKIP", detail="inputs missing")
        return

    nc = find_exact(nat.columns, "recruitment_need") or find_col(nat.columns, "recruit")
    if nc is None:
        rep.add("C_spatial", "sum(NUTS III) == national", "SKIP",
                detail="national column missing")
        return
    s_nat = float(pd.to_numeric(nat[nc], errors="coerce").sum())

    col_rec = find_exact(reg.columns, "recruitment_need")
    col_raw = find_exact(reg.columns, "recruitment_need_raw")

    if col_raw is not None:
        s_raw = float(pd.to_numeric(reg[col_raw], errors="coerce").sum())
        r_raw = rel(s_raw, s_nat)
        rep.judge("C_spatial", "sum(NUTS III) RAW == national", r_raw, TOL_SPATIAL,
                  f"raw {s_raw:,.0f} vs national {s_nat:,.0f}")
        if np.isfinite(r_raw) and abs(r_raw) > TOL_SPATIAL:
            print("           note: a positive gap is the expected signature of")
            print("           sum(max(0,.)) over regions when the national expansion is")
            print("           negative. 04_gap rakes the series for this reason.")
    else:
        rep.add("C_spatial", "sum(NUTS III) RAW == national", "SKIP",
                detail="no recruitment_need_raw column -- run the current 04_gap")

    if col_rec is not None:
        s_rec = float(pd.to_numeric(reg[col_rec], errors="coerce").sum())
        rep.judge("C_spatial", "sum(NUTS III) RECONCILED == national (closure)",
                  rel(s_rec, s_nat), TOL_RECONCILED,
                  f"reconciled {s_rec:,.0f} vs national {s_nat:,.0f}")
    else:
        rep.add("C_spatial", "sum(NUTS III) RECONCILED == national", "SKIP",
                detail="no recruitment_need column")

    sh = find_exact(reg.columns, "share_of_national")
    if sh is not None:
        tot = float(pd.to_numeric(reg[sh], errors="coerce").sum())
        rep.judge("C_spatial", "regional shares sum to 1", rel(tot, 1.0), 0.001,
                  f"sum = {tot:.6f}")

    regc = find_col(reg.columns, "region", "nuts3")
    if regc:
        n = int(reg[regc].nunique())
        rep.add("C_spatial", "24 mainland NUTS III regions",
                "PASS" if n == 24 else "WARN", float(n), 24.0, f"{n} regions found")


# ============================================================ D. BENCHMARK
def _perimeter_ratio():
    """[VAL-4] Chain stock over external benchmark stock, from 04's own reconciliation."""
    u = read_csv_safe(Path(GAP_DIR) / "universe_reconciliation.csv")
    if u is None:
        return np.nan, ""
    c_chain = find_exact(u.columns, "chain_stock_at_comparison")
    c_ext = find_exact(u.columns, "external_benchmark")
    c_rel = find_exact(u.columns, "relative_difference")
    if c_chain and c_ext:
        a, b = first_num(u, c_chain), first_num(u, c_ext)
        if np.isfinite(a) and np.isfinite(b) and b > 0:
            yr = find_exact(u.columns, "comparison_year")
            ytxt = f" at {int(first_num(u, yr))}" if yr else ""
            return a / b, f"chain {a:,.0f} vs benchmark {b:,.0f}{ytxt}"
    if c_rel:
        r = first_num(u, c_rel)
        if np.isfinite(r):
            return 1.0 + r, f"relative difference {r:+.1%}"
    return np.nan, ""


def check_benchmark(rep):
    lo, hi = BENCH_WINDOW
    print(f"\n[D] External benchmark ({BENCH_LABEL})")
    print(f"    comparison window {lo}-{hi}; our horizon is {HORIZON[0]}-{HORIZON[1]}, "
          f"so only the overlap is used")
    nat = read_csv_safe(Path(GAP_DIR) / "gap_national.csv")
    if nat is None:
        rep.add("D_benchmark", "recruitment vs benchmark", "SKIP",
                detail="gap_national.csv missing")
        return
    yc = find_col(nat.columns, "year")
    nc = find_exact(nat.columns, "recruitment_need") or find_col(nat.columns, "recruit")
    rw = find_col(nat.columns, "retirements_wave", "retire")
    if yc is None:
        rep.add("D_benchmark", "recruitment vs benchmark", "SKIP", detail="no year column")
        return

    win = year_filter(nat, yc, lo, hi)
    n_years = len(win)
    if n_years == 0:
        rep.add("D_benchmark", "recruitment vs benchmark", "SKIP",
                detail="no overlap with the benchmark window")
        return

    ratio, ratio_note = _perimeter_ratio()
    if np.isfinite(ratio):
        print(f"    [VAL-4] perimeter ratio from 04: {ratio:.3f} ({ratio_note})")
        print(f"    the benchmark covers 835 mainland PUBLIC units and excludes private")
        print(f"    education, so the raw comparison mixes a universe difference into it")
    else:
        print("    [VAL-4] no universe_reconciliation.csv -> raw comparison only")

    if nc:
        ours = float(pd.to_numeric(win[nc], errors="coerce").mean())
        bench = BENCH_RECRUIT_TOTAL / BENCH_YEARS
        rep.judge("D_benchmark", "recruitment per year vs benchmark (raw)",
                  rel(ours, bench), TOL_BENCHMARK,
                  f"ours {ours:,.0f}/yr over {n_years}y vs benchmark {bench:,.0f}/yr")
        if np.isfinite(ratio) and ratio > 0:
            adj = ours / ratio
            rep.judge("D_benchmark",
                      "recruitment per year vs benchmark (perimeter-adjusted)",
                      rel(adj, bench), TOL_BENCHMARK,
                      f"ours {adj:,.0f}/yr after dividing by {ratio:.3f} vs "
                      f"benchmark {bench:,.0f}/yr")
        else:
            rep.add("D_benchmark",
                    "recruitment per year vs benchmark (perimeter-adjusted)", "SKIP",
                    detail="perimeter ratio unavailable")
    else:
        rep.add("D_benchmark", "recruitment per year vs benchmark (raw)", "SKIP",
                detail="no recruitment column")

    ret_series = None
    if rw:
        ret_series = pd.to_numeric(win[rw], errors="coerce")
    else:
        sp = read_csv_safe(Path(SUPPLY_DIR) / "supply_projection.csv")
        if sp is not None:
            yc2 = find_col(sp.columns, "year")
            rw2 = find_col(sp.columns, "retirements_wave", "retire", "reforma")
            if yc2 and rw2:
                spw = year_filter(sp, yc2, lo, hi)
                ret_series = pd.to_numeric(spw[rw2], errors="coerce")
                print("    note: retirement wave read from supply_projection.csv")
    if ret_series is not None and ret_series.notna().any():
        ours_ret = float(ret_series.mean())
        rep.judge("D_benchmark", "retirements per year vs benchmark",
                  rel(ours_ret, BENCH_RETIRE_ANNUAL), TOL_BENCHMARK,
                  f"ours {ours_ret:,.0f}/yr vs benchmark {BENCH_RETIRE_ANNUAL:,.0f}/yr; "
                  f"note the 55-59 component is an assumption, not an estimate")
    else:
        rep.add("D_benchmark", "retirements per year vs benchmark", "SKIP",
                detail="no retirement series found")

    if nc:
        cum_ours = float(pd.to_numeric(win[nc], errors="coerce").sum())
        cum_bench = BENCH_RECRUIT_TOTAL * (n_years / BENCH_YEARS)
        rep.judge("D_benchmark", "cumulative recruitment vs benchmark (raw)",
                  rel(cum_ours, cum_bench), TOL_BENCHMARK,
                  f"ours {cum_ours:,.0f} vs benchmark {cum_bench:,.0f} "
                  f"(pro-rated to {n_years}y)")


# ============================================================ E. PLAUSIBILITY
def _in_range(v, lo, hi):
    return np.isfinite(v) and lo <= v <= hi


def check_plausibility(rep):
    print("\n[E] Plausibility ranges")

    h = read_csv_safe(Path(SUPPLY_DIR) / "supply_hazards.csv")
    if h is not None:
        bc = find_col(h.columns, "band")
        hc = (find_exact(h.columns, "hazard_flat_chain", "hazard_deployed")
              or find_col(h.columns, "hazard"))
        if bc and hc:
            row = h[h[bc].astype(str).str.contains("60", na=False)]
            if not row.empty:
                v = float(pd.to_numeric(row[hc], errors="coerce").iloc[0])
                lo, hi = BOUND_HAZARD_60
                rep.add("E_plausibility", "60+ exit hazard in range",
                        "PASS" if _in_range(v, lo, hi) else "FAIL", v, hi,
                        f"bounds [{lo}, {hi}], column '{hc}'")
            else:
                rep.add("E_plausibility", "60+ exit hazard in range", "SKIP",
                        detail="band not found")
        else:
            rep.add("E_plausibility", "60+ exit hazard in range", "SKIP",
                    detail="columns missing")
    else:
        rep.add("E_plausibility", "60+ exit hazard in range", "SKIP",
                detail="supply_hazards.csv missing")

    ch = read_csv_safe(Path(GAP_DIR) / "conversion_history.csv")
    if ch is not None:
        rc = find_exact(ch.columns, "rate") or find_col(ch.columns, "rate")
        cc = find_exact(ch.columns, "cohort_conversion")
        if rc:
            r = pd.to_numeric(ch[rc], errors="coerce").dropna()
            lo, hi = BOUND_CONVERSION
            bad = int(((r < lo) | (r > hi)).sum())
            rep.add("E_plausibility", "conversion rates in [1%, 100%]",
                    "PASS" if bad == 0 else "FAIL", float(r.max()) if len(r) else np.nan,
                    hi, f"{bad} of {len(r)} outside; max={r.max():.1%}" if len(r) else "")
        else:
            rep.add("E_plausibility", "conversion rates in range", "SKIP",
                    detail="no rate column")
        if cc:
            v = pd.to_numeric(ch[cc], errors="coerce").dropna()
            lo, hi = BOUND_COHORT_CONV
            bad = int((v > hi).sum())
            rep.add("E_plausibility", "cohort conversion <= 100%",
                    "PASS" if bad == 0 else "FAIL",
                    float(v.max()) if len(v) else np.nan, hi,
                    f"{bad} of {len(v)} above 100%; max={v.max():.0%}" if len(v) else "")
    else:
        rep.add("E_plausibility", "conversion rates in range", "SKIP",
                detail="conversion_history.csv missing")

    nat = read_csv_safe(Path(GAP_DIR) / "gap_national.csv")
    if nat is not None:
        nc = find_exact(nat.columns, "recruitment_need") or find_col(nat.columns, "recruit")
        if nc:
            v = pd.to_numeric(nat[nc], errors="coerce").dropna()
            lo, hi = BOUND_RECRUIT
            bad = int(((v < lo) | (v > hi)).sum())
            rep.add("E_plausibility", "annual recruitment in plausible range",
                    "PASS" if bad == 0 else "FAIL", float(v.mean()), hi,
                    f"{bad} of {len(v)} outside [{lo:,}, {hi:,}]")
        else:
            rep.add("E_plausibility", "annual recruitment in range", "SKIP",
                    detail="no column")
    else:
        rep.add("E_plausibility", "annual recruitment in range", "SKIP",
                detail="gap_national missing")

    if Path(MASTER_FILE).exists():
        try:
            xls = pd.ExcelFile(MASTER_FILE, engine="openpyxl")
            sh = "historical" if "historical" in xls.sheet_names else xls.sheet_names[0]
            m = pd.read_excel(MASTER_FILE, sheet_name=sh, engine="openpyxl")
            m.columns = [str(c).strip() for c in m.columns]
            yc = find_col(m.columns, "year_start", "year")
            m[yc] = pd.to_numeric(m[yc], errors="coerce")
            last = m[m[yc] == m[yc].max()]
            pairs = [("students_basic_1", "teachers_basic_1", "basic_1"),
                     ("students_pre_school", "teachers_pre_school", "pre_school"),
                     ("students_basic_2", "teachers_basic_2", "basic_2")]
            worst_lab, worst_v, bad = "", np.nan, 0
            for s, t, lab in pairs:
                if s in last.columns and t in last.columns:
                    ss = pd.to_numeric(last[s], errors="coerce").sum()
                    tt = pd.to_numeric(last[t], errors="coerce").sum()
                    if tt > 0:
                        v = ss / tt
                        lo, hi = BOUND_RATIO
                        if not _in_range(v, lo, hi):
                            bad += 1
                            worst_lab, worst_v = lab, v
                        elif not np.isfinite(worst_v):
                            worst_lab, worst_v = lab, v
            rep.add("E_plausibility", "student-teacher ratios in range",
                    "PASS" if bad == 0 else "FAIL", worst_v, BOUND_RATIO[1],
                    f"{bad} cycle(s) outside [{BOUND_RATIO[0]}, {BOUND_RATIO[1]}]"
                    + (f"; worst {worst_lab}" if bad else ""))
        except Exception as e:
            rep.add("E_plausibility", "student-teacher ratios in range", "SKIP",
                    detail=str(e)[:60])
    else:
        rep.add("E_plausibility", "student-teacher ratios in range", "SKIP",
                detail="master missing")


# ============================================================ F. HINDCAST
def _national_age_panel():
    if not Path(MASTER_FILE).exists():
        return None, None
    try:
        xls = pd.ExcelFile(MASTER_FILE, engine="openpyxl")
        sh = "historical" if "historical" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(MASTER_FILE, sheet_name=sh, engine="openpyxl")
    except Exception:
        return None, None
    df.columns = [str(c).strip() for c in df.columns]
    if not all(f"teachers_{b}" in df.columns for b in BANDS):
        return None, None
    yc = find_col(df.columns, "year_start", "year")
    df[yc] = pd.to_numeric(df[yc], errors="coerce")
    df = df.dropna(subset=[yc]); df[yc] = df[yc].astype(int)
    df = df[df[yc] >= FIRST_REAL_YEAR]
    years = sorted(df[yc].unique())
    if not years:
        return None, None
    nat = {}
    for y in years:
        s = df[df[yc] == y]
        nat[y] = np.array([float(pd.to_numeric(s[f"teachers_{b}"], errors="coerce").sum())
                           for b in BANDS])
    real = [years[0]]
    for i in range(1, len(years)):
        if not np.allclose(nat[years[i]], nat[years[i - 1]]):
            real.append(years[i])
    if len(real) < len(years):
        dropped = [y for y in years if y not in real]
        print(f"    note: placeholder year(s) dropped from the hindcast panel: {dropped}")
    return nat, real


def _estimate_hazards(nat, years):
    """[VAL-6] Inverts the projection equation with NO attrition floor.

    03_supply uses a different estimator: identity-based gross exits with a floor. This is
    deliberately independent, so the hindcast is not scoring the code that produced the
    forecast. It also means a good MASE does not certify 03's own estimator.
    """
    H = {a: [] for a in range(NB)}
    ys = sorted(years)
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        for a in range(1, NB - 1):
            if n0[a] > 0:
                H[a].append(np.clip(1 - G - (n1[a] - G * n0[a - 1]) / n0[a], *HCLIP))
        if n0[NB - 1] > 0:
            H[NB - 1].append(np.clip(1 - (n1[NB - 1] - G * n0[NB - 2]) / n0[NB - 1], *HCLIP))
    h = np.array([np.nanmean(H[a]) if H[a] else HINDCAST_BAND0_HAZARD for a in range(NB)])
    h[0] = HINDCAST_BAND0_HAZARD
    return h


def _entry_profile(nat, years):
    ys = sorted(years)
    prof = np.zeros(NB)
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        for a in range(NB - 1):
            agein = 0.0 if a == 0 else G * n0[a - 1]
            ageout = G * n0[a]
            net = (n1[a] - n0[a]) - agein + ageout
            prof[a] += max(net, 0.0)
    return prof / prof.sum() if prof.sum() > 0 else np.eye(NB)[0]


def _project(n0, h, entries, prof, steps):
    n = n0.astype(float).copy()
    out = []
    for _ in range(steps):
        new = np.zeros(NB)
        for a in range(NB):
            if a < NB - 1:
                stay = max(1 - G - h[a], 0.0)
                new[a] += n[a] * stay
                new[a + 1] += n[a] * G
            else:
                new[a] += n[a] * max(1 - h[a], 0.0)
        n = np.clip(new + entries * prof, 0, None)
        out.append(n.copy())
    return out


def check_hindcast(rep):
    print(f"\n[F] Hindcast -- out-of-sample test of a RE-IMPLEMENTED cohort model "
          f"(hold out {HINDCAST_HOLDOUT} years)")
    print("    [VAL-6] this is NOT 03_supply's deployed code: the hazards are estimated")
    print("    without an attrition floor and entries are held CONSTANT, where 03 uses an")
    print("    identity-based estimator and endogenous entry driven by 02's demand.")
    nat, years = _national_age_panel()
    if nat is None or years is None or len(years) < HINDCAST_MIN_TRAIN + HINDCAST_HOLDOUT:
        rep.add("F_hindcast", "cohort model out-of-sample MASE", "SKIP",
                detail="insufficient observed years")
        return None, None
    train_years = years[:-HINDCAST_HOLDOUT]
    test_years = years[-HINDCAST_HOLDOUT:]
    print(f"    train {train_years[0]}-{train_years[-1]} | "
          f"test {test_years[0]}-{test_years[-1]}")

    h = _estimate_hazards(nat, train_years)
    prof = _entry_profile(nat, train_years)
    ent = []
    for t, t1 in zip(train_years[:-1], train_years[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        tot = 0.0
        for a in range(NB - 1):
            agein = 0.0 if a == 0 else G * n0[a - 1]
            net = (n1[a] - n0[a]) - agein + G * n0[a]
            tot += max(net, 0.0)
        ent.append(tot)
    entries = float(np.mean(ent)) if ent else 0.0

    n_last = nat[train_years[-1]]
    steps = test_years[-1] - train_years[-1]
    proj = _project(n_last, h, entries, prof, steps)

    rows = []
    for y in test_years:
        idx = y - train_years[-1] - 1
        if idx < 0 or idx >= len(proj):
            continue
        pred = proj[idx]; obs = nat[y]
        naive = n_last
        for a in range(NB):
            rows.append({"year": int(y), "band": NICE[a], "observed": obs[a],
                         "predicted": pred[a], "naive": naive[a],
                         "abs_err": abs(pred[a] - obs[a]),
                         "abs_err_naive": abs(naive[a] - obs[a])})
    det = pd.DataFrame(rows)
    if det.empty:
        rep.add("F_hindcast", "cohort model out-of-sample MASE", "SKIP",
                detail="no test rows")
        return None, None

    mae = det["abs_err"].mean()
    mae_naive = det["abs_err_naive"].mean()
    mase = mae / mae_naive if mae_naive > 0 else np.nan

    # [VAL-3] per-band MASE. The aggregate pools bands of 800 and 26,000 teachers, so it
    # is dominated by the largest ones, where the no-change benchmark collapses.
    per = det.groupby("band", sort=False).agg(
        mae=("abs_err", "mean"), mae_naive=("abs_err_naive", "mean"),
        level=("observed", "mean"))
    # [VAL-8] Guard the denominator on SCALE, not on exact zero. replace(0, nan) only
    # catches an exact zero; a naive MAE of 1e-11 on a band of 26,000 teachers is
    # numerically zero and produced a MASE of 1.3e+14 in testing, which then became the
    # reported "worst band". A naive error below this fraction of the band level means
    # the no-change benchmark is effectively exact and the ratio carries no information.
    floor = np.maximum(MASE_DENOM_REL * per["level"].abs(), MASE_DENOM_ABS)
    usable = per["mae_naive"] >= floor
    per["mase"] = np.where(usable, per["mae"] / per["mae_naive"].where(usable), np.nan)
    per["naive_degenerate"] = ~usable
    per["beats_naive"] = per["mase"] < 1.0
    per = per.reset_index()
    n_degen = int(per["naive_degenerate"].sum())
    if n_degen:
        print(f"    [VAL-8] {n_degen} band(s) excluded from the MASE ranking: the "
              f"no-change error is numerically zero there, so the ratio is undefined "
              f"({', '.join(per.loc[per['naive_degenerate'], 'band'])})")
    lost = per[(~per["beats_naive"].fillna(False)) & (~per["naive_degenerate"])]
    n_lost = int(len(lost))
    worst_row = per.loc[per["mase"].idxmax()] if per["mase"].notna().any() else None

    status = ("PASS" if (np.isfinite(mase) and mase < 1.0)
              else ("WARN" if np.isfinite(mase) else "SKIP"))
    rep.add("F_hindcast", "cohort model MASE vs naive (aggregate)", status,
            float(mase), 1.0,
            f"MAE {mae:,.0f} vs naive {mae_naive:,.0f}; pooled across bands of very "
            f"different size -- see the per-band check")

    if n_lost == 0:
        st = "PASS"
    elif n_lost <= HINDCAST_MAX_BANDS_LOST_WARN:
        st = "WARN"
    elif n_lost < HINDCAST_MAX_BANDS_LOST_FAIL:
        st = "WARN"
    else:
        st = "FAIL"
    detail = f"{n_lost} of {len(per)} bands lose to no-change"
    if worst_row is not None and np.isfinite(worst_row["mase"]):
        detail += f"; worst {worst_row['band']} at MASE {worst_row['mase']:.2f}"
    if n_lost:
        detail += f" ({', '.join(lost['band'].tolist())})"
    rep.add("F_hindcast", "bands where the model beats naive", st,
            float(n_lost), float(HINDCAST_MAX_BANDS_LOST_FAIL), detail)

    tot_obs = det.groupby("year")["observed"].sum()
    tot_pred = det.groupby("year")["predicted"].sum()
    tot_err = float(np.nanmax(np.abs(tot_pred - tot_obs) / np.maximum(tot_obs, 1.0)))
    rep.judge("F_hindcast", "total headcount error (worst test year)", tot_err, 0.05,
              "aggregate is easier than the age structure")

    det.to_csv(RES / "hindcast_detail.csv", index=False)
    per.to_csv(RES / "hindcast_mase_by_band.csv", index=False)
    print(f"    [csv] hindcast_detail.csv ({len(det)} rows)")
    print(f"    [csv] hindcast_mase_by_band.csv")
    print(f"    per-band MASE: " + " | ".join(
        f"{r['band']}={r['mase']:.2f}" for _, r in per.iterrows()
        if np.isfinite(r["mase"])))
    return det, per


# ============================================================ G. SILENT FAILURES
def check_silent_failures(rep):
    print("\n[G] Silent-failure guards (did the upstream blocks produce a usable result?)")

    cd = read_csv_safe(Path(UNC_DIR) / "ceiling_and_diagnostics.csv")

    # G1 -- 05 must not have collapsed onto a point
    if cd is not None:
        col = find_exact(cd.columns, "simulation_degenerate")
        if col is not None:
            flag = as_bool(cd[col].iloc[0])
            if flag is None:
                rep.add("G_silent", "05 simulation is not degenerate", "SKIP",
                        detail="flag not readable")
            elif flag:
                rep.add("G_silent", "05 simulation is not degenerate", "FAIL",
                        detail="05 reports a DEGENERATE simulation: the deficit "
                               "distribution has zero width, so every interval it "
                               "publishes is meaningless")
            else:
                rep.add("G_silent", "05 simulation is not degenerate", "PASS",
                        detail="05 reports a non-degenerate simulation")
        else:
            rep.add("G_silent", "05 simulation is not degenerate", "SKIP",
                    detail="no simulation_degenerate column -- run the current 05")
    else:
        rep.add("G_silent", "05 simulation is not degenerate", "SKIP",
                detail="ceiling_and_diagnostics.csv missing")

    # G2 -- [VAL-5] recompute the band width, absolute AND relative to the median need
    fan = read_csv_safe(Path(UNC_DIR) / "deficit_fan.csv")
    if fan is not None:
        c10 = find_exact(fan.columns, "p10"); c90 = find_exact(fan.columns, "p90")
        cmn = find_exact(fan.columns, "median_need")
        if c10 and c90:
            w = (pd.to_numeric(fan[c90], errors="coerce")
                 - pd.to_numeric(fan[c10], errors="coerce")).dropna()
            if len(w):
                worst = float(w.max())
                med_band = float(w.median())
                scale = np.nan
                if cmn is not None:
                    mn = pd.to_numeric(fan[cmn], errors="coerce").dropna()
                    scale = float(mn.median()) if len(mn) else np.nan
                floor_abs = MIN_FAN_BAND
                floor_rel = (MIN_FAN_BAND_REL * scale) if np.isfinite(scale) else 0.0
                floor = max(floor_abs, floor_rel)
                rel_band = worst / scale if (np.isfinite(scale) and scale > 0) else np.nan
                note = f"median band {med_band:,.0f}, widest {worst:,.0f} teachers"
                if np.isfinite(rel_band):
                    note += f" = {rel_band:.1%} of the median need"
                if worst < floor:
                    rep.add("G_silent", "05 deficit fan has a non-zero band", "FAIL",
                            worst, floor,
                            f"widest P10-P90 is {worst:,.2f}, below the floor of "
                            f"{floor:,.1f} -- the fan is effectively a single line")
                else:
                    rep.add("G_silent", "05 deficit fan has a non-zero band", "PASS",
                            worst, floor, note)
            else:
                rep.add("G_silent", "05 deficit fan has a non-zero band", "SKIP",
                        detail="no usable percentile rows")
        else:
            rep.add("G_silent", "05 deficit fan has a non-zero band", "SKIP",
                    detail="p10/p90 columns missing")
    else:
        rep.add("G_silent", "05 deficit fan has a non-zero band", "SKIP",
                detail="deficit_fan.csv missing")

    # G3 -- every uncertainty axis must be able to vary
    if cd is not None:
        col = find_exact(cd.columns, "degenerate_axes")
        if col is not None:
            raw = cd[col].iloc[0]
            txt = "" if (pd.isna(raw) or str(raw).strip().lower() == "nan") else str(raw).strip()
            if txt:
                axes = [a.strip() for a in txt.split(",") if a.strip()]
                rep.add("G_silent", "all 05 uncertainty axes can vary", "FAIL",
                        float(len(axes)), 0.0,
                        f"{len(axes)} axis/axes carry no dispersion: {', '.join(axes)}")
            else:
                rep.add("G_silent", "all 05 uncertainty axes can vary", "PASS",
                        0.0, 0.0, "no axis reported as degenerate")
        else:
            rep.add("G_silent", "all 05 uncertainty axes can vary", "SKIP",
                    detail="no degenerate_axes column -- run the current 05")
    else:
        rep.add("G_silent", "all 05 uncertainty axes can vary", "SKIP",
                detail="ceiling_and_diagnostics.csv missing")

    # G4 -- [VAL-1] the anchor verdict, under every name 04 has used for it
    ar = read_csv_safe(Path(GAP_DIR) / "anchor_robustness.csv")
    if ar is None:
        rep.add("G_silent", "04 deficit sign is robust to the entry anchor", "SKIP",
                detail="anchor_robustness.csv missing -- run 04_gap")
    else:
        col = find_exact(ar.columns,
                         "verdict_robust_without_cap",   # current 04
                         "sign_robust_without_cap",      # previous 04
                         "sign_robust_to_anchor")        # name this suite assumed
        ga = find_exact(ar.columns, "mean_gap_anchored")
        gu = find_exact(ar.columns, "mean_gap_unanchored_raw", "mean_gap_unanchored")
        note = ""
        if ga and gu:
            va, vu = first_num(ar, ga), first_num(ar, gu)
            note = f"mean gap {va:+,.0f}/yr anchored vs {vu:+,.0f}/yr unanchored"
        if col is None:
            # [VAL-1] a missing verdict in the family built to catch silent failures is
            # itself the silent failure. Do NOT skip.
            rep.add("G_silent", "04 deficit sign is robust to the entry anchor", "FAIL",
                    detail=("anchor_robustness.csv carries no recognisable verdict "
                            "column; the 04 output is stale or renamed, so the anchor "
                            "dependence is UNCHECKED. " + note).strip())
        else:
            flag = as_bool(ar[col].iloc[0])
            if flag is None:
                rep.add("G_silent", "04 deficit sign is robust to the entry anchor",
                        "SKIP", detail=f"column '{col}' not readable as a boolean")
            elif flag:
                rep.add("G_silent", "04 deficit sign is robust to the entry anchor",
                        "PASS",
                        detail=(note or "verdict holds with and without the anchor")
                               + f" [via '{col}']")
            else:
                rep.add("G_silent", "04 deficit sign is robust to the entry anchor",
                        "FAIL",
                        detail=("the verdict DEPENDS on the external anchor -- report "
                                "this prominently; " + note).strip("; "))

    # G5 -- [VAL-7] negative needs floored in 05
    if cd is not None:
        col = find_exact(cd.columns, "negative_need_cells_floored")
        if col is not None:
            n = first_num(cd, col)
            if not np.isfinite(n):
                rep.add("G_silent", "05 had no negative recruitment needs", "SKIP",
                        detail="counter not readable")
            elif n > 0:
                rep.add("G_silent", "05 had no negative recruitment needs", "WARN",
                        float(n), 0.0,
                        f"{int(n)} (scenario, year) cell(s) had a NEGATIVE need and were "
                        f"floored at zero: demand fell faster than replacement there")
            else:
                rep.add("G_silent", "05 had no negative recruitment needs", "PASS",
                        0.0, 0.0, "no scenario pushed the need below zero")
        else:
            rep.add("G_silent", "05 had no negative recruitment needs", "SKIP",
                    detail="no negative_need_cells_floored column -- run the current 05")
    else:
        rep.add("G_silent", "05 had no negative recruitment needs", "SKIP",
                detail="ceiling_and_diagnostics.csv missing")


# ============================================================ FIGURE
def fig_summary(rep_df, per_band):
    fams = ["A_accounting", "B_crossblock", "C_spatial", "D_benchmark",
            "E_plausibility", "F_hindcast", "G_silent"]
    labels = ["Accounting", "Cross-block", "Spatial", "Benchmark",
              "Plausibility", "Hindcast", "Silent-failure"]
    statuses = ["PASS", "WARN", "FAIL", "SKIP"]
    colours = {"PASS": OK, "WARN": WARNC, "FAIL": BAD, "SKIP": SKIPC}

    ev = rep_df[~rep_df["by_construction"]]
    ta = rep_df[rep_df["by_construction"]]

    has_h = per_band is not None and not per_band.empty
    fig, axes = plt.subplots(1, 2 if has_h else 1,
                             figsize=(16 if has_h else 10, 5.6), squeeze=False)

    ax = axes[0][0]
    bottom = np.zeros(len(fams))
    for s in statuses:
        v = np.array([int(((ev.family == f) & (ev.status == s)).sum()) for f in fams],
                     float)
        ax.bar(labels, v, bottom=bottom, color=colours[s], label=s, alpha=0.92)
        for i, (b, h_) in enumerate(zip(bottom, v)):
            if h_ > 0:
                ax.text(i, b + h_ / 2, f"{int(h_)}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        bottom += v
    # [VAL-2] tautologies stacked on top, in grey, so they are visible but not counted
    tv = np.array([int((ta.family == f).sum()) for f in fams], float)
    if tv.sum() > 0:
        ax.bar(labels, tv, bottom=bottom, color=TAUTC, alpha=0.75, hatch="///",
               edgecolor="white", label="by construction (not evidence)")
        for i, (b, h_) in enumerate(zip(bottom, tv)):
            if h_ > 0:
                ax.text(i, b + h_ / 2, f"{int(h_)}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
    ax.set_title("Validation results by family", fontweight="bold")
    ax.set_ylabel("number of checks")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(alpha=0.2, axis="y"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3)

    if has_h:
        ax2 = axes[0][1]
        p = per_band.dropna(subset=["mase"])
        cols = [OK if v < 1 else BAD for v in p["mase"]]
        ax2.bar(p["band"], p["mase"], color=cols, alpha=0.9)
        ax2.axhline(1.0, color=INK, lw=1.4, ls="--")
        ax2.text(len(p) - 0.5, 1.03, "no-change benchmark", ha="right", va="bottom",
                 fontsize=8.5, color=INK)
        for i, v in enumerate(p["mase"]):
            ax2.text(i, v, f"{v:.2f}", ha="center",
                     va="bottom" if v < 1 else "bottom", fontsize=8)
        ax2.set_xticks(range(len(p)))
        ax2.set_xticklabels(p["band"], rotation=30, ha="right")
        n_lost = int((p["mase"] >= 1).sum())
        ax2.set_title(f"Hindcast MASE by age band -- the model loses in "
                      f"{n_lost} of {len(p)}", fontweight="bold")
        ax2.set_ylabel("MASE (below 1 beats no-change)")
        ax2.grid(alpha=0.2, axis="y"); ax2.spines[["top", "right"]].set_visible(False)
        ax2.text(0.99, 0.97,
                 "the aggregate MASE pools bands of 800 and 26,000\n"
                 "teachers and hides this [VAL-3]",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=8, color="0.45")

    fig.tight_layout()
    out = IMG / "validation_summary.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", out)


# ============================================================ MAIN
def main():
    print("=" * 78)
    print("06_validation -- validation suite for the teacher demand/supply pipeline")
    print("=" * 78)
    rep = Report()

    check_accounting(rep)
    check_crossblock(rep)
    check_spatial(rep)
    check_benchmark(rep)
    check_plausibility(rep)
    hind, per_band = check_hindcast(rep)
    check_silent_failures(rep)

    df = rep.frame()
    df.to_csv(RES / "validation_report.csv", index=False)

    n = len(df)
    ev = df[~df["by_construction"]]
    ta = df[df["by_construction"]]
    npass = int((ev.status == "PASS").sum())
    nwarn = int((ev.status == "WARN").sum())
    nfail = int((ev.status == "FAIL").sum())
    nskip = int((ev.status == "SKIP").sum())

    print("\n" + "=" * 78)
    print(f"SUMMARY (evidential checks only): {npass} PASS | {nwarn} WARN | "
          f"{nfail} FAIL | {nskip} SKIP  (of {len(ev)})")
    print(f"plus {len(ta)} check(s) that hold BY CONSTRUCTION and are not evidence")
    print("=" * 78)

    # [VAL-2] name them, so nobody quotes the headline as independent corroboration
    if len(ta):
        print("\nBY CONSTRUCTION -- these re-derive definitions and cannot fail:")
        for _, r in ta.iterrows():
            print(f"    {r['check']}")
            print(f"        {r['construction_reason']}")
        print("    They are retained because a corrupted file or a wrong column would")
        print("    still trip them. They are not corroboration of the model.")

    gfail = df[(df.family == "G_silent") & (df.status == "FAIL")]
    if len(gfail):
        print("\nSILENT FAILURES -- the upstream block ran but its output is not usable:")
        for _, r in gfail.iterrows():
            print(f"    {r['check']}: {r['detail']}")
        print("    Do NOT quote intervals or probabilities from that block until fixed.")

    other_fail = df[(df.status == "FAIL") & (df.family != "G_silent")]
    if len(other_fail):
        print("\nFAILURES -- fix or disclose explicitly in the report:")
        for _, r in other_fail.iterrows():
            print(f"    [{r['family']}] {r['check']}: {r['detail']}")
    if int((df.status == "WARN").sum()):
        print("\nWARNINGS -- outside tolerance but arguably explainable:")
        for _, r in df[df.status == "WARN"].iterrows():
            print(f"    [{r['family']}] {r['check']}: {r['detail']}")
    if nskip:
        print(f"\n{nskip} evidential check(s) skipped for missing inputs. Run the upstream")
        print("blocks to enable them. A skipped check is NOT a passed check.")
    if nfail == 0 and nwarn == 0 and nskip == 0:
        print("\nAll evidential checks pass. That means the pipeline is internally")
        print("coherent and externally plausible -- not that the forecast is correct.")

    fig_summary(df, per_band)

    print("\n[caveat] Validation covers coherence, plausibility, one out-of-sample test of")
    print("         a RE-IMPLEMENTED cohort model [VAL-6], and the upstream blocks' own")
    print("         self-diagnostics. It cannot validate structural assumptions:")
    print("         non-restrictive entry in 03_supply, the observed-versus-normative")
    print("         ratio in 02_demand, or the graduate-pool definition in 04_gap.")
    print("=" * 78)
    print("Done. Report:", RES / "validation_report.csv")
    print("      Figure:", IMG / "validation_summary.png")
    print("=" * 78)


if __name__ == "__main__":
    main()
