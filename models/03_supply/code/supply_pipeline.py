
from pathlib import Path
import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ============================================================ PATHS
BASE = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
TEACHERS_FILE = BASE / r"models\00_data\master_panel_nuts3_with_age.xlsx"
PANEL = TEACHERS_FILE
DEMAND_FILE = BASE / r"models\02_demand\results\teacher_demand_selected.csv"
RES = BASE / r"models\03_supply\results"
IMG = BASE / r"models\03_supply\images"

SUPPLY_FC_CANDIDATES = [
    RES / "supply_projection.csv",
    RES / "supply_universe_national.csv",
]
STUDENTS_FC_CANDIDATES = [
    BASE / r"models\02_demand\results\teacher_demand_by_cell.csv",
    BASE / r"models\02_demand\results\students_forecast.csv",
    BASE / r"models\01_students\results\students_forecast.csv",
    BASE / r"models\01_students\results\students_forecast_nuts3.xlsx",
]
SEARCH_DIRS = [
    BASE / r"models\03_supply\results",
    BASE / r"models\02_demand\results",
    BASE / r"models\01_students\results",
]

if os.environ.get("SUP_TEST") == "1":
    TEACHERS_FILE = Path("rob/master_panel_nuts3_with_age.xlsx")
    PANEL = TEACHERS_FILE
    DEMAND_FILE = Path(os.environ.get("SUP_DEMAND", "rob/none.csv"))
    RES = Path("rob/results")
    IMG = Path("rob/images")
    SUPPLY_FC_CANDIDATES = [RES / "supply_projection.csv"]
    STUDENTS_FC_CANDIDATES = [Path("rob/students_forecast.csv")]
    SEARCH_DIRS = [RES]

RES.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

# ============================================================ SHARED CONFIG
BANDS = ["lt25", "25_29", "30_34", "35_39", "40_44",
         "45_49", "50_54", "55_59", "60_plus"]
NICE = ["<25", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60+"]
NB = len(BANDS)

G = 1.0 / 5.0
FYEARS = list(range(2025, 2041))
HCLIP = (0.0, 0.6)
LAST = 2024

FIRST_REAL_YEAR = 2014
FIRST_FLOW_YEAR = 2015
SPLIT = 2024.5
BASE_ATTRITION_YOUNGMID = 0.01

# [SUP-36] 55-59 carries the same attrition floor as every other band below
# 60+, because _gross_exits charges it regardless.
# [SUP-56] Cost of that choice: the observed 55-59 hazard is then the floor in
# every year, so it stops carrying information. Stated wherever it is printed.
ATTRITION_FLOOR_ON_5559 = True

ENTRY_PROFILE_WINDOW = "full"
CHANNEL_SHARE_WINDOW = "last3"

# [SUP-15][SUP-18] 60+ hazard source.
H60_BASE_SOURCE = "legal"
H60_TARGET_EXIT_AGE = 67.0
H60_FIXED_BASE = 0.12
H60_SCENARIOS = {"base": 1.0 / (H60_TARGET_EXIT_AGE - 60.0),
                 "recent_3y": 0.12, "post_2017": np.nan,
                 "reversal": 0.25, "persistent": 0.10}

# [SUP-3][SUP-24] flat vs tenure queue. WIRED into project().
H60_MODE = "flat"

ENTRY_PROFILE_REPAIR = True
ARTEFACT_RATIO = 0.25
YOUNG_BANDS = (0, 1)

# [SUP-53] a scenario within this fraction of the base is treated as
# disconnected rather than as a result.
DEGENERATE_TOL = 1e-9

# [SUP-64] a net cohort multiplier below this is reported as genuine shrinkage
# rather than rounding.
COHORT_SHRINK_TOL = 0.99

# --- stage 2 ---
MIN_HAZARD_YEAR = FIRST_FLOW_YEAR
N_MC = 2000
MC_SEED = 20260820
SIXTY_SLOTS = 12
LEGAL_GAP = 6
H60_EARLY = 0.06
H60_PEAK = 0.55
H60_RAMP = 3

MC_RET_AGE_SD = 1.0
MC_RET_SCALE_SD = 0.10
MC_DEMAND_STEP = 0.006
MC_DEMAND_FIRST_YEAR_FRAC = 0.25

# --- colours ---
INK = "#1a1a1a"
AGEING = "#3b4a5a"
EXITC = "#b5171e"
ENTRY = "#128a4b"
STAYC = "#9aa4ad"
MIDC = "#e08a2c"

# [SUP-67] NUTS III codes carry no meaning on a chart axis. Mapping is the
# official NUTS 2024 list in force since 1 January 2024 (Regulamento delegado
# (UE) 2023/674). The panel stores codes without the "PT" prefix, so lookups
# below normalise both forms.
NUTS3_NAMES = {
    "PT111": "Alto Minho",
    "PT112": "Cávado",
    "PT119": "Ave",
    "PT11A": "Área Metropolitana do Porto",
    "PT11B": "Alto Tâmega e Barroso",
    "PT11C": "Tâmega e Sousa",
    "PT11D": "Douro",
    "PT11E": "Terras de Trás-os-Montes",
    "PT150": "Algarve",
    "PT191": "Região de Aveiro",
    "PT192": "Região de Coimbra",
    "PT193": "Região de Leiria",
    "PT194": "Viseu Dão Lafões",
    "PT195": "Beira Baixa",
    "PT196": "Beiras e Serra da Estrela",
    "PT1A0": "Grande Lisboa",
    "PT1B0": "Península de Setúbal",
    "PT1C1": "Alentejo Litoral",
    "PT1C2": "Baixo Alentejo",
    "PT1C3": "Alto Alentejo",
    "PT1C4": "Alentejo Central",
    "PT1D1": "Oeste",
    "PT1D2": "Médio Tejo",
    "PT1D3": "Lezíria do Tejo",
    "PT200": "Região Autónoma dos Açores",
    "PT300": "Região Autónoma da Madeira",
}

NUTS2_NAMES = {
    "PT11": "Norte",
    "PT15": "Algarve",
    "PT19": "Centro",
    "PT1A": "Grande Lisboa",
    "PT1B": "Península de Setúbal",
    "PT1C": "Alentejo",
    "PT1D": "Oeste e Vale do Tejo",
    "PT20": "Região Autónoma dos Açores",
    "PT30": "Região Autónoma da Madeira",
}

LEVELS = [
    ("Pre-primary", "students_pre_school", "teachers_pre_school", "#2c7fb8"),
    ("Primary (1st cycle)", "students_basic_1", "teachers_basic_1", "#41ab5d"),
    ("2nd cycle", "students_basic_2", "teachers_basic_2", "#e08a2c"),
    ("3rd cycle + secondary", ("students_basic_3", "students_secondary"),
     "teachers_basic_3_secondary", "#b5171e"),
]

_MANIFEST = {}


# ============================================================================
# SHARED FLOW ARITHMETIC
# ============================================================================

def region_label(code, scope="nuts3"):
    """[SUP-67] Turn a region code into its NUTS 2024 designation.

    The panel stores NUTS III codes both with and without the "PT" prefix
    ("1A0" and "PT1A0" both occur), so the code is normalised before lookup.
    An unmatched code is returned unchanged rather than silently relabelled,
    which keeps a bad join visible on the chart instead of hiding it.
    """
    s = str(code).strip().upper()
    if not s.startswith("PT"):
        s = "PT" + s
    table = NUTS2_NAMES if scope == "nuts2" else NUTS3_NAMES
    return table.get(s, str(code))


def band_floor(a):
    """[SUP-36] Attrition floor for band `a`, including 55-59."""
    if a == NB - 1:
        return 0.0
    if a == NB - 2 and not ATTRITION_FLOOR_ON_5559:
        return 0.0
    return BASE_ATTRITION_YOUNGMID


def flow_years(years):
    """[SUP-25][SUP-33] Years eligible for flow estimation."""
    return [y for y in sorted(years) if y >= FIRST_FLOW_YEAR]


def flow_window(years):
    """[SUP-45] First and last year of the flow series, and the gap."""
    ys = flow_years(years)
    obs = [t1 for t, t1 in zip(ys[:-1], ys[1:]) if t1 - t == 1]
    if not obs:
        return None, None, None
    return int(obs[0]), int(obs[-1]), int(obs[0]) - int(min(years))


def _net_flow(n0, n1, a):
    """[SUP-32] Net flow into band `a`: entries - exits = net."""
    agein = 0.0 if a == 0 else G * n0[a - 1]
    ageout = 0.0 if a == NB - 1 else G * n0[a]
    return (n1[a] - n0[a]) - agein + ageout


def _gross_entries(n0, n1, a, attrition):
    """[SUP-16] Gross entries: net + h*n0, floored at zero."""
    return max(_net_flow(n0, n1, a) + attrition * n0[a], 0.0)


def _gross_exits(n0, n1, a, attrition):
    """[SUP-26][SUP-32] Exits derived from the identity, not restated."""
    if a == NB - 1:
        agein = G * n0[a - 1]
        return max(agein - (n1[a] - n0[a]), 0.0)
    return max(_gross_entries(n0, n1, a, attrition) - _net_flow(n0, n1, a), 0.0)


def _implied_hazard(n0, n1, a, attrition=None):
    """[SUP-35] The hazard implied by the identity-consistent gross exits.

    [SUP-56] For any band whose net flow never falls below -floor*n0 this
    returns exactly the floor in every year, so that series is constant by
    construction and carries no behavioural information.
    """
    att = band_floor(a) if attrition is None else attrition
    if n0[a] <= 0:
        return float(np.clip(att, *HCLIP))
    return float(np.clip(_gross_exits(n0, n1, a, att) / n0[a], *HCLIP))


def _raw_hazard_unclipped(n0, n1, a):
    """[SUP-58] The same ratio WITHOUT the zero floor or the HCLIP clip."""
    if n0[a] <= 0:
        return np.nan
    if a == NB - 1:
        agein = G * n0[a - 1]
        return float((agein - (n1[a] - n0[a])) / n0[a])
    return float(band_floor(a))


def series_is_degenerate(s, tol=1e-9):
    """[SUP-56] Is a series pinned to a single value, i.e. uninformative?"""
    s = pd.Series(s).dropna()
    return bool(len(s) > 1 and (s.max() - s.min()) < tol)


def h60_base_value(h_estimated=None):
    """[SUP-15][SUP-18] Resolve the base 60+ hazard from config."""
    if H60_BASE_SOURCE == "legal":
        return float(1.0 / (H60_TARGET_EXIT_AGE - 60.0))
    if H60_BASE_SOURCE == "estimated" and h_estimated is not None \
            and np.isfinite(h_estimated) and h_estimated > 0:
        return float(h_estimated)
    return float(H60_FIXED_BASE)


def implied_mean_exit_age(h):
    """Mean age at exit implied by a CONSTANT hazard in an open 60+ band."""
    return 60.0 + 1.0 / h if (h and h > 0) else np.nan


def tenure_mean_exit_age(shift=0.0, scale=1.0):
    """[SUP-51] Mean exit age implied by the tenure queue at a given shift."""
    surv, num, tot = 1.0, 0.0, 0.0
    n = LEGAL_GAP + 3 + int(max(0, np.ceil(shift))) + 6
    for t in range(n):
        hz = rb_h60_by_tenure(t, shift, scale)
        d = surv * hz
        num += d * (60 + t + 0.5); tot += d
        surv *= (1 - hz)
    num += surv * (60 + n + 0.5); tot += surv
    return num / tot if tot > 0 else np.nan


def tenure_curve_mean_exit_age():
    """[SUP-15] Mean exit age of the unshifted queue, for comparability."""
    return tenure_mean_exit_age(0.0, 1.0)


def tenure_shift_for_exit_age(target, lo=-6.0, hi=16.0, iters=80):
    """[SUP-51] Solve for the queue shift that yields a target mean exit age.

    tenure_mean_exit_age is monotone increasing in the shift, so plain bisection
    is valid. The default bracket spans roughly 61.3 to 72.1 years of mean exit
    age, which contains every scenario in H60_SCENARIOS and the flat base.
    """
    if not np.isfinite(target):
        return 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if tenure_mean_exit_age(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def steady_state_60plus(stock_5559, mode=None):
    """[SUP-46] Steady-state 60+ headcount under the LIVE specification."""
    m = H60_MODE if mode is None else mode
    inflow = G * stock_5559
    if m != "tenure":
        h = H60_SCENARIOS["base"]
        return (inflow / h) if h > 0 else np.nan
    q = np.zeros(SIXTY_SLOTS)
    for _ in range(500):
        newq = np.zeros(SIXTY_SLOTS)
        for k in range(SIXTY_SLOTS):
            newq[min(k + 1, SIXTY_SLOTS - 1)] += q[k] * (1 - rb_h60_by_tenure(k))
        newq[0] += inflow
        if np.max(np.abs(newq - q)) < 1e-9:
            q = newq
            break
        q = newq
    return float(q.sum())


def spec_label(mode=None):
    """[SUP-40] One place that turns H60_MODE into words for a chart legend."""
    m = H60_MODE if mode is None else mode
    if m == "tenure":
        return f"tenure queue, exit age ~{tenure_curve_mean_exit_age():.1f}"
    return (f"flat h60 = {H60_SCENARIOS['base']:.4f}, "
            f"exit age ~{implied_mean_exit_age(H60_SCENARIOS['base']):.1f}")


def band_fill_diagnostics(nat, years):
    """[SUP-18][SUP-58] Is the 60+ band in steady state?

    [SUP-58] This used to restate the flow arithmetic inline, bypassing the
    zero floor and the HCLIP clip. It now calls the shared helpers, and writes
    the unguarded ratio alongside so nothing is hidden by the fix.
    """
    ys = flow_years(years)
    rows = []
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        inflow = G * n0[NB - 2]
        exits = _gross_exits(n0, n1, NB - 1, band_floor(NB - 1))
        h = _implied_hazard(n0, n1, NB - 1)
        h_raw = _raw_hazard_unclipped(n0, n1, NB - 1)
        ss = inflow / h if (np.isfinite(h) and h > 0) else np.nan
        rows.append({"year": int(t1), "stock_60plus": float(n0[NB - 1]),
                     "ageing_inflow": float(inflow), "implied_exits": float(exits),
                     "h_observed": float(h),
                     "h_observed_unclipped": float(h_raw),
                     "guard_binding": bool(np.isfinite(h_raw)
                                           and abs(h_raw - h) > 1e-12),
                     "steady_state_at_h_obs": float(ss),
                     "fill_gap_pct": float(100 * (n0[NB - 1] / ss - 1))
                     if (np.isfinite(ss) and ss > 0) else np.nan})
    return pd.DataFrame(rows)


def channel_share(eh, window=None, target_year=None):
    """[SUP-19] Young-channel share on its own window (it trends; shape does not)."""
    window = CHANNEL_SHARE_WINDOW if window is None else window
    if eh is None or eh.empty or "young_share" not in eh.columns:
        return np.nan, "unavailable"
    s = eh["young_share"].dropna()
    if s.empty:
        return np.nan, "unavailable"
    if window == "full":
        return float(s.mean()), "full window"
    if window == "last5":
        return float(s.tail(5).mean()), "last 5 years"
    if window == "trend":
        yr = eh.loc[s.index, "year"].to_numpy(float)
        if len(yr) < 2 or np.var(yr) == 0:
            return float(s.tail(3).mean()), "last 3 years (trend not estimable)"
        b, a = np.polyfit(yr, s.to_numpy(float), 1)
        ty = float(target_year if target_year else FYEARS[0])
        return float(np.clip(a + b * ty, 0.0, 1.0)), f"OLS trend to {int(ty)}"
    return float(s.tail(3).mean()), "last 3 years"


def apply_channel_share(ep, share):
    """[SUP-19] Rescale a profile so the <30 bands carry `share`, keeping shape."""
    if not np.isfinite(share):
        return np.asarray(ep, float).copy()
    w = np.asarray(ep, float).copy()
    y_idx = list(YOUNG_BANDS)
    l_idx = [a for a in range(NB - 1) if a not in y_idx]
    ys, ls = w[y_idx].sum(), w[l_idx].sum()
    if ys <= 0 or ls <= 0:
        return w
    w[y_idx] *= share / ys
    w[l_idx] *= (1.0 - share) / ls
    s = w.sum()
    return w / s if s > 0 else np.asarray(ep, float)


# ============================================================================
# STAGE 1 -- SUPPLY PROJECTION (base cohort model)
# ============================================================================

def _national_from_df(df):
    df = df.copy()
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df.dropna(subset=["year_start"]); df["year_start"] = df["year_start"].astype(int)
    all_years = sorted(df.year_start.unique())
    nat_all = {}
    for y in all_years:
        sub = df[df.year_start == y]
        nat_all[y] = np.array([float(pd.to_numeric(sub[f"teachers_{b}"],
                               errors="coerce").sum()) for b in BANDS])
    return nat_all, all_years


def load_national():
    df = pd.read_excel(TEACHERS_FILE, sheet_name="historical", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    for b in BANDS:
        if f"teachers_{b}" not in df.columns:
            raise RuntimeError(f"Missing column teachers_{b} in the 'historical' sheet.")
    nat_all, all_years = _national_from_df(df)

    placeholder = []
    ref = nat_all.get(FIRST_REAL_YEAR)
    for y in all_years:
        if y < FIRST_REAL_YEAR and ref is not None and np.allclose(nat_all[y], ref):
            placeholder.append(y)
    real_years = [y for y in all_years if y >= FIRST_REAL_YEAR]
    nat = {y: nat_all[y] for y in real_years}

    print("[data] available years:", [int(y) for y in all_years])
    if placeholder:
        print(f"[data] placeholders dropped (copies of {FIRST_REAL_YEAR}): "
              f"{[int(y) for y in placeholder]}")
    print(f"[data] real years used: {[int(y) for y in real_years]}")
    f0, f1, gap = flow_window(real_years)
    if f0 is not None:
        print(f"[data] flows estimable {f0}-{f1}: {gap} years shorter than the stock")
        print(f"       series (FIRST_FLOW_YEAR drops one, the first transition another).")
    return nat, real_years


def gross_entry_vector(n0, n1):
    """[SUP-64] The gross entries of one transition, as a band vector."""
    e = np.zeros(NB)
    for a in range(NB - 1):
        e[a] = _gross_entries(n0, n1, a, band_floor(a))
    return e


def entries_alive_at(nat, years, y1, h):
    """[SUP-64] Where the historical entrants are sitting in year `y1`.

    Each transition's gross entry vector is injected into an otherwise empty
    panel and aged forward with one_year() under the SAME hazards the projection
    uses, until year y1. Summing those propagated cohorts gives, band by band,
    the number of past entrants still present in y1.

    This replaces the [SUP-62] estimator, which located entries by an integer
    band shift and therefore discarded every transition year for which
    (y1 - t) / 5 was not a whole number: seven of nine years on the real panel.

    Note that the ONLY assumption added here is that entrants face the same band
    hazards as incumbents. That is a behavioural assumption, not an accounting
    identity, and it is what makes the resulting figure an estimate rather than
    a bound.

    Returns (vector by band, per-cohort detail frame).
    """
    ys = flow_years(years)
    acc = np.zeros(NB)
    rows = []
    zero_profile = np.zeros(NB)
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1 or t1 > y1:
            continue
        cohort = gross_entry_vector(nat[t], nat[t1])
        injected = float(cohort.sum())
        for _ in range(y1 - t1):
            cohort, _ex, _rt = one_year(cohort, h, 0.0, zero_profile)
        acc += cohort
        rows.append({"entry_year": int(t1), "entries_injected": injected,
                     "years_aged": int(y1 - t1),
                     "survivors_at_target": float(cohort.sum()),
                     "survival_rate": float(cohort.sum() / injected)
                     if injected > 0 else np.nan})
    return acc, pd.DataFrame(rows)


def cohort_trace(nat, years, h):
    """[SUP-14][SUP-62][SUP-64] Follow each band forward, net of arriving entrants.

    The gross multiplier compares band j in y0 with band j+step in y1. Those are
    not the same people: the destination band also holds everyone who entered
    during the interval and aged into it. On the real panel the entry profile
    puts 8.5% of roughly 4,500 annual entries straight into 55-59, so a gross
    multiplier of 1.00 is perfectly compatible with material attrition.

    [SUP-64] Arriving entrants are now counted by FORWARD PROPAGATION through
    the model's own ageing arithmetic, not by an integer band shift. See
    entries_alive_at for why the previous version discarded most of the window.
    """
    y0, y1 = min(years), max(years)
    span = y1 - y0
    step = int(round(span * G))
    arriving, detail = entries_alive_at(nat, years, y1, h)
    rows = []
    for j in range(NB - step):
        d = j + step
        a, b = nat[y0][j], nat[y1][d]
        ent = float(arriving[d])
        net_end = b - ent
        rows.append({"band_start": NICE[j], "year_start": y0, "n_start": a,
                     "band_end": NICE[d], "year_end": y1, "n_end": b,
                     "entrants_still_present": ent,
                     "n_end_net_of_entrants": net_end,
                     "gross_multiplier": (b / a) if a > 0 else np.nan,
                     "net_multiplier": (net_end / a) if a > 0 else np.nan})
    return pd.DataFrame(rows), detail


def estimate_flows(nat, years, repair=None, window=None):
    """Estimate band hazards and the entry profile from observed net flows.

    Standing caveat: gross entries and gross exits are NOT identified from net
    changes in stock. Separating them needs person-level records.
    """
    repair = ENTRY_PROFILE_REPAIR if repair is None else repair
    window = ENTRY_PROFILE_WINDOW if window is None else window
    ys = flow_years(years)
    Hlist = {a: [] for a in range(NB)}
    Rband = {a: [] for a in range(NB)}
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        for a in range(NB):
            Hlist[a].append(_implied_hazard(n0, n1, a))
            Rband[a].append(0.0 if a == NB - 1
                            else _gross_entries(n0, n1, a, band_floor(a)))

    h = np.array([np.nanmean(Hlist[a]) if Hlist[a] else band_floor(a)
                  for a in range(NB)])

    def _slice(v):
        return v[-5:] if window == "last5" else v

    prof_raw = np.array([np.nanmean(_slice(Rband[a])) if Rband[a] else 0.0
                         for a in range(NB)])
    entrants = float(prof_raw.sum())
    ep_raw = prof_raw / prof_raw.sum() if prof_raw.sum() > 0 else np.eye(NB)[0]

    ep, fixed = repair_entry_profile(ep_raw) if repair else (ep_raw, [])
    return h, ep, entrants, ep_raw, fixed


def legacy_hazards(nat, years):
    """[SUP-35] The pre-v9 estimator, kept only so the size of the fix is visible."""
    ys = flow_years(years)
    Hlist = {a: [] for a in range(NB)}
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        for a in range(NB):
            net = _net_flow(n0, n1, a)
            if a == NB - 1:
                X = G * n0[a - 1] - (n1[a] - n0[a])
                Hlist[a].append(np.clip(X / n0[a] if n0[a] > 0 else 0.0, *HCLIP))
            elif a == NB - 2:
                Hlist[a].append(np.clip(max(-net, 0.0) / n0[a] if n0[a] > 0 else 0.0,
                                        *HCLIP))
            else:
                extra = max(-net, 0.0) / n0[a] if n0[a] > 0 else 0.0
                Hlist[a].append(np.clip(BASE_ATTRITION_YOUNGMID + extra, *HCLIP))
    return np.array([np.nanmean(Hlist[a]) if Hlist[a] else BASE_ATTRITION_YOUNGMID
                     for a in range(NB)])


def repair_entry_profile(ep_raw):
    """[SUP-14] Guard only; with the full window it should not fire."""
    ep = ep_raw.astype(float).copy()
    fixed = []
    for a in range(1, NB - 1):
        nb_mean = float(np.mean([ep_raw[a - 1], ep_raw[a + 1]]))
        if nb_mean > 0 and ep_raw[a] < ARTEFACT_RATIO * nb_mean:
            fixed.append({"band": NICE[a], "raw_share": float(ep_raw[a]),
                          "neighbour_mean": nb_mean})
            ep[a] = nb_mean
    s = ep.sum()
    return (ep / s if s > 0 else ep_raw), fixed


def entries_history(nat, years):
    """[SUP-8][SUP-16] Annual entries, split into graduate and lateral channels."""
    ys = flow_years(years)
    rows = []
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        tot = young = 0.0
        e_lt25 = 0.0
        for a in range(NB - 1):
            inflow = _gross_entries(n0, n1, a, band_floor(a))
            tot += inflow
            if a in YOUNG_BANDS:
                young += inflow
            if a == 0:
                e_lt25 = inflow
        rows.append({"year": int(t1),
                     "new_entrants": tot,
                     "new_entrants_lt25": e_lt25,
                     "new_entrants_young": young,
                     "new_entrants_lateral": tot - young,
                     "young_share": (young / tot) if tot > 0 else np.nan})
    return pd.DataFrame(rows)


def apply_h60_scenario(h, h60):
    hh = h.copy(); hh[NB - 1] = np.clip(h60, *HCLIP); return hh


def build_transition_matrix(h):
    P = np.zeros((NB + 1, NB + 1)); EXIT = NB
    for a in range(NB):
        if a < NB - 1:
            stay = max(1 - G - h[a], 0.0)
            P[a, a] = stay; P[a, a + 1] = G; P[a, EXIT] = max(1 - stay - G, 0.0)
        else:
            stay = max(1 - h[a], 0.0)
            P[a, a] = stay; P[a, EXIT] = max(1 - stay, 0.0)
    P[EXIT, EXIT] = 1.0
    P = np.clip(P, 0, None); P = P / P.sum(axis=1, keepdims=True)
    return P


def one_year(n, h, entries, entry_profile):
    new = np.zeros(NB); exits = 0.0
    for a in range(NB):
        if a < NB - 1:
            stay = max(1 - G - h[a], 0.0)
            new[a] += n[a] * stay; new[a + 1] += n[a] * G
            exits += n[a] * (1 - stay - G)
        else:
            stay = max(1 - h[a], 0.0)
            new[a] += n[a] * stay; exits += n[a] * (1 - stay)
    retire_55plus = h[NB - 2] * n[NB - 2] + h[NB - 1] * n[NB - 1]
    new = new + entries * entry_profile
    return np.clip(new, 0, None), exits, retire_55plus


def resolve_demand(demand):
    if demand is None:
        return None, None
    covered = [y for y in FYEARS if y in demand]
    if not covered:
        return None, "demand file covers none of the projection years"
    full = {}; last = demand[max(covered)]; warn = None
    for y in FYEARS:
        if y in demand:
            full[y] = demand[y]
        else:
            full[y] = last
            warn = f"demand did not cover {y}+; extrapolated last value ({last:,.0f})"
    return full, warn


def project(stock0, h, entry_profile, demand=None, entrants_fb=0.0,
            entry="endogenous", tenure=None, tenure_shift=0.0, tenure_scale=1.0):
    """Project the workforce forward.

    [SUP-24] `tenure` (default: H60_MODE) selects the 60+ specification.
    [SUP-51] `tenure_shift` / `tenure_scale` move the queue, so the tornado and
    the specification comparison can test the dominant parameter under BOTH
    specifications.

    Entry is UNCONSTRAINED by assumption; 04_gap owns the feasibility ceiling.
    """
    tenure = (H60_MODE == "tenure") if tenure is None else tenure
    n = stock0.astype(float).copy()
    q60 = rb_seed_sixtyplus(n[NB - 1], tenure_shift, tenure_scale) if tenure else None
    if tenure:
        n[NB - 1] = q60.sum()
    out = {"stock": {}, "retire": {}, "recruit": {}, "exit": {},
           "recruit_young": {}, "recruit_lateral": {}}
    young_w = float(sum(entry_profile[a] for a in YOUNG_BANDS))
    for y in FYEARS:
        # [SUP-66] the first step sizes the entry residual; its state output is
        # DISCARDED. Only the second step advances n and q60.
        if tenure:
            pre, _pq, exits, retire = rb_step_year(n, q60, h, 0.0, entry_profile,
                                                   tenure_shift, tenure_scale)
        else:
            pre, exits, retire = one_year(n, h, 0.0, entry_profile)
        after = pre.sum()
        if entry == "endogenous" and demand is not None and y in demand:
            E = max(demand[y] - after, 0.0)
        elif entry == "replacement":
            E = exits
        elif entry == "constant":
            E = entrants_fb
        else:
            E = 0.0
        if tenure:
            n, q60, exits, retire = rb_step_year(n, q60, h, E, entry_profile,
                                                 tenure_shift, tenure_scale)
        else:
            n = pre + E * entry_profile
        out["stock"][y] = n.copy(); out["retire"][y] = retire
        out["recruit"][y] = E; out["exit"][y] = exits
        out["recruit_young"][y] = E * young_w
        out["recruit_lateral"][y] = E * (1.0 - young_w)
    return out


def _read_demand_frame():
    if not DEMAND_FILE.exists():
        print("[demand] file not found:", DEMAND_FILE)
        return None, None, None, None
    d = pd.read_csv(DEMAND_FILE)
    col = [c for c in d.columns if "demand_selected" in c.lower()] or \
          [c for c in d.columns if "demand" in c.lower()]
    yc = [c for c in d.columns if "year" in c.lower()]
    sc = [c for c in d.columns if "scenario" in c.lower()]
    if not (col and yc):
        print("[demand] year/demand columns not recognised")
        return None, None, None, None
    dd = d.copy()
    if sc and "central" in set(dd[sc[0]].astype(str)):
        dd = dd[dd[sc[0]] == "central"]
    return dd, col[0], yc[0], d


def load_demand():
    dd, col, yc, _ = _read_demand_frame()
    if dd is None:
        return None
    g = dd.groupby(yc)[col].sum()
    print(f"[demand] loaded: {len(g)} years, {g.index.min()}-{g.index.max()}")
    return {int(k): float(v) for k, v in g.items()}


def load_demand_regional(scope):
    """[SUP-2] Read demand by region straight from the block-02 contract."""
    if scope == "national":
        return None
    dd, col, yc, _ = _read_demand_frame()
    if dd is None:
        return None
    key = "nuts2_code" if scope == "nuts2" else "nuts3_code"
    if key not in dd.columns:
        if scope == "nuts2" and "nuts3_code" in dd.columns:
            dd = dd.copy()
            dd["nuts2_code"] = dd["nuts3_code"].astype(str).str[:2]
            key = "nuts2_code"
        else:
            print(f"[demand] {key} not in the 02 contract -> fixed-share fallback")
            return None
    g = dd.groupby([key, yc])[col].sum()
    out = {}
    for (k, y), v in g.items():
        out.setdefault(str(k), {})[int(y)] = float(v)
    print(f"[demand] regional demand read for {len(out)} {scope} regions")
    return out


# --------------------------------------------------------- universes
def load_by_scope(scope="national"):
    df = pd.read_excel(TEACHERS_FILE, sheet_name="historical", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df.dropna(subset=["year_start"]); df["year_start"] = df["year_start"].astype(int)
    df = df[df.year_start >= FIRST_REAL_YEAR]

    if scope == "national":
        groups = {"PT": df}
    else:
        keycol = "nuts2_code" if scope == "nuts2" else "nuts3_code"
        if keycol not in df.columns:
            print(f"[universe] column {keycol} missing; falling back to national")
            groups = {"PT": df}
        else:
            groups = {str(k): g for k, g in df.groupby(keycol)}

    universes = {}
    for key, g in groups.items():
        years = sorted(g.year_start.unique())
        nat = {}
        for y in years:
            sub = g[g.year_start == y]
            nat[y] = np.array([float(pd.to_numeric(sub[f"teachers_{b}"],
                               errors="coerce").sum()) for b in BANDS])
        universes[key] = (nat, years)
    return universes


def rake_to_national(dfu, national,
                     cols=("recruitment", "retirements_wave", "stock"),
                     proportional=(("recruitment",
                                    ("recruitment_young", "recruitment_lateral")),)):
    """[SUP-1][SUP-13] Scale regional series so they reproduce national totals."""
    out = dfu.copy()
    rows = []
    prop_map = {src: tgts for src, tgts in proportional}
    for y, g in dfu.groupby("year"):
        rec = {"year": int(y)}
        factors = {}
        for c in cols:
            if c not in dfu.columns or y not in national.index:
                continue
            reg_sum = float(g[c].sum())
            nat_val = float(national.loc[y, c]) if c in national.columns else np.nan
            f = (nat_val / reg_sum) if (reg_sum > 0 and np.isfinite(nat_val)) else 1.0
            factors[c] = f
            out.loc[out.year == y, c] = out.loc[out.year == y, c] * f
            rec[f"{c}_regional_raw"] = reg_sum
            rec[f"{c}_national"] = nat_val
            rec[f"{c}_rake_factor"] = f
            rec[f"{c}_gap_pct"] = 100 * (reg_sum / nat_val - 1) if nat_val else np.nan
        for src, tgts in prop_map.items():
            f = factors.get(src)
            if f is None:
                continue
            for t in tgts:
                if t in out.columns:
                    out.loc[out.year == y, t] = out.loc[out.year == y, t] * f
                    rec[f"{t}_rake_factor"] = f
        rows.append(rec)

    if {"recruitment", "recruitment_young", "recruitment_lateral"} <= set(out.columns):
        ident = (out["recruitment_young"] + out["recruitment_lateral"]
                 - out["recruitment"]).abs().max()
        out.attrs["channel_identity_max_abs_error"] = float(ident)
        if ident > 1e-6:
            print(f"[rake] WARNING: channel identity broken by {ident:,.4f}")
    return out, pd.DataFrame(rows)


def run_universe(scope, demand=None, national_df=None):
    universes = load_by_scope(scope)
    keys = list(universes)
    print(f"[universe={scope}] {len(universes)} region(s): "
          f"{keys[:8]}{'...' if len(keys) > 8 else ''}")

    reg_demand_all = load_demand_regional(scope)
    total_last = sum(v[0][max(v[1])].sum() for v in universes.values())
    used_fallback = False

    rows = []
    for key, (nat, years) in universes.items():
        if len(years) < 2:
            continue
        h, ep_shape, entrants, _, _ = estimate_flows(nat, years)
        reg_eh = entries_history(nat, years)
        reg_share, _ = channel_share(reg_eh)
        ep = apply_channel_share(ep_shape, reg_share)
        h = apply_h60_scenario(h, H60_SCENARIOS["base"])
        s0 = nat[max(years)]

        reg_demand = None
        if reg_demand_all is not None and key in reg_demand_all:
            reg_demand, _ = resolve_demand(reg_demand_all[key])
        elif demand is not None and total_last > 0:
            share = s0.sum() / total_last
            reg_demand = {y: demand[y] * share for y in demand}
            used_fallback = True

        reg_entry = "endogenous" if reg_demand is not None else "replacement"
        out = project(s0, h, ep, reg_demand, entrants, entry=reg_entry)
        young_w = float(sum(ep[a] for a in YOUNG_BANDS))
        for y in FYEARS:
            rows.append({"scope": scope, "region": key, "year": y,
                         "stock": out["stock"][y].sum(),
                         "recruitment": out["recruit"][y],
                         "retirements_wave": out["retire"][y],
                         "recruitment_young": out["recruit_young"][y],
                         "recruitment_lateral": out["recruit_lateral"][y],
                         "young_share_region": young_w})

    if used_fallback and scope != "national":
        print(f"[universe={scope}] WARNING: fixed-share fallback used for at least one "
              f"region; those regional paths are not independent of the national one.")

    dfu = pd.DataFrame(rows)
    rake = pd.DataFrame()
    share_stats = None
    if national_df is not None and not dfu.empty and scope != "national":
        nat_idx = national_df.set_index("year")
        dfu, rake = rake_to_national(dfu, nat_idx)
        if not rake.empty:
            g = rake.get("recruitment_gap_pct")
            r = rake.get("retirements_wave_gap_pct")
            if g is not None:
                print(f"[universe={scope}] pre-raking gap vs national: "
                      f"recruitment {g.mean():+.2f}%, retirements {r.mean():+.2f}% "
                      f"(mean over years); raked to match.")
        if "recruitment_young" in national_df.columns:
            reg_y = dfu.groupby("year")["recruitment_young"].sum()
            nat_y = national_df.set_index("year")["recruitment_young"]
            resid = 100 * (reg_y / nat_y - 1)
            print(f"[universe={scope}] channel identity young+lateral=recruitment holds "
                  f"exactly; regional young sums to {resid.mean():+.2f}% of the national "
                  f"young channel.")
            share = dfu.groupby("region")["young_share_region"].first()
            if len(share) > 1:
                ratio = float(share.max() / share.min()) if share.min() > 0 else np.inf
                # [SUP-67] report designations; the code is kept alongside so the
                # printed diagnostic can still be joined back to the panel.
                lo_lab = region_label(share.idxmin(), scope)
                hi_lab = region_label(share.idxmax(), scope)
                share_stats = {"scope": scope, "min": float(share.min()),
                               "min_region": lo_lab,
                               "min_region_code": str(share.idxmin()),
                               "max": float(share.max()),
                               "max_region": hi_lab,
                               "max_region_code": str(share.idxmax()),
                               "ratio": ratio}
                print(f"[universe={scope}] young-channel share by region: "
                      f"{100*share.min():.1f}% ({lo_lab}) to "
                      f"{100*share.max():.1f}% ({hi_lab}) "
                      f"= {ratio:.1f}x -- a graduate ceiling in 04_gap bites very "
                      f"unevenly.")
                if share.min() < 0.05:
                    print(f"            NOTE: {lo_lab} models "
                          f"{100*(1-share.min()):.0f}% of its entries as lateral, so a "
                          f"graduate ceiling would barely touch it. Worth confirming "
                          f"before\n            04_gap relies on that split.")
        if not rake.empty and "recruitment_rake_factor" in rake.columns:
            f = rake["recruitment_rake_factor"]
            print(f"[universe={scope}] raking discards {100*(1-f.mean()):.1f}% of regional "
                  f"recruitment on average (min factor {f.min():.4f}). That is regional "
                  f"heterogeneity\n            the national model does not carry -- "
                  f"zero-flooring plus {dfu.region.nunique()} sets of region-specific "
                  f"hazards.")
    return dfu, rake, share_stats


# --------------------------------------------------------- experiments
def experiment_entry_policies(nat, years, demand, entry_profile=None):
    """[SUP-22] Accepts the profile the base projection used."""
    h, ep_shape, entrants, _, _ = estimate_flows(nat, years)
    ep = entry_profile if entry_profile is not None else ep_shape
    h = apply_h60_scenario(h, H60_SCENARIOS["base"])
    s0 = nat[max(years)]
    policies = {
        "endogenous (follows demand)": ("endogenous", demand),
        "replacement (replaces exits)": ("replacement", None),
        "constant (fixed recruitment)": ("constant", None),
        "zero (ageing only)": ("zero", None),
    }
    results = {}
    for label, (mode, dem) in policies.items():
        out = project(s0, h, ep, dem, entrants, entry=mode)
        results[label] = {
            "df": pd.DataFrame({"year": FYEARS,
                                "stock": [out["stock"][y].sum() for y in FYEARS],
                                "recruitment": [out["recruit"][y] for y in FYEARS]}),
            "final_bands": out["stock"][FYEARS[-1]],
            "cum_recruit": sum(out["recruit"].values()),
        }
    return results, h, ep


def cum_recruit(h, ep, dem, s0, entry, efb, shift=0.0, scale=1.0):
    return sum(project(s0, h, ep, dem, efb, entry,
                       tenure_shift=shift, tenure_scale=scale)["recruit"].values())


def sensitivity(h, ep, dem, s0, entry, efb, eh=None, eh_h5559=None):
    """[SUP-51][SUP-53] Sensitivity that works under BOTH 60+ specifications.

    Under the tenure queue an h60 scenario writes h[8], which the queue never
    reads, so every h60 bar came out at exactly zero. The h60 scenarios are
    therefore expressed as TARGET EXIT AGES and translated into a queue shift
    when the queue is live, which makes the two modes directly comparable.

    [SUP-61] `global G` is declared at the top of the function. Python requires
    the declaration to precede every reference to the name in the same scope, so
    keeping it buried next to the ageing-rate loop made the function one stray
    `G` away from a SyntaxError at import time.
    """
    global G

    tenure_live = (H60_MODE == "tenure")
    base = cum_recruit(h, ep, dem, s0, entry, efb); rows = []

    def run(hh=None, epp=None, dd=None, shift=0.0, scale=1.0, label=""):
        val = cum_recruit(hh if hh is not None else h, epp if epp is not None else ep,
                          dd if dd is not None else dem, s0, entry, efb, shift, scale)
        rows.append({"scenario": label, "cum_recruit": val,
                     "delta_vs_base": val - base,
                     "pct": 100 * (val - base) / base if base else np.nan})

    # ---- 60+ scenarios, expressed as exit ages so both modes can run them
    for tag, h60 in H60_SCENARIOS.items():
        if tag == "base" or not np.isfinite(h60):
            continue
        age = implied_mean_exit_age(h60)
        if tenure_live:
            sh = tenure_shift_for_exit_age(age)
            run(shift=sh,
                label=f"60+ exit age ~{age:.1f} ({tag}, queue shift {sh:+.2f}y)")
        else:
            run(hh=apply_h60_scenario(h, h60),
                label=f"h60 = {h60:.4f} ({tag}, exit age ~{age:.1f})")

    for f, tag in [(1.5, "attrition 30-44 +50%"), (0.5, "attrition 30-44 -50%")]:
        hh = h.copy()
        for a in range(2, 5):
            hh[a] = np.clip(hh[a] * f, *HCLIP)
        run(hh=hh, label=tag)

    # [SUP-53] the observed 55-59 value equals the floor by construction after
    # [SUP-36], so testing it reproduces the base exactly. It is dropped here
    # and the reason is printed, instead of drawing a zero-length bar.
    tests = [(0.02, "55-59 hazard = 0.02 (early retirement returns)"),
             (0.05, "55-59 hazard = 0.05")]
    if (eh_h5559 is not None and np.isfinite(eh_h5559)
            and abs(eh_h5559 - band_floor(NB - 2)) > 1e-9):
        tests.insert(0, (eh_h5559,
                         f"55-59 hazard = {eh_h5559:.4f} (last 3 observed years)"))
    for v, tag in tests:
        hh = h.copy(); hh[NB - 2] = np.clip(v, *HCLIP)
        run(hh=hh, label=tag)

    for wnd in ("full", "last5", "trend"):
        sh_v, lab = channel_share(eh, window=wnd)
        if np.isfinite(sh_v):
            run(epp=apply_channel_share(ep, sh_v),
                label=f"channel share = {100*sh_v:.1f}% ({lab})")

    # [SUP-11] the near-zero result is a finding: with a 16-year window only
    # entrants already aged 44+ reach the 60+ band before 2040.
    def _shift_channel(ep, to_young):
        w = ep.copy()
        y_idx = list(YOUNG_BANDS); l_idx = [a for a in range(NB - 1) if a not in y_idx]
        move = 0.5 * (w[l_idx].sum() if to_young else w[y_idx].sum())
        if to_young:
            w[l_idx] *= 0.5
            w[y_idx] += move * (w[y_idx] / w[y_idx].sum() if w[y_idx].sum() > 0 else 1)
        else:
            w[y_idx] *= 0.5
            w[l_idx] += move * (w[l_idx] / w[l_idx].sum() if w[l_idx].sum() > 0 else 1)
        return w / w.sum() if w.sum() > 0 else ep

    run(epp=_shift_channel(ep, to_young=False), label="entry shifted to lateral (30+)")
    run(epp=_shift_channel(ep, to_young=True), label="entry shifted to young (<30)")

    for f, tag in [(1.1, "ageing rate +10%"), (0.9, "ageing rate -10%")]:
        g_old = G
        try:
            G = g_old * f
            run(label=tag)
        finally:
            G = g_old

    if dem is not None:
        for f, tag in [(1.05, "demand +5%"), (0.95, "demand -5%"),
                       (1.13, "demand +13% (01_students high)"),
                       (0.87, "demand -13% (01_students low)")]:
            run(dd={k: v * f for k, v in dem.items()}, label=tag)

    tor = pd.DataFrame(rows)
    # [SUP-53] separate disconnected levers from genuine near-zero results
    tor["degenerate"] = tor["delta_vs_base"].abs() < DEGENERATE_TOL
    return base, tor


# --------------------------------------------------------- feasibility
def feasibility_report(w, eh, young_w=None, share_label=""):
    """[SUP-8][SUP-20][SUP-44] Confront the requirement with realised history."""
    if eh.empty:
        return None
    e = eh["new_entrants"]
    hist_mean = float(e.mean())
    hist_median = float(e.median())
    hist_last5 = float(e.tail(5).mean())
    hist_max = float(e.max())
    hist_max_year = int(eh.loc[e.idxmax(), "year"])
    srt = e.sort_values(ascending=False).to_numpy()
    hist_second = float(srt[1]) if len(srt) > 1 else np.nan
    hist_mean_ex_max = float(e.drop(index=e.idxmax()).mean()) if len(e) > 1 else np.nan
    req_mean = float(w["recruitment"].mean())
    req_first10 = float(w.head(10)["recruitment"].mean())
    y_share = float(young_w) if (young_w is not None and np.isfinite(young_w)) \
        else (float(eh["young_share"].tail(5).mean()) if "young_share" in eh else np.nan)
    above_max = int((w["recruitment"] > hist_max).sum())
    above_second = int((w["recruitment"] > hist_second).sum()) \
        if np.isfinite(hist_second) else 0

    print("\n" + "=" * 74)
    print("ENTRY FEASIBILITY -- the assumption this block makes, and its size")
    print("=" * 74)
    print("  This block assumes recruitment is UNCONSTRAINED: whatever the demand")
    print("  path requires is hired. No qualifying-pool ceiling is applied here.")
    print("  The constraint belongs in 04_gap; what follows is the size of the ask.")
    print(f"    realised entries, mean              : {hist_mean:8,.0f}/yr")
    print(f"    realised entries, median            : {hist_median:8,.0f}/yr")
    print(f"    realised entries, last 5 years      : {hist_last5:8,.0f}/yr")
    print(f"    realised entries, best year         : {hist_max:8,.0f}  ({hist_max_year})")
    print(f"    realised entries, second best       : {hist_second:8,.0f}")
    print(f"    required, 2025-2040 mean            : {req_mean:8,.0f}/yr")
    print(f"    required, first 10 years mean       : {req_first10:8,.0f}/yr")
    if hist_mean > 0:
        print(f"    required vs mean                    : {100*(req_mean/hist_mean-1):+7.1f}%")
    if hist_median > 0:
        print(f"    required vs median                  : "
              f"{100*(req_mean/hist_median-1):+7.1f}%")
    if np.isfinite(hist_second) and hist_second > 0:
        print(f"    required vs second-best year        : "
              f"{100*(req_mean/hist_second-1):+7.1f}%")
    print(f"    years above the best year           : {above_max:2d} of {len(w)}")
    print(f"    years above the second-best year    : {above_second:2d} of {len(w)}")
    # the two facts below can look contradictory and are not: the mean can sit
    # under the second-best year while half the individual years sit above it.
    if np.isfinite(hist_second) and req_mean < hist_second and above_second > len(w) / 3:
        print(f"    NOTE: the MEAN sits below the second-best year while {above_second}")
        print("    individual years sit above it. The second reading is the operative")
        print("    one: the system would have to beat that year in most of the horizon.")
    if np.isfinite(hist_second) and hist_max > 1.05 * hist_second:
        print(f"    NOTE: {hist_max_year} is an outlier ({hist_max:,.0f} against a "
              f"second-best of {hist_second:,.0f}).")
        print(f"    Excluding it the historical mean is {hist_mean_ex_max:,.0f} and the")
        print(f"    requirement is {100*(req_mean/hist_mean_ex_max-1):+.1f}% above it.")
    if hist_max > 0 and req_mean > hist_max:
        print("    The required AVERAGE exceeds the best year ever recorded. This stops")
        print("    being a projection and becomes a claim about feasibility.")
    if np.isfinite(y_share):
        yr_req = req_mean * y_share
        obs = eh["young_share"]
        print(f"    share of entries via the <30 channel : {100*y_share:6.1f}% "
              f"({share_label or 'as used by the projection'})")
        print(f"      observed: full {100*obs.mean():.1f}% | last5 "
              f"{100*obs.tail(5).mean():.1f}% | last3 {100*obs.tail(3).mean():.1f}% "
              f"| {int(eh['year'].iloc[-1])} {100*obs.iloc[-1]:.1f}%")
        print(f"      -> graduate channel  : {yr_req:8,.0f}/yr")
        print(f"      -> re-entry/lateral  : {req_mean - yr_req:8,.0f}/yr")
        print("    The lateral channel is the larger of the two and is NOT governed")
        print("    by the initial-teacher-education pool. It is drawn from the")
        print("    contracted reserve, which shrinks as the workforce shrinks.")
    return {"hist_mean": hist_mean, "hist_median": hist_median,
            "hist_last5": hist_last5, "hist_max": hist_max,
            "hist_max_year": hist_max_year, "hist_second_best": hist_second,
            "hist_mean_excluding_max": hist_mean_ex_max,
            "req_mean": req_mean, "req_first10": req_first10,
            "years_above_best_ever": above_max,
            "years_above_second_best": above_second,
            "young_share_used": y_share, "young_share_window": share_label,
            "graduate_channel_per_year": float(req_mean * y_share)
            if np.isfinite(y_share) else np.nan,
            "lateral_channel_per_year": float(req_mean * (1 - y_share))
            if np.isfinite(y_share) else np.nan}


# --------------------------------------------------------- figures (stage 1)
def style(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def fig_flow(h, tenure=None):
    """[SUP-27][SUP-34][SUP-54] Flow diagram that matches the model."""
    tenure = (H60_MODE == "tenure") if tenure is None else tenure
    P = build_transition_matrix(h)
    ageup = np.array([P[i, i + 1] if i < NB - 1 else 0.0 for i in range(NB)])
    exit_ = np.array([P[i, NB] for i in range(NB)])
    stay_ = np.array([P[i, i] for i in range(NB)])
    at_floor = [abs(h[i] - band_floor(i)) < 1e-9 for i in range(NB - 1)] + [False]

    hz_q = [rb_h60_by_tenure(k) for k in range(SIXTY_SLOTS)] if tenure else None

    def band_colour(i):
        if i == NB - 1:
            return EXITC
        return MIDC if at_floor[i] else EXITC

    def fill(i):
        return plt.cm.YlOrRd(0.18 + 0.62 * i / (NB - 1))

    fig, ax = plt.subplots(figsize=(18, 9.2))
    xs = np.arange(NB) * 1.66; y0 = 0.0; R = 0.46; LANE = -1.9
    ex_x, ex_y = xs[NB // 2], -3.6
    ax.set_xlim(-2.9, xs[-1] + 1.4); ax.set_ylim(-4.8, 3.4); ax.axis("off")
    C = {i: (xs[i], y0) for i in range(NB)}

    ax.add_patch(Circle((ex_x, ex_y), R * 1.28, facecolor="#4b5563",
                        edgecolor=INK, lw=2.4, zorder=6))
    ax.text(ex_x, ex_y, "EXIT\nretirement /\nattrition", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=7)
    ax.add_patch(FancyArrowPatch((xs[0] - 1.6, 1.9), (xs[0] - 0.02, y0 + R * 0.55),
                 arrowstyle="-|>", mutation_scale=24, lw=3.0, color=ENTRY, zorder=4,
                 connectionstyle="arc3,rad=-0.28"))
    ax.text(xs[0] - 1.66, 2.32, "ENTRY", ha="center", fontsize=12,
            fontweight="bold", color=ENTRY)
    ax.text(xs[0] - 1.66, 1.94, "unconstrained\n(04_gap restricts)", ha="center",
            fontsize=8.5, color=ENTRY)

    for i, lab in enumerate(NICE):
        x, _ = C[i]
        ax.add_patch(Circle((x, y0), R, facecolor=fill(i), edgecolor=INK, lw=2.1, zorder=5))
        ax.text(x, y0, lab, ha="center", va="center", fontsize=10, fontweight="bold",
                color=("#1a1a1a" if i < 6 else "white"), zorder=6)
        ax.add_patch(FancyArrowPatch((x - 0.20, y0 + R * 0.92), (x + 0.20, y0 + R * 0.92),
                     arrowstyle="-|>", mutation_scale=10, lw=1.5, color=STAYC,
                     zorder=4, connectionstyle="arc3,rad=1.7"))
        lbl = "queue" if (tenure and i == NB - 1) else f"{stay_[i]:.2f}"
        ax.text(x, y0 + R + 0.62, lbl, ha="center", fontsize=7.6, color="#6b7480")

    for i in range(NB - 1):
        x1, _ = C[i]; x2, _ = C[i + 1]
        ax.add_patch(FancyArrowPatch((x1 + R, y0), (x2 - R, y0), arrowstyle="-|>",
                     mutation_scale=17, lw=2.1, color=AGEING, zorder=3))
        ax.text((x1 + x2) / 2, y0 + 0.30, f"{ageup[i]:.2f}", ha="center",
                fontsize=8.5, color=AGEING)

    order = np.argsort(-exit_)
    for i in order:
        x, _ = C[i]
        lw = 0.7 + 8.5 * exit_[i]
        col = band_colour(i)
        drop_y = LANE - 0.05 * abs(x - ex_x)
        ax.add_patch(FancyArrowPatch((x, y0 - R), (x, drop_y), arrowstyle="-",
                     mutation_scale=1, lw=lw, color=col, zorder=2, alpha=0.9))
        ax.add_patch(FancyArrowPatch((x, drop_y), (ex_x, ex_y + R * 1.28),
                     arrowstyle="-|>", mutation_scale=15, lw=lw, color=col, zorder=2,
                     alpha=0.9))
        if tenure and i == NB - 1:
            txt = f"{min(hz_q):.2f}-{max(hz_q):.2f}"
        else:
            note = "" if i == NB - 1 else ("" if at_floor[i] else "*")
            txt = f"{exit_[i]:.4f}{note}"
        ax.text(x, y0 - R - 0.30, txt, ha="center", fontsize=8.0,
                color=col, fontweight=("bold" if exit_[i] >= 0.10 else "normal"),
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
                zorder=5)

    if at_floor[NB - 2]:
        ax.text(xs[NB - 2], y0 - R - 0.78,
                "55-59 sits at the ATTRITION floor, so it is drawn amber:\n"
                "the model contains no early retirement before 60",
                ha="center", fontsize=7.6, color=MIDC, style="italic")

    if tenure:
        tail = (f"60+ is a {SIXTY_SLOTS}-slot tenure queue, hazard "
                f"{min(hz_q):.2f} to {max(hz_q):.2f} by year in band; "
                f"implied mean exit age {tenure_curve_mean_exit_age():.1f}")
    else:
        tail = (f"dominant hazard: {exit_[-1]*100:.2f}% of the 60+ cohort exits per "
                f"year (constant); implied mean exit age "
                f"{implied_mean_exit_age(exit_[-1]):.1f}")
    ax.text(ex_x, ex_y - R * 1.28 - 0.42, f"H60_MODE = {H60_MODE} -- {tail}",
            fontsize=9.5, color=EXITC, fontweight="bold", ha="center")

    ax.set_title("Teacher supply model -- cohort chain with an absorbing exit",
                 fontsize=15, fontweight="bold", pad=16, color=INK)
    leg = [Line2D([0], [0], color=AGEING, lw=2.2, label="ageing (1/5 per year, structural)"),
           Line2D([0], [0], color=STAYC, lw=1.6, label="stays in band (largest flow)"),
           Line2D([0], [0], color=EXITC, lw=3.2,
                  label="exit: retirement (hazard above the floor)"),
           Line2D([0], [0], color=MIDC, lw=2.0,
                  label=f"exit: attrition (at the {BASE_ATTRITION_YOUNGMID:.3f} floor; "
                        f"* = estimated above it)"),
           Line2D([0], [0], color=ENTRY, lw=3.0, label="entry (unconstrained here)")]
    ax.legend(handles=leg, loc="upper right", fontsize=9.5, frameon=False, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(IMG / "supply_flow_diagram.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_policies(results):
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    palette = {"endogenous (follows demand)": AGEING,
               "replacement (replaces exits)": ENTRY,
               "constant (fixed recruitment)": MIDC,
               "zero (ageing only)": EXITC}
    for label, r in results.items():
        df = r["df"]
        ax.plot(df.year, df.stock, "-o", ms=4, lw=2.2,
                color=palette.get(label, "0.4"), label=label)
        ax.annotate(f"{df.stock.iloc[-1]:,.0f}", (df.year.iloc[-1], df.stock.iloc[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=8,
                    color=palette.get(label, "0.4"), va="center")
    style(ax)
    ax.set_title(f"Stock 2025-2040 under different entry policies "
                 f"({spec_label()})", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers (national stock)")
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout()
    fig.savefig(IMG / "entry_policies_comparison.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_pyramid(nat, years, results, base_policy="endogenous (follows demand)"):
    """[SUP-28][SUP-46][SUP-55] Base policy, live specification, right direction."""
    s0 = nat[max(years)]
    sf = results[base_policy]["final_bands"]
    sr = results.get("replacement (replaces exits)", {}).get("final_bands")
    y = np.arange(NB)
    fig, ax = plt.subplots(figsize=(10, 6.8))
    hgt = 0.26
    ax.barh(y - hgt, s0, height=hgt, color=AGEING, alpha=0.85,
            label=f"{max(years)} (observed)")
    ax.barh(y, sf, height=hgt, color=ENTRY, alpha=0.85,
            label=f"2040 -- {base_policy} (BASE)")
    if sr is not None and base_policy != "replacement (replaces exits)":
        ax.barh(y + hgt, sr, height=hgt, color=MIDC, alpha=0.75,
                label="2040 -- replacement (comparison)")
    ax.set_yticks(y); ax.set_yticklabels(NICE); ax.invert_yaxis()
    style(ax)

    sh_b = 100 * sf[-1] / sf.sum()
    sub = f"60+ reaches {sh_b:.1f}% of the workforce by 2040 under the base policy"
    if sr is not None and base_policy != "replacement (replaces exits)":
        sub += f" ({100*sr[-1]/sr.sum():.1f}% under replacement)"

    ss_2040 = steady_state_60plus(sf[NB - 2])
    if np.isfinite(ss_2040) and ss_2040 > 0:
        lvl = float(sf[NB - 1])
        gap = 100 * (lvl / ss_2040 - 1)
        direction = ("above" if gap > 1 else "below" if gap < -1 else "at")
        sub += (f"\n2040 level {lvl:,.0f} vs steady state {ss_2040:,.0f} implied by the "
                f"2040 feeder: {abs(gap):.0f}% {direction} it")
        if gap > 1:
            sub += " (still shedding a legacy bulge, not converging from below)"

    ax.set_title(f"Age structure: today vs. 2040 ({spec_label()})\n{sub}",
                 fontweight="bold", fontsize=10.5)
    ax.set_xlabel("teachers (national stock)"); ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "age_pyramid_today_vs_2040.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_wave(w):
    """[SUP-40] Name the specification in the legend, not just the title."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(w.year, w.retirements_wave, "o-", color=EXITC, lw=2.2, ms=5,
            label=f"retirement wave, 55-59 + 60+ ({spec_label()})")
    ax.plot(w.year, w.recruitment, "s--", color=ENTRY, lw=2.2, ms=5,
            label="required recruitment, stage 1 (unconstrained)")
    ax.fill_between(w.year, w.retirements_wave, w.recruitment,
                    where=(w.recruitment >= w.retirements_wave), color=ENTRY, alpha=0.08)
    style(ax)
    ax.set_title("Retirement wave vs. required recruitment, 2025-2040 -- STAGE 1",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers per year (national)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "wave_vs_recruitment.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_channel(w, hist_max=None, hist_max_year=None, hist_second=None):
    """[SUP-44] Show the second-best year too; the maximum is a pandemic year."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.stackplot(w.year, w.recruitment_young, w.recruitment_lateral,
                 labels=["new-graduate channel (<30)",
                         "re-entry / lateral channel (30+)"],
                 colors=[ENTRY, MIDC], alpha=0.85)
    ax.plot(w.year, w.recruitment, color=INK, lw=1.6, label="total recruitment")
    if hist_max:
        lbl = f"best year on record ({hist_max:,.0f}"
        lbl += f", {hist_max_year})" if hist_max_year else ")"
        ax.axhline(hist_max, color=EXITC, lw=1.4, ls="--", label=lbl)
    if hist_second and np.isfinite(hist_second):
        ax.axhline(hist_second, color=EXITC, lw=1.2, ls=":", alpha=0.8,
                   label=f"second-best year ({hist_second:,.0f})")
    style(ax)
    ax.set_title("Required recruitment by entry channel -- only the lower band is\n"
                 "governed by the initial-teacher-education pool", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers per year")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMG / "recruitment_by_channel.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_tornado(t):
    """[SUP-53] Drop disconnected levers rather than drawing zero-length bars."""
    live = t[~t["degenerate"]].sort_values("delta_vs_base")
    dead = t[t["degenerate"]]
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.barh(live.scenario, live.delta_vs_base,
            color=[EXITC if d > 0 else "#2980B9" for d in live.delta_vs_base],
            edgecolor="black", lw=0.4)
    ax.axvline(0, color="k", lw=0.8)
    style(ax)
    span = max(abs(live.delta_vs_base)) if len(live) else 1.0
    for yv, (d, p) in enumerate(zip(live.delta_vs_base, live.pct)):
        ax.text(d + (1 if d >= 0 else -1) * span * 0.02, yv, f"{p:+.1f}%",
                va="center", ha="left" if d >= 0 else "right", fontsize=8.5)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_title(f"Sensitivity of cumulative recruitment 2025-2040 "
                 f"({spec_label()})", fontweight="bold")
    ax.set_xlabel("deviation in cumulative recruitment vs base (teachers)")
    if len(dead):
        fig.text(0.5, 0.005,
                 f"{len(dead)} scenario(s) returned exactly the base and are omitted: "
                 f"{', '.join(dead.scenario.head(3))}"
                 f"{'...' if len(dead) > 3 else ''}. "
                 "A zero delta means the lever is not connected to this "
                 "specification, not that the model is insensitive.",
                 ha="center", fontsize=8, color="#555")
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(IMG / "tornado_sensitivity.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_universe(dfu, scope):
    """[SUP-38][SUP-39] Correct title, and no degenerate ranking panel."""
    if dfu.empty:
        return
    agg = dfu.groupby("year")[["stock", "recruitment"]].sum().reset_index()
    n_reg = dfu.region.nunique()
    single = (scope == "national") or (n_reg < 2)

    if single:
        fig, a1 = plt.subplots(figsize=(8.5, 5.4))
    else:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))

    a1.plot(agg.year, agg.stock, "-o", color=AGEING, lw=2.2, ms=4)
    ttl = "unraked" if single else "raked to national"
    a1.set_title(f"Aggregate stock ({scope}, {ttl})", fontweight="bold")
    a1.set_xlabel("year"); a1.set_ylabel("teachers")
    style(a1)

    if not single:
        top = (dfu.groupby("region")["recruitment"].sum()
               .sort_values(ascending=False).head(10))
        # [SUP-67] label by NUTS 2024 designation, not by code
        labels = [region_label(c, scope) for c in top.index]
        unmatched = [c for c, lab in zip(top.index, labels) if lab == str(c)]
        if unmatched:
            print(f"[universe={scope}] WARNING: {len(unmatched)} region code(s) not in "
                  f"the NUTS 2024 table, shown as codes: {unmatched}")
        a2.barh(labels[::-1], top.values[::-1], color=ENTRY, edgecolor="black", lw=0.4)
        a2.set_title(f"Cumulative recruitment by region "
                     f"(top {min(10, n_reg)} of {n_reg}, {scope})", fontweight="bold")
        a2.set_xlabel("teachers 2025-2040")
        style(a2)

    fig.tight_layout()
    fig.savefig(IMG / f"universe_{scope}.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def stage_supply():
    """STAGE 1 -- base supply projection, entry policies, sensitivity, universes."""
    print("=" * 74)
    print("STAGE 1 -- supply projection (base cohort model)")
    print("=" * 74)
    print(f"[config] H60_MODE = {H60_MODE} | h60 source = {H60_BASE_SOURCE} "
          f"| flows from {FIRST_FLOW_YEAR} | entry window = {ENTRY_PROFILE_WINDOW}")
    nat, years = load_national()

    # [SUP-64] estimate_flows must run BEFORE the cohort trace, because the trace
    # now propagates entries forward through the model's own hazards.
    h, entry_profile_raw_shape, entrants, ep_raw, fixed = estimate_flows(nat, years)

    # ---- [SUP-62][SUP-64] cohort trace, net of the entrants still present
    trace, coh_detail = cohort_trace(nat, years, h)
    trace.to_csv(RES / "cohort_trace.csv", index=False)
    coh_detail.to_csv(RES / "cohort_trace_entry_propagation.csv", index=False)
    print("\n[cohort trace] following each band forward "
          f"{max(years)-min(years)} years (a band ages "
          f"{int(round((max(years)-min(years))*G))} bands over that span).")
    print("   The destination band ALSO holds everyone who entered during the")
    print("   interval and aged into it, so the gross multiplier does not measure")
    print("   cohort survival. Entrants are located by propagating each year's")
    print("   gross entries forward through the model's own ageing arithmetic")
    print("   [SUP-64], then subtracted.")
    print(f"   {'from':>6} {'->':^4} {'to':>6}{'n_start':>10}{'n_end':>10}"
          f"{'entrants':>10}{'net_end':>10}{'gross':>8}{'net':>8}")
    for _, r in trace.iterrows():
        print(f"   {r.band_start:>6} {'->':^4} {r.band_end:>6}{r.n_start:>10,.0f}"
              f"{r.n_end:>10,.0f}{r.entrants_still_present:>10,.0f}"
              f"{r.n_end_net_of_entrants:>10,.0f}"
              f"{r.gross_multiplier:>7.2f}x{r.net_multiplier:>7.2f}x")
    mn_g = float(trace["gross_multiplier"].min())
    mn_n = float(trace["net_multiplier"].min())
    n_below = int((trace["net_multiplier"] < COHORT_SHRINK_TOL).sum())
    tot_arriving = float(trace["entrants_still_present"].sum())
    print(f"   smallest multiplier: gross {mn_g:.2f}x | net {mn_n:.2f}x "
          f"({n_below} of {len(trace)} bands below {COHORT_SHRINK_TOL:.2f} on the "
          f"net measure)")
    if not coh_detail.empty:
        print(f"   {coh_detail['entries_injected'].sum():,.0f} gross entries injected "
              f"across {len(coh_detail)} cohorts; "
              f"{coh_detail['survivors_at_target'].sum():,.0f} still present in "
              f"{max(years)}")
        print(f"   (mean survival {100*coh_detail['survival_rate'].mean():.1f}%). "
              f"Entrants are assumed to face the SAME band hazards as incumbents,")
        print("   which is a behavioural assumption, not an accounting identity.")
    if mn_n >= COHORT_SHRINK_TOL:
        print("   No cohort shrank materially once arriving entrants are removed, so")
        print("   falling band HEADCOUNTS are composition rather than exits.")
    else:
        print(f"   At least one cohort DID shrink once entrants are removed "
              f"({mn_n:.2f}x).")
        print("   Falling band headcounts are NOT purely composition. The composition")
        print("   claim should not be repeated without pointing at cohort_trace.csv.")

    fill = band_fill_diagnostics(nat, years)
    fill.to_csv(RES / "h60_band_fill_diagnostics.csv", index=False)
    print("\n[60+ band] is it in steady state? (steady state = ageing inflow / h)")
    print(f"   {'yr':>5}{'stock':>9}{'inflow':>8}{'exits':>8}{'h_obs':>8}"
          f"{'steady state':>14}{'gap':>8}")
    for _, r in fill.iterrows():
        print(f"   {int(r.year):>5}{r.stock_60plus:>9,.0f}{r.ageing_inflow:>8,.0f}"
              f"{r.implied_exits:>8,.0f}{r.h_observed:>8.4f}"
              f"{r.steady_state_at_h_obs:>14,.0f}{r.fill_gap_pct:>7.0f}%")
    if not fill.empty and bool(fill["guard_binding"].any()):
        bad = fill[fill["guard_binding"]]
        print(f"   NOTE: the zero floor / HCLIP clip bound in "
              f"{', '.join(str(int(y)) for y in bad['year'])}. The unguarded ratio "
              f"is in h_observed_unclipped")
        print(f"   (min {bad['h_observed_unclipped'].min():+.4f}). A negative value "
              f"means the band grew faster than its ageing inflow that year.")
    if not fill.empty and (fill["fill_gap_pct"] < -5).all():
        print("   The band is BELOW its own steady state in every year, i.e. still")
        print("   filling. exits/stock therefore UNDERSTATES the structural hazard.")

    h60_est = float(h[NB - 1])
    h_post2017 = float(fill[fill["year"] >= 2018]["h_observed"].mean()) \
        if not fill.empty else np.nan
    H60_SCENARIOS["post_2017"] = h_post2017
    H60_SCENARIOS["base"] = h60_base_value(h60_est)

    print(f"\n[hazard 60+] base in use = {H60_SCENARIOS['base']:.4f} "
          f"(source = {H60_BASE_SOURCE}) -> exit age "
          f"{implied_mean_exit_age(H60_SCENARIOS['base']):.1f}")
    if H60_MODE == "tenure":
        print(f"   NOTE: H60_MODE = tenure, so the projection runs the QUEUE "
              f"(exit age {tenure_curve_mean_exit_age():.1f}).")
        print(f"   The flat base above is retained for the scenario grid and for the")
        print(f"   specification comparison; it is not what the projection uses.")
    print(f"   window means: full {FIRST_FLOW_YEAR}+ = {h60_est:.4f} "
          f"(exit {implied_mean_exit_age(h60_est):.1f}) | "
          f"post-2017 = {h_post2017:.4f} (exit {implied_mean_exit_age(h_post2017):.1f})")
    print("   The base is set from the statutory exit age, not from a window mean,")
    print("   because the window mean is fragile and the band is not in steady state.")
    if np.isfinite(h60_est) and H60_SCENARIOS["base"] > h60_est:
        print(f"   It is {100*(H60_SCENARIOS['base']/h60_est-1):+.1f}% above the "
              f"full-window mean and implies retirement "
              f"{implied_mean_exit_age(h60_est) - implied_mean_exit_age(H60_SCENARIOS['base']):.1f} "
              f"years EARLIER.")
        print("   The sensitivity grid prices that choice; report the range, not just")
        print("   the base.")
    for tag, v in sorted(H60_SCENARIOS.items(),
                         key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -1)):
        if not np.isfinite(v):
            continue
        ss = G * nat[LAST][NB - 2] / v
        print(f"     h60 = {v:.4f} ({tag:<11}) -> exit age {implied_mean_exit_age(v):5.1f}"
              f" | 60+ steady state {ss:8,.0f}"
              f" ({100*ss/nat[LAST].sum():4.1f}% of today's workforce)")

    h_old = legacy_hazards(nat, years)
    print("\n[hazards] by band (real regime, post-%d):" % FIRST_FLOW_YEAR)
    print(f"   {'band':>6}{'identity':>12}{'pre-v9':>10}{'ratio':>8}   note")
    for a in range(NB):
        if a == NB - 1:
            note = "estimated (60+)"
        elif abs(h[a] - band_floor(a)) < 1e-9:
            note = "at the attrition floor"
        else:
            note = "floor + estimated excess"
        ratio = (h_old[a] / h[a]) if h[a] > 0 else np.nan
        print(f"   {NICE[a]:>6}{h[a]:>12.4f}{h_old[a]:>10.4f}{ratio:>8.2f}x   {note}")
    print("   (net flows do not identify gross exits; the floor is an assumption)")

    _ys = flow_years(years)
    d_h5559 = pd.Series(
        [_implied_hazard(nat[t], nat[t1], NB - 2)
         for t, t1 in zip(_ys[:-1], _ys[1:]) if t1 - t == 1])

    r5559 = h[NB - 2] * nat[LAST][NB - 2]
    r60 = H60_SCENARIOS["base"] * nat[LAST][NB - 1]
    print(f"\n[SUP-21] composition of the retirement wave at {LAST} stocks:")
    print(f"   55-59: h = {h[NB-2]:.4f} x {nat[LAST][NB-2]:,.0f} = {r5559:,.0f}/yr "
          f"({100*r5559/(r5559+r60):.1f}%)")
    print(f"   60+  : h = {H60_SCENARIOS['base']:.4f} x {nat[LAST][NB-1]:,.0f} = "
          f"{r60:,.0f}/yr ({100*r60/(r5559+r60):.1f}%)")
    print("   Almost the whole wave is one parameter, so every statement about WHEN")
    print("   it lands is a statement about the 60+ specification alone.")
    if series_is_degenerate(d_h5559):
        print(f"   NOTE: the observed 55-59 hazard is exactly the "
              f"{band_floor(NB-2):.4f} floor in every year of the window. After "
              f"[SUP-36] the")
        print("   identity returns max(floor*n0, -net), and the net flow never falls")
        print("   below -floor*n0, so that series carries NO behavioural information.")
        print(f"   The {r5559:,.0f}/yr above is assumption, not estimate.")

    _, _, _, ep_raw_last5, _ = estimate_flows(nat, years, repair=False, window="last5")

    eh = entries_history(nat, years)
    ch_share, ch_label = channel_share(eh)
    entry_profile = apply_channel_share(entry_profile_raw_shape, ch_share)
    young_w = float(sum(entry_profile[a] for a in YOUNG_BANDS))

    print("\n[entry profile] estimation window matters (%):")
    print(f"   {'band':>6}  {'shape(full)':>12}  {'last5':>8}  {'used':>8}")
    for a in range(NB - 1):
        print(f"   {NICE[a]:>6}  {100*entry_profile_raw_shape[a]:>11.1f}  "
              f"{100*ep_raw_last5[a]:>7.1f}  {100*entry_profile[a]:>7.1f}")
    if fixed:
        for f in fixed:
            print(f"   NOTE: repair fired on {f['band']} "
                  f"({100*f['raw_share']:.1f}% vs neighbour mean "
                  f"{100*f['neighbour_mean']:.1f}%). Investigate.")
    else:
        print("   Repair did not fire; the window artefacts are gone without a heuristic.")

    obs = eh["young_share"]
    yr = eh["year"].to_numpy(float)
    slope = float(np.polyfit(yr, obs.to_numpy(float), 1)[0]) if len(yr) > 1 else np.nan
    print(f"\n[SUP-19] the young/lateral split is NOT stationary: OLS slope "
          f"{100*slope:+.2f} pp/yr")
    print(f"   observed: {int(yr[0])} {100*obs.iloc[0]:.1f}%  ->  "
          f"{int(yr[-1])} {100*obs.iloc[-1]:.1f}%")
    print(f"   full {100*obs.mean():.1f}% | last5 {100*obs.tail(5).mean():.1f}% "
          f"| last3 {100*obs.tail(3).mean():.1f}%")
    print(f"   band SHAPE from the full window, channel SPLIT from {ch_label} "
          f"-> {100*ch_share:.1f}%.")
    print(f"[entry channels] <30 = {100*young_w:.1f}% of entries, "
          f"30+ = {100*(1-young_w):.1f}%")
    print(f"[recruitment] mean annual entries (estimation window) = {entrants:,.0f}")

    h_base = apply_h60_scenario(h, H60_SCENARIOS["base"])

    demand = load_demand()
    demand, warn = resolve_demand(demand)
    if warn:
        print("[warn]", warn)
    stock0 = nat[LAST]

    base_entry = "endogenous"
    if demand is None:
        print("[warn] demand (02_demand) missing -> base projection falls back to "
              "REPLACEMENT (E=exits)")
        base_entry = "replacement"
    out = project(stock0, h_base, entry_profile, demand, entrants, entry=base_entry)
    w = pd.DataFrame({"year": FYEARS,
                      "stock": [out["stock"][y].sum() for y in FYEARS],
                      "retirements_wave": [out["retire"][y] for y in FYEARS],
                      "recruitment": [out["recruit"][y] for y in FYEARS],
                      "exits": [out["exit"][y] for y in FYEARS],
                      "recruitment_young": [out["recruit_young"][y] for y in FYEARS],
                      "recruitment_lateral": [out["recruit_lateral"][y] for y in FYEARS]})
    w["entry_constraint"] = "none (unconstrained; 04_gap applies the restriction)"
    w["h60_mode"] = H60_MODE
    w["h60_base_flat"] = H60_SCENARIOS["base"] if H60_MODE != "tenure" else np.nan
    w["tenure_mean_exit_age"] = (tenure_curve_mean_exit_age()
                                 if H60_MODE == "tenure" else np.nan)
    w.to_csv(RES / "supply_projection.csv", index=False)

    prev = np.r_[stock0.sum(), w["stock"].to_numpy()[:-1]]
    resid = np.abs(prev - w["exits"].to_numpy() + w["recruitment"].to_numpy()
                   - w["stock"].to_numpy()).max()
    print(f"[check] flow identity stock_y = stock_(y-1) - exits + recruit: "
          f"max abs residual = {resid:.6f}")

    hist_resid = 0.0
    for t, t1 in zip(_ys[:-1], _ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        e = sum(_gross_entries(n0, n1, a, band_floor(a)) for a in range(NB - 1))
        x = sum(_gross_exits(n0, n1, a, band_floor(a)) for a in range(NB))
        hist_resid = max(hist_resid, abs(e - x - (n1.sum() - n0.sum())))
    print(f"[check] historical gross flows: entries - exits - change in stock, "
          f"max abs residual = {hist_resid:.6f}")

    haz_resid = 0.0
    for t, t1 in zip(_ys[:-1], _ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        for a in range(NB - 1):
            haz_resid = max(haz_resid,
                            abs(_implied_hazard(n0, n1, a) * n0[a]
                                - _gross_exits(n0, n1, a, band_floor(a))))
    print(f"[check] hazard x stock reproduces gross exits band by band: "
          f"max abs residual = {haz_resid:.6f}")

    fill_resid = 0.0
    for _, r in fill.iterrows():
        t1 = int(r["year"]); t = t1 - 1
        if t in nat and t1 in nat:
            fill_resid = max(fill_resid,
                             abs(r["h_observed"]
                                 - _implied_hazard(nat[t], nat[t1], NB - 1)))
    print(f"[check] band_fill_diagnostics uses the shared hazard helper: "
          f"max abs residual = {fill_resid:.6f}")

    # [SUP-64] propagated survivors can never exceed the entries injected
    if not coh_detail.empty:
        inj = float(coh_detail["entries_injected"].sum())
        surv = float(coh_detail["survivors_at_target"].sum())
        ok = surv <= inj + 1e-6
        print(f"[check] cohort propagation: {surv:,.0f} survivors from {inj:,.0f} "
              f"injected entries ({'consistent' if ok else 'INCONSISTENT'}); "
              f"the gap is the attrition applied on the way.")

    pd.DataFrame({"band": NICE, "hazard_flat_chain": h_base,
                  "hazard_estimated": h,
                  "hazard_pre_v9_estimator": h_old,
                  "attrition_floor": [band_floor(a) for a in range(NB)],
                  "at_floor": [abs(h[a] - band_floor(a)) < 1e-9 for a in range(NB)],
                  "observed_series_degenerate": [
                      series_is_degenerate(pd.Series(
                          [_implied_hazard(nat[t], nat[t1], a)
                           for t, t1 in zip(_ys[:-1], _ys[1:]) if t1 - t == 1]))
                      for a in range(NB)],
                  "entry_profile_used": entry_profile,
                  "entry_profile_shape_full_window": entry_profile_raw_shape,
                  "entry_profile_raw": ep_raw,
                  "entry_profile_last5_window": ep_raw_last5,
                  "h60_mode": H60_MODE,
                  "h60_used_by_projection": (
                      "tenure_queue" if H60_MODE == "tenure"
                      else f"{H60_SCENARIOS['base']:.6f}"),
                  "h60_base_source": H60_BASE_SOURCE,
                  "h60_target_exit_age": H60_TARGET_EXIT_AGE,
                  "h60_estimated_full_window": h60_est,
                  "h60_post_2017": h_post2017,
                  "entry_profile_window": ENTRY_PROFILE_WINDOW,
                  "channel_share_window": CHANNEL_SHARE_WINDOW,
                  "channel_share_used": ch_share,
                  "attrition_floor_on_5559": ATTRITION_FLOOR_ON_5559,
                  }).to_csv(RES / "supply_hazards.csv", index=False)

    w[["year", "recruitment", "recruitment_young", "recruitment_lateral"]].to_csv(
        RES / "supply_recruitment_by_channel.csv", index=False)
    print("[csv] supply_recruitment_by_channel.csv  <- 04_gap should restrict the")
    print("      young channel only, not the total.")

    eh.to_csv(RES / "supply_entries_history.csv", index=False)
    print(f"[csv] supply_entries_history.csv  ({len(eh)} years, "
          f"{int(eh.year.min())}-{int(eh.year.max())}; "
          f"mean entries={eh['new_entrants'].mean():,.0f}/yr)")

    feas = feasibility_report(w, eh, young_w=young_w, share_label=ch_label)

    print("\n[experiment] entry policies:")
    pol, _, _ = experiment_entry_policies(nat, years, demand,
                                          entry_profile=entry_profile)
    for label, r in pol.items():
        print(f"   {label:<30} stock 2040 = {r['df'].stock.iloc[-1]:>10,.0f}"
              f" | cum. recruit = {r['cum_recruit']:>10,.0f}")
    endo = pol["endogenous (follows demand)"]["cum_recruit"]
    if base_entry == "endogenous":
        gap = abs(endo - float(w["recruitment"].sum()))
        print(f"[check] endogenous policy vs supply_projection.csv: "
              f"difference = {gap:.6f} "
              f"({'consistent' if gap < 1e-6 else 'INCONSISTENT'})")

    h5559_last3 = float(d_h5559.tail(3).mean()) if len(d_h5559) >= 3 else None
    base, tor = sensitivity(h_base, entry_profile, demand, stock0,
                            base_entry, entrants, eh=eh, eh_h5559=h5559_last3)
    tor.to_csv(RES / "supply_sensitivity.csv", index=False)
    print(f"[sensitivity] base cumulative recruitment = {base:,.0f}")
    n_dead = int(tor["degenerate"].sum())
    print(f"[sensitivity] {len(tor)-n_dead} live scenario(s), {n_dead} degenerate")
    if n_dead:
        for s in tor[tor["degenerate"]]["scenario"]:
            print(f"              DEGENERATE (lever not connected): {s}")

    print("\n[universes]")
    rake_all = []; share_all = []
    for scope in ("national", "nuts2", "nuts3"):
        dfu, rake, sstats = run_universe(scope, demand, national_df=w)
        if dfu.empty:
            continue
        dfu.to_csv(RES / f"supply_universe_{scope}.csv", index=False)
        if not rake.empty:
            rake.insert(0, "scope", scope)
            rake_all.append(rake)
        if sstats:
            share_all.append(sstats)
        fig_universe(dfu, scope)
        tot = dfu.groupby("year")["stock"].sum().iloc[-1]
        rec = dfu["recruitment"].sum()
        print(f"   {scope:<9} regions={dfu.region.nunique():<4} "
              f"stock2040={tot:,.0f} cum.recruit={rec:,.0f}")
    if rake_all:
        pd.concat(rake_all, ignore_index=True).to_csv(
            RES / "regional_raking_factors.csv", index=False)
        print("[csv] regional_raking_factors.csv  <- size of the pre-raking mismatch")

    fig_flow(h_base); fig_wave(w); fig_tornado(tor)
    fig_policies(pol); fig_pyramid(nat, years, pol, base_policy=(
        "endogenous (follows demand)" if base_entry == "endogenous"
        else "replacement (replaces exits)"))
    fig_channel(w,
                hist_max=feas["hist_max"] if feas else None,
                hist_max_year=feas["hist_max_year"] if feas else None,
                hist_second=feas["hist_second_best"] if feas else None)

    f0, f1, gap_yrs = flow_window(years)
    _MANIFEST["stage1"] = {
        "h60_mode": H60_MODE,
        "h60_used_by_projection": ("tenure_queue" if H60_MODE == "tenure"
                                   else float(H60_SCENARIOS["base"])),
        "h60_base_flat": float(H60_SCENARIOS["base"]),
        "h60_base_source": H60_BASE_SOURCE,
        "h60_estimated_full_window": h60_est,
        "h60_post_2017": h_post2017,
        "live_mean_exit_age": float(tenure_curve_mean_exit_age()
                                    if H60_MODE == "tenure"
                                    else implied_mean_exit_age(H60_SCENARIOS["base"])),
        "steady_state_60plus_live_spec_2024_feeder":
            float(steady_state_60plus(stock0[NB - 2])),
        "entry_profile_window": ENTRY_PROFILE_WINDOW,
        "channel_share_window": CHANNEL_SHARE_WINDOW,
        "channel_share_used": float(ch_share),
        "channel_share_label": ch_label,
        "attrition_floor_on_5559": ATTRITION_FLOOR_ON_5559,
        "h5559_series_degenerate": bool(series_is_degenerate(d_h5559)),
        # [SUP-60] deployed vs estimated, kept apart
        "hazards_deployed": [float(x) for x in h_base],
        "hazards_estimated": [float(x) for x in h],
        "wave_share_from_60plus": float(r60 / (r60 + r5559)),
        "first_flow_year": FIRST_FLOW_YEAR,
        "stock_window": [int(min(years)), int(max(years))],
        "flow_window": [f0, f1],
        "stock_minus_flow_gap_years": gap_yrs,
        "entry_profile_repaired": [f["band"] for f in fixed],
        "young_channel_share": young_w,
        "regional_young_share_spread": share_all,
        "base_entry_policy": base_entry,
        "flow_identity_max_residual": float(resid),
        "historical_gross_flow_residual": float(hist_resid),
        "hazard_exit_consistency_residual": float(haz_resid),
        "band_fill_helper_residual": float(fill_resid),
        "band_fill_guard_binding_years": [int(y) for y in
                                          fill.loc[fill["guard_binding"], "year"]]
        if not fill.empty else [],
        # [SUP-64] cohort trace, forward-propagated
        "cohort_trace_method": "forward_propagation",
        "cohort_trace_min_gross_multiplier": float(mn_g),
        "cohort_trace_min_net_multiplier": float(mn_n),
        "cohort_trace_bands_below_one_net": n_below,
        "cohort_trace_entrants_subtracted": float(tot_arriving),
        "cohort_trace_entries_injected": float(coh_detail["entries_injected"].sum())
        if not coh_detail.empty else 0.0,
        "cohort_trace_survivors": float(coh_detail["survivors_at_target"].sum())
        if not coh_detail.empty else 0.0,
        "cohort_trace_assumes_entrant_hazards_equal_incumbent": True,
        "sensitivity_degenerate_scenarios": n_dead,
        "entry_constraint": "none (04_gap owns the restriction)",
        "feasibility": feas,
        "cum_recruitment": float(w["recruitment"].sum()),
    }
    print("\n[figures] written to", IMG)
    print("[tables] written to", RES)
    return w, entry_profile, share_all, (h_base, stock0, demand, entrants, base_entry)


# ============================================================================
# STAGE 2 -- ROBUSTNESS (cohort-aware 60+ tenure queue + Monte Carlo)
# ============================================================================

def rb_load_national():
    df = pd.read_excel(TEACHERS_FILE, sheet_name="historical", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year_start"]); df["year_start"] = df["year_start"].astype(int)
    years = sorted(df.year_start.unique()); nat = {}
    for y in years:
        s = df[df.year_start == y]
        nat[y] = np.array([float(pd.to_numeric(s[f"teachers_{b}"], errors="coerce").sum())
                           for b in BANDS])
    return nat, years


def rb_detect_placeholder(nat, years):
    ys = sorted(years)
    for i in range(1, len(ys)):
        if not np.allclose(nat[ys[i]], nat[ys[i - 1]]):
            return ys[i]
    return ys[0]


def rb_estimate_hazards(nat, years, exclude=None, min_year=MIN_HAZARD_YEAR):
    """[SUP-30][SUP-35] Same helper as stage 1, so the two cannot drift."""
    exclude = set(exclude or [])
    ys = [y for y in sorted(years) if y >= min_year]
    H = {a: [] for a in range(NB)}
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1 or t in exclude or t1 in exclude:
            continue
        n0, n1 = nat[t], nat[t1]
        for a in range(NB):
            H[a].append(_implied_hazard(n0, n1, a))
    h = np.array([np.nanmean(H[a]) if H[a] else band_floor(a) for a in range(NB)])
    h[0] = BASE_ATTRITION_YOUNGMID
    return h


def rb_h60_by_tenure(k, shift=0.0, scale=1.0):
    """[SUP-9][SUP-57] Tenure-dependent 60+ exit hazard.

    The curve is flat at H60_EARLY until year `ramp` in the band, then rises
    linearly to H60_PEAK over `span` years and stays there.

    [SUP-57] `span` used to be written as max((LEGAL_GAP + shift) - ramp, 1)
    with ramp = H60_RAMP + shift. The shift cancels algebraically, so that
    expression was always max(LEGAL_GAP - H60_RAMP, 1) and never responded to
    the shift. The design is intentional: `shift` TRANSLATES the curve along the
    tenure axis and leaves the ramp width alone, which is what makes
    tenure_shift_for_exit_age a clean one-parameter reparameterisation of the
    mean exit age. It is now written as the constant it always was.

    `scale` multiplies the hazard level and is the second, independent lever.
    """
    ramp = H60_RAMP + shift
    if k < ramp:
        return float(np.clip(H60_EARLY * scale, *HCLIP))
    span = max(LEGAL_GAP - H60_RAMP, 1)          # constant ramp width, see [SUP-57]
    frac = min((k - ramp) / span, 1.0)
    return float(np.clip((H60_EARLY + frac * (H60_PEAK - H60_EARLY)) * scale, *HCLIP))


def rb_seed_sixtyplus(n60, shift=0.0, scale=1.0):
    haz = np.array([rb_h60_by_tenure(k, shift, scale) for k in range(SIXTY_SLOTS)])
    surv = np.cumprod(np.r_[1.0, (1 - haz[:-1])]); surv /= surv.sum()
    return n60 * surv


def rb_step_year(n_bands, q60, h, entry, ent_profile, shift=0.0, scale=1.0):
    """[SUP-9] shift/scale reach the 60+ queue."""
    new = np.zeros(NB); exits = 0.0
    for a in range(NB - 1):
        stay = max(1 - G - h[a], 0.0)
        new[a] += n_bands[a] * stay
        if a < NB - 2:
            new[a + 1] += n_bands[a] * G
        exits += n_bands[a] * (1 - stay - G)
    inflow_to_60 = n_bands[NB - 2] * G
    newq = np.zeros(SIXTY_SLOTS); q_exits = 0.0
    for k in range(SIXTY_SLOTS):
        hk = rb_h60_by_tenure(k, shift, scale)
        stay = q60[k] * (1 - hk); q_exits += q60[k] * hk
        newq[min(k + 1, SIXTY_SLOTS - 1)] += stay
    newq[0] += inflow_to_60
    exits += q_exits
    new[NB - 1] = newq.sum()
    retire55 = n_bands[NB - 2] * h[NB - 2] + q_exits
    new = new + entry * ent_profile
    return np.clip(new, 0, None), newq, exits, retire55


def rb_load_demand():
    return load_demand()


def rb_project(stock0, h, ent_profile, demand=None, shift=0.0, scale=1.0):
    n = stock0.astype(float).copy()
    q60 = rb_seed_sixtyplus(n[NB - 1], shift, scale); n[NB - 1] = q60.sum()
    out = {}
    for y in FYEARS:
        # [SUP-66] two calls per year. The FIRST sizes the entry residual and its
        # state outputs (pre, preq) are discarded; only the SECOND advances n and
        # q60. Reassigning n or q60 from the first call would double-age the year.
        pre, preq, exits, retire = rb_step_year(n, q60, h, 0.0, ent_profile, shift, scale)
        after = pre.sum()
        if demand is not None and y in demand:
            E = max(demand[y] - after, 0.0)
        else:
            E = exits
        n, q60, exits, retire = rb_step_year(n, q60, h, E, ent_profile, shift, scale)
        out[y] = {"stock": n.sum(), "p55": n[NB - 2] + n[NB - 1], "exits": exits,
                  "retire55": retire, "recruit": E}
    return out


def rb_monte_carlo(stock0, h0, ent_profile, demand, decompose=True, seed=MC_SEED):
    """[SUP-9][SUP-10][SUP-17] Monte Carlo over the parameters the model uses."""
    rng = np.random.default_rng(seed)

    def _draw_hazards(h0):
        h = h0.copy()
        h[NB - 2] = np.clip(h0[NB - 2] * np.exp(rng.normal(0, 0.20)), *HCLIP)
        for a in range(3, 7):
            h[a] = np.clip(h0[a] * np.exp(rng.normal(0, 0.15)), *HCLIP)
        return h

    def _draw_demand(demand):
        if not demand:
            return None
        inn = rng.normal(0, MC_DEMAND_STEP, len(FYEARS))
        inn[0] *= MC_DEMAND_FIRST_YEAR_FRAC
        shocks = inn.cumsum()
        return {y: demand[y] * float(np.exp(shocks[i]))
                for i, y in enumerate(FYEARS) if y in demand}

    def _run(reps, vary_haz=True, vary_ret=True, vary_dem=True):
        d = np.zeros((reps, len(FYEARS)))
        for m in range(reps):
            h = _draw_hazards(h0) if vary_haz else h0.copy()
            if vary_ret:
                shift = rng.normal(0, MC_RET_AGE_SD)
                scale = float(np.exp(rng.normal(0, MC_RET_SCALE_SD)))
            else:
                shift, scale = 0.0, 1.0
            dem = _draw_demand(demand) if vary_dem else demand
            o = rb_project(stock0, h, ent_profile, dem, shift, scale)
            d[m] = [o[y]["recruit"] for y in FYEARS]
        return d

    draws = _run(N_MC)
    p10, p50, p90 = np.percentile(draws, [10, 50, 90], axis=0)
    mc = pd.DataFrame({"year": FYEARS, "p10": p10, "p50": p50, "p90": p90,
                       "mean": draws.mean(0)})
    mc["width"] = mc["p90"] - mc["p10"]
    mc["width_pct_of_median"] = 100 * mc["width"] / mc["p50"]

    decomp = None
    if decompose:
        n_small = max(N_MC // 4, 200)
        parts = {}
        for name, kw in [("retirement_age",
                          dict(vary_haz=False, vary_ret=True, vary_dem=False)),
                         ("other_hazards",
                          dict(vary_haz=True, vary_ret=False, vary_dem=False)),
                         ("demand",
                          dict(vary_haz=False, vary_ret=False, vary_dem=True))]:
            rng = np.random.default_rng(seed + 1)
            d = _run(n_small, **kw)
            parts[name] = float(np.mean(np.var(d, axis=0)))
        tot = sum(parts.values())
        decomp = pd.DataFrame([{"source": k, "mean_variance": v,
                                "share_pct": 100 * v / tot if tot > 0 else np.nan}
                               for k, v in parts.items()]).sort_values(
                                   "share_pct", ascending=False)
    return mc, draws, decomp


def compare_specifications(stage1_inputs, entry_profile):
    """[SUP-50][SUP-52][SUP-63] Compare the two 60+ specifications properly.

    [SUP-63] The flat chain is calibrated to a mean exit age of 67.0 while the
    unshifted tenure queue implies 65.2, so a raw flat-vs-queue comparison mixes
    two different things: the SHAPE of the hazard over time in the band, and the
    LEVEL of the retirement age.

    A third path is therefore projected: the queue shifted, via
    tenure_shift_for_exit_age, to the flat spec's own mean exit age. The gap then
    splits cleanly:

        form effect = age-matched queue  vs  flat          (shape only)
        age  effect = unshifted queue    vs  age-matched   (retirement age only)
        total       = unshifted queue    vs  flat
    """
    h_base, stock0, demand, entrants, base_entry = stage1_inputs

    flat_age = implied_mean_exit_age(H60_SCENARIOS["base"])
    shift_matched = tenure_shift_for_exit_age(flat_age)

    flat = project(stock0, h_base, entry_profile, demand, entrants,
                   entry=base_entry, tenure=False)
    ten = project(stock0, h_base, entry_profile, demand, entrants,
                  entry=base_entry, tenure=True)
    ten_m = project(stock0, h_base, entry_profile, demand, entrants,
                    entry=base_entry, tenure=True, tenure_shift=shift_matched)

    cmp = pd.DataFrame({
        "year": FYEARS,
        "retire_flat": [flat["retire"][y] for y in FYEARS],
        "retire_tenure": [ten["retire"][y] for y in FYEARS],
        "retire_tenure_age_matched": [ten_m["retire"][y] for y in FYEARS],
        "recruit_flat": [flat["recruit"][y] for y in FYEARS],
        "recruit_tenure": [ten["recruit"][y] for y in FYEARS],
        "recruit_tenure_age_matched": [ten_m["recruit"][y] for y in FYEARS]})
    cmp["retire_diff"] = cmp["retire_tenure"] - cmp["retire_flat"]
    cmp["live_mode"] = H60_MODE
    cmp["flat_mean_exit_age"] = flat_age
    cmp["tenure_mean_exit_age"] = tenure_curve_mean_exit_age()
    cmp["tenure_shift_for_age_match"] = shift_matched
    cmp.to_csv(RES / "h60_specification_comparison.csv", index=False)

    a, b = cmp.iloc[0], cmp.iloc[-1]
    cum_f = float(cmp["recruit_flat"].sum())
    cum_t = float(cmp["recruit_tenure"].sum())
    cum_m = float(cmp["recruit_tenure_age_matched"].sum())

    def pct(x, y):
        return 100 * (x / y - 1) if y else np.nan

    total_gap = pct(cum_t, cum_f)
    form_gap = pct(cum_m, cum_f)
    age_gap = pct(cum_t, cum_m)

    yr1_total = pct(a.retire_tenure, a.retire_flat)
    yr1_form = pct(a.retire_tenure_age_matched, a.retire_flat)
    yr1_age = pct(a.retire_tenure, a.retire_tenure_age_matched)

    dominant = "functional form" if abs(form_gap) >= abs(age_gap) else "retirement age"
    opposed = (np.isfinite(form_gap) and np.isfinite(age_gap)
               and form_gap * age_gap < 0)

    print("\n" + "=" * 74)
    print("THE TWO 60+ SPECIFICATIONS, DECOMPOSED")
    print("=" * 74)
    print(f"  live specification: H60_MODE = {H60_MODE}")
    print(f"  flat chain   -> mean exit age {flat_age:.2f}")
    print(f"  queue        -> mean exit age {tenure_curve_mean_exit_age():.2f}  "
          f"({flat_age - tenure_curve_mean_exit_age():+.2f} years vs flat)")
    print(f"  queue shifted {shift_matched:+.3f}y -> mean exit age "
          f"{tenure_mean_exit_age(shift_matched):.2f}  (matched to the flat chain)")
    print()
    print(f"  {'':28}{'2025 retire':>13}{'2040 retire':>13}{'cum recruit':>14}")
    print(f"  {'flat':28}{a.retire_flat:>13,.0f}{b.retire_flat:>13,.0f}"
          f"{cum_f:>14,.0f}")
    print(f"  {'queue, age-matched':28}{a.retire_tenure_age_matched:>13,.0f}"
          f"{b.retire_tenure_age_matched:>13,.0f}{cum_m:>14,.0f}")
    print(f"  {'queue, as specified':28}{a.retire_tenure:>13,.0f}"
          f"{b.retire_tenure:>13,.0f}{cum_t:>14,.0f}")
    print()
    print(f"  DECOMPOSITION of the flat -> queue gap in cumulative recruitment:")
    print(f"    functional form (shape only) : {form_gap:+6.2f}%")
    print(f"    retirement age (level only)  : {age_gap:+6.2f}%")
    print(f"    total                        : {total_gap:+6.2f}%")
    print(f"    first-year retirements       : form {yr1_form:+.1f}%, "
          f"age {yr1_age:+.1f}%, total {yr1_total:+.1f}%")
    print(f"  The larger of the two is {dominant.upper()}.")
    if opposed:
        print("  The two channels have OPPOSITE SIGNS: the raw flat-vs-queue gap is a")
        print("  net of two forces pulling apart, so quoting it alone understates both.")
    if abs(form_gap) < 3.0:
        print("  Holding the exit age fixed, the two specifications are close: most of")
        print("  the raw gap is the retirement age, not the shape of the hazard.")
    else:
        print("  Even at a common exit age the two specifications disagree materially,")
        print("  so the shape of the hazard within the band is doing real work.")

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.plot(cmp.year, cmp.retire_flat, "o-", color=EXITC, lw=2.2, ms=5,
            label=f"flat -- exit age {flat_age:.1f}")
    ax.plot(cmp.year, cmp.retire_tenure_age_matched, "^-", color=MIDC, lw=2.0, ms=5,
            label=f"tenure queue, age-matched -- exit age "
                  f"{tenure_mean_exit_age(shift_matched):.1f}")
    ax.plot(cmp.year, cmp.retire_tenure, "s--", color=AGEING, lw=2.2, ms=5,
            label=f"tenure queue, as specified -- exit age "
                  f"{tenure_curve_mean_exit_age():.1f}")
    ax.fill_between(cmp.year, cmp.retire_flat, cmp.retire_tenure_age_matched,
                    color=MIDC, alpha=0.12)
    ax.fill_between(cmp.year, cmp.retire_tenure_age_matched, cmp.retire_tenure,
                    color=AGEING, alpha=0.12)
    sub = (f"cumulative recruitment: functional form {form_gap:+.1f}%, "
           f"retirement age {age_gap:+.1f}%, total {total_gap:+.1f}%")
    ax.set_title(f"Retirement wave under the two 60+ specifications, decomposed\n{sub}",
                 fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("retirements per year")
    style(ax); ax.legend(frameon=False, fontsize=9)
    foot = ("The amber band is the effect of the hazard SHAPE at a common exit age; "
            "the grey band is the effect of the exit age itself.\n"
            f"All three paths are projected here; the live run uses "
            f"H60_MODE = {H60_MODE}.")
    if opposed:
        foot += " The two effects have opposite signs and partly cancel."
    fig.text(0.5, 0.005, foot, ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(IMG / "h60_specification_comparison.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"cum_flat": cum_f, "cum_tenure": cum_t,
            "cum_tenure_age_matched": cum_m,
            "cum_gap_pct": float(total_gap),
            "form_effect_pct": float(form_gap),
            "age_effect_pct": float(age_gap),
            "effects_have_opposite_signs": bool(opposed),
            "first_year_gap_pct": float(yr1_total),
            "first_year_form_pct": float(yr1_form),
            "first_year_age_pct": float(yr1_age),
            "flat_mean_exit_age": float(flat_age),
            "tenure_mean_exit_age": float(tenure_curve_mean_exit_age()),
            "tenure_shift_for_age_match": float(shift_matched),
            "dominant_channel": dominant}


def stage_robustness(entry_profile=None, stage1_inputs=None):
    """STAGE 2 -- tenure-aware 60+ cohort model with Monte Carlo bands.

    [SUP-59] The `base_wave` argument was removed. It was never read.
    """
    print("=" * 74)
    print("STAGE 2 -- robustness (cohort-aware 60+ tenure queue + Monte Carlo)")
    print("=" * 74)
    nat, years = rb_load_national()
    first_real = rb_detect_placeholder(nat, years)
    print(f"First year the age vector changes: {first_real}  "
          f"(years < {MIN_HAZARD_YEAR} treated as placeholder)")

    if entry_profile is None:
        _, entry_profile, _, _, _ = estimate_flows(
            nat, [y for y in years if y >= FIRST_REAL_YEAR])
        print("[entry profile] stage 1 profile unavailable; re-estimated here")
    else:
        print("[entry profile] inherited from stage 1")
    ent_profile = entry_profile

    demand = rb_load_demand()
    demand, _ = resolve_demand(demand)

    h_bad = rb_estimate_hazards(nat, years, min_year=min(years))
    h_good = rb_estimate_hazards(nat, years, min_year=MIN_HAZARD_YEAR)
    print("\n60+ exit hazard:")
    print(f"  with placeholder ({min(years)}+): {h_bad[NB-1]:.4f}   <-- clip-inflated")
    print(f"  clean ({MIN_HAZARD_YEAR}+)          : {h_good[NB-1]:.4f}   <-- real regime")
    print("  NOTE: stage 2 always runs the tenure queue, and rb_step_year reads the")
    print("  60+ hazard from rb_h60_by_tenure, never from h[8]. Both figures above")
    print("  are descriptive; h_good[8] is estimated and then discarded here. The")
    print("  bands BELOW 60+ are the part of h_good the projection actually uses.")

    print(f"\n[SUP-15] implied mean exit age from the 60+ band:")
    print(f"    tenure queue                  : {tenure_curve_mean_exit_age():.1f}")
    print(f"    flat h60 = {H60_SCENARIOS['base']:.4f}      : "
          f"{implied_mean_exit_age(H60_SCENARIOS['base']):.1f}")
    print(f"    DGEEC continuity rates        : 66.8   (report reference age 67)")

    out = rb_project(nat[LAST], h_good, ent_profile, demand)
    wave = pd.DataFrame([{"year": y, **out[y]} for y in FYEARS])
    wave.to_csv(RES / "supply_wave_v3.csv", index=False)

    pd.DataFrame({"tenure_in_60plus": list(range(LEGAL_GAP + 3)),
                  "exit_hazard": [rb_h60_by_tenure(k) for k in range(LEGAL_GAP + 3)],
                  "specification": "tenure_queue",
                  "used_by": "stage_2_always",
                  "stage1_h60_mode": H60_MODE,
                  "ramp_start_year_in_band": H60_RAMP,
                  "ramp_width_years": max(LEGAL_GAP - H60_RAMP, 1),
                  "implied_mean_exit_age": tenure_curve_mean_exit_age()}
                 ).to_csv(RES / "h60_tenure_curve.csv", index=False)

    mc, draws, decomp = rb_monte_carlo(nat[LAST], h_good, ent_profile, demand)
    mc.to_csv(RES / "recruitment_montecarlo_v3.csv", index=False)
    if decomp is not None:
        decomp.to_csv(RES / "mc_variance_decomposition.csv", index=False)
        print("\n[MC] where the band comes from (one-at-a-time variance):")
        for _, r in decomp.iterrows():
            print(f"     {r['source']:<16} {r['share_pct']:5.2f}%")
        lev = (float(np.mean(list(demand.values()))) /
               float(mc["p50"].mean())) if demand else np.nan
        if np.isfinite(lev) and lev > 0:
            print(f"     demand leverage onto recruitment: {lev:.0f}x "
                  f"(recruitment is {100/lev:.1f}% of demand).")
            print("     Most of this band is block-02 uncertainty passing through.")

    cum = draws.sum(1); c10, c50, c90 = np.percentile(cum, [10, 50, 90])

    print("\nRetirement wave (first years):")
    print(wave[["year", "retire55", "recruit", "stock"]].head(6).round(0)
          .to_string(index=False))
    print(f"\nCumulative recruitment {FYEARS[0]}-{FYEARS[-1]}:  P10={c10:,.0f}  "
          f"P50={c50:,.0f}  P90={c90:,.0f}   [seed={MC_SEED}, N={N_MC}]")
    print("     This P50 is the TENURE-QUEUE path. Stage 1's base is the flat chain")
    print("     and is a different number; see the decomposition below before")
    print("     quoting either.")

    w1 = mc.iloc[0]; wl = mc.iloc[-1]
    imax = int(mc["width_pct_of_median"].idxmax()); wmax = mc.loc[imax]
    print(f"[MC] band width: {int(w1.year)} = {w1.width:,.0f} "
          f"({w1.width_pct_of_median:.1f}% of median)  ->  "
          f"{int(wl.year)} = {wl.width:,.0f} ({wl.width_pct_of_median:.1f}%)")
    print(f"[MC] widest year in RELATIVE terms: {int(wmax.year)} = "
          f"{wmax.width_pct_of_median:.1f}% of median (absolute width "
          f"{wmax.width:,.0f}).")
    if imax not in (0, len(mc) - 1):
        print("     The band does NOT widen monotonically: it opens early and narrows")
        print("     after, because demand dips and recruitment is a small residual.")

    recomputed = np.percentile(draws[:, 0], [10, 90])
    drift = max(abs(recomputed[0] - w1.p10), abs(recomputed[1] - w1.p90))
    assert drift < 1e-9, (f"MC console/file mismatch: percentiles recomputed from "
                          f"draws differ by {drift:.6f}")
    print(f"[check] MC percentiles recomputed from the draws match the written file "
          f"(max drift {drift:.2e})")

    if w1.width <= 0.05 * w1.p50:
        print("[MC] WARNING: the first-year band is near zero, which means the")
        print("     dominant parameters are not being drawn. Check the draw.")

    spec_cmp = None
    if stage1_inputs is not None:
        spec_cmp = compare_specifications(stage1_inputs, ent_profile)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(wave.year, wave.retire55, "o-", color=EXITC, lw=2.2, ms=5,
            label=f"retirement wave ({spec_label('tenure')})")
    ax.plot(wave.year, wave.recruit, "s--", color=ENTRY, lw=2.2, ms=5,
            label="required recruitment, stage 2 tenure path (unconstrained)")
    ax.fill_between(mc.year, mc.p10, mc.p90, color=ENTRY, alpha=0.12,
                    label="recruitment P10-P90 (mostly block-02 demand uncertainty)")
    ax.set_title("Retirement wave and recruitment under the tenure queue, "
                 "2025-2040 -- STAGE 2", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("teachers per year")
    style(ax); ax.legend(frameon=False)
    fig.text(0.5, 0.005, "Stage 1 publishes its own recruitment path on the flat "
             "chain; see wave_vs_recruitment.png and "
             "h60_specification_comparison.png.",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(IMG / "wave_cohort_aware.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    _MANIFEST["stage2"] = {
        "mc_seed": MC_SEED, "n_mc": N_MC,
        "h60_clean": float(h_good[NB - 1]),
        "h60_clean_used_by_projection": False,
        "tenure_mean_exit_age": float(tenure_curve_mean_exit_age()),
        "tenure_ramp_start": H60_RAMP,
        "tenure_ramp_width": max(LEGAL_GAP - H60_RAMP, 1),
        "mc_first_year_spread": float(w1.width),
        "mc_widest_year_relative": int(wmax.year),
        "mc_widest_pct_of_median": float(wmax.width_pct_of_median),
        "cum_p10": float(c10), "cum_p50": float(c50), "cum_p90": float(c90),
        "cum_p50_is_tenure_path": True,
        "specification_comparison": spec_cmp,
    }
    print("\nResults ->", RES, "\nFigures ->", IMG)


# ============================================================================
# STAGE 3 -- AGE-STRUCTURE DIAGNOSTICS
# ============================================================================

def first_existing(cands):
    for p in cands:
        if Path(p).exists():
            return Path(p)
    return None


def find_col(cols, *keywords):
    low = {c: str(c).lower() for c in cols}
    for kw in keywords:
        for c in cols:
            if kw in low[c]:
                return c
    return None


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None


def _num(d, col):
    return pd.to_numeric(d[col], errors="coerce").to_numpy() if col else None


def anchor_forecast(hist_last, fc_raw):
    fc_raw = np.asarray(fc_raw, float)
    if not np.isfinite(fc_raw).any() or hist_last is None or np.isnan(hist_last):
        return fc_raw, 1.0
    first = fc_raw[np.isfinite(fc_raw)][0]
    k = hist_last / first if first else 1.0
    return fc_raw * k, k


def load_panel():
    df = pd.read_excel(PANEL, sheet_name="historical", engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df = df.dropna(subset=["year_start"]); df["year_start"] = df["year_start"].astype(int)
    df = df[df.year_start >= FIRST_REAL_YEAR]
    return df


def national_by_year(df):
    years = sorted(df.year_start.unique())
    nat, students, tt = {}, {}, {}
    has_students = "students_total" in df.columns
    for y in years:
        s = df[df.year_start == y]
        nat[y] = np.array([float(pd.to_numeric(s[f"teachers_{b}"], errors="coerce").sum())
                           for b in BANDS])
        tt[y] = float(pd.to_numeric(s["teachers_total"], errors="coerce").sum()) \
            if "teachers_total" in df.columns else nat[y].sum()
        students[y] = float(pd.to_numeric(s["students_total"], errors="coerce").sum()) \
            if has_students else np.nan
    return years, nat, students, tt


def coverage_factor(years, nat, tt):
    facs = []
    for y in years[-3:]:
        ab = nat[y].sum()
        if ab > 0 and not np.isnan(tt[y]):
            facs.append(tt[y] / ab)
    return float(np.mean(facs)) if facs else 1.0


def load_supply_fc():
    path = first_existing(SUPPLY_FC_CANDIDATES)
    if path is None:
        for d in SEARCH_DIRS:
            for g in glob.glob(str(Path(d) / "*.csv")):
                cols = pd.read_csv(g, nrows=1).columns
                if find_col(cols, "recruit") or find_col(cols, "stock"):
                    path = Path(g); break
            if path:
                break
    if path is None:
        print("[fc supply] none found -> history only"); return None
    d = read_csv_safe(path)
    if d is None or d.empty:
        return None
    yc = find_col(d.columns, "year", "ano")
    if yc is None:
        print(f"[fc supply] no year column in {path.name}"); return None
    print(f"[fc supply] using {path.name}")
    return {"year": pd.to_numeric(d[yc], errors="coerce").astype("Int64").to_numpy(),
            "entries": _num(d, find_col(d.columns, "recruit", "entrada", "entries")),
            "exits": _num(d, find_col(d.columns, "exit", "saida", "saidas")),
            "stock": _num(d, find_col(d.columns, "stock"))}


def load_students_fc():
    """[SUP-12] Read students from the per-cell file, which is the one that has them."""
    path = first_existing(STUDENTS_FC_CANDIDATES)
    if path is None:
        for d in SEARCH_DIRS:
            for g in glob.glob(str(Path(d) / "*.csv")):
                cols = pd.read_csv(g, nrows=1).columns
                if find_col(cols, "student", "alun"):
                    path = Path(g); break
            if path:
                break
    if path is None:
        print("[fc students] none found -> ratio forecast omitted"); return None, None
    d = (pd.read_excel(path, engine="openpyxl")
         if str(path).lower().endswith(("xlsx", "xls")) else read_csv_safe(path))
    if d is None or d.empty:
        return None, None
    d.columns = [str(c).strip() for c in d.columns]
    yc = find_col(d.columns, "year", "ano")
    stu = find_col(d.columns, "student", "alun")
    if yc is None or stu is None:
        print(f"[fc students] no student column in {path.name}; "
              f"columns = {list(d.columns)[:12]}")
        return None, None
    long_by_cell = "cell" in d.columns
    if long_by_cell:
        print(f"[fc students] {path.name} is long by cell; summing across cells")
    sc = find_col(d.columns, "scenario", "cenario")
    if sc and "central" in set(d[sc].astype(str).str.lower()):
        d = d[d[sc].astype(str).str.lower() == "central"]
    d = d.copy(); d["_y"] = pd.to_numeric(d[yc], errors="coerce")
    total = {int(k): float(v) for k, v in d.groupby("_y")[stu].sum().items()
             if pd.notna(k)}

    levels = None
    if long_by_cell:
        cell_map = {"Pre-primary": ["pre_school"],
                    "Primary (1st cycle)": ["basic_1"],
                    "2nd cycle": ["basic_2"],
                    "3rd cycle + secondary": ["basic_3_secondary", "basic_3", "secondary"]}
        levels = {}
        present = set(d["cell"].astype(str))
        for name, keys in cell_map.items():
            sel = d[d["cell"].astype(str).isin([k for k in keys if k in present])]
            if not sel.empty:
                levels[name] = {int(k): float(v)
                                for k, v in sel.groupby("_y")[stu].sum().items()
                                if pd.notna(k)}
        if not levels:
            levels = None
        print(f"[fc students] using {path.name} "
              f"({'with level breakdown' if levels else 'total only'})")
        return total, levels

    lvl_map = {
        "Pre-primary": find_col(d.columns, "pre_school", "pre-school", "preescolar"),
        "Primary (1st cycle)": find_col(d.columns, "basic_1", "1_ciclo", "primar"),
        "2nd cycle": find_col(d.columns, "basic_2", "2_ciclo"),
    }
    b3 = find_col(d.columns, "basic_3", "3_ciclo")
    sec = find_col(d.columns, "secondary", "secundario")
    if any(lvl_map.values()) or b3 or sec:
        levels = {}
        for name, c in lvl_map.items():
            if c:
                levels[name] = {int(k): float(v)
                                for k, v in d.groupby("_y")[c].sum().items()
                                if pd.notna(k)}
        if b3 or sec:
            g3 = d.groupby("_y")[[c for c in [b3, sec] if c]].sum().sum(axis=1)
            levels["3rd cycle + secondary"] = {int(k): float(v)
                                               for k, v in g3.items() if pd.notna(k)}
        print(f"[fc students] using {path.name} (with level breakdown)")
    else:
        print(f"[fc students] using {path.name} (total only)")
    return total, levels


def entries_exits_by_year(years, nat):
    """[SUP-26][SUP-32][SUP-33] Entries AND exits on the same gross basis."""
    ys = flow_years(years)
    yrs, ent, ext = [], [], []
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        te = sum(_gross_entries(n0, n1, a, band_floor(a)) for a in range(NB - 1))
        tx = sum(_gross_exits(n0, n1, a, band_floor(a)) for a in range(NB))
        yrs.append(t1); ent.append(te); ext.append(tx)
    return np.array(yrs), np.array(ent), np.array(ext)


def young_entries(years, nat):
    """[SUP-16][SUP-33] Same definition AND same window as entries_history."""
    ys = flow_years(years)
    rows = []
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        rows.append({"year": t1, "stock_lt25": n1[0],
                     "entries_lt25": _gross_entries(n0, n1, 0, band_floor(0))})
    return pd.DataFrame(rows)


def exit_hazards(years, nat):
    """[SUP-33][SUP-35] Same window AND same helper as the hazard estimates."""
    ys = flow_years(years)
    rows = []
    for t, t1 in zip(ys[:-1], ys[1:]):
        if t1 - t != 1:
            continue
        n0, n1 = nat[t], nat[t1]
        rows.append({"year": t1,
                     "h60": _implied_hazard(n0, n1, NB - 1),
                     "h5559": _implied_hazard(n0, n1, NB - 2),
                     "n60_denominator": nat[t][NB - 1]})
    return pd.DataFrame(rows)


def level_ratios_hist(df, years):
    out = {name: [] for name, *_ in LEVELS}
    for y in years:
        s = df[df.year_start == y]
        for name, scol, tcol, _c in LEVELS:
            if isinstance(scol, tuple):
                stu = sum(pd.to_numeric(s[c], errors="coerce").sum() for c in scol)
            else:
                stu = pd.to_numeric(s[scol], errors="coerce").sum()
            tea = pd.to_numeric(s[tcol], errors="coerce").sum()
            out[name].append(stu / tea if tea else np.nan)
    return out


# --------------------------------------------------------- figures (stage 3)
def fig_age_lines(years, nat):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    cmap = [mpl.colors.to_hex(c) for c in plt.cm.viridis(np.linspace(0, 1, len(years)))]
    x = np.arange(NB)
    for c, y in zip(cmap, years):
        ax.plot(x, nat[y], "-", color=c, lw=1.8, alpha=0.85)
    ax.plot(x, nat[years[0]], "o-", color=cmap[0], lw=2.6, ms=6, label=f"{years[0]}")
    ax.plot(x, nat[years[-1]], "s-", color=cmap[-1], lw=2.6, ms=6, label=f"{years[-1]}")
    ax.set_xticks(x); ax.set_xticklabels(NICE, rotation=30, ha="right")
    ax.set_title("Teacher age distribution over time (one line per year)",
                 fontweight="bold")
    ax.set_xlabel("age band"); ax.set_ylabel("teachers (national headcount)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(years[0], years[-1]))
    fig.colorbar(sm, ax=ax, label="year")
    style(ax); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(IMG / "age_distribution_lines.png", dpi=160,
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_age_heatmap(years, nat):
    fig, ax = plt.subplots(figsize=(12, 6))
    Z = np.array([nat[y] for y in years])
    im = ax.imshow(Z, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_xticks(range(NB)); ax.set_xticklabels(NICE, rotation=30, ha="right")
    ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
    ax.set_title("Teacher age structure over time -- the bulge moves right",
                 fontweight="bold")
    ax.set_xlabel("age band"); ax.set_ylabel("year")
    fig.colorbar(im, ax=ax, label="teachers")
    fig.tight_layout(); fig.savefig(IMG / "age_distribution_heatmap.png", dpi=160,
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_age_stacked(years, nat):
    fig, ax = plt.subplots(figsize=(12, 6))
    shares = np.array([nat[y] / nat[y].sum() for y in years]) * 100
    cols = [mpl.colors.to_hex(c) for c in plt.cm.RdYlBu_r(np.linspace(0, 1, NB))]
    ax.stackplot(years, shares.T, labels=NICE, colors=cols, alpha=0.9)
    ax.set_title("Share of teachers by age band over time (%)", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("share of workforce (%)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylim(0, 100); ax.margins(x=0)
    fig.tight_layout(); fig.savefig(IMG / "age_distribution_stacked.png", dpi=160,
                                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mosaic(years, series_by_band, title, fname, diff=False, note=None):
    ncols = 3; nrows = int(np.ceil(NB / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.0 * nrows),
                             sharex=True, squeeze=False)
    ax = axes.flatten()
    for i in range(NB):
        a = ax[i]; v = series_by_band[i]
        yr = years[1:] if diff else years
        if diff:
            a.bar(yr, v, color=[ENTRY if x >= 0 else EXITC for x in v], alpha=0.8)
            a.axhline(0, color="0.5", lw=0.8)
        else:
            a.plot(yr, v, "-o", color=AGEING, lw=2.0, ms=4)
            a.fill_between(yr, min(v) * 0.98, v, color=AGEING, alpha=0.06)
            d0, d1 = v[0], v[-1]
            a.annotate(f"{d0:,.0f}", (yr[0], d0), textcoords="offset points",
                       xytext=(0, 6), fontsize=7, ha="center")
            if d0:
                a.annotate(f"{d1:,.0f} ({100*(d1-d0)/d0:+.0f}%)", (yr[-1], d1),
                           textcoords="offset points", xytext=(-6, -12),
                           fontsize=7, ha="right")
        a.set_title(NICE[i], fontsize=11, fontweight="bold")
        a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        style(a); a.margins(y=0.18)
    for a in ax[NB:]:
        a.set_visible(False)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8.5, color="#555")
    fig.supxlabel("year", fontsize=10); fig.supylabel("teachers", fontsize=10)
    fig.tight_layout(rect=[0.01, 0.03 if note else 0.01, 1, 0.96])
    fig.savefig(IMG / fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_mosaic_level(years, nat, cohort_note=None):
    """[SUP-64] The footnote now reflects what the net trace actually found."""
    series = [np.array([nat[y][i] for y in years]) for i in range(NB)]
    note = cohort_note or ("Falling headcounts mix cohorts ageing OUT of a band with "
                           "entrants ageing IN: see cohort_trace.csv for the "
                           "multiplier net of arriving entrants [SUP-64].")
    _mosaic(years, series, "Teacher age structure -- level by band",
            "age_bracket_mosaic.png", note=note)


def fig_mosaic_diff(years, nat):
    series = [np.diff(np.array([nat[y][i] for y in years])) for i in range(NB)]
    _mosaic(years, series, "Teacher age structure -- year-on-year change by band",
            "age_bracket_mosaic_diff.png", diff=True)


def fig_entries_lt25(d):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(d.year, d.entries_lt25, "o-", color=ENTRY, lw=2.6, ms=7,
            label="entries into <25")
    d = d.copy(); d["e3y"] = d["entries_lt25"].rolling(3, min_periods=1).mean()
    ax.plot(d.year, d.e3y, ":", color="#0b5a30", lw=2, label="3-year rolling mean")
    m = d.entries_lt25.mean(); l3 = d.entries_lt25.tail(3).mean()
    ax.axhline(m, color=ENTRY, lw=1, alpha=0.4)
    ax.axhline(l3, color="#333", lw=1, ls="--", alpha=0.6)
    ax.text(d.year.min(), m * 1.03, f"full-series mean = {m:,.0f}", fontsize=9, color=ENTRY)
    ax.text(d.year.min(), l3 * 0.90, f"last-3-year mean = {l3:,.0f}", fontsize=9,
            color="#333")
    for _, r in d.iterrows():
        ax.annotate(f"{r.entries_lt25:,.0f}", (r.year, r.entries_lt25),
                    textcoords="offset points", xytext=(0, 9), fontsize=8, ha="center")
    ax.set_title(f"New entries into teaching (<25 band), "
                 f"{int(d.year.min())}-{int(d.year.max())}", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("new young teachers entering (national)")
    style(ax); ax.legend(frameon=False); ax.set_ylim(0, None)
    fig.text(0.5, 0.005, "Same gross definition and same year window as "
             "supply_entries_history.csv.", ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(IMG / "entries_lt25_by_year.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_exit_hazard(d):
    """[SUP-56][SUP-65] Say when the 55-59 series is constant by construction."""
    degen = series_is_degenerate(d["h5559"])

    # [SUP-65] plain if/else: the label was previously built in a multi-line
    # conditional expression that an unrelated reindent could silently break.
    if degen:
        lbl5559 = (f"55-59: pinned at the {band_floor(NB-2):.3f} floor "
                   f"(no information)")
    else:
        lbl5559 = f"55-59 exit hazard (floor {band_floor(NB-2):.3f})"

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(d.year, d.h60, "o-", color=EXITC, lw=2.4, ms=7, label="60+ exit hazard")
    ax.plot(d.year, d.h5559, "s--", color=MIDC, lw=2, ms=6, label=lbl5559)
    d = d.copy(); d["h60_3y"] = d["h60"].rolling(3, min_periods=1).mean()
    ax.plot(d.year, d.h60_3y, ":", color="#7a0f14", lw=2,
            label="60+ (3-year rolling mean)")
    full = d.h60.mean(); last3 = d.h60.tail(3).mean()
    ax.axhline(full, color=EXITC, lw=1, alpha=0.4)
    ax.axhline(last3, color="#333", lw=1, ls="--", alpha=0.6)
    live = (tenure_curve_mean_exit_age() if H60_MODE == "tenure"
            else implied_mean_exit_age(H60_SCENARIOS["base"]))
    if H60_MODE != "tenure":
        ax.axhline(H60_SCENARIOS["base"], color=AGEING, lw=1.6, ls="-.",
                   label=f"model h60 = {H60_SCENARIOS['base']:.4f} "
                         f"(exit age ~{live:.1f})")
    ax.text(d.year.min(), full + 0.005,
            f"estimation-window mean = {full:.3f}", fontsize=9, color=EXITC)
    ax.text(d.year.min(), last3 - 0.015, f"last-3-year mean = {last3:.3f}",
            fontsize=9, color="#333")
    ax.set_title(f"Annual exit hazard of the 60+ and 55-59 cohorts "
                 f"({int(d.year.min())}-{int(d.year.max())})", fontweight="bold")
    ax.set_xlabel("year"); ax.set_ylabel("share of the cohort exiting that year")
    style(ax); ax.legend(frameon=False); ax.set_ylim(0, None)
    note = ""
    if "n60_denominator" in d.columns:
        n0, n1 = d.n60_denominator.iloc[0], d.n60_denominator.iloc[-1]
        note = (f"The 60+ fall is real behaviour, but the band is still filling -- "
                f"stock grew {n0:,.0f} to {n1:,.0f} and sits below its own steady "
                f"state, so exits/stock understates the structural hazard.")
    if degen:
        note += ("\nThe 55-59 line is flat by construction: after [SUP-36] the "
                 "identity returns max(floor*n0, -net) and net never falls below "
                 "-floor*n0, so it measures nothing.")
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#555")
        fig.tight_layout(rect=[0, 0.05, 1, 1])
    else:
        fig.tight_layout()
    fig.savefig(IMG / "exit_hazard_60_5559.png", dpi=160,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_entries_exits_ratio(df, years, nat, students, tt, sup, stu_total_fc,
                            stu_levels_fc):
    """[SUP-37][SUP-45] Three panels, one shared x-axis, windows computed."""
    hy, he, hx = entries_exits_by_year(years, nat)
    lvl_hist = level_ratios_hist(df, years)
    cov = coverage_factor(years, nat, tt)
    fy = sup["year"].astype(float) if sup else None
    ry = list(years)
    if len(hy) == 0:
        print("[fig] entries/exits: no flow years, figure skipped")
        return

    flow_span = f"{int(hy.min())}-{int(hy.max())}"
    stock_span = f"{ry[0]}-{ry[-1]}"
    gap_yrs = int(hy.min()) - int(ry[0])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 6.6), sharex=True)

    ax1.plot(hy, he, "-o", color=ENTRY, lw=2.4, ms=6, label="entries (history)")
    ax1.plot(hy, hx, "-s", color=EXITC, lw=2.4, ms=6, label="exits (history)")
    ax1.fill_between(hy, hx, he, where=(he >= hx), color=ENTRY, alpha=0.08)
    ax1.fill_between(hy, hx, he, where=(he < hx), color=EXITC, alpha=0.08)
    if sup and sup.get("entries") is not None:
        ax1.plot(np.r_[hy[-1], fy], np.r_[he[-1], sup["entries"]], "--o",
                 color=ENTRY, lw=2.0, ms=4, alpha=0.85, label="entries (forecast)")
        if sup.get("exits") is not None:
            ax1.plot(np.r_[hy[-1], fy], np.r_[hx[-1], sup["exits"]], "--s",
                     color=EXITC, lw=2.0, ms=4, alpha=0.85, label="exits (forecast)")
    ax1.set_title(f"Teacher entries vs. exits\nhistory {flow_span} "
                  f"({gap_yrs} years shorter than the stock series)",
                  fontweight="bold", fontsize=11)
    ax1.set_ylabel("teachers per year")
    ax1.legend(frameon=False, fontsize=8.5, loc="lower right")

    rr = [students[y] / tt[y] if (tt[y] and not np.isnan(students[y])) else np.nan
          for y in ry]
    have = np.any(~np.isnan(rr))
    if have:
        ax2.plot(ry, rr, "-o", color=AGEING, lw=2.4, ms=6, label="ratio (history)")
        m = np.nanmean(rr); ax2.axhline(m, color=AGEING, ls="--", lw=1, alpha=0.5)
        ax2.text(ry[0], m, f" history mean = {m:.1f}", fontsize=8, color=AGEING,
                 va="bottom")
    if sup and sup.get("stock") is not None and stu_total_fc:
        raw = np.array([stu_total_fc.get(int(y), np.nan) / (s * cov)
                        if (s and not np.isnan(s)) else np.nan
                        for y, s in zip(fy, sup["stock"])])
        rr_fc, _ = anchor_forecast(rr[-1] if have else None, raw)
        ay = [ry[-1]] if have else []; ar = [rr[-1]] if have else []
        ax2.plot(np.r_[ay, fy], np.r_[ar, rr_fc], "--o", color=AGEING, lw=2.0, ms=4,
                 alpha=0.85, label="ratio (forecast)")
    ax2.set_title(f"Student/teacher ratio: aggregate\nhistory {stock_span} "
                  f"(stock series)", fontweight="bold", fontsize=11)
    ax2.set_ylabel("students per teacher")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(frameon=False, fontsize=8.5, loc="upper right")

    last = df[df.year_start == years[-1]]
    for name, scol, tcol, col in LEVELS:
        hist = np.array(lvl_hist[name], float)
        ax3.plot(ry, hist, "-o", color=col, lw=2.2, ms=5, label=name)
        if (fy is not None and stu_levels_fc and name in stu_levels_fc
                and sup.get("stock") is not None and tcol in last.columns):
            denom = pd.to_numeric(last["teachers_total"], errors="coerce").sum()
            share = (pd.to_numeric(last[tcol], errors="coerce").sum() / denom) \
                if denom else np.nan
            if np.isfinite(share) and share > 0:
                raw = np.array([stu_levels_fc[name].get(int(y), np.nan)
                                / (s * cov * share)
                                if (s and not np.isnan(s)) else np.nan
                                for y, s in zip(fy, sup["stock"])])
                fc, _ = anchor_forecast(hist[-1], raw)
                ax3.plot(np.r_[ry[-1], fy], np.r_[hist[-1], fc], "--o",
                         color=col, lw=1.8, ms=3.5, alpha=0.8)
    ax3.set_title(f"Student/teacher ratio, by education level\nhistory {stock_span}",
                  fontweight="bold", fontsize=11)
    ax3.set_ylabel("students per teacher")
    ax3.legend(frameon=False, fontsize=8.5, loc="upper left",
               bbox_to_anchor=(0.0, -0.13), ncol=2)

    for a in (ax1, ax2, ax3):
        a.axvline(SPLIT, color="0.5", ls=":", lw=1.2)
        a.text(SPLIT, a.get_ylim()[1], " forecast", color="0.4", fontsize=8.5, va="top")
        a.set_xlabel("year")
        style(a)

    fig.suptitle("Teacher flows and student/teacher ratios -- history and forecast",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005,
             "History and forecast use the same gross accounting: entries - exits "
             "reproduces the observed change in stock exactly, and hazard x stock "
             "reproduces exits band by band.\n"
             f"The flow panel starts in {int(hy.min())} and the ratio panels in "
             f"{ry[0]}, a gap of {gap_yrs} years: flows are estimated from "
             f"{FIRST_FLOW_YEAR} onwards and each one needs two consecutive stock "
             "observations.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(IMG / "entries_exits_ratio.png", dpi=155, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def stage_diagnostics():
    """STAGE 3 -- descriptive age-structure diagnostics (+ optional forecast overlay)."""
    print("=" * 74)
    print("STAGE 3 -- age-structure diagnostics")
    print("=" * 74)
    df = load_panel()
    years, nat, students, tt = national_by_year(df)

    M = pd.DataFrame({NICE[i]: [nat[y][i] for y in years] for i in range(NB)},
                     index=years)
    M.index.name = "year"; M.to_csv(RES / "age_distribution_by_year.csv")

    d_ent = young_entries(years, nat)
    d_ent.to_csv(RES / "entries_lt25_by_year.csv", index=False)
    d_haz = exit_hazards(years, nat)
    d_haz.to_csv(RES / "exit_hazard_by_year.csv", index=False)

    f0, f1, gap_yrs = flow_window(years)
    print(f"[window] stock series (age_distribution_by_year): {years[0]}-{years[-1]}")
    print(f"[window] flow series (entries, exits, hazards)  : {f0}-{f1}")
    print(f"[window] the flow series is {gap_yrs} years shorter: FIRST_FLOW_YEAR "
          f"drops one and the first transition consumes another.")

    # [SUP-49] assert the prose matches the data
    assert (f0, f1) == (int(d_ent.year.min()), int(d_ent.year.max())), \
        "flow_window disagrees with the written entry series"
    assert gap_yrs == f0 - int(years[0]), "printed gap does not match the data"

    eh = entries_history(nat, years)
    if not eh.empty and not d_ent.empty:
        j = d_ent.merge(eh[["year", "new_entrants_lt25"]], on="year", how="outer",
                        indicator=True)
        missing = int((j["_merge"] != "both").sum())
        gap = (j["entries_lt25"] - j["new_entrants_lt25"]).abs().max()
        ok = (missing == 0) and (gap < 1e-6)
        print(f"[check] entries_lt25_by_year vs supply_entries_history: "
              f"rows in only one series = {missing}, max abs difference = {gap:.6f} "
              f"({'consistent' if ok else 'INCONSISTENT'})")

    sup = load_supply_fc()
    stu_total_fc, stu_levels_fc = load_students_fc()

    # [SUP-64] carry the cohort-trace verdict into the mosaic footnote
    st1 = _MANIFEST.get("stage1", {})
    mn_n = st1.get("cohort_trace_min_net_multiplier")
    if mn_n is not None:
        if mn_n >= COHORT_SHRINK_TOL:
            cohort_note = ("Falling headcounts are cohorts ageing OUT of a band: net "
                           f"of arriving entrants the smallest cohort multiplier is "
                           f"{mn_n:.2f}x, so no cohort shrank [SUP-64].")
        else:
            cohort_note = ("Falling headcounts are NOT purely composition: net of "
                           f"arriving entrants the smallest cohort multiplier is "
                           f"{mn_n:.2f}x, i.e. at least one cohort shrank [SUP-64]. "
                           "See cohort_trace.csv.")
    else:
        cohort_note = None

    figs = [fig_age_lines, fig_age_heatmap, fig_age_stacked, fig_mosaic_diff]
    for f in figs:
        f(years, nat)
    fig_mosaic_level(years, nat, cohort_note=cohort_note)
    fig_entries_lt25(d_ent)
    fig_exit_hazard(d_haz)
    fig_entries_exits_ratio(df, years, nat, students, tt, sup, stu_total_fc,
                            stu_levels_fc)
    n_figs = len(figs) + 4

    last3 = d_haz.h60.tail(3).mean()
    full = d_haz.h60.mean()
    h5559_degen = series_is_degenerate(d_haz["h5559"])
    print(f"\n[done] {n_figs} figures written to", IMG)
    print("[done] CSVs written to", RES)
    print(f"[hazard] 60+ estimation-window mean = {full:.4f} | last-3-year mean = "
          f"{last3:.4f}")
    if H60_MODE == "tenure":
        print(f"[hazard] the projection runs the tenure queue "
              f"(exit age {tenure_curve_mean_exit_age():.1f}), so there is no single")
        print(f"         model h60 to compare against.")
    else:
        print(f"[hazard] model h60 = {H60_SCENARIOS['base']:.4f} "
              f"(exit age {implied_mean_exit_age(H60_SCENARIOS['base']):.1f})")
    if h5559_degen:
        print(f"[hazard] 55-59 is pinned at the {band_floor(NB-2):.4f} floor in every "
              f"year: that series carries no behavioural information [SUP-56].")
    _MANIFEST["stage3"] = {"h60_observed_last3": float(last3),
                           "h60_observed_window_mean": float(full),
                           "h5559_series_degenerate": bool(h5559_degen),
                           "stock_window": [int(years[0]), int(years[-1])],
                           "flow_window": [f0, f1],
                           "stock_minus_flow_gap_years": gap_yrs,
                           "n_figures": n_figs}


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def main():
    # [SUP-31] Stage 1 is a hard precondition: it resolves H60_SCENARIOS["base"].
    base_wave, entry_profile, share_stats, s1_inputs = stage_supply()
    print()

    # [SUP-59] base_wave is no longer passed to stage 2: that stage reads nothing
    # from stage 1's wave.
    for name, fn in [("ROBUSTNESS",
                      lambda: stage_robustness(entry_profile=entry_profile,
                                               stage1_inputs=s1_inputs)),
                     ("DIAGNOSTICS", stage_diagnostics)]:
        try:
            fn()
        except Exception as e:
            print(f"[error] stage {name} failed: {type(e).__name__}: {e}")
        print()

    with open(RES / "supply_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(_MANIFEST, f, indent=2, default=str)
    print("[manifest] supply_run_manifest.json")
    print("\nKEY ASSUMPTIONS CARRIED FORWARD TO 04_gap:")
    print("  1. Entry is UNCONSTRAINED in this block. The qualifying-pool ceiling")
    print("     belongs in 04_gap and should bind on the <30 channel only; see")
    print("     supply_recruitment_by_channel.csv.")
    print("  2. The lateral channel is the LARGER of the two and is drawn from the")
    print("     contracted reserve, not from initial teacher education. That reserve")
    print("     shrinks as the workforce shrinks and needs a stock model of its own.")
    if share_stats:
        print("  3. Regional young share varies far more than the national average")
        print("     suggests:")
        for s in share_stats:
            print(f"       {s['scope']:<6} {100*s['min']:.1f}% ({s['min_region']}) to "
                  f"{100*s['max']:.1f}% ({s['max_region']}) = {s['ratio']:.1f}x.")
        worst = min(share_stats, key=lambda s: s["min"])
        if worst["min"] < 0.05:
            print(f"     {worst['min_region']} models "
                  f"{100*(1-worst['min']):.0f}% of its entries as lateral, so a "
                  f"graduate ceiling would barely bind there. Confirm that split")
            print("     before 04_gap relies on it.")
    else:
        print("  3. Regional young-share spread unavailable (regional scopes skipped).")
    sc = _MANIFEST.get("stage2", {}).get("specification_comparison")
    if sc:
        # [SUP-63] report the decomposition, not the conflated total
        print(f"  4. H60_MODE = {H60_MODE} is live. Flat vs tenure queue differs by "
              f"{sc['cum_gap_pct']:+.1f}% in cumulative")
        print(f"     recruitment, which DECOMPOSES into {sc['form_effect_pct']:+.1f}% "
              f"from the hazard shape and")
        print(f"     {sc['age_effect_pct']:+.1f}% from the retirement age "
              f"({sc['flat_mean_exit_age']:.1f} vs "
              f"{sc['tenure_mean_exit_age']:.1f} years). The dominant channel is")
        print(f"     {sc['dominant_channel'].upper()}.", end="")
        if sc.get("effects_have_opposite_signs"):
            print(" The two effects have OPPOSITE signs")
            print("     and partly cancel, so the total understates both.", end="")
        print(" All three paths are in")
        print("     h60_specification_comparison.csv [SUP-63].")
    else:
        print(f"  4. H60_MODE = {H60_MODE} is wired into the projection.")
    print("  5. Every historical flow series, the hazard estimator, the 60+ band-fill")
    print("     diagnostic and the projection share one arithmetic: entries - exits =")
    print("     change in stock, and hazard x stock = gross exits, band by band.")
    print("     All residuals are asserted above [SUP-58].")
    if _MANIFEST.get("stage1", {}).get("h5559_series_degenerate"):
        print("  6. The 55-59 hazard is an ASSUMPTION, not an estimate: the observed")
        print("     series is pinned at the attrition floor in every year [SUP-56].")
    st1 = _MANIFEST.get("stage1", {})
    if "cohort_trace_min_net_multiplier" in st1:
        mn_n = st1["cohort_trace_min_net_multiplier"]
        print("  7. The cohort trace subtracts arriving entrants, located by FORWARD")
        print("     PROPAGATION through the model's own ageing arithmetic [SUP-64].")
        print(f"     Smallest multiplier: gross "
              f"{st1['cohort_trace_min_gross_multiplier']:.2f}x, net {mn_n:.2f}x "
              f"({st1['cohort_trace_entrants_subtracted']:,.0f} entrants")
        print(f"     still present out of {st1['cohort_trace_entries_injected']:,.0f} "
              f"injected). Entrants are assumed to face the SAME")
        print("     band hazards as incumbents; that is behavioural, not an identity.")
        if mn_n < COHORT_SHRINK_TOL:
            print("     At least one cohort SHRANK: do not repeat the claim that falling")
            print("     band headcounts are pure composition.")


if __name__ == "__main__":
    main()
