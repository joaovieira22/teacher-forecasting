
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting\data")
PROCESSED_DIR = BASE_DIR / "02_processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

POP_PROJ_FILE = PROCESSED_DIR / "population_projections_nuts2.xlsx"
AGEING_FILE = PROCESSED_DIR / "ageing_index_nuts3.xlsx"
RENEWAL_FILE = PROCESSED_DIR / "renewal_index_nuts3.xlsx"
OUT_FILE = PROCESSED_DIR / "population_projections_nuts3.xlsx"

# ============================================================
# CONFIG
# ============================================================
VALID_NUTS2 = {"11", "15", "19", "1A", "1B", "1C", "1D", "20", "30"}

BASE_AGE_BANDS = [
    "pop_total",
    "pop_0_4", "pop_5_9", "pop_10_14", "pop_15_19", "pop_20_24",
    "pop_25_29", "pop_30_34", "pop_35_39", "pop_40_44", "pop_45_49",
    "pop_50_54", "pop_55_59", "pop_60_64", "pop_65_69", "pop_70_74",
    "pop_75_79", "pop_80_84", "pop_85_plus",
]
DERIVED_BANDS = ["pop_0_14", "pop_15_64", "pop_65_plus"]
POP_VARS = BASE_AGE_BANDS + DERIVED_BANDS

# Sensibilidade da quota aos indices (z-scored). As coortes jovens encolhem
# onde o envelhecimento e alto e crescem onde a renovacao e alta; as idosas
# fazem o contrario. Os sinais codificam o mecanismo; as magnitudes sao
# deliberadamente conservadoras.
ALPHA_YOUNG = 0.22    # 0-14
ALPHA_TEEN = 0.16     # 15-29
ALPHA_WORKING = 0.10  # 30-64
ALPHA_OLD = 0.18      # 65+

YEAR_TREND_STRENGTH = 0.015  # quanto o tilt cresce por ano de horizonte
MIN_WEIGHT = 1e-12
RANDOM_SEED = 20260818

# ============================================================
# HELPERS
# ============================================================
def code(x):
    if pd.isna(x):
        return ""
    s = re.sub(r"\.0$", "", str(x).strip())
    return s.replace(" ", "")


def num(s):
    return pd.to_numeric(s, errors="coerce")


def largest_remainder(values, target_total):
    """Arredonda para inteiros nao-negativos que somam exactamente o alvo.

    Metodo dos maiores restos (Hamilton): arredonda todos por defeito e
    distribui as unidades que faltam pelas maiores partes fraccionarias.
    """
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    target = int(round(float(target_total)))

    if values.sum() == 0:
        out = np.zeros(len(values), dtype=int)
        if len(out) and target != 0:
            out[0] = target
        return out

    floors = np.floor(values).astype(int)
    remainder = target - int(floors.sum())

    if remainder > 0:
        order = np.argsort(-(values - floors))
        chosen = order[0:remainder]
        floors[chosen] += 1
    elif remainder < 0:
        order = np.argsort(values - floors)
        need = -remainder
        for i in order:
            if need == 0:
                break
            if floors[i] > 0:
                floors[i] -= 1
                need -= 1
    return floors.astype(int)


def zscore_within_group(df, value_col, group_cols, out_col):
    """Z-score dentro de cada grupo (0 se o grupo nao tiver variancia)."""
    def _z(x):
        vals = x.astype(float)
        sd = vals.std(skipna=True)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(vals)), index=x.index)
        return (vals - vals.mean(skipna=True)) / sd

    df = df.copy()
    df[out_col] = df.groupby(group_cols)[value_col].transform(_z).fillna(0.0)
    return df


def add_derived_bands(df):
    """Calcula pop_0_14 / pop_15_64 / pop_65_plus se faltarem."""
    df = df.copy()
    if "pop_0_14" not in df.columns:
        cols = ["pop_0_4", "pop_5_9", "pop_10_14"]
        df["pop_0_14"] = df[cols].sum(axis=1, min_count=1)
    if "pop_15_64" not in df.columns:
        cols = [
            "pop_15_19", "pop_20_24", "pop_25_29", "pop_30_34", "pop_35_39",
            "pop_40_44", "pop_45_49", "pop_50_54", "pop_55_59", "pop_60_64",
        ]
        df["pop_15_64"] = df[cols].sum(axis=1, min_count=1)
    if "pop_65_plus" not in df.columns:
        cols = ["pop_65_69", "pop_70_74", "pop_75_79", "pop_80_84", "pop_85_plus"]
        df["pop_65_plus"] = df[cols].sum(axis=1, min_count=1)
    return df

# ============================================================
# LOAD PROJECTIONS (totais de controlo NUTS2)
# ============================================================
def load_population_projections(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Nao encontrei: {path}\n"
            "Corre primeiro o 05_build_population_projections.py."
        )
    xls = pd.ExcelFile(path, engine="openpyxl")
    if "panel_wide_all" in xls.sheet_names:
        sheet = "panel_wide_all"
    else:
        sheet = xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

    df = df.rename(columns={
        "location_code": "nuts2_code",
        "location": "nuts2_nome_proj",
    })
    df["year"] = df["year"].astype(int)
    df["nuts2_code"] = df["nuts2_code"].map(code)
    df["location_level"] = df["location_level"].astype(str).str.lower().str.strip()
    df["scenario"] = df["scenario"].astype(str).str.strip()

    is_n2 = df["location_level"].eq("nuts2")
    valid = df["nuts2_code"].isin(VALID_NUTS2)
    df = df[is_n2 & valid].copy()
    if df.empty:
        raise RuntimeError("Nao ha linhas NUTS2 no ficheiro de projeccoes.")

    for c in BASE_AGE_BANDS:
        if c not in df.columns:
            raise RuntimeError(f"Falta o escalao etario nas projeccoes: {c}")
        df[c] = num(df[c])

    return add_derived_bands(df)

# ============================================================
# LOAD INDICATORS (NUTS3 ja harmonizado, folha 'nuts3')
# ============================================================
def load_indicator_nuts3(path, mean_col, out_name):
    if not path.exists():
        raise FileNotFoundError(
            f"Nao encontrei: {path}\n"
            "Corre primeiro o 02_build_ageing_index.py e o 03_build_renewal_index.py."
        )
    xls = pd.ExcelFile(path, engine="openpyxl")
    if "nuts3" not in xls.sheet_names:
        raise RuntimeError(f"Folha 'nuts3' nao encontrada em {path.name}")
    df = pd.read_excel(path, sheet_name="nuts3", engine="openpyxl")

    if "ano" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"ano": "year"})

    required = ["year", "nuts2_code", "nuts3_code", mean_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Faltam colunas {missing} na folha 'nuts3' de {path.name}")

    df["year"] = df["year"].astype(int)
    df["nuts2_code"] = df["nuts2_code"].map(code)
    df["nuts3_code"] = df["nuts3_code"].map(code)
    df[out_name] = num(df[mean_col])

    keep = ["year", "nuts2_code", "nuts3_code"]
    for extra in ("nuts2_nome", "nuts3_nome"):
        if extra in df.columns:
            keep.append(extra)
    keep.append(out_name)
    return df[keep].drop_duplicates()


def build_indicator_base():
    ageing = load_indicator_nuts3(
        AGEING_FILE, "indice_envelhecimento_mean", "indice_envelhecimento_mean")
    renewal = load_indicator_nuts3(
        RENEWAL_FILE, "indice_renovacao_pop_ativa_mean",
        "indice_renovacao_pop_ativa_mean")

    ren_cols = ["year", "nuts2_code", "nuts3_code", "indice_renovacao_pop_ativa_mean"]
    base = ageing.merge(
        renewal[ren_cols],
        on=["year", "nuts2_code", "nuts3_code"],
        how="outer",
    )
    base = base[base["nuts2_code"].isin(VALID_NUTS2)].copy()

    name_cols = [c for c in ["nuts2_nome", "nuts3_nome"] if c in base.columns]
    geo_cols = ["nuts2_code", "nuts3_code"] + name_cols
    geo = (
        base.sort_values("year")
        .drop_duplicates(["nuts2_code", "nuts3_code"])
        [geo_cols]
        .sort_values(["nuts2_code", "nuts3_code"])
        .reset_index(drop=True)
    )
    if geo.empty:
        raise RuntimeError("Grelha NUTS3 vazia, construida a partir dos indicadores.")
    return base, geo

# ============================================================
# MODELO DINAMICO
# ============================================================
def prepare_indicator_panel(indicators, geo, target_years):
    """Grelha NUTS3 x ano de previsao, com indicadores propagados e z-scores.

    Os indicadores sao unidos sobre a UNIAO dos anos historicos e dos anos-alvo,
    propagados para a frente e para tras dentro de cada NUTS3, e so depois
    restringidos ao horizonte. Assim o ultimo valor estrutural conhecido entra
    sempre no horizonte, mesmo que os indicadores nao cubram o primeiro ano.
    """
    target_years = sorted(set(int(y) for y in target_years))
    ind_cols = ["indice_envelhecimento_mean", "indice_renovacao_pop_ativa_mean"]

    keep = ["year", "nuts2_code", "nuts3_code"] + ind_cols
    ind = indicators[keep].copy()
    ind["year"] = ind["year"].astype(int)

    all_years = sorted(set(ind["year"].tolist()) | set(target_years))
    years_df = pd.DataFrame({"year": all_years, "_key": 1})
    grid = geo.assign(_key=1).merge(years_df, on="_key").drop(columns="_key")

    panel = grid.merge(ind, on=["year", "nuts2_code", "nuts3_code"], how="left")
    panel = panel.sort_values(["nuts3_code", "year"])
    for c in ind_cols:
        panel[c] = panel.groupby("nuts3_code")[c].ffill().bfill()
        # ultimo recurso, se toda a serie de um NUTS3 estiver vazia
        panel[c] = panel[c].fillna(panel[c].median())

    panel = panel[panel["year"].isin(target_years)].reset_index(drop=True)

    panel = zscore_within_group(
        panel, "indice_envelhecimento_mean", ["year", "nuts2_code"], "z_ageing")
    panel = zscore_within_group(
        panel, "indice_renovacao_pop_ativa_mean", ["year", "nuts2_code"], "z_renewal")
    return panel


def score_for_pop_var(df, pop_var):
    """Tilt log-quota para um escalao etario, a partir dos indices z-scored."""
    young = ("pop_0_4", "pop_5_9", "pop_10_14", "pop_0_14")
    teen = ("pop_15_19", "pop_20_24", "pop_25_29")
    working = (
        "pop_30_34", "pop_35_39", "pop_40_44", "pop_45_49",
        "pop_50_54", "pop_55_59", "pop_60_64", "pop_15_64",
    )
    old = (
        "pop_65_69", "pop_70_74", "pop_75_79", "pop_80_84",
        "pop_85_plus", "pop_65_plus",
    )

    if pop_var in young:
        return -ALPHA_YOUNG * df["z_ageing"] + ALPHA_YOUNG * df["z_renewal"]
    if pop_var in teen:
        return -ALPHA_TEEN * df["z_ageing"] + ALPHA_TEEN * df["z_renewal"]
    if pop_var in working:
        return -0.04 * df["z_ageing"] + ALPHA_WORKING * df["z_renewal"]
    if pop_var in old:
        return ALPHA_OLD * df["z_ageing"] - 0.06 * df["z_renewal"]
    # pop_total e o resto: tilt suave
    return -0.02 * df["z_ageing"] + 0.04 * df["z_renewal"]


def allocate_nuts2_to_nuts3(proj, indicator_panel, pop_vars):
    """Reparte cada total NUTS2 pelos seus NUTS3, com reconciliacao exacta."""
    records = []
    share_records = []
    first_year = int(proj["year"].min())

    grouped = proj.groupby(["year", "scenario", "nuts2_code"])
    for keys, block in grouped:
        year, scenario, n2 = keys
        same_year = indicator_panel["year"] == year
        same_n2 = indicator_panel["nuts2_code"] == n2
        geo = indicator_panel[same_year & same_n2].copy()
        geo = geo.sort_values("nuts3_code").reset_index(drop=True)
        if geo.empty:
            continue

        target_row = block.iloc[0]
        horizon = max(int(year) - first_year, 0)
        dynamic_multiplier = 1.0 + YEAR_TREND_STRENGTH * horizon
        n = len(geo)
        base = np.ones(n, dtype=float) / n

        for pop_var in pop_vars:
            target = target_row.get(pop_var, np.nan)
            if pd.isna(target):
                continue

            score = score_for_pop_var(geo, pop_var).to_numpy(dtype=float)
            weights = base * np.exp(score * dynamic_multiplier)
            weights = np.where(np.isfinite(weights), weights, 0.0)
            weights = np.clip(weights, MIN_WEIGHT, None)
            weights = weights / weights.sum()

            values = largest_remainder(weights * float(target), target)

            for i, row in geo.iterrows():
                n2_nome = target_row.get("nuts2_nome_proj", row.get("nuts2_nome", ""))
                records.append({
                    "year": int(year),
                    "scenario": scenario,
                    "nuts2_code": n2,
                    "nuts2_nome": n2_nome,
                    "nuts3_code": row["nuts3_code"],
                    "nuts3_nome": row.get("nuts3_nome", ""),
                    "pop_var": pop_var,
                    "population": int(values[i]),
                })
                share_records.append({
                    "year": int(year),
                    "scenario": scenario,
                    "nuts2_code": n2,
                    "nuts3_code": row["nuts3_code"],
                    "nuts3_nome": row.get("nuts3_nome", ""),
                    "pop_var": pop_var,
                    "share": float(weights[i]),
                    "score": float(score[i]),
                    "indice_envelhecimento_mean": row["indice_envelhecimento_mean"],
                    "indice_renovacao_pop_ativa_mean": row["indice_renovacao_pop_ativa_mean"],
                })

    long = pd.DataFrame(records)
    shares = pd.DataFrame(share_records)
    if long.empty:
        raise RuntimeError("Nao foi produzida nenhuma imputacao NUTS3.")

    idx = [
        "year", "scenario", "nuts2_code", "nuts2_nome",
        "nuts3_code", "nuts3_nome",
    ]
    wide = long.pivot_table(
        index=idx, columns="pop_var", values="population", aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    ordered = idx + [c for c in POP_VARS if c in wide.columns]
    wide = wide[ordered]
    wide = wide.sort_values(
        ["year", "scenario", "nuts2_code", "nuts3_code"]).reset_index(drop=True)
    return wide, long, shares

# ============================================================
# CHECKS
# ============================================================
def build_reconciliation_check(nuts3_wide, proj, pop_vars):
    grp = ["year", "scenario", "nuts2_code"]
    n3_sum = nuts3_wide.groupby(grp, as_index=False)[pop_vars].sum()
    n2 = proj[grp + pop_vars].copy()
    check = n3_sum.merge(
        n2, on=grp, how="left", suffixes=("_nuts3", "_nuts2"))
    for c in pop_vars:
        check["diff_" + c] = check[c + "_nuts3"] - check[c + "_nuts2"]
    return check

# ============================================================
# MAIN
# ============================================================
def main():
    np.random.seed(RANDOM_SEED)

    print("=" * 72)
    print("09_build_population_nuts3")
    print("=" * 72)
    print(f"  projeccoes : {POP_PROJ_FILE}")
    print(f"  envelhec.  : {AGEING_FILE}")
    print(f"  renovacao  : {RENEWAL_FILE}")
    print(f"  output     : {OUT_FILE}")
    print()

    proj = load_population_projections(POP_PROJ_FILE)

    print("A ler indicadores NUTS3 (folha 'nuts3')...")
    indicators, geo = build_indicator_base()
    print(f"  regioes NUTS3 na grelha: {geo['nuts3_code'].nunique()}")

    pop_vars = [c for c in POP_VARS if c in proj.columns]
    years = sorted(proj["year"].unique())
    indicator_panel = prepare_indicator_panel(indicators, geo, years)

    print("A repartir NUTS2 -> NUTS3 com quotas dinamicas...")
    nuts3_wide, nuts3_long, shares = allocate_nuts2_to_nuts3(
        proj, indicator_panel, pop_vars)

    check = build_reconciliation_check(nuts3_wide, proj, pop_vars)
    diff_cols = [c for c in check.columns if c.startswith("diff_")]
    max_abs_diff = float(check[diff_cols].abs().to_numpy().max())
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Reconciliacao falhou. Max |diff| = {max_abs_diff}")

    summary = pd.DataFrame({
        "metric": [
            "rows_nuts3_wide", "first_year", "last_year", "n_scenarios",
            "n_nuts2", "n_nuts3", "max_abs_reconciliation_diff",
        ],
        "value": [
            len(nuts3_wide),
            int(nuts3_wide["year"].min()),
            int(nuts3_wide["year"].max()),
            nuts3_wide["scenario"].nunique(),
            nuts3_wide["nuts2_code"].nunique(),
            nuts3_wide["nuts3_code"].nunique(),
            max_abs_diff,
        ],
    })

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        nuts3_wide.to_excel(writer, sheet_name="nuts3_population", index=False)
        nuts3_long.to_excel(writer, sheet_name="nuts3_population_long", index=False)
        shares.to_excel(writer, sheet_name="dynamic_shares", index=False)
        check.to_excel(writer, sheet_name="reconciliation_check", index=False)
        indicator_panel.to_excel(writer, sheet_name="indicator_panel", index=False)
        geo.to_excel(writer, sheet_name="nuts3_geo", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 72)
    print("OUTPUT:", OUT_FILE)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"Reconciliacao: diferenca absoluta maxima = {max_abs_diff}")


if __name__ == "__main__":
    main()