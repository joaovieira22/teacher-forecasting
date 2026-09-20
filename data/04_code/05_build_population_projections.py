
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting\data")
RAW_DIR = BASE_DIR / "01_raw" / "ine" / "population_projections"
PROCESSED_DIR = BASE_DIR / "02_processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "population_projections_nuts2.xlsx"

# ============================================================
# CONFIG
# ============================================================
SCENARIO_MAP = {
    "baixo": "low",
    "central": "central",
    "alto": "high",
    "sem migracoes": "no_migration",
    "sem migracao": "no_migration",
}
SCENARIO_LABEL = {
    "low": "Baixo",
    "central": "Central",
    "high": "Alto",
    "no_migration": "Sem migracoes",
}

AGE_BANDS = {
    "pop_0_4": range(0, 5),
    "pop_5_9": range(5, 10),
    "pop_10_14": range(10, 15),
    "pop_15_19": range(15, 20),
    "pop_20_24": range(20, 25),
    "pop_25_29": range(25, 30),
    "pop_30_34": range(30, 35),
    "pop_35_39": range(35, 40),
    "pop_40_44": range(40, 45),
    "pop_45_49": range(45, 50),
    "pop_50_54": range(50, 55),
    "pop_55_59": range(55, 60),
    "pop_60_64": range(60, 65),
    "pop_65_69": range(65, 70),
    "pop_70_74": range(70, 75),
    "pop_75_79": range(75, 80),
    "pop_80_84": range(80, 85),
}
# 85+ = idades 85..100

FINAL_POP_COLS = [
    "pop_total", "pop_0_4", "pop_5_9", "pop_10_14", "pop_15_19", "pop_20_24",
    "pop_25_29", "pop_30_34", "pop_35_39", "pop_40_44", "pop_45_49", "pop_50_54",
    "pop_55_59", "pop_60_64", "pop_65_69", "pop_70_74", "pop_75_79", "pop_80_84",
    "pop_85_plus",
]

VALID_CODES = {"PT", "11", "15", "19", "1A", "1B", "1C", "1D", "20", "30"}
LEVEL = {"PT": "country"}
for c in ["11", "15", "19", "1A", "1B", "1C", "1D", "20", "30"]:
    LEVEL[c] = "nuts2"

# ============================================================
# HELPERS
# ============================================================
def norm_txt(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = unicodedata.normalize("NFKD", x)
    x = "".join(c for c in x if not unicodedata.combining(c))
    x = x.lower()
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def code(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\.0$", "", x)
    return x.replace(" ", "")


def clean_text(x):
    if pd.isna(x):
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", s)


def to_int(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in {"", "-", "\\-", "\u2013", "x", "X", "//", "nan", "NaN"}:
        return None
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except Exception:
        return None


def as_year(x):
    try:
        y = int(float(x))
        if 2025 <= y <= 2100:
            return y
    except Exception:
        return None
    return None


def normalize_sex(x):
    n = norm_txt(x)
    if n == "hm":
        return "HM"
    if n == "h":
        return "H"
    if n == "m":
        return "M"
    return None


def normalize_scenario(x):
    return SCENARIO_MAP.get(norm_txt(x))


def normalize_age(x):
    n = norm_txt(x)
    if n == "":
        return None
    if n == "total":
        return "total"
    if "menos de 1" in n:
        return 0
    m = re.match(r"^(\d+)\s*(ano|anos)", n)
    if m:
        return int(m.group(1))
    if re.match(r"^\d+$", n):
        return int(n)
    if "100" in n and ("mais" in n or "ou mais" in n):
        return 100
    return None


def excel_engine(path):
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"

# ============================================================
# HEADER MAPPING
# ============================================================
def find_row_by(df, fn, max_rows=40, min_count=3):
    best_r, best_c = None, 0
    for r in range(min(max_rows, len(df))):
        c = sum(fn(v) is not None for v in df.iloc[r].tolist())
        if c > best_c:
            best_c, best_r = c, r
    return best_r if best_c >= min_count else None


def build_col_map(df, sex_row, age_row, scen_row):
    sex_vals = df.iloc[sex_row].tolist() if sex_row is not None else []
    age_vals = df.iloc[age_row].tolist()
    scen_vals = df.iloc[scen_row].tolist()

    col_map = {}
    cur_sex = None
    cur_age = None
    for c in range(3, len(scen_vals)):
        if sex_row is not None and c < len(sex_vals):
            s = normalize_sex(sex_vals[c])
            if s is not None:
                cur_sex = s
        if c < len(age_vals):
            a = normalize_age(age_vals[c])
            if a is not None:
                cur_age = a
        scen = normalize_scenario(scen_vals[c])
        if cur_sex == "HM" and cur_age is not None and scen is not None:
            col_map[c] = (cur_age, scen)
    return col_map

# ============================================================
# PARSE ONE SHEET
# ============================================================
def parse_sheet(path, sheet):
    df = pd.read_excel(path, sheet_name=sheet, header=None, engine=excel_engine(path))
    ncols = len(df.columns)

    scen_row = find_row_by(df, normalize_scenario, min_count=4)
    if scen_row is None:
        return pd.DataFrame()

    age_row = find_row_by(df, normalize_age, max_rows=scen_row, min_count=3)
    if age_row is None:
        return pd.DataFrame()

    sex_row = find_row_by(df, normalize_sex, max_rows=age_row, min_count=1)

    col_map = build_col_map(df, sex_row, age_row, scen_row)
    if not col_map:
        return pd.DataFrame()

    rows = []
    cur_year = None
    for r in range(scen_row + 2, len(df)):
        # o ano aparece na coluna 0
        y = as_year(df.iloc[r, 0]) if ncols > 0 else None
        if y is not None:
            cur_year = y
        if cur_year is None:
            continue

        region = clean_text(df.iloc[r, 1]) if ncols > 1 else ""
        cod = code(df.iloc[r, 2]) if ncols > 2 else ""
        if cod not in VALID_CODES:
            continue

        for c, pair in col_map.items():
            age, scen = pair
            if c >= ncols:
                continue
            v = to_int(df.iloc[r, c])
            if v is None:
                continue
            rows.append({
                "year": int(cur_year),
                "location": region,
                "location_code": cod,
                "location_level": LEVEL[cod],
                "scenario": scen,
                "age": age,
                "population": v,
            })

    return pd.DataFrame(rows)


def parse_file(path):
    print(f"Ler: {path.name}")
    xls = pd.ExcelFile(path, engine=excel_engine(path))
    parts = []
    for s in xls.sheet_names:
        p = parse_sheet(path, s)
        if len(p):
            parts.append(p)
    if not parts:
        print("  sem dados extraidos")
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    anos = sorted(out["year"].dropna().unique().tolist())
    print(f"  linhas: {len(out):,}  anos: {anos[0]}-{anos[-1]}" if anos else f"  linhas: {len(out):,}")
    return out

# ============================================================
# BUILD WIDE
# ============================================================
def build_wide(long):
    long = long[long["age"] != "total"].copy()
    long["age"] = pd.to_numeric(long["age"], errors="coerce")
    long = long.dropna(subset=["age"])
    long["age"] = long["age"].astype(int)

    # a mesma celula pode aparecer em partes sobrepostas
    long = long.drop_duplicates(
        ["year", "location_code", "scenario", "age"], keep="first"
    )

    keys = ["year", "location", "location_code", "location_level", "scenario"]
    wide = long.pivot_table(
        index=keys, columns="age", values="population", aggfunc="first"
    ).reset_index()
    wide.columns.name = None

    age_cols = [c for c in wide.columns if isinstance(c, (int, np.integer))]

    for band, ages in AGE_BANDS.items():
        cols = [a for a in ages if a in age_cols]
        if cols:
            wide[band] = wide[cols].sum(axis=1, min_count=1)
        else:
            wide[band] = np.nan

    cols_85 = [a for a in age_cols if a >= 85]
    if cols_85:
        wide["pop_85_plus"] = wide[cols_85].sum(axis=1, min_count=1)
    else:
        wide["pop_85_plus"] = np.nan

    all_bands = [c for c in FINAL_POP_COLS if c != "pop_total"]
    wide["pop_total"] = wide[all_bands].sum(axis=1, min_count=1)

    wide["scenario_label"] = wide["scenario"].map(SCENARIO_LABEL)

    out_cols = keys + ["scenario_label"] + FINAL_POP_COLS
    out_cols = [c for c in out_cols if c in wide.columns]
    wide = wide[out_cols].sort_values(
        ["year", "location_level", "location_code", "scenario"]
    ).reset_index(drop=True)
    return wide, long

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("05_build_population_projections")
    print("=" * 80)
    print(f"  input : {RAW_DIR}")
    print(f"  output: {OUT_FILE}")
    print()

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de input nao encontrada: {RAW_DIR}\n"
            "Confirma que a reorganizacao da pasta data ja foi aplicada."
        )

    files = sorted(list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"Nao encontrei ficheiros Excel em: {RAW_DIR}")

    parts = []
    for f in files:
        p = parse_file(f)
        if len(p):
            parts.append(p)

    if not parts:
        raise RuntimeError("Nao consegui extrair dados de nenhum ficheiro.")

    long = pd.concat(parts, ignore_index=True)
    wide, long_clean = build_wide(long)

    n_nuts2 = wide.loc[wide["location_level"].eq("nuts2"), "location_code"].nunique()
    scen_list = ", ".join(sorted(wide["scenario"].dropna().unique()))

    summary = pd.DataFrame({
        "metric": [
            "wide_rows", "first_year", "last_year",
            "n_locations", "n_nuts2", "scenarios",
        ],
        "value": [
            len(wide),
            wide["year"].min(),
            wide["year"].max(),
            wide["location_code"].nunique(),
            n_nuts2,
            scen_list,
        ],
    })

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        wide.to_excel(writer, sheet_name="panel_wide_all", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(OUT_FILE)
    print()
    print(summary.to_string(index=False))
    print()
    print(wide.head(12).to_string(index=False))


if __name__ == "__main__":
    main()