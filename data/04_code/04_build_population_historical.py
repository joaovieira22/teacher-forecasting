
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting\data")
RAW_DIR = BASE_DIR / "01_raw" / "ine" / "population_estimates"
PROCESSED_DIR = BASE_DIR / "02_processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "population_historical_nuts3.xlsx"

# ============================================================
# CONFIG
# ============================================================
VALID_NUTS2_2024 = {"11", "15", "19", "1A", "1B", "1C", "1D", "20", "30"}

AGE_ORDER = [
    "pop_total",
    "pop_0_4",
    "pop_5_9",
    "pop_10_14",
    "pop_15_19",
    "pop_20_24",
    "pop_25_29",
    "pop_30_34",
    "pop_35_39",
    "pop_40_44",
    "pop_45_49",
    "pop_50_54",
    "pop_55_59",
    "pop_60_64",
    "pop_65_69",
    "pop_70_74",
    "pop_75_79",
    "pop_80_84",
    "pop_85_plus",
]

AGE_MAP = {
    "total": "pop_total",
    "0 4 anos": "pop_0_4",
    "5 9 anos": "pop_5_9",
    "10 14 anos": "pop_10_14",
    "15 19 anos": "pop_15_19",
    "20 24 anos": "pop_20_24",
    "25 29 anos": "pop_25_29",
    "30 34 anos": "pop_30_34",
    "35 39 anos": "pop_35_39",
    "40 44 anos": "pop_40_44",
    "45 49 anos": "pop_45_49",
    "50 54 anos": "pop_50_54",
    "55 59 anos": "pop_55_59",
    "60 64 anos": "pop_60_64",
    "65 69 anos": "pop_65_69",
    "70 74 anos": "pop_70_74",
    "75 79 anos": "pop_75_79",
    "80 84 anos": "pop_80_84",
    "85 e mais anos": "pop_85_plus",
    "85 mais anos": "pop_85_plus",
    "85 anos ou mais": "pop_85_plus",
}

# ============================================================
# HELPERS
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def norm_txt(x):
    x = clean_text(x)
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
    x = x.replace(" ", "")
    return x


def municipio_id_from_code(x):
    # Prefixo NUTS + 4 digitos do concelho.
    # 1111601 -> 1601 ; 11A0104 -> 0104 ; 1C40705 -> 0705 ; 2004301 -> 4301
    c = code(x)
    digits = re.sub(r"[^0-9]", "", c)
    if len(digits) < 4:
        return ""
    return digits[-4:]


def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s in {"", "-", "\u2013", "x", "X", "//", "nan", "NaN"}:
        return np.nan
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan


def as_year(x):
    try:
        y = int(float(x))
        if 1990 <= y <= 2100:
            return y
    except Exception:
        return None
    return None


def excel_engine(path):
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"


def is_nuts2024_file(path, df):
    name = norm_txt(path.name)
    if "2024" in name:
        return True
    for r in range(min(15, len(df))):
        joined = " ".join(norm_txt(v) for v in df.iloc[r].tolist())
        if "nuts 2024" in joined:
            return True
    return False


def normalize_age(x):
    n = norm_txt(x)
    if not n:
        return None
    return AGE_MAP.get(n)


def normalize_sex(x):
    n = norm_txt(x)
    if n == "hm":
        return "HM"
    if n == "h":
        return "H"
    if n == "m":
        return "M"
    return None


def classify_row(c):
    c = code(c)
    if c == "":
        return "empty"
    if c == "PT":
        return "country"
    if c in {"1", "2", "3"}:
        return "nuts1"
    if c in VALID_NUTS2_2024 or c in {"16", "17", "18"}:
        return "nuts2"
    if re.match(r"^[0-9A-Za-z]{3}$", c):
        return "nuts3"
    if len(c) >= 5 and municipio_id_from_code(c) != "":
        return "municipality"
    return "other"

# ============================================================
# HEADER DETECTION
# ============================================================
def find_age_row(df):
    best_row, best_count = None, 0
    for r in range(min(40, len(df))):
        ages = [normalize_age(v) for v in df.iloc[r].tolist()]
        count = sum(a is not None for a in ages)
        if count > best_count:
            best_count, best_row = count, r
    if best_count < 10:
        return None
    return best_row


def find_sex_row(df, age_row):
    best_row, best_count = None, 0
    for r in range(max(0, age_row - 8), age_row):
        vals = [normalize_sex(v) for v in df.iloc[r].tolist()]
        count = sum(v is not None for v in vals)
        if count > best_count:
            best_count, best_row = count, r
    if best_count == 0:
        return None
    return best_row


def build_column_map(df, sex_row, age_row):
    age_vals = df.iloc[age_row].tolist()
    if sex_row is not None:
        sex_vals = df.iloc[sex_row].tolist()
    else:
        sex_vals = [None] * len(age_vals)

    col_map = {}
    current_sex = None
    for c in range(0, len(age_vals)):
        s = normalize_sex(sex_vals[c]) if c < len(sex_vals) else None
        if s is not None:
            current_sex = s
        a = normalize_age(age_vals[c])
        if a is None:
            continue
        if current_sex is None:
            current_sex = "HM"
        if current_sex == "HM":
            col_map[c] = a
    return col_map


def pick_year_name_code(df, r, id_cols, ncols):
    # Devolve (year_here, name, code) procurando nas colunas de identificacao.
    year_here = None
    found_code = ""
    found_code_col = None

    for c in id_cols:
        val = df.iloc[r, c] if c < ncols else ""
        y = as_year(val)
        if y is not None:
            year_here = y
            continue
        cval = code(val)
        if cval == "":
            continue
        kind = classify_row(cval)
        if kind in {"country", "nuts1", "nuts2", "nuts3", "municipality"}:
            found_code = cval
            found_code_col = c

    name = ""
    if found_code_col is not None:
        # nome = coluna de texto imediatamente a esquerda do codigo
        for c in range(found_code_col - 1, -1, -1):
            t = clean_text(df.iloc[r, c]) if c < ncols else ""
            if t != "" and as_year(t) is None and code(t) != found_code:
                name = t
                break
        if name == "":
            for c in id_cols:
                t = clean_text(df.iloc[r, c]) if c < ncols else ""
                if t != "" and as_year(t) is None and code(t) != found_code:
                    name = t
                    break

    return year_here, name, found_code

# ============================================================
# PARSING
# ============================================================
def parse_file(path):
    print(f"Ler: {path.name}")
    df = pd.read_excel(path, header=None, engine=excel_engine(path))
    ncols = len(df.columns)

    age_row = find_age_row(df)
    if age_row is None:
        print("  sem idades detectadas")
        return pd.DataFrame(), pd.DataFrame()

    sex_row = find_sex_row(df, age_row)
    col_map = build_column_map(df, sex_row, age_row)
    if not col_map:
        print("  sem colunas HM detectadas")
        return pd.DataFrame(), pd.DataFrame()

    source_is_nuts2024 = is_nuts2024_file(path, df)

    # Colunas de identificacao = tudo antes da primeira coluna de dados.
    data_start = min(col_map.keys())
    id_cols = list(range(0, data_start))
    if not id_cols:
        id_cols = [0, 1]

    rows = []
    geo_rows = []

    current_nuts2_code = ""
    current_nuts2_nome = ""
    current_nuts3_code = ""
    current_nuts3_nome = ""
    current_year = None

    for r in range(age_row + 2, len(df)):
        y_here, nome, cod = pick_year_name_code(df, r, id_cols, ncols)
        if y_here is not None:
            current_year = y_here

        if nome == "" or cod == "":
            continue

        n = norm_txt(nome)
        if (n.startswith("populacao residente") or n.startswith("nota")
                or n.startswith("ultima atualizacao") or n.startswith("fonte")):
            break

        kind = classify_row(cod)

        if kind == "nuts2":
            current_nuts2_code = cod
            current_nuts2_nome = nome
            current_nuts3_code = ""
            current_nuts3_nome = ""
            geo_rows.append({
                "source_file": path.name,
                "source_is_nuts2024": int(source_is_nuts2024),
                "geo_level": "nuts2",
                "code": cod,
                "name": nome,
                "municipio_id": "",
                "nuts2_code_source": current_nuts2_code,
                "nuts2_nome_source": current_nuts2_nome,
                "nuts3_code_source": "",
                "nuts3_nome_source": "",
            })
            continue

        if kind == "nuts3":
            current_nuts3_code = cod
            current_nuts3_nome = nome
            geo_rows.append({
                "source_file": path.name,
                "source_is_nuts2024": int(source_is_nuts2024),
                "geo_level": "nuts3",
                "code": cod,
                "name": nome,
                "municipio_id": "",
                "nuts2_code_source": current_nuts2_code,
                "nuts2_nome_source": current_nuts2_nome,
                "nuts3_code_source": current_nuts3_code,
                "nuts3_nome_source": current_nuts3_nome,
            })
            continue

        if kind != "municipality":
            continue

        mid = municipio_id_from_code(cod)
        if mid == "":
            continue

        geo_rows.append({
            "source_file": path.name,
            "source_is_nuts2024": int(source_is_nuts2024),
            "geo_level": "municipality",
            "code": cod,
            "name": nome,
            "municipio_id": mid,
            "nuts2_code_source": current_nuts2_code,
            "nuts2_nome_source": current_nuts2_nome,
            "nuts3_code_source": current_nuts3_code,
            "nuts3_nome_source": current_nuts3_nome,
        })

        if current_year is None:
            continue

        rec = {
            "ano": int(current_year),
            "municipio_id": mid,
            "municipio_code_source": cod,
            "municipio_nome_source": nome,
            "source_file": path.name,
            "source_is_nuts2024": int(source_is_nuts2024),
        }
        has_any = False
        for c, out_col in col_map.items():
            if c >= ncols:
                continue
            v = to_num(df.iloc[r, c])
            if not pd.isna(v):
                rec[out_col] = int(round(v))
                has_any = True
        if has_any:
            rows.append(rec)

    data = pd.DataFrame(rows)
    geo = pd.DataFrame(geo_rows)
    anos = sorted(data["ano"].dropna().unique().tolist()) if len(data) else []
    print(f"  anos: {anos}")
    print(f"  municipios x ano: {len(data):,}")
    print(f"  nuts2024: {source_is_nuts2024}")
    return data, geo

# ============================================================
# CROSSWALK
# ============================================================
def build_crosswalk(geo_all):
    if geo_all.empty:
        raise RuntimeError("Sem geografia extraida.")
    is24 = geo_all["source_is_nuts2024"] == 1
    ismun = geo_all["geo_level"] == "municipality"
    geo24 = geo_all[is24 & ismun].copy()
    if geo24.empty:
        raise RuntimeError(
            "Nao encontrei municipios no ficheiro NUTS2024.\n"
            f"Confirma que o ficheiro 2024 esta em {RAW_DIR}.\n"
            "A deteccao procura '2024' no nome, ou 'nuts 2024' nas "
            "primeiras 15 linhas do ficheiro."
        )

    keep = [
        "municipio_id", "code", "name",
        "nuts2_code_source", "nuts2_nome_source",
        "nuts3_code_source", "nuts3_nome_source", "source_file",
    ]
    cw = geo24[keep].rename(columns={
        "code": "concelho_code_source_2024",
        "name": "concelho",
        "nuts2_code_source": "nuts2_code",
        "nuts2_nome_source": "nuts2_nome",
        "nuts3_code_source": "nuts3_code",
        "nuts3_nome_source": "nuts3_nome",
        "source_file": "crosswalk_source_file",
    })
    cw["concelho_norm"] = cw["concelho"].map(norm_txt)
    cw = cw.sort_values(["municipio_id", "crosswalk_source_file"])
    cw = cw.drop_duplicates("municipio_id", keep="first")
    return cw

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("04_build_population_historical")
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

    parts, geos = [], []
    for f in files:
        data, geo = parse_file(f)
        if len(data):
            parts.append(data)
        if len(geo):
            geos.append(geo)

    if not parts:
        raise RuntimeError("Nao consegui extrair dados de nenhum ficheiro.")

    raw = pd.concat(parts, ignore_index=True)
    geo_all = pd.concat(geos, ignore_index=True) if geos else pd.DataFrame()
    crosswalk = build_crosswalk(geo_all)

    for c in AGE_ORDER:
        if c not in raw.columns:
            raw[c] = np.nan
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    concelho = raw.merge(crosswalk, on="municipio_id", how="left")

    concelho["priority"] = concelho["source_is_nuts2024"].fillna(0).astype(int)
    concelho = concelho.sort_values(
        ["ano", "municipio_id", "priority", "source_file"],
        ascending=[True, True, False, True],
    )
    concelho = concelho.drop_duplicates(["ano", "municipio_id"], keep="first")

    no_n3 = concelho["nuts3_code"].isna() | concelho["nuts3_code"].eq("")
    unmapped = concelho[no_n3].copy()
    mapped = concelho[~no_n3].copy()

    group_n3 = ["ano", "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome"]

    nuts3 = (
        mapped.groupby(group_n3, as_index=False)[AGE_ORDER]
        .sum(min_count=1)
        .sort_values(["ano", "nuts2_code", "nuts3_code"])
    )
    n_conc = (
        mapped.groupby(group_n3, as_index=False)["municipio_id"]
        .nunique()
        .rename(columns={"municipio_id": "n_concelhos"})
    )
    nuts3 = nuts3.merge(n_conc, on=group_n3, how="left")

    nuts2 = (
        mapped.groupby(["ano", "nuts2_code", "nuts2_nome"], as_index=False)[AGE_ORDER]
        .sum(min_count=1)
        .sort_values(["ano", "nuts2_code"])
    )

    age_bands = [c for c in AGE_ORDER if c != "pop_total"]
    check_cols = ["ano", "nuts2_code", "nuts3_code", "nuts3_nome", "pop_total"]
    check = nuts3[check_cols].copy()
    check["sum_age_bands"] = nuts3[age_bands].sum(axis=1, min_count=1)
    check["diff_total_minus_bands"] = check["pop_total"] - check["sum_age_bands"]

    check_year = (
        mapped.groupby("ano", as_index=False)
        .agg(
            n_concelhos=("municipio_id", "nunique"),
            n_nuts3=("nuts3_code", "nunique"),
            n_nuts2=("nuts2_code", "nunique"),
        )
        .sort_values("ano")
    )

    last_year_n3 = 0
    if len(nuts3):
        ymax = nuts3["ano"].max()
        last_year_n3 = int(nuts3[nuts3["ano"] == ymax]["nuts3_code"].nunique())

    max_diff = np.nan
    if len(check):
        max_diff = check["diff_total_minus_bands"].abs().max()

    summary = pd.DataFrame({
        "metric": [
            "raw_rows", "mapped_rows", "unmapped_rows", "nuts3_rows", "nuts2_rows",
            "first_year", "last_year", "n_nuts3_latest_year", "max_abs_total_diff",
        ],
        "value": [
            len(raw), len(mapped), len(unmapped), len(nuts3), len(nuts2),
            int(mapped["ano"].min()) if len(mapped) else np.nan,
            int(mapped["ano"].max()) if len(mapped) else np.nan,
            last_year_n3,
            max_diff,
        ],
    })

    wanted = [
        "ano", "municipio_id", "concelho_code_source_2024", "concelho", "concelho_norm",
        "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome",
    ] + AGE_ORDER + ["source_file", "source_is_nuts2024"]
    concelho_out_cols = [c for c in wanted if c in mapped.columns]

    conc_sheet = mapped[concelho_out_cols].sort_values(
        ["ano", "nuts2_code", "nuts3_code", "concelho"]
    )
    cw_sheet = crosswalk.sort_values(["nuts2_code", "nuts3_code", "concelho"])

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        conc_sheet.to_excel(writer, sheet_name="concelho", index=False)
        nuts3.to_excel(writer, sheet_name="nuts3", index=False)
        nuts2.to_excel(writer, sheet_name="nuts2", index=False)
        cw_sheet.to_excel(writer, sheet_name="crosswalk_concelhos_2024", index=False)
        check.to_excel(writer, sheet_name="check_totais", index=False)
        check_year.to_excel(writer, sheet_name="checks_by_year", index=False)
        unmapped.to_excel(writer, sheet_name="unmapped", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(OUT_FILE)
    print()
    print(summary.to_string(index=False))
    print()
    print("Checks por ano:")
    print(check_year.to_string(index=False))


if __name__ == "__main__":
    main()