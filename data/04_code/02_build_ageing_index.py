
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting\data")
RAW_DIR = BASE_DIR / "01_raw" / "ine" / "ageing_index"
PROCESSED_DIR = BASE_DIR / "02_processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "ageing_index_nuts3.xlsx"

# ============================================================
# CONFIG
# ============================================================
VALUE_COL = "indice_envelhecimento"
MEAN_NAME = "indice_envelhecimento_mean"
MEDIAN_NAME = "indice_envelhecimento_median"

VALID_NUTS2_2024 = {"11", "15", "19", "1A", "1B", "1C", "1D", "20", "30"}

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
    """District+municipality 4-digit id, stable across NUTS revisions.

    1111601 -> 1601 | 11A0104 -> 0104 | 1A01115 -> 1115 | 1C40701 -> 0701
    """
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
    if "34077" in name or "2024" in name:
        return True
    for r in range(min(12, len(df))):
        joined = " ".join(norm_txt(v) for v in df.iloc[r].tolist())
        if "nuts 2024" in joined:
            return True
    return False


def find_year_row(df):
    best_row, best_count = None, 0
    for r in range(min(30, len(df))):
        years = [as_year(v) for v in df.iloc[r].tolist()]
        count = sum(y is not None for y in years)
        if count > best_count:
            best_count = count
            best_row = r
    return best_row if best_count > 0 else None


def find_year_cols(df, year_row):
    return {c: as_year(v) for c, v in enumerate(df.iloc[year_row].tolist()) if as_year(v) is not None}


def classify_row(c):
    c = code(c)
    if c == "":
        return "empty"
    if c == "PT":
        return "country"
    if c in {"1", "2", "3"}:
        return "nuts1"
    if c in VALID_NUTS2_2024 or c in {"16", "17", "18"}:  # 2013 NUTS2 codes too
        return "nuts2"
    if re.match(r"^[0-9A-Za-z]{3}$", c):
        return "nuts3"
    if len(c) >= 5 and municipio_id_from_code(c) != "":
        return "municipality"
    return "other"

# ============================================================
# PARSING
# ============================================================
def parse_file(path):
    print(f"Ler: {path.name}")
    df = pd.read_excel(path, header=None, engine=excel_engine(path))

    year_row = find_year_row(df)
    if year_row is None:
        print("  sem anos detectados")
        return pd.DataFrame(), pd.DataFrame(), False

    year_cols = find_year_cols(df, year_row)
    if not year_cols:
        print("  sem colunas de anos")
        return pd.DataFrame(), pd.DataFrame(), False

    source_is_nuts2024 = is_nuts2024_file(path, df)

    rows, geo_rows = [], []
    cur_n2c = cur_n2n = cur_n3c = cur_n3n = ""

    for r in range(year_row + 2, len(df)):
        nome = clean_text(df.iloc[r, 0]) if df.shape[1] > 0 else ""
        cod = code(df.iloc[r, 1]) if df.shape[1] > 1 else ""

        if nome == "" or cod == "":
            continue

        n = norm_txt(nome)
        if n.startswith("indice de envelhecimento") or n.startswith("nota") or n.startswith("ultima atualizacao"):
            break

        kind = classify_row(cod)

        if kind == "nuts2":
            cur_n2c, cur_n2n, cur_n3c, cur_n3n = cod, nome, "", ""
            continue
        if kind == "nuts3":
            cur_n3c, cur_n3n = cod, nome
            continue
        if kind != "municipality":
            continue

        mid = municipio_id_from_code(cod)
        if mid == "":
            continue

        geo_rows.append({
            "source_is_nuts2024": int(source_is_nuts2024),
            "municipio_id": mid,
            "concelho": nome,
            "concelho_code_source": cod,
            "nuts2_code": cur_n2c,
            "nuts2_nome": cur_n2n,
            "nuts3_code": cur_n3c,
            "nuts3_nome": cur_n3n,
        })

        for c, y in year_cols.items():
            if c >= len(df.columns):
                continue
            rows.append({
                "ano": int(y),
                "municipio_id": mid,
                "concelho_source": nome,
                VALUE_COL: to_num(df.iloc[r, c]),
                "source_file": path.name,
                "source_is_nuts2024": int(source_is_nuts2024),
            })
            
    data = pd.DataFrame(rows)
    geo = pd.DataFrame(geo_rows)
    print(f"  anos: {sorted(data['ano'].dropna().unique().tolist()) if len(data) else []}")
    print(f"  municipios x ano: {len(data):,}")
    print(f"  nuts2024: {source_is_nuts2024}")
    return data, geo, source_is_nuts2024

# ============================================================
# CROSSWALK
# ============================================================
def build_crosswalk(geo_all):
    if geo_all.empty:
        raise RuntimeError("Nao ha geografia extraida dos ficheiros INE.")
    geo24 = geo_all[geo_all["source_is_nuts2024"] == 1].copy()
    if geo24.empty:
        raise RuntimeError(
            "Nao encontrei o ficheiro NUTS2024 na pasta 01_raw\\ine\\ageing_index.\n"
            "A deteccao procura '34077' ou '2024' no nome, ou 'nuts 2024' nas "
            "primeiras 12 linhas do ficheiro."
        )
    cw = geo24[[
        "municipio_id", "concelho", "concelho_code_source",
        "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome",
    ]].copy()
    cw = cw.sort_values("municipio_id").drop_duplicates("municipio_id", keep="first")
    cw["concelho_norm"] = cw["concelho"].map(norm_txt)
    return cw

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("02_build_ageing_index")
    print("=" * 80)
    print(f"  input : {RAW_DIR}")
    print(f"  output: {OUT_FILE}")
    print()

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de input nao encontrada: {RAW_DIR}\n"
            f"Confirma que a reorganizacao da pasta data ja foi aplicada."
        )

    files = sorted(list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.xlsx")))
    if not files:
        raise FileNotFoundError(f"Nao encontrei ficheiros Excel em: {RAW_DIR}")

    parts, geos = [], []
    for f in files:
        data, geo, _ = parse_file(f)
        if len(data):
            parts.append(data)
        if len(geo):
            geos.append(geo)

    if not parts:
        raise RuntimeError("Nao consegui extrair dados de nenhum ficheiro.")

    raw = pd.concat(parts, ignore_index=True)
    geo_all = pd.concat(geos, ignore_index=True)
    crosswalk = build_crosswalk(geo_all)

    # Map every municipality-year (2013 and 2024 sources) to NUTS2024 via municipio_id.
    concelho = raw.merge(crosswalk, on="municipio_id", how="left")

    # Prefer NUTS2024 source when the same year appears in both files.
    concelho["priority"] = concelho["source_is_nuts2024"].fillna(0).astype(int)
    concelho = concelho.sort_values(
        ["ano", "municipio_id", "priority", "source_file"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["ano", "municipio_id"], keep="first")

    unmapped = concelho[concelho["nuts3_code"].isna() | concelho["nuts3_code"].eq("")].copy()
    mapped = concelho[concelho["nuts3_code"].notna() & concelho["nuts3_code"].ne("")].copy()
    mapped[VALUE_COL] = pd.to_numeric(mapped[VALUE_COL], errors="coerce")
    valid = mapped[mapped[VALUE_COL].notna()].copy()

    nuts3 = (
        valid.groupby(["ano", "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome"], as_index=False)
        .agg(**{
            MEAN_NAME: (VALUE_COL, "mean"),
            MEDIAN_NAME: (VALUE_COL, "median"),
            "n_concelhos": ("municipio_id", "nunique"),
        })
        .sort_values(["ano", "nuts2_code", "nuts3_code"])
    )

    nuts2 = (
        valid.groupby(["ano", "nuts2_code", "nuts2_nome"], as_index=False)
        .agg(**{
            MEAN_NAME: (VALUE_COL, "mean"),
            MEDIAN_NAME: (VALUE_COL, "median"),
            "n_concelhos": ("municipio_id", "nunique"),
        })
        .sort_values(["ano", "nuts2_code"])
    )

    check_year = (
        valid.groupby("ano", as_index=False)
        .agg(
            n_concelhos=("municipio_id", "nunique"),
            n_nuts3=("nuts3_code", "nunique"),
            n_nuts2=("nuts2_code", "nunique"),
            min_value=(VALUE_COL, "min"),
            max_value=(VALUE_COL, "max"),
        )
        .sort_values("ano")
    )

    summary = pd.DataFrame({
        "metric": ["raw_rows", "mapped_rows", "unmapped_rows", "valid_rows",
                   "nuts3_rows", "nuts2_rows", "first_year", "last_year", "n_nuts3_latest_year"],
        "value": [
            len(raw), len(mapped), len(unmapped), len(valid),
            len(nuts3), len(nuts2),
            int(valid["ano"].min()) if len(valid) else np.nan,
            int(valid["ano"].max()) if len(valid) else np.nan,
            int(nuts3[nuts3["ano"] == nuts3["ano"].max()]["nuts3_code"].nunique()) if len(nuts3) else 0,
        ],
    })

    conc_cols = ["ano", "municipio_id", "concelho", "concelho_code_source",
                 "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome",
                 VALUE_COL, "source_file", "source_is_nuts2024"]
    conc_cols = [c for c in conc_cols if c in mapped.columns]

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        mapped[conc_cols].sort_values(["ano", "nuts2_code", "nuts3_code", "concelho"]).to_excel(writer, sheet_name="concelho", index=False)
        nuts3.to_excel(writer, sheet_name="nuts3", index=False)
        nuts2.to_excel(writer, sheet_name="nuts2", index=False)
        crosswalk.sort_values(["nuts2_code", "nuts3_code", "concelho"]).to_excel(writer, sheet_name="crosswalk_concelhos_2024", index=False)
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