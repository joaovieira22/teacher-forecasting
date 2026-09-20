
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting\data")
RAW_DIR = BASE_DIR / "01_raw" / "ine" / "students_enrolled"
PROCESSED_DIR = BASE_DIR / "02_processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "students_nuts3.xlsx"

# ============================================================
# CONFIG
# ============================================================
LEVEL_ORDER = [
    "students_total",
    "students_pre_school",
    "students_basic_1",
    "students_basic_2",
    "students_basic_3",
    "students_secondary",
    "students_post_secondary",
]

HEADER_TOKENS = ["pre escolar", "ensino basico", "secundario"]

NUTS3_2024 = {
    "111": ("11", "Norte", "Alto Minho"),
    "112": ("11", "Norte", "Cavado"),
    "119": ("11", "Norte", "Ave"),
    "11A": ("11", "Norte", "Area Metropolitana do Porto"),
    "11B": ("11", "Norte", "Alto Tamega e Barroso"),
    "11C": ("11", "Norte", "Tamega e Sousa"),
    "11D": ("11", "Norte", "Douro"),
    "11E": ("11", "Norte", "Terras de Tras-os-Montes"),
    "191": ("19", "Centro", "Regiao de Aveiro"),
    "192": ("19", "Centro", "Regiao de Coimbra"),
    "193": ("19", "Centro", "Regiao de Leiria"),
    "194": ("19", "Centro", "Viseu Dao Lafoes"),
    "195": ("19", "Centro", "Beira Baixa"),
    "196": ("19", "Centro", "Beiras e Serra da Estrela"),
    "1D1": ("1D", "Oeste e Vale do Tejo", "Oeste"),
    "1D2": ("1D", "Oeste e Vale do Tejo", "Medio Tejo"),
    "1D3": ("1D", "Oeste e Vale do Tejo", "Leziria do Tejo"),
    "1A0": ("1A", "Grande Lisboa", "Grande Lisboa"),
    "1B0": ("1B", "Peninsula de Setubal", "Peninsula de Setubal"),
    "1C1": ("1C", "Alentejo", "Alentejo Litoral"),
    "1C2": ("1C", "Alentejo", "Baixo Alentejo"),
    "1C3": ("1C", "Alentejo", "Alto Alentejo"),
    "1C4": ("1C", "Alentejo", "Alentejo Central"),
    "150": ("15", "Algarve", "Algarve"),
    "200": ("20", "Regiao Autonoma dos Acores", "Regiao Autonoma dos Acores"),
    "300": ("30", "Regiao Autonoma da Madeira", "Regiao Autonoma da Madeira"),
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


def to_int_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        if np.isnan(x):
            return None
        return int(round(x))
    s = str(x).strip()
    if s in {"-", "\u2013", "\\-", ":"}:
        return 0
    if s in {"", "x", "X", "//", "nan", "NaN", "N/A", "n/a"}:
        return None
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except Exception:
        return None


def parse_school_year(text):
    t = clean_text(text)
    m = re.search(r"(\d{4})\s*/\s*(\d{4})", t)
    if m:
        return f"{m.group(1)}/{m.group(2)}", int(m.group(1))
    m = re.search(r"(\d{4})\s*/\s*(\d{2})", t)
    if m:
        return f"{m.group(1)}/{2000 + int(m.group(2))}", int(m.group(1))
    return None, None


def excel_engine(path):
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"


def find_header_row(df):
    best_row, best_count = None, 0
    for r in range(min(30, len(df))):
        joined = " ".join(norm_txt(v) for v in df.iloc[r].tolist())
        count = sum(1 for tok in HEADER_TOKENS if tok in joined)
        if count > best_count:
            best_count, best_row = count, r
    if best_count < 2:
        return None
    return best_row


def find_first_data_col(df, header_row):
    header = [norm_txt(v) for v in df.iloc[header_row].tolist()]
    for c, h in enumerate(header):
        if h == "total":
            return c
    for c in range(df.shape[1]):
        hits = 0
        for r in range(header_row + 1, min(header_row + 40, len(df))):
            v = to_int_or_none(df.iloc[r, c])
            if v is not None and v > 100:
                hits += 1
        if hits >= 3:
            return c
    return 2


def find_code_col(df, header_row, first_data_col):
    for c in range(first_data_col - 1, -1, -1):
        hits = 0
        for r in range(header_row + 1, min(header_row + 80, len(df))):
            if code(df.iloc[r, c]) in NUTS3_2024:
                hits += 1
        if hits >= 3:
            return c
    return 1

# ============================================================
# PARSE
# ============================================================
def parse_file(path):
    print(f"Ler: {path.name}")
    df = pd.read_excel(path, header=None, engine=excel_engine(path))

    header_row = find_header_row(df)
    if header_row is None:
        print("  sem cabecalho de niveis")
        return pd.DataFrame()

    first_data_col = find_first_data_col(df, header_row)
    code_col = find_code_col(df, header_row, first_data_col)
    year_col = 0

    rows = []
    current_sy = None
    current_ys = None

    for r in range(header_row + 1, len(df)):
        sy, ys = parse_school_year(df.iloc[r, year_col] if df.shape[1] > year_col else "")
        if sy is not None:
            current_sy, current_ys = sy, ys

        cod = code(df.iloc[r, code_col]) if df.shape[1] > code_col else ""
        if cod not in NUTS3_2024:
            continue
        if current_sy is None:
            continue

        vals = []
        for c in range(first_data_col, df.shape[1]):
            v = to_int_or_none(df.iloc[r, c])
            if v is not None:
                vals.append(v)
            if len(vals) == 7:
                break

        if len(vals) < 7:
            vals = vals + [0] * (7 - len(vals))

        nuts2_code, nuts2_nome, nuts3_nome = NUTS3_2024[cod]
        rec = {
            "school_year": current_sy,
            "year_start": current_ys,
            "nuts2_code": nuts2_code,
            "nuts2_nome": nuts2_nome,
            "nuts3_code": cod,
            "nuts3_nome": nuts3_nome,
        }
        for i, name in enumerate(LEVEL_ORDER):
            rec[name] = vals[i]
        rows.append(rec)

    data = pd.DataFrame(rows)
    anos = sorted(data['school_year'].dropna().unique().tolist()) if len(data) else []
    print(f"  linhas NUTS3: {len(data)}  anos: {anos}")
    return data

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("01_build_students")
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

    parts = []
    for f in files:
        d = parse_file(f)
        if len(d):
            parts.append(d)

    if not parts:
        raise RuntimeError("Nao consegui extrair dados de nenhum ficheiro.")

    raw = pd.concat(parts, ignore_index=True)

    for c in LEVEL_ORDER:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(int)

    panel = raw.sort_values(["year_start", "nuts3_code"]).drop_duplicates(
        ["school_year", "nuts3_code"], keep="first"
    ).reset_index(drop=True)

    out_cols = [
        "school_year", "year_start",
        "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome",
    ] + LEVEL_ORDER
    panel = panel[out_cols].sort_values(["nuts3_code", "year_start"]).reset_index(drop=True)

    level_bands = [
        "students_pre_school", "students_basic_1", "students_basic_2",
        "students_basic_3", "students_secondary", "students_post_secondary",
    ]
    check = panel[["school_year", "nuts3_code", "nuts3_nome", "students_total"]].copy()
    check["sum_levels"] = panel[level_bands].sum(axis=1)
    check["diff_total_minus_levels"] = check["students_total"] - check["sum_levels"]

    check_year = (
        panel.groupby("school_year", as_index=False)
        .agg(
            n_nuts3=("nuts3_code", "nunique"),
            students_total=("students_total", "sum"),
        )
        .sort_values("school_year")
    )
    diff_by_year = check.groupby("school_year")["diff_total_minus_levels"].apply(
        lambda s: s.abs().max()
    ).reset_index()
    diff_by_year = diff_by_year.rename(columns={"diff_total_minus_levels": "max_abs_diff"})
    check_year = check_year.merge(diff_by_year, on="school_year", how="left")

    summary = pd.DataFrame({
        "metric": ["rows", "n_nuts3", "first_year", "last_year", "max_abs_total_diff"],
        "value": [
            len(panel),
            panel["nuts3_code"].nunique(),
            panel["school_year"].min(),
            panel["school_year"].max(),
            check["diff_total_minus_levels"].abs().max() if len(check) else np.nan,
        ],
    })

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name="students_panel", index=False)
        check.to_excel(writer, sheet_name="check_totais", index=False)
        check_year.to_excel(writer, sheet_name="checks_by_year", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(OUT_FILE)
    print()
    print(summary.to_string(index=False))
    print()
    print(check_year.to_string(index=False))


if __name__ == "__main__":
    main()