
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
BASE_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

RAW_DIR = BASE_DIR / "01_raw" / "ine" / "teachers"
ANALYSIS_DIR = BASE_DIR / "03_analysis"
MODELS_DATA_DIR = MODELS_DIR / "00_data"

OUT_NAME = "teachers_panel.xlsx"

# os dois destinos finais; acrescentar aqui se algum dia houver um terceiro
FINAL_DIRS = [ANALYSIS_DIR, MODELS_DATA_DIR]

# ============================================================
# CONFIG
# ============================================================
AGE_SUFFIX = [
    "total", "lt25", "25_29", "30_34", "35_39",
    "40_44", "45_49", "50_54", "55_59", "60_plus",
]

NUTS3_TO_NUTS2 = {
    "111": ("11", "Norte"), "112": ("11", "Norte"), "119": ("11", "Norte"),
    "11A": ("11", "Norte"), "11B": ("11", "Norte"), "11C": ("11", "Norte"),
    "11D": ("11", "Norte"), "11E": ("11", "Norte"),
    "191": ("19", "Centro"), "192": ("19", "Centro"), "193": ("19", "Centro"),
    "194": ("19", "Centro"), "195": ("19", "Centro"), "196": ("19", "Centro"),
    "1D1": ("1D", "Oeste e Vale do Tejo"), "1D2": ("1D", "Oeste e Vale do Tejo"),
    "1D3": ("1D", "Oeste e Vale do Tejo"),
    "1A0": ("1A", "Grande Lisboa"),
    "1B0": ("1B", "Peninsula de Setubal"),
    "1C1": ("1C", "Alentejo"), "1C2": ("1C", "Alentejo"),
    "1C3": ("1C", "Alentejo"), "1C4": ("1C", "Alentejo"),
    "150": ("15", "Algarve"),
    "200": ("20", "Regiao Autonoma dos Acores"),
    "300": ("30", "Regiao Autonoma da Madeira"),
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


def code(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\.0$", "", x)
    return x.replace(" ", "")


def to_int(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in {"", "-", "\\-", "\u2013", "x", "X", "//", "nan", "NaN"}:
        return 0  # "-" = dado nulo / nao aplicavel -> 0
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return int(round(float(s)))
    except Exception:
        return None


def parse_school_year(s):
    s = clean_text(s)
    m = re.match(r"^(\d{4})\s*/\s*(\d{4})$", s)
    if m:
        return f"{m.group(1)}/{m.group(2)}", int(m.group(1)), int(m.group(2))
    return None, None, None


def excel_engine(path):
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"


def find_nuts3_code_in_row(series):
    """Devolve (indice_da_coluna, codigo) da primeira celula com um NUTS3 conhecido."""
    for i, v in enumerate(series.tolist()):
        c = code(v)
        if c in NUTS3_TO_NUTS2:
            return i, c
    return None, None


def numbers_after(series, start_col):
    vals = []
    tail = series.iloc[start_col + 1:].tolist()
    for v in tail:
        n = to_int(v)
        if n is not None:
            vals.append(n)
    return vals

# ============================================================
# PARSE
# ============================================================
def parse_file(path):
    print(f"Ler: {path.name}")
    df = pd.read_excel(path, header=None, engine=excel_engine(path))
    ncols = len(df.columns)

    rows = []
    current_sy = None
    current_ys = None
    current_ye = None

    for r in range(len(df)):
        row = df.iloc[r]

        # o ano letivo aparece na coluna 0 de cada bloco
        for cc in range(min(2, ncols)):
            sy, ys, ye = parse_school_year(row.iloc[cc])
            if sy is not None:
                current_sy = sy
                current_ys = ys
                current_ye = ye
                break

        if current_sy is None:
            continue

        # o codigo NUTS3 pode estar em qualquer coluna; o nome fica imediatamente antes
        code_col, n3 = find_nuts3_code_in_row(row)
        if n3 is None:
            continue

        nums = numbers_after(row, code_col)
        if len(nums) < 30:
            continue
        vals = nums[0:30]  # 30 valores: HM x10, H x10, M x10

        region = clean_text(row.iloc[code_col - 1]) if code_col >= 1 else ""
        n2c, n2n = NUTS3_TO_NUTS2[n3]

        rec = {
            "school_year": current_sy,
            "year_start": current_ys,
            "year_end": current_ye,
            "nuts2_code": n2c,
            "nuts2_nome": n2n,
            "nuts3_code": n3,
            "nuts3": region,
        }
        for i, suf in enumerate(AGE_SUFFIX):
            rec["teachers_" + suf] = vals[i]
        for i, suf in enumerate(AGE_SUFFIX):
            rec["teachers_male_" + suf] = vals[10 + i]
        for i, suf in enumerate(AGE_SUFFIX):
            rec["teachers_female_" + suf] = vals[20 + i]

        rows.append(rec)

    out = pd.DataFrame(rows)
    print(f"  linhas NUTS3: {len(out):,}")
    return out

# ============================================================
# ESCRITA DUPLA
# ============================================================
def save_final(sheets, filename):
    """Grava o mesmo livro Excel em todos os destinos finais."""
    written = []
    for d in FINAL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=name, index=False)
        written.append(path)
    return written

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("07_build_teachers_age_sex_ine")
    print("=" * 80)
    print(f"  input   : {RAW_DIR}")
    for d in FINAL_DIRS:
        print(f"  output  : {d / OUT_NAME}")
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

    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["school_year", "nuts3_code"], keep="first")

    tot = df["teachers_total"].replace(0, np.nan)
    df["teachers_55_plus"] = df[["teachers_55_59", "teachers_60_plus"]].sum(axis=1, min_count=1)
    df["share_55_plus"] = df["teachers_55_plus"] / tot
    df["share_60_plus"] = df["teachers_60_plus"] / tot
    df["teachers_male_share"] = df["teachers_male_total"] / tot
    df["teachers_female_share"] = df["teachers_female_total"] / tot
    df["check_sex_total"] = (
        df["teachers_total"] - (df["teachers_male_total"] + df["teachers_female_total"])
    )

    col_order = [
        "school_year", "year_start", "year_end",
        "nuts2_code", "nuts2_nome", "nuts3", "nuts3_code",
        "teachers_total", "teachers_lt25", "teachers_25_29", "teachers_30_34",
        "teachers_35_39", "teachers_40_44", "teachers_45_49", "teachers_50_54",
        "teachers_55_59", "teachers_60_plus",
        "teachers_male_total", "teachers_male_lt25", "teachers_male_25_29",
        "teachers_male_30_34", "teachers_male_35_39", "teachers_male_40_44",
        "teachers_male_45_49", "teachers_male_50_54", "teachers_male_55_59",
        "teachers_male_60_plus",
        "teachers_female_total", "teachers_female_lt25", "teachers_female_25_29",
        "teachers_female_30_34", "teachers_female_35_39", "teachers_female_40_44",
        "teachers_female_45_49", "teachers_female_50_54", "teachers_female_55_59",
        "teachers_female_60_plus",
        "teachers_55_plus", "share_55_plus", "share_60_plus",
        "teachers_male_share", "teachers_female_share", "check_sex_total",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order].sort_values(["nuts3", "school_year"]).reset_index(drop=True)

    check_year = (
        df.groupby("school_year", as_index=False)
        .agg(
            n_nuts3=("nuts3_code", "nunique"),
            teachers_total=("teachers_total", "sum"),
            max_abs_sex_diff=("check_sex_total", lambda s: s.abs().max()),
        )
        .sort_values("school_year")
    )

    max_sex_diff = 0
    if len(df):
        max_sex_diff = int(df["check_sex_total"].abs().max())

    summary = pd.DataFrame({
        "metric": ["rows", "n_nuts3", "first_year", "last_year", "max_abs_sex_diff"],
        "value": [
            len(df),
            df["nuts3_code"].nunique(),
            df["school_year"].min(),
            df["school_year"].max(),
            max_sex_diff,
        ],
    })

    sheets = {
        "teachers_panel": df,
        "checks_by_year": check_year,
        "summary": summary,
    }
    written = save_final(sheets, OUT_NAME)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    for p in written:
        print(p)
    print()
    print(summary.to_string(index=False))
    print()
    print(check_year.to_string(index=False))


if __name__ == "__main__":
    main()