
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
PROJECT_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
BASE_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"

RAW_DIR = BASE_DIR / "01_raw" / "dgeec" / "graduates"
ANALYSIS_DIR = BASE_DIR / "03_analysis"
MODELS_DATA_DIR = MODELS_DIR / "00_data"

OUT_NAME = "teacher_supply_panel.xlsx"

# os dois destinos finais; acrescentar aqui se algum dia houver um terceiro
FINAL_DIRS = [ANALYSIS_DIR, MODELS_DATA_DIR]

# ============================================================
# CONFIG
# ============================================================
# Usa a Tabela 3.1: diplomados que conferem nivel CITE de ensino superior,
# por area geral de educacao e formacao, 1996/97 a 2024/25.

AREA_MAP = {
    "total": "grad_total",
    "educacao": "grad_education",
    "artes e humanidades": "grad_arts_humanities",
    "ciencias sociais jornalismo e informacao": "grad_social_sciences",
    "ciencias empresariais administracao e direito": "grad_business_law",
    "ciencias naturais matematica e estatistica": "grad_science_math",
    "tecnologias da informacao e comunicacao tic": "grad_ict",
    "tecnologias da informacao e comunicacao": "grad_ict",
    "engenharia industrias transformadoras e construcao": "grad_engineering",
    "agricultura silvicultura pescas e ciencias veterinarias": "grad_agriculture",
    "saude e protecao social": "grad_health",
    "servicos": "grad_services",
    "area desconhecida": "grad_unknown",
}

AREA_ORDER = [
    "grad_total",
    "grad_education",
    "grad_arts_humanities",
    "grad_social_sciences",
    "grad_business_law",
    "grad_science_math",
    "grad_ict",
    "grad_engineering",
    "grad_agriculture",
    "grad_health",
    "grad_services",
    "grad_unknown",
]

TARGET_SHEET_TOKENS = ["tabela 3 1", "tabela 3.1", "3 1 e 3 2", "tabela 3"]

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


def to_int(x):
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float, np.integer, np.floating)):
        if np.isnan(x):
            return 0
        return int(round(x))
    s = str(x).strip()
    if s in {"", "-", "\u2013", "\\-", "x", "X", "//", "nan", "NaN", ":"}:
        return 0
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", ".").replace("\\-", "")
    if s in {"", "-"}:
        return 0
    try:
        return int(round(float(s)))
    except Exception:
        return 0


def parse_school_year_header(x):
    t = clean_text(x)
    m = re.match(r"^(\d{4})\s*/\s*(\d{2,4})$", t)
    if m:
        y1 = int(m.group(1))
        y2raw = m.group(2)
        if len(y2raw) == 4:
            y2 = int(y2raw)
        else:
            # normalizar: 1996/97 -> 1996/1997 ; 1999/00 -> 1999/2000
            y2 = (y1 // 100) * 100 + int(y2raw)
            if y2 <= y1:
                y2 += 100
        return f"{y1}/{y2}", y1
    return None, None


def excel_engine(path):
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"


def find_target_sheet(xls):
    for s in xls.sheet_names:
        n = norm_txt(s)
        for tok in TARGET_SHEET_TOKENS:
            if tok in n:
                return s
    return None

# ============================================================
# PARSE TABELA 3.1
# ============================================================
def parse_table31(path):
    print(f"Ler: {path.name}")
    xls = pd.ExcelFile(path, engine=excel_engine(path))

    sheet = find_target_sheet(xls)
    if sheet is None:
        print("  nao encontrei folha da Tabela 3.1")
        return pd.DataFrame()

    df = pd.read_excel(path, sheet_name=sheet, header=None, engine=excel_engine(path))
    ncols = len(df.columns)

    # 1) encontrar a linha do cabecalho com os anos letivos
    header_row = None
    year_cols = {}
    for r in range(min(30, len(df))):
        cols = {}
        for c, v in enumerate(df.iloc[r].tolist()):
            sy, ys = parse_school_year_header(v)
            if sy is not None:
                cols[c] = (sy, ys)
        if len(cols) >= 10:
            header_row = r
            year_cols = cols
            break

    if header_row is None:
        print("  nao encontrei linha de anos letivos")
        return pd.DataFrame()

    # 2) percorrer abaixo do cabecalho ate a segunda "TOTAL" (inicio da 3.2)
    records = []
    total_seen = 0
    for r in range(header_row + 1, len(df)):
        label = norm_txt(df.iloc[r, 0])
        if label == "":
            continue
        if (label.startswith("fonte") or label.startswith("tabela 3 2")
                or label.startswith("tabela 3.2")):
            break

        if label == "total":
            total_seen += 1
            if total_seen >= 2:
                break

        area_col = AREA_MAP.get(label)
        if area_col is None:
            continue

        for c, pair in year_cols.items():
            sy, ys = pair
            if c >= ncols:
                continue
            val = to_int(df.iloc[r, c])
            records.append({
                "school_year": sy,
                "year_start": ys,
                "area_col": area_col,
                "graduates": val,
            })

    if not records:
        print("  sem dados extraidos")
        return pd.DataFrame()

    long = pd.DataFrame(records)
    wide = long.pivot_table(
        index=["school_year", "year_start"],
        columns="area_col",
        values="graduates",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    for c in AREA_ORDER:
        if c not in wide.columns:
            wide[c] = 0
        wide[c] = pd.to_numeric(wide[c], errors="coerce").fillna(0).astype(int)

    y0 = wide["school_year"].min()
    y1 = wide["school_year"].max()
    print(f"  anos: {y0} -> {y1} ({len(wide)} anos)")
    return wide

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
    print("06_build_graduates")
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
        d = parse_table31(f)
        if len(d):
            parts.append(d)

    if not parts:
        raise RuntimeError("Nao consegui extrair a Tabela 3.1 de nenhum ficheiro.")

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values("year_start")
    panel = panel.drop_duplicates("school_year", keep="first").reset_index(drop=True)

    # variaveis derivadas para teacher supply
    panel["teacher_supply_core"] = panel["grad_education"]
    panel["teacher_supply_stem"] = (
        panel["grad_education"] + panel["grad_science_math"] + panel["grad_ict"]
    )
    panel["teacher_supply_extended"] = (
        panel["grad_education"] + panel["grad_science_math"] + panel["grad_ict"]
        + panel["grad_arts_humanities"] + panel["grad_social_sciences"]
    )
    panel["share_education"] = np.where(
        panel["grad_total"] > 0,
        panel["grad_education"] / panel["grad_total"],
        np.nan,
    )

    derived = [
        "teacher_supply_core", "teacher_supply_stem",
        "teacher_supply_extended", "share_education",
    ]
    out_cols = ["school_year", "year_start"] + AREA_ORDER + derived
    panel = panel[out_cols]

    # check: soma das areas (sem o total) contra o total publicado
    area_bands = [c for c in AREA_ORDER if c != "grad_total"]
    check = panel[["school_year", "grad_total"]].copy()
    check["sum_areas"] = panel[area_bands].sum(axis=1)
    check["diff_total_minus_areas"] = check["grad_total"] - check["sum_areas"]

    max_diff = np.nan
    if len(check):
        max_diff = check["diff_total_minus_areas"].abs().max()

    last_edu = np.nan
    last_sci = np.nan
    last_ict = np.nan
    if len(panel):
        last = panel.iloc[-1]
        last_edu = int(last["grad_education"])
        last_sci = int(last["grad_science_math"])
        last_ict = int(last["grad_ict"])

    summary = pd.DataFrame({
        "metric": [
            "rows", "first_school_year", "last_school_year", "max_abs_total_diff",
            "grad_education_last", "grad_science_math_last", "grad_ict_last",
        ],
        "value": [
            len(panel),
            panel["school_year"].min(),
            panel["school_year"].max(),
            max_diff,
            last_edu,
            last_sci,
            last_ict,
        ],
    })

    sheets = {
        "teacher_supply_panel": panel,
        "check_totais": check,
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
    print(panel.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()