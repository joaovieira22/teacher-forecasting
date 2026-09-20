
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

RAW_DIR = BASE_DIR / "01_raw" / "dgeec" / "teachers"
ANALYSIS_DIR = BASE_DIR / "03_analysis"
MODELS_DATA_DIR = MODELS_DIR / "00_data"

OUT_NAME = "teachers_panel_nuts2024.xlsx"

# os dois destinos finais; acrescentar aqui se algum dia houver um terceiro
FINAL_DIRS = [ANALYSIS_DIR, MODELS_DATA_DIR]

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
    x = x.replace("\u00ba", "").replace("\u00aa", "")
    x = re.sub(r"[^a-z0-9]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def clean_num(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        if np.isnan(x):
            return 0.0
        return float(x)
    s = str(x).strip()
    if s in {"", "-", "\u2013", "N/A", "n/a", "nan", "NaN"}:
        return 0.0
    s = s.replace("\u00a0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return 0.0


def std_school_year(x):
    t = clean_text(x)
    m = re.search(r"(\d{4})\s*/\s*(\d{4})", t)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    m = re.search(r"(\d{4})\s*/\s*(\d{2})", t)
    if m:
        y1 = int(m.group(1))
        y2short = int(m.group(2))
        y2 = (y1 // 100) * 100 + y2short
        if y2 <= y1:
            y2 += 100
        return f"{y1}/{y2}"
    return None


def infer_school_year_from_filename(name):
    m = re.search(r"cnt\D*?(\d{2})(\d{2})", norm_txt(name))
    if m:
        y1 = 2000 + int(m.group(1))
        y2 = 2000 + int(m.group(2))
        return f"{y1}/{y2}"
    m = re.search(r"(\d{4})\s*_?\s*(\d{4})", name)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def excel_engine(path):
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"


def map_col(colname):
    c = norm_txt(colname)
    if "ano letivo" in c:
        return "school_year"
    if "codigo da escola" in c:
        return "school_code"
    if "nome da escola" in c:
        return "school_name"
    if c == "rede":
        return "network"
    # NUTS III primeiro (mais especifico) para nao ser apanhado por "nuts ii"
    if "nuts iii" in c and "2024" in c:
        return "nuts3_2024_src"
    if "nuts iii" in c and "2013" in c:
        return "nuts3_2013_src"
    if "nuts ii" in c and "2024" in c:
        return "nuts2_2024_src"
    if "nuts ii" in c and "2013" in c:
        return "nuts2_2013_src"
    if c == "municipio":
        return "municipio"
    if c == "natureza":
        return "nature"
    if "ciclo de docencia" in c:
        return "teaching_cycle"
    if "funcoes principais" in c:
        return "teacher_function"
    if "grupo de recrutamento" in c:
        return "recruitment_group"
    if "docentes" in c:
        return "teachers"
    return None


def find_header_row(df):
    for r in range(min(30, len(df))):
        row = [norm_txt(v) for v in df.iloc[r].tolist()]
        joined = " ".join(row)
        if ("codigo da escola" in joined and "docentes" in joined
                and "municipio" in joined):
            return r
    return None


def read_file(path):
    df0 = pd.read_excel(path, header=None, engine=excel_engine(path))
    hr = find_header_row(df0)
    if hr is None:
        return None

    raw_headers = df0.iloc[hr].tolist()
    headers = [map_col(v) for v in raw_headers]

    # deteta no cabecalho se este ficheiro tem NUTS 2024
    file_has_nuts2024 = False
    for v in raw_headers:
        n = norm_txt(v)
        if "nuts iii" in n and "2024" in n:
            file_has_nuts2024 = True
            break

    data = df0.iloc[hr + 1:].copy()
    data.columns = headers
    keep = [c for c in data.columns if c is not None]
    data = data.loc[:, keep]
    data = data.loc[:, ~pd.Index(data.columns).duplicated()]
    data = data.dropna(how="all")

    # ano letivo
    if "school_year" in data.columns:
        data["school_year"] = data["school_year"].map(std_school_year)
        if data["school_year"].isna().all():
            data["school_year"] = infer_school_year_from_filename(path.name)
    else:
        data["school_year"] = infer_school_year_from_filename(path.name)

    text_cols = [
        "municipio", "nature", "teaching_cycle", "teacher_function",
        "recruitment_group", "nuts2_2024_src", "nuts3_2024_src",
        "nuts2_2013_src", "nuts3_2013_src",
    ]
    for c in text_cols:
        if c in data.columns:
            data[c] = data[c].map(clean_text)
        else:
            data[c] = ""

    if "teachers" in data.columns:
        data["teachers"] = data["teachers"].map(clean_num)
    else:
        data["teachers"] = 0.0

    data["municipio_norm"] = data["municipio"].map(norm_txt)
    data["source_file"] = path.name
    data["has_nuts2024"] = int(file_has_nuts2024)

    keep_rows = (data["teachers"] > 0) & (data["municipio_norm"] != "")
    data = data[keep_rows].copy()
    return data

# ============================================================
# FLAGS
# ============================================================
def contains(x, patterns):
    n = norm_txt(x)
    for p in patterns:
        if p in n:
            return True
    return False


def _has_cycle(x, n):
    """True se o ciclo n aparecer no texto, robusto a '1.o', '1 o', '1o'.

    Motivo: unicodedata.NFKD decompoe o ordinal masculino, pelo que
    'Ensino basico - 1.o Ciclo' -> 'ensino basico 1 o ciclo'.
    Um simples 'in "1 ciclo"' falha; usamos regex tolerante ao 'o' intermedio.
    """
    t = norm_txt(x)
    pat = r"(^|\D)" + str(n) + r"\s*o?\s*ciclo"
    return bool(re.search(pat, t))


def add_flags(df):
    df = df.copy()
    df["flag_public"] = df["nature"].map(lambda x: contains(x, ["publico"]))
    df["flag_private"] = df["nature"].map(lambda x: contains(x, ["privado"]))
    df["flag_teaching"] = df["teacher_function"].map(
        lambda x: contains(x, ["com funcoes letivas"]))
    df["flag_non_teaching"] = df["teacher_function"].map(
        lambda x: contains(x, ["sem funcoes letivas", "nao letivas", "sem funcoes"]))
    df["flag_pre_school"] = df["teaching_cycle"].map(
        lambda x: contains(x, ["pre escolar"]))
    # ciclos por regex tolerante ao 'o' do ordinal (ver _has_cycle)
    df["flag_basic_1"] = df["teaching_cycle"].map(lambda x: _has_cycle(x, 1))
    df["flag_basic_2"] = df["teaching_cycle"].map(lambda x: _has_cycle(x, 2))
    df["flag_basic_3_sec"] = df["teaching_cycle"].map(
        lambda x: _has_cycle(x, 3) or contains(x, ["secundario"]))
    df["flag_special_ed"] = df["teaching_cycle"].map(
        lambda x: contains(x, ["educacao especial"]))

    groups = {
        "math": ["matematica"],
        "portuguese": ["portugues"],
        "english": ["ingles"],
        "history": ["historia"],
        "geography": ["geografia"],
        "physics_chemistry": ["fisica e quimica"],
        "biology_geology": ["biologia e geologia"],
        "informatics": ["informatica"],
        "physical_education": ["educacao fisica"],
    }
    for name, pats in groups.items():
        col = "flag_" + name
        df[col] = df["recruitment_group"].map(lambda x, p=pats: contains(x, p))
    return df

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
def first_nonempty(series):
    vals = series.map(clean_text).replace("", np.nan).dropna()
    return vals.iloc[0] if len(vals) else ""


def main():
    print("=" * 80)
    print("08_build_teachers_cycle_subject_dgeec")
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
        raise FileNotFoundError(f"Nao encontrei ficheiros em: {RAW_DIR}")

    parts = []
    for f in files:
        d = read_file(f)
        if d is not None and len(d):
            anos = d["school_year"].dropna().unique().tolist()
            ano = anos[0] if anos else "?"
            print(f"Ler: {f.name}  linhas: {len(d):,}  ano: {ano}")
            parts.append(d)

    if not parts:
        raise RuntimeError("Nao consegui extrair dados de nenhum ficheiro.")

    raw = pd.concat(parts, ignore_index=True)

    # ---- crosswalk municipio -> NUTS 2024 (dos ficheiros que ja tem NUTS2024) ----
    has_n3 = raw["nuts3_2024_src"].map(norm_txt) != ""
    cw_src = raw[has_n3].copy()
    if cw_src.empty:
        raise RuntimeError(
            "Nenhum ficheiro tem NUTS 2024.\n"
            "Nao consigo construir o crosswalk municipio -> NUTS 2024."
        )

    crosswalk = (
        cw_src.groupby("municipio_norm")
        .agg(
            municipio=("municipio", "first"),
            nuts2_2024=("nuts2_2024_src", first_nonempty),
            nuts3_2024=("nuts3_2024_src", first_nonempty),
        )
        .reset_index()
    )

    # ---- aplicar o crosswalk a todos os registos, por nome de municipio ----
    cw_cols = ["municipio_norm", "nuts2_2024", "nuts3_2024"]
    df = raw.merge(crosswalk[cw_cols], on="municipio_norm", how="left")

    no_n3 = df["nuts3_2024"].isna() | df["nuts3_2024"].eq("")
    unmapped = df[no_n3].copy()
    mapped = df[~no_n3].copy()

    mapped = add_flags(mapped)

    group_cols = ["school_year", "nuts2_2024", "nuts3_2024"]

    def agg_flag(frame, flag, out):
        sub = frame[frame[flag]]
        if len(sub) == 0:
            return pd.DataFrame(columns=group_cols + [out])
        return sub.groupby(group_cols, as_index=False).agg(**{out: ("teachers", "sum")})

    base = mapped.groupby(group_cols, as_index=False).agg(
        teachers_total=("teachers", "sum"),
        schools_count=("school_code", pd.Series.nunique),
        municipalities_count=("municipio", pd.Series.nunique),
    )

    flag_specs = [
        ("flag_public", "teachers_public"),
        ("flag_private", "teachers_private"),
        ("flag_teaching", "teachers_teaching_functions"),
        ("flag_non_teaching", "teachers_non_teaching_functions"),
        ("flag_pre_school", "teachers_pre_school"),
        ("flag_basic_1", "teachers_basic_1"),
        ("flag_basic_2", "teachers_basic_2"),
        ("flag_basic_3_sec", "teachers_basic_3_secondary"),
        ("flag_special_ed", "teachers_special_education"),
        ("flag_math", "teachers_math"),
        ("flag_portuguese", "teachers_portuguese"),
        ("flag_english", "teachers_english"),
        ("flag_history", "teachers_history"),
        ("flag_geography", "teachers_geography"),
        ("flag_physics_chemistry", "teachers_physics_chemistry"),
        ("flag_biology_geology", "teachers_biology_geology"),
        ("flag_informatics", "teachers_informatics"),
        ("flag_physical_education", "teachers_physical_education"),
    ]

    panel = base.copy()
    for flag, out in flag_specs:
        piece = agg_flag(mapped, flag, out)
        panel = panel.merge(piece, on=group_cols, how="left")

    teacher_cols = [c for c in panel.columns if c.startswith("teachers_")]
    panel[teacher_cols] = panel[teacher_cols].fillna(0)

    panel = panel.rename(columns={
        "nuts2_2024": "nuts2_2024_nome",
        "nuts3_2024": "nuts3_2024_nome",
    })

    tot = panel["teachers_total"]
    panel["share_public"] = np.where(tot > 0, panel["teachers_public"] / tot, np.nan)
    panel["share_private"] = np.where(tot > 0, panel["teachers_private"] / tot, np.nan)
    panel["share_non_teaching"] = np.where(
        tot > 0, panel["teachers_non_teaching_functions"] / tot, np.nan)

    panel = panel.sort_values(["nuts3_2024_nome", "school_year"]).reset_index(drop=True)

    check_year = (
        panel.groupby("school_year", as_index=False)
        .agg(
            n_nuts3=("nuts3_2024_nome", "nunique"),
            teachers_total=("teachers_total", "sum"),
        )
        .sort_values("school_year")
    )

    # diagnostico dos ciclos: basic_1 e basic_2 nao devem estar a zero
    cycle_cols = [
        "teachers_pre_school", "teachers_basic_1", "teachers_basic_2",
        "teachers_basic_3_secondary", "teachers_special_education",
    ]
    cycle_sums = []
    for c in cycle_cols:
        cycle_sums.append(int(panel[c].sum()) if c in panel.columns else 0)
    cycle_check = pd.DataFrame({"cycle_col": cycle_cols, "sum": cycle_sums})

    summary = pd.DataFrame({
        "metric": [
            "raw_rows", "mapped_rows", "unmapped_rows", "panel_rows",
            "first_year", "last_year", "n_nuts3", "crosswalk_municipios",
        ],
        "value": [
            len(raw), len(mapped), len(unmapped), len(panel),
            panel["school_year"].min(), panel["school_year"].max(),
            panel["nuts3_2024_nome"].nunique(), len(crosswalk),
        ],
    })

    unmapped_sheet = (
        unmapped[["school_year", "municipio", "source_file"]].drop_duplicates()
    )

    sheets = {
        "teachers_panel_nuts2024": panel,
        "crosswalk_municipio_2024": crosswalk.sort_values("municipio"),
        "checks_by_year": check_year,
        "cycle_check": cycle_check,
        "unmapped": unmapped_sheet,
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
    print("Ciclos (soma de docentes) -- basic_1 e basic_2 nao devem ser 0:")
    print(cycle_check.to_string(index=False))
    print()
    print("Checks por ano:")
    print(check_year.to_string(index=False))


if __name__ == "__main__":
    main()
