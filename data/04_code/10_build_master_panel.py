
from pathlib import Path
import re
import unicodedata
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting\data")
PROCESSED_DIR = BASE_DIR / "02_processed"
ANALYSIS_DIR = BASE_DIR / "03_analysis"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TEACHERS_FILE = ANALYSIS_DIR / "teachers_panel_nuts2024.xlsx"
STUDENTS_FILE = PROCESSED_DIR / "students_nuts3.xlsx"
AGEING_FILE = PROCESSED_DIR / "ageing_index_nuts3.xlsx"
RENEWAL_FILE = PROCESSED_DIR / "renewal_index_nuts3.xlsx"
POP_HIST_FILE = PROCESSED_DIR / "population_historical_nuts3.xlsx"
POP_FUTURE_FILE = PROCESSED_DIR / "population_projections_nuts3.xlsx"

OUT_FILE = PROCESSED_DIR / "master_panel_nuts3.xlsx"

# ============================================================
# CONFIG
# ============================================================
# 24 NUTS3 do Continente (exclui Acores 200 e Madeira 300)
NUTS3_CONTINENTE = {
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
}
CONTINENTE_CODES = set(NUTS3_CONTINENTE.keys())

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
    x = x.replace(" ", "")
    return x


def year_start_from_school_year(sy):
    if pd.isna(sy):
        return np.nan
    m = re.search(r"(\d{4})", str(sy))
    if m:
        return int(m.group(1))
    return np.nan


def school_year_label(y):
    if pd.isna(y):
        return ""
    y = int(y)
    return str(y) + "/" + str(y + 1)


def read_sheet(path, preferred):
    if not path.exists():
        raise FileNotFoundError(
            "Nao encontrei: " + str(path) + "\n"
            "Confirma que os scripts anteriores ja correram."
        )
    xls = pd.ExcelFile(path, engine="openpyxl")
    for s in preferred:
        if s in xls.sheet_names:
            return pd.read_excel(path, sheet_name=s, engine="openpyxl")
    first = xls.sheet_names[0]
    return pd.read_excel(path, sheet_name=first, engine="openpyxl")


# mapa nome_norm -> nuts3_code (para casar os professores).
# Os aliases acentuados que existiam antes eram redundantes: o norm_txt ja
# remove acentos, por isso "Cavado" e "Cavado" com acento dao a mesma chave.
NAME_TO_CODE = {}
for c in NUTS3_CONTINENTE:
    n3n = NUTS3_CONTINENTE[c][2]
    NAME_TO_CODE[norm_txt(n3n)] = c

# ============================================================
# LOADERS
# ============================================================
def load_teachers():
    df = read_sheet(TEACHERS_FILE, ["teachers_panel_nuts2024"])
    df.columns = [str(c).strip() for c in df.columns]

    # identificar coluna do nome NUTS3
    n3name_col = None
    for cand in ["nuts3_2024_nome", "nuts3_2024", "nuts3_nome", "nuts3"]:
        if cand in df.columns:
            n3name_col = cand
            break
    if n3name_col is None:
        raise RuntimeError("Nao encontrei coluna de NUTS3 no ficheiro de professores.")

    df["year_start"] = df["school_year"].map(year_start_from_school_year)
    df["nuts3_code"] = df[n3name_col].map(lambda x: NAME_TO_CODE.get(norm_txt(x), ""))

    sem_codigo = df["nuts3_code"] == ""
    unmapped = sorted(set(df.loc[sem_codigo, n3name_col].map(str)))
    if unmapped:
        print("AVISO professores: NUTS3 sem codigo:", unmapped[0:10])

    df = df[df["nuts3_code"].isin(CONTINENTE_CODES)].copy()

    teacher_cols = []
    for c in df.columns:
        if c.startswith("teachers_") or c.startswith("share_"):
            teacher_cols.append(c)
        elif c in ("schools_count", "municipalities_count"):
            teacher_cols.append(c)

    keep = ["year_start", "nuts3_code"] + teacher_cols
    return df[keep].drop_duplicates(["year_start", "nuts3_code"])


def load_students():
    df = read_sheet(STUDENTS_FILE, ["students_panel"])
    df.columns = [str(c).strip() for c in df.columns]
    df["year_start"] = df["school_year"].map(year_start_from_school_year)
    df["nuts3_code"] = df["nuts3_code"].map(code)
    df = df[df["nuts3_code"].isin(CONTINENTE_CODES)].copy()
    student_cols = [c for c in df.columns if c.startswith("students_")]
    keep = ["year_start", "nuts3_code"] + student_cols
    return df[keep].drop_duplicates(["year_start", "nuts3_code"])


def load_indicator(path, cols):
    df = read_sheet(path, ["nuts3"])
    df.columns = [str(c).strip() for c in df.columns]
    if "ano" in df.columns:
        df = df.rename(columns={"ano": "year_start"})
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df["nuts3_code"] = df["nuts3_code"].map(code)
    df = df[df["nuts3_code"].isin(CONTINENTE_CODES)].copy()
    have = [c for c in cols if c in df.columns]
    keep = ["year_start", "nuts3_code"] + have
    return df[keep].drop_duplicates(["year_start", "nuts3_code"])


def load_pop_hist():
    df = read_sheet(POP_HIST_FILE, ["nuts3"])
    df.columns = [str(c).strip() for c in df.columns]
    if "ano" in df.columns:
        df = df.rename(columns={"ano": "year_start"})
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df["nuts3_code"] = df["nuts3_code"].map(code)
    df = df[df["nuts3_code"].isin(CONTINENTE_CODES)].copy()

    pop_cols = [c for c in df.columns if c.startswith("pop_")]
    rename = {}
    for c in pop_cols:
        rename[c] = "hist_" + c
    df = df.rename(columns=rename)

    keep = ["year_start", "nuts3_code"] + list(rename.values())
    return df[keep].drop_duplicates(["year_start", "nuts3_code"])


def load_pop_future():
    df = read_sheet(POP_FUTURE_FILE, ["nuts3_population"])
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={"year": "year_start"})
    df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")
    df["nuts3_code"] = df["nuts3_code"].map(code)
    df = df[df["nuts3_code"].isin(CONTINENTE_CODES)].copy()
    return df

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("10_build_master_panel")
    print("=" * 80)
    print("  professores : " + str(TEACHERS_FILE))
    print("  alunos      : " + str(STUDENTS_FILE))
    print("  envelhec.   : " + str(AGEING_FILE))
    print("  renovacao   : " + str(RENEWAL_FILE))
    print("  pop hist    : " + str(POP_HIST_FILE))
    print("  pop futuro  : " + str(POP_FUTURE_FILE))
    print("  output      : " + str(OUT_FILE))
    print()

    log = []

    teachers = load_teachers()
    students = load_students()
    ageing = load_indicator(
        AGEING_FILE,
        ["indice_envelhecimento_mean", "indice_envelhecimento_median"],
    )
    renewal = load_indicator(
        RENEWAL_FILE,
        ["indice_renovacao_pop_ativa_mean", "indice_renovacao_pop_ativa_median"],
    )
    pop_hist = load_pop_hist()

    # ---- HISTORICO: ancora = professores (series longas 2011-2024) ----
    hist = teachers.copy()
    hist = hist.merge(students, on=["year_start", "nuts3_code"], how="outer")
    hist = hist.merge(ageing, on=["year_start", "nuts3_code"], how="left")
    hist = hist.merge(renewal, on=["year_start", "nuts3_code"], how="left")
    hist = hist.merge(pop_hist, on=["year_start", "nuts3_code"], how="left")

    # metadados NUTS
    meta_rows = []
    for c in NUTS3_CONTINENTE:
        v = NUTS3_CONTINENTE[c]
        meta_rows.append({
            "nuts3_code": c,
            "nuts2_code": v[0],
            "nuts2_nome": v[1],
            "nuts3_nome": v[2],
        })
    meta = pd.DataFrame(meta_rows)

    hist = hist.merge(meta, on="nuts3_code", how="left")
    hist["school_year"] = hist["year_start"].map(school_year_label)
    hist["data_scope"] = "historical"
    hist["scenario"] = "observed"

    front = [
        "data_scope", "school_year", "year_start", "scenario",
        "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome",
    ]
    other = [c for c in hist.columns if c not in front]
    hist = hist[front + other]
    hist = hist.sort_values(["nuts3_code", "year_start"]).reset_index(drop=True)

    hist_y0 = int(hist["year_start"].min())
    hist_y1 = int(hist["year_start"].max())
    log.append({
        "bloco": "historical",
        "linhas": len(hist),
        "anos": str(hist_y0) + "-" + str(hist_y1),
        "n_nuts3": hist["nuts3_code"].nunique(),
    })

    # ---- FUTURO: populacao imputada + indicadores (ultimo valor conhecido) ----
    fut = load_pop_future()

    last_age = ageing.sort_values("year_start")
    last_age = last_age.drop_duplicates("nuts3_code", keep="last")
    last_age = last_age[["nuts3_code", "indice_envelhecimento_mean"]]

    last_ren = renewal.sort_values("year_start")
    last_ren = last_ren.drop_duplicates("nuts3_code", keep="last")
    last_ren = last_ren[["nuts3_code", "indice_renovacao_pop_ativa_mean"]]

    fut = fut.merge(last_age, on="nuts3_code", how="left")
    fut = fut.merge(last_ren, on="nuts3_code", how="left")
    fut["data_scope"] = "forecast"
    fut["school_year"] = fut["year_start"].map(school_year_label)

    fut_front = [c for c in front if c in fut.columns]
    fut_other = [c for c in fut.columns if c not in fut_front]
    fut = fut[fut_front + fut_other]
    fut = fut.sort_values(["scenario", "year_start", "nuts3_code"])
    fut = fut.reset_index(drop=True)

    fut_y0 = int(fut["year_start"].min())
    fut_y1 = int(fut["year_start"].max())
    log.append({
        "bloco": "forecast",
        "linhas": len(fut),
        "anos": str(fut_y0) + "-" + str(fut_y1),
        "n_nuts3": fut["nuts3_code"].nunique(),
    })

    # ---- MASTER: concatenar ----
    all_cols = list(dict.fromkeys(list(hist.columns) + list(fut.columns)))
    master = pd.concat(
        [hist.reindex(columns=all_cols), fut.reindex(columns=all_cols)],
        ignore_index=True,
    )

    # ---- checks ----
    # O agg e montado condicionalmente: antes, se a coluna faltasse, produzia
    # uma coluna chamada teachers_total com uma CONTAGEM de linhas la dentro.
    agg_kwargs = {"n_nuts3": ("nuts3_code", "nunique")}
    if "teachers_total" in hist.columns:
        agg_kwargs["teachers_total"] = ("teachers_total", "sum")
    if "students_total" in hist.columns:
        agg_kwargs["students_total"] = ("students_total", "sum")

    check_hist = hist.groupby("year_start", as_index=False).agg(**agg_kwargs)

    merge_report = pd.DataFrame(log)

    summary = pd.DataFrame({
        "metric": [
            "master_rows", "hist_rows", "fut_rows", "n_nuts3",
            "hist_first", "hist_last", "fut_first", "fut_last",
        ],
        "value": [
            len(master), len(hist), len(fut),
            master["nuts3_code"].nunique(),
            hist_y0, hist_y1, fut_y0, fut_y1,
        ],
    })

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="master_all", index=False)
        hist.to_excel(writer, sheet_name="historical", index=False)
        fut.to_excel(writer, sheet_name="forecast", index=False)
        check_hist.to_excel(writer, sheet_name="checks_hist_by_year", index=False)
        merge_report.to_excel(writer, sheet_name="merge_report", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    print("=" * 80)
    print("OUTPUT")
    print("=" * 80)
    print(OUT_FILE)
    print()
    print(summary.to_string(index=False))
    print()
    print("Historico por ano:")
    print(check_hist.to_string(index=False))


if __name__ == "__main__":
    main()