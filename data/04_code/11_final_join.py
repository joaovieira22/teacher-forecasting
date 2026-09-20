from pathlib import Path
import re
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "02_processed"
ANALYSIS_DIR = DATA_DIR / "03_analysis"
MODELS_DATA_DIR = BASE_DIR / "models" / "00_data"

MASTER_FILE = PROCESSED_DIR / "master_panel_nuts3.xlsx"
TEACHERS_FILE = ANALYSIS_DIR / "teachers_panel.xlsx"

OUT_NAME = "master_panel_nuts3_with_age.xlsx"

# os dois destinos finais; acrescentar aqui se algum dia houver um terceiro
FINAL_DIRS = [ANALYSIS_DIR, MODELS_DATA_DIR]

# cenario a eliminar dos projetados
DROP_SCENARIOS = {
    "no_migration",
    "sem_migracoes",
    "sem migracoes",
    "sem migracoes",
}

# ============================================================
# HELPERS
# ============================================================
def code(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.sub(r"\.0$", "", x)
    return x.replace(" ", "")


def norm(x):
    return re.sub(r"\s+", "_", str(x).strip().lower())


def year_start_from_school_year(sy):
    if pd.isna(sy):
        return np.nan
    m = re.search(r"(\d{4})", str(sy))
    if m:
        return int(m.group(1))
    return np.nan


def read_first_sheet(path, preferred):
    if not path.exists():
        raise FileNotFoundError(
            "Nao encontrei: " + str(path) + "\n"
            "Confirma que os scripts 07 e 10 ja correram."
        )
    xls = pd.ExcelFile(path, engine="openpyxl")
    sh = xls.sheet_names[0]
    for s in preferred:
        if s in xls.sheet_names:
            sh = s
            break
    df = pd.read_excel(path, sheet_name=sh, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric(df, col):
    """Coluna como numerico, ou uma serie de NaN se a coluna nao existir."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


AGE_COUNT_COLS = [
    "teachers_lt25", "teachers_25_29", "teachers_30_34", "teachers_35_39",
    "teachers_40_44", "teachers_45_49", "teachers_50_54", "teachers_55_59",
    "teachers_60_plus", "teachers_55_plus",
]
SEX_COLS = [
    "teachers_male_total", "teachers_female_total",
    "teachers_male_share", "teachers_female_share",
]
SHARE_COLS = ["share_55_plus", "share_60_plus"]

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
            for name in sheets:
                sheets[name].to_excel(writer, sheet_name=name, index=False)
        written.append(path)
    return written

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("11_final_join")
    print("=" * 70)
    print("  master   : " + str(MASTER_FILE))
    print("  docentes : " + str(TEACHERS_FILE))
    for d in FINAL_DIRS:
        print("  output   : " + str(d / OUT_NAME))
    print()

    # ---- master ----
    master = read_first_sheet(MASTER_FILE, ["master_all", "master_panel_nuts3"])
    master["nuts3_code"] = master["nuts3_code"].map(code)
    if "year_start" not in master.columns and "school_year" in master.columns:
        master["year_start"] = master["school_year"].map(year_start_from_school_year)
    master["year_start"] = pd.to_numeric(master["year_start"], errors="coerce")
    master["year_start"] = master["year_start"].astype("Int64")

    # ---- teachers_panel (estrutura etaria) ----
    tp = read_first_sheet(TEACHERS_FILE, ["teachers_panel"])
    tp["nuts3_code"] = tp["nuts3_code"].map(code)
    if "year_start" not in tp.columns and "school_year" in tp.columns:
        tp["year_start"] = tp["school_year"].map(year_start_from_school_year)
    tp["year_start"] = pd.to_numeric(tp["year_start"], errors="coerce")
    tp["year_start"] = tp["year_start"].astype("Int64")

    # completar teachers_55_plus a partir das duas bandas
    tem_55_59 = "teachers_55_59" in tp.columns
    tem_60_plus = "teachers_60_plus" in tp.columns
    if tem_55_59 and tem_60_plus:
        reconstruido = numeric(tp, "teachers_55_59") + numeric(tp, "teachers_60_plus")
        if "teachers_55_plus" in tp.columns:
            tp["teachers_55_plus"] = numeric(tp, "teachers_55_plus").fillna(reconstruido)
        else:
            tp["teachers_55_plus"] = reconstruido

    # completar as quotas a partir das contagens.
    # numeric() devolve uma serie de NaN se a coluna faltar, por isso nao ha
    # risco de pd.to_numeric(None) como acontecia com o tp.get().
    tot = numeric(tp, "teachers_total").replace(0, np.nan)
    if "teachers_55_plus" in tp.columns:
        calc = numeric(tp, "teachers_55_plus") / tot
        tp["share_55_plus"] = numeric(tp, "share_55_plus").fillna(calc)
    if "teachers_60_plus" in tp.columns:
        calc = numeric(tp, "teachers_60_plus") / tot
        tp["share_60_plus"] = numeric(tp, "share_60_plus").fillna(calc)

    # trazer SO o que ainda nao esta no master
    candidate = AGE_COUNT_COLS + SEX_COLS + SHARE_COLS
    bring = []
    for c in candidate:
        if c in tp.columns and c not in master.columns:
            bring.append(c)

    if not bring:
        print("AVISO: nenhuma coluna etaria nova para juntar.")

    add = tp[["year_start", "nuts3_code"] + bring].copy()
    for c in bring:
        add[c] = pd.to_numeric(add[c], errors="coerce")
    add = add.drop_duplicates(["year_start", "nuts3_code"])

    # ---- separar historico / forecast ----
    if "data_scope" in master.columns:
        hist = master[master["data_scope"] == "historical"].copy()
        fut = master[master["data_scope"] == "forecast"].copy()
    else:
        hist = master[master["year_start"] <= 2024].copy()
        fut = master[master["year_start"] > 2024].copy()

    # ---- estrutura etaria SO no historico ----
    hist = hist.merge(add, on=["year_start", "nuts3_code"], how="left")

    # interpolar lacunas dentro de cada NUTS3
    hist = hist.sort_values(["nuts3_code", "year_start"])
    for c in bring:
        grupo = hist.groupby("nuts3_code")[c]
        hist[c] = grupo.transform(lambda x: x.interpolate(limit_direction="both"))

    # ---- eliminar o cenario no_migration nos projetados ----
    drop_norm = set()
    for s in DROP_SCENARIOS:
        drop_norm.add(norm(s))
    if "scenario" in fut.columns:
        manter = fut["scenario"].map(lambda s: norm(s) not in drop_norm)
        fut = fut[manter].copy()

    # ---- ordenar colunas ----
    def order_cols(df):
        wanted = [
            "data_scope", "school_year", "year_start", "scenario",
            "nuts2_code", "nuts2_nome", "nuts3_code", "nuts3_nome",
        ]
        keys = [c for c in wanted if c in df.columns]
        rest = [c for c in df.columns if c not in keys]
        sort_by = []
        for c in ["scenario", "year_start", "nuts3_code"]:
            if c in df.columns:
                sort_by.append(c)
        out = df[keys + rest]
        if sort_by:
            out = out.sort_values(sort_by)
        return out.reset_index(drop=True)

    hist = order_cols(hist)
    fut = order_cols(fut)

    # ---- remover colunas totalmente em branco ----
    def drop_all_blank(df):
        keep = []
        for c in df.columns:
            s = df[c]
            vazia = s.isna().all()
            if not vazia:
                texto = s.astype(str).str.strip()
                vazia = (texto == "").all()
            if not vazia:
                keep.append(c)
        return df[keep]

    hist = drop_all_blank(hist)
    fut = drop_all_blank(fut)

    # ---- gravar nos dois destinos ----
    sheets = {"historical": hist, "forecast": fut}
    written = save_final(sheets, OUT_NAME)

    print("=" * 70)
    print("OUTPUT")
    print("=" * 70)
    for p in written:
        print(p)
    print()
    print("historical:", hist.shape, "| forecast:", fut.shape)
    if "scenario" in fut.columns:
        cenarios = sorted(fut["scenario"].dropna().unique())
        print("forecast scenarios:", cenarios)
    if bring:
        print("colunas etarias juntas:", len(bring))


if __name__ == "__main__":
    main()