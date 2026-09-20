
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================ PATHS
BASE = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
FIG_DIR = BASE / "report" / "figures"
if os.environ.get("DIAG_TEST") == "1":
    FIG_DIR = Path("fig_out")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================ STYLE
INK   = "#1a1a1a"
MUTED = "#6b7280"
GREY  = "#9aa4b2"
RAW   = "#e8eef6"; RAW_E   = "#4a7fa5"
PROC  = "#e6f2ea"; PROC_E  = "#128a4b"
DERIV = "#fff4e6"; DERIV_E = "#c2761a"
ANAL  = "#fdecec"; ANAL_E  = "#b5171e"
MODEL = "#f3f0fa"; MODEL_E = "#6b4fa0"
NOTE  = "#fff8e1"; NOTE_E  = "#b8860b"

LEFT, RIGHT = 0.150, 0.985


def row(n, gap=0.013, left=LEFT, right=RIGHT, weights=None):
    """Evenly spaced boxes across the drawing area."""
    span = right - left - gap * (n - 1)
    if weights is None:
        weights = [1.0] * n
    total = sum(weights)
    positions, x = [], left
    for weight in weights:
        width = span * weight / total
        positions.append((x, width))
        x += width + gap
    return positions


def box(ax, x, y, w, h, title, lines, fc, ec, fs=9.5, tw="bold"):
    """Draw a rounded box with a title and evenly spaced detail lines."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=2,
    ))
    pad = 0.016
    ax.text(
        x + w / 2, y + h - pad, title,
        ha="center", va="top", fontsize=fs, fontweight=tw,
        color=INK, zorder=3, linespacing=1.22,
    )
    if not lines:
        return
    n_title = title.count("\n") + 1
    top = y + h - pad - 0.021 * n_title
    bottom = y + pad * 0.7
    step = (top - bottom) / max(len(lines), 1)
    for i, line in enumerate(lines):
        ax.text(
            x + w / 2, top - step * (i + 0.5), line,
            ha="center", va="center", fontsize=fs - 1.5,
            color=MUTED, zorder=3,
        )


def arrow(ax, x0, y0, x1, y1, color=GREY, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, linestyle=ls,
        shrinkA=3, shrinkB=3, zorder=1,
    ))


def band(ax, y, h, label, color, alpha=0.05):
    ax.add_patch(FancyBboxPatch(
        (0.010, y), 0.980, h,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=color, edgecolor="none", alpha=alpha, zorder=0,
    ))
    ax.text(
        0.020, y + h / 2, label,
        ha="left", va="center", fontsize=11.5,
        fontweight="bold", color=color, zorder=3,
    )


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save_figure(fig, filename, dpi=200):
    """Save a Matplotlib figure safely on Windows and OneDrive."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIG_DIR / filename

    if output_path.exists():
        try:
            output_path.unlink()
        except PermissionError as exc:
            raise PermissionError(
                "Cannot replace the figure because it is open or locked:\n"
                f"{output_path}"
            ) from exc

    try:
        with output_path.open("wb") as file_handle:
            fig.savefig(
                file_handle,
                format="png",
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
            )
    except OSError as exc:
        raise OSError(
            "Could not write the figure.\n"
            f"Output directory: {FIG_DIR!r}\n"
            f"Output file: {output_path!r}\n"
            "Check whether the OneDrive folder is available locally and whether "
            "the existing PNG is open in another application."
        ) from exc

    print("[ok]", output_path)
    return output_path


# ============================================================ FIG 1 - pipeline
def fig_pipeline():
    fig, ax = blank((17.0, 12.0))

    ax.text(0.5, 0.990, "Data preparation pipeline",
            ha="center", va="top", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.962,
            "teacher_demand_forecasting   ·   24 NUTS III regions (NUTS 2024), "
            "mainland Portugal   ·   historical to 2024, projected from 2025",
            ha="center", va="top", fontsize=9.5, color=MUTED)

    band(ax, 0.812, 0.128, "01_raw", RAW_E)
    r_y, r_h = 0.822, 0.108
    raw_pos = row(4)
    raw_specs = [
        ("INE population", ["estimates by age, municipality", "projections by age, NUTS II", "two NUTS vintages"]),
        ("INE demographic indices", ["ageing index", "working-age renewal index", "municipality x year"]),
        ("INE education", ["pupils by level, NUTS III", "teachers by age and sex, NUTS III"]),
        ("DGEEC education", ["teachers by cycle and group, school", "graduates by field, country"]),
    ]
    for (x, w), (title, lines) in zip(raw_pos, raw_specs):
        box(ax, x, r_y, w, r_h, title, lines, RAW, RAW_E)

    band(ax, 0.470, 0.320, "parsers\n01-08", PROC_E)
    pa_y, pb_y, p_h = 0.636, 0.484, 0.140
    pa_pos = row(4)
    pb_pos = row(4)
    pa_specs = [
        ("students_nuts3", ["01 · build_students", "7 level series", "02_processed"]),
        ("ageing_index_nuts3", ["02 · build_ageing_index", "mean and median", "02_processed"]),
        ("renewal_index_nuts3", ["03 · build_renewal_index", "mean and median", "02_processed"]),
        ("population_historical_nuts3", ["04 · build_population_historical", "18 bands plus total", "02_processed"]),
    ]
    pb_specs = [
        ("population_projections_nuts2", ["05 · build_population_projections", "4 scenarios, ages 0-100", "02_processed"]),
        ("teacher_supply_panel", ["06 · build_graduates", "11 fields plus total", "03_analysis + models/00_data"]),
        ("teachers_panel", ["07 · build_teachers_age_sex_ine", "9 age bands x sex", "03_analysis + models/00_data"]),
        ("teachers_panel_nuts2024", ["08 · build_teachers_cycle_subject_dgeec", "cycle, function, group", "03_analysis + models/00_data"]),
    ]
    for (x, w), (title, lines) in zip(pa_pos, pa_specs):
        box(ax, x, pa_y, w, p_h, title, lines, PROC, PROC_E, fs=9)
    for (x, w), (title, lines) in zip(pb_pos, pb_specs):
        box(ax, x, pb_y, w, p_h, title, lines, PROC, PROC_E, fs=9)

    band(ax, 0.238, 0.212, "derived\nand joined\n09-11", ANAL_E)
    a_y, a_h = 0.254, 0.170
    a_pos = row(3, gap=0.040, weights=[1.0, 1.0, 1.0])
    box(ax, a_pos[0][0], a_y, a_pos[0][1], a_h, "population_projections_nuts3",
        ["09 · build_population_nuts3", "NUTS II split into NUTS III", "index-driven shares, CONSTRUCTED", "02_processed"],
        DERIV, DERIV_E, fs=9.5)
    box(ax, a_pos[1][0], a_y, a_pos[1][1], a_h, "master_panel_nuts3",
        ["10 · build_master_panel", "reads six files", "24 NUTS III, history + forecast", "02_processed"],
        ANAL, ANAL_E, fs=9.5)
    box(ax, a_pos[2][0], a_y, a_pos[2][1], a_h, "master_panel_nuts3_with_age",
        ["11 · final_join", "adds teacher age structure", "historical / forecast sheets", "03_analysis + models/00_data"],
        ANAL, ANAL_E, fs=9.5)

    band(ax, 0.062, 0.140, "models", MODEL_E)
    m_y, m_h = 0.076, 0.084
    model_pos = row(6, gap=0.010)
    model_specs = [("01_students", "enrolment"), ("02_demand", "posts"),
                   ("03_supply", "workforce"), ("04_gap", "deficit"),
                   ("05_uncertainty", "Monte Carlo"), ("06_validation", "tests")]
    for (x, w), (name, subtitle) in zip(model_pos, model_specs):
        box(ax, x, m_y, w, m_h, name, [subtitle], MODEL, MODEL_E, fs=9)

    rc = [x + w / 2 for x, w in raw_pos]
    pac = [x + w / 2 for x, w in pa_pos]
    pbc = [x + w / 2 for x, w in pb_pos]

    arrow(ax, rc[0], r_y, pac[3], pa_y + p_h)
    arrow(ax, rc[0], r_y, pbc[0], pb_y + p_h)
    arrow(ax, rc[1], r_y, pac[1], pa_y + p_h)
    arrow(ax, rc[1], r_y, pac[2], pa_y + p_h)
    arrow(ax, rc[2], r_y, pac[0], pa_y + p_h)
    arrow(ax, rc[2], r_y, pbc[2], pb_y + p_h)
    arrow(ax, rc[3], r_y, pbc[3], pb_y + p_h)
    arrow(ax, rc[3], r_y, pbc[1], pb_y + p_h)

    x09 = a_pos[0][0] + a_pos[0][1] / 2
    x10 = a_pos[1][0] + a_pos[1][1] / 2
    x11 = a_pos[2][0] + a_pos[2][1] / 2

    arrow(ax, pac[1], pa_y, x09, a_y + a_h, color=DERIV_E, lw=1.2)
    arrow(ax, pac[2], pa_y, x09, a_y + a_h, color=DERIV_E, lw=1.2)
    arrow(ax, pbc[0], pb_y, x09, a_y + a_h, color=DERIV_E, lw=1.2)
    for source_x in (pac[0], pac[1], pac[2], pac[3]):
        arrow(ax, source_x, pa_y, x10, a_y + a_h, lw=1.0)
    arrow(ax, pbc[3], pb_y, x10, a_y + a_h, lw=1.0)
    arrow(ax, a_pos[0][0] + a_pos[0][1], a_y + a_h / 2,
          a_pos[1][0], a_y + a_h / 2, color=DERIV_E, lw=2.0)
    arrow(ax, a_pos[1][0] + a_pos[1][1], a_y + a_h / 2,
          a_pos[2][0], a_y + a_h / 2, color=ANAL_E, lw=2.0)
    ax.text((a_pos[1][0] + a_pos[1][1] + a_pos[2][0]) / 2,
            a_y + a_h / 2 + 0.013, "+ age", ha="center", va="bottom",
            fontsize=8.5, color=ANAL_E, fontweight="bold")
    arrow(ax, pbc[2], pb_y, x11, a_y + a_h, color=ANAL_E, lw=1.2)

    for x, w in model_pos:
        arrow(ax, x11, a_y, x + w / 2, m_y + m_h, lw=0.9)
    x_gap = model_pos[3][0] + model_pos[3][1] / 2
    arrow(ax, pbc[1], pb_y, x_gap, m_y + m_h,
          color=PROC_E, lw=1.4, ls=(0, (4, 2)))

    for i in range(len(model_pos) - 1):
        x_end = model_pos[i][0] + model_pos[i][1]
        arrow(ax, x_end, m_y + m_h / 2,
              model_pos[i + 1][0], m_y + m_h / 2,
              color=MODEL_E, lw=1.2)

    ax.text(0.5, 0.048,
            "Every municipality file is re-anchored to NUTS 2024 by its DICO before "
            "aggregation.   The dashed green arrow is teacher_supply_panel, which "
            "bypasses the master panel and is read by 04_gap directly.",
            ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")
    ax.text(0.5, 0.024,
            "population_projections_nuts3 is CONSTRUCTED, not observed: script 09 "
            "splits each NUTS II total across its NUTS III regions using the two "
            "demographic indices.   06_validation tests the preceding blocks and "
            "feeds none of them.",
            ha="center", va="top", fontsize=8.5, color=DERIV_E, style="italic")

    try:
        save_figure(fig, "data_pipeline.png")
    finally:
        plt.close(fig)


# ============================================================ FIG 2 - cleaning
def fig_cleaning():
    fig, ax = blank((16.5, 11.0))

    ax.text(0.5, 0.990, "Cleaning decisions that affect the results",
            ha="center", va="top", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.5, 0.960,
            "Transformations that are invisible in the final files and must appear "
            "in the methodological note",
            ha="center", va="top", fontsize=9.5, color=MUTED)

    ax.add_patch(FancyBboxPatch((0.010, 0.700), 0.980, 0.238,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=NOTE_E, edgecolor="none", alpha=0.05, zorder=0))
    ax.text(0.028, 0.920, "1 · Re-anchoring NUTS 2013 to NUTS 2024",
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=NOTE_E)
    ax.text(0.028, 0.890,
            "The INE municipality code is NUTS2 + NUTS3 + DICO concatenated. The "
            "prefix changes length between vintages; the DICO is always the last\n"
            "four digits, and that is the join key. Scripts 02, 03 and 04 use it. "
            "Script 08 cannot: DGEEC identifies municipalities by name only.",
            ha="left", va="top", fontsize=9, color=INK, linespacing=1.5)

    columns = [(0.070, "municipality"), (0.240, "2013 code"), (0.370, "NUTS III 2013"),
               (0.550, "2024 code"), (0.680, "NUTS III 2024"), (0.880, "DICO")]
    for x, label in columns:
        ax.text(x, 0.822, label, fontsize=8.5, fontweight="bold",
                color=NOTE_E if label == "DICO" else MUTED)
    ax.plot([0.065, 0.930], [0.814, 0.814], color=NOTE_E, lw=0.8, alpha=0.4)

    examples = [
        ("Serta", "16I0509", "Medio Tejo", "1950509", "Beira Baixa"),
        ("Alcobaca", "16B1001", "Centro", "1D11001", "Oeste"),
        ("Almada", "1701503", "Lisbon MA", "1B01503", "Setubal Pen."),
        ("Evora", "1870705", "Central Alent.", "1C40705", "Central Alent."),
    ]
    for i, (name, code13, nuts13, code24, nuts24) in enumerate(examples):
        y = 0.790 - i * 0.022
        moved = nuts13 != nuts24
        ax.text(0.070, y, name, fontsize=8.5, color=INK, va="center")
        ax.text(0.240, y, code13, fontsize=8.5, color=MUTED,
                family="monospace", va="center")
        ax.text(0.370, y, nuts13, fontsize=8.5, color=MUTED, va="center")
        ax.text(0.550, y, code24, fontsize=8.5, color=MUTED,
                family="monospace", va="center")
        ax.text(0.680, y, nuts24, fontsize=8.5, va="center",
                color=ANAL_E if moved else MUTED,
                fontweight="bold" if moved else "normal")
        ax.text(0.880, y, code13[-4:], fontsize=8.5, color=NOTE_E,
                fontweight="bold", family="monospace", va="center")

    ax.text(0.028, 0.706,
            "The codes above illustrate the rule. Script 01 has no crosswalk at all: "
            "it keeps only rows whose NUTS III code is in the 2024 dictionary.",
            ha="left", va="top", fontsize=8.5, color=MUTED, style="italic")

    ax.add_patch(FancyBboxPatch((0.010, 0.474), 0.980, 0.212,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=NOTE_E, edgecolor="none", alpha=0.05, zorder=0))
    ax.text(0.028, 0.668, "2 · De-duplication of overlapping years",
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=NOTE_E)

    d_y, d_h = 0.516, 0.106
    box(ax, 0.075, d_y, 0.215, d_h, "NUTS 2024 file",
        ["recent years", "source_is_nuts2024 = 1"], NOTE, NOTE_E, fs=9.5)
    box(ax, 0.330, d_y, 0.215, d_h, "NUTS 2013 file",
        ["earlier years", "source_is_nuts2024 = 0"], NOTE, NOTE_E, fs=9.5)
    box(ax, 0.660, d_y, 0.240, d_h, "one row per (year, DICO)",
        ["overlap resolved", "2024 vintage retained"], PROC, PROC_E, fs=9.5)
    arrow(ax, 0.290, d_y + d_h * 0.62, 0.660, d_y + d_h * 0.62,
          color=NOTE_E, lw=1.6)
    arrow(ax, 0.545, d_y + d_h * 0.34, 0.660, d_y + d_h * 0.42,
          color=NOTE_E, lw=1.6)
    ax.text(0.5, 0.494,
            "sort on source_is_nuts2024 descending, then drop_duplicates on "
            "(ano, municipio_id) keeping the first.   The size of any revision in "
            "the overlap years is not recorded.",
            ha="center", va="top", fontsize=8.8, color=INK)

    ax.add_patch(FancyBboxPatch((0.010, 0.238), 0.980, 0.216,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=NOTE_E, edgecolor="none", alpha=0.05, zorder=0))
    ax.text(0.028, 0.436, "3 · Aggregation municipality to NUTS III",
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=NOTE_E)

    g_y, g_h = 0.254, 0.144
    box(ax, 0.060, g_y, 0.400, g_h, "Counts: sum them",
        ["population, pupils, teachers, graduates",
         "NUTS III = sum of its municipalities",
         "the identity closes exactly"], PROC, PROC_E, fs=10)

    ax.add_patch(FancyBboxPatch(
        (0.520, g_y), 0.420, g_h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=ANAL, edgecolor=ANAL_E, linewidth=1.5, zorder=2))
    ax.text(0.730, g_y + g_h - 0.018, "Indices: never average them",
            ha="center", va="top", fontsize=10, fontweight="bold", color=INK, zorder=3)
    ax.text(0.730, g_y + g_h - 0.046,
            "scripts 02 and 03 write the mean across municipalities",
            ha="center", va="center", fontsize=8.6, color=MUTED, zorder=3)
    ax.text(0.730, g_y + g_h - 0.067,
            "the components are in population_historical_nuts3",
            ha="center", va="center", fontsize=8.6, color=MUTED, zorder=3)
    ax.text(0.730, g_y + 0.028,
            r"$\dfrac{1}{n}\sum_i \dfrac{P^{65+}_i}{P^{0\mathrm{-}14}_i}"
            r" \;\neq\; \dfrac{\sum_i P^{65+}_i}{\sum_i P^{0\mathrm{-}14}_i}$",
            ha="center", va="center", fontsize=11, color=ANAL_E, zorder=3)

    ax.add_patch(FancyBboxPatch((0.010, 0.018), 0.980, 0.200,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=NOTE_E, edgecolor="none", alpha=0.05, zorder=0))
    ax.text(0.028, 0.200, "4 · Limitations on record",
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=NOTE_E)

    limitations = [
        "The index columns are named _mean and _median: they are means across municipalities, not the NUTS III index.",
        "Script 09 allocates pop_total and every age band separately, each with its own tilt. At NUTS III the bands do not sum to the total.",
        "The base weight in script 09 is uniform, so the allocated share ignores how many people live in each NUTS III region.",
        "teachers_total comes from DGEEC (08) and the age bands from INE (07): the bands need not sum to the total beside them.",
        "Teacher age structure exists only in the historical block; 11_final_join interpolates gaps and repeats values at the edges.",
    ]
    for i, text in enumerate(limitations):
        y = 0.166 - i * 0.030
        ax.text(0.038, y, "•", fontsize=10, color=NOTE_E,
                fontweight="bold", ha="left", va="center")
        ax.text(0.054, y, text, ha="left", va="center", fontsize=8.8, color=INK)

    try:
        save_figure(fig, "data_cleaning_decisions.png")
    finally:
        plt.close(fig)


# ============================================================ MAIN
def main():
    print("=" * 74)
    print("plot_data_pipeline_diagram")
    print("=" * 74)
    print(f"  figures -> {FIG_DIR}")
    print()
    fig_pipeline()
    fig_cleaning()
    print()
    print("Done:")
    print("  data_pipeline.png            01_raw -> parsers 01-08 -> 09/10/11 -> models")
    print("  data_cleaning_decisions.png  re-anchoring, de-duplication, aggregation, limits")
    print("=" * 74)


if __name__ == "__main__":
    main()
