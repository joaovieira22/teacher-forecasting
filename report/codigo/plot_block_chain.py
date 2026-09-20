
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# ============================================================ PATHS
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
FIG_DIR = BASE_DIR / "report" / "figures"
if os.environ.get("CHAIN_TEST") == "1":
    FIG_DIR = Path("chain_out")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG_DIR / "block_chain_acyclicity.png"

# ============================================================ STYLE
INK      = "#1a1a1a"
MUTED    = "#6b7280"
DATA_FC  = "#e8eef6"; DATA_EC  = "#4a7fa5"
BLOCK_FC = "#f3f0fa"; BLOCK_EC = "#6b4fa0"
VAL      = "#2c3e50"       # value flow: the acyclic part
BACK     = "#C0392B"       # backward read: the cycle
TEST     = "#7f8c8d"       # test-only read, terminal

WHITE_BOX = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.96)

# ============================================================ GEOMETRY
BW, BH = 2.62, 1.18
GAP = 5.15
XS = [i * GAP for i in range(6)]
CX = [x + BW / 2 for x in XS]
BY = 0.0
BMID = BY + BH / 2

BLOCKS = [
    ("01_students",    "enrolment by\nNUTS III and cycle"),
    ("02_demand",      "teaching posts\nimplied by enrolment"),
    ("03_supply",      "workforce, cohort chain\nUNCONSTRAINED entry"),
    ("04_gap",         "applies the ITE\nrestriction"),
    ("05_uncertainty", "Monte Carlo over\nthe 04 parameters"),
    ("06_validation",  "tests the chain,\nfeeds none of it"),
]

# (x0, width, filename, producing script)
DATA_Y, DATA_H = 5.15, 0.95
DATA = [
    (0.00, 3.15, "master_panel_nuts3.xlsx", "script 10"),
    (7.10, 7.00, "master_panel_nuts3_with_age.xlsx", "script 11"),
    (18.40, 4.35, "teacher_supply_panel.xlsx", "script 06"),
]

# adjacent value edges: (from, to, label)
VALUE_EDGES = [
    (0, 1, "students_forecast\n_nuts3.xlsx"),
    (1, 2, "teacher_demand\n_selected.csv"),
    (2, 3, "supply_projection.csv\nsupply_recruitment\n_by_channel.csv"),
    (3, 4, "pipeline_gap.csv\ngap_national.csv"),
]

# non-adjacent value edges, arced above: (from, to, label, rad, label_y)
VALUE_ARCS = [
    (1, 3, "teacher_demand_selected.csv", -0.22, 2.30),
    (2, 4, "recruitment_montecarlo_v3.csv   (the demand sigma)", -0.34, 3.35),
]

# backward reads, arced below: (from, to, label, rad, label_y)
BACK_EDGES = [
    (2, 1, "supply_projection.csv", -0.36, -1.05),
    (3, 1, "gap_national.csv    \u00b7    gap_recruitment_nuts3.csv", -0.26, -1.85),
]


def box(ax, x, y, w, h, title, sub, fc, ec, fs=10.8, sub_fs=8.1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=1.7, zorder=5))
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=INK, zorder=6)
    ax.text(x + w / 2, y + h * 0.25, sub, ha="center", va="center",
            fontsize=sub_fs, color=MUTED, zorder=6, linespacing=1.25)


def edge(ax, x0, y0, x1, y1, color, lw, ls, rad, scale=14, alpha=1.0, z=2):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=scale,
        color=color, lw=lw, linestyle=ls, alpha=alpha,
        connectionstyle="arc3,rad=" + str(rad),
        shrinkA=2, shrinkB=2, zorder=z))


def save_figure(fig, path, dpi=220):
    """Write a figure through an open binary stream.

    Pillow raises OSError [Errno 22] when it opens a OneDrive path itself,
    so the file handle is created here and handed to savefig.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            path.unlink()
        except PermissionError as exc:
            raise PermissionError(
                "Cannot replace the figure because it is open or locked:\n"
                + str(path)
            ) from exc
    with path.open("wb") as handle:
        fig.savefig(handle, format=path.suffix.lstrip("."), dpi=dpi,
                    bbox_inches="tight", facecolor="white")


def main():
    fig, ax = plt.subplots(figsize=(22.0, 10.4))
    ax.set_axis_off()

    # ------------------------------------------------ data layer
    for x0, w, name, origin in DATA:
        box(ax, x0, DATA_Y, w, DATA_H, name, origin, DATA_FC, DATA_EC,
            fs=9.6, sub_fs=8.2)
    ax.text(-2.15, DATA_Y + DATA_H / 2, "data layer\n(00_data)", ha="center",
            va="center", fontsize=11.0, fontweight="bold", color=DATA_EC,
            linespacing=1.3)

    # ------------------------------------------------ blocks
    for i, (name, sub) in enumerate(BLOCKS):
        box(ax, XS[i], BY, BW, BH, name, sub, BLOCK_FC, BLOCK_EC)
    ax.text(-2.15, BMID, "model\nblocks", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color=BLOCK_EC, linespacing=1.3)

    # ------------------------------------------------ data -> blocks
    edge(ax, 1.58, DATA_Y, CX[0], BY + BH, DATA_EC, 1.6, "-", 0.04, z=3)
    for x_src, target in ((8.40, CX[1]), (10.60, CX[2]), (12.90, CX[3] - 0.50)):
        edge(ax, x_src, DATA_Y, target, BY + BH, DATA_EC, 1.5, "-", 0.06,
             alpha=0.80, z=3)
    edge(ax, 19.80, DATA_Y, CX[3] + 0.55, BY + BH, DATA_EC, 1.9, "-", -0.16,
         z=3)
    ax.text(22.95, DATA_Y - 0.30,
            "read by 04 directly:\nit never enters\nthe master panel",
            ha="left", va="top", fontsize=8.4, color=DATA_EC, style="italic",
            linespacing=1.4)

    # ------------------------------------------------ adjacent value edges
    for a, b, lab in VALUE_EDGES:
        edge(ax, XS[a] + BW, BMID, XS[b], BMID, VAL, 2.2, "-", 0.0, scale=17,
             z=4)
        ax.text((XS[a] + BW + XS[b]) / 2, BMID + 0.14, lab,
                ha="center", va="bottom", fontsize=7.0, color=VAL,
                linespacing=1.30, zorder=7, bbox=WHITE_BOX)

    # ------------------------------------------------ arced value edges
    for a, b, lab, rad, ytxt in VALUE_ARCS:
        edge(ax, CX[a], BY + BH, CX[b], BY + BH, VAL, 1.7, "-", rad, scale=14,
             z=4)
        ax.text((CX[a] + CX[b]) / 2, ytxt, lab, ha="center", va="center",
                fontsize=8.2, color=VAL, zorder=7, bbox=WHITE_BOX)

    # ------------------------------------------------ backward reads
    for a, b, lab, rad, ytxt in BACK_EDGES:
        edge(ax, CX[a], BY, CX[b], BY, BACK, 2.2, (0, (6, 3)), rad, scale=17,
             alpha=0.93, z=4)
        ax.text((CX[a] + CX[b]) / 2, ytxt, lab, ha="center", va="center",
                fontsize=8.2, color=BACK, zorder=7, bbox=WHITE_BOX)

    # ------------------------------------------------ 06 test bus
    bus_y = -3.05
    for i in range(5):
        ax.plot([CX[i], CX[i]], [BY - 0.08, bus_y], color=TEST, lw=1.0,
                ls=(0, (1, 3)), zorder=1)
    ax.plot([CX[0], CX[4]], [bus_y, bus_y], color=TEST, lw=1.2,
            ls=(0, (1, 3)), zorder=1)
    edge(ax, CX[4], bus_y, CX[5], BY, TEST, 1.5, (0, (1, 3)), -0.26, z=1)
    ax.text(CX[2], bus_y - 0.32,
            "06_validation opens the outputs of every preceding block and "
            "tests them. Nothing it writes is read by any block,\n"
            "so a failure there cannot alter a result upstream.",
            ha="center", va="top", fontsize=8.8, color=TEST, style="italic",
            linespacing=1.5)

    # ------------------------------------------------ residuals
    ax.text(CX[2], -4.35,
            "The forward chain's arithmetic closes. Largest absolute residuals "
            "reported by 03_supply: flow identity 2.9e-11, historical gross "
            "flows 9.1e-13,\nhazard times stock against gross exits 1.1e-13, "
            "band-fill helper exactly zero. Acyclicity is what makes those "
            "identities checkable in a single pass.",
            ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.55)

    # ------------------------------------------------ legend
    handles = [
        Line2D([0], [0], color=VAL, lw=2.2,
               label="value flow: a quantity the next block computes with"),
        Line2D([0], [0], color=DATA_EC, lw=1.7,
               label="value flow from the data layer"),
        Line2D([0], [0], color=BACK, lw=2.2, ls=(0, (6, 3)),
               label="BACKWARD read: opened for consistency checks only"),
        Line2D([0], [0], color=TEST, lw=1.5, ls=(0, (1, 3)),
               label="terminal read: 06 tests, and feeds nothing"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, -0.055), fontsize=9.4, handlelength=2.8,
              columnspacing=2.2)

    fig.suptitle("Block chaining: what runs before what, and on which file",
                 fontsize=17, fontweight="bold", y=0.975)
    ax.text(0.5, 1.020,
            "Mainland Portugal, 24 NUTS III regions, 2025-2040.    Blocks are "
            "run by hand in numerical order; the arrows are the contracts "
            "between them.",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10.0, color=MUTED)

    ax.set_xlim(-3.60, 29.30)
    ax.set_ylim(-5.60, 6.55)
    fig.subplots_adjust(top=0.90)

    save_figure(fig, OUT)
    save_figure(fig, OUT.with_suffix(".pdf"))
    plt.close(fig)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
