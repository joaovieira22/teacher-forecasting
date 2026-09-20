
from pathlib import Path
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D

# ============================================================ PATHS
BASE_DIR = Path(r"C:\Users\NJ183BX\OneDrive - EY\Desktop\teacher_demand_forecasting")
FIG_DIR = BASE_DIR / "report" / "figures"
if os.environ.get("DAG_TEST") == "1":
    FIG_DIR = Path("dag_out")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT = FIG_DIR / "dag_mecanismo_professores.png"

# ============================================================ PALETTE
COL = {
    "driver":   "#1A5276",
    "aluno":    "#E67E22",
    "procura":  "#8E44AD",
    "oferta":   "#27AE60",
    "reposic":  "#C0392B",
    "pipeline": "#16A085",
    "lateral":  "#8B5E3C",
    "assump":   "#4A235A",
    "param":    "#5D6D7E",
    "outcome":  "#000000",
    "unobs":    "#95A5A6",
    "noreport": "#7F8C8D",
}

ROLE_LABEL = {
    "driver":   "Exogenous driver (demography)",
    "aluno":    "Enrolment forecast (01)",
    "procura":  "Demand and need (02, 03)",
    "oferta":   "Workforce stock (03)",
    "reposic":  "Exits: retirement and attrition (03)",
    "pipeline": "Graduate channel, under 30 (04)",
    "lateral":  "Lateral channel, 30 and over (04)",
    "param":    "Estimated parameter",
    "assump":   "External assumption",
    "outcome":  "Target: teacher gap",
}

XS = 1.30  # horizontal stretch, applied once so the layout stays editable

# id -> (label, x, y, role)
NODES = {
    "pop":      ("Population by age\n(NUTS III x scenario)", -8.6, 4.6, "driver"),
    "al_pre":   ("Enrolment\npre-school",            -6.5, 7.6, "aluno"),
    "al_b1":    ("Enrolment\n1st cycle",             -6.5, 5.9, "aluno"),
    "al_b2":    ("Enrolment\n2nd cycle",             -6.5, 4.2, "aluno"),
    "al_b3s":   ("Enrolment\n3rd cycle + secondary", -6.5, 2.5, "aluno"),

    "ratio":    ("Student-teacher ratio\n(estimated on HISTORY)", -4.2, 8.3, "param"),
    "beta":     ("Enrolment elasticity\n(3rd+sec only, beta)",    -4.2, 6.4, "param"),
    "demand":   ("Teacher demand\n(required posts)",              -3.9, 4.2, "procura"),
    "short_u":  ("Unobserved past\nshortage",                     -1.9, 8.3, "unobs"),

    "idade":    ("Teacher age structure\n(9 bands)",  -8.6, 0.4, "reposic"),
    "exits":    ("Exits\n(retirement + attrition)",   -6.5, 0.4, "reposic"),
    "stock":    ("Teacher stock\n(t-1)",              -6.5, -1.6, "oferta"),
    "need":     ("Recruitment need\n= demand - stock + exits", -1.4, 4.2, "procura"),

    "sigma":    ("Young share of entries\n(03 entry profile, 0.277)", -1.4, 1.9, "param"),
    "need_y":   ("Need, graduate\nchannel (<30)", 1.4, 5.4, "procura"),
    "need_l":   ("Need, lateral\nchannel (30+)",  1.4, 3.2, "procura"),

    "diplom":   ("Graduates\n(broad field, DGEEC)",       -8.6, -4.2, "pipeline"),
    "pool":     ("Graduate pool\n(3y rolling, lag 1)",    -6.5, -4.2, "pipeline"),
    "anchor":   ("External anchor\n(~2,000/yr, x1.624)",  -8.6, -6.4, "assump"),
    "ceiling":  ("Conversion ceiling\n(85% of a cohort)", -6.5, -6.4, "assump"),
    "rate":     ("Conversion rate\n(entries / pool)",     -4.3, -5.3, "param"),
    "ent_y":    ("Feasible entries,\ngraduate channel",   -1.4, -4.2, "pipeline"),

    "reserve":  ("Contracted reserve\n(UNMEASURED)",           -8.6, -8.8, "unobs"),
    "lat_rate": ("Reserve depletion rate\n(0.0271 of stock)",  -4.3, -8.8, "param"),
    "ent_l":    ("Feasible entries,\nlateral channel",         -1.4, -6.8, "lateral"),

    "gap_y":    ("Gap, graduate\n= need_y - ent_y", 4.0, 5.4, "outcome"),
    "gap_l":    ("Gap, lateral\n= need_l - ent_l",  4.0, 3.2, "outcome"),
    "gap":      ("TEACHER GAP\n= max(gap_y,0) + max(gap_l,0)", 6.6, 4.3, "outcome"),
    "netted":   ("netted gap = 0\n(cross-credits channels)",   6.6, 1.3, "noreport"),
}

# ============================================================ EDGES
# (source, target, kind, rad)
EDGES = [
    ("pop", "al_pre", "causal", 0.06),
    ("pop", "al_b1", "causal", 0.06),
    ("pop", "al_b2", "causal", 0.0),
    ("pop", "al_b3s", "causal", -0.06),

    ("al_pre", "demand", "identity", -0.05),
    ("al_b1", "demand", "identity", -0.05),
    ("al_b2", "demand", "identity", 0.0),
    ("ratio", "demand", "identity", 0.10),
    ("al_b3s", "demand", "estimated", 0.05),
    ("beta", "demand", "estimated", -0.10),

    ("short_u", "ratio", "causal", 0.14),
    ("short_u", "gap_y", "causal", -0.10),

    ("idade", "exits", "causal", 0.0),
    ("idade", "stock", "identity", -0.08),
    ("exits", "stock", "identity", 0.0),
    ("demand", "need", "identity", 0.0),
    ("stock", "need", "identity", 0.16),
    ("exits", "need", "identity", 0.10),

    ("need", "need_y", "estimated", 0.05),
    ("need", "need_l", "estimated", -0.05),
    ("sigma", "need_y", "estimated", -0.10),
    ("sigma", "need_l", "estimated", -0.06),

    ("diplom", "pool", "identity", 0.0),
    ("pool", "ent_y", "identity", 0.0),
    ("rate", "ent_y", "identity", -0.08),
    ("anchor", "rate", "assumption", -0.10),
    ("ceiling", "rate", "assumption", -0.06),

    ("reserve", "lat_rate", "hypothetical", 0.0),
    ("stock", "lat_rate", "estimated", -0.12),
    ("stock", "ent_l", "identity", 0.30),
    ("lat_rate", "ent_l", "identity", -0.10),

    ("need_y", "gap_y", "identity", 0.0),
    ("ent_y", "gap_y", "identity", -0.16),
    ("need_l", "gap_l", "identity", 0.0),
    ("ent_l", "gap_l", "identity", -0.22),
    ("gap_y", "gap", "identity", 0.0),
    ("gap_l", "gap", "identity", 0.0),
    ("gap_y", "netted", "identity", 0.10),
    ("gap_l", "netted", "identity", 0.06),
]

# The edge this figure exists to show. Feasible entries should close the stock
# recursion; nothing writes them back, so 03 keeps the unconstrained path.
MISSING = [
    ("ent_y", "stock", 0.34),
    ("ent_l", "stock", 0.46),
]

EDGE_STYLE = {
    "causal":       dict(color="#6b7280", lw=1.5, ls="-"),
    "identity":     dict(color="#aab2ba", lw=1.2, ls=(0, (5, 3))),
    "estimated":    dict(color="#5D6D7E", lw=1.6, ls=(0, (7, 2, 1, 2))),
    "assumption":   dict(color="#4A235A", lw=1.6, ls=(0, (1, 2))),
    "hypothetical": dict(color="#95A5A6", lw=1.3, ls=(0, (1, 3))),
}

R = 0.40
MISSING_COL = "#C0392B"


def layout():
    return {k: (v[1] * XS, v[2]) for k, v in NODES.items()}


def trimmed_endpoints(pos, a, b):
    xa, ya = pos[a]
    xb, yb = pos[b]
    dx, dy = xb - xa, yb - ya
    d = float(np.hypot(dx, dy))
    if d == 0.0:
        return None
    ux, uy = dx / d, dy / d
    return (xa + ux * R, ya + uy * R), (xb - ux * R, yb - uy * R)


def draw_edge(ax, pos, a, b, kind, rad):
    ends = trimmed_endpoints(pos, a, b)
    if ends is None:
        return
    start, end = ends
    st = EDGE_STYLE[kind]
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=12,
        color=st["color"], lw=st["lw"], linestyle=st["ls"],
        connectionstyle=f"arc3,rad={rad}", zorder=1))


def main():
    fig, ax = plt.subplots(figsize=(20.5, 19.0))
    ax.set_axis_off()
    ax.set_aspect("equal")          # circles must be circles
    pos = layout()

    for a, b, kind, rad in EDGES:
        draw_edge(ax, pos, a, b, kind, rad)

    # ---------------------------------------------------- the absent feedback
    for a, b, rad in MISSING:
        start, end = trimmed_endpoints(pos, a, b)
        ax.add_patch(FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=16,
            color=MISSING_COL, lw=2.2, linestyle=(0, (6, 4)),
            connectionstyle=f"arc3,rad={rad}", zorder=4, alpha=0.9))

    box_x, box_y = 5.4, -2.6
    ax.text(box_x, box_y, "THIS EDGE DOES NOT EXIST",
            ha="center", va="center", fontsize=12.5, fontweight="bold",
            color=MISSING_COL, zorder=6,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fdecec",
                      edgecolor=MISSING_COL, linewidth=1.8))
    ax.text(box_x, box_y - 0.85,
            "03_supply solves the recursion with E = max(demand - after, 0)\n"
            "and no ceiling, so stock(t) equals demand(t) in every year.\n"
            "04_gap then finds 6,765 of those recruitments infeasible, and\n"
            "nothing revises the stock that 03 already published.",
            ha="center", va="top", fontsize=9.2, color=MISSING_COL, zorder=6,
            linespacing=1.55)
    # leader line, from the box to a point that actually sits on the red arc
    ax.add_patch(FancyArrowPatch(
        (box_x - 2.35, box_y - 0.05), (-2.05, -3.45),
        arrowstyle="-", linestyle=(0, (2, 3)),
        color=MISSING_COL, lw=1.1, alpha=0.75, zorder=3,
        connectionstyle="arc3,rad=0.10"))

    # ---------------------------------------------------- nodes
    for label, x, y, role in NODES.values():
        x = x * XS
        if role == "unobs":
            fc, ec, lw, ls, hatch = "white", COL["unobs"], 2.0, (0, (3, 2)), None
        elif role == "noreport":
            fc, ec, lw, ls, hatch = "white", COL["noreport"], 1.6, "-", "////"
        else:
            fc, ec, lw, ls, hatch = COL[role], "white", 1.6, "-", None
        ax.add_patch(Circle((x, y), R, facecolor=fc, edgecolor=ec,
                            linewidth=lw, linestyle=ls, hatch=hatch,
                            alpha=0.97, zorder=5))
        weight = "bold" if role == "outcome" else "normal"
        ax.text(x, y - R - 0.22, label, ha="center", va="top",
                fontsize=8.4, color="#1a1a1a", fontweight=weight,
                linespacing=1.15, zorder=6)

    # ---------------------------------------------------- legend
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=11,
                      markerfacecolor=COL[r], markeredgecolor="white", label=l)
               for r, l in ROLE_LABEL.items()]
    handles += [
        Line2D([0], [0], marker="o", linestyle="", markersize=11,
               markerfacecolor="white", markeredgecolor=COL["unobs"],
               label="Unobserved"),
        Line2D([0], [0], marker="o", linestyle="", markersize=11,
               markerfacecolor="white", markeredgecolor=COL["noreport"],
               label="Computed, not reportable"),
        Line2D([0], [0], color="#6b7280", lw=1.6, label="causal / forecast link"),
        Line2D([0], [0], color="#aab2ba", lw=1.4, ls=(0, (5, 3)),
               label="accounting identity"),
        Line2D([0], [0], color="#5D6D7E", lw=1.6, ls=(0, (7, 2, 1, 2)),
               label="estimated parameter enters here"),
        Line2D([0], [0], color="#4A235A", lw=1.6, ls=(0, (1, 2)),
               label="external assumption"),
        Line2D([0], [0], color="#95A5A6", lw=1.4, ls=(0, (1, 3)),
               label="hypothetical: the reserve is never observed"),
        Line2D([0], [0], color=MISSING_COL, lw=2.2, ls=(0, (6, 4)),
               label="ABSENT feedback: entries never revise the stock"),
    ]
    ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, -0.015), fontsize=9.4,
              handlelength=2.4, columnspacing=1.8)

    # ---------------------------------------------------- titles and notes
    fig.text(0.5, 0.985,
             "Teacher-gap mechanism, and the feedback the chain does not carry",
             ha="center", va="top", fontsize=17, fontweight="bold")
    fig.text(0.5, 0.964,
             "Mainland Portugal, 24 NUTS III regions, 2025-2040.    "
             "Demand is an identity on enrolment and a historically estimated ratio, "
             "except the third cycle and secondary, which uses an elasticity.",
             ha="center", va="top", fontsize=10, color="#555555")
    fig.text(0.5, 0.947,
             "The gap is summed PER CHANNEL with no cross-credit: netting the two "
             "gives exactly zero in all sixteen years, so the entire 6,765 arises "
             "from refusing that credit.",
             ha="center", va="top", fontsize=10, color=MISSING_COL)

    fig.text(0.5, 0.028,
             "One period of a recursive supply block: stock(t) = stock(t-1) - exits + entries.\n"
             "Exits is a parent of BOTH the lateral channel and recruitment need; the second "
             "path dominates, since 94.7% of the retirement wave comes from the 60+ band.\n"
             "The contracted reserve is unmeasured anywhere in the chain. It carries about 70% "
             "of the variance of max(gap_y,0) + max(gap_l,0), a quantity in which the graduate\n"
             "term is zero in nearly every replicate, so that share is in practice the lateral "
             "channel's own decomposition and is not comparable to the graduate panel.",
             ha="center", va="top", fontsize=8.6, color="#666666", linespacing=1.55)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.9, max(xs) + 2.0)
    ax.set_ylim(min(ys) - 1.9, max(ys) + 1.1)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.925, bottom=0.155)

    fig.savefig(str(OUT), dpi=220, facecolor="white")
    fig.savefig(str(OUT.with_suffix(".pdf")), facecolor="white")
    plt.close(fig)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
