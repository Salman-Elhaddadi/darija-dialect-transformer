"""Render the two figures the README argues from.

Reads persisted analysis output only (reports/error_analysis.json,
reports/comparison.json) — it never recomputes a metric, so a figure cannot
disagree with the table it sits next to.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORTS = Path("reports")
FIGURES = Path("figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Fixed categorical slots, assigned in table order and never cycled.
SERIES = {
    "lr": "#2a78d6",
    "mlp": "#eb6834",
    "marbert_frozen": "#1baf7a",
    "marbert_finetuned": "#eda100",
}
PRETTY = {
    "lr": "LR",
    "mlp": "MLP",
    "marbert_frozen": "MARBERT frozen",
    "marbert_finetuned": "MARBERT fine-tuned",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": AXIS,
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9)


def plot_error_by_length(analysis):
    quartiles = ["Q1\nshortest", "Q2", "Q3", "Q4\nlongest"]
    rates = analysis["error_rate_by_quartile"]
    models = [m for m in SERIES if m in rates]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    style(ax)
    x = range(4)
    series = {}
    for model in models:
        values = [v * 100 for v in rates[model]]
        series[model] = values
        ax.plot(x, values, color=SERIES[model], linewidth=2, marker="o",
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
                label=PRETTY[model], zorder=3)

    # Direct label at each line end: identity never rests on colour alone, which
    # the palette contrast check requires for the low-contrast slots. Models can
    # finish within a hair of each other (LR and MLP converge by Q4), so the
    # labels are pushed apart to a legible minimum instead of overprinting.
    ax.set_ylim(bottom=0)
    min_gap = 0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    placed = 0.0
    for model, values in sorted(series.items(), key=lambda kv: kv[1][-1]):
        y = max(values[-1], placed + min_gap) if placed else values[-1]
        ax.annotate(f" {PRETTY[model]}", (3, y), color=INK_SECONDARY,
                    fontsize=9, va="center", ha="left", xytext=(8, 0),
                    textcoords="offset points")
        placed = y

    ax.set_xticks(list(x))
    ax.set_xticklabels(quartiles)
    ax.set_ylabel("error rate (%)")
    ax.set_xlim(-0.15, 3.9)
    ax.set_title("Shorter text is harder — for every model", fontsize=12,
                 color=INK, loc="left", pad=14)
    ax.legend(frameon=False, fontsize=8, ncol=len(models), loc="upper center",
              bbox_to_anchor=(0.5, -0.13), labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES / "error_by_length.png", dpi=200, facecolor=SURFACE)
    print(f"wrote {FIGURES}/error_by_length.png")


def plot_tradeoff(comparison):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)

    # Two colours for the real distinction (classical vs transformer); the
    # per-model identity is carried by the direct label, not by a 4th hue.
    for row in comparison:
        transformer = "MARBERT" in row["model"]
        colour = SERIES["marbert_frozen"] if transformer else SERIES["lr"]
        f1 = float(str(row["macro_f1"]).split(" ")[0])
        size = float(row["size_mb"])
        ax.scatter(float(row["p50_ms"]), f1, s=110, color=colour, zorder=3,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(f"  {row['model'].split(' (')[0]}\n  {size:.1f} MB" if size < 10
                    else f"  {row['model'].split(' (')[0]}\n  {size:.0f} MB",
                    (float(row["p50_ms"]), f1), color=INK_SECONDARY, fontsize=8.5,
                    va="center", ha="left", xytext=(8, 0), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("CPU p50 latency per request, ms (log scale)")
    ax.set_ylabel("test macro-F1")
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    ax.set_title("What the accuracy costs", fontsize=12, color=INK, loc="left", pad=14)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=SERIES["lr"], markeredgecolor=SURFACE, label="classical"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=SERIES["marbert_frozen"], markeredgecolor=SURFACE,
                   label="transformer"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right",
              labelcolor=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_latency.png", dpi=200, facecolor=SURFACE)
    print(f"wrote {FIGURES}/accuracy_vs_latency.png")


def main():
    FIGURES.mkdir(exist_ok=True)
    plot_error_by_length(json.loads((REPORTS / "error_analysis.json").read_text()))
    plot_tradeoff(json.loads((REPORTS / "comparison.json").read_text()))


if __name__ == "__main__":
    main()
