"""Chart helpers: four chart types, each saved for light and dark backgrounds.

Rules followed: the title states the finding, one message and one axis per
chart, one colour when the marks measure the same thing and two only when two
series are compared (always with a legend), thin marks, hairline gridlines,
and numbers printed at the marks that matter.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "blue": "#3987e5", "orange": "#d95926"},
}
SOURCE = "Data: Upworthy Research Archive (Matias, Munger, Aubin Le Quere and Ebersole), osf.io/jd64p.  Analysis: Elisha Abraham."
DPI = 200
WIDTH = 7.2

_installed = {font.name for font in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next(
    (name for name in ("Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans") if name in _installed), "sans-serif")


def _figure(theme: dict, title: str, subtitle: str, height: float):
    fig = plt.figure(figsize=(WIDTH, height), dpi=DPI)
    fig.patch.set_facecolor(theme["surface"])
    top = 1 - 0.19 / height
    fig.text(0.04, top, title, fontsize=12.5, fontweight="semibold", color=theme["ink"], ha="left", va="top")
    fig.text(0.04, top - 0.30 / height, subtitle, fontsize=9, color=theme["ink2"], ha="left", va="top")
    fig.text(0.04, 0.10 / height, SOURCE, fontsize=6.8, color=theme["muted"], ha="left", va="bottom")
    return fig


def _axes(fig, theme: dict, box: list[float], grid: str, baseline: str):
    ax = fig.add_axes(box)
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side == baseline)
    if baseline:
        ax.spines[baseline].set_color(theme["axis"])
        ax.spines[baseline].set_linewidth(0.8)
    ax.grid(axis=grid, color=theme["grid"], linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=9, colors=theme["muted"], pad=5)
    return ax


def _per_pixel(ax) -> tuple[float, float]:
    """Data units covered by one logical pixel (1/100 inch) on each axis (linear axes only)."""
    ax.figure.canvas.draw()
    scale = ax.figure.dpi / 100
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([(0, 0), (scale, scale)])
    return abs(x1 - x0), abs(y1 - y0)


def _legend(fig, theme: dict, entries: list[tuple[str, str]], x: float, y: float, height: float) -> None:
    """A row of colour swatches with names, in figure coordinates."""
    for colour, text in entries:
        fig.add_artist(Rectangle((x, y - 0.05 / height), 0.10 / WIDTH, 0.10 / height, facecolor=colour, edgecolor="none",
                                 transform=fig.transFigure))
        label = fig.text(x + 0.16 / WIDTH, y, text, fontsize=8.5, color=theme["ink2"], ha="left", va="center")
        fig.canvas.draw()
        x += 0.16 / WIDTH + label.get_window_extent().width / fig.bbox.width + 0.22 / WIDTH


def _save(fig, folder: Path, name: str, theme_name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}_{theme_name}.png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def _extra(subtitle: str) -> float:
    return 0.17 * subtitle.count("\n")  # room for a second subtitle line


def histogram(counts: list[int], expected: float, title: str, subtitle: str, folder: Path, name: str,
              x_title: str, note: str) -> None:
    """Columns for equal-width bins between 0 and 1, with a line for the count expected in each bin by chance."""
    height = 4.0 + _extra(subtitle)
    width = 1 / len(counts)
    for theme_name, theme in THEMES.items():
        fig = _figure(theme, title, subtitle, height)
        ax = _axes(fig, theme, [0.09, 0.78 / height, 0.87, 1 - (1.72 + _extra(subtitle)) / height], grid="y", baseline="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(counts) * 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
        dx, dy = _per_pixel(ax)
        for i, count in enumerate(counts):
            left, bar = i * width + dx, width - 2 * dx  # a two-pixel gap between columns
            ax.add_patch(FancyBboxPatch((left, 0), bar, count, boxstyle=f"round,pad=0,rounding_size={3 * dx}",
                                        mutation_aspect=dy / dx, facecolor=theme["blue"], edgecolor="none", zorder=3))
            ax.add_patch(Rectangle((left, 0), bar, max(count - 3 * dy, 0), facecolor=theme["blue"], edgecolor="none", zorder=3))
        ax.axhline(expected, color=theme["ink2"], linewidth=1, linestyle=(0, (4, 3)), zorder=4)
        ax.text(0.995, expected + 7 * dy, note, fontsize=8.5, color=theme["ink2"], ha="right", va="bottom")
        ax.text(width / 2 + 8 * dx + width / 2, counts[0], f"{counts[0]:,} tests", fontsize=9, color=theme["ink"], ha="left", va="top")
        ax.set_xlabel(x_title, fontsize=9, color=theme["ink2"], labelpad=6)
        _save(fig, folder, name, theme_name)


def curve(xs: list[float], ys: list[float], marks: list[dict], title: str, subtitle: str, folder: Path, name: str,
          x_title: str, x_ticks: list[int], y_ticks: list[int]) -> None:
    """One falling line on a log x axis. marks: dicts with x, y, text and where the label sits ("left" or "right")."""
    height = 4.0 + _extra(subtitle)
    for theme_name, theme in THEMES.items():
        fig = _figure(theme, title, subtitle, height)
        ax = _axes(fig, theme, [0.09, 0.78 / height, 0.87, 1 - (1.72 + _extra(subtitle)) / height], grid="y", baseline="bottom")
        ax.set_xscale("log")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(0, y_ticks[-1] * 1.08)
        ax.set_xticks(x_ticks, [f"{tick:,}" for tick in x_ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y_ticks, [f"{tick}%" if tick else "0" for tick in y_ticks])
        ax.plot(xs, ys, color=theme["blue"], linewidth=1.5, solid_capstyle="round", zorder=3)
        for mark in marks:
            ax.plot(mark["x"], mark["y"], "o", markersize=6.5, color=theme["blue"], markeredgecolor=theme["surface"],
                    markeredgewidth=1.5, zorder=4)
            side = mark.get("side", "right")
            ax.annotate(mark["text"], (mark["x"], mark["y"]), xytext=(9 if side == "right" else -9, 6), textcoords="offset points",
                        ha="left" if side == "right" else "right", va="bottom", fontsize=8.5, color=theme["ink"], linespacing=1.3)
        ax.set_xlabel(x_title, fontsize=9, color=theme["ink2"], labelpad=6)
        _save(fig, folder, name, theme_name)


def measured_against_true(points: list[tuple[float, float]], title: str, subtitle: str, folder: Path, name: str,
                          limit: int, x_title: str, y_title: str, line_note: str, diagonal_note: str) -> None:
    """Points joined by a line, against the diagonal where both axes would be equal. Values in percent."""
    height = 4.6 + _extra(subtitle)
    for theme_name, theme in THEMES.items():
        fig = _figure(theme, title, subtitle, height)
        ax = _axes(fig, theme, [0.11, 0.78 / height, 0.62, 1 - (1.72 + _extra(subtitle)) / height], grid="both", baseline="bottom")
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(theme["axis"])
        ax.spines["left"].set_linewidth(0.8)
        ticks = list(range(0, limit + 1, 20))
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xticks(ticks, [f"{tick}%" if tick else "0" for tick in ticks])
        ax.set_yticks(ticks, [f"{tick}%" if tick else "0" for tick in ticks])
        ax.plot([0, limit], [0, limit], color=theme["ink2"], linewidth=1, linestyle=(0, (4, 3)), zorder=2)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        ax.plot(xs, ys, color=theme["blue"], linewidth=1.5, zorder=3)
        ax.plot(xs, ys, "o", markersize=6.5, color=theme["blue"], markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=4)
        ax.text(1.03, ys[-1] / limit, line_note, transform=ax.transAxes, fontsize=8.5, color=theme["ink"], ha="left", va="center",
                linespacing=1.3)
        ax.text(1.03, 0.97, diagonal_note, transform=ax.transAxes, fontsize=8.5, color=theme["ink2"], ha="left", va="top",
                linespacing=1.3)
        ax.set_xlabel(x_title, fontsize=9, color=theme["ink2"], labelpad=6)
        ax.set_ylabel(y_title, fontsize=9, color=theme["ink2"], labelpad=6)
        _save(fig, folder, name, theme_name)


def paired_forest(rows: list[dict], series: tuple[str, str], title: str, subtitle: str, folder: Path, name: str,
                  x_ticks: list[float], x_title: str) -> None:
    """Two estimates with 95% intervals per row, on a log axis around 1.

    rows: dicts with label and, for each of the two series, a (ratio, low, high) tuple under "first" and
    "second". A row with neither is a group heading. The number printed is the second series.
    """
    height = 2.17 + _extra(subtitle) + 0.40 * len(rows)
    for theme_name, theme in THEMES.items():
        fig = _figure(theme, title, subtitle, height)
        ax = _axes(fig, theme, [0.40, 0.84 / height, 0.38, 1 - (2.02 + _extra(subtitle)) / height], grid="x", baseline="")
        ax.set_xscale("log")
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xticks(x_ticks, [f"{tick:g}" for tick in x_ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(range(len(rows)), [row["label"] for row in rows])
        ax.tick_params(axis="y", colors=theme["ink2"], pad=6)
        ax.axvline(1, color=theme["ink2"], linewidth=0.8, zorder=2)
        for y, (row, tick) in enumerate(zip(rows, ax.get_yticklabels())):
            if "second" not in row:
                tick.set_color(theme["ink"])
                tick.set_fontweight("semibold")
                continue
            for key, shift, colour in (("first", -0.15, theme["blue"]), ("second", 0.15, theme["orange"])):
                ratio, low, high = row[key]
                ax.hlines(y + shift, low, high, color=theme["ink2"], linewidth=1, zorder=3)
                ax.plot(ratio, y + shift, "o", markersize=6.5, color=colour, markeredgecolor=theme["surface"],
                        markeredgewidth=1.3, zorder=4)
            ratio, low, high = row["second"]
            ax.text(1.04, y, f"{ratio:.3f}  ({low:.3f} to {high:.3f})", transform=ax.get_yaxis_transform(), fontsize=9,
                    color=theme["ink"], ha="left", va="center")
        ax.set_xlabel(x_title, fontsize=9, color=theme["ink2"], labelpad=6)
        _legend(fig, theme, [(theme["blue"], series[0]), (theme["orange"], series[1])], 0.40,
                1 - (0.93 + _extra(subtitle)) / height, height)
        _save(fig, folder, name, theme_name)
