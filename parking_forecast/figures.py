"""README figure (1600x900) and phone-sized dark post figure (960x700)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .features import STEP_MIN  # noqa: E402

FONTS = ["Segoe UI", "DejaVu Sans"]
DARK = {
    "bg": "#16213A", "grid": "#2A3750", "text": "#F8FAFC", "muted": "#A9B6CB", "accent": "#38BDF8",
    "green": "#34D399", "amber": "#FBBF24", "red": "#F87171", "grey": "#64748B", "violet": "#A78BFA",
}
SOURCE = "Source: City of Zurich car-park guidance (CC0) via open archive; weather: Open-Meteo (CC BY 4.0)"


def pick_example(art: dict, free, h: int = 4, days: int = 3) -> tuple[int, int, int]:
    """Lot index and [start, stop) origin rows of the busiest window (most full/not-full crossings) in the test period."""
    common = art["common"]
    arr = free.to_numpy(dtype=float)[common + h]
    full = arr <= art["thr"]
    flips = np.abs(np.diff(full.astype(int), axis=0))
    w = days * 96
    best = (0, 0, 0)
    score_best = -1
    for li in range(arr.shape[1]):
        c = np.concatenate([[0], np.cumsum(flips[:, li])])
        for s in range(0, max(1, len(common) - w), 48):
            score = c[min(s + w, len(c) - 1)] - c[s]
            nan_frac = np.isnan(arr[s : s + w, li]).mean()
            if nan_frac < 0.05 and score > score_best:
                score_best, best = score, (li, s, min(s + w, len(common)))
    return best


def _style_axes(ax, fg, grid, muted):
    ax.grid(True, color=grid, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(grid)
    ax.tick_params(colors=muted)


def make_figure(m: dict, art: dict, free, path: Path) -> None:
    plt.rcParams["font.family"] = FONTS
    fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=100, facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.12, hspace=0.42, wspace=0.18)
    c_naive, c_pers, c_gbm = "#94A3B8", "#F59E0B", "#0284C7"
    hs = list(m["by_horizon_min"])
    x = np.arange(len(hs))

    ax = axs[0, 0]
    for k, (key, lab, col) in enumerate([("mae_seasonal_naive", "seasonal naive", c_naive), ("mae_persistence", "persistence", c_pers), ("mae_gbm", "gradient boosting", c_gbm)]):
        vals = [m["by_horizon_min"][h][key] for h in hs]
        bars = ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=lab, color=col)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_xticks(x, [f"{h} min" for h in hs])
    ax.set_ylabel("mean absolute error (spaces)")
    ax.set_title("A. Error vs forecast horizon (test: 16 Oct - 31 Dec 2022)", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center")
    ax.set_ylim(0, max(m["by_horizon_min"][h]["mae_seasonal_naive"] for h in hs) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[0, 1]
    li, s, e = pick_example(art, free)
    common = art["common"]
    g = art["grid"][4]
    idx = free.index[common[s:e] + 4]
    act = free.to_numpy(dtype=float)[common[s:e] + 4, li]
    ax.fill_between(idx, g["q10"][s:e, li], g["q90"][s:e, li], color="#BAE6FD", label="80 % interval (60 min ahead)")
    ax.plot(idx, g["q50"][s:e, li], color=c_gbm, linewidth=1.6, label="median forecast")
    ax.plot(idx, act, color="#0F172A", linewidth=1.2, label="observed")
    ax.axhline(art["thr"][li], color="#DC2626", linestyle="--", linewidth=1.0, label="'full' threshold")
    ax.set_ylabel("free spaces")
    ax.set_title(f"B. Example: {free.columns[li].replace('_', ' ')}, 60-min-ahead forecast", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylim(None, float(np.nanmax(g["q90"][s:e, li])) * 1.3)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(alpha=0.3)

    ax = axs[1, 0]
    raw = [m["by_horizon_min"][h]["coverage_raw"] * 100 for h in hs]
    cal = [m["by_horizon_min"][h]["coverage_calibrated"] * 100 for h in hs]
    b1 = ax.bar(x - 0.18, raw, 0.36, color="#94A3B8", label="quantile GBM, raw")
    b2 = ax.bar(x + 0.18, cal, 0.36, color=c_gbm, label="after conformal calibration")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{b.get_height():.1f}", ha="center", fontsize=10)
    ax.axhline(80, color="#DC2626", linestyle="--", linewidth=1)
    ax.text(len(hs) - 0.52, 80.6, "nominal 80 %", color="#DC2626", fontsize=10, ha="right", va="bottom")
    ax.set_ylim(60, 96)
    ax.set_xticks(x, [f"{h} min" for h in hs])
    ax.set_ylabel("interval coverage on test (%)")
    ax.set_title("C. Do the 80 % intervals hold?", loc="left", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1, 1]
    gd = m["guidance"]
    labels = list(gd)
    keys = [("hit_stay", "stay at intended", "#F87171"), ("hit_nearest", "nearest alternative", "#94A3B8"), ("hit_most_free_now", "most free now", "#F59E0B"), ("hit_forecast", "forecast-guided", "#059669")]
    for k, (key, lab, col) in enumerate(keys):
        vals = [gd[h][key] * 100 for h in labels]
        bars = ax.bar(np.arange(len(labels)) + (k - 1.5) * 0.2, vals, 0.2, label=lab, color=col)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=10)
    ax.set_xticks(np.arange(len(labels)), [f"arrive in {h} min\n({gd[h]['cases']} cases)" for h in labels])
    ax.set_ylim(0, 135)
    ax.set_ylabel("recommended park had space on arrival (%)")
    fw = m["full_warning"]["gbm"]
    ax.set_title(
        f"D. Re-routing; full-by warning precision {fw['precision'] * 100:.0f} %, recall {fw['recall'] * 100:.0f} %",
        loc="left", fontsize=12, fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=10, loc="upper center", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.text(0.06, 0.945, "Car-park free-space forecast, 15 min to 2 h ahead", fontsize=19, fontweight="bold", ha="left")
    fig.text(0.06, 0.905, f"{len(free.columns)} car parks in Zurich, 2022 full-year archive; chronological split, nothing tuned on the test period", fontsize=12, color="#475569", ha="left")
    fig.text(0.06, 0.02, SOURCE, fontsize=9, color="#64748B", ha="left")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def make_post_figure(m: dict, path: Path) -> None:
    plt.rcParams["font.family"] = FONTS
    d = DARK
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 7.0), dpi=100, facecolor=d["bg"])
    fig.subplots_adjust(left=0.09, right=0.97, top=0.76, bottom=0.2, wspace=0.3)
    hs = list(m["by_horizon_min"])
    x = np.arange(len(hs))

    ax = axs[0]
    ax.set_facecolor(d["bg"])
    naive = [m["by_horizon_min"][h]["mae_seasonal_naive"] for h in hs]
    gbm = [m["by_horizon_min"][h]["mae_gbm"] for h in hs]
    b1 = ax.bar(x - 0.2, naive, 0.4, color=d["grey"], label="seasonal naive")
    b2 = ax.bar(x + 0.2, gbm, 0.4, color=d["accent"], label="forecast model")
    for b in (*b1, *b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{b.get_height():.1f}", ha="center", color=d["text"], fontsize=12)
    ax.set_xticks(x, [f"{h} min" for h in hs])
    ax.set_ylim(0, max(naive) * 1.3)
    ax.set_title("Error in free spaces (MAE)", color=d["text"], fontsize=13, loc="left", pad=8)
    ax.legend(frameon=False, labelcolor=d["text"], fontsize=11, loc="upper right")
    _style_axes(ax, d["text"], d["grid"], d["text"])
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", visible=False)

    ax = axs[1]
    ax.set_facecolor(d["bg"])
    h0 = "30" if "30" in m["guidance"] else next(iter(m["guidance"]))
    gd = m["guidance"][h0]
    items = [
        ("stay", gd["hit_stay"], d["red"]),
        ("nearest", gd["hit_nearest"], d["grey"]),
        ("most free\nnow", gd["hit_most_free_now"], d["amber"]),
        ("forecast", gd["hit_forecast"], d["green"]),
    ]
    bars = ax.bar(range(4), [v * 100 for _, v, _ in items], 0.66, color=[c for _, _, c in items])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{b.get_height():.0f} %", ha="center", color=d["text"], fontsize=13, fontweight="bold")
    ax.set_xticks(range(4), [n for n, _, _ in items])
    ax.set_ylim(0, 118)
    ax.set_title("Space on arrival (% of cases)", color=d["text"], fontsize=13, loc="left", pad=8)
    _style_axes(ax, d["text"], d["grid"], d["text"])
    ax.tick_params(labelsize=12)
    ax.grid(axis="x", visible=False)

    fig.text(0.04, 0.945, "Will the car park have space when you arrive?", color=d["text"], fontsize=19, fontweight="bold", ha="left")
    fig.text(0.04, 0.895, "Open data: Zurich car-park feed archive, 36 car parks, test 16 Oct to 31 Dec 2022", color=d["muted"], fontsize=12, ha="left")
    fig.text(0.04, 0.08, f"Right: arrive in {h0} min, {gd['cases']} cases where the intended car park was forecast full.", color=d["muted"], fontsize=11, ha="left")
    fig.text(0.04, 0.035, "Forecast ties a live-count rule here. One year, one city. Weather: Open-Meteo (CC BY 4.0).", color=d["muted"], fontsize=11, ha="left")
    fig.savefig(path, dpi=100, facecolor=d["bg"])
    plt.close(fig)


def make_figures(m: dict, art: dict, free, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    make_figure(m, art, free, out_dir / "figure.png")
    make_post_figure(m, out_dir / "post-figure.png")
    print(f"wrote {out_dir / 'figure.png'} and {out_dir / 'post-figure.png'}")


__all__ = ["make_figure", "make_figures", "make_post_figure", "pick_example", "STEP_MIN"]
