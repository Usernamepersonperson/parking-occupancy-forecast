"""Command line: demo (reproduce the README numbers), fetch (rebuild data from open sources), collect (live poller)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .advise import advisory
from .data import collect_loop, fetch_archive, fetch_weather
from .evaluate import nearest_lots
from .pipeline import build_from_cache, load_committed, run_experiment, save_metrics

ROOT = Path(__file__).resolve().parent.parent


def _fmt(m: dict) -> str:
    lines = ["horizon  MAE naive  MAE persist  MAE GBM  gain   cover(80%)  width"]
    for h, v in m["by_horizon_min"].items():
        lines.append(
            f"{h:>4} min  {v['mae_seasonal_naive']:8.2f}  {v['mae_persistence']:10.2f}  {v['mae_gbm']:7.2f}  "
            f"{v['gbm_vs_naive_pct']:4.0f}%  {v['coverage_calibrated']:9.3f}  {v['mean_width_spaces']:5.1f}"
        )
    fw = m["full_warning"]
    for k in ("gbm", "seasonal_naive"):
        w = fw[k]
        lines.append(
            f"full warning [{k}]: precision {w['precision']:.2f}  recall {w['recall']:.2f}  "
            f"median lead {w['lead_median_min']:.0f} min  ({w['events']} events)"
        )
    for h, g in m["guidance"].items():
        lines.append(
            f"guidance @{h} min ({g['cases']} cases): stay {g['hit_stay']:.2f}  nearest {g['hit_nearest']:.2f}  "
            f"most-free-now {g['hit_most_free_now']:.2f}  forecast {g['hit_forecast']:.2f}"
        )
    return "\n".join(lines)


def cmd_demo(args) -> int:
    free, w, lots = load_committed(ROOT / "data")
    if args.fast:
        free = free.iloc[: 96 * 120]
    m, art = run_experiment(
        free, w, lots, max_iter=40 if args.fast else 120, train_stride=8 if args.fast else 4, ablations=args.ablations and not args.fast, verbose=True
    )
    out = Path(args.out)
    save_metrics(m, out / "metrics.json")
    print(_fmt(m))
    near = nearest_lots(lots.reindex(free.columns), k=3)
    names = list(free.columns)
    g = art["grid"]
    full4 = (g[4]["q50"] <= art["thr"]).sum(axis=1)
    k = int(np.argmax(full4)) if full4.max() > 0 else 0
    lot = int(np.argmax(g[4]["q50"][k] <= art["thr"])) if full4.max() > 0 else 0
    adv = advisory(g, k, lot, art["thr"], names, near)
    print(f"example advisory at {free.index[art['common'][k]]}: {adv['lot']} full by {adv['full_by_min']} min, reroute to {adv['reroute_to']}")
    import pickle

    cache = ROOT / "cache"
    cache.mkdir(exist_ok=True)
    keep = {k: art[k] for k in ("cap", "grid", "common", "thr", "splits")}
    (cache / "last_run.pkl").write_bytes(pickle.dumps((m, keep, free)))
    if not args.no_figures:
        from .figures import make_figures

        make_figures(m, art, free, ROOT / "docs")
    return 0


def cmd_figures(args) -> int:
    """Redraw the figures from the last demo run (cache/last_run.pkl, written by `demo`) without retraining."""
    import pickle

    from .figures import make_figures

    m, art, free = pickle.loads((ROOT / "cache" / "last_run.pkl").read_bytes())  # noqa: S301 (our own cache file)
    make_figures(m, art, free, ROOT / "docs")
    return 0


def cmd_fetch(args) -> int:
    cache = Path(args.cache)
    fetch_archive(args.year, cache / "archive")
    fetch_weather(f"{args.year}-01-01", f"{args.year}-12-31").to_csv(cache / f"weather_{args.year}.csv")
    free, w = build_from_cache(cache, args.year)
    free.round(0).astype("Int64").to_csv(ROOT / "data" / "parking_2022_15min.csv.gz", index_label="t", compression="gzip")
    w.round(2).to_csv(ROOT / "data" / "weather_2022.csv", index_label="t")
    print(f"wrote {free.shape[1]} car parks x {len(free)} steps")
    return 0


def cmd_collect(args) -> int:
    collect_loop(Path(args.out), args.interval, args.max_polls)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="parking-forecast", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="train on the committed 2022 data and print the evaluation")
    d.add_argument("--fast", action="store_true", help="small, quick run for smoke tests")
    d.add_argument("--out", default=str(ROOT / "results"))
    d.add_argument("--no-figures", action="store_true")
    d.add_argument("--ablations", action="store_true", help="also fit the no-weather / no-calendar variants (slower)")
    d.set_defaults(fn=cmd_demo)
    g = sub.add_parser("figures", help="redraw docs/*.png from the last demo run")
    g.set_defaults(fn=cmd_figures)
    f = sub.add_parser("fetch", help="download the open archive + weather into cache/ and rebuild data/")
    f.add_argument("--year", type=int, default=2022)
    f.add_argument("--cache", default=str(ROOT / "cache"))
    f.set_defaults(fn=cmd_fetch)
    c = sub.add_parser("collect", help="poll the live feed into a CSV")
    c.add_argument("--out", default=str(ROOT / "cache" / "live_feed.csv"))
    c.add_argument("--interval", type=int, default=300)
    c.add_argument("--max-polls", type=int, default=None)
    c.set_defaults(fn=cmd_collect)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
