#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_prediction_charts.py

Stage 5d: show the confusion / bar / importance charts that 5c no longer
opens during training (Agg backend, 2026-09-15). Interactive matplotlib
only — safe because training is already finished.

Reads:
  acbl/debug_predictions_{club|tournament}_{target}.parquet
  (falls back to debug_predictions_{target}.parquet)
  acbl/SavedModels/acbl_{mode}_predicted_{target}_torch_model_importance.csv

Shows one figure set per completed model, then blocks on plt.show() until
the windows are closed.
"""

import os
import sys

# 5c sets MPLBACKEND=Agg. This step must use a GUI backend.
os.environ.pop("MPLBACKEND", None)

import matplotlib
for _backend in ("TkAgg", "QtAgg", "Qt5Agg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue

import argparse
import datetime
import pathlib

import polars as pl
import matplotlib.pyplot as plt

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / "mlBridge"
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f"mlBridge not found at {_MLBRIDGE}")
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)

from mlBridge.mlBridgeAiLib import analyze_prediction_results

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")
savedModelsPath = acblPath.joinpath("SavedModels")

Y_NAMES = ["Declarer_Direction", "Contract", "Pct_NS"]

_MODE_LABEL = {"club": "Club", "tournament": "Tournament"}
_AXIS_NOUN = {
    "Declarer_Direction": "Direction",
    "Contract": "Level and strain",
    "Level_Strain": "Level and strain",
    "Pct_NS": "Matchpoint percentage",
}


def _mode_label(mode: str) -> str:
    return _MODE_LABEL.get(mode, mode)


def _target_label(y_name: str) -> str:
    return "Contract" if y_name == "Level_Strain" else y_name


def _axis_noun(y_name: str) -> str:
    return _AXIS_NOUN.get(y_name, y_name)


def _model_pth(mode: str, y_name: str) -> pathlib.Path:
    return savedModelsPath / (
        f"acbl_{mode}_predicted_{y_name.lower()}_torch_model.pth"
    )


def _file_date(path: pathlib.Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _training_date(mode: str, y_name: str) -> str:
    for path in (
        _model_pth(mode, y_name),
        _importance_path(mode, y_name),
        _prediction_path(mode, y_name),
    ):
        dated = _file_date(path)
        if dated:
            return dated
    return "unknown date"


def _figure_title(mode: str, y_name: str, chart_type: str, trained: str) -> str:
    return (
        f"{_mode_label(mode)} · {_target_label(y_name)} · {chart_type} · trained {trained}"
    )


def _set_window_title(fig, title: str) -> None:
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def _apply_top_title(fig, title: str) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold")
    _set_window_title(fig, title)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    except Exception:
        fig.subplots_adjust(top=0.90)


def _chart_type_from_figure(fig) -> str:
    titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
    if any(t.startswith("Confusion Matrix") for t in titles) and len(titles) <= 1:
        return "Confusion matrix"
    if any(t == "Metrics Distribution" for t in titles):
        return "Metrics distribution"
    if any(t == "Predictions vs Actuals" for t in titles):
        return "Regression analysis"
    if any(t.startswith("Precision") for t in titles):
        return "Classification analysis"
    if any("importance" in t.lower() for t in titles):
        return "Feature importance"
    return "Prediction analysis"


def _relabel_axes(fig, y_name: str) -> None:
    noun = _axis_noun(y_name)
    for ax in fig.axes:
        title = ax.get_title()
        if not title:
            continue
        if title.startswith("Confusion Matrix"):
            ax.set_xlabel(f"Predicted {noun.lower()}")
            ax.set_ylabel(f"Actual {noun.lower()}")
        elif title in (
            "Precision by Class",
            "Recall by Class",
            "F1 Score by Class",
            "Accuracy by Class",
        ):
            ax.set_xlabel(noun)
        elif title == "True Class Distribution":
            ax.set_xlabel(noun)
            ax.set_ylabel("Frequency")
        elif title == "Predictions vs Actuals":
            ax.set_xlabel(f"Actual {noun.lower()}")
            ax.set_ylabel(f"Predicted {noun.lower()}")
        elif title == "Error Distribution":
            ax.set_xlabel(f"Prediction error ({noun.lower()})")
            ax.set_ylabel("Frequency")
        elif title == "Value Distributions":
            ax.set_xlabel(noun)
            ax.set_ylabel("Frequency")
        elif title == "Residuals vs Predicted":
            ax.set_xlabel(f"Predicted {noun.lower()}")
            ax.set_ylabel(f"Residual ({noun.lower()})")
        elif title == "Data Quality Overview":
            ax.set_xlabel("Data-quality category")
            ax.set_ylabel("Frequency")
        elif title == "Absolute Error Distribution":
            ax.set_xlabel(f"Absolute error ({noun.lower()})")
            ax.set_ylabel("Frequency")
        elif title == "Metrics Distribution":
            ax.set_xlabel("Precision, recall, or F1 score")
            ax.set_ylabel("Frequency")


def _annotate_new_figures(
    before_ids: set[int], mode: str, y_name: str, trained: str
) -> None:
    for num in plt.get_fignums():
        if num in before_ids:
            continue
        fig = plt.figure(num)
        _relabel_axes(fig, y_name)
        _apply_top_title(
            fig, _figure_title(mode, y_name, _chart_type_from_figure(fig), trained)
        )


def _prediction_path(mode: str, y_name: str) -> pathlib.Path | None:
    qualified = acblPath / f"debug_predictions_{mode}_{y_name.lower()}.parquet"
    legacy = acblPath / f"debug_predictions_{y_name.lower()}.parquet"
    if qualified.exists():
        return qualified
    if legacy.exists():
        return legacy
    return None


def _importance_path(mode: str, y_name: str) -> pathlib.Path | None:
    p = savedModelsPath / (
        f"acbl_{mode}_predicted_{y_name.lower()}_torch_model_importance.csv"
    )
    return p if p.exists() else None


def _plot_importance_bars(
    path: pathlib.Path, mode: str, y_name: str, trained: str, top_n: int = 50
) -> None:
    df = pl.read_csv(path)
    if "feature" not in df.columns or "importance" not in df.columns:
        print(f"  skip importance (unexpected columns): {path.name}")
        return
    top = df.sort("importance", descending=True).head(top_n)
    fig, ax = plt.subplots(figsize=(12, max(4, top.height * 0.22)))
    ax.barh(top["feature"].to_list()[::-1], top["importance"].to_list()[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Feature")
    _apply_top_title(fig, _figure_title(mode, y_name, "Feature importance", trained))


def show_charts(modes: list[str], targets: list[str]) -> int:
    shown = 0
    for mode in modes:
        for y_name in targets:
            print(f"\n===== {mode} {y_name} =====")
            trained = _training_date(mode, y_name)
            print(f"  trained: {trained}")
            pred_path = _prediction_path(mode, y_name)
            if pred_path is None:
                print(f"  no debug_predictions parquet for {mode}/{y_name}")
            else:
                print(f"  predictions: {pred_path.name}")
                pred_df = pl.read_parquet(pred_path)
                plot_y = y_name
                if y_name == "Contract" and "Contract" in pred_df.columns:
                    pred_df = pred_df.with_columns([
                        pl.col("Contract").cast(pl.Utf8).str.slice(0, 2)
                        .cast(pl.Categorical).alias("Level_Strain"),
                        pl.col("Contract_Pred").cast(pl.Utf8).str.slice(0, 2)
                        .cast(pl.Categorical).alias("Level_Strain_Pred"),
                    ])
                    plot_y = "Level_Strain"
                before_ids = set(plt.get_fignums())
                analyze_prediction_results(pred_df, plot_y)
                _annotate_new_figures(before_ids, mode, y_name, trained)
                shown += 1

            imp_path = _importance_path(mode, y_name)
            if imp_path is None:
                print(f"  no importance csv for {mode}/{y_name}")
            else:
                print(f"  importance: {imp_path.name}")
                _plot_importance_bars(imp_path, mode, y_name, trained)
                shown += 1

    if shown == 0:
        print("No prediction or importance artifacts found. Run 5c first.")
        return 1

    print(f"\nShowing {shown} chart group(s). Close the figure windows to finish 5d.")
    plt.show()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show 5c prediction charts (interactive; run after training)."
    )
    parser.add_argument("--club", action="store_true")
    parser.add_argument("--tournament", action="store_true")
    parser.add_argument(
        "--target", "-y", action="append", choices=Y_NAMES, default=None,
    )
    args = parser.parse_args()
    if not args.club and not args.tournament:
        modes = ["club", "tournament"]
    else:
        modes = []
        if args.club:
            modes.append("club")
        if args.tournament:
            modes.append("tournament")
    targets = args.target if args.target else list(Y_NAMES)
    return show_charts(modes, targets)


if __name__ == "__main__":
    sys.exit(main())
