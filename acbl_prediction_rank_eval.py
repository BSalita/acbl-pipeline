#!/usr/bin/env python3
"""Score ACBL predictions the way the postmortem uses them.

Section ranking (club and tournament): each pair's predicted session
percentage is the mean of its board predictions (``Pct_NS_Pred`` for North-South, the complement for
East-West). Pairs are ranked against the other pairs in the same section and
direction, as Mitchell games are scored, and compared with the actual ranks
by Spearman correlation. Variants:

- ``model_raw``: the model output
- ``model_section_centered``: shifted so each section averages 50%, as the
  postmortem does (cannot change ranks within a direction, only pair error)
- ``model_board_centered``: shifted so each board in a section averages 50%
- ``elo_line_*``: a line on the seat-average Elo gap, the baseline to beat

Joint contract/declarer: on a sample of test boards, accuracy of each
classifier alone and of one contract chosen from both models' probabilities.

Inputs are the test parquet written by ``acbl_prediction_data.py`` and the
row-aligned ``debug_predictions_*`` files written by
``acbl_prediction_train.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import polars as pl  # noqa: E402

ACBL_DIR = Path(r"E:\bridge\data\acbl")
MODELS_DIR = ACBL_DIR / "SavedModels"
MIN_PAIR_BOARDS = 8
MIN_SECTION_PAIRS = 4
Z95 = 1.959963984540054
JOINT_SAMPLE_ROWS = 300_000
ELO_FIT_FROM = date(2025, 1, 1)
SECTION = ["session_id", "section_id"]
META_COLUMNS = (
    "session_id",
    "section_id",
    "Board",
    "Pair_Number_NS",
    "Pair_Number_EW",
    "Pct_NS",
    *[f"Elo_R_{seat}_EventStart" for seat in "NESW"],
)


def seat_elo_gap() -> pl.Expr:
    return (
        (pl.col("Elo_R_N_EventStart") + pl.col("Elo_R_S_EventStart")) / 2.0
        - (pl.col("Elo_R_E_EventStart") + pl.col("Elo_R_W_EventStart")) / 2.0
    ).cast(pl.Float64)


def fit_elo_line(train_path: Path) -> tuple[float, float]:
    """Least-squares ``Pct_NS = intercept + slope * gap`` on recent training rows."""
    stats = (
        pl.scan_parquet(train_path)
        .filter(pl.col("Date") >= ELO_FIT_FROM)
        .select(seat_elo_gap().alias("x"), pl.col("Pct_NS").cast(pl.Float64).alias("y"))
        .drop_nulls()
        .select(
            pl.col("x").mean().alias("mx"),
            pl.col("y").mean().alias("my"),
            pl.cov("x", "y").alias("cxy"),
            pl.col("x").var().alias("vx"),
            pl.len().alias("rows"),
        )
        .collect()
        .row(0, named=True)
    )
    if not stats["vx"]:
        raise ValueError("Elo gap has no variance in the training rows")
    slope = stats["cxy"] / stats["vx"]
    print(f"Elo line fitted on {stats['rows']:,} training rows since {ELO_FIT_FROM}")
    return stats["my"] - slope * stats["mx"], slope


def center(column: str, over: list[str]) -> pl.Expr:
    return (pl.col(column) - pl.col(column).mean().over(over) + 0.5).clip(0.0, 1.0)


def pair_sessions(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """One row per pair per section and direction with 8+ boards: actual and predicted mean."""
    pieces = []
    for direction, number, east_west in (("NS", "Pair_Number_NS", False), ("EW", "Pair_Number_EW", True)):
        def side(name: str) -> pl.Expr:
            return (1.0 - pl.col(name)) if east_west else pl.col(name)

        pieces.append(
            frame.group_by([*SECTION, number])
            .agg(
                side("Pct_NS").mean().alias("actual"),
                *[side(name).mean().alias(name) for name in columns],
                pl.len().alias("boards"),
            )
            .filter(pl.col("boards") >= MIN_PAIR_BOARDS)
            .drop(number)
            .with_columns(pl.lit(direction).alias("direction"))
        )
    return pl.concat(pieces, how="vertical")


def interval_calibration(frame: pl.DataFrame, column: str) -> dict:
    """Spread for a pair's predicted session percentage: widening * board_sd / sqrt(boards).

    One pair's board errors move together, so board_sd / sqrt(boards) alone is
    too narrow. ``widening`` is the RMS of pair_error * sqrt(boards) / board_sd,
    which makes the 95% range (1.96 x spread) cover about 95% of pair sessions.
    """
    board_sd = float((frame[column] - frame["Pct_NS"]).std())
    pairs = pair_sessions(frame, [column]).with_columns(
        (pl.col(column) - pl.col("actual")).alias("error"),
        pl.col("boards").cast(pl.Float64).sqrt().alias("root_boards"),
    )
    widening = float(
        pairs.select(((pl.col("error") * pl.col("root_boards") / board_sd) ** 2).mean().sqrt()).item()
    )

    def coverage(factor: float) -> float:
        return float(
            pairs.select(
                (pl.col("error").abs() <= Z95 * factor * board_sd / pl.col("root_boards")).mean()
            ).item()
        )

    return {
        "column": column,
        "board_sd": board_sd,
        "widening": widening,
        "pairs": pairs.height,
        "coverage_95_plain": coverage(1.0),
        "coverage_95_widened": coverage(widening),
        "pair_error_sd": float(pairs["error"].std()),
        "median_boards": float(pairs["boards"].median()),
    }


def section_rank_quality(frame: pl.DataFrame, columns: list[str]) -> dict:
    """Spearman of predicted and actual pair percentage per section and direction."""
    pairs = pair_sessions(frame, columns)
    groups = (
        pairs.group_by([*SECTION, "direction"])
        .agg(
            pl.len().alias("pairs"),
            *[pl.corr(name, "actual", method="spearman").alias(name) for name in columns],
        )
        .filter(pl.col("pairs") >= MIN_SECTION_PAIRS)
    )
    if groups.is_empty():
        raise ValueError(
            f"No section direction has {MIN_SECTION_PAIRS}+ pairs with {MIN_PAIR_BOARDS}+ boards"
        )
    kept = pairs.join(groups.select(*SECTION, "direction"), on=[*SECTION, "direction"], how="semi")
    result = {"section_directions": groups.height, "pairs": kept.height}
    for name in columns:
        result[name] = {
            "mean_spearman": float(groups[name].fill_nan(None).mean()),
            "pair_mae": float((kept[name] - kept["actual"]).abs().mean()),
        }
    result["constant_half_pair_mae"] = float((kept["actual"] - 0.5).abs().mean())
    return result


def evaluate_ranking(side: str, suffix: str, predictions_dir: Path | None = None) -> dict:
    """predictions_dir: where debug_predictions_<side>_pct_ns.parquet is
    (default ACBL_DIR; acbl_prediction_train.py --out-dir passes its own)."""
    test_path = ACBL_DIR / f"acbl_{side}_prediction_data_test{suffix}.parquet"
    missing = sorted(set(META_COLUMNS) - set(pl.read_parquet_schema(test_path)))
    if missing:
        raise ValueError(f"{test_path.name} lacks {missing}; sections cannot be ranked")
    train_path = ACBL_DIR / f"acbl_{side}_prediction_data_train{suffix}.parquet"
    predictions_path = Path(predictions_dir or ACBL_DIR) / f"debug_predictions_{side}_pct_ns.parquet"
    meta = pl.read_parquet(test_path, columns=list(META_COLUMNS))
    predicted = pl.read_parquet(predictions_path)
    if predicted.height != meta.height:
        raise RuntimeError(f"{predictions_path.name} has {predicted.height:,} rows, test has {meta.height:,}")
    aligned = (meta["Pct_NS"] == predicted["Pct_NS"]) | (meta["Pct_NS"].is_null() & predicted["Pct_NS"].is_null())
    if not aligned.all():
        raise RuntimeError(f"{predictions_path.name} is not row-aligned with {test_path.name}")

    intercept, slope = fit_elo_line(train_path)
    frame = (
        meta.with_columns(predicted["Pct_NS_Pred"].cast(pl.Float64).alias("model_raw"))
        .filter(pl.col("Pct_NS").is_not_null() & pl.col("Pct_NS").is_finite())
        .with_columns(
            pl.col("Pct_NS").cast(pl.Float64),
            (intercept + slope * seat_elo_gap()).clip(0.0, 1.0).alias("elo_line_raw"),
        )
        .with_columns(
            center("model_raw", SECTION).alias("model_section_centered"),
            center("model_raw", [*SECTION, "Board"]).alias("model_board_centered"),
            center("elo_line_raw", [*SECTION, "Board"]).alias("elo_line_board_centered"),
        )
    )
    columns = [
        "model_raw",
        "model_section_centered",
        "model_board_centered",
        "elo_line_raw",
        "elo_line_board_centered",
    ]
    quality = section_rank_quality(frame, columns)
    quality["board_mae"] = {
        name: float((frame[name] - frame["Pct_NS"]).abs().mean()) for name in columns
    }
    quality["board_mae"]["constant_half"] = float((frame["Pct_NS"] - 0.5).abs().mean())
    quality["elo_line"] = {"intercept": intercept, "slope_per_100_elo": slope * 100.0}
    quality["interval"] = interval_calibration(frame, "model_section_centered")
    quality["test_rows"] = frame.height
    return quality


def _schema(models_dir: Path, model_name: str) -> dict:
    return json.loads((models_dir / f"{model_name}_schema.json").read_text(encoding="utf-8"))


def evaluate_joint(side: str, suffix: str, models_dir: Path, device: str) -> dict:
    from mlBridge.ffbridge_prediction_features import (
        ACBL_DECLARER_WEIGHT,
        contract_declarer,
        joint_contract_and_declarer,
    )
    from mlBridge.mlBridgeAiLib import predict_model

    contract_model = f"acbl_{side}_predicted_contract_torch_model"
    direction_model = f"acbl_{side}_predicted_declarer_direction_torch_model"
    schemas = {name: _schema(models_dir, name) for name in (contract_model, direction_model)}
    features = list(
        dict.fromkeys(
            name
            for schema in schemas.values()
            for name in schema["feature_dtypes"]
        )
    )
    test_path = ACBL_DIR / f"acbl_{side}_prediction_data_test{suffix}.parquet"
    total = pl.scan_parquet(test_path).select(pl.len()).collect().item()
    every = max(1, total // JOINT_SAMPLE_ROWS)
    started = datetime.now()
    sample = (
        pl.scan_parquet(test_path)
        .select(*features, "Contract", "Declarer_Direction")
        .gather_every(every)
        .collect()
    )
    print(f"Joint sample: every {every}th of {total:,} test rows -> {sample.height:,} "
          f"({(datetime.now() - started).total_seconds():.0f}s)")
    outputs = {}
    for name in (contract_model, direction_model):
        schema = schemas[name]
        frame = sample.select(list(schema["feature_dtypes"]))
        predicted = predict_model(models_dir, name, frame, device=device, return_probs=True)
        class_to_idx = schema["class_to_idx"]
        classes = sorted(class_to_idx, key=class_to_idx.get)
        base = schema["target_column"]
        outputs[name] = (classes, predicted[f"{base}_Probs"].list.to_array(len(classes)).to_numpy(), predicted[f"{base}_Pred"])
    contract_classes, contract_probs, contract_alone = outputs[contract_model]
    direction_classes, direction_probs, direction_alone = outputs[direction_model]
    joint_contract, joint_direction = joint_contract_and_declarer(
        contract_classes, contract_probs, direction_classes, direction_probs,
        declarer_weight=ACBL_DECLARER_WEIGHT,
    )
    frame = pl.DataFrame(
        {
            "contract": sample["Contract"].cast(pl.Utf8),
            "direction": sample["Declarer_Direction"].cast(pl.Utf8),
            "contract_alone": contract_alone.cast(pl.Utf8),
            "direction_alone": direction_alone.cast(pl.Utf8),
            "joint_contract": joint_contract,
            "joint_direction": joint_direction,
        }
    ).with_columns(
        pl.col("contract_alone").map_elements(contract_declarer, return_dtype=pl.Utf8).alias("contract_model_direction")
    )
    played = frame.filter(pl.col("direction").is_in(["N", "E", "S", "W"]))
    result = {
        "declarer_weight": ACBL_DECLARER_WEIGHT,
        "rows": frame.height,
        "played_rows": played.height,
        "contract_alone": frame.select((pl.col("contract_alone") == pl.col("contract")).mean()).item(),
        "contract_joint": frame.select((pl.col("joint_contract") == pl.col("contract")).mean()).item(),
        "direction_alone": played.select((pl.col("direction_alone") == pl.col("direction")).mean()).item(),
        "direction_from_contract_model": played.select(
            (pl.col("contract_model_direction") == pl.col("direction")).mean()
        ).item(),
        "direction_joint": played.select((pl.col("joint_direction") == pl.col("direction")).mean()).item(),
        "alone_models_agree": frame.filter(pl.col("contract_alone") != "PASS")
        .select((pl.col("contract_model_direction") == pl.col("direction_alone")).mean())
        .item(),
        "pass_predicted_alone": frame.select((pl.col("contract_alone") == "PASS").mean()).item(),
        "pass_predicted_joint": frame.select((pl.col("joint_contract") == "PASS").mean()).item(),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Section-rank and joint contract/declarer evaluation for ACBL.")
    parser.add_argument("--side", choices=("club", "tournament"), default="club")
    parser.add_argument("--input-suffix", default="_v2")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    started = datetime.now()
    print(f"Start {started:%Y-%m-%d %H:%M:%S}")
    metrics = {"side": args.side}
    metrics["ranking"] = evaluate_ranking(args.side, args.input_suffix)
    print(json.dumps(metrics["ranking"], indent=2))
    print(f"Ranking done ({(datetime.now() - started).total_seconds():.0f}s)")
    if not args.skip_joint:
        metrics["joint_contract_declarer"] = evaluate_joint(args.side, args.input_suffix, args.models_dir, args.device)
        print(json.dumps(metrics["joint_contract_declarer"], indent=2))
    out = ACBL_DIR / f"acbl_{args.side}_prediction_rank_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    ended = datetime.now()
    print(f"Wrote {out}\nEnd {ended:%Y-%m-%d %H:%M:%S} (elapsed {(ended - started).total_seconds():.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
