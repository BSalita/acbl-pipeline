"""Row-count and feature-set experiments on the finished club prediction files.

Reads acbl_club_prediction_data_{train,test}_v2.parquet. Does not rebuild
stages 5a or 5b and does not write into SavedModels.

Samples are cached under e:/bridge/data/acbl/experiments/. Each trial trains
with the same architecture, epoch cap, and early-stop patience as stage 5c.
The test set is one fixed 500,000-row draw, so trials are comparable.
Validation is one shard out of about twenty (~5%), not half the sample.

Full-run club test numbers to beat or match (10,068,124 rows):
  Declarer_Direction accuracy 0.6644
  Contract accuracy 0.3795
  Pct_NS MAE 0.2595
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import polars as pl
import pyarrow.parquet as pq

_SRC_DIR = Path(__file__).resolve().parent.parent
for _p in (_SRC_DIR, _SRC_DIR / "mlBridge", Path(__file__).resolve().parent):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)

from acbl_hp_search_lib import PCT_NS_DROP_PATTERNS  # noqa: E402
from acbl_prediction_train import (  # noqa: E402
    _cast_enums_to_categorical,
    _prepare_target_frame,
)
from mlBridge.mlBridgeAiLib import (  # noqa: E402
    create_torch_shards,
    generate_and_save_schema,
    predict_model,
    train_model_from_shards,
)

ACBL = Path("e:/bridge/data/acbl")
TRAIN_SRC = ACBL / "acbl_club_prediction_data_train_v2.parquet"
TEST_SRC = ACBL / "acbl_club_prediction_data_test_v2.parquet"
EXP = ACBL / "experiments"
MODELS = EXP / "models"
RESULTS = EXP / "results.csv"
TEST_ROWS = 500_000
SEED = 42
FULL_BASELINE = {
    "Declarer_Direction": {"accuracy": 0.6644},
    "Contract": {"accuracy": 0.3795},
    "Pct_NS": {"mae": 0.2595},
}
IMPORTANCE = {
    "Declarer_Direction": ACBL / "SavedModels" / "acbl_club_predicted_declarer_direction_torch_model_importance.csv",
    "Contract": ACBL / "SavedModels" / "acbl_club_predicted_contract_torch_model_importance.csv",
    "Pct_NS": ACBL / "SavedModels" / "acbl_club_predicted_pct_ns_torch_model_importance.csv",
}
ROW_LADDER = [100_000, 500_000, 2_000_000, 8_000_000]
FEATURE_SETS = ["top100", "top500", "top1500", "drop_noisy"]


def _hparams(y_name: str) -> dict:
    """Same overrides stage 5c uses for club."""
    hp = {
        "layers": [2048, 1024, 512, 256],
        "dropout": 0.10,
        "epochs": 20,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "bs": 8192,
        "y_range": None,
        "patience": 3,
    }
    if y_name == "Declarer_Direction":
        hp.update(layers=[1024, 512, 256, 128], weight_decay=0.0, patience=5)
    elif y_name == "Contract":
        hp.update(dropout=0.20, lr=3e-4, weight_decay=0.0, patience=5)
    elif y_name == "Pct_NS":
        hp.update(layers=[512, 256, 128], dropout=0.05, lr=3e-4, weight_decay=1e-4, patience=5, y_range=None)
    return hp


def sample_parquet(src: Path, dest: Path, n_rows: int, seed: int) -> Path:
    """Draw n_rows from randomly chosen row groups and cache the result."""
    if dest.exists():
        have = pq.ParquetFile(dest).metadata.num_rows
        if have == n_rows:
            print(f"Reusing sample {dest.name} ({have:,} rows)")
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(src)
    rng = random.Random(seed)
    order = list(range(pf.num_row_groups))
    rng.shuffle(order)
    chosen: list[int] = []
    total = 0
    target = max(n_rows, int(n_rows * 1.05))
    for index in order:
        chosen.append(index)
        total += pf.metadata.row_group(index).num_rows
        if total >= target:
            break
    print(f"Reading {len(chosen)} row groups ({total:,} rows) from {src.name} for a {n_rows:,}-row sample")
    t0 = time.time()
    table = pf.read_row_groups(sorted(chosen))
    df = pl.from_arrow(table)
    del table
    if df.height > n_rows:
        df = df.sample(n=n_rows, seed=seed, shuffle=True)
    df.write_parquet(dest, compression="zstd", compression_level=3)
    print(f"Wrote {dest.name}: {df.height:,} rows in {time.time() - t0:.1f}s")
    del df
    gc.collect()
    return dest


def _column_from_importance_name(name: str) -> str | None:
    if name.startswith("_CAT_"):
        return name[len("_CAT_"):]
    if name.startswith("_EMBEDDED_CAT_"):
        rest = name[len("_EMBEDDED_CAT_"):]
        return rest.rsplit("_", 1)[0]
    if name.startswith("_"):
        return None
    return name


def select_features(df: pl.DataFrame, y_name: str, feature_set: str) -> pl.DataFrame:
    if feature_set == "all":
        return df
    keep_target = [y_name] if y_name in df.columns else []
    available = set(df.columns)
    if feature_set == "drop_noisy":
        import re
        patterns = [re.compile(p) for p in PCT_NS_DROP_PATTERNS]
        drop = [c for c in df.columns if c != y_name and any(p.match(c) for p in patterns)]
        print(f"drop_noisy removing {len(drop):,} columns")
        return df.drop(drop)
    if feature_set.startswith("top"):
        k = int(feature_set[3:])
        ranked = pl.read_csv(IMPORTANCE[y_name])["feature"].to_list()
        chosen: list[str] = []
        for name in ranked:
            col = _column_from_importance_name(str(name))
            if col and col in available and col != y_name and col not in chosen:
                chosen.append(col)
            if len(chosen) >= k:
                break
        print(f"{feature_set} kept {len(chosen):,} features")
        return df.select(chosen + keep_target)
    raise ValueError(f"Unknown feature set {feature_set}")


def _metrics(pred: pl.DataFrame, y_name: str) -> dict:
    pred_col = f"{y_name}_Pred"
    if y_name == "Pct_NS":
        y = pred[y_name].cast(pl.Float64)
        p = pred[pred_col].cast(pl.Float64)
        mae = float((y - p).abs().mean())
        y_std = float(y.std())
        p_std = float(p.std())
        ratio = p_std / y_std if y_std else float("nan")
        return {"accuracy": "", "majority": "", "macro_f1": "", "mae": f"{mae:.4f}", "variance_ratio": f"{ratio:.3f}"}
    yt = pred[y_name].cast(pl.Utf8)
    yp = pred[pred_col].cast(pl.Utf8)
    n = pred.height
    acc = float((yt == yp).fill_null(False).mean())
    counts = yt.value_counts()
    majority = float(counts["count"].max()) / n if n else float("nan")
    labels = sorted({v for v in yt.drop_nulls().to_list()} | {v for v in yp.drop_nulls().to_list()})
    index = {label: i for i, label in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for truth, guess in zip(yt.to_list(), yp.to_list()):
        if truth in index and guess in index:
            cm[index[truth]][index[guess]] += 1
    f1s = []
    for i in range(len(labels)):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels))) - tp
        fn = sum(cm[i]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "accuracy": f"{acc:.4f}",
        "majority": f"{majority:.4f}",
        "macro_f1": f"{macro:.3f}",
        "mae": "",
        "variance_ratio": "",
    }


def _append_result(row: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS.exists()
    fields = [
        "timestamp", "target", "train_rows", "feature_set", "n_features", "test_rows",
        "accuracy", "majority", "macro_f1", "mae", "variance_ratio", "seconds",
        "epochs", "model_name", "status", "note",
    ]
    with RESULTS.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def _done(target: str, train_rows: int, feature_set: str) -> bool:
    if not RESULTS.exists():
        return False
    with RESULTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("status") == "ok"
                and row.get("target") == target
                and row.get("feature_set") == feature_set
                and int(row.get("train_rows") or 0) == train_rows
            ):
                return True
    return False


def _load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    with RESULTS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_trial(y_name: str, train_rows: int, feature_set: str, epochs: int | None = None) -> dict:
    if _done(y_name, train_rows, feature_set) and epochs is None:
        print(f"Skip finished trial {y_name} rows={train_rows} features={feature_set}")
        return {}
    hp = _hparams(y_name)
    if epochs is not None:
        hp["epochs"] = epochs
        hp["patience"] = 0
    test_path = sample_parquet(TEST_SRC, EXP / f"club_test_rows{TEST_ROWS}_seed{SEED}.parquet", TEST_ROWS, SEED)
    train_path = sample_parquet(
        TRAIN_SRC,
        EXP / f"club_train_rows{train_rows}_seed{SEED}.parquet",
        train_rows,
        SEED + train_rows,
    )
    print(f"Loading train sample {train_path.name}")
    train_df = _prepare_target_frame(_cast_enums_to_categorical(pl.read_parquet(train_path)), y_name, for_training=True)
    train_df = select_features(train_df, y_name, feature_set)
    model_name = f"exp_club_{y_name.lower()}_r{train_rows}_{feature_set}"
    MODELS.mkdir(parents=True, exist_ok=True)
    for path in MODELS.glob(f"{model_name}_shard_*.pt"):
        path.unlink()
    t0 = time.time()
    schema = generate_and_save_schema(
        train_df,
        MODELS,
        model_name,
        y_name,
        layers=hp["layers"],
        dropout=hp["dropout"],
        apply_scaling_parameters=True,
        y_range=hp["y_range"],
        verbose=False,
    )
    n_features = len(schema.get("feature_dtypes") or {})
    print(f"Training {model_name} on {train_df.height:,} rows x {n_features:,} features")
    n_rows = train_df.height
    shard_rows = max(1_000, n_rows // 20)
    if math.ceil(n_rows / shard_rows) < 2:
        shard_rows = max(1, n_rows // 2)
    create_torch_shards(train_df, schema, shard_rows_count=shard_rows, apply_scaling=True)
    del train_df
    gc.collect()
    train_model_from_shards(
        schema,
        epochs=hp["epochs"],
        bs=hp["bs"],
        lr=hp["lr"],
        use_amp=False,
        weight_decay=hp["weight_decay"],
        device="cuda",
        seed=SEED,
        early_stop_patience=hp["patience"],
        class_weights=None,
        verbose=True,
    )
    print(f"Loading test sample {test_path.name}")
    test_df = _prepare_target_frame(_cast_enums_to_categorical(pl.read_parquet(test_path)), y_name, for_training=False)
    test_df = select_features(test_df, y_name, feature_set)
    y_series = test_df[y_name]
    features = list(schema["feature_dtypes"].keys())
    missing = [c for c in features if c not in test_df.columns]
    if missing:
        raise RuntimeError(f"Test sample missing {len(missing)} training features, e.g. {missing[:5]}")
    pred = predict_model(MODELS, model_name, test_df.select(features), device="cuda")
    pred = pred.with_columns(y_series.alias(y_name))
    metrics = _metrics(pred, y_name)
    seconds = round(time.time() - t0, 1)
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": y_name,
        "train_rows": train_rows,
        "feature_set": feature_set,
        "n_features": n_features,
        "test_rows": TEST_ROWS,
        "seconds": seconds,
        "epochs": hp["epochs"],
        "model_name": model_name,
        "status": "ok",
        "note": "",
        **metrics,
    }
    _append_result(row)
    print(
        f"RESULT {y_name} rows={train_rows:,} features={feature_set} "
        f"n_features={n_features} acc={metrics['accuracy']} mae={metrics['mae']} "
        f"in {seconds:.0f}s"
    )
    for path in MODELS.glob(f"{model_name}_shard_*.pt"):
        path.unlink()
    del test_df, pred
    gc.collect()
    return row


def _metric_value(row: dict) -> float:
    if row["target"] == "Pct_NS":
        return -float(row["mae"])
    return float(row["accuracy"])


def _ok_rows(target: str, feature_set: str) -> list[dict]:
    return [
        row for row in _load_results()
        if row.get("status") == "ok" and row.get("target") == target and row.get("feature_set") == feature_set
        and int(row["train_rows"]) in ROW_LADDER
    ]


def run_plan() -> None:
    """Row ladder on Declarer_Direction, then feature cuts, then the other targets."""
    for n_rows in ROW_LADDER[:3]:
        run_trial("Declarer_Direction", n_rows, "all")
    curve = sorted(_ok_rows("Declarer_Direction", "all"), key=lambda row: int(row["train_rows"]))
    by_n = {int(row["train_rows"]): row for row in curve}
    if 2_000_000 in by_n and 500_000 in by_n:
        gap = float(by_n[2_000_000]["accuracy"]) - float(by_n[500_000]["accuracy"])
        print(f"Declarer_Direction 500,000 -> 2,000,000 accuracy gap {gap:.4f}")
        if gap > 0.01:
            run_trial("Declarer_Direction", ROW_LADDER[3], "all")
            curve = sorted(_ok_rows("Declarer_Direction", "all"), key=lambda row: int(row["train_rows"]))
    best = max(curve, key=_metric_value)
    best_acc = float(best["accuracy"])
    chosen_rows = int(best["train_rows"])
    for row in curve:
        if best_acc - float(row["accuracy"]) <= 0.01:
            chosen_rows = int(row["train_rows"])
            break
    print(f"Row plateau: {chosen_rows:,} train rows (best accuracy {best_acc:.4f})")
    for feature_set in FEATURE_SETS:
        run_trial("Declarer_Direction", chosen_rows, feature_set)
    feature_rows = [
        row for row in _load_results()
        if row.get("status") == "ok" and row.get("target") == "Declarer_Direction"
        and int(row["train_rows"]) == chosen_rows
    ]
    all_row = next(row for row in feature_rows if row["feature_set"] == "all")
    all_acc = float(all_row["accuracy"])
    chosen_features = "all"
    for feature_set in FEATURE_SETS:
        match = next((row for row in feature_rows if row["feature_set"] == feature_set), None)
        if match and all_acc - float(match["accuracy"]) <= 0.01:
            chosen_features = feature_set
            break
    print(f"Feature set within 0.01 of all-features: {chosen_features}")
    for y_name in ("Contract", "Pct_NS"):
        run_trial(y_name, chosen_rows, "all")
        if chosen_features != "all":
            run_trial(y_name, chosen_rows, chosen_features)
    print("Plan complete. Results:", RESULTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample-size and feature experiments on club _v2 data.")
    parser.add_argument("--plan", action="store_true", help="Run the row ladder, then feature cuts.")
    parser.add_argument("--target", default="Declarer_Direction", choices=list(FULL_BASELINE))
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--feature-set", default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override the stage-5c epoch cap (smoke tests).")
    args = parser.parse_args()
    if args.plan:
        run_plan()
    else:
        run_trial(args.target, args.rows, args.feature_set, epochs=args.epochs)


if __name__ == "__main__":
    main()
