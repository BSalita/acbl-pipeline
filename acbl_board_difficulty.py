#!/usr/bin/env python3
"""
Per-board bidding and declarer-play difficulty.

One row per (session_id, Board). Two numbers for each skill:

* perplexity — slope of the outcome on pair Elo, per 100 Elo points above
  1500. Positive means a stronger pair does better on this deal.
* difficulty — how the same fit looks at a fixed reference rating of 1700.
  Bidding difficulty is one minus the predicted par-contract rate. Play
  difficulty is the predicted tricks below double dummy.

Bidding skill is the Elo_R_*_Before of the pair that declares a making par
contract (from ParContracts; N/S map to NS and E/W map to EW). When that
side is ambiguous, the sign of Par_NS is the fallback. Par_Declarer in the
parquet is the declarer's par *score*, not a direction.

Bidding outcomes, all pre-play:

* Is_Par_Contract — the declared contract's double-dummy score is at least
  par for the declaring side. This is the headline slope.
* Is_Par_Suit — the declared strain appears in ParContracts.
* MP from the par side: MP_DD_Score_NS - MP_Par_NS when the par side is
  NS, and the opposite difference when the par side is EW. Positive slope
  means stronger pairs bid a contract worth more matchpoints at double dummy.

Play outcome is DD_Tricks_Diff (tricks taken minus double-dummy tricks in
the contract played), after subtracting a global defender-Elo effect.
Sacrifices and passed-out hands are dropped.

Club fields are small, so each raw slope is an empirical-Bayes mix of the
within-board OLS slope and the global slope. Boards with fewer than 8
usable tables, or with a par-side / declarer Elo spread under 15 points,
keep the global slope and a shrinkage weight of 0. A field that narrow
does not identify a per-100-Elo slope, and those slopes otherwise dominate
the prior variance.

Inputs:

* e:/bridge/data/acbl/acbl_{club|tournament}_elo_ratings.parquet
  — board-result columns plus Elo_R_*_Before. The pre-Elo augmented file
  does not carry those ratings.
* e:/bridge/data/acbl/acbl_{club|tournament}_board_results_augmented.parquet
  — Contract, MP_Par_NS, MP_DD_Score_NS, joined on session, board, round,
  pair numbers, and the four player ids. Pair numbers repeat across
  sections, so the player ids keep each table on its own row.

Output:

* acbl_{club|tournament}_board_difficulty.parquet
* acbl_{club|tournament}_board_difficulty_summary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from datetime import datetime

import numpy as np
import polars as pl


ACBL_DATA_DIR = pathlib.Path("e:/bridge/data/acbl")

ELO_CENTER = 1500.0
ELO_UNIT = 100.0
ELO_REFERENCE = 1700.0
MIN_TABLES = 8
LARGE_FIELD_TABLES = 16
MIN_LARGE_FIELDS = 50
SXX_MIN = 1e-8
# A per-100-Elo slope is not identified when the field spans only a few
# rating points. Those boards produce OLS slopes of tens of units, and that
# tail dominates the method-of-moments prior variance. 15 points is enough
# to drop the tail; on club data the prior variance falls from ~13 to ~0.008.
MIN_ELO_SPREAD = 15.0
# Residual SE below this is float dust on an exact fit (constant outcome,
# or a line the tables hit exactly). Treat it as known so the prior does
# not pull a measured zero slope.
SE_ZERO = 1e-8

JOIN_KEYS = [
    "session_id",
    "Board",
    "Round",
    "Pair_Number_NS",
    "Pair_Number_EW",
    "Player_ID_N",
    "Player_ID_E",
    "Player_ID_S",
    "Player_ID_W",
]
GROUP_KEYS = ["session_id", "Board"]

S_REF = (ELO_REFERENCE - ELO_CENTER) / ELO_UNIT


def _fmt_elapsed(start: float) -> str:
    secs = time.time() - start
    if secs < 60:
        return f"{secs:.1f}s"
    mins, s = divmod(secs, 60)
    return f"{int(mins)}m{int(s):02d}s"


def _finite(col: str) -> pl.Expr:
    c = pl.col(col)
    return c.is_not_null() & c.is_not_nan()


def _skill(elo_col: str) -> pl.Expr:
    return ((pl.col(elo_col) - ELO_CENTER) / ELO_UNIT).cast(pl.Float64)


def add_par_side(df: pl.DataFrame) -> pl.DataFrame:
    """Side that declares a making par contract, else the sign of Par_NS."""
    making = (
        pl.col("ParContracts")
        .list.eval(
            pl.element().filter(pl.element().struct.field("Result").fill_null(-1) >= 0)
        )
        .list.eval(
            pl.when(pl.element().struct.field("Pair_Direction").is_in(["N", "S", "NS"]))
            .then(pl.lit("NS"))
            .when(pl.element().struct.field("Pair_Direction").is_in(["E", "W", "EW"]))
            .then(pl.lit("EW"))
            .otherwise(None)
        )
        .list.drop_nulls()
        .list.unique()
    )
    n_strains = (
        pl.col("ParContracts")
        .list.eval(pl.element().struct.field("Strain"))
        .list.drop_nulls()
        .list.unique()
        .list.len()
        .fill_null(0)
        .cast(pl.UInt16)
    )
    return df.with_columns(making.alias("_making"), n_strains.alias("n_par_strains")).with_columns(
        pl.when(pl.col("_making").list.len() == 1)
        .then(pl.col("_making").list.first())
        .when(pl.col("Par_NS") > 0)
        .then(pl.lit("NS"))
        .when(pl.col("Par_NS") < 0)
        .then(pl.lit("EW"))
        .otherwise(None)
        .alias("par_side")
    ).drop("_making")


def load_tables(side: str, max_rows: int | None) -> pl.DataFrame:
    """Join Elo rows to contract and matchpoint columns from the augmented file."""
    elo_path = ACBL_DATA_DIR / f"acbl_{side}_elo_ratings.parquet"
    aug_path = ACBL_DATA_DIR / f"acbl_{side}_board_results_augmented.parquet"
    if not elo_path.exists():
        raise FileNotFoundError(elo_path)
    if not aug_path.exists():
        raise FileNotFoundError(aug_path)

    elo_names = set(pl.scan_parquet(elo_path).collect_schema().names())
    aug_names = set(pl.scan_parquet(aug_path).collect_schema().names())
    keys = [key for key in JOIN_KEYS if key in elo_names and key in aug_names]
    missing = [key for key in JOIN_KEYS if key not in keys and key != "Round"]
    if missing:
        raise RuntimeError(f"{side}: join keys missing from elo or augmented file: {missing}")
    if "Round" not in keys:
        print("  Round is absent; joining without it", flush=True)

    analysis_cols = [
        "Par_NS",
        "ParContracts",
        "Elo_R_NS_Before",
        "Elo_R_EW_Before",
        "Is_Par_Contract",
        "Is_Par_Suit",
        "Declarer_Pair_Direction",
        "Is_Sacrifice",
        "DD_Tricks_Diff",
    ]
    elo_cols = list(dict.fromkeys(keys + analysis_cols))
    aug_cols = list(dict.fromkeys(keys + ["Contract", "MP_Par_NS", "MP_DD_Score_NS"]))

    print(f"reading {elo_path.name}", flush=True)
    elo_lf = pl.scan_parquet(elo_path).select(elo_cols)
    if max_rows is not None:
        elo_lf = elo_lf.head(max_rows)
    elo = elo_lf.collect()
    print(f"  elo rows: {elo.height:,}", flush=True)

    print(f"reading {aug_path.name} (contract, par matchpoints)", flush=True)
    aug_lf = pl.scan_parquet(aug_path).select(aug_cols)
    if max_rows is not None:
        sessions = elo.get_column("session_id").unique().to_list()
        aug_lf = aug_lf.filter(pl.col("session_id").is_in(sessions))
    aug = aug_lf.collect()
    # Nulls do not match in a join. Round is null on some club sessions;
    # both sides use the same sentinel so those tables still line up.
    if "Round" in keys:
        elo = elo.with_columns(pl.col("Round").fill_null(0))
        aug = aug.with_columns(pl.col("Round").fill_null(0))
    before = aug.height
    aug = aug.unique(subset=keys, keep="first")
    if aug.height < before:
        print(f"  dropped {before - aug.height:,} duplicate augmented keys", flush=True)

    df = elo.join(aug, on=keys, how="left")
    if df.height != elo.height:
        raise RuntimeError(
            f"{side}: join changed row count from {elo.height:,} to {df.height:,}"
        )
    missing_contract = int(df.get_column("Contract").null_count())
    print(
        f"  joined {df.height:,} rows; contract nulls {missing_contract:,}",
        flush=True,
    )
    return df


def prepare_rows(df: pl.DataFrame) -> pl.DataFrame:
    """Par side, scaled Elo, and the three bidding / one play outcomes."""
    df = add_par_side(df)
    for col in ("Elo_R_NS_Before", "Elo_R_EW_Before"):
        df = df.with_columns(
            pl.when(pl.col(col).is_nan()).then(None).otherwise(pl.col(col)).alias(col)
        )
    df = df.with_columns(
        pl.when(pl.col("par_side") == "NS")
        .then(pl.col("Elo_R_NS_Before"))
        .when(pl.col("par_side") == "EW")
        .then(pl.col("Elo_R_EW_Before"))
        .otherwise(None)
        .alias("elo_par"),
        pl.when(pl.col("Declarer_Pair_Direction") == "NS")
        .then(pl.col("Elo_R_NS_Before"))
        .when(pl.col("Declarer_Pair_Direction") == "EW")
        .then(pl.col("Elo_R_EW_Before"))
        .otherwise(None)
        .alias("elo_dec"),
        pl.when(pl.col("Declarer_Pair_Direction") == "NS")
        .then(pl.col("Elo_R_EW_Before"))
        .when(pl.col("Declarer_Pair_Direction") == "EW")
        .then(pl.col("Elo_R_NS_Before"))
        .otherwise(None)
        .alias("elo_def"),
        pl.col("Is_Par_Contract").cast(pl.Float64).alias("y_par"),
        pl.col("Is_Par_Suit").cast(pl.Float64).alias("y_suit"),
        pl.when(pl.col("par_side") == "EW")
        .then(pl.col("MP_Par_NS") - pl.col("MP_DD_Score_NS"))
        .otherwise(pl.col("MP_DD_Score_NS") - pl.col("MP_Par_NS"))
        .cast(pl.Float64)
        .alias("y_mp"),
        pl.col("DD_Tricks_Diff").cast(pl.Float64).alias("y_tricks"),
    )
    df = df.with_columns(
        _skill("elo_par").alias("s_par"),
        _skill("elo_dec").alias("s_dec"),
        _skill("elo_def").alias("s_def"),
    )
    return df


def _sums_expr(s: str, y: str) -> list[pl.Expr]:
    return [
        pl.len().alias("n"),
        pl.col(s).sum().alias("sum_s"),
        pl.col(y).sum().alias("sum_y"),
        (pl.col(s) * pl.col(y)).sum().alias("sum_sy"),
        (pl.col(s) ** 2).sum().alias("sum_s2"),
        (pl.col(y) ** 2).sum().alias("sum_y2"),
    ]


def outcome_frame(df: pl.DataFrame, s: str, y: str, elo: str) -> pl.DataFrame:
    """Rows with a finite skill and outcome, kept as Float64 sums per board."""
    sub = df.filter(_finite(s) & _finite(y) & _finite(elo))
    if sub.is_empty():
        return pl.DataFrame(
            schema={
                "session_id": df.schema["session_id"],
                "Board": df.schema["Board"],
                "n": pl.UInt32,
                "sum_s": pl.Float64,
                "sum_y": pl.Float64,
                "sum_sy": pl.Float64,
                "sum_s2": pl.Float64,
                "sum_y2": pl.Float64,
                "elo_spread": pl.Float64,
            }
        )
    return sub.group_by(GROUP_KEYS).agg(
        _sums_expr(s, y) + [pl.col(elo).std(ddof=0).cast(pl.Float64).alias("elo_spread")]
    )


def global_slope(df: pl.DataFrame, s: str, y: str) -> float:
    """OLS slope of y on s across tables, ignoring the board grouping."""
    sub = df.filter(_finite(s) & _finite(y))
    if sub.height < MIN_TABLES:
        return 0.0
    sums = sub.select(_sums_expr(s, y)).row(0, named=True)
    beta, _se, _ms, _my = beta_se_from_sums(
        np.array([sums["n"]], dtype=np.float64),
        np.array([sums["sum_s"]], dtype=np.float64),
        np.array([sums["sum_y"]], dtype=np.float64),
        np.array([sums["sum_sy"]], dtype=np.float64),
        np.array([sums["sum_s2"]], dtype=np.float64),
        np.array([sums["sum_y2"]], dtype=np.float64),
        min_n=3,
    )
    if not np.isfinite(beta[0]):
        return 0.0
    return float(beta[0])


def beta_se_from_sums(
    n: np.ndarray,
    sum_s: np.ndarray,
    sum_y: np.ndarray,
    sum_sy: np.ndarray,
    sum_s2: np.ndarray,
    sum_y2: np.ndarray,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row OLS slope and residual SE from sufficient statistics.

    Returns beta, se, mean_s, mean_y. beta/se are NaN when n < min_n or the
    skill column has no spread. mean_s and mean_y are NaN when n == 0.
    """
    n = n.astype(np.float64, copy=False)
    sum_s = sum_s.astype(np.float64, copy=False)
    sum_y = sum_y.astype(np.float64, copy=False)
    sum_sy = sum_sy.astype(np.float64, copy=False)
    sum_s2 = sum_s2.astype(np.float64, copy=False)
    sum_y2 = sum_y2.astype(np.float64, copy=False)

    positive = n > 0
    mean_s = np.divide(sum_s, n, out=np.full(n.shape, np.nan), where=positive)
    mean_y = np.divide(sum_y, n, out=np.full(n.shape, np.nan), where=positive)
    sxx = np.zeros(n.shape, dtype=np.float64)
    syy = np.zeros(n.shape, dtype=np.float64)
    sxy = np.zeros(n.shape, dtype=np.float64)
    sxx[positive] = sum_s2[positive] - (sum_s[positive] ** 2) / n[positive]
    syy[positive] = sum_y2[positive] - (sum_y[positive] ** 2) / n[positive]
    sxy[positive] = sum_sy[positive] - sum_s[positive] * sum_y[positive] / n[positive]

    ok = positive & (n >= min_n) & (sxx > SXX_MIN) & np.isfinite(sxx)
    beta = np.full(n.shape, np.nan, dtype=np.float64)
    se = np.full(n.shape, np.nan, dtype=np.float64)
    beta[ok] = sxy[ok] / sxx[ok]
    sse = np.zeros(n.shape, dtype=np.float64)
    sse[ok] = syy[ok] - (beta[ok] ** 2) * sxx[ok]
    sse = np.maximum(sse, 0.0)
    se[ok] = np.sqrt(sse[ok] / (n[ok] - 2.0) / sxx[ok])
    return beta, se, mean_s, mean_y


def prior_variance(n: np.ndarray, beta: np.ndarray, se: np.ndarray) -> float:
    """Method-of-moments tau^2 from large fields: Var(beta) - mean(se^2)."""
    large = (n >= LARGE_FIELD_TABLES) & np.isfinite(beta) & np.isfinite(se)
    if int(large.sum()) < MIN_LARGE_FIELDS:
        large = (n >= MIN_TABLES) & np.isfinite(beta) & np.isfinite(se)
    if int(large.sum()) < 2:
        return 0.0
    return float(max(0.0, np.var(beta[large], ddof=1) - np.mean(se[large] ** 2)))


def shrink_slopes(
    n: np.ndarray,
    beta: np.ndarray,
    se: np.ndarray,
    mean_s: np.ndarray,
    mean_y: np.ndarray,
    beta0: float,
    tau2: float,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision-weighted mix of the raw slope and the global slope.

    kind 'miss' writes difficulty as one minus the predicted rate at Elo 1700,
    clipped to [0, 1]. kind 'tricks' writes predicted tricks below double dummy
    (the negation of the predicted trick difference).
    """
    se2 = np.where(np.isfinite(se), se ** 2, np.nan)
    measured = np.isfinite(beta) & np.isfinite(se2)
    weight = np.zeros(n.shape, dtype=np.float64)
    zero_se = measured & (se <= SE_ZERO)
    weight[zero_se] = 1.0
    pos = measured & (se > SE_ZERO)
    if tau2 > 0.0:
        weight[pos] = tau2 / (tau2 + se2[pos])
    raw_or_zero = np.where(np.isfinite(beta), beta, 0.0)
    published = weight * raw_or_zero + (1.0 - weight) * beta0
    has_rows = n > 0
    published = np.where(has_rows, published, np.nan)
    weight = np.where(has_rows, weight, np.nan)

    pred = mean_y + published * (S_REF - mean_s)
    if kind == "miss":
        difficulty = 1.0 - np.clip(pred, 0.0, 1.0)
    elif kind == "tricks":
        difficulty = -pred
    else:
        raise ValueError(f"unknown difficulty kind {kind!r}")
    difficulty = np.where(has_rows & np.isfinite(pred), difficulty, np.nan)
    return published, weight, difficulty


def attach_outcome(
    boards: pl.DataFrame,
    stats: pl.DataFrame,
    beta0: float,
    prefix: str,
    kind: str,
) -> tuple[pl.DataFrame, dict]:
    """Left-join one outcome's shrunk slope onto the board frame."""
    if stats.is_empty():
        n = np.zeros(boards.height, dtype=np.float64)
        beta = np.full(boards.height, np.nan)
        se = np.full(boards.height, np.nan)
        mean_s = np.full(boards.height, np.nan)
        mean_y = np.full(boards.height, np.nan)
        elo_spread = np.full(boards.height, np.nan)
        tau2 = 0.0
    else:
        aligned = boards.select(GROUP_KEYS).join(stats, on=GROUP_KEYS, how="left")
        n = aligned.get_column("n").fill_null(0).to_numpy().astype(np.float64)
        sum_s = aligned.get_column("sum_s").fill_null(0).to_numpy()
        sum_y = aligned.get_column("sum_y").fill_null(0).to_numpy()
        sum_sy = aligned.get_column("sum_sy").fill_null(0).to_numpy()
        sum_s2 = aligned.get_column("sum_s2").fill_null(0).to_numpy()
        sum_y2 = aligned.get_column("sum_y2").fill_null(0).to_numpy()
        elo_spread = aligned.get_column("elo_spread").to_numpy().astype(np.float64)
        beta, se, mean_s, mean_y = beta_se_from_sums(
            n, sum_s, sum_y, sum_sy, sum_s2, sum_y2, MIN_TABLES
        )
        flat = ~(np.isfinite(elo_spread) & (elo_spread >= MIN_ELO_SPREAD))
        beta = np.where(flat, np.nan, beta)
        se = np.where(flat, np.nan, se)
        tau2 = prior_variance(n, beta, se)

    published, weight, difficulty = shrink_slopes(
        n, beta, se, mean_s, mean_y, beta0, tau2, kind
    )
    slope_name = f"{prefix}_perplexity" if prefix in ("bid", "play") else f"{prefix}_slope"
    difficulty_name = f"{prefix}_difficulty" if prefix in ("bid", "play") else f"{prefix}_difficulty_unused"
    boards = boards.with_columns(
        pl.Series(f"n_{prefix}", n.astype(np.int32)),
        pl.Series(slope_name, published),
        pl.Series(f"{prefix}_weight", weight),
        pl.Series(difficulty_name, difficulty),
        pl.Series(f"{prefix}_slope_raw", beta),
        pl.Series(f"{prefix}_slope_se", se),
        pl.Series(f"{prefix}_elo_spread", elo_spread),
    )
    meta = {
        "prior_slope": beta0,
        "tau2": tau2,
        "n_measured": int(np.isfinite(beta).sum()),
    }
    return boards, meta


def defender_coefficient(df: pl.DataFrame) -> float:
    """Partial slope of DD_Tricks_Diff on defender Elo, given declarer Elo.

    Fit on every play-eligible table. The within-board play slope is then
    estimated on tricks after this defender term is removed, so a strong
    pair declaring against a weak pair is not credited for the defense.
    """
    sub = df.filter(_finite("s_dec") & _finite("s_def") & _finite("y_tricks"))
    if sub.height < MIN_TABLES:
        return 0.0
    sums = sub.select(
        pl.len().alias("n"),
        pl.col("s_dec").sum().alias("sum_d"),
        pl.col("s_def").sum().alias("sum_f"),
        pl.col("y_tricks").sum().alias("sum_y"),
        (pl.col("s_dec") ** 2).sum().alias("sum_d2"),
        (pl.col("s_def") ** 2).sum().alias("sum_f2"),
        (pl.col("s_dec") * pl.col("s_def")).sum().alias("sum_df"),
        (pl.col("s_dec") * pl.col("y_tricks")).sum().alias("sum_dy"),
        (pl.col("s_def") * pl.col("y_tricks")).sum().alias("sum_fy"),
    ).row(0, named=True)
    xtx = np.array(
        [
            [sums["n"], sums["sum_d"], sums["sum_f"]],
            [sums["sum_d"], sums["sum_d2"], sums["sum_df"]],
            [sums["sum_f"], sums["sum_df"], sums["sum_f2"]],
        ],
        dtype=np.float64,
    )
    xty = np.array([sums["sum_y"], sums["sum_dy"], sums["sum_fy"]], dtype=np.float64)
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        return 0.0
    return float(coef[2])


def play_frame(df: pl.DataFrame) -> tuple[pl.DataFrame, float]:
    """Non-sacrifice tables with a declarer, defender effect removed."""
    play = df.filter(
        pl.col("Declarer_Pair_Direction").is_in(["NS", "EW"])
        & pl.col("Is_Sacrifice").eq(False)
        & _finite("s_dec")
        & _finite("s_def")
        & _finite("y_tricks")
        & _finite("elo_dec")
    )
    gamma = defender_coefficient(play)
    print(f"  defender Elo coefficient: {gamma:+.4f} tricks per 100 Elo", flush=True)
    play = play.with_columns(
        (pl.col("y_tricks") - gamma * pl.col("s_def")).alias("y_play")
    )
    return play, gamma


def contract_features(df: pl.DataFrame, play: pl.DataFrame) -> pl.DataFrame:
    """Distinct contracts, and trick spread inside the modal contract."""
    n_contracts = df.group_by(GROUP_KEYS).agg(
        pl.col("Contract").drop_nulls().n_unique().cast(pl.UInt16).alias("n_contracts"),
        pl.col("Contract").null_count().cast(pl.UInt16).alias("n_contract_null"),
    )
    counted = (
        play.filter(pl.col("Contract").is_not_null())
        .group_by(GROUP_KEYS + ["Contract"])
        .agg(pl.len().alias("n_in_contract"), pl.col("y_tricks").std(ddof=0).alias("spread"))
        .sort(GROUP_KEYS + ["n_in_contract"], descending=[False, False, True])
    )
    modal = counted.group_by(GROUP_KEYS, maintain_order=True).agg(
        pl.col("spread").first().alias("trick_spread"),
        pl.col("n_in_contract").first().alias("n_modal_contract"),
    )
    # A single table has no spread.
    modal = modal.with_columns(
        pl.when(pl.col("n_modal_contract") >= 2)
        .then(pl.col("trick_spread"))
        .otherwise(None)
        .alias("trick_spread")
    ).drop("n_modal_contract")
    return n_contracts.join(modal, on=GROUP_KEYS, how="left")


def _dist(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"n": 0}
    pcts = [10, 25, 50, 75, 90]
    out = {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
    }
    for p in pcts:
        out[f"p{p}"] = float(np.percentile(v, p))
    return out


def _mean(values: np.ndarray) -> float | None:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return None
    return float(np.mean(v))


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    xs = x[ok]
    ys = y[ok]
    if xs.std() == 0 or ys.std() == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def sanity_report(boards: pl.DataFrame) -> dict:
    """Three checks from the plan, plus the metric distributions."""
    n_bid = boards.get_column("n_bid").to_numpy()
    n_play = boards.get_column("n_play").to_numpy()
    n_contracts = boards.get_column("n_contracts").fill_null(0).to_numpy()
    n_contract_null = boards.get_column("n_contract_null").fill_null(0).to_numpy()
    n_strains = boards.get_column("n_par_strains").fill_null(0).to_numpy()
    raw = boards.get_column("bid_slope_raw").to_numpy()
    published = boards.get_column("bid_perplexity").to_numpy()
    play_pub = boards.get_column("play_perplexity").to_numpy()
    play_raw = boards.get_column("play_slope_raw").to_numpy()
    spread = boards.get_column("trick_spread").to_numpy().astype(np.float64)

    measured = n_bid >= MIN_TABLES
    uniform = measured & (n_contracts == 1) & (n_contract_null == 0)
    multi_contract = measured & (n_contracts >= 3)
    one_strain = measured & (n_strains == 1)
    multi_strain = measured & (n_strains >= 2)
    play_measured = n_play >= MIN_TABLES

    uniform_abs = np.abs(raw[uniform]) if uniform.any() else np.array([])
    uniform_finite = uniform & np.isfinite(raw)
    report = {
        "distributions": {
            "bid_perplexity": _dist(published),
            "bid_difficulty": _dist(boards.get_column("bid_difficulty").to_numpy()),
            "bid_suit_slope": _dist(boards.get_column("bid_suit_slope").to_numpy()),
            "bid_mp_slope": _dist(boards.get_column("bid_mp_slope").to_numpy()),
            "play_perplexity": _dist(play_pub),
            "play_difficulty": _dist(boards.get_column("play_difficulty").to_numpy()),
            "bid_weight": _dist(boards.get_column("bid_weight").to_numpy()),
            "play_weight": _dist(boards.get_column("play_weight").to_numpy()),
        },
        "uniform_contract": {
            "n_boards": int(uniform.sum()),
            "mean_raw_slope": _mean(raw[uniform]),
            "mean_abs_raw_slope": _mean(uniform_abs),
            "fraction_raw_slope_at_zero": (
                float(np.mean(np.abs(raw[uniform_finite]) < 1e-8)) if uniform_finite.any() else None
            ),
            "n_uniform_unidentified": int(uniform.sum() - uniform_finite.sum()),
            "mean_published_perplexity": _mean(published[uniform]),
            "multi_contract_mean_abs_raw_slope": _mean(np.abs(raw[multi_contract])),
            "n_multi_contract": int(multi_contract.sum()),
        },
        "par_strains": {
            "n_one_strain": int(one_strain.sum()),
            "n_multi_strain": int(multi_strain.sum()),
            "mean_raw_slope_one_strain": _mean(raw[one_strain]),
            "mean_raw_slope_multi_strain": _mean(raw[multi_strain]),
            "mean_published_one_strain": _mean(published[one_strain]),
            "mean_published_multi_strain": _mean(published[multi_strain]),
        },
        "trick_spread": {
            "n_boards": int((play_measured & np.isfinite(spread)).sum()),
            "pearson_published": _pearson(play_pub[play_measured], spread[play_measured]),
            "pearson_raw": _pearson(play_raw[play_measured], spread[play_measured]),
        },
    }
    return report


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def print_summary(side: str, report: dict, meta: dict) -> None:
    print(f"\n--- {side} board difficulty ---")
    priors = meta["priors"]
    print(
        "  priors (per 100 Elo): "
        f"par={priors['bid']['prior_slope']:+.4f} "
        f"suit={priors['bid_suit']['prior_slope']:+.4f} "
        f"mp={priors['bid_mp']['prior_slope']:+.4f} "
        f"play={priors['play']['prior_slope']:+.4f}"
    )
    print(
        "  tau^2: "
        f"par={priors['bid']['tau2']:.6f} "
        f"suit={priors['bid_suit']['tau2']:.6f} "
        f"mp={priors['bid_mp']['tau2']:.6f} "
        f"play={priors['play']['tau2']:.6f} "
        f"defender_coef={meta['defender_coefficient']:+.4f}"
    )
    for name, dist in report["distributions"].items():
        if dist.get("n", 0) == 0:
            print(f"  {name}: no finite values")
            continue
        print(
            f"  {name}: n={dist['n']:,} mean={dist['mean']:+.4f} std={dist['std']:.4f} "
            f"p10={dist['p10']:+.4f} p50={dist['p50']:+.4f} p90={dist['p90']:+.4f}"
        )
    u = report["uniform_contract"]
    print(
        "  uniform contract (n_bid>=8, one contract): "
        f"n={u['n_boards']:,} mean_raw={_fmt(u['mean_raw_slope'])} "
        f"mean_|raw|={_fmt(u['mean_abs_raw_slope'])} "
        f"fraction_raw~0={_fmt(u['fraction_raw_slope_at_zero'], 3)} "
        f"unidentified={u['n_uniform_unidentified']:,} "
        f"mean_published={_fmt(u['mean_published_perplexity'])}"
    )
    print(
        "  three-or-more contracts: "
        f"n={u['n_multi_contract']:,} mean_|raw|={_fmt(u['multi_contract_mean_abs_raw_slope'])}"
    )
    s = report["par_strains"]
    print(
        "  par strains (n_bid>=8): "
        f"one n={s['n_one_strain']:,} raw={_fmt(s['mean_raw_slope_one_strain'])} "
        f"published={_fmt(s['mean_published_one_strain'])}; "
        f"multi n={s['n_multi_strain']:,} raw={_fmt(s['mean_raw_slope_multi_strain'])} "
        f"published={_fmt(s['mean_published_multi_strain'])}"
    )
    t = report["trick_spread"]
    print(
        "  play perplexity vs within-contract trick spread: "
        f"n={t['n_boards']:,} "
        f"pearson_published={_fmt(t['pearson_published'])} "
        f"pearson_raw={_fmt(t['pearson_raw'])}"
    )


def analyze(side: str, max_rows: int | None) -> tuple[pl.DataFrame, dict]:
    t0 = time.time()
    print(f"\n=== {side} ===")
    print(f"start: {datetime.now().isoformat(timespec='seconds')}")
    df = prepare_rows(load_tables(side, max_rows))

    print("bidding slopes", flush=True)
    boards = df.group_by(GROUP_KEYS).agg(
        pl.len().cast(pl.UInt32).alias("n_tables"),
        pl.col("n_par_strains").first(),
    )
    bid_rows = df.filter(pl.col("par_side").is_not_null())
    par_stats = outcome_frame(bid_rows, "s_par", "y_par", "elo_par")
    suit_stats = outcome_frame(bid_rows, "s_par", "y_suit", "elo_par")
    mp_stats = outcome_frame(bid_rows, "s_par", "y_mp", "elo_par")
    beta_par = global_slope(bid_rows, "s_par", "y_par")
    beta_suit = global_slope(bid_rows, "s_par", "y_suit")
    beta_mp = global_slope(bid_rows, "s_par", "y_mp")

    boards, meta_par = attach_outcome(boards, par_stats, beta_par, "bid", "miss")
    boards, meta_suit = attach_outcome(boards, suit_stats, beta_suit, "bid_suit", "miss")
    boards, meta_mp = attach_outcome(boards, mp_stats, beta_mp, "bid_mp", "miss")

    boards = boards.drop(
        [c for c in (
            "bid_suit_difficulty_unused", "bid_mp_difficulty_unused",
            "bid_suit_elo_spread", "bid_mp_elo_spread",
            "bid_suit_slope_raw", "bid_mp_slope_raw",
            "bid_suit_slope_se", "bid_mp_slope_se",
            "n_bid_suit", "n_bid_mp",
        ) if c in boards.columns]
    )

    print("play slopes", flush=True)
    play, gamma = play_frame(df)
    play_stats = outcome_frame(play, "s_dec", "y_play", "elo_dec")
    beta_play = global_slope(play, "s_dec", "y_play")
    boards, meta_play = attach_outcome(boards, play_stats, beta_play, "play", "tricks")

    print("contract features", flush=True)
    features = contract_features(df, play)
    boards = boards.join(features, on=GROUP_KEYS, how="left")
    boards = boards.with_columns(
        pl.col("n_contracts").fill_null(0).cast(pl.UInt16),
        pl.col("n_contract_null").fill_null(0).cast(pl.UInt16),
    )

    keep = [
        "session_id",
        "Board",
        "n_tables",
        "n_bid",
        "n_play",
        "bid_elo_spread",
        "play_elo_spread",
        "bid_perplexity",
        "bid_difficulty",
        "bid_weight",
        "bid_slope_raw",
        "bid_slope_se",
        "bid_suit_slope",
        "bid_suit_weight",
        "bid_mp_slope",
        "bid_mp_weight",
        "play_perplexity",
        "play_difficulty",
        "play_weight",
        "play_slope_raw",
        "play_slope_se",
        "n_par_strains",
        "n_contracts",
        "n_contract_null",
        "trick_spread",
    ]
    boards = boards.select(keep).sort(GROUP_KEYS)

    report = sanity_report(boards)
    meta = {
        "side": side,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_boards": boards.height,
        "max_rows": max_rows,
        "elo_center": ELO_CENTER,
        "elo_unit": ELO_UNIT,
        "elo_reference": ELO_REFERENCE,
        "min_tables": MIN_TABLES,
        "min_elo_spread": MIN_ELO_SPREAD,
        "defender_coefficient": gamma,
        "priors": {
            "bid": meta_par,
            "bid_suit": meta_suit,
            "bid_mp": meta_mp,
            "play": meta_play,
        },
        "sanity": report,
        "elapsed": _fmt_elapsed(t0),
    }
    print_summary(side, report, meta)
    print(f"  elapsed {_fmt_elapsed(t0)}", flush=True)
    return boards, meta


def write_outputs(side: str, boards: pl.DataFrame, meta: dict, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"acbl_{side}_board_difficulty.parquet"
    json_path = out_dir / f"acbl_{side}_board_difficulty_summary.json"
    boards.write_parquet(parquet_path)
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {parquet_path}")
    print(f"wrote {json_path}")


def run_self_test() -> None:
    """Numeric checks for the slope, shrinkage, and defender adjustment."""
    # Exact slope 0.2 on a 20-table board. y = 0.5 + 0.2 * s, s = -1..1.
    s = np.linspace(-1.0, 1.0, 20)
    y = 0.5 + 0.2 * s
    n = np.array([20.0])
    beta, se, mean_s, mean_y = beta_se_from_sums(
        n, np.array([s.sum()]), np.array([y.sum()]), np.array([(s * y).sum()]),
        np.array([(s ** 2).sum()]), np.array([(y ** 2).sum()]), min_n=8,
    )
    assert abs(beta[0] - 0.2) < 1e-8, beta
    assert se[0] < 1e-6, se
    assert abs(mean_y[0] - 0.5) < 1e-8

    # Constant outcome: raw slope 0 and SE 0, so the published slope stays 0
    # even when the global prior is not.
    y0 = np.ones(10)
    s0 = np.linspace(-1.0, 1.0, 10)
    beta0_hat, se0, ms0, my0 = beta_se_from_sums(
        np.array([10.0]), np.array([s0.sum()]), np.array([y0.sum()]),
        np.array([(s0 * y0).sum()]), np.array([(s0 ** 2).sum()]), np.array([(y0 ** 2).sum()]),
        min_n=8,
    )
    assert abs(beta0_hat[0]) < 1e-8
    assert se0[0] < 1e-6
    published, weight, difficulty = shrink_slopes(
        np.array([10.0]), beta0_hat, se0, ms0, my0, beta0=0.15, tau2=0.01, kind="miss",
    )
    assert weight[0] == 1.0
    assert abs(published[0]) < 1e-8
    # Everyone found par, so difficulty at 1700 is 0.
    assert abs(difficulty[0]) < 1e-9

    # Below the minimum, the published slope is the prior and the weight is 0.
    # mean y = 0.25, mean s = 0, prior 0.1 -> rate at s=2 is 0.45, difficulty 0.55.
    small_s = np.array([0.0, 0.0, 1.0, -1.0])
    small_y = np.array([0.0, 0.0, 1.0, 0.0])
    b_s, se_s, ms_s, my_s = beta_se_from_sums(
        np.array([4.0]), np.array([small_s.sum()]), np.array([small_y.sum()]),
        np.array([(small_s * small_y).sum()]), np.array([(small_s ** 2).sum()]),
        np.array([(small_y ** 2).sum()]), min_n=8,
    )
    assert not np.isfinite(b_s[0])
    published, weight, difficulty = shrink_slopes(
        np.array([4.0]), b_s, se_s, ms_s, my_s, beta0=0.1, tau2=0.01, kind="miss",
    )
    assert weight[0] == 0.0
    assert abs(published[0] - 0.1) < 1e-9
    assert abs(difficulty[0] - (1.0 - 0.45)) < 1e-9

    # Tricks-below at the reference rating: mean tricks diff -0.4, mean skill 0,
    # slope 0.1 -> prediction at s=2 is -0.2, difficulty (tricks below DD) is 0.2.
    published, _w, difficulty = shrink_slopes(
        np.array([20.0]), np.array([0.1]), np.array([0.0]),
        np.array([0.0]), np.array([-0.4]), beta0=0.0, tau2=1.0, kind="tricks",
    )
    assert abs(published[0] - 0.1) < 1e-9
    assert abs(difficulty[0] - 0.2) < 1e-9

    # Defender partial coefficient. y = 1 + 0.3*declarer - 0.5*defender.
    rng = np.random.default_rng(0)
    n_rows = 400
    s_dec = rng.normal(size=n_rows)
    s_def = rng.normal(size=n_rows)
    y_tricks = 1.0 + 0.3 * s_dec - 0.5 * s_def
    frame = pl.DataFrame({"s_dec": s_dec, "s_def": s_def, "y_tricks": y_tricks})
    gamma = defender_coefficient(frame)
    assert abs(gamma - (-0.5)) < 1e-8, gamma

    # Prior variance: true slopes differ, sampling noise is removed.
    beta = np.array([0.0, 0.2, 0.4, 0.6])
    se = np.array([0.1, 0.1, 0.1, 0.1])
    n_large = np.array([20.0, 20.0, 20.0, 20.0])
    tau2 = prior_variance(n_large, beta, se)
    expected = float(np.var(beta, ddof=1) - 0.01)
    assert abs(tau2 - expected) < 1e-9, (tau2, expected)
    print("self-test ok")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club", action="store_true", help="Club games only")
    parser.add_argument("--tournament", action="store_true", help="Tournament games only")
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=ACBL_DATA_DIR,
        help="Directory for the parquet and summary JSON",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Read only the first N Elo rows (smoke test)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Check the slope math and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.club and args.tournament:
        sides = ["club", "tournament"]
    elif args.club:
        sides = ["club"]
    elif args.tournament:
        sides = ["tournament"]
    else:
        sides = ["club", "tournament"]

    overall = time.time()
    for side in sides:
        boards, meta = analyze(side, args.max_rows)
        write_outputs(side, boards, meta, args.out_dir)
    print(f"\nTOTAL elapsed: {_fmt_elapsed(overall)}")


if __name__ == "__main__":
    main()
