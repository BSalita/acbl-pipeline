# ACBL Model Training Results

Append-only log of prediction-model training runs (`acbl_prediction_train.py`,
stage 5c of `acbl_all.bat`). One dated entry per run, newest first.

Purpose: longitudinal comparison across retrains. Raw console output lives in
`logs/`; numbers that justify code decisions (architectures, thresholds) stay
as comments next to the code they justify (see e.g. the hp-search notes in
`acbl_prediction_train.py`).

Reading the metrics:

- **Declarer_Direction / Contract** (classification): accuracy on the held-out
  test split vs the majority-class baseline. Macro F1 is heavily penalized by
  rare classes (doubled/redoubled contracts), so accuracy + lift is the
  headline number.
- **Pct_NS** (regression): MAE and variance ratio (pred_std / actual_std).
  Single-board matchpoint percentage is mostly noise; the empirical ceiling
  found by hp search (17 trials, 2026-04) is MAE ~0.263, variance ratio ~0.165.
  Matching that ceiling is success, not underfitting to be "fixed".

---

## 2026-10-19 → 2026-10-20 (tournament retrain without ids and dates)

5b tournament rebuilt (22 min) so the test parquet keeps the String
`event_id` / `session_id` / `section_id` (6,035 sections, no duplicate
section/board/NS-pair rows). 5c tournament retrained all 3 targets (7h 29m)
with `Date`, `game_date`, `section_id`, and `Club` added to the schema
blacklist. Tournament models had only used `Date` and `game_date` of these.
Club is not retrained yet (waiting for full October data); the served club
models still take `section_id`, `Club`, and dates as inputs.

| Metric (tournament test, 1.27M rows) | 2026-10-08 | 2026-10-20 |
|---|---|---|
| Declarer_Direction accuracy | 0.6522 | 0.6516 |
| Contract alone / joint (318k sample) | 0.3826 / 0.3830 | 0.3823 / 0.3824 |
| Declarer alone / joint (318k sample) | 0.6529 / 0.6598 | 0.6511 / 0.6580 |
| Pct_NS board MAE (constant 0.5: 0.2554) | 0.2514 | 0.2516 |
| Pct_NS variance ratio | 0.164 | 0.145 |

Accuracy is unchanged within noise, so the dates were not carrying signal.

### Tournament rank (first measurement; 11,965 section/directions, 122,366 pairs)

| Ranker | Spearman | Pair session-% MAE |
|---|---|---|
| Pct_NS model | 0.375 | 0.0575 |
| Same, section-centered | 0.375 | 0.0573 |
| Same, board-centered | 0.376 | 0.0573 |
| Seat-Elo gap line | 0.338 | 0.0584 |
| Constant 50% | | 0.0636 |

The model beats Elo alone by +0.037 (club: +0.023 on the old club models).

### Caveats / follow-ups

Found while fixing the out-of-range Pct_NS rows (`drop_out_of_range_pct`,
2026-10-20). Both are left as is for now:

- **Pct_NS + Pct_EW != 1 on in-range rows**: 14,021 tournament rows (of 19.0M)
  and 38,857 club rows (of 71.8M) in the model-data shards. Each side is
  within [0,1]; the pattern (e.g. MP 4.24 / 3.52 on a top of 8) looks like
  ACBL adjusted or factored scores, which are real recorded results, so they
  are kept. Open question: whether adjusted scores (average-plus /
  average-minus) belong in the Pct_NS training target at all.
- **Club MP_Top can be understated**: club data has no ACBL top, so
  `add_board_matchpoint_top` in `mlBridge/mlBridgeAugmentLib.py` uses
  tables-in-frame - 1. When some of a board's rows are missing from our data,
  the top is too small and every Pct on that board is inflated, not just the
  rows that end up above 1.0 (which are now dropped). Not measured. Fixing it
  means deriving the club top from ACBL's matchpoints (e.g. the board's max
  MP_NS + MP_EW) instead of the row count, in ACBL code rather than the
  shared library, which FFBridge and postmortem also use.

---

## 2026-10-18 (evaluation only, no retrain: club rank and joint pick)

`acbl_prediction_rank_eval.py` on the current `_v2` test parquets and
SavedModels. Joint pick = argmax of log P(contract) + 0.3 x log P(declarer of
that contract's seat); weight 0.3 picked on a tune half, checked on a report
half (weight 1.0 cost 0.14 points of contract accuracy).

### Club rank (92,573 section/directions, pairs with >= 8 boards)

| Ranker | Spearman | Pair session-% MAE |
|---|---|---|
| Pct_NS model (mean of board predictions) | 0.456 | 0.0556 |
| Same, section-centered | 0.456 | 0.0544 |
| Same, board-centered | 0.456 | 0.0544 |
| Seat-Elo gap line (fit on 2025+) | 0.433 | |

Section centering is rank-neutral within a direction (ACBL matchpoints within
section and direction), so only the MAE moves. Tournament prediction data has
no session or section id, so it is not ranked. 5c now writes
`acbl_club_prediction_rank_metrics.json` after club Pct_NS.

### Joint contract/declarer pick (served by postmortem-acbl)

| Mode | Rows | Contract alone -> joint | Declarer alone -> joint | Models agree |
|---|---|---|---|---|
| club | 305k | 0.3787 -> 0.3787 | 0.6630 -> 0.6644 | 85.4% -> 100% |
| tournament | 318k | 0.3826 -> 0.3830 | 0.6529 -> 0.6598 | 83.6% -> 100% |

---

## 2026-07-10 → 2026-07-11 (full retrain, both modes, all 3 targets)

- Hardware: RTX 5080, 192 GB RAM, NVMe E:
- Input: `_v2` prediction parquets (test cutoff 2026-01-01)
  - club: 55,239,033 train / 4,331,680 test rows x 6,042 cols
    (train file salvaged 2026-07-08: one reboot-corrupted row group dropped,
    122,880 rows = 0.22% of May–Jun 2023)
  - tournament: 15,522,539 train / 419,448 test rows x 6,033 cols
- Wall clock: 30.4 h total (club 23.5 h, tournament 6.7 h); 20 epochs,
  early-stop patience 5. Per-step breakdown in `acbl_all.bat` 5c TIME tag.

### Club

| Target | Test metric | Baseline | Notes |
|---|---|---|---|
| Declarer_Direction | accuracy 0.6536 (macro F1 0.657) | majority 0.2615 | lift +0.39; best val epoch 18/20 |
| Contract | accuracy 0.4235 (macro F1 0.200) | majority 0.1513 | 36 effective classes; low macro F1 = rare-class imbalance, not weak model |
| Pct_NS | MAE 0.2593, variance ratio 0.168 | mean-prediction MAE ~0.25 | at hp-search ceiling (0.263 / 0.165); pruned to 1,347 features |

### Tournament

| Target | Test metric | Baseline | Notes |
|---|---|---|---|
| Declarer_Direction | accuracy 0.6422 (macro F1 0.642) | majority 0.2614 | lift +0.38 |
| Contract | accuracy 0.4364 (macro F1 0.195) | majority 0.1596 | schema has 413 classes, 36 in test |
| Pct_NS | MAE 0.2543, variance ratio 0.134 | mean-prediction MAE ~0.25 | early stop at epoch 15 (best epoch 10) |

### Assessment

- Both classifiers healthy: smooth val-loss curves, no overfit
  (club DD train/val acc 0.657/0.660), club and tournament agree within
  ~1 pt, predicted label distributions match real bridge frequencies
  (3N/4M dominate Contract; N/E/S/W balanced for DD).
- Feature importances sensible: Dealer + per-seat Elo dominate the
  classifiers; MasterPoints + Elo dominate Pct_NS.
- Pct_NS at its documented empirical ceiling on both modes.

### Caveats / follow-ups

- **Pct_NS target outliers**: club max 1.38, tournament max 9.99 — a
  matchpoint percentage should be <= 1.0. Upstream data-quality bug
  (9.99 looks like a sentinel). Clean before next retrain. FIXED 2026-10-20:
  `drop_out_of_range_pct` drops rows with Pct_NS or Pct_EW outside [0,1] in
  tournament cleaning (4,756 of 32.7M rows) and in club step 2 after
  augmentation; takes effect on the next run from stage 2 (tournament) or
  step 2 (club).
- **`session_id` / `event_id` in top-10 classifier importances**: raw IDs
  acting as time/venue proxies. FIXED 2026-07-12: added
  `^(event_id|session_id|group_id|session_number)$` to the default
  `blacklist_patterns` in `mlBridge/mlBridgeAiLib.py`; takes effect
  automatically on the next 5c training run (no 5a/5b rerun needed).
- Tournament Contract schema carries 413 classes but only 36 appear in
  test; collapsing ultra-rare doubled/redoubled variants would likely
  nudge accuracy up.
