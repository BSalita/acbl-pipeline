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
  (9.99 looks like a sentinel). Clean before next retrain.
- **`session_id` / `event_id` in top-10 classifier importances**: raw IDs
  acting as time/venue proxies. FIXED 2026-07-12: added
  `^(event_id|session_id|group_id|session_number)$` to the default
  `blacklist_patterns` in `mlBridge/mlBridgeAiLib.py`; takes effect
  automatically on the next 5c training run (no 5a/5b rerun needed).
- Tournament Contract schema carries 413 classes but only 36 appear in
  test; collapsing ultra-rare doubled/redoubled variants would likely
  nudge accuracy up.
