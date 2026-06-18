import argparse
import os
import re
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import get, load_default_config, parse_config_path


class TeeStream:
    def __init__(self, stream, logfile):
        self._stream = stream
        self._logfile = logfile
        self._buffer = ''

    def write(self, data):
        self._stream.write(data)
        self._buffer += data
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._write_log_line(line)

    def flush(self):
        self._stream.flush()
        if self._buffer:
            if '\r' not in self._buffer:
                self._write_log_line(self._buffer)
            self._buffer = ''
        self._logfile.flush()

    def isatty(self):
        return getattr(self._stream, 'isatty', lambda: False)()

    def fileno(self):
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, 'encoding', 'utf-8')

    @property
    def errors(self):
        return getattr(self._stream, 'errors', 'strict')

    def _write_log_line(self, line):
        if not line:
            self._logfile.write('\n')
            return
        if '\r' in line:
            return
        self._logfile.write(line + '\n')


def install_tee(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    logfile = open(log_path, 'a', encoding='utf-8', buffering=1)
    sys.stdout = TeeStream(original_stdout, logfile)
    sys.stderr = TeeStream(original_stderr, logfile)
    return original_stdout, original_stderr, logfile


def row_minmax(scores):
    scores = scores.astype(np.float32, copy=False)
    min_v = scores.min(axis=1, keepdims=True)
    max_v = scores.max(axis=1, keepdims=True)
    denom = np.maximum(max_v - min_v, 1e-12)
    return (scores - min_v) / denom


def uncertainty_tiebreak(scores, helper_scores, weight=0.02, margin=0.05):
    """Small helper-score boost only for rows where the neural top choice is uncertain."""
    weight = float(weight)
    margin = float(margin)
    if weight <= 0:
        return scores

    neural = row_minmax(scores)
    helper = row_minmax(helper_scores)
    if neural.shape[1] < 2:
        return neural

    sorted_scores = np.sort(neural, axis=1)
    top_margin = sorted_scores[:, -1] - sorted_scores[:, -2]
    if margin <= 0:
        row_weight = np.full((neural.shape[0], 1), weight, dtype=np.float32)
    else:
        uncertainty = np.clip(1.0 - top_margin / margin, 0.0, 1.0)
        row_weight = (weight * uncertainty).astype(np.float32)[:, None]
    return row_minmax(neural + row_weight * helper)


class TemporalFeatureStore:
    """Strictly time-aware features for candidate link ranking."""

    FEATURE_DIM = 12

    def __init__(self, src, dst, t, decay_scale=0.0, max_co_items=20):
        self.pair_times = defaultdict(list)
        self.src_times = defaultdict(list)
        self.dst_times = defaultdict(list)
        self.src_dst_times = defaultdict(lambda: defaultdict(list))
        self.src_item_sets = defaultdict(set)
        self.dst_source_sets = defaultdict(set)
        self.max_co_items = int(max_co_items)

        for src_i, dst_i, time_i in zip(src, dst, t):
            src_i = int(src_i)
            dst_i = int(dst_i)
            time_i = int(time_i)
            self.pair_times[(src_i, dst_i)].append(time_i)
            self.src_times[src_i].append(time_i)
            self.dst_times[dst_i].append(time_i)
            self.src_dst_times[src_i][dst_i].append(time_i)
            self.src_item_sets[src_i].add(dst_i)
            self.dst_source_sets[dst_i].add(src_i)

        for history in self.pair_times.values():
            history.sort()
        for history in self.src_times.values():
            history.sort()
        for history in self.dst_times.values():
            history.sort()
        for dst_histories in self.src_dst_times.values():
            for history in dst_histories.values():
                history.sort()

        if len(t) > 1:
            time_span = max(int(np.max(t)) - int(np.min(t)), 1)
        else:
            time_span = 1
        self.decay_scale = float(decay_scale) if decay_scale > 0 else max(time_span / 50.0, 1.0)

    def _prefix(self, history, query_time):
        return bisect_left(history, int(query_time))

    def _recency_and_count(self, history, query_time):
        idx = self._prefix(history, query_time)
        if idx == 0:
            return 0.0, 0
        delta = max(int(query_time) - history[idx - 1], 0)
        return float(np.exp(-delta / self.decay_scale)), idx

    def source_history_items(self, src, query_time):
        items = []
        for dst_i, history in self.src_dst_times.get(int(src), {}).items():
            if self._prefix(history, query_time) > 0:
                items.append(int(dst_i))
        return items

    def source_recent_items(self, src, query_time, max_items=None):
        items = []
        for dst_i, history in self.src_dst_times.get(int(src), {}).items():
            idx = self._prefix(history, query_time)
            if idx > 0:
                items.append((history[idx - 1], int(dst_i)))
        items.sort(reverse=True)
        if max_items is not None and max_items > 0:
            items = items[:max_items]
        return [dst_i for _, dst_i in items]

    def candidate_matrix(self, src, t, candidates):
        features = np.zeros((*candidates.shape, self.FEATURE_DIM), dtype=np.float32)

        for row_idx, (src_i, time_i) in enumerate(zip(src, t)):
            src_i = int(src_i)
            time_i = int(time_i)
            recent_items = self.source_recent_items(src_i, time_i, self.max_co_items)
            recent_rank = {int(dst_i): rank for rank, dst_i in enumerate(recent_items)}
            src_history_len = len(recent_items)
            src_event_count = self._prefix(self.src_times.get(src_i, []), time_i)
            for col_idx, dst_i in enumerate(candidates[row_idx]):
                dst_i = int(dst_i)
                pair_recency, pair_count = self._recency_and_count(
                    self.pair_times.get((src_i, dst_i), []),
                    time_i,
                )
                dst_recency, dst_count = self._recency_and_count(
                    self.dst_times.get(dst_i, []),
                    time_i,
                )

                features[row_idx, col_idx, 0] = 1.0 if pair_count > 0 else 0.0
                features[row_idx, col_idx, 1] = pair_recency
                features[row_idx, col_idx, 2] = np.log1p(pair_count)
                features[row_idx, col_idx, 3] = dst_recency
                features[row_idx, col_idx, 4] = np.log1p(dst_count)

                cand_sources = self.dst_source_sets.get(dst_i, set())
                co_hits = 0
                co_strength = 0.0
                for hist_dst in recent_items:
                    hist_sources = self.dst_source_sets.get(hist_dst, set())
                    if not hist_sources or not cand_sources:
                        continue
                    common_sources = len(cand_sources & hist_sources)
                    if common_sources > 0:
                        co_hits += 1
                        co_strength += np.log1p(min(common_sources, 1000))

                denom = max(src_history_len, 1)
                features[row_idx, col_idx, 5] = np.log1p(src_history_len)
                features[row_idx, col_idx, 6] = np.log1p(len(cand_sources))
                features[row_idx, col_idx, 7] = co_hits / denom
                features[row_idx, col_idx, 8] = co_strength / denom
                if dst_i in recent_rank:
                    rank = recent_rank[dst_i]
                    features[row_idx, col_idx, 9] = 1.0 / float(rank + 1)
                    features[row_idx, col_idx, 10] = 1.0 if rank == 0 else 0.0
                features[row_idx, col_idx, 11] = pair_count / max(src_event_count, 1)

        return features

    def heuristic_score(self, src, t, candidates):
        features = self.candidate_matrix(src, t, candidates)
        scores = (
            2.0 * features[:, :, 0]
            + 4.0 * features[:, :, 1]
            + 1.5 * features[:, :, 2]
            + 0.8 * features[:, :, 3]
            + 0.35 * features[:, :, 4]
            + 2.5 * features[:, :, 9]
            + 1.0 * features[:, :, 10]
            + 2.0 * features[:, :, 11]
        )
        return row_minmax(scores)


class CandidateSampler:
    """Builds positive-first candidate lists for local MRR and hard negatives."""

    def __init__(self, src, dst, t, feature_store, seed=1):
        self.src = np.asarray(src, dtype=np.int32)
        self.dst = np.asarray(dst, dtype=np.int32)
        self.t = np.asarray(t, dtype=np.int64)
        self.feature_store = feature_store
        self.rng = np.random.default_rng(seed)
        self.dst_pool = np.unique(self.dst).astype(np.int32)
        self.sorted_order = np.argsort(self.t)
        self.sorted_t = self.t[self.sorted_order]
        self.sorted_dst = self.dst[self.sorted_order]

        if len(self.dst_pool) == 0:
            raise ValueError('CandidateSampler needs at least one destination node')

    def sample(self, src, positive_dst, t, num_candidates=100):
        if num_candidates < 2:
            raise ValueError('num_candidates must be at least 2')
        if len(self.dst_pool) < num_candidates:
            raise ValueError(f'dst_pool has {len(self.dst_pool)} items, needs at least {num_candidates}')

        src = np.asarray(src, dtype=np.int32)
        positive_dst = np.asarray(positive_dst, dtype=np.int32)
        t = np.asarray(t, dtype=np.int64)
        candidates = np.empty((len(src), num_candidates), dtype=np.int32)
        candidates[:, 0] = positive_dst

        for row_idx, (src_i, pos_i, time_i) in enumerate(zip(src, positive_dst, t)):
            selected = {int(pos_i)}
            negatives = []

            self._add_source_history(negatives, selected, src_i, time_i, num_candidates)
            self._add_recent_popular(negatives, selected, time_i, num_candidates)
            self._add_random(negatives, selected, num_candidates)

            candidates[row_idx, 1:] = np.asarray(negatives[:num_candidates - 1], dtype=np.int32)

        return candidates

    def _add_source_history(self, negatives, selected, src_i, time_i, num_candidates):
        items = self.feature_store.source_history_items(src_i, time_i)
        if items:
            self.rng.shuffle(items)
        for dst_i in items:
            self._maybe_add(negatives, selected, dst_i, num_candidates)

    def _add_recent_popular(self, negatives, selected, time_i, num_candidates):
        end = np.searchsorted(self.sorted_t, int(time_i), side='left')
        start = max(0, end - 20 * num_candidates)
        recent_dst = self.sorted_dst[start:end][::-1]
        for dst_i in recent_dst:
            self._maybe_add(negatives, selected, dst_i, num_candidates)

    def _add_random(self, negatives, selected, num_candidates):
        while len(negatives) < num_candidates - 1:
            dst_i = int(self.rng.choice(self.dst_pool))
            self._maybe_add(negatives, selected, dst_i, num_candidates)

    @staticmethod
    def _maybe_add(negatives, selected, dst_i, num_candidates):
        if len(negatives) >= num_candidates - 1:
            return
        dst_i = int(dst_i)
        if dst_i in selected:
            return
        selected.add(dst_i)
        negatives.append(dst_i)


def reciprocal_rank(scores, positive_col=0):
    positive_scores = scores[:, positive_col:positive_col + 1]
    greater = (scores > positive_scores).sum(axis=1)
    ties = (scores == positive_scores).sum(axis=1)
    ranks = 1.0 + greater + 0.5 * (ties - 1)
    return 1.0 / ranks


def mean_reciprocal_rank(scores, positive_col=0):
    return float(np.mean(reciprocal_rank(scores, positive_col)))


def candidate_ranking_metrics(scores, positive_col=0):
    scores = row_minmax(scores)
    y_true = np.zeros_like(scores, dtype=np.int32)
    y_true[:, positive_col] = 1
    ap100 = float(average_precision_score(y_true.ravel(), scores.ravel()))
    auc100 = float(roc_auc_score(y_true.ravel(), scores.ravel()))
    return {
        'MRR': mean_reciprocal_rank(scores, positive_col),
        'AP100': ap100,
        'AUC100': auc100,
        'AP': ap100,
        'AUC': auc100,
    }


def tune_blend_alpha(neural_scores, heuristic_scores):
    best_alpha = 1.0
    best_mrr = -1.0
    for alpha in np.linspace(0.0, 1.0, 21):
        blended = alpha * neural_scores + (1.0 - alpha) * heuristic_scores
        mrr = mean_reciprocal_rank(blended)
        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = float(alpha)
    return best_alpha, best_mrr


def build_local_validation(src, dst, t, train_end, num_candidates=100, seed=1, max_rows=None):
    val_src = src[train_end:]
    val_dst = dst[train_end:]
    val_t = t[train_end:]

    if max_rows is not None and max_rows > 0 and max_rows < len(val_src):
        idx = np.linspace(0, len(val_src) - 1, max_rows, dtype=np.int64)
        val_src = val_src[idx]
        val_dst = val_dst[idx]
        val_t = val_t[idx]

    feature_store = TemporalFeatureStore(src[:train_end], dst[:train_end], t[:train_end])
    sampler = CandidateSampler(src[:train_end], dst[:train_end], t[:train_end], feature_store, seed=seed)
    candidates = sampler.sample(val_src, val_dst, val_t, num_candidates=num_candidates)
    return val_src, val_dst, val_t, candidates, feature_store


def check_dataset(data_dir: str, dataset: str) -> None:
    train_path = os.path.join(data_dir, dataset, 'train.csv')
    test_path = os.path.join(data_dir, dataset, 'test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Missing train file: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'Missing test file: {test_path}')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for col in ['src', 'dst', 'time']:
        if col not in train_df.columns:
            raise ValueError(f'{dataset} train.csv missing column: {col}')
    for col in ['src', 'time']:
        if col not in test_df.columns:
            raise ValueError(f'{dataset} test.csv missing column: {col}')

    cand_cols = test_df.columns[2:]
    if len(cand_cols) != 100:
        raise ValueError(f'{dataset} test.csv candidate columns != 100, got {len(cand_cols)}')

    if train_df[['src', 'dst', 'time']].isnull().any().any():
        raise ValueError(f'{dataset} train.csv has null values in required columns')
    if test_df[['src', 'time']].isnull().any().any() or test_df[cand_cols].isnull().any().any():
        raise ValueError(f'{dataset} test.csv has null values')

    train_rows = len(train_df)
    dup_ratio = 1.0 - (len(train_df[['src', 'dst', 'time']].drop_duplicates()) / max(1, train_rows))

    print(f'[{dataset}]')
    print(f'  train rows: {train_rows}')
    print(f'  test rows: {len(test_df)}')
    print(f'  unique src: {train_df["src"].nunique()}')
    print(f'  unique dst: {train_df["dst"].nunique()}')
    print(f'  train time range: [{int(train_df["time"].min())}, {int(train_df["time"].max())}]')
    print(f'  duplicate ratio (src,dst,time): {dup_ratio:.4f}')
    print('  status: OK')


DECIMAL_8 = re.compile(r'^\d+\.\d{8}$')


def check_one_submission_file(pred_path: str, test_path: str, dataset: str) -> None:
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f'Missing prediction file: {pred_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'Missing test file: {test_path}')

    test_rows = len(pd.read_csv(test_path))

    with open(pred_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]

    if len(lines) != test_rows:
        raise ValueError(f'{dataset}: line count mismatch, pred={len(lines)} test={test_rows}')

    for i, line in enumerate(lines, start=1):
        parts = line.split(',')
        if len(parts) != 100:
            raise ValueError(f'{dataset}: line {i} has {len(parts)} columns, expected 100')
        for j, p in enumerate(parts, start=1):
            p = p.strip()
            try:
                v = float(p)
            except ValueError as exc:
                raise ValueError(f'{dataset}: line {i} col {j} is not float: {p}') from exc
            if v < 0.0 or v > 1.0:
                raise ValueError(f'{dataset}: line {i} col {j} out of [0,1]: {v}')
            if not DECIMAL_8.match(p):
                raise ValueError(f'{dataset}: line {i} col {j} not fixed 8 decimals: {p}')

    print(f'[{dataset}] submission file OK: {pred_path}')


def check_data_main(argv=None) -> None:
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--data_dir', type=str, default=get(config, 'paths', 'data_dir', './data_A'))
    parser.add_argument('--datasets', type=str, nargs='+', default=config.get('datasets', ['dataset1', 'dataset2']))
    parser.add_argument('--log_dir', type=str, default=get(config, 'paths', 'log_dir', './outputs/logs'))
    args = parser.parse_args(argv)

    log_name = f'check_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    install_tee(os.path.join(args.log_dir, 'check_data', log_name))

    for ds in args.datasets:
        check_dataset(args.data_dir, ds)


def check_submission_main(argv=None) -> None:
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--data_dir', type=str, default=get(config, 'paths', 'data_dir', './data_A'))
    parser.add_argument('--submission_dir', type=str, default=get(config, 'paths', 'submission_dir', './outputs/submission'))
    parser.add_argument('--datasets', type=str, nargs='+', default=config.get('datasets', ['dataset1', 'dataset2']))
    parser.add_argument('--log_dir', type=str, default=get(config, 'paths', 'log_dir', './outputs/logs'))
    args = parser.parse_args(argv)

    log_name = f'check_submission_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    install_tee(os.path.join(args.log_dir, 'check_submission', log_name))

    for ds in args.datasets:
        pred_file = os.path.join(args.submission_dir, f'{ds}.csv')
        test_file = os.path.join(args.data_dir, ds, 'test.csv')
        check_one_submission_file(pred_file, test_file, ds)


def local_mrr_main(argv=None):
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    local = config.get('local_mrr', {})
    common = config.get('common', {})
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=get(config, 'paths', 'data_dir', './data_A'))
    parser.add_argument('--val_ratio', type=float, default=local.get('val_ratio', 0.15))
    parser.add_argument('--num_candidates', type=int, default=local.get('num_candidates', 100))
    parser.add_argument('--max_rows', type=int, default=local.get('max_rows', 5000))
    parser.add_argument('--seed', type=int, default=common.get('seed', 1))
    args = parser.parse_args(argv)

    df = pd.read_csv(os.path.join(args.data_dir, args.dataset, 'train.csv'))
    src = df['src'].values.astype(np.int32)
    dst = df['dst'].values.astype(np.int32)
    t = df['time'].values.astype(np.int64)
    train_end = len(df) - int(len(df) * args.val_ratio)

    val_src, _, val_t, candidates, feature_store = build_local_validation(
        src,
        dst,
        t,
        train_end,
        num_candidates=args.num_candidates,
        seed=args.seed,
        max_rows=args.max_rows,
    )
    heuristic_scores = feature_store.heuristic_score(val_src, val_t, candidates)
    mrr = mean_reciprocal_rank(heuristic_scores)

    print(f'{args.dataset} local MRR@{args.num_candidates}: {mrr:.6f}')
    print(f'rows={len(val_src)}, train_end={train_end}, seed={args.seed}')
