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


class TemporalFeatureStore:
    """Strictly left-looking temporal and pair features for candidate ranking."""

    FEATURE_NAMES = (
        'pair_seen',
        'pair_recency',
        'pair_count_log',
        'dst_recency',
        'dst_count_log',
        'src_history_items_log',
        'dst_sources_log',
        'co_item_ratio',
        'co_item_strength',
        'pair_recent_rank',
        'pair_is_last_item',
        'pair_source_share',
        'pair_log_gap',
        'dst_log_gap',
        'pair_recent_frequency',
        'dst_recent_frequency_log',
        'pair_gap_stability',
        'pair_mean_gap_log',
        'temporal_co_strength',
    )
    FEATURE_DIM = len(FEATURE_NAMES)

    def __init__(self, src, dst, t, decay_scale=0.0, max_co_items=20):
        self.pair_times = defaultdict(list)
        self.src_times = defaultdict(list)
        self.dst_times = defaultdict(list)
        self.src_dst_times = defaultdict(lambda: defaultdict(list))
        self.src_item_sets = defaultdict(set)
        self.dst_source_sets = defaultdict(set)
        self.max_co_items = int(max_co_items)
        self._co_strength_cache = {}
        self._co_strength_cache_limit = 250000

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
        self.max_time = int(np.max(t)) if len(t) else -1
        self.time_span = float(time_span)
        self.log_time_span = float(np.log1p(time_span))
        self.decay_scale = float(decay_scale) if decay_scale > 0 else max(time_span / 50.0, 1.0)

    def _prefix(self, history, query_time):
        return bisect_left(history, int(query_time))

    def _recency_and_count(self, history, query_time):
        idx = self._prefix(history, query_time)
        if idx == 0:
            return 0.0, 0
        delta = max(int(query_time) - history[idx - 1], 0)
        return float(np.exp(-delta / self.decay_scale)), idx

    def _history_features(self, history, query_time, include_intervals=True):
        """Return bounded TAMI-style gap features using events strictly before query_time."""
        idx = self._prefix(history, query_time)
        if idx == 0:
            return 0.0, 0, 0.0, 0.0, 0.0, 0.0

        query_time = int(query_time)
        delta = max(query_time - history[idx - 1], 0)
        recency = float(np.exp(-delta / self.decay_scale))
        log_gap = float(np.log1p(delta) / max(self.log_time_span, 1e-12))
        window_start = query_time - self.decay_scale
        recent_start = bisect_left(history, window_start, 0, idx)
        recent_frequency = float(idx - recent_start) / max(idx, 1)

        stability = 0.0
        mean_gap_log = 0.0
        if include_intervals and idx >= 2:
            recent_history = np.asarray(history[max(0, idx - 6):idx], dtype=np.float64)
            gaps = np.diff(recent_history)
            mean_gap = float(np.mean(gaps))
            gap_cv = float(np.std(gaps) / max(mean_gap, 1.0))
            stability = 1.0 / (1.0 + gap_cv)
            mean_gap_log = float(np.log1p(mean_gap) / max(self.log_time_span, 1e-12))

        return recency, idx, log_gap, recent_frequency, stability, mean_gap_log

    def source_history_items(self, src, query_time):
        items = []
        for dst_i, history in self.src_dst_times.get(int(src), {}).items():
            if self._prefix(history, query_time) > 0:
                items.append(int(dst_i))
        return items

    def source_recent_items(self, src, query_time, max_items=None):
        return [dst_i for _, dst_i in self.source_recent_history(src, query_time, max_items)]

    def source_recent_history(self, src, query_time, max_items=None):
        items = []
        for dst_i, history in self.src_dst_times.get(int(src), {}).items():
            idx = self._prefix(history, query_time)
            if idx > 0:
                items.append((history[idx - 1], int(dst_i)))
        items.sort(reverse=True)
        if max_items is not None and max_items > 0:
            items = items[:max_items]
        return items

    def _active_sources(self, dst, query_time):
        sources = self.dst_source_sets.get(int(dst), set())
        if int(query_time) > self.max_time:
            return sources
        return {
            src_i
            for src_i in sources
            if self._prefix(self.src_dst_times[src_i][int(dst)], query_time) > 0
        }

    def _common_source_strength(self, left_dst, right_dst, query_time):
        left_dst = int(left_dst)
        right_dst = int(right_dst)
        key = (left_dst, right_dst) if left_dst <= right_dst else (right_dst, left_dst)
        fully_observed = int(query_time) > self.max_time
        if fully_observed and key in self._co_strength_cache:
            return self._co_strength_cache[key]

        left_sources = self._active_sources(left_dst, query_time)
        right_sources = self._active_sources(right_dst, query_time)
        if not left_sources or not right_sources:
            value = 0.0
        else:
            common_sources = len(left_sources & right_sources)
            value = float(np.log1p(min(common_sources, 1000))) if common_sources else 0.0

        if fully_observed:
            if len(self._co_strength_cache) >= self._co_strength_cache_limit:
                self._co_strength_cache.clear()
            self._co_strength_cache[key] = value
        return value

    def candidate_matrix(self, src, t, candidates, feature_set='enhanced'):
        feature_set = str(feature_set).strip().lower()
        if feature_set not in {'basic', 'enhanced'}:
            raise ValueError(f'Unknown temporal feature set: {feature_set}')
        enhanced = feature_set == 'enhanced'
        feature_dim = self.FEATURE_DIM if enhanced else 12
        features = np.zeros((*candidates.shape, feature_dim), dtype=np.float32)

        for row_idx, (src_i, time_i) in enumerate(zip(src, t)):
            src_i = int(src_i)
            time_i = int(time_i)
            recent_history = self.source_recent_history(src_i, time_i, self.max_co_items)
            recent_items = [dst_i for _, dst_i in recent_history]
            recent_rank = {int(dst_i): rank for rank, dst_i in enumerate(recent_items)}
            recent_item_weights = {
                int(dst_i): float(np.exp(-max(time_i - int(last_time), 0) / self.decay_scale))
                for last_time, dst_i in recent_history
            }
            src_history_len = len(recent_items)
            src_event_count = self._prefix(self.src_times.get(src_i, []), time_i)
            for col_idx, dst_i in enumerate(candidates[row_idx]):
                dst_i = int(dst_i)
                pair_history = self.pair_times.get((src_i, dst_i), [])
                dst_history = self.dst_times.get(dst_i, [])
                if enhanced:
                    pair_recency, pair_count, pair_log_gap, pair_recent_frequency, pair_gap_stability, pair_mean_gap_log = self._history_features(
                        pair_history,
                        time_i,
                    )
                    dst_recency, dst_count, dst_log_gap, dst_recent_frequency, _, _ = self._history_features(
                        dst_history,
                        time_i,
                        include_intervals=False,
                    )
                else:
                    pair_recency, pair_count = self._recency_and_count(pair_history, time_i)
                    dst_recency, dst_count = self._recency_and_count(dst_history, time_i)

                features[row_idx, col_idx, 0] = 1.0 if pair_count > 0 else 0.0
                features[row_idx, col_idx, 1] = pair_recency
                features[row_idx, col_idx, 2] = np.log1p(pair_count)
                features[row_idx, col_idx, 3] = dst_recency
                features[row_idx, col_idx, 4] = np.log1p(dst_count)

                co_hits = 0
                co_strength = 0.0
                temporal_co_strength = 0.0
                for hist_dst in recent_items:
                    common_strength = self._common_source_strength(dst_i, hist_dst, time_i)
                    if common_strength > 0:
                        co_hits += 1
                        co_strength += common_strength
                        if enhanced:
                            temporal_co_strength += recent_item_weights[hist_dst] * common_strength

                denom = max(src_history_len, 1)
                features[row_idx, col_idx, 5] = np.log1p(src_history_len)
                features[row_idx, col_idx, 6] = np.log1p(len(self._active_sources(dst_i, time_i)))
                features[row_idx, col_idx, 7] = co_hits / denom
                features[row_idx, col_idx, 8] = co_strength / denom
                if dst_i in recent_rank:
                    rank = recent_rank[dst_i]
                    features[row_idx, col_idx, 9] = 1.0 / float(rank + 1)
                    features[row_idx, col_idx, 10] = 1.0 if rank == 0 else 0.0
                features[row_idx, col_idx, 11] = pair_count / max(src_event_count, 1)
                if enhanced:
                    features[row_idx, col_idx, 12] = pair_log_gap
                    features[row_idx, col_idx, 13] = dst_log_gap
                    features[row_idx, col_idx, 14] = pair_recent_frequency
                    features[row_idx, col_idx, 15] = np.log1p(dst_count * dst_recent_frequency)
                    features[row_idx, col_idx, 16] = pair_gap_stability
                    features[row_idx, col_idx, 17] = pair_mean_gap_log
                    features[row_idx, col_idx, 18] = temporal_co_strength / denom

        return features

    def heuristic_score(self, src, t, candidates):
        features = self.candidate_matrix(src, t, candidates, feature_set='basic')
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
    """Builds positive-first candidate lists with random negatives."""

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

        for row_idx, pos_i in enumerate(positive_dst):
            selected = {int(pos_i)}
            negatives = []

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
    rr = reciprocal_rank(scores, positive_col)
    ranks = 1.0 / rr
    y_true = np.zeros_like(scores, dtype=np.int32)
    y_true[:, positive_col] = 1
    ap100 = float(average_precision_score(y_true.ravel(), scores.ravel()))
    auc100 = float(roc_auc_score(y_true.ravel(), scores.ravel()))
    return {
        'MRR': float(np.mean(rr)),
        'Hits@1': float(np.mean(ranks <= 1.0)),
        'Hits@3': float(np.mean(ranks <= 3.0)),
        'Hits@10': float(np.mean(ranks <= 10.0)),
        'MedianRank': float(np.median(ranks)),
        'AP100': ap100,
        'AUC100': auc100,
        'AP': ap100,
        'AUC': auc100,
    }


def candidate_slice_metrics(scores, feature_store, src, t, candidates, positive_col=0):
    """MRR diagnostics for the repeat/new, gap and popularity slices in develop.md."""
    scores = np.asarray(scores)
    positive_candidates = candidates[:, positive_col:positive_col + 1]
    positive_features = feature_store.candidate_matrix(src, t, positive_candidates)[:, 0, :]
    rr = reciprocal_rank(scores, positive_col)
    seen = positive_features[:, 0] > 0.5
    slices = {
        'overall': np.ones(len(rr), dtype=bool),
        'repeat': seen,
        'novel': ~seen,
    }

    if seen.any():
        seen_gap = positive_features[seen, 12]
        gap_split = float(np.median(seen_gap))
        slices['short_gap'] = seen & (positive_features[:, 12] <= gap_split)
        slices['long_gap'] = seen & (positive_features[:, 12] > gap_split)

    popularity = positive_features[:, 4]
    popularity_split = float(np.median(popularity))
    slices['head'] = popularity > popularity_split
    slices['tail'] = popularity <= popularity_split

    result = {}
    for name, mask in slices.items():
        count = int(mask.sum())
        result[name] = {
            'count': count,
            'MRR': float(np.mean(rr[mask])) if count else float('nan'),
        }
    return result


def format_slice_metrics(prefix, metrics):
    parts = []
    for name in ('overall', 'repeat', 'novel', 'short_gap', 'long_gap', 'head', 'tail'):
        if name not in metrics:
            continue
        value = metrics[name]
        parts.append(f'{name}={value["MRR"]:.6f}(n={value["count"]})')
    return f'{prefix}: ' + ', '.join(parts)


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
    pair_repeat_ratio = float(train_df[['src', 'dst']].duplicated().mean())
    src_nodes = set(train_df['src'].unique())
    dst_nodes = set(train_df['dst'].unique())
    type_overlap = len(src_nodes & dst_nodes)
    type_overlap_ratio = type_overlap / max(min(len(src_nodes), len(dst_nodes)), 1)
    graph_type = 'bipartite' if type_overlap == 0 else 'homogeneous/overlapping'

    print(f'[{dataset}]')
    print(f'  train rows: {train_rows}')
    print(f'  test rows: {len(test_df)}')
    print(f'  unique src: {train_df["src"].nunique()}')
    print(f'  unique dst: {train_df["dst"].nunique()}')
    print(f'  train time range: [{int(train_df["time"].min())}, {int(train_df["time"].max())}]')
    print(f'  duplicate ratio (src,dst,time): {dup_ratio:.4f}')
    print(f'  repeat interaction ratio (src,dst): {pair_repeat_ratio:.4f}')
    print(f'  graph type: {graph_type} (src/dst overlap={type_overlap_ratio:.4f})')
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
    args = parser.parse_args(argv)

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
