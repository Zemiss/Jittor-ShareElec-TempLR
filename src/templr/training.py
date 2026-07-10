import argparse
import os
import os.path as osp
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (
    PROJECT_ROOT,
    as_bool,
    get,
    get_model_config,
    get_run_model_name,
    load_default_config,
    parse_config_path,
)
from .baseline import test_baseline_competition, train_baseline
from .core import (
    CandidateSampler,
    TemporalFeatureStore,
    candidate_ranking_metrics,
    candidate_slice_metrics,
    format_slice_metrics,
    install_tee,
    mean_reciprocal_rank,
    row_minmax,
    tune_blend_alpha,
)
from .models import build_model
from .runtime import (
    configure_jittor_backend,
    seed_everything,
    test_competition,
    train_model,
)


def count_parameters(model):
    return int(sum(np.prod(p.shape) for p in model.parameters()))


def display_path(path):
    try:
        return osp.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


def print_model_banner(args, model):
    return


def is_baseline_family(model_name):
    return model_name.lower() == 'baseline'


def uses_bpr_objective(model_name, objective):
    return is_baseline_family(model_name) or str(objective).lower() == 'bpr'


def normalize_selection_metric(value):
    return ''.join(ch for ch in str(value).upper() if ch.isalnum())


def resolve_mynet_rerank(args):
    return bool(getattr(args, 'use_rerank', False))


def normalize_rerank_feature_set(value):
    feature_set = str(value).strip().lower()
    if feature_set not in {'basic', 'enhanced'}:
        raise ValueError(f'Unknown rerank feature set: {value}')
    return feature_set


def parse_rerank_dual_head(value):
    if value is None or str(value).strip().lower() == 'auto':
        return None
    return as_bool(value)


def resolve_rerank_dual_head(dataset, value):
    if value is not None:
        return bool(value)
    return str(dataset).strip().lower() != 'dataset2'


def learned_rerank_feature_names(dual_head=True, feature_set='enhanced'):
    feature_set = normalize_rerank_feature_set(feature_set)
    temporal_names = TemporalFeatureStore.FEATURE_NAMES
    if feature_set == 'basic':
        temporal_names = temporal_names[:12]
    base_names = ('neural_score',) + temporal_names
    if not dual_head:
        return base_names
    return (
        tuple(f'repeat:{name}' for name in base_names)
        + tuple(f'novel:{name}' for name in base_names)
        + ('repeat_gate',)
    )


def learned_rerank_matrix(
    feature_store,
    src,
    t,
    candidates,
    neural_scores,
    dual_head=True,
    feature_set='enhanced',
):
    feature_set = normalize_rerank_feature_set(feature_set)
    features = feature_store.candidate_matrix(src, t, candidates, feature_set=feature_set)
    neural_norm = row_minmax(neural_scores)
    num_rows, num_candidates = neural_norm.shape
    flat_features = features.reshape(num_rows * num_candidates, features.shape[-1])
    base = np.concatenate([neural_norm.reshape(-1, 1), flat_features], axis=1)
    if not dual_head:
        return base

    repeat_gate = flat_features[:, 0:1]
    novel_gate = 1.0 - repeat_gate
    return np.concatenate(
        [base * repeat_gate, base * novel_gate, repeat_gate],
        axis=1,
    )


def train_learned_reranker(
    feature_store,
    src,
    t,
    candidates,
    neural_scores,
    dual_head=True,
    regularization=1.0,
    feature_set='enhanced',
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = learned_rerank_matrix(
        feature_store,
        src,
        t,
        candidates,
        neural_scores,
        dual_head=dual_head,
        feature_set=feature_set,
    )
    num_candidates = candidates.shape[1]
    row_x = x.reshape(len(src), num_candidates, x.shape[-1])
    pos_x = row_x[:, :1, :]
    neg_x = row_x[:, 1:, :]
    pos_minus_neg = (pos_x - neg_x).reshape(-1, x.shape[-1])
    neg_minus_pos = -pos_minus_neg
    x_pair = np.vstack([pos_minus_neg, neg_minus_pos])
    y = np.concatenate(
        [
            np.ones(len(pos_minus_neg), dtype=np.int32),
            np.zeros(len(neg_minus_pos), dtype=np.int32),
        ]
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=max(float(regularization), 1e-6),
            fit_intercept=False,
            max_iter=300,
            solver='lbfgs',
            random_state=0,
        ),
    )
    model.fit(x_pair, y)
    model.rerank_alpha = 0.0
    model.rerank_margin = 0.0
    model.rerank_dual_head = bool(dual_head)
    model.rerank_feature_set = normalize_rerank_feature_set(feature_set)
    model.feature_names_ = learned_rerank_feature_names(dual_head, feature_set)
    return model


def predict_learned_rerank_scores(model, feature_store, src, t, candidates, neural_scores, batch_rows=10000):
    scores = np.empty_like(neural_scores, dtype=np.float32)
    for start in range(0, len(src), int(batch_rows)):
        end = min(start + int(batch_rows), len(src))
        x = learned_rerank_matrix(
            feature_store,
            src[start:end],
            t[start:end],
            candidates[start:end],
            neural_scores[start:end],
            dual_head=bool(getattr(model, 'rerank_dual_head', False)),
            feature_set=getattr(model, 'rerank_feature_set', 'basic'),
        )
        if hasattr(model, 'decision_function'):
            chunk_scores = model.decision_function(x).reshape(end - start, candidates.shape[1])
        else:
            chunk_scores = model.predict_proba(x)[:, 1].reshape(end - start, candidates.shape[1])
        scores[start:end] = chunk_scores.astype(np.float32)
    return row_minmax(scores)


def apply_learned_rerank_on_uncertain_rows(neural_scores, learned_scores, margin, alpha=0.0, coverage=None):
    neural_norm = row_minmax(neural_scores)
    learned_norm = row_minmax(learned_scores)
    margin = float(margin)
    alpha = min(max(float(alpha), 0.0), 1.0)
    if margin < 0:
        scores = alpha * neural_norm + (1.0 - alpha) * learned_norm
        return scores, np.ones(neural_norm.shape[0], dtype=bool)
    if neural_norm.shape[1] < 2:
        mask = np.ones(neural_norm.shape[0], dtype=bool)
    else:
        sorted_scores = np.sort(neural_norm, axis=1)
        top_margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        if coverage is None:
            mask = top_margin < margin
        else:
            keep_rows = int(round(float(coverage) * neural_norm.shape[0]))
            keep_rows = min(max(keep_rows, 1), neural_norm.shape[0])
            selected = np.argpartition(top_margin, keep_rows - 1)[:keep_rows]
            mask = np.zeros(neural_norm.shape[0], dtype=bool)
            mask[selected] = True
    scores = neural_norm.copy()
    scores[mask] = alpha * neural_norm[mask] + (1.0 - alpha) * learned_norm[mask]
    return scores, mask


def tune_learned_rerank_policy(neural_scores, learned_scores):
    neural_norm = row_minmax(neural_scores)
    learned_norm = row_minmax(learned_scores)
    best = {
        'mrr': mean_reciprocal_rank(neural_norm),
        'alpha': 1.0,
        'margin': -1.0,
        'coverage': 1.0,
        'scores': neural_norm,
        'mask': np.ones(neural_norm.shape[0], dtype=bool),
    }
    margins = [-1.0, 0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    alphas = np.linspace(0.0, 1.0, 11)
    for margin in margins:
        for alpha in alphas:
            scores, mask = apply_learned_rerank_on_uncertain_rows(
                neural_norm,
                learned_norm,
                margin,
                alpha=alpha,
            )
            if not mask.any():
                continue
            mrr = mean_reciprocal_rank(scores)
            if mrr > best['mrr']:
                best = {
                    'mrr': mrr,
                    'alpha': float(alpha),
                    'margin': float(margin),
                    'coverage': float(mask.mean()),
                    'scores': scores,
                    'mask': mask,
                }
    return best


def split_rerank_validation(num_rows, train_ratio=0.8):
    num_rows = int(num_rows)
    if num_rows < 2:
        raise ValueError('Rerank validation needs at least two rows for train/holdout split')
    split = int(num_rows * float(train_ratio))
    split = min(max(split, 1), num_rows - 1)
    return slice(0, split), slice(split, num_rows)


def format_rerank_summary(prefix, before_metrics, after_metrics, mask, alpha, margin, coverage=None):
    mrr_before = before_metrics['MRR']
    mrr_after = after_metrics['MRR']
    ap_before = before_metrics['AP100']
    ap_after = after_metrics['AP100']
    hits1_before = before_metrics['Hits@1']
    hits1_after = after_metrics['Hits@1']
    hits3_before = before_metrics['Hits@3']
    hits3_after = after_metrics['Hits@3']
    hits10_before = before_metrics['Hits@10']
    hits10_after = after_metrics['Hits@10']
    median_before = before_metrics['MedianRank']
    median_after = after_metrics['MedianRank']
    delta = mrr_after - mrr_before
    rel = 100.0 * delta / max(abs(mrr_before), 1e-12)
    applied = int(mask.sum())
    total = len(mask)
    coverage_text = f', coverage={100.0 * float(coverage):.1f}%' if coverage is not None else ''
    return (
        f'{prefix}: MRR {mrr_before:.6f} -> {mrr_after:.6f} '
        f'({delta:+.6f}, {rel:+.2f}%), '
        f'Hits@1 {hits1_before:.6f} -> {hits1_after:.6f}, '
        f'Hits@3 {hits3_before:.6f} -> {hits3_after:.6f}, '
        f'Hits@10 {hits10_before:.6f} -> {hits10_after:.6f}, '
        f'MedianRank {median_before:.1f} -> {median_after:.1f}, '
        f'AP100 {ap_before:.6f} -> {ap_after:.6f}; '
        f'applied {applied}/{total} ({100.0 * applied / max(total, 1):.1f}%), '
        f'alpha={alpha:.2f}, margin={margin:.4f}{coverage_text}'
    )


def format_reranker_weights(model, top_k=10):
    names = tuple(getattr(model, 'feature_names_', ()))
    try:
        coefficients = model.named_steps['logisticregression'].coef_[0]
    except (AttributeError, KeyError):
        return 'Rerank feature weights: unavailable'
    if len(names) != len(coefficients):
        names = tuple(f'feature_{idx}' for idx in range(len(coefficients)))
    order = np.argsort(np.abs(coefficients))[::-1][:int(top_k)]
    summary = ', '.join(f'{names[idx]}={coefficients[idx]:+.3f}' for idx in order)
    return f'Rerank top feature weights: {summary}'


def reranker_path(save_dir, dataset, model_tag):
    return osp.join(save_dir, 'reranker', f'{dataset}_{model_tag}_reranker.pkl')


def save_learned_reranker(model, path):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_learned_reranker(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def assert_left_history_only(full_neighbor_sampler, node_ids, query_times, num_neighbors, label, max_rows=256):
    if len(node_ids) == 0:
        return
    sample_size = min(len(node_ids), max_rows)
    sample_idx = np.linspace(0, len(node_ids) - 1, sample_size, dtype=np.int64)
    sampled_nodes = np.asarray(node_ids, dtype=np.int32)[sample_idx]
    sampled_times = np.asarray(query_times, dtype=np.int64)[sample_idx]
    neighbors, _, history_times = full_neighbor_sampler.get_historical_neighbors_left(
        node_ids=sampled_nodes,
        node_interact_times=sampled_times,
        num_neighbors=num_neighbors,
    )
    neighbors = np.asarray(neighbors)
    history_times = np.asarray(history_times)
    valid = neighbors != 0
    leaked = valid & (history_times >= sampled_times[:, None])
    if leaked.any():
        row, col = np.argwhere(leaked)[0]
        raise ValueError(
            f'Temporal leakage detected in {label}: '
            f'node={int(sampled_nodes[row])}, query_time={int(sampled_times[row])}, '
            f'history_time={int(history_times[row, col])}'
        )
    return


def resolve_model_defaults(config, argv=None):
    default_model_name = get_run_model_name(config)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=default_model_name, choices=['baseline', 'mynet'])
    args, _ = parser.parse_known_args(argv)
    model_name = args.model_name.lower()
    return model_name, get_model_config(config, model_name)


def build_parser(config, argv=None):
    train_cfg = config.get('train', {})
    model_name, model_cfg = resolve_model_defaults(config, argv)
    common = config.get('common', {})
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=parse_config_path())
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_dir', type=str, default=get(config, 'paths', 'data_dir', './data_A'), help='Data directory')
    parser.add_argument(
        '--save_model_dir',
        '--save_dir',
        dest='save_model_dir',
        type=str,
        default=get(config, 'paths', 'save_model_dir', get(config, 'paths', 'model_dir', './models')),
        help='Model save directory',
    )
    parser.add_argument('--output_dir', type=str, default=get(config, 'paths', 'output_dir', './outputs'), help='Output directory')
    parser.add_argument(
        '--submission_dir',
        type=str,
        default=get(config, 'paths', 'submission_dir', './outputs/submission'),
        help='Submission output directory',
    )
    parser.add_argument('--prediction_file', type=str, default='', help='Write prediction CSV to this exact path')
    parser.add_argument('--epochs', type=int, default=model_cfg.get('epochs', train_cfg.get('epochs', 100)), help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=model_cfg.get('batch_size', train_cfg.get('batch_size', 200)), help='Batch size')
    parser.add_argument('--early_stop', type=int, default=model_cfg.get('early_stop', train_cfg.get('early_stop', 10)), help='Early stopping patience')
    parser.add_argument('--use_cuda', type=int, default=common.get('use_cuda', 0), choices=[0, 1], help='Use CUDA (1) or CPU (0)')
    parser.add_argument('--blend_mode', type=str, default=train_cfg.get('blend_mode', 'auto'), choices=['auto', 'none'])
    parser.add_argument('--blend_alpha', type=float, default=train_cfg.get('blend_alpha', -1.0))
    parser.add_argument('--tune_val_samples', type=int, default=train_cfg.get('tune_val_samples', 5000))
    parser.add_argument('--val_candidates', type=int, default=train_cfg.get('val_candidates', 100))
    parser.add_argument('--train_candidates', type=int, default=train_cfg.get('train_candidates', 32))
    parser.add_argument(
        '--objective',
        type=str,
        default=model_cfg.get('objective', train_cfg.get('objective', 'bpr')),
        choices=['bpr', 'sampled_softmax'],
    )
    parser.add_argument(
        '--selection_metric',
        type=normalize_selection_metric,
        default=normalize_selection_metric(model_cfg.get('selection_metric', train_cfg.get('selection_metric', 'MRR'))),
        choices=['MRR', 'AP100', 'AUC100', 'AP', 'AUC'],
    )
    parser.add_argument(
        '--extra_eval_metrics',
        type=as_bool,
        default=as_bool(train_cfg.get('extra_eval_metrics', True)),
    )
    parser.add_argument('--heuristic_decay', type=float, default=train_cfg.get('heuristic_decay', 0.0))
    parser.add_argument('--binary_aux_weight', type=float, default=train_cfg.get('binary_aux_weight', 0.0))
    parser.add_argument(
        '--use_rerank',
        type=as_bool,
        default=as_bool(model_cfg.get('use_rerank', train_cfg.get('use_rerank', False))),
    )
    parser.add_argument(
        '--rerank_margin',
        type=float,
        default=model_cfg.get('rerank_margin', train_cfg.get('rerank_margin', 0.05)),
    )
    parser.add_argument(
        '--rerank_dual_head',
        type=parse_rerank_dual_head,
        default=parse_rerank_dual_head(model_cfg.get('rerank_dual_head', 'auto')),
    )
    parser.add_argument(
        '--rerank_regularization',
        type=float,
        default=model_cfg.get('rerank_regularization', 1.0),
    )
    parser.add_argument(
        '--rerank_feature_set',
        type=normalize_rerank_feature_set,
        default=normalize_rerank_feature_set(model_cfg.get('rerank_feature_set', 'enhanced')),
        choices=['basic', 'enhanced'],
    )
    parser.add_argument('--num_negatives', type=int, default=model_cfg.get('num_negatives', 1))
    parser.add_argument('--log_dir', type=str, default=get(config, 'paths', 'log_dir', './outputs/logs'))
    parser.add_argument('--seed', type=int, default=common.get('seed', 1), help='Random seed')
    parser.add_argument('--num_neighbors', type=int, default=model_cfg.get('num_neighbors', 30))
    parser.add_argument('--hidden_size', type=int, default=model_cfg.get('hidden_size', 64))
    parser.add_argument('--model_name', type=str, default=model_name, choices=['baseline', 'mynet'])
    parser.add_argument('--dropout', type=float, default=model_cfg.get('dropout', 0.1))
    parser.add_argument('--shortcut_scale', type=float, default=model_cfg.get('shortcut_scale', 0.35))
    parser.add_argument('--pair_feature_dim', type=int, default=model_cfg.get('pair_feature_dim', 9))
    parser.add_argument('--time_frequencies', type=int, default=model_cfg.get('time_frequencies', 8))
    parser.add_argument('--max_co_items', type=int, default=model_cfg.get('max_co_items', 20))
    parser.add_argument('--attention_heads', type=int, default=model_cfg.get('attention_heads', 4))
    parser.add_argument('--dst_num_neighbors', type=int, default=model_cfg.get('dst_num_neighbors', 10))
    parser.add_argument('--temperature', type=float, default=model_cfg.get('temperature', 0.2))
    parser.add_argument('--feature_scale', type=float, default=model_cfg.get('feature_scale', 1.0))
    parser.add_argument('--use_pair_features', type=as_bool, default=as_bool(model_cfg.get('use_pair_features', False)))
    parser.add_argument('--use_mixer', type=as_bool, default=as_bool(model_cfg.get('use_mixer', False)))
    parser.add_argument('--use_fourier_time', type=as_bool, default=as_bool(model_cfg.get('use_fourier_time', False)))
    parser.add_argument('--use_time_bias', type=as_bool, default=as_bool(model_cfg.get('use_time_bias', True)))
    parser.add_argument('--use_repeat_bias', type=as_bool, default=as_bool(model_cfg.get('use_repeat_bias', True)))
    parser.add_argument('--use_dst_time', type=as_bool, default=as_bool(model_cfg.get('use_dst_time', True)))
    parser.add_argument('--use_repeat_profile', type=as_bool, default=as_bool(model_cfg.get('use_repeat_profile', True)))
    parser.add_argument('--use_seen_new_scorer', type=as_bool, default=as_bool(model_cfg.get('use_seen_new_scorer', True)))
    parser.add_argument('--use_seen_gate', type=as_bool, default=as_bool(model_cfg.get('use_seen_gate', True)))
    parser.add_argument('--shortcut_dropout', type=float, default=model_cfg.get('shortcut_dropout', 0.0))
    parser.add_argument('--lr', type=float, default=model_cfg.get('lr', train_cfg.get('lr', 0.00005)))
    parser.add_argument('--weight_decay', type=float, default=model_cfg.get('weight_decay', train_cfg.get('weight_decay', 0.00001)))
    return parser


def resolve_use_cuda(argv=None):
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--use_cuda', type=int, default=get(config, 'common', 'use_cuda', 0), choices=[0, 1])
    args, _ = parser.parse_known_args(argv)
    return args.use_cuda


def main(argv=None):
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    configure_jittor_backend(resolve_use_cuda(argv))

    import jittor as jt
    from jittor_geometric.data import TemporalData
    from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler

    args = build_parser(config, argv).parse_args(argv)
    args.rerank_dual_head = resolve_rerank_dual_head(args.dataset, args.rerank_dual_head)
    jt.flags.use_cuda = args.use_cuda
    seed_everything(args.seed, jt)

    log_name = f'{args.dataset}_{datetime.now().strftime("%m%d_%H%M")}.log'
    install_tee(osp.join(args.log_dir, log_name))

    model_name = args.model_name.lower()
    model_tag = model_name.upper()
    print('=' * 80)
    print(f'{model_tag} Competition - Dataset: {args.dataset}')
    print(f'Seed: {args.seed}')
    print(f'Config: {display_path(args.config)}')
    print('=' * 80)

    df = pd.read_csv(f'{args.data_dir}/{args.dataset}/train.csv')
    test_df = pd.read_csv(f'{args.data_dir}/{args.dataset}/test.csv')

    src_np = df['src'].values.astype(np.int32)
    dst_np = df['dst'].values.astype(np.int32)
    t_np = df['time'].values.astype(np.int32)
    edge_ids_np = np.arange(len(df), dtype=np.int32) + 1

    test_src = test_df['src'].values.astype(np.int32)
    test_time = test_df['time'].values.astype(np.int32)
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

    num_total = len(df)
    num_val = int(num_total * 0.15)
    num_train = num_total - num_val

    train_data = TemporalData(
        src=jt.Var(src_np[:num_train]),
        dst=jt.Var(dst_np[:num_train]),
        t=jt.Var(t_np[:num_train]),
        edge_ids=jt.Var(edge_ids_np[:num_train]),
    )
    val_data = TemporalData(
        src=jt.Var(src_np[num_train:]),
        dst=jt.Var(dst_np[num_train:]),
        t=jt.Var(t_np[num_train:]),
        edge_ids=jt.Var(edge_ids_np[num_train:]),
    )
    full_data = TemporalData(
        src=jt.Var(src_np),
        dst=jt.Var(dst_np),
        t=jt.Var(t_np),
        edge_ids=jt.Var(edge_ids_np),
    )

    baseline_family = is_baseline_family(model_name)
    bpr_objective = uses_bpr_objective(model_name, args.objective)
    mynet_rerank_enabled = resolve_mynet_rerank(args)

    bpr_num_negatives = max(1, int(args.num_negatives if model_name == 'mynet' else 1))

    if bpr_objective:
        train_sampler = None
        train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=float(bpr_num_negatives))
        val_loader = TemporalDataLoader(val_data, batch_size=args.batch_size, neg_sampling_ratio=1.0)
    else:
        train_feature_store = TemporalFeatureStore(
            src_np[:num_train],
            dst_np[:num_train],
            t_np[:num_train],
            decay_scale=args.heuristic_decay,
            max_co_items=args.max_co_items,
        )
        train_sampler = CandidateSampler(
            src_np[:num_train],
            dst_np[:num_train],
            t_np[:num_train],
            train_feature_store,
            seed=args.seed,
        )
        train_loader = TemporalDataLoader(train_data, batch_size=args.batch_size, neg_sampling_ratio=0.0)
        val_loader = None

    full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=args.seed)

    val_len = len(src_np) - num_train
    val_eval_size = min(args.tune_val_samples, val_len)
    if val_eval_size <= 0:
        raise ValueError('Validation split is empty; cannot tune or early stop on MRR')

    val_eval_idx = np.linspace(0, val_len - 1, val_eval_size, dtype=np.int64)
    val_src_eval = src_np[num_train:][val_eval_idx]
    val_dst_eval = dst_np[num_train:][val_eval_idx]
    val_time_eval = t_np[num_train:][val_eval_idx]

    assert_left_history_only(
        full_neighbor_sampler,
        src_np[:num_train],
        t_np[:num_train],
        args.num_neighbors,
        'train_src_history',
    )
    assert_left_history_only(
        full_neighbor_sampler,
        val_src_eval,
        val_time_eval,
        args.num_neighbors,
        'val_src_history',
    )
    test_probe_rows = min(len(test_src), 32)
    if test_probe_rows > 0:
        assert_left_history_only(
            full_neighbor_sampler,
            test_candidates[:test_probe_rows].reshape(-1),
            np.repeat(test_time[:test_probe_rows], test_candidates.shape[1]),
            1,
            'test_candidate_dst_last_update',
            max_rows=512,
        )

    if bpr_objective:
        val_feature_store = TemporalFeatureStore(
            src_np[:num_train],
            dst_np[:num_train],
            t_np[:num_train],
            decay_scale=args.heuristic_decay,
            max_co_items=args.max_co_items,
        )
        val_heuristic_scores = None
        val_sampler = CandidateSampler(
            src_np[:num_train],
            dst_np[:num_train],
            t_np[:num_train],
            val_feature_store,
            seed=args.seed,
        )
        val_candidates = val_sampler.sample(
            val_src_eval,
            val_dst_eval,
            val_time_eval,
            num_candidates=args.val_candidates,
        )
    else:
        val_feature_store = TemporalFeatureStore(
            src_np[:num_train],
            dst_np[:num_train],
            t_np[:num_train],
            decay_scale=args.heuristic_decay,
            max_co_items=args.max_co_items,
        )
        val_sampler = CandidateSampler(
            src_np[:num_train],
            dst_np[:num_train],
            t_np[:num_train],
            val_feature_store,
            seed=args.seed,
        )
        val_candidates = val_sampler.sample(
            val_src_eval,
            val_dst_eval,
            val_time_eval,
            num_candidates=args.val_candidates,
        )
        val_heuristic_scores = None
        if args.blend_mode == 'auto':
            val_heuristic_scores = val_feature_store.heuristic_score(val_src_eval, val_time_eval, val_candidates)

    max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
    node_size = max_node + 1
    dst_min = min(int(dst_np.min()), int(test_candidates.min()))
    src_min = int(src_np.min())

    model = build_model(args, node_size)
    model.set_min_idx(src_min, dst_min)
    print_model_banner(args, model)
    optimizer = jt.nn.Adam(
        list(model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_path = args.save_model_dir
    os.makedirs(save_path, exist_ok=True)

    print(
        f'\nTraining for {args.epochs} epoch(s) with early stopping '
        f'(patience={args.early_stop}, selection_metric={args.selection_metric}, '
        f'extra_eval_metrics={args.extra_eval_metrics})...'
    )
    if bpr_objective and model_name == 'mynet' and bpr_num_negatives != 1:
        print(f'Using {bpr_num_negatives} random negatives per positive for mynet BPR training.')
    best_val_neural_scores = None
    if bpr_objective:
        best_score = train_baseline(
            jt,
            model,
            optimizer,
            train_loader,
            val_loader,
            full_neighbor_sampler,
            args.num_neighbors,
            args.epochs,
            save_path,
            args.dataset,
            val_src_eval,
            val_time_eval,
            val_candidates,
            args.batch_size,
            args.early_stop,
            model_tag,
            args.selection_metric,
            args.extra_eval_metrics,
            bpr_num_negatives,
        )
    else:
        best_score, best_val_neural_scores = train_model(
            jt,
            model,
            optimizer,
            train_loader,
            train_sampler,
            args.train_candidates,
            val_src_eval,
            val_time_eval,
            val_candidates,
            full_neighbor_sampler,
            args.num_neighbors,
            args.epochs,
            save_path,
            args.dataset,
            args.batch_size,
            args.early_stop,
            model_tag,
            val_feature_store,
            args.binary_aux_weight,
            args.selection_metric,
        )

    print('\nGenerating predictions using best model...')
    best_model_path = osp.join(save_path, f'{args.dataset}_{model_tag}_best.pkl')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f'Best model not found: {best_model_path}')
    model.load_state_dict(jt.load(best_model_path))
    print(f'Loaded best model from {best_model_path}')

    test_feature_store = None
    needs_test_features = (
        not baseline_family
        and (
            (not bpr_objective)
            or getattr(model, 'supports_pair_features', False)
            or args.blend_mode == 'auto'
            or (model_name == 'mynet' and mynet_rerank_enabled)
        )
    )
    if needs_test_features:
        test_feature_store = TemporalFeatureStore(
            src_np,
            dst_np,
            t_np,
            decay_scale=args.heuristic_decay,
            max_co_items=args.max_co_items,
        )

    learned_reranker = None
    if model_name == 'mynet' and mynet_rerank_enabled:
        print('\nRerank: training validation ranker...')
        if best_val_neural_scores is None:
            if bpr_objective:
                best_val_neural_scores = test_baseline_competition(
                    jt,
                    model,
                    val_src_eval,
                    val_time_eval,
                    val_candidates,
                    full_neighbor_sampler,
                    args.num_neighbors,
                    args.batch_size,
                    desc='Rerank val',
                )
            else:
                best_val_neural_scores = test_competition(
                    jt,
                    model,
                    val_src_eval,
                    val_time_eval,
                    val_candidates,
                    full_neighbor_sampler,
                    args.num_neighbors,
                    args.batch_size,
                    desc='Rerank val',
                    feature_store=val_feature_store,
                )

        rerank_train_slice, rerank_holdout_slice = split_rerank_validation(len(val_src_eval))
        learned_reranker = train_learned_reranker(
            val_feature_store,
            val_src_eval[rerank_train_slice],
            val_time_eval[rerank_train_slice],
            val_candidates[rerank_train_slice],
            best_val_neural_scores[rerank_train_slice],
            dual_head=args.rerank_dual_head,
            regularization=args.rerank_regularization,
            feature_set=args.rerank_feature_set,
        )
        print(format_reranker_weights(learned_reranker))
        learned_reranker_path = reranker_path(save_path, args.dataset, model_tag)
        save_learned_reranker(learned_reranker, learned_reranker_path)
        learned_val_scores = predict_learned_rerank_scores(
            learned_reranker,
            val_feature_store,
            val_src_eval[rerank_holdout_slice],
            val_time_eval[rerank_holdout_slice],
            val_candidates[rerank_holdout_slice],
            best_val_neural_scores[rerank_holdout_slice],
        )
        holdout_neural_scores = best_val_neural_scores[rerank_holdout_slice]
        rerank_policy = tune_learned_rerank_policy(holdout_neural_scores, learned_val_scores)
        learned_reranker.rerank_alpha = rerank_policy['alpha']
        learned_reranker.rerank_margin = rerank_policy['margin']
        learned_reranker.rerank_coverage = rerank_policy['coverage']
        save_learned_reranker(learned_reranker, learned_reranker_path)
        gated_val_scores = rerank_policy['scores']
        val_rerank_mask = rerank_policy['mask']
        neural_val_metrics = candidate_ranking_metrics(holdout_neural_scores)
        gated_val_metrics = candidate_ranking_metrics(gated_val_scores)
        print(format_rerank_summary(
            'Rerank holdout',
            neural_val_metrics,
            gated_val_metrics,
            val_rerank_mask,
            learned_reranker.rerank_alpha,
            learned_reranker.rerank_margin,
            learned_reranker.rerank_coverage,
        ))
        print(format_slice_metrics(
            'Rerank holdout neural slices',
            candidate_slice_metrics(
                holdout_neural_scores,
                val_feature_store,
                val_src_eval[rerank_holdout_slice],
                val_time_eval[rerank_holdout_slice],
                val_candidates[rerank_holdout_slice],
            ),
        ))
        print(format_slice_metrics(
            'Rerank holdout upgraded slices',
            candidate_slice_metrics(
                gated_val_scores,
                val_feature_store,
                val_src_eval[rerank_holdout_slice],
                val_time_eval[rerank_holdout_slice],
                val_candidates[rerank_holdout_slice],
            ),
        ))

    if bpr_objective:
        neural_scores = test_baseline_competition(
            jt,
            model,
            test_src,
            test_time,
            test_candidates,
            full_neighbor_sampler,
            args.num_neighbors,
            args.batch_size,
            desc='Predict',
        )
    else:
        neural_scores = test_competition(
            jt,
            model,
            test_src,
            test_time,
            test_candidates,
            full_neighbor_sampler,
            args.num_neighbors,
            args.batch_size,
            feature_store=test_feature_store,
        )

    if model_name == 'mynet' and mynet_rerank_enabled:
        print('\nApplying learned mynet reranker...')
        if learned_reranker is None:
            raise RuntimeError('Learned reranker was not trained before test prediction')
        learned_scores = predict_learned_rerank_scores(
            learned_reranker,
            test_feature_store,
            test_src,
            test_time,
            test_candidates,
            neural_scores,
        )
        scores, test_rerank_mask = apply_learned_rerank_on_uncertain_rows(
            neural_scores,
            learned_scores,
            getattr(learned_reranker, 'rerank_margin', args.rerank_margin),
            alpha=getattr(learned_reranker, 'rerank_alpha', 0.0),
            coverage=getattr(learned_reranker, 'rerank_coverage', None),
        )
        print(
            f'Rerank test: applied {int(test_rerank_mask.sum())}/{len(test_rerank_mask)} '
            f'({100.0 * test_rerank_mask.mean():.1f}%), '
            f'alpha={getattr(learned_reranker, "rerank_alpha", 0.0):.2f}, '
            f'margin={getattr(learned_reranker, "rerank_margin", args.rerank_margin):.4f}'
        )
    elif not baseline_family and args.blend_mode == 'auto':
        print('\nBuilding temporal heuristic scores...')
        heuristic_scores = test_feature_store.heuristic_score(test_src, test_time, test_candidates)

        if args.blend_alpha >= 0:
            blend_alpha = min(max(args.blend_alpha, 0.0), 1.0)
            print(f'Using user blend alpha: {blend_alpha:.2f}')
        else:
            print(f'Tuning blend alpha on {val_eval_size} validation rows with MRR...')
            if best_val_neural_scores is None:
                best_val_neural_scores = test_competition(
                    jt,
                    model,
                    val_src_eval,
                    val_time_eval,
                    val_candidates,
                    full_neighbor_sampler,
                    args.num_neighbors,
                    args.batch_size,
                    desc='Blend tuning',
                    feature_store=val_feature_store,
                )
            blend_alpha, best_val_mrr = tune_blend_alpha(row_minmax(best_val_neural_scores), val_heuristic_scores)
            neural_mrr = mean_reciprocal_rank(row_minmax(best_val_neural_scores))
            heuristic_mrr = mean_reciprocal_rank(val_heuristic_scores)
            print(
                f'Validation MRR - {model_tag}: {neural_mrr:.6f}, '
                f'heuristic: {heuristic_mrr:.6f}, blended: {best_val_mrr:.6f} '
                f'(alpha={blend_alpha:.2f})'
            )

        scores = blend_alpha * row_minmax(neural_scores) + (1.0 - blend_alpha) * heuristic_scores
    else:
        scores = neural_scores

    print(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')

    output_file = args.prediction_file or osp.join(args.submission_dir, f'{args.dataset}.csv')
    output_parent = osp.dirname(output_file)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')
    print(f'Prediction written -> {output_file}')

    print('\n' + '=' * 80)
    print('DONE')
    print('=' * 80)
    return best_score
