import argparse
import os
import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd

from .baseline import test_baseline_competition
from .config import (
    as_bool,
    get,
    get_model_config,
    get_run_model_name,
    load_default_config,
    parse_config_path,
)
from .core import TemporalFeatureStore, install_tee, row_minmax
from .models import build_model
from .runtime import configure_jittor_backend, seed_everything, test_competition
from .training import display_path, is_baseline_family


def resolve_model_defaults(config, argv=None):
    default_model_name = get_run_model_name(config)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=default_model_name, choices=['baseline', 'mynet'])
    args, _ = parser.parse_known_args(argv)
    model_name = args.model_name.lower()
    return model_name, get_model_config(config, model_name)


def build_parser(config, argv=None):
    common = config.get('common', {})
    train = config.get('train', {})
    model_name, model_cfg = resolve_model_defaults(config, argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=parse_config_path())
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default=get(config, 'paths', 'data_dir', './data_A'))
    parser.add_argument('--save_dir', type=str, default=get(config, 'paths', 'model_dir', './models'))
    parser.add_argument('--output_dir', type=str, default=get(config, 'paths', 'submission_dir', './outputs/submission'))
    parser.add_argument('--prediction_file', type=str, default='')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=train.get('batch_size', 200))
    parser.add_argument('--use_cuda', type=int, default=common.get('use_cuda', 0), choices=[0, 1])
    parser.add_argument('--seed', type=int, default=common.get('seed', 1))
    parser.add_argument('--blend_mode', type=str, default=train.get('blend_mode', 'none'), choices=['auto', 'none'])
    parser.add_argument('--blend_alpha', type=float, default=train.get('blend_alpha', -1.0))
    parser.add_argument('--heuristic_decay', type=float, default=train.get('heuristic_decay', 0.0))
    parser.add_argument('--mynet_heuristic_alpha', type=float, default=train.get('mynet_heuristic_alpha', 1.0))
    parser.add_argument('--log_dir', type=str, default=get(config, 'paths', 'log_dir', './outputs/logs'))
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
    return parser


def write_prediction(path, scores):
    parent = osp.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in scores:
            f.write(','.join([f'{p:.8f}' for p in row]) + '\n')


def main(argv=None):
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    args = build_parser(config, argv).parse_args(argv)
    configure_jittor_backend(args.use_cuda)

    import jittor as jt
    from jittor_geometric.data import TemporalData
    from jittor_geometric.dataloader.temporal_dataloader import get_neighbor_sampler

    seed_everything(args.seed, jt)

    log_name = f'{args.dataset}_{datetime.now().strftime("%m%d_%H%M")}.log'
    install_tee(osp.join(args.log_dir, 'test', log_name))

    model_name = args.model_name.lower()
    model_tag = model_name.upper()
    print('=' * 80)
    print(f'{model_tag} Test - Dataset: {args.dataset}')
    print(f'Seed: {args.seed}')
    print(f'Config: {display_path(args.config)}')
    print('=' * 80)

    train_df = pd.read_csv(osp.join(args.data_dir, args.dataset, 'train.csv'))
    test_df = pd.read_csv(osp.join(args.data_dir, args.dataset, 'test.csv'))

    src_np = train_df['src'].values.astype(np.int32)
    dst_np = train_df['dst'].values.astype(np.int32)
    t_np = train_df['time'].values.astype(np.int32)
    edge_ids_np = np.arange(len(train_df), dtype=np.int32) + 1

    test_src = test_df['src'].values.astype(np.int32)
    test_time = test_df['time'].values.astype(np.int32)
    test_candidates = test_df.iloc[:, 2:].values.astype(np.int32)

    full_data = TemporalData(
        src=jt.Var(src_np),
        dst=jt.Var(dst_np),
        t=jt.Var(t_np),
        edge_ids=jt.Var(edge_ids_np),
    )
    full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=args.seed)

    max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
    node_size = max_node + 1
    dst_min = min(int(dst_np.min()), int(test_candidates.min()))
    src_min = int(src_np.min())

    model = build_model(args, node_size)
    model.set_min_idx(src_min, dst_min)

    checkpoint = args.checkpoint or osp.join(args.save_dir, f'{args.dataset}_{model_tag}_best.pkl')
    if not osp.exists(checkpoint):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    model.load_state_dict(jt.load(checkpoint))
    print(f'Loaded model from {checkpoint}')

    baseline_family = is_baseline_family(model_name)
    feature_store = None
    needs_features = (
        not baseline_family
        and (
            getattr(model, 'supports_pair_features', False)
            or args.blend_mode == 'auto'
            or (model_name == 'mynet' and args.mynet_heuristic_alpha < 1.0)
        )
    )
    if needs_features:
        feature_store = TemporalFeatureStore(
            src_np,
            dst_np,
            t_np,
            decay_scale=args.heuristic_decay,
            max_co_items=args.max_co_items,
        )

    if baseline_family:
        scores = test_baseline_competition(
            jt,
            model,
            test_src,
            test_time,
            test_candidates,
            full_neighbor_sampler,
            args.num_neighbors,
            args.batch_size,
        )
    else:
        scores = test_competition(
            jt,
            model,
            test_src,
            test_time,
            test_candidates,
            full_neighbor_sampler,
            args.num_neighbors,
            args.batch_size,
            feature_store=feature_store,
        )

    if model_name == 'mynet' and args.mynet_heuristic_alpha < 1.0:
        alpha = min(max(args.mynet_heuristic_alpha, 0.0), 1.0)
        print(f'Blending mynet scores with temporal heuristic (alpha={alpha:.2f})...')
        heuristic_scores = feature_store.heuristic_score(test_src, test_time, test_candidates)
        scores = alpha * row_minmax(scores) + (1.0 - alpha) * heuristic_scores

    output_file = args.prediction_file or osp.join(args.output_dir, f'{args.dataset}.csv')
    write_prediction(output_file, scores)
    print(f'Scores shape: {scores.shape}, range: [{scores.min():.6f}, {scores.max():.6f}]')
    print(f'Prediction written -> {output_file}')
    print('=' * 80)
