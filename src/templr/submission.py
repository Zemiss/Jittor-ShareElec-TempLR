import argparse
import os
import subprocess
import sys
import zipfile

from .config import as_bool, get, get_model_config, get_run_model_name, load_default_config, parse_config_path
from .core import check_dataset, check_one_submission_file


DEFAULT_DATASET_STRATEGIES = {
    'dataset1': {
        'blend_mode': 'auto',
        'blend_alpha': -1.0,
        'heuristic_decay': 0.0,
    },
    'dataset2': {
        'blend_mode': 'none',
        'blend_alpha': 1.0,
        'heuristic_decay': 0.0,
    },
}


def run_cmd(cmd, cwd=None):
    print('> ' + ' '.join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_strategy(dataset, args):
    strategy = dict(DEFAULT_DATASET_STRATEGIES.get(dataset, DEFAULT_DATASET_STRATEGIES['dataset1']))
    if args.blend_mode != 'auto':
        strategy['blend_mode'] = args.blend_mode
    if args.blend_alpha >= 0:
        strategy['blend_alpha'] = args.blend_alpha
    if args.heuristic_decay > 0:
        strategy['heuristic_decay'] = args.heuristic_decay
    return strategy


def resolve_model_defaults(config, argv=None):
    default_model_name = get_run_model_name(config)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_name', type=str, default=default_model_name, choices=['baseline', 'mynet'])
    args, _ = parser.parse_known_args(argv)
    model_name = args.model_name.lower()
    return model_name, get_model_config(config, model_name)


def normalize_selection_metric(value):
    return ''.join(ch for ch in str(value).upper() if ch.isalnum())


def build_parser(config, argv=None):
    submit = config.get('submission', {})
    train = config.get('train', {})
    common = config.get('common', {})
    model_name, model_cfg = resolve_model_defaults(config, argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=parse_config_path())
    parser.add_argument('--data_dir', type=str, default=get(config, 'paths', 'data_dir', './data_A'))
    parser.add_argument('--save_dir', type=str, default=get(config, 'paths', 'model_dir', './models'))
    parser.add_argument('--submission_dir', type=str, default=get(config, 'paths', 'submission_dir', './outputs/submission'))
    parser.add_argument('--zip_name', type=str, default=submit.get('zip_name', 'result.zip'))
    parser.add_argument('--datasets', type=str, nargs='+', default=config.get('datasets', ['dataset1', 'dataset2']))
    parser.add_argument('--epochs', type=int, default=model_cfg.get('epochs', train.get('epochs', 100)))
    parser.add_argument('--batch_size', type=int, default=model_cfg.get('batch_size', train.get('batch_size', 200)))
    parser.add_argument('--early_stop', type=int, default=model_cfg.get('early_stop', train.get('early_stop', 10)))
    parser.add_argument('--use_cuda', type=int, default=common.get('use_cuda', 0), choices=[0, 1])
    parser.add_argument('--blend_mode', type=str, default=train.get('blend_mode', 'auto'), choices=['auto', 'none'])
    parser.add_argument('--blend_alpha', type=float, default=train.get('blend_alpha', -1.0))
    parser.add_argument('--tune_val_samples', type=int, default=train.get('tune_val_samples', 5000))
    parser.add_argument('--val_candidates', type=int, default=train.get('val_candidates', 100))
    parser.add_argument(
        '--objective',
        type=str,
        default=model_cfg.get('objective', train.get('objective', 'bpr')),
        choices=['bpr', 'sampled_softmax'],
    )
    parser.add_argument(
        '--selection_metric',
        type=normalize_selection_metric,
        default=normalize_selection_metric(model_cfg.get('selection_metric', train.get('selection_metric', 'MRR'))),
        choices=['MRR', 'AP100', 'AUC100', 'AP', 'AUC'],
    )
    parser.add_argument(
        '--extra_eval_metrics',
        type=as_bool,
        default=as_bool(train.get('extra_eval_metrics', True)),
    )
    parser.add_argument('--heuristic_decay', type=float, default=train.get('heuristic_decay', 0.0))
    parser.add_argument('--binary_aux_weight', type=float, default=train.get('binary_aux_weight', 0.0))
    parser.add_argument('--mynet_heuristic_alpha', type=float, default=train.get('mynet_heuristic_alpha', 1.0))
    parser.add_argument('--num_negatives', type=int, default=model_cfg.get('num_negatives', 1))
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
    parser.add_argument('--lr', type=float, default=model_cfg.get('lr', train.get('lr', 0.00005)))
    parser.add_argument('--weight_decay', type=float, default=model_cfg.get('weight_decay', train.get('weight_decay', 0.00001)))
    parser.add_argument('--seed', type=int, default=submit.get('seed', common.get('seed', 1)))
    parser.add_argument('--log_dir', type=str, default=get(config, 'paths', 'log_dir', './outputs/logs'))
    return parser


def main(argv=None):
    config_path = parse_config_path(argv)
    config = load_default_config(config_path)
    args = build_parser(config, argv).parse_args(argv)

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_script = os.path.join(project_root, 'scripts', 'run.py')
    py = sys.executable

    for ds in args.datasets:
        check_dataset(args.data_dir, ds)

    os.makedirs(args.submission_dir, exist_ok=True)

    for ds in args.datasets:
        strategy = resolve_strategy(ds, args)
        dst_file = os.path.join(args.submission_dir, f'{ds}.csv')

        run_cmd([
            py, run_script,
            'train',
            '--config', args.config,
            '--dataset', ds,
            '--data_dir', args.data_dir,
            '--save_dir', args.save_dir,
            '--prediction_file', dst_file,
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--early_stop', str(args.early_stop),
            '--use_cuda', str(args.use_cuda),
            '--seed', str(args.seed),
            '--blend_mode', strategy['blend_mode'],
            '--blend_alpha', str(strategy['blend_alpha']),
            '--tune_val_samples', str(args.tune_val_samples),
            '--val_candidates', str(args.val_candidates),
            '--objective', args.objective,
            '--selection_metric', args.selection_metric,
            '--extra_eval_metrics', str(args.extra_eval_metrics),
            '--heuristic_decay', str(strategy['heuristic_decay']),
            '--binary_aux_weight', str(args.binary_aux_weight),
            '--mynet_heuristic_alpha', str(args.mynet_heuristic_alpha),
            '--num_negatives', str(args.num_negatives),
            '--model_name', args.model_name,
            '--dropout', str(args.dropout),
            '--shortcut_scale', str(args.shortcut_scale),
            '--pair_feature_dim', str(args.pair_feature_dim),
            '--time_frequencies', str(args.time_frequencies),
            '--max_co_items', str(args.max_co_items),
            '--attention_heads', str(args.attention_heads),
            '--dst_num_neighbors', str(args.dst_num_neighbors),
            '--temperature', str(args.temperature),
            '--feature_scale', str(args.feature_scale),
            '--use_pair_features', str(args.use_pair_features),
            '--use_mixer', str(args.use_mixer),
            '--use_fourier_time', str(args.use_fourier_time),
            '--use_time_bias', str(args.use_time_bias),
            '--use_repeat_bias', str(args.use_repeat_bias),
            '--use_dst_time', str(args.use_dst_time),
            '--use_repeat_profile', str(args.use_repeat_profile),
            '--use_seen_new_scorer', str(args.use_seen_new_scorer),
            '--use_seen_gate', str(args.use_seen_gate),
            '--shortcut_dropout', str(args.shortcut_dropout),
            '--lr', str(args.lr),
            '--weight_decay', str(args.weight_decay),
            '--log_dir', args.log_dir,
        ])

        print(f'Wrote prediction -> {dst_file}')

    for ds in args.datasets:
        pred_file = os.path.join(args.submission_dir, f'{ds}.csv')
        test_file = os.path.join(args.data_dir, ds, 'test.csv')
        check_one_submission_file(pred_file, test_file, ds)

    zip_path = os.path.join(args.submission_dir, args.zip_name)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for ds in args.datasets:
            filename = f'{ds}.csv'
            filepath = os.path.join(args.submission_dir, filename)
            zf.write(filepath, arcname=filename)

    print(f'Submission package created: {zip_path}')
