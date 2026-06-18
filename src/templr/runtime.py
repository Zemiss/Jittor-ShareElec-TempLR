import os
import os.path as osp
import random
import sys
import time

import numpy as np
from tqdm import tqdm

from .core import candidate_ranking_metrics, mean_reciprocal_rank, row_minmax


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m{seconds:04.1f}s'
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s'


def _without_cuda_paths(value):
    blocked = ('cuda', 'cudnn', 'nvidia')
    return os.pathsep.join(
        path
        for path in value.split(os.pathsep)
        if path and not any(token in path.lower() for token in blocked)
    )


def configure_jittor_backend(use_cuda):
    os.environ['JT_SYNC'] = '1'
    os.environ['PATH'] = _without_cuda_paths(os.environ.get('PATH', ''))
    os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(
        path
        for path in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        if path == '/usr/lib/wsl/lib'
    )
    if use_cuda == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['CUDA_HOME'] = ''
        os.environ['CUDA_PATH'] = ''
        os.environ['nvcc_path'] = ''
    else:
        cuda_home = sys.prefix
        driver_lib = '/usr/lib/wsl/lib'
        os.environ['CUDA_HOME'] = cuda_home
        os.environ['CUDA_PATH'] = cuda_home
        os.environ['nvcc_path'] = osp.join(cuda_home, 'bin', 'nvcc')
        os.environ['cuda_archs'] = '89'
        if osp.isdir(driver_lib):
            os.environ['LD_LIBRARY_PATH'] = driver_lib


def seed_everything(seed, jt):
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


def build_dst_last_update_times(jt, full_neighbor_sampler, batch_time, test_dst):
    dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
        node_ids=test_dst.flatten().numpy(),
        node_interact_times=np.broadcast_to(batch_time[:, np.newaxis], (len(batch_time), test_dst.shape[1])).flatten(),
        num_neighbors=1,
    )
    dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
    dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
    return jt.Var(dst_last_update_time)


def build_dst_history(full_neighbor_sampler, batch_time, test_dst_np, num_neighbors):
    flat_dst = test_dst_np.reshape(-1)
    flat_time = np.broadcast_to(batch_time[:, np.newaxis], test_dst_np.shape).reshape(-1)
    neighbors, _, history_times = full_neighbor_sampler.get_historical_neighbors_left(
        node_ids=flat_dst,
        node_interact_times=flat_time,
        num_neighbors=num_neighbors,
    )
    dst_shape = (*test_dst_np.shape, num_neighbors)
    return np.asarray(neighbors).reshape(dst_shape), np.asarray(history_times).reshape(dst_shape)


def forward_model(
    jt,
    model,
    src_neighb_seq_adj,
    neighbor_num,
    src_neighb_interact_times,
    batch_time,
    test_dst_adj,
    dst_last_update_time,
    pair_features=None,
    src_ids=None,
    dst_neighb_seq=None,
    dst_neighb_interact_times=None,
):
    kwargs = {}
    if getattr(model, 'supports_pair_features', False) and pair_features is not None:
        kwargs['pair_features'] = jt.Var(pair_features)
    if getattr(model, 'supports_src_ids', False) and src_ids is not None:
        kwargs['src_ids'] = jt.Var(src_ids)
    if getattr(model, 'supports_dst_history', False) and dst_neighb_seq is not None:
        kwargs['dst_neighb_seq'] = jt.Var(dst_neighb_seq)
        kwargs['dst_neighb_interact_times'] = jt.Var(dst_neighb_interact_times)
    return model.forward(
        src_neighb_seq_adj,
        jt.Var(neighbor_num),
        jt.Var(src_neighb_interact_times),
        jt.Var(batch_time),
        test_dst=test_dst_adj,
        dst_last_update_times=dst_last_update_time,
        **kwargs,
    )


def evaluate_mrr(
    jt,
    model,
    val_src,
    val_time,
    val_candidates,
    full_neighbor_sampler,
    num_neighbors,
    batch_size,
    feature_store=None,
):
    scores = test_competition(
        jt,
        model,
        val_src,
        val_time,
        val_candidates,
        full_neighbor_sampler,
        num_neighbors,
        batch_size,
        desc='Validation MRR',
        feature_store=feature_store,
    )
    return candidate_ranking_metrics(scores), scores


def train_model(
    jt,
    model,
    optimizer,
    train_loader,
    train_sampler,
    train_candidates,
    val_src,
    val_time,
    val_candidates,
    full_neighbor_sampler,
    num_neighbors,
    num_epochs,
    save_path,
    dataset_name,
    batch_size,
    early_stop_patience=10,
    model_tag='model',
    val_feature_store=None,
    binary_aux_weight=0.0,
    selection_metric='MRR',
):
    os.makedirs(save_path, exist_ok=True)
    selection_metric = ''.join(ch for ch in str(selection_metric).upper() if ch.isalnum())
    if selection_metric in {'AP', 'AUC'}:
        raise ValueError(f'{selection_metric} selection requires BPR/main.py-style validation; use AP100 or AUC100 here')
    best_score = 0
    patience_counter = 0
    best_val_scores = None
    best_checkpoint = osp.join(save_path, f'{dataset_name}_{model_tag}_best.pkl')

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []
        rank_losses = []
        binary_losses = []
        train_tqdm = tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch+1}')

        for batch_data in train_tqdm:
            src_np = np.asarray(batch_data.src.numpy(), dtype=np.int32)
            dst_np = np.asarray(batch_data.dst.numpy(), dtype=np.int32)
            t_np = np.asarray(batch_data.t.numpy(), dtype=np.int64)

            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src_np, node_interact_times=t_np, num_neighbors=num_neighbors)
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            if neighbor_num.sum() == 0:
                continue

            candidate_dst = train_sampler.sample(
                src_np,
                dst_np,
                t_np,
                num_candidates=train_candidates,
            )
            test_dst = jt.Var(candidate_dst)
            dst_last_update_time = build_dst_last_update_times(jt, full_neighbor_sampler, t_np, test_dst)

            src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
            test_dst_adj = test_dst - model.dst_min_idx + 1
            src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)
            pair_features = None
            if getattr(model, 'supports_pair_features', False):
                pair_features = train_sampler.feature_store.candidate_matrix(src_np, t_np, candidate_dst)
            dst_neighb_seq = None
            dst_neighb_times = None
            if getattr(model, 'supports_dst_history', False):
                dst_neighb_seq, dst_neighb_times = build_dst_history(
                    full_neighbor_sampler,
                    t_np,
                    candidate_dst,
                    getattr(model, 'dst_num_neighbors', num_neighbors),
                )

            logits = forward_model(
                jt,
                model,
                src_neighb_seq_adj,
                neighbor_num,
                src_neighb_interact_times,
                t_np,
                test_dst_adj,
                dst_last_update_time,
                pair_features,
                src_np,
                dst_neighb_seq,
                dst_neighb_times,
            ).squeeze(-1)
            scaled_logits = logits / max(float(getattr(model, 'temperature', 1.0)), 1e-6)
            rank_loss = jt.nn.logsumexp(scaled_logits, dim=1) - scaled_logits[:, 0]
            rank_loss = rank_loss.mean()
            loss = rank_loss
            if binary_aux_weight > 0:
                pos_loss = jt.log(1.0 + jt.exp(-logits[:, 0])).mean()
                neg_loss = jt.log(1.0 + jt.exp(logits[:, 1:])).mean()
                binary_loss = 0.5 * (pos_loss + neg_loss)
                loss = loss + binary_aux_weight * binary_loss
                binary_losses.append(binary_loss.item())
            rank_losses.append(rank_loss.item())

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            train_losses.append(loss.item())
            train_tqdm.set_description(f'Epoch {epoch+1}, loss: {loss.item():.4f}')

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}')
        if binary_aux_weight > 0:
            rank_loss = float(np.mean(rank_losses)) if rank_losses else float('nan')
            binary_loss = float(np.mean(binary_losses)) if binary_losses else float('nan')
            print(
                f'Epoch {epoch+1}, Loss Detail: rank={rank_loss:.4f}, '
                f'binary_aux={binary_loss:.4f}, binary_aux_weight={binary_aux_weight:.3f}'
            )
        val_start = time.time()
        val_metrics, val_scores = evaluate_mrr(
            jt,
            model,
            val_src,
            val_time,
            val_candidates,
            full_neighbor_sampler,
            num_neighbors,
            batch_size,
            feature_store=val_feature_store,
        )
        val_seconds = time.time() - val_start
        epoch_seconds = time.time() - epoch_start
        current_score = val_metrics[selection_metric]
        print(
            f'Epoch {epoch+1}, Val MRR@{val_candidates.shape[1]}: {val_metrics["MRR"]:.6f}, '
            f'AP100: {val_metrics["AP100"]:.6f}, AUC100: {val_metrics["AUC100"]:.6f} '
            f'(val_time={format_duration(val_seconds)}, epoch_time={format_duration(epoch_seconds)})'
        )

        if current_score > best_score:
            best_score = current_score
            best_val_scores = val_scores
            patience_counter = 0
            os.makedirs(save_path, exist_ok=True)
            jt.save(model.state_dict(), best_checkpoint)
            print(f'  -> New best {selection_metric}: {best_score:.6f}, model saved!')
        else:
            patience_counter += 1
            print(
                f'  -> No improvement for {patience_counter} epoch(s), '
                f'best {selection_metric}: {best_score:.6f}'
            )

        print('=' * 80)

        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {epoch+1} epochs!')
            print(f'Best validation {selection_metric}: {best_score:.6f}')
            break

    return best_score, best_val_scores


def test_competition(
    jt,
    model,
    test_src,
    test_time,
    test_candidates,
    full_neighbor_sampler,
    num_neighbors,
    batch_size=200,
    desc='Testing',
    feature_store=None,
):
    model.eval()
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    pbar = tqdm(range(num_batches), dynamic_ncols=True, desc=desc)
    for batch_idx in pbar:
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        batch_src = test_src[start:end]
        batch_time = test_time[start:end]
        batch_cand = test_candidates[start:end]

        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=batch_src, node_interact_times=batch_time, num_neighbors=num_neighbors)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst = jt.Var(batch_cand)
        dst_last_update_time = build_dst_last_update_times(jt, full_neighbor_sampler, batch_time, test_dst)

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)
        pair_features = None
        if getattr(model, 'supports_pair_features', False) and feature_store is not None:
            pair_features = feature_store.candidate_matrix(batch_src, batch_time, batch_cand)
        dst_neighb_seq = None
        dst_neighb_times = None
        if getattr(model, 'supports_dst_history', False):
            dst_neighb_seq, dst_neighb_times = build_dst_history(
                full_neighbor_sampler,
                batch_time,
                batch_cand,
                getattr(model, 'dst_num_neighbors', num_neighbors),
            )

        logits = forward_model(
            jt,
            model,
            src_neighb_seq_adj,
            neighbor_num,
            src_neighb_interact_times,
            batch_time,
            test_dst_adj,
            dst_last_update_time,
            pair_features,
            batch_src,
            dst_neighb_seq,
            dst_neighb_times,
        )
        probs = jt.sigmoid(logits.squeeze(-1)).numpy()
        all_scores.append(probs)

    return np.vstack(all_scores)
