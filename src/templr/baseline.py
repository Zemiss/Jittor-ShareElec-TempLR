import os
import os.path as osp
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .core import candidate_ranking_metrics


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m{seconds:04.1f}s'
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s'


def build_dst_last_update_times(jt, full_neighbor_sampler, batch_time, test_dst):
    dst_last_neighbor, _, dst_last_update_time = full_neighbor_sampler.get_historical_neighbors_left(
        node_ids=test_dst.flatten().numpy(),
        node_interact_times=np.broadcast_to(batch_time[:, np.newaxis], (len(batch_time), test_dst.shape[1])).flatten(),
        num_neighbors=1,
    )
    dst_last_update_time = np.array(dst_last_update_time).reshape(len(test_dst), -1)
    dst_last_update_time[dst_last_neighbor.reshape(len(test_dst), -1) == 0] = -100000
    return jt.Var(dst_last_update_time)


def evaluate_ap_auc(jt, model, loader, full_neighbor_sampler, num_neighbors):
    model.eval()
    ap_list, auc_list = [], []
    loader_tqdm = tqdm(loader, dynamic_ncols=True, desc='Validation')
    for batch_data in loader_tqdm:
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst)
        t = jt.array(batch_data.t)
        neg_dst = jt.array(batch_data.neg_dst)

        src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
            node_ids=src.numpy(),
            node_interact_times=t.numpy(),
            num_neighbors=num_neighbors,
        )
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        pos_item = jt.Var(dst).unsqueeze(1)
        neg_item = jt.Var(neg_dst).unsqueeze(1)
        test_dst = jt.cat([pos_item, neg_item], dim=1)
        dst_last_update_time = build_dst_last_update_times(
            jt,
            full_neighbor_sampler,
            t.numpy(),
            test_dst,
        )

        pos_score, neg_score = model.predict(
            src_neighb_seq=jt.Var(src_neighb_seq),
            src_neighb_seq_len=jt.Var(neighbor_num),
            src_neighb_interact_times=jt.Var(src_neighb_interact_times),
            cur_pred_times=jt.Var(t),
            test_dst=test_dst,
            dst_last_update_times=dst_last_update_time,
        )

        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_score = np.concatenate([pos_score, neg_score.flatten()])
        ap_list.append(average_precision_score(y_true, y_score))
        auc_list.append(roc_auc_score(y_true, y_score))

    return {'AP': float(np.mean(ap_list)), 'AUC': float(np.mean(auc_list))}


def train_baseline(
    jt,
    model,
    optimizer,
    train_loader,
    val_loader,
    full_neighbor_sampler,
    num_neighbors,
    num_epochs,
    save_path,
    dataset_name,
    val_src,
    val_time,
    val_candidates,
    batch_size,
    early_stop_patience=10,
    model_tag='BASELINE',
    selection_metric='MRR',
    extra_eval_metrics=True,
):
    selection_metric = ''.join(ch for ch in str(selection_metric).upper() if ch.isalnum())
    extra_eval_metrics = bool(extra_eval_metrics)
    candidate_metrics = {'MRR', 'AP100', 'AUC100'}
    best_score = 0.0
    patience_counter = 0
    best_checkpoint = osp.join(save_path, f'{dataset_name}_{model_tag}_best.pkl')

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_tqdm = tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch + 1}')

        for batch_data in train_tqdm:
            src = jt.array(batch_data.src)
            dst = jt.array(batch_data.dst)
            t = jt.array(batch_data.t)
            neg_dst = jt.array(batch_data.neg_dst)

            src_neighb_seq, _, src_neighb_interact_times = full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(),
                node_interact_times=t.numpy(),
                num_neighbors=num_neighbors,
            )
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            if neighbor_num.sum() == 0:
                continue

            pos_item = jt.Var(dst).unsqueeze(-1)
            neg_item = jt.Var(neg_dst).unsqueeze(-1)
            test_dst = jt.cat([pos_item, neg_item], dim=-1)
            dst_last_update_time = build_dst_last_update_times(
                jt,
                full_neighbor_sampler,
                t.numpy(),
                test_dst,
            )

            loss, _, _ = model.calculate_loss(
                src_neighb_seq=jt.Var(src_neighb_seq),
                src_neighb_seq_len=jt.Var(neighbor_num),
                src_neighb_interact_times=jt.Var(src_neighb_interact_times),
                cur_pred_times=jt.Var(t),
                test_dst=test_dst,
                dst_last_update_times=dst_last_update_time,
            )

            optimizer.zero_grad()
            optimizer.step(loss)
            jt.sync_all()
            train_losses.append(loss.item())
            train_tqdm.set_description(f'Epoch {epoch + 1}, loss: {loss.item():.4f}')

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
        val_start = time.time()
        val_metrics = {}
        if extra_eval_metrics or selection_metric in candidate_metrics:
            val_scores = test_baseline_competition(
                jt,
                model,
                val_src,
                val_time,
                val_candidates,
                full_neighbor_sampler,
                num_neighbors,
                batch_size,
                desc='Validation MRR',
            )
            val_metrics.update(candidate_ranking_metrics(val_scores))
        if selection_metric in {'AP', 'AUC'}:
            pair_metrics = evaluate_ap_auc(jt, model, val_loader, full_neighbor_sampler, num_neighbors)
            val_metrics.update(pair_metrics)
        val_seconds = time.time() - val_start
        epoch_seconds = time.time() - epoch_start
        metric_parts = [f'Epoch {epoch + 1}, Val']
        if 'MRR' in val_metrics:
            metric_parts.append(
                f'MRR@{val_candidates.shape[1]}: {val_metrics["MRR"]:.6f}, '
                f'AP100: {val_metrics["AP100"]:.6f}, AUC100: {val_metrics["AUC100"]:.6f}'
            )
        if selection_metric in {'AP', 'AUC'}:
            metric_parts.append(f'AP: {val_metrics["AP"]:.6f}, AUC: {val_metrics["AUC"]:.6f}')
        metric_text = ' - '.join(metric_parts)
        metric_text += f' (val_time={format_duration(val_seconds)}, epoch_time={format_duration(epoch_seconds)})'
        print(metric_text)

        current_score = val_metrics[selection_metric]
        if current_score > best_score:
            best_score = current_score
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
            print(f'\nEarly stopping triggered after {epoch + 1} epochs!')
            print(f'Best validation {selection_metric}: {best_score:.6f}')
            break

    return best_score


def test_baseline_competition(
    jt,
    model,
    test_src,
    test_time,
    test_candidates,
    full_neighbor_sampler,
    num_neighbors,
    batch_size=200,
    desc='Testing',
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
            node_ids=batch_src,
            node_interact_times=batch_time,
            num_neighbors=num_neighbors,
        )
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst = jt.Var(batch_cand)
        dst_last_update_time = build_dst_last_update_times(
            jt,
            full_neighbor_sampler,
            batch_time,
            test_dst,
        )

        src_neighb_seq_adj = jt.Var(src_neighb_seq) - model.dst_min_idx + 1
        test_dst_adj = test_dst - model.dst_min_idx + 1
        src_neighb_seq_adj = jt.where(src_neighb_seq_adj < 0, jt.zeros_like(src_neighb_seq_adj), src_neighb_seq_adj)

        logits = model.forward(
            src_neighb_seq_adj,
            jt.Var(neighbor_num),
            jt.Var(src_neighb_interact_times),
            jt.Var(batch_time),
            test_dst=test_dst_adj,
            dst_last_update_times=dst_last_update_time,
        )
        probs = jt.sigmoid(logits.squeeze(-1)).numpy()
        all_scores.append(probs)

    return np.vstack(all_scores)
