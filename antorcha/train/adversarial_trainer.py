import sys as _sys
from collections.abc import Iterable as _Iterable

import numpy as _np
import torch as _torch
import tqdm as _tqdm
from torch import nn as _nn


def train_adv_one_epoch(
        model: _nn.Module,
        dataset: _Iterable,
):
    losses = [[] for _ in range(len(model.loss_names))]

    dataset = _tqdm.tqdm(dataset, file=_sys.stdout, desc='Training... ', leave=False)
    for (x,) in dataset:
        batch_loss = model.train_adv(x)
        for b_loss, loss in zip(batch_loss, losses):
            loss.append(b_loss)

    return tuple(_np.nanmean(_np.array(loss, dtype=_np.float))
                 for loss in losses)


def test_adv_one_epoch(
        model: _nn.Module,
        dataset: _Iterable,
        metrics=False
):
    losses = [[] for _ in range(len(model.loss_names))]

    with _torch.no_grad():
        if metrics:
            model.metric.reset()
        dataset = _tqdm.tqdm(dataset, file=_sys.stdout, desc='Testing... ', leave=False)
        for (x,) in dataset:
            *batch_loss, generated_images = model.test_adv(x)
            for b_loss, loss in zip(batch_loss, losses):
                loss.append(b_loss.item())
            if metrics:
                model.metric.accumulate_batch(generated_images)

    metric_results = model.metric.results() if metrics else ()
    return tuple(_np.array(loss).mean() for loss in losses if loss), metric_results


# [TODO] adding support for lr scheduler
def fit_adv(
        model,
        train_ld: _Iterable,
        test_ld: _Iterable,
        metrics=False,
        epochs=30
):
    def gen_loss_str(losses, precision=6):
        return ', '.join(f'{name} = {loss:.{precision}f}'
                         for name, loss in zip(model.loss_names, losses))

    def gen_results_str(results, result_names, precision=6):
        res_strs = [f'\t\t| {m} = {results[i]:.{precision}f}'
                    for i, m in enumerate(result_names)]
        return '\n'.join(res_strs)

    model.eval()
    te_losses, _ = test_adv_one_epoch(model, test_ld)
    print(f'Initial: {gen_loss_str(te_losses)}\n')

    for i in range(epochs):
        model.train()
        tr_losses = train_adv_one_epoch(model, train_ld)
        model.eval()
        te_losses, metric_results = test_adv_one_epoch(model, test_ld, metrics)

        print(f'Epoch {i:>2}: training: {gen_loss_str(tr_losses)}')
        print(f'\t\t|  testing: {gen_loss_str(te_losses)}')

        if metric_results:
            print(gen_results_str(metric_results, model.metric.metric_names), end='\n')

        print()

