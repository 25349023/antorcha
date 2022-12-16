import collections.abc as _abc
import sys as _sys

import numpy as _np
import torch as _torch
import tqdm as _tqdm
from torch import nn as _nn


def train_one_epoch(
        model: _nn.Module,
        dataset: _abc.Iterable,
        loss_fns: list,
        optim: _torch.optim.Optimizer
) -> float:
    losses = []
    for x, y in _tqdm.tqdm(dataset, file=_sys.stdout, desc='Training... ', leave=False):
        pred = model(x)
        loss = model.loss(*loss_fns, pred, y)
        losses.append(loss.item())

        optim.zero_grad()
        loss.backward()
        optim.step()
    return _np.array(losses).mean()


def test_one_epoch(
        model: _nn.Module,
        dataset: _abc.Iterable,
        loss_fns: list,
        metrics=False
):
    losses = []
    with _torch.no_grad():
        if metrics:
            model.metric.reset()
        for x, y in _tqdm.tqdm(dataset, file=_sys.stdout, desc='Testing... ', leave=False):
            pred: _torch.Tensor = model(x)
            loss = model.loss(*loss_fns, pred, y)
            losses.append(loss.item())
            if metrics:
                model.metric.accumulate(pred.cpu().numpy(), y.cpu().numpy())

    metric_results = model.metric.results() if metrics else ()

    return _np.array(losses).mean(), *metric_results


def fit(
        model: _nn.Module,
        train_ld: _abc.Iterable,
        test_ld: _abc.Iterable,
        loss_fn,
        optim: _torch.optim.Optimizer,
        metrics=False,
        scheduler=None,
        epochs=20,
):
    def gen_results_str(results, result_names, precision=6):
        res_strs = [f'\t| {m} = {results[i]:.{precision}f}'
                    for i, m in enumerate(result_names)]
        return '\n'.join(res_strs)

    for i in range(epochs):
        model.train()
        train_loss = train_one_epoch(model, train_ld, loss_fn, optim)
        model.eval()
        test_loss, *metric_results = test_one_epoch(model, test_ld, loss_fn, metrics)

        print(f'Epoch {i:>2}: training loss = {train_loss:.6f}, testing loss = {test_loss:.6f}')
        if metric_results:
            print(gen_results_str(metric_results, model.metric.metric_names), end='\n\n')

        if scheduler is not None:
            scheduler.step()
