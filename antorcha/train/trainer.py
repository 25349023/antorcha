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
    losses, metric_results = [], []
    with _torch.no_grad():
        for x, y in _tqdm.tqdm(dataset, file=_sys.stdout, desc='Testing... ', leave=False):
            pred: _torch.Tensor = model(x)
            loss = model.loss(*loss_fns, pred, y)
            losses.append(loss.item())
            if metrics:
                metric_results.append(model.metrics(pred, y))

    if metrics:
        # transposing the result matrix
        metric_results = zip(*metric_results)
        metric_results = tuple(_np.array(m).mean() for m in metric_results)

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
    for i in range(epochs):
        model.train()
        train_loss = train_one_epoch(model, train_ld, loss_fn, optim)
        model.eval()
        test_loss, *metric_results = test_one_epoch(model, test_ld, loss_fn, metrics)

        print(f'Epoch {i:>2}: training loss = {train_loss:.6f}, testing loss = {test_loss:.6f}')
        if metric_results:
            m_strs = [f'\t| {m} = {metric_results[i]:.6f}'
                      for i, m in enumerate(model.metric_names)]
            print('\n'.join(m_strs), end='\n\n')

        if scheduler is not None:
            scheduler.step()
