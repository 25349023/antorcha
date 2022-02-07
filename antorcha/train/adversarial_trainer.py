import sys as _sys
from collections.abc import Iterable as _Iterable

import numpy as _np
import torch as _torch
import tqdm as _tqdm
from torch import nn as _nn

from antorcha.toy_models import gans as _gans


def train_adv_one_epoch(
        model: _nn.Module,
        dataset: _Iterable,
):
    losses = [[] for _ in range(len(model.loss_names))]

    for (x,) in _tqdm.tqdm(dataset, file=_sys.stdout, desc='Training... ', leave=False):
        batch_loss = model.train_adv(x)
        for b_loss, loss in zip(batch_loss, losses):
            loss.append(b_loss)

    return tuple(_np.array(loss).mean() for loss in losses)


def test_adv_one_epoch(
        model: _nn.Module,
        dataset: _Iterable,
):
    losses = [[] for _ in range(len(model.loss_names))]

    with _torch.no_grad():
        for (x,) in _tqdm.tqdm(dataset, file=_sys.stdout, desc='Testing... ', leave=False):
            batch_loss = model.test_adv(x)
            for b_loss, loss in zip(batch_loss, losses):
                loss.append(b_loss.item())

    return tuple(_np.array(loss).mean() for loss in losses)


def fit_adv(
        model: _gans.GenerativeAdversarialNetwork,
        train_ld: _Iterable,
        test_ld: _Iterable,
        epochs=30
):
    def gen_loss_str(losses, precision=6):
        return ', '.join(f'{name} = {loss:.{precision}f}'
                         for name, loss in zip(model.loss_names, losses))

    model.eval()
    te_losses = test_adv_one_epoch(model, test_ld)
    print(f'Initial: {gen_loss_str(te_losses)}\n')

    for i in range(epochs):
        model.train()
        tr_losses = train_adv_one_epoch(model, train_ld)
        model.eval()
        te_losses = test_adv_one_epoch(model, test_ld)

        print(f'Epoch {i:>2}: training: {gen_loss_str(tr_losses)}')
        print(f'\t\t|  testing: {gen_loss_str(te_losses)}\n')
