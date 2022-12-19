import abc as _abc

import numpy as _np
import torch as _torch


class Metric(_abc.ABC):
    metric_names = []

    @_abc.abstractmethod
    def reset(self):
        """Reset the metric internal state"""

    @_abc.abstractmethod
    def accumulate_batch(self, *args):
        """Accumulate the metric result w.r.t. the prediction and ground truth"""

    @_abc.abstractmethod
    def results(self) -> tuple:
        """Return the metric results in tuple"""


class Collection(Metric):
    """ Collection of metrics """

    def __init__(self, *metrics: Metric):
        self.metrics = metrics
        self.metric_names = [name for m in metrics for name in m.metric_names]

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def accumulate_batch(self, *args):
        for metric in self.metrics:
            metric.accumulate_batch(*args)

    def results(self) -> tuple:
        return tuple(res for m in self.metrics for res in m.results())


class Accuracy(Metric):
    metric_names = ['Accuracy']

    def __init__(self):
        self.batch_results = None

    def reset(self):
        self.batch_results = []

    def accumulate_batch(self, pred, gt):
        self.batch_results.append((pred.argmax(axis=1) == gt).mean())

    def results(self) -> tuple:
        return _np.array(self.batch_results).mean(),


class KLDivergence(Metric):
    """ KL Divergence between N(mu, var) and N(0, 1) """

    metric_names = ['KL Divergence']

    def __init__(self, dist):
        """
        Initialize KL Divergence metrics with the given distributional encoder

        :param dist: target gaussian distribution, it should have two attributes: `mu` and `log_var`
        """
        self.batch_results = None
        self.dist = dist

    @staticmethod
    def kl_loss_fn(mu, log_var):
        kl = 0.5 * _torch.sum(mu ** 2 + log_var.exp() - 1 - log_var, dim=1)
        return _torch.mean(kl)

    def reset(self):
        self.batch_results = []

    def accumulate_batch(self, *args):
        self.batch_results.append(self.kl_loss_fn(self.dist.mu, self.dist.log_var).item())

    def results(self) -> tuple:
        return _np.array(self.batch_results).mean(),


class WassersteinDistance(Metric):
    metric_names = ['Wasserstein Distance']

    def __init__(self, w_approx):
        """
        Initialize Wasserstein Distance metrics with the given distributional encoder

        :param w_approx: model that approximates the W Distance, it should have `w_dist` attribute
        """
        self.batch_results = None
        self.w_approx = w_approx

    def reset(self):
        self.batch_results = []

    def accumulate_batch(self, *args):
        self.batch_results.append(self.w_approx.w_dist)

    def results(self) -> tuple:
        return _np.array(self.batch_results).mean(),


class MeanIoU(Metric):
    metric_names = ['mIoU']

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.intersections = _np.zeros(self.num_classes)
        self.unions = _np.zeros(self.num_classes)

    def reset(self):
        self.intersections = _np.zeros(self.num_classes)
        self.unions = _np.zeros(self.num_classes)

    def accumulate_batch(self, pred, gt):
        pred_seg = pred.argmax(axis=1)

        for c in range(self.num_classes):
            pred_idx = pred_seg == c
            gt_idx = gt == c

            intersection = (pred_idx & gt_idx).sum()
            union = (pred_idx | gt_idx).sum()

            self.intersections[c] += intersection
            self.unions[c] += union

    def results(self) -> tuple:
        return (self.intersections / (self.unions + 1e-15)).mean(),
