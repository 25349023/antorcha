import abc

import numpy as _np
import torch as _torch


class Metric(abc.ABC):
    metric_names = []

    @abc.abstractmethod
    def reset(self):
        """Reset the metric internal state"""

    @abc.abstractmethod
    def accumulate(self, pred, gt):
        """Accumulate the metric result w.r.t. the prediction and ground truth"""

    @abc.abstractmethod
    def results(self) -> tuple:
        """Return the metric results in tuple"""


class Accuracy(Metric):
    metric_names = ['Accuracy']

    def __init__(self):
        self.batch_results = None

    def reset(self):
        self.batch_results = []

    def accumulate(self, pred: _np.ndarray, gt: _np.ndarray):
        self.batch_results.append((pred.argmax(axis=1) == gt).mean())

    def results(self) -> tuple:
        return _np.array(self.batch_results).mean(),
