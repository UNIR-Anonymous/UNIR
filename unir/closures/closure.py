import fnmatch
from collections import OrderedDict

import torch

from utils.meters import AverageMeter


class Closure(object):

    def forward(self, *input):
        self._set_batch_size(*input)
        self._forward(*input)
        self._update_metrics()

    def backward(self):
        self._backward()
        self._update_metrics()

    def step(self):
        self._step()
        metrics = OrderedDict()
        for n in self._metrics:
            metrics[n] = self._metrics[n].get()
        self._reset_metrics()
        self._update_images()

        return metrics, self._images

    def _forward(self, *input):
        raise NotImplementedError

    def _backward(self):
        raise NotImplementedError

    def _step(self):
        raise NotImplementedError

    def _register_metrics(self, metric_names):
        self._metric_names = metric_names
        self._metrics = OrderedDict()
        self._images = OrderedDict()

    def _update_metrics(self, n=1):
        for pattern in self._metric_names:
            for name in fnmatch.filter(self.__dict__, pattern):
                if name not in self._metrics:
                    self._metrics[name] = AverageMeter()
                v = self.__dict__[name]
                if isinstance(v, torch.Tensor):
                    v = v.item()
                self._metrics[name].update(v, n)

    def _update_images(self):
        for k, v in self.__dict__.items():
            if type(v) == torch.Tensor:
                self._images[k] = v.to("cpu")

    def _reset_metrics(self):
        for m in self._metrics.values():
            m.reset()

    def _set_batch_size(self, *input, batch_dim=0):
        for t in input:
            if isinstance(t, torch.Tensor):
                self.curr_batch_size = t.shape[batch_dim]
                break


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
