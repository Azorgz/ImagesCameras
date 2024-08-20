from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from warnings import warn

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

__version__ = '1.0'


class image_histogram:

    def __init__(self, image=None, density=False, weight=None):
        if image is None:
            self.hist = []
            self.bins = None
            self.colors = None
            self.depth = None
            self.channels = None
            self.density = density
            self.weight = weight
        else:
            depth = image.depth
            bins = torch.linspace(0, 1, 2 ** depth + 1)
            if image.batch_size > 1:
                histo = [torch.histogram(c, bins=bins, density=density, weight=weight) for c in image]
            else:
                histo = [torch.histogram(image.reset_layers_order()[:, c, :, :], bins=bins, density=density, weight=weight) for
                         c in range(image.channel_num)]
            self.hist = [h.hist for h in histo]
            self.bins = histo[0].bin_edges
            self.colors = ['r', 'g', 'b', 'm', 'k', 'y']
            self.depth = image.depth
            self.channels = image.channel_num
            self.density = density
            self.weight = weight

    def __add__(self, other, in_place=False):
        # assert isinstance(other, image_histogram)
        # if self.depth != other.depth or self.channels != other.channels or all(self.bins != other.bins) or len(
        #         self.hist) != len(other.hist):
        #     raise ValueError("Incompatible histogram dimensions")
        if in_place:
            self.hist = [a + b for a, b in zip(self.hist, other.hist)]
            return self
        else:
            hist = self.clone()
            hist.hist = [s_h + o_h for s_h, o_h in zip(hist.hist, other.hist)]
            return hist

    def clone(self):
        new = image_histogram(image=None, density=self.density, weight=self.weight)
        new.hist = self.hist.copy()
        new.bins = self.bins.clone()
        new.colors = self.colors.copy()
        new.depth = self.depth
        new.channels = self.channels
        return new

    def show(self, min_value=0):
        for h, color in zip(self.hist, self.colors):
            if min_value:
                val_0 = h.clone()
                val_0[h <= min_value] = torch.nan
                val_1 = h.clone()
                val_1[h > min_value] = torch.nan
                plt.plot(self.bins[:-1], val_0, color=color)
                plt.plot(self.bins[:-1], val_1, color=color, linestyle='--')
            else:
                plt.plot(self.bins[:-1], h, color=color)
        plt.show()

    # def resample(self, nb_bin=256):

    def clip(self):
        mini = 0
        maxi = len(self.bins)
        for h in self.hist:
            temp = torch.diff(torch.cumsum(h, dim=0))
            mini = max(mini, np.max(temp))
            maxi = min(maxi, np.min(temp))
        maxi = -1 if maxi == len(self.bins) else maxi
        #     for idx, h in enumerate(self.hist):
        #         h[mini] += torch.sum(h[:mini])
        #         h[maxi] = torch.sum(h[maxi:])
        #         self.hist[idx] = h[mini:maxi]
        # self.bins = self.bins[mini, maxi]
        return mini, maxi


