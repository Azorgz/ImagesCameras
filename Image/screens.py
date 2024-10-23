from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor


class Screen:

    def __init__(self, data: Tensor,
                 save: str = '',
                 split_batch: bool = False,
                 split_channel: bool = False):
        self.data = data
        self.path = save or None
        self.split_batch = split_batch
        self.split_channel = split_channel

    def __call__(self):
        pass


class MatplotLibScreen(Screen, plt.Figure):
    def __init__(self, data: Tensor, *args,
                 num: str | None = None,
                 cmap: str = 'gray',
                 save: str = '',
                 split_batch: bool = False,
                 split_channel: bool = False,
                 roi: list = None,
                 point: Union[list, Tensor] = None,
                 **kwargs):
        self.cmap = cmap
        super().__init__(num=num, save=save, split_batch=split_batch, split_channel=split_channel, **kwargs)
        self.roi = roi
        self.point = point
        self.channels_names = data.channel_names if data.channel_names else np.arange(0, data.channel_num).tolist()
        if data.batch_size > 1:
            self.multiple = True
            if split_batch:
                self.data_display = [*data.to_numpy().squeeze()]
                if split_channel:
                    self.multiple = False
                    self.data_display = [[*i.moveaxis(-1, 0)] for i in self.data_display]
                elif data.channel_num != 3:
                    self.data_display = [i.moveaxis(-1, 0)[c] for i in self.data_display for c in i.shape[-1]]
                else:
                    self.multiple = False
            else:
                self.data_display = [*data.to_numpy().squeeze()]
                if split_channel:
                    self.data_display = [[*i.moveaxis(-1, 0)] for i in self.data_display]
                elif data.channel_num != 3 and data.channel_num > 1:
                    self.data_display = [i.moveaxis(-1, 0)[c] for i in self.data_display for c in i.shape[-1]]
        else:
            self.data_display = data.to_numpy().squeeze()
            if split_channel:
                self.multiple = False
                self.data_display = [i for i in self.data_display.moveaxis(-1, 0)]
            elif data.channel_num != 3:
                self.multiple = True
                self.data_display = [i for i in self.data_display.moveaxis(-1, 0)]
            else:
                self.multiple = False


    def __call__(self):
        if not self.multiple:
            pass







