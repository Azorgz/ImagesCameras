from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from warnings import warn

import numpy as np
import torch
from torch import Tensor

__version__ = '1.0'

"""
Base classes for ImageTensor, largely inspired from Kornia Image files
https://github.com/kornia/kornia/blob/main/kornia/image/base.py
"""

list_modality = ['Any', 'Visible', 'Multimodal', 'Depth']
mode_list = np.array(['UNKNOWN', 'BINARY', 'GRAY', 'RGB', 'RGBA', 'CMYK', 'LAB', 'HSV', 'XYZ'])
mode_dict = {'UNKNOWN': 0, 'BINARY': 1, 'GRAY': 2, 'RGB': 3,
             'RGBA': 4, 'CMYK': 5, 'LAB': 6, 'HSV': 7, 'XYZ': 8}


@dataclass()
class Modality:
    r"""Data class to represent image modality.
    modality, either :
    Visible (3,4 channels)
    Multimodal (2 + channels)
    Any (1 channel lengthwave)
    Depth (1 channel depth values)"""
    _modality: str

    def __init__(self, modality):
        if modality.upper() == 'VIS' or modality.upper() == 'VISIBLE':
            self._modality = 'Visible'
        elif modality.upper() == 'MULTI' or modality.upper() == 'MULTIMODAL':
            self._modality = 'Multimodal'
        elif modality.upper() == 'DEPTH':
            self._modality = 'Depth'
        else:
            self._modality = modality

    @property
    def private_modality(self):
        if self._modality in list_modality:
            return self._modality
        else:
            return 'Any'

    @property
    def modality(self):
        return self._modality


# @dataclass(frozen=True)
# class Modality(Enum):
#     r"""Data class to represent image modality.
#     modality, either :
#     Visible (3,4 channels)
#     Multimodal (2 + channels)
#     Any (1 channel lengthwave)
#     Depth (1 channel depth values)"""
#     Any = 0
#     Visible = 1
#     Multimodal = 2
#     Depth = 3


@dataclass()
class Shape:
    r"""Data class to represent image shape.

    Args:
        batch : batch size
        channels : number of channels
        height: image height.
        width: image width.

    Example:
        >>> shape = Shape(3, 4, 20, 30)
        >>> shape.channel
        4
        >>> shape.width
        30
    """
    batch: int | Tensor
    channel: int | Tensor
    height: int | Tensor
    width: int | Tensor


@dataclass()
class ImageSize:
    r"""Data class to represent image shape.

    Args:
        height: image height.
        width: image width.

    Example:
        >>> size = ImageSize(3, 4)
        >>> size.height
        3
        >>> size.width
        4
    """

    height: int | Tensor
    width: int | Tensor


class ColorSpace(Enum):
    r"""Enum that represents the color space of an image."""

    UNKNOWN = 0  # in case of multi band images
    BINARY = 1  # in case of binary mask images (1, 3, or 4 channel(s))
    GRAY = 2  # in case of grayscale images (1 channel)
    RGB = 3  # in case of color images (3 channels)
    RGBA = 4  # in case of color images (4 channels)
    CMYK = 5  # in case of color images in CMYK mode (4 channels)
    LAB = 6  # in case of color images in LAB mode (3 channels)
    HSV = 7  # in case of color images in HSV mode (3 channels)
    XYZ = 8  # in case of color images in XYZ mode (3 channels)


@dataclass()
class PixelFormat:
    r"""Data class to represent the pixel format of an image.

    Args:
        colorspace: color space.
        bit_depth: the number of bits per channel.

    Example:
        >>> pixel_format = PixelFormat(ColorSpace.RGB, 8)
        >>> pixel_format.colorspace
        <ColorSpace.RGB: 2>
        >>> pixel_format.bit_depth
        8
    """
    colorspace: ColorSpace
    bit_depth: int


@dataclass()
class Channel:
    r"""Enum that represents the channels order of an image."""
    pos: int
    num_ch: int


@dataclass()
class Dims:
    r"""list that represents the dims order of an image."""
    batch: int = 0
    channels: int = 1
    height: int = 2
    width: int = 3

    def __init__(self, *args, batch: int = 0, channels: int = 1, height: int = 2, width: int = 3):
        if len(args) == 4:
            self.batch, self.channels, self.height, self.width = args
        else:
            self.batch, self.channels, self.height, self.width = [batch, channels, height, width]

    def permute(self, dims):
        temp = np.array(dims)
        self.batch, self.channels, self.height, self.width = [int(np.argwhere(d == temp)) for d in self.dims]
        # for i, d in enumerate(dims):
        #     self.pos_layer[d] = i
        #  = np.array(self.dims)[dims].tolist()

    @property
    def dims(self):
        return [self.batch, self.channels, self.height, self.width]

    # @property
    # def dims(self):
    #     temp = np.array(self.pos_layer)
    #     return [int(np.argwhere(temp == i)) for i in range(4)]

    @property
    def layers(self):
        return np.array(['batch', 'channels', 'height', 'width'])[np.argsort(self.dims)].tolist()


@dataclass()
class Batch:
    r"""Class that represents the batch dimension."""
    batched: bool
    batch_size: int


@dataclass()
class Pad:
    r"""Data class to represent image pad.

    Args:
        left: pad left.
        right: pad right.
        top: pad top.
        bottom: pad bottom.

    Example:
        >>> pad = Pad(3, 4, 15, 30)
        >>> pad.left
        3
        >>> pad.bottom
        30
    """
    left: int | Tensor = 0
    right: int | Tensor = 0
    top: int | Tensor = 0
    bottom: int | Tensor = 0
    mode: str = 'constant'


@dataclass()
class ImageLayout:
    """Data class to represent the layout of an image.
    """
    _modality: Modality
    image_size: ImageSize
    channel: Channel
    pixel_format: PixelFormat
    channel_names = ['']
    batch: Batch
    dims: Dims

    def __init__(self, modality: Modality,
                 image_size: ImageSize,
                 channel: Channel,
                 pixel_format: PixelFormat,
                 batch: Batch,
                 dims: Dims = None,
                 pad: Pad = None,
                 colormap: str = None,
                 channel_names: list = None
                 ):
        self._modality = modality
        self.colormap = colormap
        self.image_size = image_size
        self.channel = channel
        self.pixel_format = pixel_format
        self.batch = batch
        if pad is not None:
            self.pad = pad
        else:
            self.pad = Pad
        if dims is not None:
            self.dims = dims
        else:
            self.dims = Dims()
        if channel_names is not None:
            try:
                assert len(channel_names) == channel.num_ch
                self.channel_names = channel_names
            except AssertionError:
                warn("The given channel names don't match the number of channels")
                self.channel_names = None
        else:
            self.channel_names = None

        self._CHECK_MODALITY_VALIDITY()
        self._CHECK_COLOR_VALIDITY()
        self._CHECK_DEPTH_VALIDITY()
        self._CHECK_LAYERS_VALIDITY()
        self._CHECK_BATCH_SIZE()

    def __eq__(self, other):
        return self.private_modality == other.private_modality and \
            self.image_size == other.image_size and \
            self.channel == other.channel and \
            self.pixel_format == other.pixel_format and \
            self.channel_names == other.channel_names and \
            self.batch == other.batch and \
            self.dims == other.dims and \
            self.colormap == other.colormap

    def __str__(self) -> str:
        str_print = f"# --------------------------------- Image Layout -------------------------------- #\n"
        str_print += f"Modality: {self.modality}\n"
        str_print += f"Image size: {self.image_size.height} x {self.image_size.width} (height x width)\n"
        if self.pad.left != 0 or self.pad.right != 0 or self.pad.top != 0 or self.pad.bottom != 0:
            str_print += f"Pad: left : {self.pad.left}px x  right : {self.pad.right}px x top : {self.pad.top}px x bottom : {self.pad.bottom}px  | Mode : {self.pad.mode}\n"
        if self.channel_names is not None:
            str_print += f"Channel names: {' | '.join(self.channel_names)}\n"
        if self.colormap is not None:
            str_print += f"Colormap: {self.colormap}\n"
        str_print += f"Pixel format: {self.pixel_format.colorspace.name} | {self.channel.num_ch} x {self.pixel_format.bit_depth} bits\n"
        if self.batch.batched:
            str_print += f"Batch size: {self.batch.batch_size}\n"
        str_print += f"Layers: {' x '.join(self.dims.layers)} || {' x '.join([str(s) for s in self.shape])}\n"
        str_print += f"# -------------------------------------------------------------------------------- #\n"
        return str_print

    def clone(self):
        return ImageLayout(modality=Modality(self.modality),
                           image_size=ImageSize(self.image_size.height, self.image_size.width),
                           channel=Channel(self.channel.pos, self.channel.num_ch),
                           pixel_format=PixelFormat(self.pixel_format.colorspace, self.pixel_format.bit_depth),
                           channel_names=self.channel_names,
                           pad=Pad(self.pad.left, self.pad.right, self.pad.top, self.pad.bottom),
                           batch=Batch(self.batch.batched, self.batch.batch_size),
                           dims=Dims(*self.dims.dims),
                           colormap=self.colormap)

    def _CHECK_MODALITY_VALIDITY(self):
        if self.modality == 'Visible':
            assert self.channel.num_ch in [3, 4]
        elif self.modality == 'Multimodal':
            assert self.channel.num_ch > 1
        else:
            assert self.channel.num_ch == 1 or (self.channel.num_ch == 3 and self.colormap is not None)

    def _CHECK_LAYERS_VALIDITY(self):
        assert self.dims.batch != self.dims.channels != self.dims.height != self.dims.width
        assert self.dims.channels == self.channel.pos

    def _CHECK_COLOR_VALIDITY(self):
        cs = self.pixel_format.colorspace.name
        if cs == "GRAY":
            assert self.channel.num_ch == 1 and self.colormap is None
        elif cs in ['RGB', 'HSV', 'XYZ', 'LAB']:
            assert self.channel.num_ch == 3
        elif cs in ['RGBA', 'CMYK']:
            assert self.channel.num_ch == 4

    def _CHECK_DEPTH_VALIDITY(self):
        if self.pixel_format.colorspace.name == 'BINARY':
            assert self.pixel_format.bit_depth == 1
        else:
            assert self.pixel_format.bit_depth in [8, 16, 32, 64]

    def _CHECK_BATCH_SIZE(self):
        if self.batch.batched:
            assert self.batch.batch_size > 1

    def update(self, **kwargs):
        verif = []
        if 'height' in kwargs or 'width' in kwargs or 'image_size' in kwargs:
            h = kwargs['height'] if 'height' in kwargs else self.image_size.height
            w = kwargs['width'] if 'width' in kwargs else self.image_size.width
            h, w = (kwargs['image_size'][0], kwargs['image_size'][1]) if 'image_size' in kwargs else (h, w)
            self._update_image_size(height=h, width=w)
        if 'pos' in kwargs or 'num_ch' in kwargs:
            pos = kwargs['pos'] if 'pos' in kwargs else self.channel.pos
            num_ch = kwargs['num_ch'] if 'num_ch' in kwargs else self.channel.num_ch
            verif.extend(self._update_channel(pos=pos, num_ch=num_ch))
        if 'colorspace' in kwargs or 'bit_depth' in kwargs:
            cs = kwargs['colorspace'] if 'colorspace' in kwargs else self.pixel_format.colorspace
            if not isinstance(cs, ColorSpace):
                assert cs in mode_dict, 'The required colospace is not implemented (/update layout)'
                cs = ColorSpace(mode_dict[cs])
            bd = kwargs['bit_depth'] if 'bit_depth' in kwargs else self.pixel_format.bit_depth
            verif.extend(self._update_pixel_format(colorspace=cs, bit_depth=bd))
        if 'batch_size' in kwargs:
            batched = kwargs['batch_size'] > 1
            batch_size = kwargs['batch_size']
            verif.extend(self._update_batch(batched=batched, batch_size=batch_size))
        if 'channel_names' in kwargs:
            self._update_channel_names(kwargs['channel_names'])
        if 'dims' in kwargs:
            self._update_dims(kwargs['dims'])
        if 'modality' in kwargs:
            verif.extend(self._update_modality(kwargs['modality']))
        if 'colormap' in kwargs:
            verif.extend(self._update_colormap(kwargs['colormap']))
        if 'pad' in kwargs:
            self._update_pad(kwargs['pad'])
        if len(verif) > 0:
            verif_done = []
            for v in verif:
                if v not in verif_done:
                    v()
                    verif_done.append(v)

    def _update_pad(self, pad):
        p = self.pad
        p.left, p.right, p.top, p.bottom, p.mode = pad
        self._update_image_size(height=pad[2] + pad[3] + self.image_size.height,
                                width=pad[0] + pad[1] + self.image_size.width)

    def _update_colormap(self, colormap):
        self.colormap = colormap
        return [self._CHECK_COLOR_VALIDITY,
                self._CHECK_MODALITY_VALIDITY]

    def _update_modality(self, modality):
        if self.modality == 'Depth':
            return []
        self._modality = Modality(modality)
        return [self._CHECK_MODALITY_VALIDITY]

    def _update_channel(self, **kwargs):
        self.channel = Channel(**kwargs)

        return [self._CHECK_LAYERS_VALIDITY,
                self._CHECK_COLOR_VALIDITY]

    def _update_dims(self, dims):
        try:
            assert len(dims) == 4
            self.dims.permute(dims)
        except AssertionError:
            warn("The given channel dims don't match the number of dimensions")

    def _update_channel_names(self, names):
        try:
            assert len(names) == self.channel.num_ch
            self.channel_names = names
        except AssertionError:
            warn("The given channel names don't match the number of channels")
            self.channel_names = None

    def _update_image_size(self, **kwargs):
        self.image_size = ImageSize(**kwargs)

    def _update_pixel_format(self, **kwargs):
        self.pixel_format = PixelFormat(**kwargs)
        return [self._CHECK_COLOR_VALIDITY,
                self._CHECK_DEPTH_VALIDITY]

    def _update_batch(self, **kwargs):
        self.batch = Batch(**kwargs)
        return [self._CHECK_BATCH_SIZE]

    @property
    def shape(self):
        size = np.array(self.dims.dims)
        sorted_idx = np.argsort(size)
        return torch.Size(np.array([self.batch.batch_size,
                                    self.channel.num_ch,
                                    self.image_size.height,
                                    self.image_size.width])[sorted_idx])

    @property
    def private_modality(self):
        return self._modality.private_modality

    @property
    def modality(self):
        return self._modality.modality
