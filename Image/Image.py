from __future__ import annotations

# --------- Import dependencies -------------------------------- #
import math
import warnings
from itertools import cycle
from os.path import *
from typing import Union, Iterable

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from functorch.einops import rearrange
from kornia.enhance import equalize, equalize_clahe
from matplotlib import pyplot as plt, patches
from matplotlib.pyplot import ion, subplot2grid
from matplotlib.widgets import Slider
from torch import Tensor, _C
from torch.overrides import get_default_nowrap_functions

# --------- Import local classes -------------------------------- #
from .base import Modality, ImageLayout, mode_list
from .colorspace import colorspace_fct
from .encoder import Encoder, Decoder
from .histogram import image_histogram
from .utils import find_best_grid, CHECK_IMAGE_SHAPE, CHECK_IMAGE_FORMAT, in_place_fct, find_class, switch_colormap, \
    draw_rectangle, color_tensor

matplotlib.use('TkAgg')

__version__ = '1.0'


class ImageTensor(Tensor):
    """
    A class defining the general basic framework of a TensorImage.
    The modality per default is VIS (visible light), but can be multimodal or LWIR, SWIR...
    It can use all the methods from Torch plus some new ones.
    To create a new instance:
    --> From file (a str or path pointing towards an image file)
    --> From numpy array (shape h, w, c / h, w, c, a or h, w in case of mono-layered image)
    --> From a torch tensor (shape c, h, w  or b, c, h, w for batched tensor)
    An instance is created using a numpy array or a path to an image file or a PIL image
    """
    _image_layout: ImageLayout = None
    _mode_list = mode_list
    _name: str = None

    # _modality: str = None
    # _name: str = 'new image'
    # _im_pad: tuple = None
    # _colorspace: str = None
    # _pixel_depth: int = None
    # _channel_pos: int | Tensor = None
    # _channel_num: int | Tensor = None
    # _channelNames: list = None
    # _layer_name: list = None
    # _batched: bool = False

    def __init__(self, *args, **kwargs):
        super(ImageTensor, self).__init__()

    # ------- Instance creation methods ---------------------------- #
    @staticmethod
    def __new__(cls, inp, *args,
                name: str = None,
                device: torch.device = None,
                modality: str = None,
                colorspace: str = None,
                channel_names=None,
                batched=None,
                permute_image=True,
                normalize=True,
                **kwargs):
        # Input array is a path to an image OR an already formed ndarray instance
        if isinstance(inp, str):
            name = basename(inp).split('.')[0] if name is None else name
            d = Decoder(inp)
            inp, batched = d.value, d.batched
            permute_image = True  # The image has been created from a path
        if isinstance(inp, ImageTensor) or isinstance(inp, DepthTensor):
            if isinstance(inp, DepthTensor):
                normalize = False
            inp_ = inp.to_tensor()
            image_layout = inp.image_layout.clone()
            name = str(inp.name)
        elif isinstance(inp, np.ndarray) or isinstance(inp, Tensor):
            valid, inp_, dims, image_size, channel, batch = CHECK_IMAGE_SHAPE(inp, batched, permute_image)
            if colorspace is not None:
                colorspace = int(np.argwhere(mode_list == colorspace)[0][0])
            inp_, pixelformat, mod, channel_names = CHECK_IMAGE_FORMAT(inp_, colorspace, dims,
                                                                       channel_names=channel_names)
            modality = mod if modality is None else Modality(modality)
            image_layout = ImageLayout(modality, image_size, channel, pixelformat, batch, dims,
                                       channel_names=channel_names)
        else:
            raise NotImplementedError

        if isinstance(device, torch.device):
            inp_ = inp_.to(device)
        if normalize and inp_.max() > 1:
            if image_layout.pixel_format.bit_depth == 8:
                inp_ /= 255
            elif image_layout.pixel_format.bit_depth == 16:
                inp_ /= 65535
            else:
                inp_ = (inp_ - inp_.min()) / (inp_.max() - inp_.min())

        image = super().__new__(cls, inp_)
        # add the new attributes to the created instance of Image
        image._image_layout = image_layout
        image._name = name
        return image

    @classmethod
    def rand(cls, *args, batch: int | list = 1, channel: int = 3, height: int = 100, width: int = 100,
             depth: int | str = 32) -> ImageTensor:
        """
        Instance creation of Random images
        :param batch: batch dimension
        :param channel: channel dimension (1 : Any modality, 3 or 4 : RGB-A, other: Multimodal)
        :param height: height of the ImageTensor
        :param width: width of the ImageTensor
        :param depth: depth of the pixel (32 or 64 bits)
        :return: A new instance of randomized ImageTensor
        """
        dtype_dict = {str(32): torch.float32, str(64): torch.float64}
        assert str(depth) in dtype_dict, 'depth must be either 32 or 64 bits'
        dtype = dtype_dict[str(depth)]

        if len(args) > 0:
            if len(args) == 1:
                batch = args[0]
            elif len(args) == 2:
                height, width = args
                channel = 1
            elif len(args) == 3:
                channel, channel, height = args
            elif len(args) == 4:
                batch, channel, height, width = args
            else:
                *batch, channel, height, width = args
        if isinstance(batch, int):
            batch = [batch]
        assert all([b >= 1 for b in batch])
        batched = sum(batch) > 1
        assert channel >= 1 and height >= 1 and width >= 1
        return cls(torch.rand([*batch, channel, height, width], dtype=dtype),
                   name='Random Image', batched=batched, permute_image=True)

    @classmethod
    def randint(cls, *args, batch: int | list = 1, channel: int = 3, height: int = 100, width: int = 100,
                depth: int | str = 8, low=0, high=None) -> ImageTensor:
        """
        Instance creation of Random images
        :param high: Upper bound of the created instance
        :param low: Lower bound of the created instance
        :param batch: batch dimension
        :param channel: channel dimension (1 : Any modality, 3 or 4 : RGB-A, other: Multimodal)
        :param height: height of the ImageTensor
        :param width: width of the ImageTensor
        :param depth: depth of the pixel (8, 16 or 32 bits)
        :return: A new instance of randomized ImageTensor
        """
        dtype_dict = {str(8): torch.uint8, str(16): torch.uint16, str(32): torch.uint32}
        assert str(depth) in dtype_dict, 'depth must be either 8, 16 or 32 bits'
        dtype = dtype_dict[str(depth)]
        if dtype == torch.uint8:
            high_ = 255
        elif dtype == torch.uint16:
            high_ = 65535
        elif dtype == torch.uint32:
            high_ = 4294967295
        else:
            high_ = 18446744073709551615
        high = min(high, high_) if high is not None else high_
        if len(args) > 0:
            if len(args) == 1:
                batch = args[0]
            elif len(args) == 2:
                height, width = args
                channel = 1
            elif len(args) == 3:
                channel, channel, height = args
            elif len(args) == 4:
                batch, channel, height, width = args
            else:
                *batch, channel, height, width = args
        if isinstance(batch, int):
            batch = [batch]
        assert all([b >= 1 for b in batch])
        batched = sum(batch) > 1
        assert channel >= 1 and height >= 1 and width >= 1
        return cls(torch.randint(low, high, [*batch, channel, height, width], dtype=dtype),
                   name='Random Image', batched=batched, permute_image=True, normalize=False)

    # ------- utils methods ---------------------------- #
    # ------- Torch function call method ---------------------------- #

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        # res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if kwargs is None:
            kwargs = {}
        with _C.DisableTorchFunctionSubclass():
            res = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return res
            elif res.__class__ is not Tensor:
                return res
            else:
                arg = find_class(args, cls)
                if arg is not None:
                    if arg.shape == res.shape:
                        new = cls(res)
                        new.name = arg.name
                        new.permute(arg.layers_name, in_place=True)
                        new.image_layout.update(pad=(*arg.image_layout.pad.to_list(), arg.image_layout.pad.mode))
                        new.image_layout.colormap = arg.image_layout.colormap
                        return new
                    else:
                        return res
                else:
                    return res

    # ------- clone methods ---------------------------- #
    def pass_attr(self, image, *args):
        if len(args) > 0:
            for arg in args:
                if arg == 'image_layout':
                    self.__dict__[arg] = image.__dict__[arg].clone()
                self.__dict__[arg] = image.__dict__[arg].copy()
        else:
            self.image_layout = image.image_layout.clone()
            self.name = image.name

    def clone(self, *args) -> ImageTensor:
        """
        Function to clone an ImageTensor
        :param image: ImageTensor to clone
        :return: cloned ImageTensor
        """
        new = self.__class__(self, normalize=False)
        return new

    def to(self, dst, *args, in_place=False, **kwargs):
        out = in_place_fct(self, in_place)
        out.data = self.to_tensor().to(dst, *args, **kwargs)
        return out

    # ------- basic methods ---------------------------- #
    # def __eq__(self, other):
    #     if isinstance(other, ImageTensor):
    #         eq = True
    #         if torch.sum(Tensor(self.data) - Tensor(other.data)) != 0:
    #             eq = False
    #         elif self.image_layout != other.image_layout:
    #             eq = False
    #         return eq
    #     else:
    #         return Tensor(self.data) == other

    def __str__(self):
        return str(self.to_tensor())

    def __lt__(self, other):
        return self.BINARY(other, method='lt')

    def __le__(self, other):
        return self.BINARY(other, method='le')

    def __eq__(self, other):
        return self.BINARY(other, method='eq')

    def __ne__(self, other):
        return self.BINARY(other, method='ne')

    def __ge__(self, other):
        return self.BINARY(other, method='ge')

    def __gt__(self, other):
        return self.BINARY(other, method='gt')

    def __mul__(self, other):
        out = self.clone()
        with _C.DisableTorchFunctionSubclass():
            out.data = torch.mul(self, other)
        return out

    def __add__(self, other):
        out = self.clone()
        with _C.DisableTorchFunctionSubclass():
            out.data = torch.add(self, other)
        return out

    def __div__(self, other):
        out = self.clone()
        with _C.DisableTorchFunctionSubclass():
            out.data = torch.div(self, other)
        return out

    def __sub__(self, other):
        out = self.clone()
        with _C.DisableTorchFunctionSubclass():
            out.data = torch.sub(self, other)
        return out

    def pprint(self):
        print(self.image_layout)

    def hist(self, density=False, weight=None):
        return image_histogram(self, density, weight)

    # -------  Image manipulation methods || Size changes  ---------------------------- #
    @torch.no_grad()
    def batch(self, images: Iterable or ImageTensor, *args, in_place=False):
        """
        Function to batch ImageTensor together
        :param in_place: If True the current instance of Imagetensor is modified, otherwise a new one is returned
        :param images: list of ImageTensor to batch
        :return: batched ImageTensor
        NO GRAD
        """
        if len(args) > 0:
            images = [images, *args]
        if isinstance(images, ImageTensor):
            images = [images]
        assert isinstance(images, list)
        assert len(images) > 0
        batch = self.batch_size
        for i, im in enumerate(images):
            assert isinstance(im, ImageTensor), 'Only ImageTensors are supported'
            if im.shape != self.shape:
                images[i] = im.match_shape(self).to_tensor()
            else:
                images[i] = im.to_tensor()
            batch += im.batch_size
        out = in_place_fct(self, in_place)
        out.data = torch.concatenate([self.to_tensor(), *images], dim=0)
        out.batch_size = batch
        if not in_place:
            return out

    def stack(self, *args, dim: int | str = 0, in_place: bool = False):
        """
        Function to stack ImageTensor together. It's a redirection of other stacking fct in order to preserve the
        dimensions of the initial ImageTensor
        :param in_place: bool wether to create a new instance or not
        :param iter_tensor: list of ImageTensor to stack
        :param dim: dimension along which to stack
        :return: stacked ImageTensor
        """
        # if the list to stack is within self:
        if isinstance(self, list):
            out = in_place_fct(self[0], in_place)
            if len(self) == 1:
                return self[0]
            else:
                iter_tensor = [*self[1:]]
        elif len(args) > 0:
            out = in_place_fct(self, in_place)
            iter_tensor = [*args]
        else:
            out = in_place_fct(self, in_place)
            return out
        if dim == 'b' or dim == 'batch' or dim == out.image_layout.dims.dims[0]:
            fct = out.batch
            out.batch_size = out.batch_size + sum([im.batch_size for im in iter_tensor])
        elif dim == 'c' or dim == 'channel' or dim == out.image_layout.dims.dims[1]:
            raise NotImplementedError
        elif dim == 'h' or dim == 'height' or dim == out.image_layout.dims.dims[2]:
            fct = out.vstack
            out.image_size = out.image_size[0] + sum([im.image_size[0] for im in iter_tensor]), self.image_size[1]
        elif dim == 'w' or dim == 'width' or dim == out.image_layout.dims.dims[3]:
            fct = out.hstack
            out.image_size = out.image_size[0], out.image_size[1] + sum([im.image_size[1] for im in iter_tensor])
        else:
            raise NotImplementedError
        out.data = fct(iter_tensor, in_place=False)

        if not in_place:
            return out

    def pad(self, size, in_place=False, value: float = 0, mode: str = 'constant', **kwargs):
        '''
        Pad the image to match the given Tensor/Array size or with the list of padding indicated (left, right, top, bottom)
        Use the same optional parameter than torch Pad
        :param value: fill value for 'constant' padding. Default: 0
        :param mode: 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        :param in_place: if in_place is True the current instance of ImageTensor will be modified
        :param size: image to replicate on size or Iterable with either 2 values : (height, width) or 4 values : (left, right, top, bottom)
        :return: a copy of self (or self if in_place) but padded
        '''
        # Selection of self or creation of a new instance according the bool "in_place" parameter
        out = in_place_fct(self, in_place)
        # Save the current layer order and reset it
        layers = out.layers_name
        out.reset_layers_order(in_place=True)

        # Determine the padding. Either it's given, either we compute it from the target image size
        if isinstance(size, list) or isinstance(size, tuple):
            assert len(size) == 2 or len(size) == 4
            if len(size) == 2:
                pad_l, pad_r = int(size[1]), int(size[1])
                pad_t, pad_b = int(size[0]), int(size[0])
            elif len(size) == 4:
                pad_l, pad_r, pad_t, pad_b = int(size[0]), int(size[1]), int(size[2]), int(size[3])
            else:
                pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0
            pad_tuple = (pad_l, pad_r, pad_t, pad_b)
        else:
            if isinstance(size, ImageTensor):
                h, w = size.image_size
            elif isinstance(size, Tensor):
                h, w = size.shape[-2:]
            elif isinstance(size, np.ndarray):
                h, w = size.shape[0], size.shape[1]
            else:
                h, w = 0, 0
            h_ref, w_ref = self.image_size
            try:
                assert w >= w_ref and h >= h_ref
            except AssertionError:
                return self.clone()
            pad_l = int(math.ceil((w - w_ref) / 2))
            pad_r = int(math.floor((w - w_ref) / 2))
            pad_t = int(math.ceil((h - h_ref) / 2))
            pad_b = int(math.floor((h - h_ref) / 2))
            pad_tuple = (pad_l, pad_r, pad_t, pad_b)
        # Pad the output and update its layout

        data = F.pad(out.to_tensor(), pad_tuple, value=value, mode=mode)
        out.data = data
        out.image_layout.update(pad=(*pad_tuple, mode), height=out.shape[-2], width=out.shape[-1])
        out.permute(layers)

        if not in_place:
            return out

    def unpad(self, in_place=False):
        """
        Crop back the image to its original size, removing the padding
        """
        out = in_place_fct(self, in_place)
        if out.image_layout.pad.to_list() != [0, 0, 0, 0]:
            layers = out.layers_name
            pad = out.image_layout.pad
            out.reset_layers_order(in_place=True)
            out.data = out[:, :, pad.top:self.image_size[0] - pad.bottom, pad.left:self.image_size[1] - pad.right]
            out.image_layout.update(pad=(0, 0, 0, 0, 'constant'), height=out.shape[-2], width=out.shape[-1])
            out.permute(layers, in_place=True)
        if not in_place:
            return out

    def hstack(self, *args, in_place=False, **kwargs):
        assert all([im.image_size[0] == self.image_size[0] for im in args])
        layers = self.layers_name
        out = in_place_fct(self, in_place).permute(['h', 'w', 'b', 'c'])
        stack = [out.to_tensor()]
        for im in args:
            stack.append(im.permute(['h', 'w', 'b', 'c']).to_tensor())
        out.data = torch.hstack(stack)
        out.reset_layers_order(in_place=True)
        out.image_size = out.shape[-2:]
        out.permute(layers, in_place=True)
        if not in_place:
            return out

    def vstack(self, *args, in_place=False, **kwargs):
        assert all([im.image_size[1] == self.image_size[1] for im in args])
        layers = self.layers_name
        out = in_place_fct(self, in_place).permute(['h', 'w', 'b', 'c'])
        stack = [out.to_tensor()]
        for im in args:
            stack.append(im.permute(['h', 'w', 'b', 'c']).to_tensor())
        out.data = torch.vstack(stack)
        out.reset_layers_order(in_place=True)
        out.image_layout.update(image_size=out.shape[-2:])
        out.permute(layers, in_place=True)
        if not in_place:
            return out

    def pyrDown(self, in_place=False, **kwargs):
        layers = self.layers_name
        out = in_place_fct(self, in_place).reset_layers_order(in_place=False)
        # downsample
        out.data = F.interpolate(out,
                                 scale_factor=1 / 2,
                                 mode='bilinear',
                                 align_corners=True)

        out.image_size = out.shape[-2:]
        out.permute(layers)
        if not in_place:
            return out

    def pyrUp(self, in_place=False, **kwargs):
        layers = self.layers_name
        out = in_place_fct(self, in_place).reset_layers_order(in_place=False)
        # upsample
        out.data = F.interpolate(out,
                                 scale_factor=2,
                                 mode='bilinear',
                                 align_corners=True)

        out.image_size = out.shape[-2:]
        out.permute(layers)
        if not in_place:
            return out

    def resize(self, shape, keep_ratio=False, in_place=False, **kwargs):
        layers = self.layers_name
        out = in_place_fct(self, in_place).reset_layers_order(in_place=False)
        if keep_ratio:
            ratio = torch.tensor(self.image_size) / torch.tensor(shape)
            ratio = ratio.max()
            out.data = F.interpolate(out.to_tensor(), mode='bilinear', scale_factor=float(1 / ratio))
            out.image_size = out.shape[-2:]
        else:
            out.data = F.interpolate(out.to_tensor(), size=shape, mode='bilinear', align_corners=True)
            out.image_size = out.shape[-2:]
        out.permute(layers, in_place=True)
        if not in_place:
            return out

    def squeeze(self, *args, **kwargs):
        return self.to_tensor().squeeze(*args, **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return self.to_tensor().unsqueeze(*args, **kwargs)

    def match_shape(self, other: Union[Tensor, tuple, list], keep_ratio=False, in_place=False, **kwargs):
        """
        Take as input either a Tensor based object to match on size or
        an Iterable describing the new size to get
        :param in_place: If True modify the current instance
        :param other: Tensor like or Iterable
        :param keep_ratio: to match on size while keeping the original ratio
        :return: ImageTensor
        """
        layers = self.layers_name
        out = in_place_fct(self, in_place)
        out.reset_layers_order()
        if isinstance(other, tuple) or isinstance(other, list):
            shape = other
            assert len(other) == 2
        elif isinstance(other, ImageTensor):  # or isinstance(other, DepthTensor):
            shape = other.image_size
        else:
            shape = other.shape[-2:]
        if keep_ratio:
            ratio = torch.tensor(self.image_size) / torch.tensor(shape)
            ratio = ratio.max()
            out.data = F.interpolate(out.to_tensor(), mode='bilinear', scale_factor=float((1 / ratio).cpu().numpy()))
            out.image_size = out.shape[-2:]
            out.pad(other, in_place=True)
        else:
            out.data = F.interpolate(out.to_tensor(), size=shape, mode='bilinear', align_corners=True)
        out.image_size = out.shape[-2:]
        out.permute(layers, in_place=True)
        if not in_place:
            return out

    def crop(self, crop: Iterable, center: bool = False, xyxy=True):
        """
        Crop the image following the top-left / height / width norm
        :param crop: coordinates xy of the reference point, height and width (x, y, h, w)
        :param center: If True, crop the image around the center
        :param xyxy: If True, crop according to xyxy coordinates, otherwise it will be considered as (x, y, w, h)
        """
        if not isinstance(crop, Iterable) or len(crop) != 4:
            raise ValueError("Crop coordinates should be provided as (x, y, h, w)")
        y, x, w, h = crop
        if xyxy:
            y, x, y1, x1 = crop
            h, w = x1 - x, y1 - y
        if center:
            x = x - h // 2
            y = y - w // 2
        out = self.to_tensor()
        x, y, h, w = int(x), int(y), int(h), int(w)
        return ImageTensor(out[:, :, x:x + h, y:y + w])

    def apply_patch(self, patch: Tensor, anchor: tuple,
                    center: bool = False,
                    shape: float | tuple = None,
                    zeros_as_transparent: bool = True,
                    in_place: bool = False):
        """
        Apply a patch to the image tensor at the given anchor
        :param patch: tensor of patch data
        :param anchor: coordinates (x, y) of the anchor point (top-left corner)
        :param center: If True, apply the patch around the anchor
        :param shape: Shape of the patch (h, w) if not provided, it will be inferred from the patch
        :param zeros_as_transparent: If True, replace zeros in the patch with pixel from the image
        :param in_place: If True modify the current instance
        """
        layers = self.layers_name
        out = in_place_fct(self, in_place)
        out.reset_layers_order()
        x, y = anchor
        if shape is not None:
            if isinstance(shape, tuple):
                patch = F.interpolate(patch, shape)
            elif isinstance(shape, float):
                patch = F.interpolate(patch, scale_factor=shape)
            else:
                raise NotImplementedError
        h, w = patch.shape[-2:]
        if center:
            x = x - h // 2
            y = y - w // 2
        assert all((patch.shape[0] == self.shape[0],
                    patch.shape[1] == self.shape[1])), 'The given patch does not fit the image batch and channel'
        assert ((0 <= x < self.image_size[0]) or (0 <= y < self.image_size[1]) or
                (0 <= x + h < self.image_size[0])) or (
                       0 <= y + w < self.image_size[1]), 'No pixel are overwrite with this patch'
        # pad the image if necessary
        pad = [0, 0, 0, 0]
        pad[2] = -x if x < 0 else 0
        pad[3] = x + h - out.image_size[0] if x + h > out.image_size[0] else 0
        pad[0] = -y if y < 0 else 0
        pad[1] = y + w - out.image_size[1] if y + w > out.image_size[1] else 0
        out.pad(pad, in_place=True)
        x = x + pad[2]
        y = y + pad[0]
        if zeros_as_transparent:
            out[:, :, x:x + h, y:y + w] *= (patch == 0) * 1.
            out[:, :, x:x + h, y:y + w] += patch
        else:
            out[:, :, x:x + h, y:y + w] = patch
        out.permute(layers, in_place=True)
        out.unpad()
        if not in_place:
            return out

    # -------  Layers manipulation methods  ---------------------------- #
    def reset_layers_order(self, in_place: bool = False):
        if in_place:
            self.permute(self.image_layout.dims.dims, in_place=True)
        else:
            return self.permute(self.image_layout.dims.dims, in_place=False)

    def put_channel_at(self, idx=1, in_place: bool = False):
        out = in_place_fct(self, in_place)
        if idx == out.channel_pos:
            return out
        out.data = torch.movedim(out.to_tensor(), self.channel_pos, idx)
        dims = np.array(self.image_layout.dims.dims)
        d = dims[self.channel_pos]
        dims = np.delete(dims, self.channel_pos)
        dims = np.insert(dims, idx, d)
        out.image_layout.update(pos=idx, dims=dims)
        if not in_place:
            return out

    def permute(self, dims: Iterable | int | str, *args, in_place: bool = False):
        """
        Similar to permute torch function but with Layers tracking.
        Work with a list of : layer indexes, layer names ('batch', 'height', 'width', 'channels')
        :param in_place: bool to state if a new tensor has to be generated or not
        :param dims: List of new dimension order (length = 4)
        :return: ImageTensor or Nothing
        """
        if isinstance(dims, int) or dims in ['b', 'c', 'h', 'w', 'batch', 'channels', 'height', 'width']:
            assert len(args) + 1 == 4, "4 indexes are needed for permutation"
            dims = [dims, *args]
        elif isinstance(dims, Iterable):
            assert len(dims) == 4, "4 indexes are needed for permutation"
        dims = dims.copy()
        if any(isinstance(d, str) for d in dims):
            layers = np.array(self.layers_name)
            for idx, d in enumerate(dims):
                if isinstance(d, str):
                    if d == 'batch' or d == 'b':
                        dims[idx] = int(np.argwhere(layers == 'batch'))
                    elif d == 'height' or d == 'h':
                        dims[idx] = int(np.argwhere(layers == 'height'))
                    elif d == 'width' or d == 'w':
                        dims[idx] = int(np.argwhere(layers == 'width'))
                    elif d == 'channels' or d == 'c' or d == 'channel':
                        dims[idx] = int(np.argwhere(layers == 'channels'))
                    else:
                        raise ValueError(f'Unknown dimension {d}')
        assert len(np.unique(dims)) == len(dims), 'Dimension position must be unique (/permute)'
        new_channel_pos = int(np.argwhere((np.array(dims) == self.channel_pos)))
        out = in_place_fct(self, in_place)
        out.data = torch.permute(out.to_tensor(), dims)
        out.image_layout.update(pos=new_channel_pos, dims=dims)
        if not in_place:
            return out

    def extract_from_batch(self, idx):
        if not self.batched:
            return self
        elif idx <= self.batch_size:
            layers = self.layers_name
            batch_split = self.reset_layers_order(in_place=False)
            if isinstance(idx, list):
                batch_split = ImageTensor(batch_split[idx], batched=True, permute_image=False)
            else:
                batch_split = ImageTensor(batch_split[idx].unsqueeze(0), permute_image=False)
            batch_split.permute(layers, in_place=True)
            batch_split.pass_attr(self)
            batch_split.image_layout.update(batch_size=batch_split.shape[0])
            return batch_split
        else:
            raise IndexError('Batch index out of range')

    # -------  Value manipulation methods  ---------------------------- #
    def histo_equalization(self, mini=0, maxi=0, in_place=False, filtering=True, clahe=False):
        out = in_place_fct(self, in_place)
        if clahe:
            out.data = equalize_clahe(out)
        else:
            if mini == 0 and maxi == 0:
                hist = out.hist()
                mini, maxi = hist.clip()
            out.data = out.clip(mini, maxi)
            out.normalize(in_place=True)
            if filtering:
                out.data = equalize(out) / 2 + out / 2
        return out

    @torch.no_grad()
    def interpolate(self, *args):
        """
        :param items:  List/array or Tensor of shape (N,2) of tuple of coordinate to interpolate
        :return: Tensor of interpolated values
        """
        items = Tensor(args)
        while len(items.shape) < 4:
            items = items.unsqueeze(0)
        try:
            device = items.device
        except AttributeError:
            device = 'cpu'
        h, w = self.image_size
        N = self.batch_size
        grid = items.repeat(N, 1, 1, 1)
        grid[:, :, :, 0] = grid[:, :, :, 0] * 2 / w - 1
        grid[:, :, :, 1] = grid[:, :, :, 1] * 2 / h - 1
        return F.grid_sample(self.to_tensor().to(device), grid, align_corners=True).squeeze().T

    def normalize(self, return_minmax=False, keep_abs_max=False, in_place=False, **kwargs):
        out = in_place_fct(self, in_place)
        if keep_abs_max:
            a = torch.abs(out.to_tensor())
            m = a.min()
            M = a.max()
            out.data = (a - m) / (M - m)
        else:
            m = out.min()
            M = out.max()
            out.data = (out.to_tensor() - m) / (M - m)
        if return_minmax:
            return out, m, M
        else:
            return out

    # -------  type conversion methods ---------------------------- #
    def to_opencv(self, datatype=None):
        """
        :return: np.ndarray
        """
        if self.colorspace == 'RGB':
            out = self.permute(['b', 'h', 'w', 'c']).to_numpy(datatype=datatype)[..., [2, 1, 0]].squeeze()

        elif self.colorspace == 'RGBA':
            out = self.permute(['b', 'h', 'w', 'c']).to_numpy(datatype=datatype)[..., [2, 1, 0, 3]].squeeze()
        else:
            out = self.permute(['b', 'h', 'w', 'c']).to_numpy(datatype=datatype).squeeze()
        return np.ascontiguousarray(out)

    def to_numpy(self, datatype=None):
        """
        :param datatype: np.uint8, np.uint16, np.float32, np.float64
        :return: np.ndarray
        """
        if datatype is None:
            if self.depth == 8:
                datatype = np.uint8
            elif self.depth == 16:
                datatype = np.uint16
            elif self.depth == 32:
                datatype = np.float32
            else:
                datatype = np.float64
        if datatype == np.uint8:
            if self.min() < 0 or self.max() > 255:
                out = (self.normalize().to_tensor() * 255).to(torch.uint8)
            elif self.max() <= 1:
                out = (self.to_tensor() * 255).to(torch.uint8)
            else:
                out = self.to_tensor().to(torch.uint8)
        elif datatype == np.uint16:
            if self.min() < 0 or self.max() > 65535:
                out = (self.normalize().to_tensor() * 65535).to(torch.uint16)
            elif self.max() <= 1:
                out = (self.to_tensor() * 65535).to(torch.uint16)
            else:
                out = self.to_tensor().to(torch.uint16)
        else:
            out = self.to_tensor()
        if self.requires_grad:
            numpy_array = np.ascontiguousarray(Tensor.numpy(out.detach().cpu()), dtype=datatype)
        else:
            numpy_array = np.ascontiguousarray(Tensor.numpy(out.cpu()), dtype=datatype)
        return numpy_array

    def to_tensor(self):
        """
        Remove all attributes to keep only the data as a torch tensor.
        :return: Tensor
        """
        with _C.DisableTorchFunctionSubclass():
            out = torch.Tensor(self)
        return out

    # -------  Drawings methods  ---------------------------- #
    def draw_rectangle(self, pts: list = None, roi: list = None, color=None, fill=None, width: int = 3, in_place=False):
        """
        Draw a rectangle in the image using the homonym kornia fct with as entry either the top-left/bottom-right pts coordinates
        or a ROI with left/right/top/bottom margin
        """
        out = in_place_fct(self, in_place)
        assert pts is not None or roi is not None, 'Either a ROI or top-left/bottom-right points are needed'
        if roi is not None:
            roi = Tensor(roi).squeeze()
            assert roi.shape[-1] == 4, '4 points are needed for a ROI'
            if len(roi.shape) == 1:
                roi = roi.unsqueeze(0)
            pts = [[roi[i][0], roi[i][2], roi[i][1], roi[i][3]] for i in range(roi.shape[0])]
        if not isinstance(pts, Tensor):
            pts = Tensor(pts).repeat([out.batch_size, 1, 1])
        if color is None:
            if self.channel_num == 3:
                list_color = cycle(['red', 'green', 'blue', 'yellow', 'chartreuse', 'violet', 'aqua'])
                color = []
                for c, _ in zip(list_color, range(pts.shape[1])):
                    color.append(color_tensor(c).repeat([out.batch_size, 1, 1]))
                color = torch.cat(color, dim=1)
            else:
                color = Tensor([1])
        out.data = draw_rectangle(out, pts, color, fill, width=width)
        if not in_place:
            return out

    # -------  Display Methods ------------------------------------------------- #

    @torch.no_grad()
    def show(self,
             num: str | None = None,
             cmap: str = 'gray',
             roi: list = None,
             point: Union[list, Tensor] = None,
             save: str = '',
             split_batch: bool = False,
             split_channel: bool = False,
             opencv: bool = False):
        split_channel = split_channel and self.channel_num > 1
        # If the ImageTensor is multimodal or batched then we will plot a matrix of images for each mod / image
        if self.modality == 'Multimodal' or self.batch_size > 1:
            if not opencv:
                self._multiple_show_matplot(num=num, cmap=cmap, split_batch=split_batch, split_channel=split_channel)
            else:
                self._multiple_show_opencv(num=num, cmap=cmap, split_batch=split_batch, split_channel=split_channel)

        # Else we will plot a Grayscale image or a ColorImage
        else:
            num = self.name if num is None else num
            channels_names = self.channel_names if self.channel_names else np.arange(0, self.channel_num).tolist()
            if split_channel:
                im_display = self.permute(['b', 'c', 'h', 'w']).to_numpy().squeeze()
                fig, axe = plt.subplots(1, 1, num=num)
                plt.subplots_adjust(left=0.15)
                # Make a vertical slider to control the channel.
                axe_channel = plt.axes((0.03, 0.05, 0.05, 0.8))
                channel_slider = Slider(
                    ax=axe_channel,
                    label='Channel of image',
                    valmin=0,
                    valmax=self.channel_num - 1,
                    valstep=1,
                    valinit=0,
                    orientation="vertical")

                def update(i):
                    match self.colorspace:
                        case 'RGB':
                            cmap_ = ['Reds', 'Greens', 'Blues'][int(channel_slider.val)]
                        case _:
                            cmap_ = None if self.p_modality != 'Any' else cmap
                    axe.imshow(im_display[int(i)], cmap_)
                    axe.set_title(f"Channel {channels_names[int(channel_slider.val)]}")
                    if point is not None:
                        for center in point.squeeze():
                            center = center.cpu().long().numpy()
                            circle = patches.Circle(center, 5, linewidth=2, edgecolor='r', facecolor='none')
                            axe.add_patch(circle)
                    if roi is not None:
                        for r, color in zip(roi, ['r', 'g', 'b']):
                            rect = patches.Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2]
                                                     , linewidth=2, edgecolor=color, facecolor='none')
                            axe.add_patch(rect)
                    axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    plt.show()

                channel_slider.on_changed(update)
                update(0)
            else:
                im_display = self.permute(['b', 'h', 'w', 'c']).to_numpy().squeeze()
                fig, axe = plt.subplots(ncols=1, nrows=1, num=num, squeeze=False)
                axe.imshow(im_display, cmap=None if self.p_modality != 'Any' else cmap)
                if point is not None:
                    for center in point.squeeze():
                        center = center.cpu().long().numpy()
                        circle = patches.Circle(center, 5, linewidth=2, edgecolor='r', facecolor='none')
                        axe.add_patch(circle)
                if roi is not None:
                    for r, color in zip(roi, ['r', 'g', 'b']):
                        rect = patches.Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2]
                                                 , linewidth=2, edgecolor=color, facecolor='none')
                        axe.add_patch(rect)
                axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                if save:
                    fig.savefig(f'{save}.png', bbox_inches='tight', dpi=300)
                plt.show()
            return axe

    @torch.no_grad()
    def _multiple_show_matplot(self,
                               num: str = None,
                               cmap: str = 'gray',
                               split_batch: bool = False,
                               split_channel: bool = False):
        num = self.name if not num else num
        channels_names = self.channel_names if self.channel_names else np.arange(0, self.channel_num).tolist()

        if split_batch and split_channel:
            im_display = self.permute(['b', 'c', 'h', 'w']).to_numpy().squeeze()
            fig, axes = plt.subplots(1, 1, num=num)
            plt.subplots_adjust(left=0.15, bottom=0.15)
            # Make a horizontal slider to control the batch.
            axe_batch = plt.axes((0.15, 0.05, 0.75, 0.05))
            batch_slider = Slider(
                ax=axe_batch,
                label='Batch number',
                valmin=0,
                valmax=self.batch_size - 1,
                valstep=1,
                valinit=0)
            # Make a vertical slider to control the channel.
            axe_channel = plt.axes((0.03, 0.15, 0.05, 0.8))
            channel_slider = Slider(
                ax=axe_channel,
                label='Channel',
                valmin=0,
                valmax=self.channel_num - 1,
                valstep=1,
                valinit=0,
                orientation="vertical")

            def update(i):
                match self.colorspace:
                    case 'RGB':
                        cmap_ = ['Reds', 'Greens', 'Blues'][int(channel_slider.val)]
                    case _:
                        cmap_ = None if self.p_modality != 'Any' else cmap
                axes.imshow(im_display[int(batch_slider.val), int(channel_slider.val)], cmap_)
                axes.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axes.set_title(f" Image {i} from batch, Channel {channels_names[int(channel_slider.val)]}")
                plt.show()

            batch_slider.on_changed(update)
            channel_slider.on_changed(update)
            update(0)
        elif split_batch:
            im_display = self.permute(['b', 'h', 'w', 'c']).to_numpy().squeeze()
            if self.channel_num == 3 or self.channel_num == 1:
                fig, axes = plt.subplots(1, 1, num=num)
                im_display = im_display[:, None]
            else:
                rows, cols = find_best_grid(self.channel_num)
                fig, axes = plt.subplots(rows, cols, num=num)
                axes = axes.flatten()
                im_display = im_display.moveaxis(-1, 1)
            plt.subplots_adjust(bottom=0.15)
            # Make a horizontal slider to control the batch.
            axe_batch = plt.axes((0.15, 0.05, 0.75, 0.05))
            batch_slider = Slider(
                ax=axe_batch,
                label='Batch number',
                valmin=0,
                valmax=self.batch_size - 1,
                valstep=1,
                valinit=0)

            def update(i):
                for j, axe in enumerate(axes):
                    if j < im_display.shape[1]:
                        axe.imshow(im_display[int(batch_slider.val), j], cmap)
                        axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    else:
                        if axe.axes is not None:
                            axe.remove()
                fig.suptitle(f" Image {i} from batch, Channel {channels_names[int(channel_slider.val)]}")
                plt.show()

            batch_slider.on_changed(update)
            update(0)
        elif split_channel:
            im_display = self.permute(['b', 'c', 'h', 'w']).to_numpy().squeeze()
            rows, cols = find_best_grid(self.batch_size)
            fig, axes = plt.subplots(rows, cols, num=num)
            axes = axes.flatten()
            plt.subplots_adjust(left=0.15)
            # Make a vertical slider to control the channel.
            axe_channel = plt.axes((0.03, 0.05, 0.05, 0.8))
            channel_slider = Slider(
                ax=axe_channel,
                label='Channel',
                valmin=0,
                valmax=self.channel_num - 1,
                valstep=1,
                valinit=0,
                orientation="vertical")


            def update(i):
                match self.colorspace:
                    case 'RGB':
                        cmap_ = ['Reds', 'Greens', 'Blues'][int(channel_slider.val)]
                    case _:
                        cmap_ = None if self.p_modality != 'Any' else cmap
                for j, axe in enumerate(axes):
                    if j < im_display.shape[0]:
                        axe.imshow(im_display[j, int(i)], cmap_)
                        axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    else:
                        if axe.axes is not None:
                            axe.remove()
                fig.suptitle(f"Channel {channels_names[int(channel_slider.val)]}")
                plt.show()

            channel_slider.on_changed(update)
            update(0)
        else:
            im_display = rearrange(self.permute(['b', 'c', 'h', 'w']), 'b c h w -> (b c) h w').detach().cpu().numpy()
            rows, cols = find_best_grid(self.batch_size * self.channel_num)
            fig, axes = plt.subplots(rows, cols, num=num)
            axes = axes.flatten()
            for j, axe in enumerate(axes):
                if j < im_display.shape[0]:
                    axe.imshow(im_display[j], cmap)
                    axe.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                else:
                    if axe.axes is not None:
                        axe.remove()
            plt.show()

        return fig, axes

    # -------  Data inspection and storage methods  ---------------------------- #

    def save(self, path, name=None, ext=None, keep_colorspace=False, depth=None, **kwargs):
        encod = Encoder(self.depth if depth is None else depth, self.modality, self.batched)
        encod(self, path, name=name, keep_colorspace=keep_colorspace)

    # ---------------- Properties -------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name

    @property
    def image_layout(self) -> ImageLayout:
        return self._image_layout

    @image_layout.setter
    def image_layout(self, value) -> None:
        self._image_layout = value

    # ---------------- Inherited from layout -------------------------------- #
    @property
    def im_shape(self) -> Tensor:
        assert self.image_layout.shape == self.shape
        return self.image_layout.shape

    @property
    def batched(self) -> bool:
        return self.image_layout.batch.batched

    @property
    def batch_size(self) -> int:
        return self.image_layout.batch.batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.image_layout.update(batch_size=value)

    @property
    def image_size(self) -> tuple:
        return self.image_layout.image_size.height, self.image_layout.image_size.width

    @image_size.setter
    def image_size(self, value):
        self.image_layout.update(image_size=value)

    @property
    def im_pad(self):
        return {'left': self.image_layout.pad.left,
                'right': self.image_layout.pad.right,
                'top': self.image_layout.pad.top,
                'bottom': self.image_layout.pad.bottom,
                'mode': self.image_layout.pad.mode}

    @im_pad.setter
    def im_pad(self, pad) -> None:
        self.image_layout.update(pad=pad)

    @property
    def depth(self) -> int:
        return self.image_layout.pixel_format.bit_depth

    @depth.setter
    def depth(self, value) -> None:
        self.image_layout.update(bit_depth=value)

    @property
    def modality(self) -> str:
        return self.image_layout.modality

    @property
    def p_modality(self) -> str:
        return self.image_layout.private_modality

    @property
    def mode_list(self) -> list:
        return self._mode_list

    @property
    def layers_name(self) -> list:
        return self.image_layout.dims.layers

    @layers_name.setter
    def layers_name(self, dims) -> None:
        self.image_layout.update(dims=dims)

    @property
    def channel_pos(self) -> int:
        return self.image_layout.channel.pos

    @channel_pos.setter
    def channel_pos(self, pos) -> None:
        self.image_layout.update(pos=pos if pos >= 0 else pos + self.ndim)

    @property
    def channel_names(self) -> list:
        return self.image_layout.channel_names

    @channel_names.setter
    def channel_names(self, names) -> None:
        self.image_layout.update(channel_names=names)

    @property
    def channel_num(self) -> int:
        return self.image_layout.channel.num_ch

    @channel_num.setter
    def channel_num(self, num) -> None:
        self.image_layout.update(num_ch=num)

    @property
    def colormap(self):
        return self.image_layout.colormap

    @colormap.setter
    def colormap(self, value) -> None:
        self.image_layout.update(colormap=value)

    @property
    def colorspace(self) -> str:
        return self.image_layout.pixel_format.colorspace.name

    @colorspace.setter
    def colorspace(self, v) -> None:
        """
        :param c_mode: str following the Modes of a Pillow Image
        :param colormap: to convert a GRAYSCALE image to a Palette (=colormap) colored image
        """
        if self.modality == 'Multimodal':
            warnings.warn('Multimodal Images cannot get a colorspace')
            return
        if isinstance(v, list) or isinstance(v, tuple):
            colorspace = v[0]
            colormap = v[1]['colormap'] if v[1] else None
        else:
            colorspace = v
            colormap = None
        if colorspace == self.colorspace:
            if self.colormap == colormap or colormap is None:
                return
            else:
                switch_colormap(self, colormap, in_place=True)
        else:
            colorspace_change_fct = colorspace_fct(f'{self.colorspace}_to_{colorspace}')
            if self.batched:
                im = []
                for i in range(self.batch_size):
                    batch_split = self.extract_from_batch(i)
                    colorspace_change_fct(batch_split, colormap=colormap)
                    im.append(batch_split)
                self.data = torch.cat(im, dim=int(np.argwhere(np.array(self.layers_name) == 'batch')))
                self.image_layout.update(colorspace=im[0].colorspace,
                                         num_ch=im[0].channel_num,
                                         colormap=im[0].colormap,
                                         channel_names=im[0].channel_names,
                                         modality=im[0].modality)
            else:
                colorspace_change_fct(self, colormap=colormap)

    # ---------------- Colorspace change functions -------------------------------- #
    def RGB(self, cmap=None):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'rgb' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'RGB', {'colormap': cmap}
        return im

    def RGBA(self, cmap='gray', **kwargs):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'rgba' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'RGBA', {'colormap': cmap}
        return im

    def GRAY(self):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'gray' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'GRAY', {}
        return im

    def CMYK(self, cmap=None, **kwargs):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'cmyk' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'CMYK', {'colormap': cmap}
        return im

    def YCrCb(self, cmap=None, **kwargs):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'YCrCb' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'YCrCb', {'colormap': cmap}
        return im

    def LAB(self, cmap=None):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'lab' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'LAB', {'colormap': cmap}
        return im

    def LUV(self, cmap=None):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'luv' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'LUV', {'colormap': cmap}
        return im

    def HSV(self, cmap=None, **kwargs):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'hsv' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'HSV', {'colormap': cmap}
        return im

    def HLS(self, cmap=None, **kwargs):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'hls' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'HLS', {'colormap': cmap}
        return im

    def XYZ(self, cmap=None, **kwargs):
        """
        Implementation equivalent at the attribute setting : im.colorspace = 'xyz' but create a new ImageTensor
        """
        im = self.clone()
        im.colorspace = 'XYZ', {'colormap': cmap}
        return im

    def BINARY(self, threshold=0.5, method='gt', keepchannel=True):
        """
        Implementation equivalent at the attribute setting : im.colorspace = '1' but create a new ImageTensor
        """
        methods = {'lt': torch.lt, 'le': torch.le,
                   'eq': torch.eq, 'ne': torch.ne,
                   'ge': torch.ge, 'gt': torch.gt}
        func = methods[method]
        im = ImageTensor(self)
        im.pass_attr(self)
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if isinstance(threshold, Tensor) and len(threshold.shape) == 2:
            batch = im.batch_size
            assert threshold.shape == im.image_size
            threshold = threshold.repeat([batch, 1, 1, 1])
        with _C.DisableTorchFunctionSubclass():
            im.data = func(im if keepchannel else im.sum(dim=1, keepdim=True), threshold)
        im.image_layout.update(colorspace='BINARY', bit_depth=1)
        im.permute(layers, in_place=True)
        return im


class DepthTensor(ImageTensor):
    """
    A SubClass of Image Tensor to deal with Disparity/Depth value > 1.
    If the Tensor is modified, the maximum value always be referenced
    The DepthTensor takes metrics values in between 0 and 1000 meters.
    The Tensor will be encoded in uint16, allowing a good precision (1/66 meters)
    """
    _max_value = 0
    _min_value = 0
    _scaled = False
    _scale_factor = 65.535

    def __init__(self, *args, **kwargs):
        super(DepthTensor, self).__init__()

    # ------- Instance creation methods ---------------------------- #
    @staticmethod
    def __new__(cls, inp: Union[Tensor, np.ndarray, str], *args,
                name: str = None,
                device: torch.device = None,
                batched=False,
                scaled=False,
                permute_image=True,
                **kwargs):
        inp_str = isinstance(inp, str)
        inp = super().__new__(cls, inp, device=device, batched=batched, name=name, modality='Depth',
                              channels_name=['Depth'], normalize=False, permute_image=permute_image)
        assert inp.channel_num == 1, 'Depth Tensor must have at most one channel'
        if not inp_str:
            inp.depth = 16
            try:
                if not scaled:
                    assert inp.max() <= 1000, 'A Depth Tensor must be contained in 0-1000 meters'
            except AssertionError:
                inp.data = torch.clamp(inp.to_tensor(), 0, 1000)
        else:
            assert inp.depth == 16, 'A Depth Tensor must be saved in a 16 bit file format'
            inp /= 65.535
        inp.scaled = scaled
        if scaled:
            inp.max_value = inp.max() / 65.535
            inp.min_value = inp.min() / 65.535
        else:
            inp.max_value = inp.max()
            inp.min_value = inp.min()
        return inp

    @classmethod
    def rand(cls, batch: int = 1, height: int = 100, width: int = 100, low=0, high=1000, **kwargs) -> ImageTensor:
        """
        Instance creation of Random images
        :param batch: batch dimension
        :param height: height of the ImageTensor
        :param width: width of the ImageTensor
        :param depth: depth of the pixel (16, 32 or 64 bits)
        :return: A new instance of randomized ImageTensor
        """
        device = kwargs['device'] if 'device' in kwargs else None
        assert batch >= 1 and height >= 1 and width >= 1
        high = min(high, 1000) if high is not None else 1000
        return cls(torch.randint(0, high * 65, [batch, 1, height, width], dtype=torch.uint16) / 65,
                   name='Random Depth', batched=batch > 1, device=device)

    @classmethod
    def randint(cls, batch: int = 1, height: int = 100, width: int = 100, scaled=True, low=0, high=1000,
                **kwargs) -> DepthTensor:
        """
        Instance creation of Random images
        :param high: Upper bound of the created instance
        :param low: Lower bound of the created instance
        :param scaled: If True the returned ImageTensor will take integer values
        :param batch: batch dimension
        :param height: height of the ImageTensor
        :param width: width of the ImageTensor
        """
        device = kwargs['device'] if 'device' in kwargs else None
        assert batch >= 1 and height >= 1 and width >= 1
        high = min(high, 1000) if high is not None else 1000
        return cls(torch.randint(0, high + 1, [batch, 1, height, width], dtype=torch.uint16),
                   name='Random Depth', batched=batch > 1, device=device)

    # ------- clone methods ---------------------------- #

    def clone(self):
        new = self.__class__(self, name=self.name, scaled=self.scaled, batched=self.batched, device=self.device)
        return new

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # print(f"Calling '{func.__name__}' for Subclass")
        # res = super().__torch_function__(func, types, args=args, kwargs=kwargs)
        if kwargs is None:
            kwargs = {}
        with _C.DisableTorchFunctionSubclass():
            res = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return res
            elif res.__class__ is not Tensor:
                return res
            else:
                arg = find_class(args, cls)
                if arg is not None:
                    if arg.shape == res.shape and arg.dtype == res.dtype:
                        new = cls(res, batched=arg.batched)
                        new.name = arg.name
                        new.permute(arg.layers_name, in_place=True)
                        return new
                    else:
                        return res
                else:
                    return res

    def save(self, path, name=None, save_image=False, **kwargs):
        encod = Encoder(self.depth, self.modality, self.batched, ext='tiff')
        if save_image:
            ImageTensor(self.inverse_depth(remove_zeros=True)
                        .normalize()).RGB().save(path, name=name)
        else:
            name = self.name if name is None else name

            im_to_save = self.clamp(0, 1000).scale()
            encod(im_to_save, path, name=name, keep_colorspace=False)

    def clamp(self, mini=None, maxi=None, *, out=None, in_place=False) -> DepthTensor or None:
        out = in_place_fct(self, in_place)
        out.max_value = min(maxi, out.max_value)
        out.min_value = max(mini, out.min_value)
        out.data = torch.clamp(out.to_tensor(), min=mini, max=maxi)
        if not in_place:
            return out

    def show(self, num=None, cmap='jet', roi: list = None, point: Union[list, Tensor] = None, true_value=False):
        matplotlib.use('TkAgg')
        """
        :param num: Number of images to show
        :param cmap: Colormap to use
        :param roi: Region of Interest
        :param point: Point to show
        """
        if true_value and self.max_value <= 255:
            im = ImageTensor(self.unscale(), normalize=False)
            im.depth = 8
            im.show(num=num, cmap=f'{cmap}', roi=roi, point=point)

        else:
            im = ImageTensor(self.inverse_depth(remove_zeros=True))
            im.show(num=num, cmap=f'{cmap}', roi=roi, point=point)

    def inverse_depth(self, remove_zeros=False, remove_max=True, factor=1, in_place=False, **kwargs):
        out = in_place_fct(self, in_place)
        if remove_zeros:
            out[out == 0] = out.max()
        if remove_max:
            out[out == out.max()] = out.min()
        out.data = factor / torch.log(out.to_tensor() + 10)
        out.normalize(in_place=True)
        if not in_place:
            return out

    # -------------------- Properties related to DepthTensor only -------------------- #
    @property
    def scale_factor(self):
        return self._scale_factor

    @property
    def scaled(self):
        return self._scaled

    @scaled.setter
    def scaled(self, value):
        self._scaled = value

    def scale(self):
        new = self.clone()
        if not new.scaled:
            new.scaled = True
            return new * self.scale_factor  # (new.max_value - new.min_value) + new.min_value
        else:
            return new

    def unscale(self):
        new = self.clone()
        if new.scaled:
            new.scaled = False
            return new / self.scale_factor  # (new - new.min_value) / (new.max_value - new.min_value)
        else:
            return new

    @property
    def colorspace(self) -> str:
        return self.image_layout.pixel_format.colorspace.name

    @colorspace.setter
    def colorspace(self, v) -> None:
        """
        :param c_mode: str following the Modes of a Pillow Image
        :param colormap: to convert a GRAYSCALE image to a Palette (=colormap) colored image
        """
        if isinstance(v, list) or isinstance(v, tuple):
            colorspace = v[0]
            colormap = v[1]['colormap'] if v[1] else 'inferno'
        else:
            colorspace = v
            colormap = 'inferno'
        if colorspace == self.colorspace:
            return
        elif self.colorspace == 'BINARY':
            warnings.warn("The Mask image can't be colored")
            return
        else:
            if colorspace not in ['RGB', 'GRAY', 'BINARY']:
                warnings.warn("Those colorspaces are not implemented for DepthTensor")
                return
            colorspace_change_fct = colorspace_fct(f'{self.colorspace}_to_{colorspace}')
            colorspace_change_fct(self, colormap=colormap)

    @property
    def max_value(self):
        return self._max_value

    @max_value.setter
    def max_value(self, v):
        self._max_value = v

    @property
    def min_value(self):
        return self._min_value

    @min_value.setter
    def min_value(self, v):
        self._min_value = v
