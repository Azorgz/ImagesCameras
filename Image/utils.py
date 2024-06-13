import time
from typing import Union

import PIL.Image
import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .base import ImageSize, Channel, ColorSpace, PixelFormat, Modality, Batch, Dims


def time_fct(func, reps=1, exclude_first=False):
    def wrapper(*args, **kwargs):
        if exclude_first:
            start = time.time()
            res = func(*args, **kwargs)
            first = time.time() - start
        start = time.time()
        for i in range(reps):
            res = func(*args, **kwargs)
        timed = time.time() - start
        print("------------------------------------ TIME FUNCTION ---------------------------------------------")
        try:
            print(
                f"Function {func.__name__} executed  {reps} times in : {timed} seconds, average = {timed / reps} seconds"
                f"{f', first occurence : {first}' if exclude_first else ''}")
        except AttributeError:
            print(
                f"\nFunction {func.__class__.__name__} executed  {reps} times in : {timed} seconds, average = {timed / reps} seconds"
                f"{f', first occurence : {first}' if exclude_first else ''}")
        print("------------------------------------------------------------------------------------------------")
        return res

    return wrapper


def in_place_fct(obj: Tensor, inplace: bool):
    if inplace:
        return obj
    else:
        return obj.clone()


def wrap_colorspace(wrapper, fct):
    def wrapper_fct(im, **kwargs):
        wrapper(im, **kwargs)
        fct(im, **kwargs)
        return im

    return wrapper_fct


def find_class(args, class_name):
    arg = None
    for idx, a in enumerate(args):
        if isinstance(a, class_name):
            return a
        elif isinstance(a, list) or isinstance(a, tuple):
            arg = find_class(a, class_name)
    return arg


def update_channel_pos(im):
    shape = np.array(im.shape)
    channel_pos = np.argwhere(shape == 3)
    channel_pos = channel_pos[0][0] if len(channel_pos >= 1) else \
        (np.argwhere(shape == 1)[0][0] if len(np.argwhere(shape == 1)) >= 1 else None)
    if channel_pos is None:
        return -1
    else:
        return int(channel_pos)


def pil_to_numpy(im):
    """
    Converts a PIL Image object to a NumPy array.
    Source : Fast import of Pillow images to NumPy / OpenCV arrays Written by Alex Karpinsky

    Args:
        im (PIL.Image.Image): The input PIL Image object.

    Returns:
        numpy.ndarray: The NumPy array representing the image.
    """
    im.load()

    # Unpack data
    e = Image._getencoder(im.mode, "raw", im.mode)
    e.setimage(im.im)

    # NumPy buffer for the result
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))

    bufsize, s, offset = 65536, 0, 0
    while not s:
        l, s, d = e.encode(bufsize)

        mem[offset:offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError("encoder error %d in tobytes" % s)
    return data


def find_best_grid(param):
    srt = int(np.floor(np.sqrt(param)))
    i = 0
    while srt * (srt + i) < param:
        i += 1
    return srt, srt + i


def CHECK_IMAGE_SHAPE(im: Union[np.ndarray, Tensor, PIL.Image.Image], batched=False, permute=False):
    """
    Return first a boolean to indicate whether the image shape is valid or not
    Return the image with channels at the right positions,
    the ImageSize, the Channel and the Batch
    :param permute: Weither if the image has to be permuted has b -c -h -w or not
    :param batched: to specify if the given Tensor is batched or not
    :param im: image to check
    :return: channels order
    """
    if isinstance(im, Image.Image):
        im = pil_to_numpy(im)
    if isinstance(im, np.ndarray):
        im = torch.from_numpy(im)
    if batched:
        batched_im = im.clone()
        im = im[0]
    b = -1
    im = im.squeeze()
    valid = True
    if len(im.shape) > 4:
        return False

    elif len(im.shape) == 4:
        if batched:
            return False
        b_ = 1
        b = 0
        # Batch of images
        im0 = im[0].clone()
    else:
        im0 = im.clone()
        b_ = 0

    if len(im0.shape) == 3:
        # Color image or batch of monochrome images
        c, h, w = im0.shape
        if c == 3:
            # Channel first
            c, h, w = 0 + b_, 1 + b_, 2 + b_
        elif w == 3:
            # Channel last
            c, h, w = 2 + b_, 0 + b_, 1 + b_
        elif c == 4:
            # Channel first, alpha coeff
            c, h, w = 0 + b_, 1 + b_, 2 + b_
        elif w == 4:
            # Channel last, alpha coeff
            c, h, w = 2 + b_, 0 + b_, 1 + b_
        else:
            # batch of monochrome images or batch of multimodal images
            # Channel first
            c, h, w = 0 + b_, 1 + b_, 2 + b_

    elif len(im0.shape) == 2:
        # Monochrome image
        b, c, h, w = -2, -1, 0, 1
    else:
        # Unknown image shape
        return False
    im = batched_im.movedim(0, -1).squeeze() if batched else im
    while len(im.shape) < 4:
        im = im.unsqueeze(-1)
    b, c, h, w = b % 4, c % 4, h % 4, w % 4
    if permute:
        im = im.permute(b, c, h, w)
        dims = Dims()
    else:
        dims = Dims(b, c, h, w)
    return (valid,
            im,
            dims,
            ImageSize(im.shape[dims.height], im.shape[dims.width]),
            Channel(dims.channels, im.shape[dims.channels]),
            Batch(im.shape[dims.batch] > 1, im.shape[dims.batch]))


def CHECK_IMAGE_FORMAT(im, colorspace, dims, channel_names=None, scale=True):
    # Depth format
    if im.dtype == torch.uint8:
        bit_depth = 8
        im = im / 1. if scale else im / 255
    elif im.dtype == torch.uint16:
        bit_depth = 16
        # Promotion not implemented as torch 2.3.0 for uint16, uint32, uint64
        im = im / (256 ** 2 - 1)
        im = im * (256 ** 2 - 1) if scale else im
    elif im.dtype == torch.float32 or im.dtype == torch.uint32:
        bit_depth = 32
        if im.dtype == torch.uint32:
            im = im / (256 ** 4 - 1)
            im = im * (256 ** 4 - 1) if scale else im
    elif im.dtype == torch.float64:
        bit_depth = 64
    elif im.dtype == torch.bool:
        bit_depth = 1
    else:
        raise NotImplementedError

    # Colorspace check
    c = im.shape[dims.channels]
    if colorspace is None:
        if bit_depth == 1:
            # BINARY MODE, ANY MODALITY
            colorspace = ColorSpace(1)
            if c == 1:
                modality = 'Any'
            elif c in (3, 4):
                modality = 'Visible'
            else:
                modality = 'Multimodal'
            channel_names = ['Mask']*c if channel_names is None else channel_names
        elif c == 1:
            # GRAY MODE, ANY MODALITY
            colorspace = ColorSpace(2)
            modality = 'Any'
            channel_names = ['Any'] if channel_names is None else channel_names

        elif c == 3:
            # RGB MODE, VISIBLE MODALITY
            colorspace = ColorSpace(3)
            modality = 'Visible'
            channel_names = ['Red', 'Green', 'Blue'] if channel_names is None else channel_names
        elif c == 4:
            # RGBA MODE, VISIBLE MODALITY
            colorspace = ColorSpace(4)
            modality = 'Visible'
            channel_names = ['Red', 'Green', 'Blue', 'Alpha'] if channel_names is None else channel_names

        else:
            # UNKNOWN MODE, MULTIMODAL MODALITY
            colorspace = ColorSpace(0)
            modality = 'Multimodal'
            channel_names = ['Any'] * c if channel_names is None else channel_names

    else:
        if bit_depth == 1:
            # BINARY MODE
            modality = 'Any'
        elif c == 1:
            # GRAY MODE
            modality = 'Any'
        elif c == 3:
            # RGB MODE
            modality = 'Visible'
        elif c == 4:
            # RGBA MODE
            modality = 'Visible'
        else:
            # UNKNOWN MODE
            modality = 'Multimodal'
    return im, PixelFormat(colorspace, bit_depth), Modality(modality), channel_names
