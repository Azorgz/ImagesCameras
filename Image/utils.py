from typing import Union
import PIL.Image
import numpy as np
import torch
from PIL import Image
from torch import Tensor

# --------- Import local classes -------------------------------- #
from .base import ImageSize, Channel, ColorSpace, PixelFormat, Modality, Batch, Dims


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


def retrieve_grayscale_from_colormap(im, **kwargs):
    assert im.colormap is not None
    return im.LAB()[:, :1, :, :]


def switch_colormap(im, colormap, in_place=True, **kwargs):
    im = in_place_fct(im, in_place)
    if im.colormap == colormap or im.colormap is None:
        return im
    else:
        temp = im.__class__(retrieve_grayscale_from_colormap(im)).RGB(colormap)
        temp.colorspace = im.colorspace, {'colormap': colormap}
        im.data = temp
        im.image_layout.update(colormap=colormap)
        return im


def draw_rectangle(
        image: torch.Tensor,
        rectangle: torch.Tensor,
        color: [torch.Tensor] = None,
        fill: [bool] = None,
        width: int = 1,
        in_place=False) -> torch.Tensor:
    r"""Draws N rectangles on a batch of image tensors.

        Args:
            image (torch.Tensor): is tensor of BxCxHxW.
            rectangle (torch.Tensor): represents number of rectangles to draw in BxNx4
                N is the number of boxes to draw per batch index[x1, y1, x2, y2]
                4 is in (top_left.x, top_left.y, bot_right.x, bot_right.y).
            color (torch.Tensor, optional): a size 1, size 3, BxNx1, or BxNx3 tensor.
                If C is 3, and color is 1 channel it will be broadcasted Default: None (black).
            fill (bool, optional): is a flag used to fill the boxes with color if True. Default: False.
            width (int): The line width. Default: 1. (Not implemented yet).
        Returns:
            torch.Tensor: This operation modifies image inplace but also returns the drawn tensor for
            convenience with same shape the of the input BxCxHxW.
        """
    batch, c, h, w = image.shape
    batch_rect, num_rectangle, num_points = rectangle.shape
    assert batch == batch_rect, "Image batch and rectangle batch must be equal"
    assert num_points == 4, "Number of points in rectangle must be 4"

    # clone rectangle, in case it's been expanded assignment from clipping causes problems
    rectangle = rectangle.long().clone()

    width = width // 2
    # clip rectangle to hxw bounds
    rectangle[:, :, 1::2] = torch.clamp(rectangle[:, :, 1::2], width, h - (1 + width))
    rectangle[:, :, ::2] = torch.clamp(rectangle[:, :, ::2], width, w - (1 + width))

    if color is None:
        color = torch.tensor([0.0] * c).expand(batch, num_rectangle, c)

    if fill is None:
        fill = False

    if len(color.shape) == 1:
        color = color.expand(batch, num_rectangle, c)

    b, n, color_channels = color.shape

    if color_channels == 1 and c == 3:
        color = color.expand(batch, num_rectangle, c)
    out = in_place_fct(image, in_place)
    for b in range(batch):
        for n in range(num_rectangle):
            if fill:
                out[b, :, int(rectangle[b, n, 1]):int(rectangle[b, n, 3] + 1),
                int(rectangle[b, n, 0]):int(rectangle[b, n, 2] + 1)] = color[b, n, :, None, None]
            else:
                left = int(rectangle[b, n, 0])
                top = int(rectangle[b, n, 1]) - width
                right = int(rectangle[b, n, 2] + 1) + width
                bottom = int(rectangle[b, n, 3] + 1) + width
                # Vertical left
                out[b, :, top - width:bottom + width, left - width: left + width] = color[b, n, :]
                # Vertical right
                out[b, :, top - width:bottom + width, right - width: right + width] = color[b, n, :]
                # Horizontal top
                out[b, :, top - width:top + width, left - width: right + width] = color[b, n, :]
                # Horizontal bottom
                out[b, :, bottom - width:bottom + width, left - width: right + width] = color[b, n, :]
    return out


def find_class(args, class_name):
    arg = None
    for idx, a in enumerate(args):
        if isinstance(a, class_name):
            return a
        elif isinstance(a, list) or isinstance(a, tuple):
            arg = find_class(a, class_name)
    return arg


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
            channel_names = ['Mask'] * c if channel_names is None else channel_names
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
