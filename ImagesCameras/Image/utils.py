from typing import Union
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from matplotlib.colors import CSS4_COLORS as css_color
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


def color_tensor(name) -> Tensor:
    assert name in css_color, 'This color does not exist'
    string_color = css_color[name][1:]
    r = int(string_color[:2], 16) / 255
    g = int(string_color[2:4], 16) / 255
    b = int(string_color[4:], 16) / 255
    return torch.tensor([r, g, b])


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
                top = int(rectangle[b, n, 1])
                right = int(rectangle[b, n, 2])
                bottom = int(rectangle[b, n, 3])
                # Vertical left
                out[b, :, top - width:bottom + width + 1, left - width: left + width + 1] = color[b, n, :, None, None]
                # Vertical right
                out[b, :, top - width:bottom + width + 1, right - width: right + width + 1] = color[b, n, :, None, None]
                # Horizontal top
                out[b, :, top - width:top + width + 1, left - width: right + width + 1] = color[b, n, :, None, None]
                # Horizontal bottom
                out[b, :, bottom - width:bottom + width + 1, left - width: right + width + 1] = color[b, n, :, None,
                                                                                                None]
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


def CHECK_IMAGE_SHAPE(im: Union[np.ndarray, Tensor, Image.Image], batched: bool | None = False, permute=False):
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
    im_ = im.squeeze()
    # GRAYSCALE IMAGE
    if im_.ndim == 2:
        valid = True
        dims = Dims(0, 1, 2, 3)
        im = im_[None, None]
        batch = Batch([1])

    # COLOR IMAGE or BATCHED GRAYSCALE IMAGE
    elif im_.ndim == 3:
        # The BATCH DIMS WILL ALWAYS BE THE 1ST ONE
        if batched:
            valid = True
            dims = Dims(0, 1, 2, 3)
            im = im_[:, None]
            batch = Batch([im_.shape[0]])
        # The IMAGE IS NOT BATCHED, CHANNEL WILL BE THE DIM=3/4 or the 1ST ONE
        elif batched is not None:
            dim_list = [0, 1, 2]
            # Color or multispectral image
            s = torch.tensor(im_.shape)
            if 3 in s:
                c = dim_list.pop(np.argwhere(s == 3)[0][0])
                h, w = dim_list
            else:
                c, h, w = dim_list
            valid = True
            dims = Dims(0, c + 1, h + 1, w + 1)
            im = im_[None]
            batch = Batch([1])
        # The IMAGE MAY BE BATCHED, CHANNEL WILL BE THE DIM=3/4 or THE IMAGE IS BATCHED
        else:
            dim_list = [0, 1, 2]
            if im.ndim > 3:
                s = torch.tensor(im.shape[-3:])
                if 1 in s:
                    # THE IMAGE IS BATCHED
                    c = dim_list.pop(np.argwhere(s == 1)[0][0])
                    h, w = dim_list
                    valid = True
                    dims = Dims(0, c + 1, h + 1, w + 1)
                    batch = Batch([im.shape[:-3]])
                    im = im_[:, None]
                else:
                    # THE IMAGE IS NOT BATCHED
                    valid = True
                    dims = Dims(0, 1, 2, 3)
                    im = im_[None]
                    batch = Batch([1])
            # THE IMAGE IS NOT BATCHED
            else:
                # Color image possible
                s = torch.tensor(im_.shape)
                if 3 in s:
                    c = dim_list.pop(np.argwhere(s == 3)[0][0])
                    h, w = dim_list
                    valid = True
                    dims = Dims(0, c + 1, h + 1, w + 1)
                    im = im_[None]
                    batch = Batch([1])
                # Image is batched
                else:
                    valid = True
                    dims = Dims(0, 1, 2, 3)
                    im = im_[None]
                    batch = Batch([1])

    # THE IMAGE IS BATCHED
    elif im_.ndim == 4:
        dim_list = [0, 1, 2, 3]
        # s = torch.tensor(im_.shape)
        # # Color image
        # if 3 in s:
        #     c = dim_list.pop(np.argwhere(s == 3)[0][0])
        #     b, h, w = dim_list
        # else:
        b, c, h, w = dim_list
        valid = True
        dims = Dims(b, c, h, w)
        batch = Batch([im_.shape[b]])
        im = im_
    # THE IMAGE IS BATCHED WITH A BATCH SHAPE
    elif im_.ndim > 4:
        dim_list = np.arange(0, im.ndim).tolist()
        s = torch.tensor(im.shape)
        # # Color image
        # if 3 in s:
        #     c = dim_list.pop(np.argwhere(s == 3)[0][0])
        #     *b, h, w = dim_list
        # else:
        *b, c, h, w = dim_list
        valid = True
        im = im_.flatten(0, len(b) - 1)
        dims = Dims(0, 1, 2, 3)
        batch = Batch(s[b])
    else:
        raise ValueError

    if permute:
        im = im.permute(*dims.dims)
        dims.permute(dims.dims)

    return (valid,
            im,
            dims,
            ImageSize(im.shape[dims.height], im.shape[dims.width]),
            Channel(dims.channels, im.shape[dims.channels]),
            batch)


def CHECK_IMAGE_FORMAT(im, colorspace, dims, channel_names=None, scale=True):
    # Depth format
    im_ = im/1.
    if im.dtype == torch.uint8:
        bit_depth = 8
        im = im_ if scale else im / 255
    elif im.dtype == torch.uint16:
        bit_depth = 16
        # Promotion not implemented as torch 2.3.0 for uint16, uint32, uint64
        im = im_ if scale else im/(256 ** 2 - 1)
    elif im.dtype == torch.float32 or im.dtype == torch.uint32:
        bit_depth = 32
        if im.dtype == torch.uint32:
            im = im_ if scale else im / (256 ** 4 - 1)
        else:
            im = im_
    elif im.dtype == torch.float64:
        bit_depth = 64
        im = im_
    elif im.dtype == torch.bool:
        bit_depth = 1
        im = im_
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
        colorspace = ColorSpace(colorspace)
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
