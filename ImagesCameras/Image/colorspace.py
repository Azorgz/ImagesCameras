import numpy as np
import torch
from kornia.color import *
from matplotlib import colormaps as cm
from torch import Tensor

from .utils import wrap_colorspace, switch_colormap

__all__ = ['RGBA_to_GRAY', 'RGBA_to_RGB',  # RGBA
           'RGB_to_GRAY', 'GRAY_to_RGB',  # GRAY
           'RGB_to_HSV', 'HSV_to_RGB',  # HSV
           'RGB_to_HLS', 'HLS_to_RGB',  # HSL
           'RGB_to_CMYK', 'CMYK_to_RGB',  # CMYK
           'XYZ_to_LAB', 'LAB_to_XYZ', 'RGB_to_XYZ', 'XYZ_to_RGB',  # XYZ
           'LAB_to_RGB', 'RGB_to_LAB',  # LAB
           'LUV_to_RGB', 'RGB_to_LUV',  # LUV
           'YCbCr_to_RGB', 'RGB_to_YCbCr',  # YCbCr
           'BINARY_to_GRAY', 'BINARY_to_RGB',
           'colorspace_fct']

__version__ = '1.0'


# -------- BINARY ---------------------#
class BINARY_to_GRAY:
    def __init__(self):
        pass

    def __call__(self, im, **kwargs):
        """
        Converts a binary image to grayscale.
        Args:
            im (torch.Tensor): The input binary image tensor.

        Returns:
            torch.Tensor: The grayscale image tensor.
        """
        assert im.colorspace == 'BINARY', "Wrong number of dimensions (/BINARY_to_GRAY)"
        im.data = im.to_tensor().to(torch.float)
        im.image_layout.update(colorspace='GRAY', num_ch=1, bit_depth=8)
        return im


class BINARY_to_RGB:
    def __init__(self):
        pass

    def __call__(self, im, colormap='gray', **kwargs):
        """
        Converts a binary image to RGB.
        Args:
            im (torch.Tensor): The input binary image tensor.

        Returns:
            torch.Tensor: The grayscale image tensor.
        """
        assert im.colorspace == 'BINARY', "Wrong number of dimensions (/BINARY_to_RGB)"
        bin_to_gray = BINARY_to_GRAY()
        gray_to_rgb = GRAY_to_RGB()
        im = bin_to_gray(im)
        return gray_to_rgb(im, colormap=colormap)


# -------- RGBA -----------------------#
class RGBA_to_GRAY:
    luma = {'SDTV': torch.Tensor([0.299, 0.587, 0.114, 1]),
            'Adobe': torch.Tensor([0.212, 0.701, 0.087, 1]),
            'HDTV': torch.Tensor([0.2126, 0.7152, 0.0722, 1]),
            'HDR': torch.Tensor([0.2627, 0.6780, 0.0593, 1])}

    def __init__(self):
        pass

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGBA image to grayscale using the specified luma coefficient or a default one.
        Warning : The Alpha transparency is lost during the operation
        Args:
            im (torch.Tensor): The input RGB image tensor.
            luma (str, optional): The name of the luma coefficient to use. Defaults to None.

        Returns:
            torch.Tensor: The grayscale image tensor.

        Raises:
            AssertionError: If the specified luma coefficient is not found in the dictionary.
        """
        assert im.colorspace == 'RGBA', "Wrong number of dimensions (/RGBA_to_GRAY)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if luma is not None:
            assert luma in self.luma
            im.data = torch.sum(torch.mul(im.permute(0, 2, 3, 1)[..., :-1], self.luma[luma]), dim=-1).unsqueeze(1)
        else:
            im.data = torch.sum(im[:, :-1, :, :] / 3, dim=1).unsqueeze(1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='GRAY', num_ch=1, channel_names=['Gray'])


class RGBA_to_RGB:

    def __init__(self):
        pass

    def __call__(self, im, **kwargs):
        """
        Converts an RGBA image to RGB.
        Warning : The Alpha transparency is lost during the operation
        Args:
            im (torch.Tensor): The input RGBA image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.

        Raises:
            AssertionError: If the specified luma coefficient is not found in the dictionary.
        """
        assert im.colorspace == 'RGBA', "Starting Colorspace (/RGBA_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        im.data = im.to_tensor()[:, :-1, :, :]
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])


# -------- GRAY -----------------------#
class RGB_to_GRAY:
    luma = {'SDTV': torch.Tensor([0.299, 0.587, 0.114, 1]),
            'Adobe': torch.Tensor([0.212, 0.701, 0.087, 1]),
            'HDTV': torch.Tensor([0.2126, 0.7152, 0.0722, 1]),
            'HDR': torch.Tensor([0.2627, 0.6780, 0.0593, 1])}

    def __init__(self):
        pass

    def __call__(self, im, luma: str = None, **kwargs):
        """
        Converts an RGB image to grayscale using the specified luma coefficient or a default one.

        Args:
            im (torch.Tensor): The input RGB image tensor.
            luma (str, optional): The name of the luma coefficient to use. Defaults to None.

        Returns:
            torch.Tensor: The grayscale image tensor.

        Raises:
            AssertionError: If the specified luma coefficient is not found in the dictionary.
        """
        assert im.colorspace == 'RGB', "Wrong number of dimensions (/RGB_to_GRAY)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if luma is not None:
            assert luma in self.luma
            im.data = torch.sum(torch.mul(im.permute(0, 2, 3, 1), self.luma[luma]), dim=-1).unsqueeze(1)
        else:
            im.data = torch.sum(im.to_tensor() / 3, dim=1, keepdim=True)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='GRAY', num_ch=1, modality='Any', colormap=None, channel_names=['Gray'])


class GRAY_to_RGB:

    def __call__(self, im, colormap='gray', **kwargs):
        """
        Converts an GRAY image to the RGB colorspace following a colormap.

        Args:
            im (torch.Tensor): The input GRAY image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """
        if colormap is None:
            colormap = 'gray'
        assert im.colorspace == 'GRAY', "Starting Colorspace (/GRAY_to_RGB)"
        assert im.channel_num == 1
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if im.depth == 8:
            depth, datatype = 8, torch.uint8
        elif im.depth == 16:
            depth, datatype = 16, torch.uint16
        else:
            im.depth = 8
            depth, datatype = 8, torch.uint8
        num = 2 ** (depth)
        x = np.linspace(0.0, 1.0, num)
        cmap_rgb = Tensor(cm[colormap](x)[:, :3]).to(im.device).squeeze()
        temp = (im.to_tensor().squeeze(1) * (num - 1)).to(datatype).long()
        im.data = cmap_rgb[temp].permute(0, 3, 1, 2).clamp(0.0, 1.0)
        # ------- Permute back the layers ----------- #
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, colormap=colormap, channel_names=['Red', 'Green', 'Blue'])


# -------- HSV -----------------------#
class RGB_to_HSV:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the HSV colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_HSV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = self.normalize(rgb_to_hsv(im))
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='HSV', num_ch=3, channel_names=['Hue', 'Saturation', 'Value'])

    @staticmethod
    def normalize(im):
        h, s, v = im.split(1, 1)
        h = h / (2 * torch.pi)
        return torch.cat([h, s, v], dim=1)


class HSV_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an HSV image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'HSV', "Starting Colorspace (/HSV_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = hsv_to_rgb(self.denormalize(im)).clamp(0.0, 1.0)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])

    @staticmethod
    def denormalize(im):
        h, s, v = im.split(1, 1)
        h = h * 2 * torch.pi
        return torch.cat([h, s, v], dim=1)


# -------- HLS -----------------------#

class RGB_to_HLS:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the HLS colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HLS image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_HLS)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = self.normalize(rgb_to_hls(im))
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='HLS', num_ch=3, channel_names=['Hue', 'Lightness', 'Saturation'])

    @staticmethod
    def normalize(im):
        h, l, s = im.split(1, 1)
        h = h / (2 * torch.pi)
        return torch.cat([h, l, s], dim=1)


class HLS_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an HLS image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input HLS image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'HLS', "Starting Colorspace (/HLS_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = hls_to_rgb(self.denormalize(im)).clamp(0.0, 1.0)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])

    @staticmethod
    def denormalize(im):
        h, l, s = im.split(1, 1)
        h = h * 2 * torch.pi
        return torch.cat([h, l, s], dim=1)


# -------- CMYK -----------------------#
class RGB_to_CMYK:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the HSV colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The HSV image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_HSV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        R, G, B = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:, :, :].data)
        # ------- Black Key K ---------------- #
        K = 1 - torch.max(im, dim=1, keepdim=True)[0]
        mask = K != 1
        # ------- Cyan ---------------- #
        C = torch.zeros_like(R, dtype=im.dtype)
        C[mask] = ((1 - R - K) / (1 - K))[mask]
        # ------- Magenta ---------------- #
        M = torch.zeros_like(R, dtype=im.dtype)
        M[mask] = ((1 - G - K) / (1 - K))[mask]
        # ------- Yellow ---------------- #
        Y = torch.zeros_like(R, dtype=im.dtype)
        Y[mask] = ((1 - B - K) / (1 - K))[mask]

        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([C, M, Y, K], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='CMYK', num_ch=4, channel_names=['Cyan', 'Magenta', 'Yellow', 'Black key'])


class CMYK_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an CMYK image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input CMYK image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'CMYK', "Starting Colorspace (/CMYK_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        C, M, Y, K = Tensor(im[:, :1, :, :].data), Tensor(im[:, 1:2, :, :].data), Tensor(im[:, 2:3, :, :].data), Tensor(
            im[:, 3:, :, :].data)
        # ------- R ---------------- #
        R = (1 - C) * (1 - K)
        # ------- G ---------------- #
        G = (1 - M) * (1 - K)
        # ------- B ---------------- #
        B = (1 - Y) * (1 - K)
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([R, G, B], dim=1).clamp(0.0, 1.0)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])


# -------- YCbCr -----------------------#
class RGB_to_YCbCr:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the YCbCr colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The YCbCr image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_YCbCr)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = rgb_to_ycbcr(im)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='YCbCr', num_ch=3,
                               channel_names=['Luma', 'Blue Chrominance', 'Red Chrominance'])


class YCbCr_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an CMYK image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input CMYK image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'YCbCr', "Starting Colorspace (/YCbCr_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = ycbcr_to_rgb(im).clamp(0.0, 1.0)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])


# -------- XYZ -----------------------#
class RGB_to_XYZ:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_XYZ)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = rgb_to_xyz(im)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='XYZ', num_ch=3, channel_names=['X', 'Y', 'Z'])


class XYZ_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an XYZ image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input XYZ image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'XYZ', "Starting Colorspace (/XYZ_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = xyz_to_rgb(im).clamp(0.0, 1.0)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])


class XYZ_to_LAB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an XYZ image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input XYZ image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'XYZ', "Starting Colorspace (/XYZ_to_LAB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            im = switch_colormap(im, colormap, **kwargs)
        # ------- to LAB ---------------- #
        mask = Tensor(im.data) > (6 / 29) ** 3
        temp = im.clone()
        temp[mask] = temp[mask] ** (1 / 3)
        temp[~mask] = 1 / 3 * (29 / 6) ** 2 * temp[~mask] + 4 / 29
        X, Y, Z = temp[:, :1, :, :], temp[:, 1:2, :, :], temp[:, 2:, :, :]
        # ------- to LAB ---------------- #
        L = ((116 * Y) - 16) / 100
        a = (500 * (X - Y) + 128) / 256
        b = (200 * (Y - Z) + 128) / 256
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([L, a, b], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='LAB', num_ch=3, channel_names=['Luminance', 'A', 'B'])


class LAB_to_XYZ:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an XYZ image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input XYZ image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'LAB', "Starting Colorspace (/LAB_to_XYZ)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            im = switch_colormap(im, colormap, **kwargs)
        temp = im.clone()
        L = (temp[:, :1, :, :] * 100 + 16) / 116
        a = (temp[:, 1:2, :, :] * 256 - 128) / 500
        b = (temp[:, 2:, :, :] * 256 - 128) / 200
        # ------- Y ---------------- #
        mask = L > (6 / 29) ** 3
        Y = torch.zeros_like(L)
        Y[mask] = L[mask] ** 3
        Y[~mask] = 3 * (6 / 29) ** 2 * (L[~mask] - 4 / 29)
        # ------- X ---------------- #
        mask = L + a > (6 / 29) ** 3
        X = L + a
        X[mask] = (X[mask]) ** 3
        X[~mask] = 3 * (6 / 29) ** 2 * (X[~mask] - 4 / 29)
        # ------- Z ---------------- #
        mask = L - b > (6 / 29) ** 3
        Z = L - b
        Z[mask] = (Z[mask]) ** 3
        Z[~mask] = 3 * (6 / 29) ** 2 * (Z[~mask] - 4 / 29)
        # ------- Stack the layers ----------- #
        im.data = torch.concatenate([X, Y, Z], dim=1)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='XYZ', num_ch=3, channel_names=['X', 'Y', 'Z'])


# -------- LAB -----------------------#
class RGB_to_LAB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the LAB colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The LAB image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_LAB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = self.normalize(rgb_to_lab(im.to_tensor()))
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='LAB', num_ch=3, channel_names=['Luminance', 'A', 'B'])

    @staticmethod
    def normalize(im):
        l, a, b = im.split(1, 1)
        l = l / 100
        a = (a + 128) / 255 * 2 - 1
        b = (b + 128) / 255 * 2 - 1
        return torch.cat([l, a, b], dim=1)


class LAB_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an LAB image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input LAB image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'LAB', "Starting Colorspace (/LAB_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = lab_to_rgb(self.denormalize(im.to_tensor()), clip=True)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])

    @staticmethod
    def denormalize(im):
        l, a, b = im.split(1, 1)
        l = l * 100
        a = (a + 1) / 2 * 255 - 128
        b = (b + 1) / 2 * 255 - 128
        return torch.cat([l, a, b], dim=1)


# -------- LUV -----------------------#
class RGB_to_LUV:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an RGB image to the LUV colorspace.

        Args:
            im (torch.Tensor): The input RGB image tensor.

        Returns:
            torch.Tensor: The LUV image tensor.
        """

        assert im.colorspace == 'RGB', "Starting Colorspace (/RGB_to_LUV)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = self.normalize(rgb_to_luv(im))
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='LUV', num_ch=3, channel_names=['Luminance', 'U', 'V'])

    @staticmethod
    def normalize(im):
        l, u, v = im.split(1, 1)
        l = l / 100
        u = (u + 134) / (220 + 134)
        v = (v + 140) / (122 + 140)
        return torch.cat([l, u, v], dim=1)


class LUV_to_RGB:

    def __call__(self, im, colormap: str = None, **kwargs):
        """
        Converts an LUV image to the RGB colorspace.

        Args:
            im (torch.Tensor): The input LUV image tensor.

        Returns:
            torch.Tensor: The RGB image tensor.
        """

        assert im.colorspace == 'LUV', "Starting Colorspace (/LUV_to_RGB)"
        layers = im.layers_name
        im.reset_layers_order(in_place=True)
        if colormap is not None:
            switch_colormap(im, colormap, **kwargs)
        im.data = luv_to_rgb(self.denormalize(im)).clamp(0.0, 1.0)
        im.permute(layers, in_place=True)
        im.image_layout.update(colorspace='RGB', num_ch=3, channel_names=['Red', 'Green', 'Blue'])

    @staticmethod
    def denormalize(im):
        l, u, v = im.split(1, 1)
        l = l * 100
        u = u * (220 + 134) - 134
        v = v * (122 + 140) - 140
        return torch.cat([l, u, v], dim=1)


def colorspace_fct(colorspace_change):
    if colorspace_change not in __all__:
        wrapper = colorspace_fct(f"{colorspace_change.split('_')[0]}_to_RGB")
        colorspace_change = f"RGB_to_{colorspace_change.split('_')[-1]}"
    else:
        wrapper = None

    # -------- RGBA -----------------------#
    if colorspace_change == 'RGBA_to_GRAY':
        fct = RGBA_to_GRAY()
    elif colorspace_change == 'RGBA_to_RGB':
        fct = RGBA_to_RGB()
    # -------- GRAY -----------------------#
    elif colorspace_change == 'RGB_to_GRAY':
        fct = RGB_to_GRAY()
    elif colorspace_change == 'GRAY_to_RGB':
        fct = GRAY_to_RGB()
    # -------- BINARY ---------------------#
    elif colorspace_change == 'BINARY_to_RGB':
        fct = BINARY_to_RGB()
    elif colorspace_change == 'BINARY_to_GRAY':
        fct = BINARY_to_GRAY()
    # -------- LAB -----------------------#
    elif colorspace_change == 'RGB_to_LAB':
        fct = RGB_to_LAB()
    elif colorspace_change == 'LAB_to_RGB':
        fct = LAB_to_RGB()
    # -------- LUV -----------------------#
    elif colorspace_change == 'RGB_to_LUV':
        fct = RGB_to_LUV()
    elif colorspace_change == 'LUV_to_RGB':
        fct = LUV_to_RGB()
    # -------- XYZ -----------------------#
    elif colorspace_change == 'XYZ_to_LAB':
        fct = XYZ_to_LAB()
    elif colorspace_change == 'LAB_to_XYZ':
        fct = LAB_to_XYZ()
    elif colorspace_change == 'XYZ_to_RGB':
        fct = XYZ_to_RGB()
    elif colorspace_change == 'RGB_to_XYZ':
        fct = RGB_to_XYZ()
    # -------- HSV -----------------------#
    elif colorspace_change == 'HSV_to_RGB':
        fct = HSV_to_RGB()
    elif colorspace_change == 'RGB_to_HSV':
        fct = RGB_to_HSV()
    # -------- HLS -----------------------#
    elif colorspace_change == 'HLS_to_RGB':
        fct = HLS_to_RGB()
    elif colorspace_change == 'RGB_to_HLS':
        fct = RGB_to_HLS()
    # -------- CMYK -----------------------#
    elif colorspace_change == 'CMYK_to_RGB':
        fct = CMYK_to_RGB()
    elif colorspace_change == 'RGB_to_CMYK':
        fct = RGB_to_CMYK()
    # -------- YCbCr -----------------------#
    elif colorspace_change == 'YCbCr_to_RGB':
        fct = YCbCr_to_RGB()
    elif colorspace_change == 'RGB_to_YCbCr':
        fct = RGB_to_YCbCr()
    else:
        raise NotImplementedError

    if wrapper is not None:
        fct = wrap_colorspace(wrapper, fct)
    return fct


# The classic CIE ΔE2000 implementation, which operates on two L*a*b* colors, and returns their difference.
# "l" ranges from 0 to 100, while "a" and "b" are unbounded and commonly clamped to the range of -128 to 127.
def color_distance(img1, img2):
    eps = 1e-4
    # Working in Python with the CIEDE2000 color-difference formula.
    # k_l, k_c, k_h are parametric factors to be adjusted according to
    # different viewing parameters such as textures, backgrounds...
    img1 = img1.LAB() if img1.colorspace != 'LAB' else img1
    img2 = img2.LAB() if img2.colorspace != 'LAB' else img2
    l_1, a_1, b_1 = img1[:, :1] * 100, (img1[:, 1:2] + 1) * 255 / 2 - 128, (img1[:, 2:] + 1) * 255 / 2 - 128
    l_2, a_2, b_2 = img2[:, :1] * 100, (img2[:, 1:2] + 1) * 255 / 2 - 128, (img2[:, 2:] + 1) * 255 / 2 - 128
    k_l = k_c = k_h = 1.0
    n = (torch.sqrt(a_1 * a_1 + b_1 * b_1 + eps) + torch.sqrt(a_2 * a_2 + b_2 * b_2 + eps)) * 0.5
    n = n**7
    # A factor involving chroma raised to the power of 7 designed to make
    # the influence of chroma on the total color difference more accurate.
    n = 1.0 + 0.5 * (1.0 - torch.sqrt(n / (n + 6103515625.0)) + eps)
    # Application of the chroma correction factor.
    c_1 = torch.sqrt(a_1 * a_1 * n * n + b_1 * b_1 + eps)
    c_2 = torch.sqrt(a_2 * a_2 * n * n + b_2 * b_2 + eps)
    # atan2 is preferred over atan because it accurately computes the angle of
    # a point (x, y) in all quadrants, handling the signs of both coordinates.
    h_1 = torch.atan2(b_1, a_1 * n)
    h_2 = torch.atan2(b_2, a_2 * n)
    h_1 += 2.0 * torch.pi * (h_1 < 0.0)
    h_2 += 2.0 * torch.pi * (h_2 < 0.0)
    n = torch.abs(h_2 - h_1)
    # Cross-implementation consistent rounding.
    n = torch.where((torch.pi - 1E-14 < n) * (n < torch.pi + 1E-14), torch.pi, n)
    # When the hue angles lie in different quadrants, the straightforward
    # average can produce a mean that incorrectly suggests a hue angle in
    # the wrong quadrant, the next lines handle this issue.
    h_m = (h_1 + h_2) * 0.5
    h_d = (h_2 - h_1) * 0.5
    mask = torch.pi < n
    h_d[mask] += torch.pi
    # 📜 Sharma’s formulation doesn’t use the next line, but the one after it,
    # and these two variants differ by ±0.0003 on the final color differences.
    h_m[mask] += torch.pi
    # h_m += pi if h_m < pi else -pi
    p = 36.0 * h_m - 55.0 * torch.pi
    n = (c_1 + c_2) * 0.5
    n = n**7
    # The hue rotation correction term is designed to account for the
    # non-linear behavior of hue differences in the blue region.
    r_t = -2.0 * torch.sqrt(n / (n + 6103515625.0) + eps) \
          * torch.sin(torch.pi / 3.0 * torch.exp(p * p / (-25.0 * torch.pi * torch.pi)))
    n = (l_1 + l_2) * 0.5
    n = (n - 50.0) * (n - 50.0)
    # Lightness.
    l = (l_2 - l_1) / (k_l * (1.0 + 0.015 * n / torch.sqrt(20.0 + n + eps)))
    # These coefficients adjust the impact of different harmonic
    # components on the hue difference calculation.
    t = 1.0 + 0.24 * torch.sin(2.0 * h_m + torch.pi * 0.5) \
        + 0.32 * torch.sin(3.0 * h_m + 8.0 * torch.pi / 15.0) \
        - 0.17 * torch.sin(h_m + torch.pi / 3.0) \
        - 0.20 * torch.sin(4.0 * h_m + 3.0 * torch.pi / 20.0)
    n = c_1 + c_2
    # Hue.
    h = 2.0 * torch.sqrt(c_1 * c_2 + eps) * torch.sin(h_d) / (k_h * (1.0 + 0.0075 * n * t) + eps)
    # Chroma.
    c = (c_2 - c_1) / (k_c * (1.0 + 0.0225 * n) + eps)
    # Returns the square root so that the DeltaE 2000 reflects the actual geometric
    # distance within the color space, which ranges from 0 to approximately 185.
    return torch.sqrt(l**2 + h**2 + c**2 + c * h * r_t + eps)
