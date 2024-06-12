import numpy as np
import torch
import torch.utils.data
from kornia.geometry import compute_correspond_epilines

from ..Image import ImageTensor
from .drawing import drawlines


def show_epipolar(im_src, im_dst, F_mat, pts_src, pts_dst) -> None:
    """
    Draw the epipolar lines one both image.
    :param im_src: image source of F_mat
    :param im_dst: image destination of F_mat
    :param F_mat: Fundamental matrix going from im_src to im_dst
    :param args: **
    :param pts_src: Optional pts src to draw them as well
    :param pts_dst: Optional pts dst to draw them as well
    :param kwargs:
    :return:
    """
    epipolar_lines_dst = compute_correspond_epilines(pts_src, F_mat).squeeze().cpu().numpy()
    epipolar_lines_src = compute_correspond_epilines(pts_dst, F_mat[0].transpose(-2, -1)).squeeze().cpu().numpy()
    if im_src.im_type == 'RGB':
        im_src_w_line = im_src.opencv()
    else:
        im_src_w_line = im_src.RGB(cmap='gray').opencv()
    if im_dst.im_type == 'RGB':
        im_dst_w_line = im_dst.opencv()
    else:
        im_dst_w_line = im_dst.RGB(cmap='gray').opencv()
    im_src_w_line = ImageTensor(drawlines(im_src_w_line, epipolar_lines_src, pts_src)[..., [2, 1, 0]])
    im_dst_w_line = ImageTensor(drawlines(im_dst_w_line, epipolar_lines_dst, pts_dst)[..., [2, 1, 0]])
    im_dst_w_line = im_dst_w_line.pad(im_src_w_line)
    im_src_w_line = im_src_w_line.pad(im_dst_w_line)
    B, C, H, W = im_src_w_line.shape
    torch.stack([im_src_w_line, im_dst_w_line], dim=3).view([B, C, H, 2 * W]).show()


def gen_error_colormap():
    cols = np.array(
        [[0 / 3.0, 0.1875 / 3.0, 49, 54, 149],
         [0.1875 / 3.0, 0.375 / 3.0, 69, 117, 180],
         [0.375 / 3.0, 0.75 / 3.0, 116, 173, 209],
         [0.75 / 3.0, 1.5 / 3.0, 171, 217, 233],
         [1.5 / 3.0, 3 / 3.0, 224, 243, 248],
         [3 / 3.0, 6 / 3.0, 254, 224, 144],
         [6 / 3.0, 12 / 3.0, 253, 174, 97],
         [12 / 3.0, 24 / 3.0, 244, 109, 67],
         [24 / 3.0, 48 / 3.0, 215, 48, 39],
         [48 / 3.0, np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols
