from typing import Literal

import cv2
import cv2 as cv
import kornia
import numpy as np
import torch
from kornia import create_meshgrid
from kornia.feature.responses import harris_response
from kornia.geometry import hflip, vflip
from kornia.morphology import opening
from torch import Tensor, FloatTensor
from skimage.segmentation import flood
from torch.nn.functional import conv2d

# --------- Import local classes -------------------------------- #
from ..Image import ImageTensor


def draw_grid(im_cv, grid_size: int = 24):
    im_gd_cv = np.full_like(im_cv, 255.0)
    im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)

    height, width = im_cv.shape
    color = (0, 0, 255)
    for x in range(0, width - 1, grid_size):
        cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1)  # (0, 0, 0)
    for y in range(0, height - 1, grid_size):
        cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)
    im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
    return im_gd_ts


def drawlines(img, lines, pts):
    """
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    """
    r, c = img.shape[:2]
    for r, pt in zip(lines, pts.squeeze().cpu().numpy()):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv.circle(img, (int(pt[0]), int(pt[1])), 5, color, -1)
    return img


def extract_roi_from_images(mask: ImageTensor, *args, return_pts=True):
    """
    :param mask:
    :param args:
    :param return_pts: return points of the corners of the ROI, in the order top_left, top_right, bot_left, bot_right
    :return:
    """
    roi = []
    pts = []
    mask = ImageTensor(mask) if not isinstance(mask, ImageTensor) else mask.clone()
    mask_valid = 1 - extract_external_occlusion(mask)
    or_mask_valid = mask_valid * 1.
    and_mask_valid = mask_valid * 1.
    split_sides = split_point_closer_side(mask)
    split_mask, left_grid, right_grid, top_grid, bottom_grid = split_sides

    if len(args) > 0:
        for m in args:
            m = ImageTensor(m) if not isinstance(m, ImageTensor) else m.clone()
            assert m.image_size == mask.image_size
            m = 1 - extract_external_occlusion(m)
            or_mask_valid += m
            and_mask_valid *= m

    and_mask_to_crop = (and_mask_valid == 0) * 1.0
    or_mask_to_crop = (or_mask_valid == 0) * 1.0

    m_transfo = [ImageTensor(or_mask_to_crop), ImageTensor(and_mask_to_crop)]
    m_roi = [ImageTensor(or_mask_to_crop), ImageTensor(and_mask_to_crop)]

    for m_ in m_roi:
        # first OR mask --> At least one image with information, second AND mask --> All images with information
        m_ = m_.to_tensor().to(torch.float)
        if m_.any():
            left = m_ * (split_mask == 0) * left_grid
            right = m_ * (split_mask == 1) * right_grid
            top = m_ * (split_mask == 2) * top_grid
            bottom = m_ * (split_mask == 3) * bottom_grid

            roi.append([int(left.max()),
                        int(right_grid.max() - right.max()),
                        int(top.max()),
                        int(bottom_grid.max() - bottom.max())])
        else:
            roi.append([0, mask.image_size[1], 0, mask.image_size[0]])
    if return_pts:
        for m_ in m_transfo:
            m_ = m_.to_tensor().to(torch.float)
            corner_map = harris_response(m_).squeeze()
            center = int(corner_map.shape[0] / 2), int(corner_map.shape[1] / 2)
            top_l = corner_map[:center[0], : center[1]]
            top_r = corner_map[:center[0], center[1]:]
            bot_l = corner_map[center[0]:, :center[1]]
            bot_r = corner_map[center[0]:, center[1]:]
            top_left = torch.argmax(top_l)
            top_left = top_left % center[1] - 1, top_left // center[1] - 1
            top_right = torch.argmax(top_r)
            top_right = top_right % center[1] + center[1] - 1, top_right // center[1] - 1,
            bot_left = torch.argmax(bot_l)
            bot_left = bot_left % center[1] - 1, bot_left // center[1] + center[0] - 1
            bot_right = torch.argmax(bot_r)
            bot_right = bot_right % center[1] + center[1] - 1, bot_right // center[1] + center[0] - 1
            pts.append([top_left, top_right, bot_left, bot_right])
    if return_pts:
        return roi[1], roi[0], FloatTensor(pts[0]), FloatTensor(pts[1])
    else:
        return roi[1], roi[0]


def extract_external_occlusion(mask: ImageTensor) -> Tensor:
    if (mask == 0).any():
        mask.pad((1, 1), in_place=True, value=0)
        new = mask.to_numpy().squeeze()
        new = flood(new, (0, 0))
        mask.data = Tensor(new).to(mask.device).unsqueeze(0).unsqueeze(0)
        mask.unpad(in_place=True)
    else:
        return mask * 0.
    return mask


def split_point_closer_side(mask: ImageTensor):
    grid = create_meshgrid(mask.image_size[0], mask.image_size[1], device=mask.device)  # [1 H W 2]
    left, top = grid[:, :, :, 0], grid[:, :, :, 1]
    right, bottom = hflip(left), vflip(top)
    conc = torch.cat([left, right, top, bottom], dim=0)
    return (torch.argmin(conc, dim=0), (left + 1) / 2 * (mask.image_size[1] - 1),
            (right + 1) / 2 * (mask.image_size[1] - 1),
            (top + 1) / 2 * (mask.image_size[0] - 1),
            (bottom + 1) / 2 * (mask.image_size[0] - 1))


def get_common_roi(img_a, img_b, mode: Literal['lrtb', 'xyxy', 'xywh'] = 'lrtb', inner: bool = True):
    """
    Find the common ROI between two tensor (B, C, H, W).
    return ROIs of the form: Tensor(B, 4) with (y1, y2, x1, x2) if mode='lrtb', (x1, y1, x2, y2) if mode='xyxy' and (x, y, w, h) if mode='xywh'
    inner: if True, the ROI will contain only valid pixels.
    """
    img_a = img_a.sum(1)
    img_b = img_b.sum(1)
    assert img_a.shape == img_b.shape, "Both images must have the same shape"
    assert img_a.dim() == 3, "Input images must be 4D tensors (B, C, H, W)"

    mask_a = img_a > 0
    mask_b = img_b > 0
    B = img_a.shape[0]
    rois = torch.zeros([B, 4], dtype=torch.int64, device=img_a.device)
    for b in range(B):
        common_mask = (mask_a[b] & mask_b[b]) * 1. if inner else (mask_a[b] | mask_b[b]) * 1.
        common_mask = opening(common_mask.unsqueeze(0).unsqueeze(0),
                              kernel=torch.ones((5, 5)).to(img_a.device)).squeeze()
        common_mask = ~extract_external_occlusion(ImageTensor(common_mask)).squeeze().to(torch.bool)
        coords = torch.nonzero(common_mask)

        if coords.shape[0] == 0:  # No area in common, return the whole image as ROI
            if mode == 'lrtb':
                rois[b] = torch.tensor([0, img_a.shape[2], 0, img_a.shape[1]], device=img_a.device)
            elif mode == 'xyxy':
                rois[b] = torch.tensor([0, 0, img_a.shape[2], img_a.shape[1]], device=img_a.device)
            elif mode == 'xywh':
                rois[b] = torch.tensor([0, 0, img_a.shape[2], img_a.shape[1]], device=img_a.device)
            continue

        y_min, x_min = coords.min(dim=0).values
        y_max, x_max = coords.max(dim=0).values

        if mode == 'lrtb':
            rois[b] = torch.tensor([y_min, y_max + 1, x_min, x_max + 1], device=img_a.device)
        elif mode == 'xyxy':
            rois[b] = torch.tensor([x_min, y_min, x_max + 1, y_max + 1], device=img_a.device)
        elif mode == 'xywh':
            rois[b] = torch.tensor([x_min, y_min, x_max - x_min + 1, y_max - y_min + 1], device=img_a.device)

    return rois


def crop_to_common_roi(img_a, img_b, inner=True):
    cls_a = img_a.__class__
    cls_b = img_b.__class__
    outer_rois, inner_rois = [], []
    for b in range(img_a.shape[0]):
        inner_roi, outer_roi = extract_roi_from_images(img_a[b].sum(0) > 0, img_b[b].sum(0) > 0, return_pts=False)
        outer_rois.append(outer_roi)
        inner_rois.append(inner_roi)
    cropped_a = []
    cropped_b = []
    for b in range(img_a.shape[0]):
        x1, x2, y1, y2 = inner_rois[b] if inner else outer_rois[b]
        cropped_a.append(img_a[b:b + 1, :, y1:y2, x1:x2])
        cropped_b.append(img_b[b:b + 1, :, y1:y2, x1:x2])
    return cls_a(torch.cat(cropped_a)), cls_b(torch.cat(cropped_b))

