import cv2 as cv
import numpy as np
import torch
from kornia import create_meshgrid
from kornia.feature.responses import harris_response
from kornia.geometry import hflip, vflip
from torch import Tensor, FloatTensor
from skimage.segmentation import flood

from ..Image import ImageTensor


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
    roi = []
    pts = []
    or_mask = 0
    and_mask = 1
    mask = ImageTensor(mask) if not isinstance(mask, ImageTensor) else mask.clone()
    mask = extract_external_occlusion(mask)
    split_sides = split_point_closer_side(mask)
    split_mask, left_grid, right_grid, top_grid, bottom_grid = split_sides

    if len(args) > 0:
        for m in args:
            m = ImageTensor(m) if not isinstance(m, ImageTensor) else m.clone()
            assert m.image_size == mask.image_size
            m = extract_external_occlusion(m)
            or_mask += m.GRAY()
            and_mask *= m.GRAY()

    if not isinstance(or_mask, int):
        m_transfo = [ImageTensor(mask.GRAY()), ImageTensor(or_mask)]
        m_roi = [ImageTensor(or_mask + mask.GRAY() > 0),
                 ImageTensor(and_mask * mask.GRAY() > 0)]
    else:
        m_transfo = [ImageTensor(mask.GRAY())]
        m_roi = [ImageTensor(or_mask + mask.GRAY() > 0)]

    for m_ in m_roi:
        m_ = m_.to_tensor().to(torch.float)
        left = m_ * (split_mask == 0)*left_grid
        right = m_ * (split_mask == 1)*right_grid
        top = m_ * (split_mask == 2)*top_grid
        bottom = m_ * (split_mask == 3)*bottom_grid

        roi.append([int(top.max()), int(left.max()),
                    int(bottom_grid.max() - bottom.max()),
                    int(right_grid.max() - right.max())])
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
    if not isinstance(or_mask, int):
        if return_pts:
            return roi[1], roi[0], FloatTensor(pts[0]), FloatTensor(pts[1])
        else:
            return roi[1], roi[0]
    else:
        if return_pts:
            return roi[0], FloatTensor(pts[0])
        else:
            return roi[0]


def extract_external_occlusion(mask: ImageTensor) -> Tensor:
    mask.pad((1, 1), in_place=True, value=0)
    new = mask.to_numpy()
    new = flood(new, (0, 0))
    mask.data = Tensor(new).to(mask.device)
    new.unpad(in_place=True)
    return mask.unpad()


def split_point_closer_side(mask: ImageTensor):
    grid = create_meshgrid(mask.image_size[0], mask.image_size[1], device=mask.device)  # [1 H W 2]
    left, top = grid[:, :, :, 0], grid[:, :, :, 1]
    right, bottom = hflip(left), vflip(top)
    conc = torch.cat([left, right, top, bottom], dim=0)
    return (torch.argmin(conc, dim=0), (left + 1)/2*(mask.image_size[1]-1),
            (right + 1)/2*(mask.image_size[1]-1),
            (top + 1)/2*(mask.image_size[0]-1),
            (bottom + 1)/2*(mask.image_size[0]-1))

