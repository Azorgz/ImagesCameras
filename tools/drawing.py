import cv2 as cv
import numpy as np
import torch
from kornia.feature.responses import harris_response
from kornia.morphology import closing, opening, dilation
from torch import Tensor, FloatTensor

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


def extract_roi_from_images(mask: ImageTensor, *args):
    roi = []
    pts = []
    or_mask = 0
    and_mask = 1
    mask = ImageTensor(mask) if not isinstance(mask, ImageTensor) else mask
    mask = extract_external_occlusion(mask)
    if len(args) > 0:
        for m in args:
            m = ImageTensor(m) if not isinstance(m, ImageTensor) else m
            assert m.image_size == mask.image_size
            m = extract_external_occlusion(m)
            or_mask += m.GRAY()
            and_mask *= m.GRAY()

    if not isinstance(or_mask, int):
        m_transfo = [ImageTensor(mask.GRAY()).pad([1, 1, 1, 1]), ImageTensor(or_mask).pad([1, 1, 1, 1])]
        m_roi = [ImageTensor(or_mask + mask.GRAY() > 0).pad([1, 1, 1, 1]),
                 ImageTensor(and_mask * mask.GRAY() > 0).pad([1, 1, 1, 1])]
    else:
        m_transfo = [ImageTensor(mask.GRAY()).pad([1, 1, 1, 1])]
        m_roi = [ImageTensor(or_mask + mask.GRAY() > 0).pad([1, 1, 1, 1])]

    for m_ in m_roi:
        m_ = m_.to_tensor().to(torch.float)
        corner_map = harris_response(m_).squeeze()
        center = int(corner_map.shape[0] / 2), int(corner_map.shape[1] / 2)
        top_l = corner_map[:center[0], : center[1]]
        top_r = corner_map[:center[0], center[1]:]
        bot_l = corner_map[center[0]:, :center[1]]
        bot_r = corner_map[center[0]:, center[1]:]
        top_left = torch.argmax(top_l)
        top_left = top_left // center[1] - 1, top_left % center[1] - 1
        top_right = torch.argmax(top_r)
        top_right = top_right // center[1] - 1, top_right % center[1] + center[1] - 1
        bot_left = torch.argmax(bot_l)
        bot_left = bot_left // center[1] + center[0] - 1, bot_left % center[1] - 1
        bot_right = torch.argmax(bot_r)
        bot_right = bot_right // center[1] + center[0] - 1, bot_right % center[1] + center[1] - 1
        roi.append([int(max(top_left[0], top_right[0])), int(max(top_left[1], bot_left[1])),
                    int(min(bot_left[0], bot_right[0])), int(min(bot_right[1], top_right[1]))])

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
        return roi[1], roi[0], FloatTensor(pts[0]), FloatTensor(pts[1])
    else:
        return roi[0], FloatTensor(pts[0])


def extract_external_occlusion(mask: ImageTensor) -> Tensor:
    mask.pad((50, 50), in_place=True, value=1)
    temp = mask.clone()
    new = mask.clone()
    kernel = torch.ones([5, 20], device=mask.device if isinstance(mask, Tensor) else 'cpu')
    for i in range(3):
        temp = opening(temp*1., kernel)
    temp = dilation(temp, kernel)
    new.data = temp
    new.unpad(in_place=True)
    return ImageTensor(new == mask.unpad())


def extract_roi_from_map(mask_left: Tensor, mask_right: Tensor):
    roi = []
    pts = []

    m_roi = [ImageTensor(mask_right + mask_left > 0).pad([1, 1, 1, 1]),
             ImageTensor(mask_right * mask_left > 0).pad([1, 1, 1, 1])]
    m_transfo = [ImageTensor(mask_left).pad([1, 1, 1, 1]), ImageTensor(mask_right).pad([1, 1, 1, 1])]

    for m_ in m_roi:
        m_ = m_.to_tensor().to(torch.float)
        corner_map = Tensor(harris_response(m_)).squeeze()
        center = int(corner_map.shape[0] / 2), int(corner_map.shape[1] / 2)
        top_l = corner_map[:center[0], : center[1]]
        top_r = corner_map[:center[0], center[1]:]
        bot_l = corner_map[center[0]:, :center[1]]
        bot_r = corner_map[center[0]:, center[1]:]
        top_left = torch.argmax(top_l)
        top_left = top_left // center[1] - 1, top_left % center[1] - 1
        top_right = torch.argmax(top_r)
        top_right = top_right // center[1] - 1, top_right % center[1] + center[1] - 1
        bot_left = torch.argmax(bot_l)
        bot_left = bot_left // center[1] + center[0] - 1, bot_left % center[1] - 1
        bot_right = torch.argmax(bot_r)
        bot_right = bot_right // center[1] + center[0] - 1, bot_right % center[1] + center[1] - 1
        roi.append([int(max(top_left[0], top_right[0])), int(max(top_left[1], bot_left[1])),
                    int(min(bot_left[0], bot_right[0])), int(min(bot_right[1], top_right[1]))])

    for m_ in m_transfo:
        m_ = m_.to_tensor().to(torch.float)
        corner_map = Tensor(harris_response(m_)).squeeze()
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
    return roi[1], roi[0], FloatTensor(pts[0]), FloatTensor(pts[1])
