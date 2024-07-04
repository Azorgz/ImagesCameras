import os
from os.path import basename

import cv2 as cv
import numpy as np
import torch
from torch import Tensor
from torchvision.transforms.functional import to_pil_image


class Encoder:

    def __init__(self, depth: int, modality: str, batched: bool, ext: str = None):
        if depth == 8:
            self.datatype = torch.uint8
        elif depth == 16:
            self.datatype = torch.uint16
        elif depth == 32:
            self.datatype = torch.float32
        elif depth == 64:
            self.datatype = torch.float64
        self.depth = depth

        if modality == 'Multimodal':
            self.ext = 'npy'
        elif batched:
            self.ext = 'tiff'
        else:
            if depth > 16:
                self.ext = 'tiff'
            else:
                self.ext = ext if ext is not None else 'png'

    def __call__(self, im: Tensor, path, *args, name=None, keep_colorspace=False, **kwargs):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if self.depth == 8 and keep_colorspace:
            if im.colorspace in ['LAB', 'HSV', 'CMYK', 'RGBA']:
                mode = im.colorspace
            elif im.colorspace == 'GRAY':
                mode = 'L'
            elif im.colorspace == 'BINARY':
                mode = '1'
            image = to_pil_image(im, mode)
            image.save(path + '/' + name)
        else:
            im = im.RGB() if im.colorspace in ['LAB', 'HSV', 'CMYK'] else im
            if im.batched:
                if im.channel_num not in [1, 3, 4]:
                    name = name + f".{self.ext}" if name is not None else im.name + f".{self.ext}"
                    np.save(path + '/' + name, im.to_numpy())
                else:
                    name = name + f".{self.ext}" if name is not None else im.name + f".{self.ext}"
                    if not cv.imwritemulti(path + f'/{name}', im.to_opencv()):
                        raise Exception("Could not write image")
            else:
                name = name + f".{self.ext}" if name is not None else im.name + f".{self.ext}"
                im = im.to_opencv(datatype=np.uint8) if im.depth == 1 else im.to_opencv()
                if not cv.imwrite(path + f'/{name}', im):
                    raise Exception("Could not write image")


class Decoder:

    def __init__(self, filename):
        ext = basename(filename).split('.')[1]
        self.batched = False
        if ext.upper() == 'TIFF':
            try:
                valid, inp_ = cv.imreadmulti(filename, flags=-1)
                assert valid
                inp = np.stack(inp_)
                self.batched = True
            except AssertionError:
                inp = cv.imread(filename, cv.IMREAD_LOAD_GDAL)
            if inp.shape[-1] == 3:
                self.value = inp[..., [2, 1, 0]]
            elif inp.shape[-1] == 4:
                self.value = inp[..., [2, 1, 0, 3]]
            elif len(inp.shape) == 4:
                self.batched = True
                self.value = inp
            else:
                self.value = np.expand_dims(inp, 1)
        elif ext.upper() == 'NPY':
            self.value = np.load(filename)
            if self.value.shape[0] > 1:
                self.batched = True
        else:
            inp = cv.imread(filename, -1)
            if inp.shape[-1] == 3:
                self.value = inp[..., [2, 1, 0]]
            elif inp.shape[-1] == 4:
                self.value = inp[..., [2, 1, 0, 3]]
            else:
                self.value = inp
        assert self.value is not None, f'No Image found at {filename}'
