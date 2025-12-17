import os
from os.path import basename

import cv2 as cv
import numpy as np
from tifffile import imread, TiffFile
from torch import Tensor
from torchvision.transforms.functional import to_pil_image


class Encoder:

    def __init__(self, depth: int, modality: str, batched: bool, ext: str = None):
        if depth == 8:
            self.datatype = np.uint8
        elif depth == 16:
            self.datatype = np.uint16
            depth = 16
        elif depth == 32:
            self.datatype = np.uint16
            depth = 16
        elif depth == 64:
            self.datatype = np.uint16
            depth = 16
        self.depth = depth

        if ext is not None:
            self.ext = ext
        elif modality == 'Multimodal':
            self.ext = 'npy'
        elif batched:
            self.ext = 'tiff'
        else:
            if depth > 16:
                self.ext = 'tiff'
            else:
                self.ext = 'png'

    def __call__(self, im: Tensor, path, *args, name=None, keep_colorspace=False, **kwargs):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        path = str(path)
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
            im = im.RGB() if im.colorspace not in ['RGB', 'GRAY', 'BINARY'] else im*1
            if im.batched:
                if im.channel_num not in [1, 3, 4]:
                    name = name + f".{self.ext}" if name is not None else im.name + f".{self.ext}"
                    np.save(path + '/' + name, im.to_numpy(datatype=self.datatype))
                else:
                    name = name + f".{self.ext}" if name is not None else im.name + f".{self.ext}"
                    if not cv.imwritemulti(path + f'/{name}', im.to_opencv(datatype=self.datatype)):
                        raise Exception("Could not write image")
            else:
                name = name + f".{self.ext}" if name is not None else im.name + f".{self.ext}"
                im = im.to_opencv(datatype=np.uint8) if im.depth == 1 else im.to_opencv(datatype=self.datatype)
                if not cv.imwrite(path + f'/{name}', im):
                    raise Exception("Could not write image")


class Decoder:

    batched = False
    channels_name = None
    shape = None
    value = None

    def __init__(self, filename, batched=False):
        ext = basename(filename).split('.')[1]
        self.batched = False
        if ext.upper() == 'TIFF' or ext.upper() == 'TIF':
            inp = imread(filename)
            tiff_data = TiffFile(filename)
            if hasattr(tiff_data, 'shaped_metadata'):
                if tiff_data.shaped_metadata is not None and len(tiff_data.shaped_metadata) > 0:
                    tiff_data = tiff_data.shaped_metadata[0]
                    self.channels_name = [str(c) for c in tiff_data["wavelength"]]
                    self.shape = tiff_data["nrows"], tiff_data["ncols"], tiff_data["nbands"]
            if len(inp.shape) == 3 and np.min(inp.shape) > 1:
                inp = np.transpose(inp, (-1, 0, 1))
            self.batched = batched
            inp = self.concatanate_gray(inp)
            self.value = inp
        elif ext.upper() == 'NPY':
            self.value = np.load(filename)
            self.batched = batched
            if len(self.value.shape) >= 3 and np.min(self.value.shape) > 1:
                if batched and len(self.value.shape) == 3:
                    self.batched = True
                elif batched and len(self.value.shape) >= 3:
                    self.batched = True
                    channel_pos = np.argmin(self.value.shape[-3:])
                    self.value = np.moveaxis(self.value, channel_pos, -3)
                else:
                    self.batched = False
                    channel_pos = np.argmin(self.value.shape)
                    self.value = np.moveaxis(self.value, channel_pos, 0)
            else:
                self.batched = False
        else:
            inp = cv.imread(filename, -1)
            assert inp is not None, f'No Image found at {filename}'
            if inp.shape[-1] == 3:
                inp = self.concatanate_gray(inp)
            if inp.shape[-1] == 3:
                self.value = inp[..., [2, 1, 0]]
            elif inp.shape[-1] == 4:
                if all((inp[..., -1] / 255).flatten().tolist()):
                    self.value = inp[..., [2, 1, 0]]
                else:
                    self.value = inp[..., [2, 1, 0, 3]]
            else:
                self.value = inp
        assert self.value is not None, f'No Image found at {filename}'

    @staticmethod
    def concatanate_gray(image):
        truth = np.var(image, axis=-1) == 0
        if truth.all():
            return image[:, :, :1]
        else:
            return image


