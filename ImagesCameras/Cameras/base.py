import itertools
import os
from dataclasses import dataclass
from glob import glob
from itertools import chain
from typing import Iterable

import torch

from .utils import homogeneous
from ..Image.base import Modality


class Sensor:
    r"""Data class to represent the camera sensor.

    Args:
        pixelSize: float | Iterable[float, float], micrometer size of a pixel (h, w) or u
        size: Iterable[float, float], size in milimiters of the sensor (h, w)
        resolution: Iterable[int, int]
        modality: str

    Example:
        >>> sensor = Sensor(16, [4, 5], [500, 600], 'Infrared')
    """
    pixelSize: list[float]
    size: list[float]
    resolution: list[int]
    modality: Modality
    name: str = 'NoName'

    def __init__(self, pixelSize: float | Iterable[float],
                 size: Iterable[float],
                 resolution: Iterable[int],
                 modality: Modality,
                 sensor_name: str = None):
        self.pixelSize = [*pixelSize] if isinstance(pixelSize, Iterable) else [pixelSize, pixelSize]
        self.size = list(size)
        self.resolution = list(resolution)
        self.modality = Modality(modality)
        if sensor_name is not None:
            self.name = sensor_name

    def repr(self):
        return {
            'pixel_size': ((self.pixelSize[0] * 10 ** 6, self.pixelSize[1] * 10 ** 6), 'um'),
            'sensor_size': ((self.size[0] * 10 ** 3, self.size[1] * 10 ** 3), 'mm'),
            'width': (self.resolution[1], 'pixels'),
            'height': (self.resolution[0], 'pixels')}

    def save_dict(self):
        return {
            'pixelSize': self.pixelSize,
            'size': self.size,
            'resolution': self.resolution,
            'modality': self.modality.modality,
            'sensor_name': self.name}

    @property
    def aspect_ratio(self):
        return self.pixelSize[0] / self.pixelSize[1]


class Intrinsics:
    r"""Data class to represent the intrinsic parameters of the camera.

    Args:
        fx: float, focal length in x direction
        fy: float, focal length in y direction
        cx: float, principal point in x direction
        cy: float, principal point in y direction
        skew: float, skew coefficient

    Example:
        >>> intrinsics = Intrinsics(1000, 1000, 500, 500, 0)
    """
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float

    def __init__(self, fx=1., fy=1., cx=0., cy=0., skew=0.):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.skew = skew

    def repr(self):
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'skew': self.skew}

    def save_dict(self):
        return {
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
            'skew': self.skew}

    @property
    def matrix(self):
        return torch.tensor([[self.fx, self.skew, self.cx, 0.],
                             [0., self.fy, self.cy, 0.],
                             [0., 0., 1., 0.]]).unsqueeze(0)


class Extrinsics:
    r"""Data class to represent the extrinsic parameters of the camera.

    Args:
        rotation: 3x3 array, rotation matrix
        translation: 3x1 array, translation vector

    Example:
        >>> extrinsics = Extrinsics(torch.eye(3), torch.zeros((3, 1)))
    """
    rotation: torch.tensor
    translation: torch.tensor

    def __init__(self, rotation=None, translation=None):
        self.rotation = rotation if rotation is not None else torch.eye(3)
        self.translation = translation if translation is not None else torch.zeros((3, 1))

    def repr(self):
        return {
            'rotation': self.rotation,
            'translation': self.translation}

    def save_dict(self):
        return {
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist()}

    @property
    def matrix(self):
        rt = torch.hstack((self.rotation, self.translation.reshape(3, 1)))
        return torch.vstack((rt, torch.tensor([0., 0., 0., 1.])))

    def pose(self):
        r_inv = self.rotation.T
        t_inv = -r_inv @ self.translation
        return Extrinsics(r_inv, t_inv)

    def project(self, points):
        """ Project a 3D points from World to Camera coordinates."""
        points = homogeneous(points)
        rt = self.matrix
        return (rt @ points.T).T[:, :3]

    def unproject(self, points):
        """ Unproject a 3D points from Camera to World coordinates."""
        points = homogeneous(points)
        rt = self.pose().matrix
        return (rt @ points.T).T[:, :3]


class Distortion:
    r"""Data class to represent the distortion of the camera.

    Args:
        k1: float, radial distortion coefficient
        k2: float, radial distortion coefficient
        p1: float, tangential distortion coefficient
        p2: float, tangential distortion coefficient
        k3: float, radial distortion coefficient

    Example:
        >>> distortion = Distortion(0.1, 0.01, 0.001, 0.0001, 0.00001)
    """
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

    def __init__(self, k1=0., k2=0., p1=0., p2=0., k3=0.):
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3

    def repr(self):
        return {
            'k1': self.k1,
            'k2': self.k2,
            'p1': self.p1,
            'p2': self.p2,
            'k3': self.k3}

    def save_dict(self):
        return {
            'k1': self.k1,
            'k2': self.k2,
            'p1': self.p1,
            'p2': self.p2,
            'k3': self.k3}

    def distort(self, points):
        r"""Apply distortion to a set of points.

        Args:
            points: Nx2 array of points to be distorted.

        Returns:
            Nx2 array of distorted points.
        """
        x = points[:, 0]
        y = points[:, 1]
        r2 = x ** 2 + y ** 2
        radial = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
        x_distorted = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x ** 2)
        y_distorted = y * radial + self.p1 * (r2 + 2 * y ** 2) + 2 * self.p2 * x * y
        return torch.stack([x_distorted, y_distorted], dim=-1)

    def undistort(self, points, iterations=5):
        r"""Apply undistortion to a set of points using the iterative method.

        Args:
            points: Nx2 array of points to be undistorted.
            iterations: Number of iterations for the undistortion process.

        Returns:
            Nx2 array of undistorted points.
        """
        x_distorted = points[:, 0]
        y_distorted = points[:, 1]
        x_undistorted = x_distorted.copy()
        y_undistorted = y_distorted.copy()

        for _ in range(iterations):
            r2 = x_undistorted ** 2 + y_undistorted ** 2
            radial = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
            x_undistorted = (x_distorted - 2 * self.p1 * x_undistorted * y_undistorted - self.p2 * (
                        r2 + 2 * x_undistorted ** 2)) / radial
            y_undistorted = (y_distorted - self.p1 * (
                        r2 + 2 * y_undistorted ** 2) - 2 * self.p2 * x_undistorted * y_undistorted) / radial

        return torch.stack([x_undistorted, y_undistorted], dim=-1)


class Data:
    r"""Data class to represent the data produced by the camera.

    Args:
        path : str | path | stream url
        fps : int
        file : Generator

    Example:
        >>> data = data('path', 30)
        >>> data.path
        'path'
        >>> data.fps
        30
    """
    path: str
    fps: int
    ext_available = ['/*.png', '/*.jpg', '/*.jpeg', '/*.tif', '/*.tiff']

    def __init__(self, path, fps, list_file=None):
        self.path = path
        self.fps = fps
        if list_file is None:
            if isinstance(path, list):
                gen = []
                for p in path:
                    gen.extend(self._init_path(p))
                self.generator = gen
            else:
                self.generator = list(self._init_path(path))
        else:
            self.generator = list_file

    def _init_path(self, path):
        try:
            if path is not None:
                assert os.path.exists(path)

                def path_generator(p):
                    for ext in self.ext_available:
                        for f in sorted(glob(p + ext)):
                            yield f

                return path_generator(path)
            else:
                return self.generator

        except AssertionError:
            print(f'There is no image file at {path}, a source is necessary to configure a new camera')
            raise AssertionError

    def update_path(self, path):
        self.path = path
        self.reset_generator()

    def reset_generator(self):
        if isinstance(self.path, list):
            gen = []
            for p in self.path:
                gen.append(self._init_path(p))
            self.generator = list(chain(*gen))
        elif self.path is not None:
            self.generator = list(self._init_path(self.path))

    def __call__(self, *args, **kwargs):
        if not args:
            return self.generator
        else:
            gen = []
            self.reset_generator()
            for idx in args:
                gen.append(self.generator[idx])
            return list(gen)
