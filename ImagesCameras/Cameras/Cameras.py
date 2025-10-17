import inspect
import math
import os
import warnings
import cv2
import imagesize
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from types import FrameType
from typing import cast, Union

from einops import repeat
from kornia.geometry import PinholeCamera, axis_angle_to_rotation_matrix, transform_points, depth_to_3d_v2, \
    rotation_matrix_to_quaternion, quaternion_to_rotation_matrix, quaternion_to_axis_angle
from torch import Tensor, nn, tensor
from torch.nn import MaxPool2d
from yaml import safe_load

from ..Image import ImageTensor, DepthTensor
from .Sensors import load_sensor
from .base import Data, Sensor
from ..tools.misc import print_tuple
from .utils import intrinsics_parameters_wo_matrix, intrinsics_parameters_from_matrix

ext_available = ['/*.png', '/*.jpg', '/*.jpeg', '/*.tif', '/*.tiff']


class Camera(PinholeCamera):
    """
    A camera object subclassing the camera from Kornia.
    :param path: Always required (valid path or in the future stream url)
    :param device: not mandatory (defined as gpu if a gpu is available)
    :param name: not mandatory, defined as 'BaseCam' by default
    :param id: not mandatory, defined as 'BaseCam' by default
    :param is_positioned: not mandatory, defined as False by default. This parameter is True if the relative pose of the camera in the camera rig is known
    :param aperture: only cosmetic parameter for now
    :param intrinsics: if define it will create a camera with this matrix. If no other parameters is specified, the FOV/f/pixel size/sensor size will be defined by default
    :param f: focal in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or pixel_size...
    :param pixel_size: pixel size in micrometer, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as sensor_size or focal...
    :param sensor_size: sensor size in milimeters, if set with intrinsic, it will fix the other intrinsics values, if the intrinsics matrix is not given another set of parameters will be necessary as focal or pixel_size...
    :param HFOV: horizontal angular field of view in degrees
    :param VFOV: vertical angular field of view in degrees
    :param sensor_resolution: defined by the images find in path resolution. If set it's only a control value emetting a warning if the source images are not at matching resolution
    :param extrinsics: if define the camera will be positionned using that matrix, ignoring the following parameters. If no pose parameter is given, the pose will be canonical by default
    :param x: relative position in x defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param y: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param z: relative position in y defined in meter, in the Camera Setup frame. Overwritten by extrinsics
    :param rx: relative rotation around x defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param ry: relative rotation around y defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param rz: relative rotation around z defined in radian (or degree if in_degree is set to True), in the Camera Setup frame. Overwritten by extrinsics
    :param in_degree: set to True if you express the position angles in degrees
    """
    _is_positioned = False
    _is_ref = False
    _setup = None
    _name = 'BaseCam'
    _id = 'BaseCam'
    _f = None

    def __init__(self,
                 # Data source
                 path: (str or Path) = None,
                 files: list = None,
                 fps=30,
                 # General parameters
                 device=None,
                 name='BaseCam',
                 id='BaseCam',
                 sensor_name=None,
                 is_positioned=False,
                 # Intrinsic args
                 intrinsics: Union[Tensor, np.ndarray] = None,
                 distortion: Union[Tensor, np.ndarray] = None,
                 f: (float or tuple) = None,
                 pixel_size: (float or tuple) = None,
                 sensor_size: tuple = None,
                 HFOV: float = None,
                 VFOV: float = None,
                 aspect_ratio: float = 1.0,
                 sensor_resolution: tuple = None,
                 # Extrinsic args
                 extrinsics: Union[Tensor, np.ndarray] = None,
                 x: float = None,
                 y: float = None,
                 z: float = None,
                 rx: float = None,
                 ry: float = None,
                 rz: float = None,
                 from_file=False,
                 **kwargs) -> None:

        if os.path.isfile(from_file):
            with open(from_file, "r") as file:
                cam = safe_load(file)
            cam['intrinsics'] = np.array(cam['intrinsics'])
            cam['extrinsics'] = np.array(cam['extrinsics'])
            self.__init__(**cam)
        else:
            # General parameters #################################################################
            self._id = id
            self._name = name
            self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.data = Data(path, fps, list_file=files)

            # Intrinsics parameters definition #################################################################
            h, w, modality, im_calib = self._init_resolution_()
            self._modality = modality
            self._im_calib = im_calib
            # Sensor
            if sensor_name is not None and sensor_name != 'NoName':
                self.sensor = load_sensor(sensor_name)
                intrinsics, parameters = self._init_intrinsics_(
                    intrinsics=intrinsics,
                    f=f,
                    pixel_size=self.sensor.pixelSize,
                    sensor_size=self.sensor.size,
                    HFOV=HFOV,
                    VFOV=VFOV,
                    aspect_ratio=self.sensor.aspect_ratio,
                    sensor_resolution=self.sensor.resolution)
            else:
                sensor_resolution = sensor_resolution if sensor_resolution is not None else (h, w)
                intrinsics, parameters = self._init_intrinsics_(
                    intrinsics=intrinsics,
                    f=f,
                    pixel_size=pixel_size,
                    sensor_size=sensor_size,
                    HFOV=HFOV,
                    VFOV=VFOV,
                    aspect_ratio=aspect_ratio,
                    sensor_resolution=sensor_resolution)
                self.sensor = Sensor(parameters['pixel_size'], parameters['sensor_size'], sensor_resolution, modality)
            if self.sensor.resolution[1] != w or self.sensor.resolution[0] != h:
                warnings.warn(
                    f'The specified sensor resolution {self.sensor.resolution} and the source image resolution {(h, w)} are different')

            self._f = parameters['f']

            # Extrinsic parameters definition #################################################################
            if extrinsics is None:
                self.is_positioned = False
                extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
            else:
                extrinsics = torch.tensor(extrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
                self.is_positioned = is_positioned

            h = torch.tensor(self.sensor_resolution[0]).unsqueeze(0).to(self.device)
            w = torch.tensor(self.sensor_resolution[1]).unsqueeze(0).to(self.device)
            super(Camera, self).__init__(intrinsics, extrinsics, h, w)

    def display_src(self):
        for im in self.data():
            image = ImageTensor(im).to_opencv()
            cv2.imshow(f'Camera {self.name} : {self.data.fps}fps', image)
            cv2.waitKey(int(30 / self.data.fps))

    def __str__(self):
        optical_parameter = self.optical_parameter()
        global_parameter = {i: j for i, j in self.save_dict().items() if i not in optical_parameter}
        string1 = '\n'.join([': '.join([str(key), str(v)]) for key, v in global_parameter.items()])
        gap = "\n\noptical parameters : \n".upper()
        string2 = '\n'.join([': '.join([str(key), print_tuple(v)]) for key, v in optical_parameter.items()])
        return string1 + gap + string2

    def clone(self) -> "Camera":
        r"""Return a deep copy of the current object instance."""
        path = self.path if self.path is not None else self.files
        path_key = 'path' if self.path is not None else 'files'
        cam_dict = {'name': self.name,
                    'id': self.id,
                    'f': [self.f[0] * 1e3, self.f[1] * 1e3],
                    'intrinsics': self.intrinsics.squeeze().detach(),
                    'extrinsics': self.extrinsics.squeeze().detach(),
                    'is_ref': self.is_ref,
                    'is_positioned': self.is_positioned,
                    f'{path_key}': path} | self.sensor.save_dict()
        return self.__class__(**cam_dict)

    def optical_parameter(self):
        return {'f': (self.f[0] * 10 ** 3, "mm"),
                'HFOV': (self.HFOV, '°'),
                'VFOV': (self.VFOV, '°')} | self.sensor.repr()

    def save_dict(self):
        path = self.path if self.path is not None else self.files
        path_key = 'path' if self.path is not None else 'files'
        return {'name': self.name,
                'id': self.id,
                'f': [self.f[0] * 1e3, self.f[1] * 1e3],
                'intrinsics': self.intrinsics.detach().squeeze().cpu().numpy().tolist(),
                'extrinsics': self.extrinsics.detach().squeeze().cpu().numpy().tolist(),
                'is_ref': self.is_ref,
                'is_positioned': self.is_positioned,
                f'{path_key}': path} | self.sensor.save_dict()

    def _init_resolution_(self) -> tuple:
        im_path = ''
        im_size = None
        for f in self.data():
            if im_size is None:
                im_size = imagesize.get(f)
            else:
                try:
                    assert im_size == imagesize.get(f), \
                        f'Several images size have been found, image_size enforced at {im_size}'
                except AssertionError:
                    break
            if 'calibration_image' in f:
                if im_path == '':
                    im_path = f
        if im_path == '':
            im_path = self.data(0)[0]
        im_calib = ImageTensor(im_path, device=self.device)
        _, c, h, w = im_calib.shape
        modality = im_calib.modality
        return h, w, modality, im_calib

    def _init_intrinsics_(self, **kwargs):
        if kwargs['intrinsics'] is not None:
            intrinsics, parameters = intrinsics_parameters_from_matrix(**kwargs)
            intrinsics = torch.tensor(intrinsics, dtype=torch.double).unsqueeze(0).to(self.device)
        elif not all([v is None for k, v in list(kwargs.items()) if k != 'aspect_ratio' and k != 'sensor_resolution']):
            parameters = intrinsics_parameters_wo_matrix(**kwargs)
            intrinsics = self._init_intrinsics_matrix(kwargs['sensor_resolution'][0], kwargs['sensor_resolution'][1],
                                                      parameters['f'], parameters['pixel_size'], None)
        else:
            intrinsics = self._init_intrinsics_matrix(kwargs['sensor_resolution'][0], kwargs['sensor_resolution'][1],
                                                      None, None, None)[0]
            kwargs['intrinsics'] = intrinsics
            intrinsics, parameters = intrinsics_parameters_from_matrix(**kwargs)
            intrinsics = intrinsics.unsqueeze(0)
        return intrinsics, parameters

    def _init_intrinsics_matrix(self, h: Union[int, None], w: Union[int, None], f: Union[tuple, list, None],
                                pixel_size: Union[tuple, list, None], center: Union[tuple, list, None],
                                **kwargs) -> Tensor:
        """
        :param h: Height of the images (required if center is None, otherwise ignored)
        :param w: Width of the images (required if center is None, otherwise ignored)
        :param f: Focal length (required if w, h is None)
        :param pixel_size: Pixel size of the images (only used if f is not None)
        :param center: Center of the images (required if w, h is None)
        :return: Init the intrinsic matrix of the camera with default parameter
        """
        if f is not None and pixel_size is not None:
            d = f[0] / pixel_size[0], f[1] / pixel_size[1]
        elif f is not None:
            d = f[0], f[1]
        else:
            d = (np.sqrt(h ** 2 + w ** 2), np.sqrt(h ** 2 + w ** 2))
        if center is not None:
            cx = center[0]
            cy = center[1]
        else:
            cx = (w / 2)
            cy = (h / 2)
        return torch.tensor([[d[0], 0, cx, 0],
                             [0, d[1], cy, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=torch.double).unsqueeze(0).to(self.device)

    def _init_extrinsics_(self, x, y, z, rx, ry, rz, **kwargs) -> Tensor:
        """ :return: init the camera in the space at the position O(0, 0, 0) wo rotation """
        x = x if x is not None else 0
        y = y if y is not None else 0
        z = z if z is not None else 0
        mat_tr = Tensor([x, y, z, 1])
        rx = rx if rx is not None else 0
        ry = ry if ry is not None else 0
        rz = rz if rz is not None else 0
        mat_rot = axis_angle_to_rotation_matrix(Tensor([rx, ry, rz]).unsqueeze(0))
        mat = torch.zeros([1, 4, 4])
        mat[:, :3, :3] = mat_rot
        mat[:, :, -1] = mat_tr
        return mat.inverse().to(dtype=torch.double).to(self.device)

    def reset(self):
        """Only settable by the _del_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '_del_camera_':
            self.is_ref = False
            self.is_positioned = False

    def update_setup(self, camera_ref, cameras) -> None:
        self.setup = cameras
        self.is_ref = self.id == camera_ref
        self.is_positioned = True if self.is_ref else self.is_positioned

    def update_id(self, idx) -> None:
        setattr(self, 'id', f'{self.id}_{idx}')

    def update_pos(self, extrinsics=None, x=None, y=None, z=None, x_pix=None, y_pix=None, rx=None, ry=None, rz=None):
        if extrinsics is None:
            x = x if x is not None else (
                x_pix * self.pixel_size[0] / self.f[0] if x_pix is not None else self.extrinsics[0, 0, 3])
            y = y if y is not None else (
                y_pix * self.pixel_size[1] / self.f[1] if y_pix is not None else self.extrinsics[0, 1, 3])
            self.extrinsics = self._init_extrinsics_(x, y, z, rx, ry, rz)
        else:
            self.extrinsics = extrinsics
        self.is_positioned = True

    def pixel_size_at(self, distance=0):
        """
        :param distance: distance of interest or list of a point of interest coordinates
        :return: The size of a pixel in the image at such a distance
        """
        if isinstance(distance, list) or isinstance(distance, tuple):
            if len(distance) == 3:
                distance = Tensor(distance).to(self.device)
                distance = torch.sqrt(torch.sum((distance - self.center) ** 2))
        if not distance:
            distance = torch.sqrt(torch.sum(self.center ** 2))
        fov_v = 2 * math.tan(self.VFOV / 360 * math.pi) * distance
        fov_h = 2 * math.tan(self.HFOV / 360 * math.pi) * distance
        return fov_h / self.width, fov_v / self.height

    def __getitem__(self, index, autopad=False, **kwargs):
        im_path = self.files[index]
        im = ImageTensor(im_path, device=self.device, normalize=True)
        if autopad:
            temp = torch.zeros(*im.shape[:-2], self.sensor_resolution[1], self.sensor_resolution[0])
            im = im.pad(temp)
        return im

    def random_image(self, autopad=False, **kwargs):
        # list_im = os.listdir(self.path))
        index = torch.randint(0, len(self.files), [1])
        im = self.__getitem__(index, autopad=autopad)
        return im, index

    def depth_to_3D(self, depth: DepthTensor, image_texture: Tensor = None, euclidian_depth=False, **kwargs):
        # Project the points in a 3D space according the camera intrinsics
        points_3D: Tensor = depth_to_3d_v2(depth[:, 0, :, :], self.camera_matrix[0], normalize_points=euclidian_depth)
        # Camera frame to world frame
        points_3D = transform_points(self.extrinsics.inverse()[:, None], points_3D.to(torch.float64))
        # If texture is given, the final point-cloud will be of shape (b x h x w x 6)
        if image_texture is not None:
            assert depth.shape[-2:] == image_texture.shape[-2:]
            points_3D = torch.stack([points_3D, image_texture.permute([0, 2, 3, 1])], dim=3)
        return points_3D

    def project_world_to_image(self, pointcloud: Tensor, image=None, level=1,
                               from_world_frame=True) -> ImageTensor or DepthTensor:
        """
        :param from_world_frame: If True, project first the Pointcloud within the Camera frame
        :param pointcloud: 3D point cloud in the World frame coordinate system
        :param image: image to texture the cloud with. Has to be the same size as the cloud. If None a depth image is printed
        :param level: projection gaussian pyramid level
        :return: ImageTensor
        """
        if from_world_frame:
            pointcloud = transform_points(self.extrinsics[:, None], pointcloud.to(torch.float64))
        cloud_in_camera_frame = torch.concatenate([self.project(pointcloud), pointcloud[:, :, :, -1:]], dim=3)
        if image is not None:
            assert cloud_in_camera_frame.shape[1:3] == image.shape[-2:]
            b, cha, h, w = image.shape
            image_flatten = image.reshape([b, cha, w * h])  # shape b x c x H*W
            projectedImage = ImageTensor(torch.zeros([b, cha, self.sensor_resolution[1],
                                                      self.sensor_resolution[0]])).to(dtype=pointcloud.dtype)
            sample = image_flatten.to(pointcloud.dtype)
        else:
            cha = 1
            b = pointcloud.shape[0]
            projectedImage = DepthTensor(torch.zeros([b, cha, self.sensor_resolution[1],
                                                      self.sensor_resolution[0]])).to(dtype=pointcloud.dtype)
        # Put all the point into a H*W x 3 vector
        c = cloud_in_camera_frame.reshape((cloud_in_camera_frame.shape[0],
                                           cloud_in_camera_frame.shape[1] *
                                           cloud_in_camera_frame.shape[2], 3))  # B x H*W x 3
        # Sort the point by decreasing depth
        indexes = torch.argsort(c[..., -1], descending=True)
        # c_ = c.clone()
        c_sorted = c.clone()
        for j in range(b):
            c_sorted[j, ...] = c[j, indexes[j], :]
        if image is not None:
            for j in range(b):
                sample[j] = sample[j, :, indexes[j]]  # B x C x H*W
        else:
            sample = c_sorted[..., 2:].permute(0, 2, 1)  # B x 1 x H*W
        conv_upsampling = MaxPool2d((3, 5), stride=1, padding=(1, 2), dilation=1)
        for i in reversed(range(level)):
            im_size = self.sensor_resolution[1] // (2 ** i), self.sensor_resolution[0] // (2 ** i)
            # Normalize the point cloud over the sensor resolution
            c_ = c_sorted[..., [0, 1]] / (2 ** i)
            # Transform the landing positions in accurate pixels
            c_x = torch.round(c_[..., :1]).to(torch.int32)
            c_y = torch.round(c_[..., 1:]).to(torch.int32)
            # Remove the point landing outside the image
            mask_out = ((c_x < 0) + (c_x >= im_size[1]) +
                        (c_y < 0) + (c_y >= im_size[0])) != 0
            c_x = c_x * mask_out
            c_y = c_y * mask_out
            image_at_level = torch.zeros([b, cha, im_size[0], im_size[1]], dtype=pointcloud.dtype).to(pointcloud.device)
            image_at_level[:, :, c_y.squeeze(), c_x.squeeze()] = sample
            temp = F.interpolate(image_at_level, projectedImage.shape[-2:])
            projectedImage[temp > 0] = temp[temp > 0]
            projectedImage[projectedImage == 0] = conv_upsampling(projectedImage)[projectedImage == 0]
        return projectedImage

    @property
    def center(self):
        return Tensor([self.tx, self.ty, self.tz]).to(self.device)

    @property
    def f(self):
        return self._f

    @property
    def path(self):
        return self.data.path

    @path.setter
    def path(self, value):
        self.data.update_path(value)

    @property
    def pixel_size(self):
        return self.sensor.pixelSize

    @property
    def sensor_size(self):
        return self.sensor.size

    @property
    def sensor_resolution(self):
        return self.sensor.resolution

    @property
    def aspect_ratio(self):
        return self.sensor.aspect_ratio

    @property
    def modality(self):
        return self.sensor.modality.modality

    @property
    def private_modality(self):
        return self.sensor.modality.private_modality

    @property
    def VFOV(self):
        return 2 * torch.arctan(torch.tensor([self.sensor_size[0] / (2 * self.f[1])])) * 180 / torch.pi

    @property
    def HFOV(self):
        return 2 * torch.arctan(torch.tensor([self.sensor_size[1] / (2 * self.f[0])])) * 180 / torch.pi

    @property
    def pixel_FOV(self):
        return self.VFOV / self.sensor_resolution[0], self.HFOV / self.sensor_resolution[1]

    @property
    def name(self):
        return self._name

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def files(self, *args):
        return self.data(*args)

    @property
    def extrinsics(self):
        return self._extrinsics

    @extrinsics.setter
    def extrinsics(self, value):
        if not isinstance(value, Tensor):
            value = Tensor(value)
        if value.device != self.device:
            value = value.to(self.device)
        if value.dtype != torch.float64:
            value = torch.tensor(value.detach().clone(), dtype=torch.double)
        if value.shape != torch.Size([1, 4, 4]):
            value = value.unsqueeze(0)
        self._extrinsics = value

    @property
    def setup(self):
        return self._setup

    @setup.setter
    def setup(self, setup):
        """Only settable by the update_rig method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_setup' or name == '_reset_':
            self._setup = setup

    @setup.deleter
    def setup(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def im_calib(self):
        return self._im_calib

    @im_calib.setter
    def im_calib(self, im_calib):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__init__':
            self._im_calib = im_calib

    @im_calib.deleter
    def im_calib(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def is_ref(self):
        return self._is_ref

    @is_ref.setter
    def is_ref(self, is_ref):
        """Only settable by the __init__ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'update_camera_ref' or name == 'reset':
            self._is_ref = is_ref

    @is_ref.deleter
    def is_ref(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def is_positioned(self):
        return self._is_positioned

    @is_positioned.setter
    def is_positioned(self, is_positioned):
        """Only settable by the _register_camera_ method"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == 'register_camera' or name == '_reset_' or '__init__' or '__new__':
            self._is_positioned = is_positioned

    @is_positioned.deleter
    def is_positioned(self):
        warnings.warn("The attribute can't be deleted")


class LearnableCamera(Camera, nn.Module):

    def __init__(self, *args,
                 freeze_pos: bool = False, freeze_intrinsics: bool = False,
                 freeze_skew: bool = True, freeze_c: bool = False, freeze_f: bool = False,
                 freeze_x: bool = False, freeze_y: bool = False, freeze_z: bool = False,
                 freeze_rx: bool = False, freeze_ry: bool = False, freeze_rz: bool = False, **kwargs):
        Camera.__init__(self, *args, **kwargs)
        nn.Module.__init__(self)
        self._freeze_pos = freeze_pos
        self._freeze_intrinsics = freeze_intrinsics
        self._freeze_skew = freeze_skew
        self._freeze_c = freeze_c
        self._freeze_f = freeze_f
        self._freeze_x = freeze_x
        self._freeze_y = freeze_y
        self._freeze_z = freeze_z
        self._freeze_rx = freeze_rx
        self._freeze_ry = freeze_ry
        self._freeze_rz = freeze_rz

        r0, rx, ry, rz = rotation_matrix_to_quaternion(self._extrinsics[:, :3, :3]).to(self.device).split(1, -1)
        x, y, z = torch.cat([self._extrinsics[:, :3, 3].unsqueeze(1).to(self.device),
                             torch.zeros([self._extrinsics.shape[0], 1, 3]).to(self.device)], dim=1).split(1, -1)
        fx = tensor([float((self.HFOV / 45).detach().cpu())], dtype=torch.float64, device=self.device)
        fy = tensor([float((self.VFOV / 45).detach().cpu())], dtype=torch.float64, device=self.device)
        cx = self._intrinsics[:, 0, 2] / self.sensor_resolution[1]
        cy = self._intrinsics[:, 1, 2] / self.sensor_resolution[0]
        skew = self._intrinsics[:, 0, 1]
        self._set_learnable_parameters(fx, fy, cx, cy, skew, x, y, z, r0, rx, ry, rz)

    def to(self, device) -> 'LearnableCamera':
        self.device = device
        super().to(device)
        return self

    def _set_learnable_parameters(self, fx=None, fy=None, cx=None, cy=None, skew=None,
                                  x=None, y=None, z=None, r0=None, rx=None, ry=None, rz=None):
        if fx is not None:
            self._fx = nn.Parameter(fx, requires_grad=not self.freeze_f).to(self.device)
        if fy is not None:
            self._fy = nn.Parameter(fy, requires_grad=not self.freeze_f).to(self.device)
        if cx is not None:
            self._cx = nn.Parameter(cx, requires_grad=not self.freeze_c).to(self.device)
        if cy is not None:
            self._cy = nn.Parameter(cy, requires_grad=not self.freeze_c).to(self.device)
        if skew is not None:
            self._skew = nn.Parameter(skew, requires_grad=not self.freeze_skew).to(self.device)
        if x is not None:
            self._x = nn.Parameter(x, requires_grad=not self.freeze_x).to(self.device)
        if y is not None:
            self._y = nn.Parameter(y, requires_grad=not self.freeze_y).to(self.device)
        if z is not None:
            self._z = nn.Parameter(z, requires_grad=not self.freeze_z).to(self.device)
        if r0 is not None:
            self._r0 = nn.Parameter(r0, requires_grad=not self.freeze_pos).to(self.device)
        if rx is not None:
            self._rx = nn.Parameter(rx, requires_grad=not self.freeze_rx).to(self.device)
        if ry is not None:
            self._ry = nn.Parameter(ry, requires_grad=not self.freeze_ry).to(self.device)
        if rz is not None:
            self._rz = nn.Parameter(rz, requires_grad=not self.freeze_rz).to(self.device)

        self.optimizable_parameters = {'fx': self._fx, 'fy': self._fy,
                                       'cx': self._cx, 'cy': self._cy,
                                       'skew': self._skew,
                                       'x': self._x, 'y': self._y, 'z': self._z,
                                       'r0': self._r0, 'rx': self._rx, 'ry': self._ry, 'rz': self._rz}

        return self.optimizable_parameters

    # def update_parameters(self):
    #     self._fx, self._fy = (self.optimizable_parameters['fx'],
    #                           self.optimizable_parameters['fy'])
    #     self._cx, self._cy = (self.optimizable_parameters['cx'],
    #                           self.optimizable_parameters['cy'])
    #     self.skew = self.optimizable_parameters['s']

    @property
    def fx(self) -> Tensor:
        return self.sensor_resolution[1] / (2 * torch.tan((self._fx * torch.pi) / 8))

    @fx.setter
    def fx(self, value: Tensor):
        self._fx = value
        self._f = torch.stack([self.fx * self.pixel_size[1], self.fy * self.pixel_size[0]], dim=0)

    @property
    def fy(self) -> Tensor:
        return self.sensor_resolution[0] / (2 * torch.tan((self._fy * torch.pi) / 8))

    @fy.setter
    def fy(self, value: Tensor):
        self._fy = value
        self._f = torch.stack([self.fx * self.pixel_size[1], self.fy * self.pixel_size[0]], dim=0)

    @property
    def f(self) -> Tensor:
        return self._f

    @f.setter
    def f(self, value: Tensor):
        self._f = value

    @property
    def freeze_f(self):
        return self._freeze_f

    @freeze_f.setter
    def freeze_f(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_f
        self._freeze_f = value
        self._fx.requires_grad = not self._freeze_f
        self._fy.requires_grad = not self._freeze_f

    @property
    def cx(self) -> Tensor:
        cx = self._cx * self.sensor_resolution[1]
        return cx

    @cx.setter
    def cx(self, value: Tensor):
        self._cx = value

    @property
    def cy(self) -> Tensor:
        cy = self._cy * self.sensor_resolution[0]
        return cy

    @cy.setter
    def cy(self, value: Tensor):
        self._cy = value

    @property
    def freeze_c(self):
        return self._freeze_c

    @freeze_c.setter
    def freeze_c(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_c
        self._freeze_c = value
        self._cx.requires_grad = not self._freeze_c
        self._cy.requires_grad = not self._freeze_c

    @property
    def skew(self) -> Tensor:
        return self._skew

    @skew.setter
    def skew(self, value: Tensor):
        self._skew = value

    @property
    def freeze_skew(self):
        return self._freeze_skew

    @freeze_skew.setter
    def freeze_skew(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_skew
        self._freeze_skew = value
        self._skew.requires_grad = not self._freeze_skew

    @property
    def intrinsics(self):
        firstline = torch.stack([self.fx, self.skew, self.cx, torch.tensor([0], device=self.device)], dim=1)
        secondline = torch.stack(
            [torch.tensor([0], device=self.device), self.fy, self.cy, torch.tensor([0], device=self.device)], dim=1)
        thirdline = repeat(torch.tensor([0, 0, 1, 0], device=self.device), 'c -> b c', b=firstline.shape[0])
        fourthline = repeat(torch.tensor([0, 0, 0, 1], device=self.device), 'c -> b c', b=firstline.shape[0])
        return torch.stack([firstline, secondline, thirdline, fourthline], dim=1).to(self.device)  #  b 4 4

    @intrinsics.setter
    def intrinsics(self, value: Tensor):
        assert value.shape[-2] == 3 and value.shape[-1] == 3
        assert value.dtype == torch.float64, "double are needed"
        fx, fy, cx, cy, skew = value[..., :3, :3].to(self.device).split(1, -1)
        self._set_learnable_parameters(fx=fx, fy=fy, cx=cx, cy=cy, skew=skew)

    @property
    def extrinsics(self):
        rotation = self.rotation_matrix  # shape bx3x3
        translation = self.translation_vector.unsqueeze(-1)  # shape bx3x1
        base = repeat(torch.tensor([0, 0, 0, 1]).to(self.device), 'c -> b () c', b=rotation.shape[0])  # shape bx1x4
        return torch.cat([torch.cat([rotation, translation], dim=-1), base], dim=1)

    @extrinsics.setter
    def extrinsics(self, value: Tensor):
        r0, rx, ry, rz = rotation_matrix_to_quaternion(value[..., :3, :3]).to(self.device).split(1, -1)
        x, y, z = torch.cat([value[:, :3, -1].unsqueeze(1).to(self.device),
                             torch.zeros([self._extrinsics.shape[0], 1, 3]).to(self.device)], dim=1).split(1, -1)
        self._set_learnable_parameters(x=x, y=y, z=z, r0=r0, rx=rx, ry=ry, rz=rz)

    @property
    def r0(self) -> Tensor:  # shape bx1
        return self._r0

    @r0.setter
    def r0(self, value: Tensor):
        self._r0 = value

    @property
    def rx(self) -> Tensor:  # shape bx1
        return self._rx

    @rx.setter
    def rx(self, value: Tensor):
        self._rx = value

    @property
    def ry(self) -> Tensor:  # shape bx1
        return self._ry

    @ry.setter
    def ry(self, value: Tensor):
        self._ry = value

    @property
    def rz(self) -> Tensor:  # shape bx1
        return self._rz

    @rz.setter
    def rz(self, value: Tensor):
        self._rz = value

    @property
    def rotation_angles(self):
        return quaternion_to_axis_angle(self.rotation_quaternion)

    @property
    def rotation_matrix(self) -> Tensor:
        return quaternion_to_rotation_matrix(self.rotation_quaternion)

    @property
    def rotation_quaternion(self) -> Tensor:  # shape bx4
        rotation_quaternion = torch.cat([self.r0, self.rx, self.ry, self.rz], dim=-1)
        rotation_quaternion = rotation_quaternion / torch.linalg.vector_norm(rotation_quaternion)
        return rotation_quaternion

    @property
    def x(self) -> Tensor:  # shape bx1
        x = self._x[:, 0] + self._x[:, 1] * 10
        return x

    @x.setter
    def x(self, value: Tensor):
        self._x = value

    @property
    def y(self) -> Tensor:  # shape bx3
        y = self._y[:, 0] + self._y[:, 1] * 10
        return y

    @y.setter
    def y(self, value: Tensor):
        self._y = value

    @property
    def z(self) -> Tensor:  # shape bx3
        z = self._z[:, 0] + self._z[:, 1] * 10
        return z

    @z.setter
    def z(self, value: Tensor):
        self._z = value

    @property
    def translation_vector(self) -> Tensor:  # shape bx3
        translation_vector = torch.cat([self.x, self.y, self.z], dim=-1)
        return translation_vector

    @property
    def freeze_x(self):
        return self._freeze_x

    @freeze_x.setter
    def freeze_x(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_x
        self._freeze_x = value
        self._x.requires_grad = not self._freeze_x

    @property
    def freeze_y(self):
        return self._freeze_y

    @freeze_y.setter
    def freeze_y(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_y
        self._freeze_y = value
        self._y.requires_grad = not self._freeze_y

    @property
    def freeze_z(self):
        return self._freeze_z

    @freeze_z.setter
    def freeze_z(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_z
        self._freeze_z = value
        self._z.requires_grad = not self._freeze_z

    @property
    def freeze_rx(self):
        return self._freeze_rx

    @freeze_rx.setter
    def freeze_rx(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_rx
        self._freeze_rx = value
        self._rx.requires_grad = not self._freeze_rx

    @property
    def freeze_ry(self):
        return self._freeze_ry

    @freeze_ry.setter
    def freeze_ry(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_ry
        self._freeze_ry = value
        self._ry.requires_grad = not self._freeze_ry

    @property
    def freeze_rz(self):
        return self._freeze_rz

    @freeze_rz.setter
    def freeze_rz(self, value: bool = 2):
        value = value != 0 if value != 2 else not self._freeze_rz
        self._freeze_rz = value
        self._rz.requires_grad = not self._freeze_rz

    def freeze_pos(self, value):
        self.freeze_x = value
        self.freeze_y = value
        self.freeze_z = value
        self.freeze_rx = value
        self.freeze_ry = value
        self.freeze_rz = value

    def freeze_intrinsics(self, value):
        self.freeze_f = value
        self.freeze_c = value
        self.freeze_skew = value

    def freeze(self):
        self.freeze_intrinsics(True)
        self.freeze_pos(True)

    def unfreeze(self):
        self.freeze_intrinsics(False)
        self.freeze_pos(False)
