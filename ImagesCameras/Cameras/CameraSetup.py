import inspect
import os
import warnings
import numpy as np
import torch


from types import FrameType
from typing import cast, Union


from kornia.geometry import relative_transformation, rotation_matrix_to_angle_axis
from oyaml import safe_load, dump
from torch import eye
from torch.linalg import vector_norm

from .StereoSetup import StereoSetup, DepthSetup
from .Cameras import Camera
from ..tools.misc import path_leaf


class CameraSetup:
    """
    A class that manage all the cameras, there relative positions, and the exiting stereo pairs
    The CameraSetup can be initialized from a yaml file or from a list of cameras. It's used as a collection of all the
    cameras called by there respective id (that parameter could differ from the camera's name)
    """
    _cameras = {}
    _cameras_IR = {}
    _cameras_RGB = {}
    _World2Setup = eye(4)


    def __init__(self, *args, device=None, from_file=False, name=None, **kwargs):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if from_file:
            assert os.path.exists(from_file)
            self._init_from_file_(from_file, self.device)
            self.name = path_leaf(from_file)
        else:
            self(*args, **kwargs)
            self.name = name if name is not None else 'Camera Setup'

    def __new__(cls, *args, **kwargs):
        new = super(CameraSetup, cls).__new__(cls)
        new.cameras = {}
        new.cameras_IR = {}
        new.cameras_RGB = {}
        return new

    def __call__(self, *args, print_info=False, **kwargs):
        cameras = []
        if len(args) > 0:
            for cam in args:
                if isinstance(cam, Camera):
                    if cam in self.cameras.values():
                        cameras.append(self._del_camera_(cam, print_info))
                    else:
                        cameras.append(self._add_camera_(cam, print_info))
                elif cam in self.cameras.keys():
                    cameras.append(self._del_camera_(cam, print_info))
                else:
                    warnings.warn('Only a Camera can be added/deleted')
        return cameras

    def __str__(self):
        list_ir = {key: f' ({"Ref" if cam.is_ref else ("In position" if cam.is_positioned else "Position unknown")})'
                   for key, cam
                   in self.cameras_IR.items()}
        list_rgb = {key: f' ({"Ref" if cam.is_ref else ("In position" if cam.is_positioned else "Position unknown")})'
                    for key, cam
                    in self.cameras_RGB.items()}
        string = f'Cameras IR : {", ".join([self.cameras[key].name + f" ({key})" + list_ir[key] for key in self.cameras_IR.keys()])}\n'
        string += f'Cameras RGB : {", ".join([self.cameras[key].name + f" ({key})" + list_rgb[key] for key in self.cameras_RGB.keys()])}\n'
        string += f'The Reference Camera is : {self.camera_ref}'
        return string

    def _init_from_file_(self, path, device=None):
        with open(path, "r") as file:
            conf = safe_load(file)
        if 'World2Setup' in conf.keys():
            self.World2Setup = torch.tensor(conf['World2Setup'], dtype=torch.float32, device=device)
        cameras = {}
        for cam, v in conf['cameras'].items():
            v['intrinsics'] = np.array(v['intrinsics'])
            v['device'] = device
            v['extrinsics'] = np.array(v['extrinsics'])
            cameras[cam] = Camera(**v)
            self(cameras[cam])

    def save(self, path, name='Setup_Camera.yaml'):
        dict_setup = {'cameras': {key: cam.save_dict() for key, cam in self.cameras.items()},
                      'World2Setup': self.World2Setup.detach().cpu().numpy().tolist()}
        name = f'{path}/{name}'
        with open(name, 'w') as file:
            dump(dict_setup, file)

    def _add_camera_(self, camera: Camera, print_info) -> Camera:
        k = 0
        while camera.id in self.cameras.keys():
            camera.update_id(k)
            k += 1
        self.cameras[camera.id] = camera
        if camera.modality != 'Visible':
            self.cameras_IR[camera.id] = camera
        else:
            self.cameras_RGB[camera.id] = camera
        if print_info:
            print(f'The {camera.modality} Camera {camera.name} has been added to the Rig as {camera.id}')
        return camera

    def _del_camera_(self, camera: Union[Camera, str], print_info) -> Camera:
        for key, v in self.cameras.items():
            if camera == key:
                if self.cameras[camera].modality != 'Visible':
                    self.cameras_IR.pop(camera)
                else:
                    self.cameras_RGB.pop(camera)
                camera = self.cameras.pop(camera)
                break
            elif camera is v:
                if self.cameras[camera.id].modality != 'Visible':
                    self.cameras_IR.pop(camera.id)
                else:
                    self.cameras_RGB.pop(camera.id)
                camera = self.cameras.pop(camera.id)
                break
        if print_info:
            print(f'The {camera.modality} Camera {camera.name} has been removed from the Rig')
        return camera

    def _set_default_new_ref_(self, name):
        """Here the variable name specify the camera which CANT be the new Reference"""
        new = None
        for key, cam in self.cameras.items():
            if new is None and key != name:
                new = key
            if cam.is_positioned and key != name:
                new = key
                break
        self.update_camera_ref(new)

    # @torch.no_grad()
    # def recover_pose_from_keypoints(self, cam: Union[Camera, str], *args, ref=None, t=None) -> Camera:
    #     """
    #     This function re-set the extrinsic parameters of a camera using consecutively the fundamental matrix
    #     determination by the 8-points method, then extracting the Essential matrix from it (according the intrinsic is known...)
    #     and finally recovering R and T matrix from Essential.
    #     :param t: The known translation distance for one direction (scaling parameter)
    #     :param cam: Camera to position
    #     :param ref: is no ref is given, the reference camera of the system will be used
    #     :return: the newly positioned camera
    #     """
    #     if args:
    #         cams = [cam, *args]
    #         t = t if (t is not None and len(t) == len(cams)) else None
    #         for idx, c in enumerate(cams):
    #             t_cam = t[idx] if t is not None else t
    #             ref = self.recover_pose_from_keypoints(c, ref=ref, t=t_cam)
    #     else:
    #         if not isinstance(cam, str):
    #             cam = cam.id
    #         if self._check_if_cam_is_in_setup_(cam):
    #             Kpts_gen = KeypointsGenerator(self.device, detector='SIFT', matcher='SNN')
    #             cam_ref = ref if ref is not None and isinstance(ref, str) else self._camera_ref
    #             ref = ref if isinstance(ref, Tensor) else None
    #             assert cam_ref != cam
    #             assert self.cameras[cam_ref].is_positioned
    #
    #             K1, K2 = self.cameras[cam_ref].intrinsics[:, :3, :3].to(torch.float32), \
    #                 self.cameras[cam].intrinsics[:, :3, :3].to(torch.float32)
    #             # Keypoints extraction followings the chosen method
    #             im_dst = self.cameras[cam].im_calib
    #             im_src = self.cameras[cam_ref].im_calib
    #
    #             calib_method = 'auto'
    #             # calib_method = 'manual' if im_src.modality != im_dst.modality else 'auto'
    #             pts_src, pts_dst = Kpts_gen(im_src, im_dst, pts_ref=ref, method=calib_method, min_kpt=20, th=0.85,
    #                                         draw_result=True)
    #             pts_src, pts_dst = pts_src.to(self.device), pts_dst.to(self.device)
    #             # pts_src_opencv, pts_dst_opencv = np.float32(pts_src.squeeze().cpu()), np.float32(pts_dst.squeeze().cpu())
    #
    #             # Computation of the fundamental matrix and display of the epipolar lines
    #             # F_mat, mask = cv.findFundamentalMat(pts_src_opencv, pts_dst_opencv, cv.FM_LMEDS)
    #             # _, H1, H2 = cv.stereoRectifyUncalibrated(pts_src_opencv, pts_dst_opencv, F_mat, im_dst.shape[-2:])
    #             # img1_rectified = ImageTensor(cv.warpPerspective(im_src.opencv(), H1, (im_src.shape[-1], im_src.shape[-2]))[..., [2, 1, 0]])
    #             # img2_rectified = ImageTensor(cv.warpPerspective(im_dst.opencv(), H2, (im_dst.shape[-1], im_dst.shape[-2]))[..., [2, 1, 0]])
    #             # pts_src, pts_dst = Kpts_gen(img1_rectified, img2_rectified, pts_ref=ref, method=calib_method, min_kpt=20, th=0.85)
    #             # pts_src_opencv, pts_dst_opencv = np.float32(pts_src.squeeze().cpu()), np.float32(pts_dst.squeeze().cpu())
    #             # F_mat, mask = cv.findFundamentalMat(pts_src_opencv, pts_dst_opencv, cv.FM_LMEDS)
    #             # _, H1, H2 = cv.stereoRectifyUncalibrated(pts_src_opencv, pts_dst_opencv, F_mat, img1_rectified.shape[-2:])
    #             # img1_rectified = ImageTensor(cv.warpPerspective(img1_rectified.opencv(), H1, (img1_rectified.shape[-1], img1_rectified.shape[-2]))[..., [2, 1, 0]])
    #             # img2_rectified = ImageTensor(cv.warpPerspective(img2_rectified.opencv(), H2, (img2_rectified.shape[-1], img2_rectified.shape[-2]))[..., [2, 1, 0]])
    #             # img1_rectified.show()
    #             # img2_rectified.show()
    #             # im_src_new = ImageTensor(cv.warpPerspective(im_src.opencv(), H1, (im_src.shape[-1]+100, im_src.shape[-2]+100))[..., [2, 1, 0]])
    #             # im_dst_new = ImageTensor(cv.warpPerspective(im_dst.opencv(), H2, (im_dst.shape[-1]+100, im_dst.shape[-2]+100))[..., [2, 1, 0]])
    #             # (im_src_new*0.5 + im_dst_new).show()
    #             # F_mat = cv.normalizeFundamental(F_mat)
    #             # pts_src, pts_dst = pts_src.to(self.device), pts_dst.to(self.device)
    #             # pts_src_norm = normalize_points_with_intrinsics(pts_src, K1)
    #             # pts_dst_norm = normalize_points_with_intrinsics(pts_dst, K2)
    #             F_mat = find_fundamental(pts_src, pts_dst)
    #             F_mat = Tensor(F_mat).to(self.device).unsqueeze(0)
    #             show_epipolar(im_src, im_dst, F_mat, pts_src, pts_dst)
    #
    #             # Extraction of the Essential Matrix from the Fundamental & decomposition into motion
    #             E_mat = essential_from_fundamental(F_mat.squeeze(0), K1, K2)
    #             R, T, pts_3d = motion_from_essential_choose_solution(E_mat, K1, K2, pts_src, pts_dst, mask=None)
    #
    #             if t[0]:
    #                 T = T / T[0, 0, 0] * t[0]
    #             elif t[1]:
    #                 T = T / T[0, 1, 0] * t[1]
    #             elif t[2]:
    #                 T = T / T[0, 2, 0] * t[2]
    #             extrinsics = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    #             extrinsics[0, :3, :3] = R.inverse().squeeze()
    #             extrinsics[0, :3, -1] = -T.squeeze()
    #             self.cameras[cam].update_pos(extrinsics)
    #             return pts_src

    def move_camera_from_its_position(self, name, dx=None, dy=None, dz=None, drx=None, dry=None, drz=None):
        """
        Move a Camera in the Setup from its position to its position + the given delta in each direction/angle
        The units are meters/rad
        :param name: id of the Camera to be moved
        :param dx: displacement in x direction in meter
        :param dy: displacement in y direction in meter
        :param dz: displacement in z direction in meter
        :param drx: rotation around the x-axis in rad
        :param dry: rotation around the y-axis in rad
        :param drz: rotation around the z-axis in rad
        :return: None
        """
        if self._check_if_cam_is_in_setup_(name):
            if name == self.camera_ref:
                self.update_camera_relative_position(name, x=dx, y=dy, z=dz, rx=drx, ry=dry, rz=drz)
            else:
                rx, ry, rz = rotation_matrix_to_angle_axis(self.cameras[name].extrinsics[:, :3, :3]).to(
                    device='cpu').numpy()
                x, y, z = self.cameras[name].translation_vector[0, :, 0].to(device='cpu').numpy()
                self.update_camera_relative_position(name, x=dx + x, y=dy + y, z=dz + z,
                                                     rx=drx + rx, ry=dry + ry, rz=drz + rz)

    def _check_if_cam_is_in_setup_(self, cam: Union[Camera, str]):
        if not isinstance(cam, str):
            name = cam.id
        else:
            name = cam
        try:
            assert name in self.cameras.keys(), "This Camera doesn't exist"
        except AssertionError:
            return False
        return True

    @property
    def World2Setup(self):
        return self._World2Setup

    @World2Setup.setter
    def World2Setup(self, value):
        self._World2Setup = value

    @property
    def pos(self):
        return {key: cam.extrinsics for key, cam in self.cameras.items()}

    @property
    def nb_cameras_IR(self):
        return len(self.cameras_IR)

    @property
    def nb_cameras_RGB(self):
        return len(self.cameras_RGB)

    @property
    def nb_cameras(self):
        return len(self.cameras)

    def __len__(self):
        return self.nb_cameras

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__':
            self._cameras = value

    @cameras.deleter
    def cameras(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def cameras_IR(self):
        return self._cameras_IR

    @cameras_IR.setter
    def cameras_IR(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__':
            self._cameras_IR = value

    @cameras_IR.deleter
    def cameras_IR(self):
        warnings.warn("The attribute can't be deleted")

    @property
    def cameras_RGB(self):
        return self._cameras_RGB

    @cameras_RGB.setter
    def cameras_RGB(self, value):
        """Only settable by the _update_camera_ref_, _del_camera_, _set_default_new_ref_  methods"""
        # Ref: https://stackoverflow.com/a/57712700/
        name = cast(FrameType, cast(FrameType, inspect.currentframe()).f_back).f_code.co_name
        if name == '__new__':
            self._cameras_RGB = value

    @cameras_RGB.deleter
    def cameras_RGB(self):
        warnings.warn("The attribute can't be deleted")
