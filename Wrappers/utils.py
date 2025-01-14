import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Float
from kornia.filters import MedianBlur, bilateral_blur, max_blur_pool2d
from kornia.morphology import dilation, closing
from torch import Tensor

from ..Image import DepthTensor, ImageTensor


# from classes.Image import ImageCustom

def projector(cloud, image_size, post_process, image=None, return_occlusion=False, upsample=1 / 2, numpy=False,
              grid=False):
    if not grid:
        if not numpy:
            res = projection(cloud, image_size, image, 3, return_depth=False)
        else:
            cloud_np = cloud.cpu().numpy()
            if image is not None:
                image_np = image.detach().cpu().numpy()
            else:
                image_np = None
            res = project_cloud_to_image_np(cloud_np, image_size, image_np, upsample)
            res = Tensor(res).to(cloud.device)
    else:
        res = project_grid_to_image(cloud, image_size, image)
    return post_process_proj(res, post_process, return_occlusion, image, image_size)


def project_cloud_to_image(cloud, image_size, image=None, upsample=1):
    if image is not None:
        assert cloud.shape[1:3] == image.shape[-2:]
        _, cha, _, _ = image.shape
        if upsample > 1:
            image = F.interpolate(image, scale_factor=upsample)
        image_flatten = torch.tensor(image.flatten(start_dim=2), dtype=cloud.dtype).squeeze(0)  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    if upsample > 1:
        cloud = F.interpolate(cloud.permute([0, 3, 1, 2]), scale_factor=upsample).permute([0, 2, 3, 1])
    c = torch.tensor(cloud.flatten(start_dim=0, end_dim=2))  # H*W x 3
    # Remove the point landing outside the image
    c[:, 0] *= cloud.shape[2] / image_size[1]
    c[:, 1] *= cloud.shape[1] / image_size[0]
    mask_out = ((c[:, 0] < 0) + (c[:, 0] >= (cloud.shape[2] - 1)) + (c[:, 1] < 0) + (
            c[:, 1] >= (cloud.shape[1] - 1))) != 0
    # Sort the point by decreasing depth
    _, indexes = c[:, -1].sort(descending=True, dim=0)
    c[mask_out] = 0
    c_sorted = c[indexes, :]
    if image is not None:
        sample = image_flatten[:, indexes]
    else:
        sample = c_sorted[:, 2:].permute([1, 0])
    # Transform the landing positions in accurate pixels
    c_x = torch.floor(c_sorted[:, 0:1]).to(torch.int)
    dist_c_x = torch.abs(c_x - c_sorted[:, 0:1])
    c_X = torch.ceil(c_sorted[:, 0:1]).to(torch.int)
    dist_c_X = torch.abs(c_X - c_sorted[:, 0:1])
    c_y = torch.floor(c_sorted[:, 1:2]).to(torch.int)
    dist_c_y = torch.abs(c_y - c_sorted[:, 1:2])
    c_Y = torch.floor(c_sorted[:, 1:2]).to(torch.int)
    dist_c_Y = torch.abs(c_Y - c_sorted[:, 1:2])

    rays = [torch.concatenate([c_x, c_y], dim=1),
            torch.concatenate([c_X, c_y], dim=1),
            torch.concatenate([c_x, c_Y], dim=1),
            torch.concatenate([c_X, c_Y], dim=1)]
    dists = [2 - torch.sqrt(dist_c_x ** 2 + dist_c_y ** 2),
             2 - torch.sqrt(dist_c_X ** 2 + dist_c_y ** 2),
             2 - torch.sqrt(dist_c_x ** 2 + dist_c_Y ** 2),
             2 - torch.sqrt(dist_c_X ** 2 + dist_c_Y ** 2)]
    if image is not None:
        result = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
    else:
        result = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
    total_dist = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
    for dist, ray in zip(dists, rays):
        temp = torch.zeros([1, cha, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
        temp_dist = torch.zeros([1, 1, cloud.shape[1], cloud.shape[2]]).to(cloud.dtype).to(cloud.device)
        temp[0, :, ray[:, 1], ray[:, 0]] = sample
        temp_dist[0, 0, ray[:, 1], ray[:, 0]] = 1 / dist[:, 0]
        result += temp * temp_dist
        total_dist += temp_dist
    result[total_dist != 0] /= total_dist[total_dist != 0]
    return result


def project_cloud_to_image_np(cloud, image_size, image=None, upsample=1 / 2):
    if image is not None:
        assert cloud.shape[1:3] == image.shape[-2:]
        b, cha, h, w = image.shape
        if upsample != 1:
            image = F.interpolate(Tensor(image), scale_factor=upsample).numpy()
        image_flatten = image.reshape([cha, w * h * upsample ** 2])  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    if upsample != 1:
        cloud = F.interpolate(Tensor(cloud).permute([0, 3, 1, 2]), scale_factor=upsample).permute([0, 2, 3, 1]).numpy()
    c = cloud.reshape((cloud.shape[0] * cloud.shape[1] * cloud.shape[2], 3))  # H*W x 3
    # Remove the point landing outside the image
    c[:, 0] *= cloud.shape[2] / image_size[1]
    c[:, 1] *= cloud.shape[1] / image_size[0]
    mask_out = ((c[:, 0] < 0) + (c[:, 0] >= (cloud.shape[2] - 1)) + (c[:, 1] < 0) + (
            c[:, 1] >= (cloud.shape[1] - 1))) != 0
    # Sort the point by decreasing depth
    indexes = np.flip(c[:, -1].argsort(0, 'quicksort'))
    c[mask_out] = 0
    c_sorted = c[indexes, :]
    if image is not None:
        sample = image_flatten[:, indexes].transpose([1, 0])
    else:
        sample = c_sorted[:, 2:]
    # Transform the landing positions in accurate pixels
    c_x = np.int32(np.floor(c_sorted[:, 0:1]))
    dist_c_x = np.abs(c_x - c_sorted[:, 0:1])
    c_X = np.int32(np.ceil(c_sorted[:, 0:1]))
    dist_c_X = np.abs(c_X - c_sorted[:, 0:1])
    c_y = np.int32(np.floor(c_sorted[:, 1:2]))
    dist_c_y = np.abs(c_y - c_sorted[:, 1:2])
    c_Y = np.int32(np.floor(c_sorted[:, 1:2]))
    dist_c_Y = np.abs(c_Y - c_sorted[:, 1:2])

    rays = [np.concatenate([c_x, c_y], axis=1),
            np.concatenate([c_X, c_y], axis=1),
            np.concatenate([c_x, c_Y], axis=1),
            np.concatenate([c_X, c_Y], axis=1)]
    dists = [2 - np.sqrt(dist_c_x ** 2 + dist_c_y ** 2),
             2 - np.sqrt(dist_c_X ** 2 + dist_c_y ** 2),
             2 - np.sqrt(dist_c_x ** 2 + dist_c_Y ** 2),
             2 - np.sqrt(dist_c_X ** 2 + dist_c_Y ** 2)]
    if image is not None:
        result = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
    else:
        result = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
    total_dist = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
    for dist, ray in zip(dists, rays):
        temp = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        temp_dist = np.zeros([1, 1, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        temp[0, :, ray[:, 1].squeeze(), ray[:, 0].squeeze()] = sample
        temp_dist[0, 0, ray[:, 1].squeeze(), ray[:, 0].squeeze()] = 1 / dist[:, 0]
        result += temp * temp_dist
        total_dist += temp_dist
    result[total_dist != 0] /= total_dist[total_dist != 0]
    return result


def project_cloud_to_image_new(cloud, image_size, image=None, level=1):
    if image is not None:
        assert cloud.shape[1:3] == image.shape[-2:]
        b, cha, h, w = image.shape
        image_flatten = image.reshape([cha, w * h])  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 3 vector
    c = cloud.reshape((cloud.shape[0] * cloud.shape[1] * cloud.shape[2], 3))  # H*W x 3
    for i in range(level):
        im_size = image_size // (2 ** i)
        # c_ = F.interpolate(Tensor(cloud).permute([0, 3, 1, 2]), scale_factor=1/2**i).permute([0, 2, 3, 1]).numpy()
        c_ = c.copy()
        # Remove the point landing outside the image
        c_[:, 0] *= c_.shape[0] / im_size[1]
        c_[:, 1] *= c_.shape[1] / im_size[0]
        mask_out = ((c_[:, 0] < 0) + (c_[:, 0] >= (cloud.shape[2] - 1)) +
                    (c_[:, 1] < 0) + (c_[:, 1] >= (cloud.shape[1] - 1))) != 0
        # Sort the point by decreasing depth
        indexes = np.flip(c_[:, -1].argsort(0, 'quicksort'))
        c_[mask_out] = 0
        c_sorted = c_[indexes, :]
        if image is not None:
            sample = image_flatten[:, indexes].transpose([1, 0])
        else:
            sample = c_sorted[:, 2:]
        # Transform the landing positions in accurate pixels
        c_x = np.int32(np.floor(c_sorted[:, 0:1]))
        dist_c_x = np.abs(c_x - c_sorted[:, 0:1])
        c_X = np.int32(np.ceil(c_sorted[:, 0:1]))
        dist_c_X = np.abs(c_X - c_sorted[:, 0:1])
        c_y = np.int32(np.floor(c_sorted[:, 1:2]))
        dist_c_y = np.abs(c_y - c_sorted[:, 1:2])
        c_Y = np.int32(np.floor(c_sorted[:, 1:2]))
        dist_c_Y = np.abs(c_Y - c_sorted[:, 1:2])

        rays = [np.concatenate([c_x, c_y], axis=1),
                np.concatenate([c_X, c_y], axis=1),
                np.concatenate([c_x, c_Y], axis=1),
                np.concatenate([c_X, c_Y], axis=1)]
        dists = [2 - np.sqrt(dist_c_x ** 2 + dist_c_y ** 2),
                 2 - np.sqrt(dist_c_X ** 2 + dist_c_y ** 2),
                 2 - np.sqrt(dist_c_x ** 2 + dist_c_Y ** 2),
                 2 - np.sqrt(dist_c_X ** 2 + dist_c_Y ** 2)]
        if image is not None:
            result = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        else:
            result = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        total_dist = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
        for dist, ray in zip(dists, rays):
            temp = np.zeros([1, cha, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
            temp_dist = np.zeros([1, 1, cloud.shape[1], cloud.shape[2]], dtype=cloud.dtype)
            temp[0, :, ray[:, 1].squeeze(), ray[:, 0].squeeze()] = sample
            temp_dist[0, 0, ray[:, 1].squeeze(), ray[:, 0].squeeze()] = 1 / dist[:, 0]
            result += temp * temp_dist
            total_dist += temp_dist
        result[total_dist != 0] /= total_dist[total_dist != 0]
        return result


def post_process_proj(result, post_process, return_occlusion, image, image_size):
    mask = result == 0
    if return_occlusion:
        occ = mask * 1.
    if post_process:
        blur = MedianBlur(post_process)
        kernel = torch.ones([5, 3], device=result.device if isinstance(result, Tensor) else 'cpu')
        res_ = result.clone() if isinstance(result, Tensor) else result.copy()
        res_ = blur(res_)
        if return_occlusion:
            occ = closing(occ, kernel)
            occ = blur(occ)
            occ = ImageTensor(F.interpolate(occ, image_size), device=result.device, permute_image=True).BINARY(
                threshold=0, method='gt', keepchannel=False)
        if image is None:
            res_ = dilation(res_, kernel)
        result[mask] = res_[mask]
    result = F.interpolate(result, image_size)
    if image is not None:
        result = ImageTensor(result, device=result.device, permute_image=True)
    else:
        result = DepthTensor(result, device=result.device, permute_image=True)
    if return_occlusion:
        return result, occ
    else:
        return result


def project_grid_to_image(grid, image_size, image=None):
    if image is not None:
        assert grid.shape[1:3] == image.shape[-2:]
        _, cha, _, _ = image.shape
        image_flatten = torch.tensor(image.flatten(start_dim=2), dtype=grid.dtype).squeeze(0)  # shape c x H*W
    else:
        cha = 1
    # Put all the point into a H*W x 2 vector
    c = torch.abs(grid.flatten(start_dim=0, end_dim=2))  # H*W x 3
    # Remove the point outside the dst frame
    c[:, 0] *= grid.shape[2] / image_size[1]
    mask_out = (c >= image_size[1] - 1) + (c < 0)
    c[mask_out] = 0
    # Sort the point by increasing disp
    _, indexes = torch.abs(c[:, -1]).sort(descending=False, dim=0)
    # c[mask_out] = 0
    c_sorted = c[indexes, :]
    if image is not None:
        sample = image_flatten[:, indexes]
    else:
        sample = c_sorted[:, 2:].permute([1, 0])
    # Transform the landing positions in accurate pixels
    c_x = torch.floor(c_sorted[:, 0:1]).to(torch.int)
    dist_c_x = torch.abs(c_x - c_sorted[:, 0:1])
    c_X = torch.ceil(c_sorted[:, 0:1]).to(torch.int)
    dist_c_X = torch.abs(c_X - c_sorted[:, 0:1])
    c_y = torch.floor(c_sorted[:, 1:2]).to(torch.int)

    rays = [torch.concatenate([c_x, c_y], dim=1),
            torch.concatenate([c_X, c_y], dim=1)]
    dists = [1 - dist_c_x, 1 - dist_c_X]

    if image is not None:
        result = torch.zeros([1, cha, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
    else:
        result = torch.zeros([1, cha, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
    total_dist = torch.zeros([1, cha, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
    for dist, ray in zip(dists, rays):
        temp = torch.zeros([1, 1, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
        temp_dist = torch.zeros([1, 1, grid.shape[1], grid.shape[2]]).to(grid.dtype).to(grid.device)
        temp[0, :, ray[:, 1], ray[:, 0]] = sample
        temp_dist[0, 0, ray[:, 1], ray[:, 0]] = 1 / dist[:, 0]
        result += temp * temp_dist
        total_dist += temp_dist
    result[total_dist != 0] /= total_dist[total_dist != 0]
    return result


def projection(cloud: Float[Tensor, "batch height width xyz"],
               image_size: tuple[int, int],
               image: Float[Tensor, "batch channel height width "] = None,
               level=1, return_depth=True):
    cloud_size = cloud.shape[1:3]
    if image is not None:
        assert cloud_size == image.shape[-2:]
        b, cha, h, w = image.shape
    else:
        b, h, w, xyz = cloud.shape
        image = cloud.permute(0, -1, 1, 2)[:, -1:]
        cha = 1
    device = cloud.device
    image_flatten = rearrange(image, 'b c h w -> b (h w) c')  # shape  c x b*H*W
    # Put all the point into a H*W x 3 vector
    cloud = rearrange(cloud, 'b h w xyz -> b (h w) xyz')  # B x H*W x 3
    cloud_x, cloud_y, cloud_z = cloud.split(1, -1)
    cloud_norm = torch.cat([cloud_x / image_size[1], cloud_y / image_size[0], cloud_z], dim=-1)
    mask_in = ((cloud_norm[:, :, 0] < 0) + (cloud_norm[:, :, 0] >= 1) +
               (cloud_norm[:, :, 1] < 0) + (cloud_norm[:, :, 1] >= 1)) == 0
    # Remove the point landing outside the image
    cloud_norm = cloud_norm * repeat(mask_in, 'b p -> b p xyz', xyz=3)
    # Sort the point by decreasing depth
    indexes = repeat(cloud_norm[..., -1].argsort(stable=True, descending=True, dim=-1), 'b p -> b p xyz', xyz=3)
    cloud_sorted = torch.gather(cloud_norm, 1, indexes)
    c_x, c_y, c_z = cloud_sorted.split(1, -1)
    # cloud_sorted = torch.stack([c_[index] for c_, index in zip(cloud_norm, indexes)])
    sample_sorted = rearrange(torch.gather(image_flatten, 1, indexes[:, :, :cha]), 'b p c -> b c p')
    image_layers = [(image_size[0] // (2 ** i), image_size[1] // (2 ** i)) for i in reversed(range(level))]
    number_points_per_layers = [min(i_s[0]*i_s[1]/(cloud_size[0]*cloud_size[1]), 1)*c_x.shape[1] for i_s in image_layers]
    layer = None
    if return_depth:
        depth = None
    for layer_size, i in zip(image_layers, number_points_per_layers):
        if layer is None:
            if return_depth:
                depth = torch.zeros([b, 1, layer_size[0], layer_size[1]], dtype=image.dtype, device=device)
            layer = torch.zeros([b, cha, layer_size[0], layer_size[1]], dtype=image.dtype, device=device)
        else:
            layer = F.interpolate(layer, size=layer_size, mode='bilinear', align_corners=True)
            if return_depth:
                depth = F.interpolate(depth, size=layer_size, mode='bilinear', align_corners=True)
        c_x_ = torch.floor(layer_size[1] * c_x[:, :i].squeeze(-1)).to(torch.int)
        c_y_ = torch.floor(layer_size[0] * c_y[:, :i].squeeze(-1)).to(torch.int)
        for j in range(b):
            layer[j, :, c_y_[j], c_x_[j]] = sample_sorted[j, :i]
            layer = bilateral_blur(layer, (3, 3), 0.1, (1.5, 1.5))
            if return_depth:
                depth[j, 0, c_y_[j], c_x_[j]] = c_z[j].squeeze(-1).to(torch.float)[j]
                depth = max_blur_pool2d(depth, 3)
    layer = ImageTensor(layer)
    # occlusion = layer.BINARY(threshold=0, method='eq', keepchannel=False)
    if return_depth:
        return depth.squeeze(1), layer
    else:
        return layer
