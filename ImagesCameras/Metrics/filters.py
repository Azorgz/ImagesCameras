import torch
import torch.nn.functional as F


def differentiable_box_filter(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Computes a differentiable box filter using 2D average pooling."""
    pad = kernel_size // 2
    return F.avg_pool2d(
        img,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
        count_include_pad=False,
    )


def differentiable_wiener_filter(
    img: torch.Tensor, kernel_size: int = 5, eps: float = 0.05
) -> torch.Tensor:
    """Implements the adaptive Wiener-like filter (Eq. 2-5 in the paper)

    in a fully differentiable manner using PyTorch operations.
    """
    # img shape: (B, C, H, W)
    box_mean = differentiable_box_filter(img, kernel_size)
    box_mean_sq = differentiable_box_filter(img**2, kernel_size)

    # Local variance estimation (m_v in the paper)
    local_var = torch.clamp(box_mean_sq - box_mean**2, min=1e-6)

    # Weight coefficient q (Eq. 3)
    q = local_var / (local_var + eps)

    # Wiener filter approximation output (Eq. 2)
    filtered = box_mean + q * (img - box_mean)
    return filtered


def differentiable_gw_canny_edges(
    img: torch.Tensor, light_condition: str = "weak", steepness: float = 10.0
) -> torch.Tensor:
    """Extracts edges from an IR or visible image using the proposed adaptive
    GW-Canny workflow, keeping the entire pipeline differentiable.
    """
    # 1. Filtering Optimization: Wiener filter for weak light, Gaussian-approx (box) for strong light
    if light_condition.lower() == "weak":
        smoothed = differentiable_wiener_filter(img)
    else:
        smoothed = differentiable_box_filter(img, kernel_size=3)

    # 2. Gradient Magnitude & Direction via Sobel Filters
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device)
        .view(1, 1, 3, 3)
        .repeat(img.shape[1], 1, 1, 1)
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device)
        .view(1, 1, 3, 3)
        .repeat(img.shape[1], 1, 1, 1)
    )

    # Use group convolution to process multi-channel images independently
    grad_x = F.conv2d(smoothed, sobel_x, padding=1, groups=img.shape[1])
    grad_y = F.conv2d(smoothed, sobel_y, padding=1, groups=img.shape[1])

    grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
    grad_orient = torch.atan2(grad_y + 1e-8, grad_x + 1e-8)  # modulo pi

    # Normalize gradient magnitude to [0, 1] range for soft-thresholding stability
    mag_min = grad_mag.amin(dim=(-2, -1), keepdim=True)
    mag_max = grad_mag.amax(dim=(-2, -1), keepdim=True)
    norm_grad = (grad_mag - mag_min) / (mag_max - mag_min + 1e-6)

    # 3. Differentiable Thresholding Approximation (replacing hard Kittler-Illingworth steps)
    # Instead of discrete index selection, we use soft Sigmoid gates centered around
    # an adaptive mean threshold derived from the image statistics.
    adaptive_thresh = norm_grad.mean(dim=(-2, -1), keepdim=True)

    # Soft edge response map using smooth sigmoid activation to preserve gradients
    edge_response = torch.sigmoid(steepness * (norm_grad - adaptive_thresh))
    if edge_response.shape[1] > 1:
        edge_response = torch.max(edge_response, dim=1, keepdim=True)[0]  # Average across channels
    if grad_orient.shape[1] > 1:
        grad_orient = torch.max(grad_orient, dim=1, keepdim=True)[0]
    response = torch.cat([edge_response, grad_orient], dim=1)
    return response