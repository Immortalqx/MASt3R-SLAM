import torch
import torch.nn.functional as F
import mast3r_slam.image as img_utils
from mast3r_slam.config import config
import mast3r_slam_backends


def match(X11, X21, D11, D21, idx_1_to_2_init=None):
    """
    统一入口：进行**迭代投影匹配**（iterative projection matching）。

    作用：给定两幅图像的稠密 3D/描述子预测，返回从图1到图2的像素对应关系与有效掩码。

    参数：
    - X11: (b,h,w,3) 图1的 3D 点/射线图（通常为单位方向或规范坐标）。
    - X21: (b,h,w,3) 图2的 3D 点图。
    - D11: (b,h,w,f) 图1的逐像素描述子。
    - D21: (b,h,w,f) 图2的逐像素描述子。
    - idx_1_to_2_init: (b,h*w) 或 None，线性索引形式的**初始化匹配**（可来自上一帧）。

    返回：
    - idx_1_to_2: (b,h*w) 从图1到图2的线性索引（每个图1像素在图2中的匹配位置）。
    - valid_match2: (b,h*w,1) 有效匹配布尔掩码。
    """
    idx_1_to_2, valid_match2 = match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init)
    return idx_1_to_2, valid_match2


def pixel_to_lin(p1, w):
    """
    将像素坐标 `(u, v)` 转为**线性索引**。

    公式（行主序）：
        \[\; i = u + w \cdot v \;\]
    其中 `w` 为图像宽度；`u∈[0,w-1], v∈[0,h-1]`。

    参数：
    - p1: (..., 2) 最后一维为 (u, v)。
    - w: 宽度。

    返回：
    - idx_1_to_2: 与 `p1` 前置维相同的整型线性索引张量。
    """
    idx_1_to_2 = p1[..., 0] + (w * p1[..., 1])
    return idx_1_to_2


def lin_to_pixel(idx_1_to_2, w):
    """
    将**线性索引**还原为像素坐标 `(u, v)`。

    公式：
        \[\; u = i \bmod w,\quad v = \left\lfloor i / w \right\rfloor \;\]

    参数：
    - idx_1_to_2: (...,) 线性索引。
    - w: 宽度。

    返回：
    - p: (..., 2) 像素坐标 (u, v)。
    """
    u = idx_1_to_2 % w
    v = idx_1_to_2 // w
    p = torch.stack((u, v), dim=-1)
    return p


def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    """
    为**迭代投影**后端准备输入：
    1) 从图1的 3D 点图构造“射线+梯度”的多通道特征图；
    2) 将图2的 3D 点展平成向量，并做单位化；
    3) 根据可选的线性索引初始化，生成图1中的初始像素猜测 `p_init`。

    参数：
    - X11: (b,h,w,3) 图1三维点/方向。
    - X21: (b,h,w,3) 图2三维点。
    - idx_1_to_2_init: (b,h*w) 或 None。

    返回：
    - rays_with_grad_img: (b,h,w,c) 图1的“单位射线 + x/y 梯度”特征（c=3+3+3）。
    - pts3d_norm: (b,h*w,3) 图2的单位化三维点向量。
    - p_init: (b,h*w,2) 图1上的初始像素位置 (u,v)。
    """
    b, h, w, _ = X11.shape
    device = X11.device

    # Ray image
    # 译：把图1的 3D 向量单位化，作为“视线方向图”
    rays_img = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  # (b,c,h,w)

    # 对每个通道计算 x/y 方向梯度，增强几何引导（边缘/纹理）
    gx_img, gy_img = img_utils.img_gradient(rays_img)
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)
    rays_with_grad_img = rays_with_grad_img.permute(0, 2, 3, 1).contiguous()  # (b,h,w,c)

    # 3D points to project
    # 译：将图2的 3D 点展平为向量，并单位化（仅保留方向）
    X21_vec = X21.view(b, -1, 3)
    pts3d_norm = F.normalize(X21_vec, dim=-1)

    # Initial guesses of projections
    # 译：准备初始投影猜测；若无提供，则用“恒等映射”初始化
    if idx_1_to_2_init is None:
        # Reset to identity mapping
        # 译：恒等映射（每个像素匹配到同一位置）
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    p_init = p_init.float()

    return rays_with_grad_img, pts3d_norm, p_init


def match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init=None):
    """
    核心：**迭代投影 + 几何/描述子联合验证** 的匹配器。

    步骤：
    1) `prep_for_iter_proj` 生成：图1的“单位射线+梯度”特征图、图2的单位 3D 向量、初始像素猜测；
    2) 调用 CUDA 后端 `iter_proj(...)` 进行**迭代投影**，得到图1上的像素位置 `p1` 以及几何有效性 `valid_proj2`；
    3) 计算**3D 几何一致性**：取图1在 `p1` 的 3D 与图2对应位置的 3D 做 L2 距离，阈值化得到 `valid_dists2`；
    4) 若配置 `radius>0`，调用 `refine_matches(...)` 在局部邻域内用描述子作细化搜索；
    5) 将像素坐标转换为线性索引返回。

    数学要点：
    - 单位化：\(\hat{r} = X/\|X\|\)；
    - 3D 距离：\(\;\mathrm{dist}(a,b) = \|a-b\|_2\;\)。

    参数/返回同 `match(...)`，其中 `valid_match2` 结合了投影合法性与距离阈值过滤。
    """
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    device = X11.device

    # 预处理：射线+梯度特征、单位 3D、初始像素
    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )

    # 调用 CUDA 后端进行“迭代投影”：从图2的每个 3D 方向，迭代寻找在图1上的最好落点
    p1, valid_proj2 = mast3r_slam_backends.iter_proj(
        rays_with_grad_img,
        pts3d_norm,
        p_init,
        cfg["max_iter"],
        cfg["lambda_init"],
        cfg["convergence_thresh"],
    )
    p1 = p1.long()

    # Check for occlusion based on distances
    # 译：用 3D 距离做“遮挡/错误匹配”剔除（距离过大视为无效）
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)
    # 取出图1在 p1 位置的 3D，并与图2的 3D 做逐像素 L2 距离
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    valid_proj2 = valid_proj2 & valid_dists2

    # 可选：在 (u,v) 周围的 radius 邻域进行描述子细化匹配（更稳健）
    if cfg["radius"] > 0:
        (p1,) = mast3r_slam_backends.refine_matches(
            D11.half(),
            D21.view(b, h * w, -1).half(),
            p1,
            cfg["radius"],
            cfg["dilation_max"],
        )

    # Convert to linear index
    # 译：像素 -> 线性索引，作为系统通用的匹配表示
    idx_1_to_2 = pixel_to_lin(p1, w)

    return idx_1_to_2, valid_proj2.unsqueeze(-1)
