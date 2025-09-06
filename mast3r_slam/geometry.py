import lietorch
import torch


def skew_sym(x):
    """
    构造 3D 向量的**反对称矩阵**（亦称“叉乘矩阵”）。

    作用：给定向量 `x = [x, y, z]^T`，返回矩阵 \([x]_\times\) 使得对任意向量 `v` 有：

        [x]_× v = x × v

    公式：

        [x]_× = \begin{bmatrix}
            0 & -z & \;y \\
            z & 0 & -x \\
           -y & x & 0
        \end{bmatrix}

    参数：
    - x: (..., 3) 形状的张量。
    返回：
    - (..., 3, 3) 形状的反对称矩阵。
    """
    b = x.shape[:-1]
    x, y, z = x.unbind(dim=-1)
    o = torch.zeros_like(x)
    # 逐元素拼接成反对称矩阵
    return torch.stack([o, -z, y, z, o, -x, -y, x, o], dim=-1).view(*b, 3, 3)


def point_to_dist(X):
    """
    计算点到相机光心的**欧氏距离**。

    公式：
        d = \|X\|_2 = \sqrt{x^2 + y^2 + z^2}

    参数：
    - X: (..., 3) 相机坐标系下的 3D 点。
    返回：
    - d: (..., 1) 距离标量。
    """
    d = torch.linalg.norm(X, dim=-1, keepdim=True)
    return d


def point_to_ray_dist(X, jacobian=False):
    """
    将 3D 点分解为**单位视线方向 r**与**深度 d**，并可选返回雅可比。

    定义：
        d = \|X\|_2,  \quad r = X / d
        \quad rd = [r_x, r_y, r_z, d]

    若需要雅可比，给出：
        \frac{\partial r}{\partial X} = \frac{1}{d}\Big(I - \frac{1}{d^2} X X^\top\Big),
        \qquad \frac{\partial d}{\partial X} = r^\top

    参数：
    - X: (..., 3) 3D 点。
    - jacobian: 是否返回雅可比。
    返回：
    - rd: (..., 4) = [r, d]
    - (可选) drd_dX: (..., 4, 3) 对 X 的导数。
    """
    b = X.shape[:-1]

    d = point_to_dist(X)
    d_inv = 1.0 / d
    r = d_inv * X
    rd = torch.cat((r, d), dim=-1)  # 拼为长度4的向量 [r, d]
    if not jacobian:
        return rd
    else:
        d_inv_2 = d_inv ** 2
        I = torch.eye(3, device=X.device, dtype=X.dtype).repeat(*b, 1, 1)
        # ∂r/∂X = (1/d) (I - (1/d^2) X X^T)
        dr_dX = d_inv.unsqueeze(-1) * (
                I - d_inv_2.unsqueeze(-1) * (X.unsqueeze(-1) @ X.unsqueeze(-2))
        )
        # ∂d/∂X = r^T
        dd_dX = r.unsqueeze(-2)
        # 合并为对 [r; d] 的雅可比
        drd_dX = torch.cat((dr_dX, dd_dX), dim=-2)
        return rd, drd_dX


def constrain_points_to_ray(img_size, Xs, K):
    """
    将点 **约束到由其像素坐标定义的成像视线**（保持各自的深度不变）。

    思路：
    1) 根据图像尺寸生成每个像素的坐标 `uv`；
    2) 用相机内参 K 与深度 z，把 `uv` **反投影**成 3D：

        X = z * K^{-1} [u, v, 1]^T

    这样可将 `Xs` 的 (x,y) 纠正到与像素对应的射线上，常用于带内参的优化初始化/约束。

    参数：
    - img_size: (H, W)
    - Xs: (B, N, 3) 或 (1, N, 3) 点集合（仅使用其 z 作为深度）。
    - K: (3,3) 内参矩阵。
    返回：
    - Xs_corr: 与 `Xs` 同形状，落在各自像素射线上的点。
    """
    uv = get_pixel_coords(Xs.shape[0], img_size, device=Xs.device, dtype=Xs.dtype).view(
        *Xs.shape[:-1], 2
    )
    # 按像素的 (u,v) 与原深度 z 反投影到相机坐标
    Xs = backproject(uv, Xs[..., 2:3], K)
    return Xs


def act_Sim3(X: lietorch.Sim3, pC: torch.Tensor, jacobian=False):
    """
    在 **Sim(3)** 群上作用点：`pW = s R pC + t`。

    其中：
    - `R` 为旋转，`t` 为平移，`s` 为尺度。
    - `pC` 为相机坐标点，`pW` 为世界坐标点。

    雅可比（按最常见的小扰动参数顺序拼接）：

        ∂pW/∂t = I_{3×3}
        ∂pW/∂ω ≈ -[pW]_×      （旋转的小扰动用左乘近似，见李代数）
        ∂pW/∂s = pW           （对尺度的导数）

    返回的总雅可比形状为 (..., 3, 7)，按 [t(3), ω(3), s(1)] 拼接。

    参数：
    - X: lietorch.Sim3 变换（包含 s, R, t）。
    - pC: (..., 3) 待变换的点。
    - jacobian: 是否返回雅可比。
    返回：
    - pW: (..., 3)
    - (可选) J: (..., 3, 7)
    """
    pW = X.act(pC)
    if not jacobian:
        return pW
    # 对平移的导数是 I
    dpC_dt = torch.eye(3, device=pW.device).repeat(*pW.shape[:-1], 1, 1)
    # 对旋转的小扰动（李代数）的导数近似为 -[pW]_×
    dpC_dR = -skew_sym(pW)
    # 对尺度的导数：∂(sRp+t)/∂s = (Rp) = pW 在当前坐标下（与实现一致）
    dpc_ds = pW.reshape(*pW.shape[:-1], -1, 1)
    return pW, torch.cat([dpC_dt, dpC_dR, dpc_ds], dim=-1)


def decompose_K(K):
    """
    从内参矩阵中取出 `fx, fy, cx, cy`。

    公式：
        K = \begin{bmatrix}
            f_x & 0 & c_x \\
            0 & f_y & c_y \\
            0 & 0 & 1
        \end{bmatrix}
    """
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    return fx, fy, cx, cy


def project_calib(P, K, img_size, jacobian=False, border=0, z_eps=0.0):
    """
    **有内参投影**：将相机坐标系下的点投影到像素，并输出 `log(z)`，可选返回雅可比和有效掩码。

    投影模型（针孔）：
        u = f_x * x / z + c_x,\quad v = f_y * y / z + c_y,\quad \log z = \log(z)

    对 (u, v, log z) 关于 (x, y, z) 的雅可比：
        \frac{\partial(u,v,\log z)}{\partial(x,y,z)} = \begin{bmatrix}
            f_x/z & 0 & -f_x x/z^2 \\
            0 & f_y/z & -f_y y/z^2 \\
            0 & 0 & 1/z
        \end{bmatrix}

    额外：
    - 仅当 `u,v` 在图像边界内且 `z > z_eps` 时视为有效；
    - 为避免 `log(0)`，对无效深度处将 `logz` 置 0。

    参数：
    - P: (..., 3) 相机坐标下 3D 点。
    - K: (3,3) 内参矩阵。
    - img_size: (H, W)
    - jacobian: 是否返回雅可比。
    - border: 边界留白像素，用于可见性裁剪。
    - z_eps: 深度阈值。

    返回：
    - pz: (..., 3) = [u, v, log z]
    - (可选) dpz_dP: (..., 3, 3) 雅可比
    - valid: (..., 1) 布尔掩码（是否在前方且在图像内）
    """
    b = P.shape[:-1]
    # 将 K 广播到批次维度
    K_rep = K.repeat(*b, 1, 1)

    # 齐次坐标下投影并归一化
    p = (K_rep @ P[..., None]).squeeze(-1)
    p = p / p[..., 2:3]
    p = p[..., :2]

    u, v = p.split([1, 1], dim=-1)
    x, y, z = P.split([1, 1, 1], dim=-1)

    # 检查是否落在图像范围内
    valid_u = (u > border) & (u < img_size[1] - 1 - border)
    valid_v = (v > border) & (v < img_size[0] - 1 - border)
    # 检查是否在相机前方
    valid_z = z > z_eps
    # 综合有效性
    valid = valid_u & valid_v & valid_z

    # 深度取对数；无效处避免 NaN
    logz = torch.log(z)
    invalid_z = torch.logical_not(valid_z)
    logz[invalid_z] = 0.0

    # 输出 [u, v, log z]
    pz = torch.cat((p, logz), dim=-1)

    if not jacobian:
        return pz, valid
    else:
        fx, fy, cx, cy = decompose_K(K)
        z_inv = 1.0 / z[..., 0]
        dpz_dP = torch.zeros(*b + (3, 3), device=P.device, dtype=P.dtype)
        # 按上面的解析式填充雅可比
        dpz_dP[..., 0, 0] = fx
        dpz_dP[..., 1, 1] = fy
        dpz_dP[..., 0, 2] = -fx * x[..., 0] * z_inv
        dpz_dP[..., 1, 2] = -fy * y[..., 0] * z_inv
        dpz_dP *= z_inv[..., None, None]
        dpz_dP[..., 2, 2] = z_inv  # 第三行对应 ∂log z/∂z = 1/z
        return pz, dpz_dP, valid


def backproject(p, z, K):
    """
    **反投影**：由像素 `p=(u,v)` 与深度 `z` 还原相机坐标下 3D 点。

    公式：
        x = (u - c_x)/f_x * z,\quad y = (v - c_y)/f_y * z,\quad z = z

    参数：
    - p: (..., 2) 像素坐标。
    - z: (..., 1) 深度。
    - K: (3,3) 内参矩阵。
    返回：
    - P: (..., 3) 相机坐标点。
    """
    tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
    tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
    # 便于广播：先构造 dP/dz 的列向量，再乘以 z
    dP_dz = torch.empty(p.shape[:-1] + (3, 1), device=z.device, dtype=K.dtype)
    dP_dz[..., 0, 0] = tmp1
    dP_dz[..., 1, 0] = tmp2
    dP_dz[..., 2, 0] = 1.0
    P = torch.squeeze(z[..., None, :] * dP_dz, dim=-1)
    return P


def get_pixel_coords(b, img_size, device, dtype):
    """
    生成批量的**像素网格坐标** `uv`。

    约定：采用 `indexing="xy"`，即 `u` 沿宽度（x 轴），`v` 沿高度（y 轴）。

    参数：
    - b: 批大小（生成 b 份网格）。
    - img_size: (H, W)
    - device, dtype: 张量属性。
    返回：
    - uv: (b, H, W, 2) 整型网格，最后一维为 (u, v)。
    """
    h, w = img_size
    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    uv = torch.stack((u, v), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    uv = uv.to(device=device, dtype=dtype)
    return uv
