import torch
import torch.nn.functional as F


def img_gradient(img):
    """
    图像梯度（Scharr 3×3 近似）计算函数。

    作用：对输入图像按通道计算 x/y 方向的一阶导数（边缘强度与方向的基础）。

    输入/输出：
    - 输入形状：`img ∈ R^{B×C×H×W}`。
    - 输出为 `(gx, gy)`，与 `img` 同形状（逐通道）
      其中 `gx`/`gy` 分别是 x/y 方向梯度。

    核心公式：离散卷积形式
        \[
        g_x = I * K_x, \quad g_y = I * K_y
        \]
    其中 `*` 表示二维卷积，`K_x, K_y` 为 3×3 的 Scharr 方向核（已做 1/32 归一化）：
        \[
        K_x = \tfrac{1}{32}
        \begin{bmatrix}
        -3 & 0 & 3 \\
        -10 & 0 & 10 \\
        -3 & 0 & 3
        \end{bmatrix},\quad
        K_y = \tfrac{1}{32}
        \begin{bmatrix}
        -3 & -10 & -3 \\
         0 & \;0 & 0 \\
         3 & 10 & 3
        \end{bmatrix}.
        \]

    备注：若需要梯度幅值与方向，可在外部计算
        \[\; |g| = \sqrt{g_x^2 + g_y^2},\; \theta = \operatorname{atan2}(g_y, g_x).\]

    返回：
    - gx, gy：与 `img` 同形状的梯度张量。
    """
    device = img.device
    dtype = img.dtype
    b, c, h, w = img.shape

    # 构造 x 方向的 Scharr 卷积核（3×3），并做 1/32 归一化
    gx_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    # repeat 到通道维：做“分组卷积”时每个通道使用同一核
    gx_kernel = gx_kernel.repeat(c, 1, 1, 1)  # 形状 (C,1,3,3)

    # 构造 y 方向的 Scharr 卷积核（3×3）
    gy_kernel = (1.0 / 32.0) * torch.tensor(
        [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]],
        requires_grad=False,
        device=device,
        dtype=dtype,
    )
    gy_kernel = gy_kernel.repeat(c, 1, 1, 1)  # 形状 (C,1,3,3)

    # 边界处理：reflect 反射填充，能在边缘保持梯度连续性，避免零填充导致的边缘暗化
    # groups=C：逐通道独立卷积（depthwise），输入 (B,C,H,W) -> 输出 (B,C,H,W)
    gx = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gx_kernel,
        groups=img.shape[1],
    )

    gy = F.conv2d(
        F.pad(img, (1, 1, 1, 1), mode="reflect"),
        gy_kernel,
        groups=img.shape[1],
    )

    return gx, gy
