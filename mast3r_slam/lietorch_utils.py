import einops
import lietorch
import torch


def as_SE3(X):
    """
    将输入转换为 SE3 类型（李代数库 lietorch 的 SE3 类）。

    输入情况：
    - 如果 X 已经是 lietorch.SE3，则直接返回；
    - 如果 X 是 Sim3（包含平移 t、四元数 q、尺度 s 的形式），
      则把它的数据分离成 (t, q, s)，丢掉尺度 s，只保留 (t, q)，
      构造等价的 SE3 对象返回。

    公式：
        Sim3: [t, q, s]   (t ∈ R^3, q ∈ R^4, s ∈ R)
        SE3:  [t, q]      (丢弃尺度项)

    作用：
    - 用于在只需要刚体变换（旋转+平移）的地方，把含尺度的 Sim3 退化为 SE3。
    """

    # 如果已经是 SE3 对象，直接返回
    if isinstance(X, lietorch.SE3):
        return X

    # 否则，X 很可能是 Sim3。它内部有 data 属性，形状 (..., 8) = [t(3), q(4), s(1)]
    # einops.rearrange 把数据拍平成 (N, c)，其中 c=8
    t, q, s = einops.rearrange(X.data.detach().cpu(), "... c -> (...) c").split(
        [3, 4, 1], -1
    )

    # 丢弃 s，只用 [t, q] 拼接，构造一个新的 SE3
    T_WC = lietorch.SE3(torch.cat([t, q], dim=-1))
    return T_WC
