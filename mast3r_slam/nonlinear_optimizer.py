import math
import torch


def check_convergence(
        iter,
        rel_error_threshold,
        delta_norm_threshold,
        old_cost,
        new_cost,
        delta,
        verbose=False,
):
    """
    收敛判定函数：基于**相对代价下降**与**更新步长范数**的双重阈值。

    记：
        \begin{aligned}
        \Delta J &= J_{old} - J_{new} \\
        r_{dec} &= \left|\frac{\Delta J}{J_{old}}\right| \\
        \|\Delta x\|_2 &= \|\, \texttt{delta} \,\|_2
        \end{aligned}

    判定条件（满足其一即视为收敛）：
        \[\quad r_{dec} < \texttt{rel\_error\_threshold}\quad \text{或}\quad \|\Delta x\|_2 < \texttt{delta\_norm\_threshold}\, .\]

    作用：在非线性最小二乘（如高斯-牛顿/Levenberg-Marquardt）迭代中，用于决定是否提前停止迭代。

    参数：
    - iter: 当前迭代次数（仅用于可选打印）。
    - rel_error_threshold: 相对误差下降阈值。
    - delta_norm_threshold: 参数更新步长的二范数阈值。
    - old_cost: 上一次迭代的代价 J_old。
    - new_cost: 本次迭代的代价 J_new。
    - delta: 本次求解得到的参数增量向量（可为批量）。
    - verbose: 是否打印调试信息。

    返回：
    - converged: bool，是否满足收敛条件。
    """
    cost_diff = old_cost - new_cost  # 代价下降量 ΔJ
    rel_dec = math.fabs(cost_diff / old_cost)  # 相对下降率 |ΔJ / J_old|
    delta_norm = torch.linalg.norm(delta)  # 步长范数 ||Δx||_2

    # 满足任一阈值即认为收敛（稳健与效率折中）
    converged = rel_dec < rel_error_threshold or delta_norm < delta_norm_threshold
    if verbose:
        print(
            f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}"
        )

    # print(f"{iter=} | {new_cost=} {cost_diff=} {rel_dec=} {delta_norm=} | {converged=}")
    return converged


def huber(r, k=1.345):
    """
    **Huber 权重函数**（用于 M-估计/鲁棒加权）。

    对标准化残差 `r`，其**权重**定义为：
        \[
        w(r) = \begin{cases}
            1, & |r| < k \\
            \dfrac{k}{|r|}, & |r| \ge k
        \end{cases}
        \]

    说明：当残差较小（|r|<k）时按二范数处理；当残差较大时按线性增长抑制（相当于把大残差的影响“截断/减弱”）。本实现直接返回权重 `w`，通常在**迭代再加权最小二乘（IRLS）**中以 `\sqrt{w}` 作为信息矩阵的缩放因子。

    参数：
    - r: 残差（可为张量，支持广播）。
    - k: 截断阈值（默认 1.345）。

    返回：
    - w: 与 `r` 同形状的权重张量。
    """
    unit = torch.ones((1), dtype=r.dtype, device=r.device)
    r_abs = torch.abs(r)
    mask = r_abs < k
    # |r|<k 用 1， 否则用 k/|r|
    w = torch.where(mask, unit, k / r_abs)
    return w


def tukey(r, t=4.6851):
    """
    **Tukey biweight（双二次）权重函数**，更强地抑制离群点。

    对标准化残差 `r`，其**权重**定义为：
        \[
        w(r) = \begin{cases}
            \big(1 - (r/t)^2\big)^2, & |r| < t \\
            0, & |r| \ge t
        \end{cases}
        \]

    说明：Tukey 在 |r|≥t 时直接给 0 权重（完全忽略），因此对极端离群点更鲁棒；但也可能丢失有用信息，适合离群比例较高或噪声厚尾的场景。

    参数：
    - r: 残差（可为张量，支持广播）。
    - t: 截断阈值（默认 4.6851）。

    返回：
    - w: 与 `r` 同形状的权重张量。
    """
    zero = torch.tensor(0.0, dtype=r.dtype, device=r.device)
    r_abs = torch.abs(r)
    tmp = 1 - torch.square(r_abs / t)
    tmp2 = tmp * tmp
    # |r|<t 用 (1-(r/t)^2)^2，否则 0
    w = torch.where(r_abs < t, tmp2, zero)
    return w
