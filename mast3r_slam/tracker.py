import torch
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric


class FrameTracker:
    """
    帧跟踪器：
    - 维护当前关键帧集合，与新来的普通帧进行匹配与位姿优化。
    - 既支持无标定的“射线距离”误差（ray distance）优化，也支持有标定的像素+深度误差优化。
    - 位姿变量在 Sim(3) 上优化（包含尺度），以应对尺度漂移。
    """

    def __init__(self, model, frames, device):
        self.cfg = config["tracking"]  # 跟踪相关超参数
        self.model = model  # 用于特征/匹配的模型（Mast3r）
        self.keyframes = frames  # 关键帧管理器/列表
        self.device = device

        self.reset_idx_f2k()

    # 原注释：Initialize with identity indexing of size (1,n)
    # 译：以“恒等索引”进行初始化（形状大致为 (1, n)）。这里置空，首次匹配时由匹配器给出。
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        """
        主流程：
        1) 取最新关键帧，与当前帧进行不对称匹配，得到索引映射和置信度；
        2) 根据是否有内参，准备测量与初始位姿；
        3) 在 Sim(3) 上迭代优化当前帧相对关键帧的位姿；
        4) 用优化后的位姿把当前帧点云/点图映射到关键帧，更新关键帧点图；
        5) 依据匹配质量判断是否生成新关键帧；
        6) 返回（是否新关键帧、可视化/调试数据、是否丢弃）。
        """
        keyframe = self.keyframes.last_keyframe()

        # 与关键帧进行不对称匹配（可利用上一帧到关键帧的索引作为初始化）
        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # 为下一次跟踪保存索引（避免每次从零匹配）
        self.idx_f2k = idx_f2k.clone()

        # 去掉 batch 维度（匹配器通常按批处理返回）
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        # 组合双方的置信度（几何/语义等）为匹配质量 Qk
        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # 注册/位姿估计前，先把当前帧的点图更新为“自体/规范坐标”版本
        frame.update_pointmap(Xff, Cff)

        use_calib = config["use_calib"]  # 是否使用相机内参 K
        img_size = frame.img.shape[-2:]
        if use_calib:
            K = keyframe.K  # 用关键帧的内参做投影模型
        else:
            K = None

        # 准备参与优化的点、位姿与测量（像素与深度），以及置信度/有效性掩码
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # 有效性判据：
        # 使用“平均置信度”阈值和匹配质量阈值，筛选可用于优化的点
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q  # 参与优化的匹配
        valid_kf = valid_match_k & valid_Q  # 用于关键帧更新/统计的匹配

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            # 匹配太少，跳过本帧（不更新位姿、不新建关键帧）
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # 位姿优化：
            if not use_calib:
                # 无内参：基于“射线距离（ray distance）”的几何误差
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                # 有内参：基于像素(u,v) + log深度 的重投影误差
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            # 线性化求解的 Cholesky 可能失败（病态或数值问题）
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        # 写回当前帧的世界位姿（世界->相机）
        frame.T_WC = T_WCf

        # 用估计的相对位姿把关键帧点图更新到当前帧坐标系后再回写
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # 将过滤/变换后的关键帧写回到关键帧管理器末尾
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # 关键帧选择策略：
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        # 统计：当前帧有效匹配里，能覆盖到的关键帧点的“唯一比例”（去重后数量/总数）
        unique_frac_f = (
                torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        # 如果匹配比例或唯一覆盖比例太低，则需要新建关键帧
        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]

        # 如果创建了新关键帧，则清空索引，下一次匹配重新开始
        if new_kf:
            self.reset_idx_f2k()

        return (
            new_kf,
            [
                keyframe.X_canon,  # 关键帧规范坐标点
                keyframe.get_average_conf(),  # 关键帧平均置信度
                frame.X_canon,  # 当前帧规范坐标点
                frame.get_average_conf(),  # 当前帧平均置信度
                Qkf,  # 关键帧侧匹配质量
                Qff,  # 当前帧侧匹配质量
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        """
        根据是否使用内参，准备优化所需的数据：
        - Xf, Xk: 两帧的规范坐标点（可能会被约束到成像射线）
        - T_WCf, T_WCk: 当前帧/关键帧的初始世界位姿
        - Cf, Ck: 两侧点的平均置信度（用于筛选）
        - meas_k: 关键帧的“测量”向量（[u, v, log(z)]），仅在有内参时构造
        - valid_meas_k: 测量的有效掩码（z>eps）
        """
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # 平均置信度（通常来自网络输出的多通道置信度的均值）
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            # 将点约束到对应像素的视线（根据K和图像尺寸得到的规范化射线方向），
            # 目的是减少自由度，提高投影模型的稳定性
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # 构造关键帧像素坐标网格 uv_k，拼接 log 深度 形成测量向量
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # 避免 log(<=0) 的无效计算：仅当深度 z > depth_eps 才有效
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        # 根据匹配索引，把当前帧点与关键帧点一一对齐
        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        """
        一次高斯-牛顿（带 Huber 鲁棒核）的线性化求解：
        - sqrt_info: 每个残差的平方根信息（1/σ），含有效性掩码与置信度权重
        - r: 残差向量（z - h(x)）
        - J: 雅可比（对优化变量的导数）
        返回：增量 tau_j 以及 cost
        """
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )  # Huber 加权：抑制离群点
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # 加权雅可比（展平 batch/点）
        b = (robust_sqrt_info * r).view(-1, 1)  # 加权残差
        H = A.T @ A  # 近似海森
        g = -A.T @ b  # 负梯度
        cost = 0.5 * (b.T @ b).item()

        # Cholesky 分解解正规方程（数值稳定性更好）
        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        """
        无内参情形（不使用投影）：
        - 使用“点到成像射线的距离”作为几何误差：rd_k - rd_f
        - 在 Sim(3) 上优化相对位姿 T_CkCf（关键帧坐标 -> 当前帧坐标）
        """
        last_error = 0
        # 信息矩阵的平方根（含有效性与匹配置信度），ray 与 dist 两类残差拼接
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # 先用两帧的世界位姿构造初始相对位姿（不包含尺度的注释原文，此处为 Sim3）
        T_CkCf = T_WCk.inv() * T_WCf

        # 预计算关键帧侧的射线距离（常量项）
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            # 把当前帧点变换到关键帧坐标系，并拿到对 Sim3 的雅可比
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            # 计算当前帧点相对关键帧光心的射线距离及其对点的雅可比
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # 残差 r = z - h(x)
            r = rd_k - rd_f_Ck
            # 链式法则得到对变量（T_CkCf）的雅可比
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            # 在群上“回代”更新：T <- T ⊞ tau
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            # 收敛判定：相对误差、步长范数、代价下降
            if check_convergence(
                    step,
                    self.cfg["rel_error"],
                    self.cfg["delta_norm"],
                    old_cost,
                    new_cost,
                    tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # 把相对位姿回到世界：T_WCf = T_WCk * T_CkCf
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
            self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        """
        有内参情形：
        - 使用像素投影 (u,v) 与 log 深度的残差。
        - 仅在投影有效（在图像边界内且深度>eps）时计入误差。
        """
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # 初始相对位姿
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            # 当前帧点到关键帧坐标
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            # 进行带边界与深度检查的投影，返回像素+log深度及其雅可比
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            # 投影有效 且 深度测量有效 才参与优化
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # 残差：关键帧测量（uv, log z） 与 当前帧投影 之差
            r = meas_k - pzf_Ck
            # 链式法则得到雅可比
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                    step,
                    self.cfg["rel_error"],
                    self.cfg["delta_norm"],
                    old_cost,
                    new_cost,
                    tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # 回到世界坐标
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
