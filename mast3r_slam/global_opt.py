import lietorch
import torch
from mast3r_slam.config import config
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.geometry import (
    constrain_points_to_ray,
)
from mast3r_slam.mast3r_utils import mast3r_match_symmetric
import mast3r_slam_backends


class FactorGraph:
    """
    全局/局部位姿优化的**因子图管理器**（滑动窗口）。

    作用：
    - 在关键帧之间建立“匹配因子”（边），并存储双向的对应关系与匹配置信度；
    - 准备好用于高斯-牛顿（Gauss-Newton）优化的稀疏结构/观测；
    - 调用后端 CUDA 核心（`mast3r_slam_backends`）执行批量优化，更新关键帧位姿 `T_WC`（Sim3）。

    记号：
    - 关键帧索引集合 `V`；边集合 `E \subseteq V\times V`；
    - `T_WC^i ∈ Sim(3)` 为第 i 个关键帧的世界到相机位姿；
    - `X^i` 为第 i 帧的规范点图（按像素展平）。

    两种残差模型：
    1) **无内参（rays）**：使用“视线方向 + 距离”的差异作为残差。
       对于匹配点对 `(p_i, p_j)`，记
       \[
       \mathrm{rd}(X) = [\,r_x, r_y, r_z, d\,],\quad r=\tfrac{X}{\|X\|},\ d=\|X\|\, .
       \]
       以 `T_{C_j C_i} = (T_WC^j)^{-1} T_WC^i`，有
       \[
       r_m = \mathrm{rd}(X^j)\;-
       \mathrm{rd}(\,T_{C_j C_i}\, X^i\,)\, .
       \]
       代价：
       \[
       J_\text{rays} = \sum_{(i,j)\in E}\! \sum_{m\in \mathcal{M}_{ij}}\!
       \Big( \tfrac{\|\Delta r\|_2^2}{\sigma_{\text{ray}}^2} + \tfrac{\Delta d^2}{\sigma_{\text{dist}}^2} \Big)\, ,
       \]
       其中 `\Delta r` 为三维方向差、`\Delta d` 为距离差，并结合匹配置信度与鲁棒核加权。

    2) **有内参（calib）**：使用像素投影与 log 深度的差异。
       投影模型见 `geometry.project_calib`：
       \[ u= f_x x/z + c_x,\ v = f_y y/z + c_y,\ \log z = \log(z). \]
       残差：
       \[
       r_m = \begin{bmatrix} u^j\\ v^j\\ \log z^j \end{bmatrix}
             -
             \begin{bmatrix} u'\\ v'\\ \log z' \end{bmatrix}, \quad
       [u',v',\log z'] = \Pi\!\big(T_{C_j C_i} X^i; K\big) .
       \]
       代价：
       \[
       J_\text{calib} = \sum_{(i,j)\in E}\! \sum_{m\in \mathcal{M}_{ij}}\!
       \Big( \tfrac{\Delta u^2 + \Delta v^2}{\sigma_{\text{pixel}}^2} + \tfrac{\Delta (\log z)^2}{\sigma_{\text{depth}}^2} \Big)\, ,
       \]
       同样结合置信度与鲁棒核。

    其它：
    - `window_size` 控制滑窗大小；`pin` 表示前若干个关键帧**固定**（解除尺度/位姿自由度的规约），其余参与优化并被回写。
    """

    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda"):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        # 边的端点（i->j）与双向索引、有效性与置信度缓存；均与窗口对齐
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.window_size = self.cfg["window_size"]

        self.K = K

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        """
        为成对关键帧 `(ii, jj)` **建立匹配因子**（双向），并根据匹配质量筛边。

        流程：
        1) 取出两端关键帧的稀疏特征 `feat` 与其网格位置 `pos`；
        2) 进行**对称匹配** `mast3r_match_symmetric`，得到 `idx_i2j / idx_j2i`、有效性与置信度；
        3) 计算双向的匹配质量 `Qj, Qi`，并计算**有效匹配比例**：
           \[
           \mathrm{match\_frac}_j = \frac{\#\{\text{valid in }j\}}{N_j},\quad
           \mathrm{match\_frac}_i = \frac{\#\{\text{valid in }i\}}{N_i} .
           \]
           只有当 `min( match_frac_j, match_frac_i ) ≥ min_match_frac` 才保留该边；
           对**相邻**关键帧（`jj = ii+1`）放宽（始终保留），避免轨迹断裂。

        参数：
        - ii, jj: 列表/张量，候选边的端点索引序列（一一对应）。
        - min_match_frac: 有效匹配比例阈值。
        - is_reloc: 若为重定位阶段，遇到无效边时直接返回 False（不加入）。

        返回：
        - added_new_edges: 是否新增了至少一条有效边。
        """
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        # 将批次索引展开，用于从 Qii 中按 idx_i2j 采样
        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        # 双向综合质量：Q = sqrt(Q_src * Q_tgt)
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        # 质量与有效性门限
        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # 原注释：Saying we need both edge directions to be above thrhreshold to accept either
        # 译：只有**双向**匹配比例都过阈值，才接受该边（更稳健）；相邻帧例外
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        # 过滤掉无效边，并裁剪所有与边对齐的数据
        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        # 追加到图结构缓存
        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        """返回当前图里出现过的**去重关键帧索引**，按升序排序。"""
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    def prep_two_way_edges(self):
        """
        将已有边扩展为**双向列表**，并把对应的索引/掩码/质量一并拼接。

        返回：
        - ii, jj: 拼接后的端点（包含 i→j 与 j→i）
        - idx_ii2jj: 对应 i→j 方向的像素索引映射
        - valid_match: 有效性掩码
        - Q_ii2jj: 综合匹配质量
        """
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        """
        收集参与优化的关键帧的**规范点图 X**、当前位姿 `T_WC` 与平均置信度 `C`。
        """
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs

    def solve_GN_rays(self):
        """
        使用**无内参（rays）残差**进行高斯-牛顿优化，更新关键帧位姿（Sim3）。

        代价函数见类文档 `J_\text{rays}`。实现细节：
        - `pin`: 固定前 `pin` 个关键帧不动（充当锚点，消除尺度/位姿自由度）；
        - 加权：结合匹配质量 `Q` 与置信度 `C`（由后端核实现），并使用鲁棒核（Huber 等）。
        - 调用 CUDA 核心 `mast3r_slam_backends.gauss_newton_rays` 完成批量线性化与求解。
        """
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        max_iter = self.cfg["max_iters"]
        sigma_ray = self.cfg["sigma_ray"]
        sigma_dist = self.cfg["sigma_dist"]
        delta_thresh = self.cfg["delta_norm"]

        # 取出连续的 Sim3 参数向量视图，供 CUDA 核心原地更新
        pose_data = T_WCs.data[:, 0, :]
        mast3r_slam_backends.gauss_newton_rays(
            pose_data,
            Xs,
            Cs,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            sigma_ray,
            sigma_dist,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # 更新关键帧位姿（注意固定前 pin 个不回写）
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])

    def solve_GN_calib(self):
        """
        使用**有内参（calib）残差**进行高斯-牛顿优化，更新关键帧位姿（Sim3）。

        代价函数见类文档 `J_\text{calib}`。实现细节：
        - 先将各帧点**约束到像素射线**上（`constrain_points_to_ray`），减少自由度；
        - 同样固定前 `pin` 个关键帧，调用 CUDA 核心 `gauss_newton_calib`；
        - 核心内部做投影边界/深度有效性检查与加权。
        """
        K = self.K
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx)

        # 将点约束到各自像素射线（保持 z），提升数值稳定性
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]

        img_size = self.frames[0].img.shape[-2:]
        height, width = img_size

        mast3r_slam_backends.gauss_newton_calib(
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        # 回写优化后的关键帧位姿（跳过固定的 pin 个）
        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
