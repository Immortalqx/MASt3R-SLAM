import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    """
    一帧图像与其关联数据的容器（最小处理单元）。

    字段说明：
    - frame_id: 数据集中该帧的索引/编号。
    - img: 预处理后的张量图像（通常为网络输入），形状 [3, H, W]。
    - img_shape: 下采样/裁剪后的有效分辨率（H, W）。
    - img_true_shape: 原图的真实分辨率（用于可视化/反投影等）。
    - uimg: 未归一化的图像（0~1），常用于UI/可视化。
    - T_WC: 世界到相机的位姿（Sim3，含尺度），默认单位阵。
    - X_canon: 该帧的“规范点图”（按像素展平后的3D点，单位方向或带尺度）。
    - C: 与 X_canon 对应的一维置信度（逐点）。
    - feat: 稀疏特征（例如每 16x16 patch 的特征向量）。
    - pos: 特征对应的二维网格位置索引（与 feat 对齐）。
    - N: 聚合计数（点图融合时用于统计/平均）。
    - N_updates: 该帧点图被更新的次数（根据 filtering_mode 决定写入策略）。
    - K: 相机内参（仅在 use_calib=True 时使用）。
    """

    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None

    def get_score(self, C):
        """根据配置选择聚合评分方式，用于 best_score 模式的比较。"""
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            # 原注释：Is this slower than mean? Is it worth it?
            # 译：比均值更慢吗？值不值得？（中值更鲁棒，但可能更耗时）
            score = torch.median(C)
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        """
        按“点图融合策略”更新该帧的规范点图与置信度。
        可选策略（config.tracking.filtering_mode）：
        - first：只在第一次更新时写入（其后保持不变）。
        - recent：始终以最新观测覆盖（更跟随当前，但不稳）。
        - best_score：比较 C 的汇总分数（均值/中值），更优者覆盖。
        - indep_conf：逐点比较置信度，更高者逐点替换。
        - weighted_pointmap：对点与置信度按权重做加权平均（逐点）。
        - weighted_spherical：先做笛卡尔->球坐标加权，再还原。
        """
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            # 首次写入：直接克隆保存，并初始化计数
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            # 仅保留“第一次”的观测
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            # 始终使用最新结果覆盖
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            # 依据整体分数比较，优者为王
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            # 逐像素独立比较置信度：新更高则逐点替换
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            # 逐点做加权平均（C 作为权重）
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":
            # 在球坐标系中加权，可在方向性强的场景更稳定

            def cartesian_to_spherical(P):
                # 将 (x, y, z) 转为 (r, phi, theta)
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)  # 方位角 φ ∈ (-π, π]
                theta = torch.acos(z / r)  # 极角 θ ∈ [0, π]
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                # (r, phi, theta) 还原回 (x, y, z)
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        # 更新计数器：记录该帧点图被写入的次数
        self.N_updates += 1
        return

    def get_average_conf(self):
        """返回平均置信度（若尚无 C 则返回 None）。"""
        return self.C / self.N if self.C is not None else None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    """
    工厂函数：从原始图像字典构造 Frame。
    - 负责按配置缩放到固定分辨率（img_size），并保留真值尺寸。
    - 根据 dataset.img_downsample 进一步做整数下采样（影响 uimg 与 img_shape）。
    - 将位姿 T_WC（Sim3）一并写入 Frame。
    """
    img = resize_img(img, img_size)  # 外部工具：返回 dict {img, true_shape, unnormalized_img}
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0  # 转为 0~1 浮点
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # 对未归一化图像与 shape 一致地下采样
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample
    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)
    return frame


class SharedStates:
    """
    进程/线程间共享的“当前帧状态”。
    - 使用 multiprocessing.Manager 创建的共享内存/同步原语。
    - 主要用于重定位（RELOC）、可视化以及某些异步模块读取当前帧信息。
    - 图像以 GPU 张量存放（.share_memory_），uimg 放在 CPU 便于UI访问。
    """

    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()  # 读写锁，保护一致性
        self.paused = manager.Value("i", 0)  # 是否暂停（0/1）
        self.mode = manager.Value("i", Mode.INIT)  # 当前运行模式
        self.reloc_sem = manager.Value("i", 0)  # 重定位“信号量”
        self.global_optimizer_tasks = manager.list()  # 全局优化任务队列（索引）
        self.edges_ii = manager.list()  # 可能用于图优化的边集合
        self.edges_jj = manager.list()

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)  # 假设每 16x16 取一个 patch

        # fmt:off
        # 原注释：shared state for the current frame (used for reloc/visualization)
        # 译：当前帧的共享状态（用于重定位/可视化）
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # fmt: on

    def set_frame(self, frame):
        """把一个 Frame 的数据写入共享缓冲（供其他进程读取）。"""
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            self.X[:] = frame.X_canon
            self.C[:] = frame.C
            self.feat[:] = frame.feat
            self.pos[:] = frame.pos

    def get_frame(self):
        """从共享缓冲构造一个新的 Frame（浅拷贝视图为主）。"""
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img,
                self.img_shape,
                self.img_true_shape,
                self.uimg,
                lietorch.Sim3(self.T_WC),
            )
            frame.X_canon = self.X
            frame.C = self.C
            frame.feat = self.feat
            frame.pos = self.pos
            return frame

    def queue_global_optimization(self, idx):
        """将关键帧索引加入全局优化任务队列。"""
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        """发出一次重定位请求（计数+1）。"""
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        """消耗一次重定位请求（计数-1，若为0则不动）。"""
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1


class SharedKeyframes:
    """
    关键帧环形缓冲（共享内存版本）。
    - 支持 __getitem__/__setitem__ 以 Frame 视图的形式读写。
    - 内部维护 n_size 作为有效长度，buffer 为最大容量。
    - 存放关键帧图像/点图/特征/位姿等，用于跟踪与全局优化。
    """

    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # 原注释：put all of the data into a frame
            # 译：把数组切片打包成一个 Frame 返回
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx],
                self.img_shape[idx],
                self.img_true_shape[idx],
                self.uimg[idx],
                lietorch.Sim3(self.T_WC[idx]),
            )
            kf.X_canon = self.X[idx]
            kf.C = self.C[idx]
            kf.feat = self.feat[idx]
            kf.pos = self.pos[idx]
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # 原注释：set the attributes
            # 译：把 Frame 的各字段写回共享缓冲中对应位置
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            self.X[idx] = value.X_canon
            self.C[idx] = value.C
            self.feat[idx] = value.feat
            self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates
            self.is_dirty[idx] = True  # 标记该关键帧已被更新，便于异步消费
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
