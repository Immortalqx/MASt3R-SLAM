import torch
import numpy as np
from mast3r.retrieval.processor import Retriever
from mast3r.retrieval.model import how_select_local

from asmk import io_helpers


class RetrievalDatabase(Retriever):
    """
    基于 **ASMK/IVF** 的关键帧检索数据库（继承自 `Retriever`）。

    作用：
    - 维护倒排文件（IVF）索引与代码本（codebook），支持增量添加关键帧特征；
    - 为给定查询帧返回最相似的历史关键帧索引（可用于回环、全局重定位、选边）。

    记号：
    - 局部特征向量 `q \in \mathbb{R}^d`；代码本中心 `c_m`；
    - 量化（Multiple Assignment, MA）：将每个 `q` 赋给其**最近的 k 个**中心：
      \[
      m^*(q) = \operatorname{arg\,topk}_m\; \big( -\|q - c_m\|^2 \big)\, .
      \]
    - 本实现中**距离矩阵**按等价式高效计算：
      \[
      \|q - c_m\|^2 = \|q\|^2 + \|c_m\|^2 - 2\, q^\top c_m\, .
      \]

    注：相似度聚合与检索由后端 `asmk` 的 `kernel` 与 `inverted file` 完成，这里负责数据准备/调度。
    """

    def __init__(self, modelname, backbone=None, device="cuda"):
        super().__init__(modelname, backbone, device)

        # 创建倒排文件（IVF）构建器与代码本等资源
        self.ivf_builder = self.asmk.create_ivf_builder()

        # 关键帧计数器与 id 列表（自管理，避免与 IVF 内部 id 冲突）
        self.kf_counter = 0
        self.kf_ids = []

        # 查询时统一使用的 dtype/device（减少频繁 cast）
        self.query_dtype = torch.float32
        self.query_device = device
        # 预取代码本中心并放到设备上
        self.centroids = torch.from_numpy(self.asmk.codebook.centroids).to(
            device=self.query_device, dtype=self.query_dtype
        )

    # Mirrors forward_local in extract_local_features from retrieval/model.py
    def prep_features(self, backbone_feat):
        """
        将主干特征做**检索头**的预处理与选择：
        1) 预白化 `prewhiten`；
        2) 线性投影 `projector`，若 `residual=True` 则与预白化特征做残差相加；
        3) 注意力 `attention` 计算每个位置被选中的重要性；
        4) 后白化 `postwhiten`；
        5) 通过 `how_select_local` 选出 top-K 局部特征。

        返回：`topk_features`（形状约为 `[B, K, D]`）。
        """
        retrieval_model = self.model

        # extract_features_and_attention without the encoding!
        backbone_feat_prewhitened = retrieval_model.prewhiten(backbone_feat)
        proj_feat = retrieval_model.projector(backbone_feat_prewhitened) + (
            0.0 if not retrieval_model.residual else backbone_feat_prewhitened
        )
        attention = retrieval_model.attention(proj_feat)
        proj_feat_whitened = retrieval_model.postwhiten(proj_feat)

        # how_select_local in
        topk_features, _, _ = how_select_local(
            proj_feat_whitened, attention, retrieval_model.nfeat
        )

        return topk_features

    def update(self, frame, add_after_query, k, min_thresh=0.0):
        """
        用单帧更新数据库，并可选在**查询之后**再把该帧加入 IVF：
        - 提取可检索的局部特征 `feat = prep_features(frame.feat)`；
        - 若数据库非空，先**查询**得到最相似的 `k` 张历史关键帧；
        - 若 `add_after_query=True`，将本帧的特征增量写入倒排文件。

        返回：`topk_image_inds`（满足阈值的历史关键帧 id 列表，按相似度排序）。
        """
        feat = self.prep_features(frame.feat)
        id = self.kf_counter  # 使用自有的递增 id（否则容易与 IVF 的内部 id 混淆）

        feat_np = feat[0].cpu().numpy()  # 假设一次只处理单帧！
        id_np = id * np.ones(feat_np.shape[0], dtype=np.int64)

        database_size = self.ivf_builder.ivf.n_images
        # print("Database size: ", database_size, self.kf_counter)

        # Only query if already an image
        topk_image_inds = []
        topk_codes = None  # Change this if actualy querying
        if self.kf_counter > 0:
            ranks, ranked_scores, topk_codes = self.query(feat_np, id_np)

            # ranked_scores 是按“候选图片顺序”排列的，需要按 ranks 还原为“真实图片索引顺序”
            scores = np.empty_like(ranked_scores)
            scores[np.arange(ranked_scores.shape[0])[:, None], ranks] = ranked_scores
            scores = torch.from_numpy(scores)[0]

            topk_images = torch.topk(scores, min(k, database_size))

            valid = topk_images.values > min_thresh
            topk_image_inds = topk_images.indices[valid]
            topk_image_inds = topk_image_inds.tolist()

        if add_after_query:
            self.add_to_database(feat_np, id_np, topk_codes)

        return topk_image_inds

    # The reason we need this function is becasue kernel and inverted file not defined when manually updating ivf_builder
    def query(self, feat, id):
        """
        用当前倒排索引对一组局部特征进行检索，返回排序结果与（可选的）量化编码。

        记：`params = asmk.params["query_ivf"]`，内部会：
        1) 量化（multiple assignment）；
        2) 通过 `kernel.aggregate_image` 聚合为图级表示；
        3) 用 `ivf.search` 在倒排文件里**按相似度**检索。

        返回：`ranks, scores, topk_codes`。
        """
        step_params = self.asmk.params.get("query_ivf")

        images2, ranks, scores, topk = self.accumulate_scores(
            self.asmk.codebook,
            self.ivf_builder.kernel,
            self.ivf_builder.ivf,
            feat,
            id,
            params=step_params,
        )

        return ranks, scores, topk

    def add_to_database(self, feat_np, id_np, topk_codes):
        """
        将当前帧的局部特征及其图像 id 增量加入倒排文件；
        如已从 `query` 拿到 `topk_codes`（量化中心索引），可直接复用以加速构建。
        """
        self.add_to_ivf_custom(feat_np, id_np, topk_codes)

        # Bookkeeping
        # 记录并递增关键帧计数
        self.kf_ids.append(id_np[0])
        self.kf_counter += 1

    def quantize_custom(self, qvecs, params):
        """
        自定义**向量量化（MA-Topk）**：对每个查询特征选择最近的 `k` 个代码本中心。

        数学：给定 `qvecs∈\mathbb{R}^{N×d}`、代码本中心 `C=[c_1,...,c_M]`，
        构造距离矩阵 `D∈\mathbb{R}^{N×M}`：
        \[
        D_{n,m} = \|q_n - c_m\|^2
                 = \|q_n\|^2 + \|c_m\|^2 - 2\, q_n^\top c_m\, .
        \]
        然后对每一行取**最小的 k 个**索引。

        返回：`topk.indices`（形状 `N×k`）。
        """
        # Using trick for efficient distance matrix
        l2_dists = (
                torch.sum(qvecs ** 2, dim=1)[:, None]
                + torch.sum(self.centroids ** 2, dim=1)[None, :]
                - 2 * (qvecs @ self.centroids.mT)
        )
        k = params["quantize"]["multiple_assignment"]
        topk = torch.topk(l2_dists, k, dim=1, largest=False)
        return topk.indices

    def accumulate_scores(self, cdb, kern, ivf, qvecs, qimids, params):
        """
        对一批查询图像（其局部特征 `qvecs` 与对应图像 id `qimids`）
        在代码本 `cdb`、核函数 `kern` 与倒排文件 `ivf` 下**累积相似度得分**。

        步骤：
        1) 以 `qimids` 为键把 `qvecs` 切成若干子批（每个图像一批）；
        2) 用 `quantize_custom` 做 MA 量化，得到每个局部特征的 top-k 中心；
        3) `kern.aggregate_image`：把 (向量, 量化索引) 聚合为图级表示；
        4) `ivf.search`：在倒排索引中检索，得到候选图片的 `ranks, scores`；
        5) 累积上述结果并返回，也把量化索引作为 `topk_codes` 一并返回。

        返回：
        - `images2`：查询图像 id 列表；
        - `ranks`：每张查询图像对应候选库中图片的排序索引（按相似度降序）；
        - `scores`：对应的相似度分数矩阵；
        - `topk`：量化得到的中心索引（供后续复用）。
        """
        # 相似度函数闭包，由 kernel 提供具体实现
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])

        acc = []
        slices = list(io_helpers.slice_unique(qimids))
        for imid, seq in slices:
            # Calculate qvecs to centroids distance matrix (without forming diff!)
            qvecs_torch = torch.from_numpy(qvecs[seq]).to(
                device=self.query_device, dtype=self.query_dtype
            )
            # MA 量化 -> 每个局部特征的 top-k 中心索引
            topk_inds = self.quantize_custom(qvecs_torch, params)
            topk_inds = topk_inds.cpu().numpy()
            quantized = (qvecs, topk_inds)

            # 聚合为图级表示，并在倒排文件中检索
            aggregated = kern.aggregate_image(*quantized, **params["aggregate"])
            ranks, scores = ivf.search(
                *aggregated, **params["search"], similarity_func=similarity_func
            )
            acc.append((imid, ranks, scores, topk_inds))

        imids_all, ranks_all, scores_all, topk_all = zip(*acc)

        return (
            np.array(imids_all),
            np.vstack(ranks_all),
            np.vstack(scores_all),
            np.vstack(topk_all),
        )

    def add_to_ivf_custom(self, vecs, imids, topk_codes=None):
        """Add descriptors and cooresponding image ids to the IVF

        :param np.ndarray vecs: 2D array of local descriptors
        :param np.ndarray imids: 1D array of image ids
        :param bool progress: step at which update progress printing (None to disable)
        """
        ivf_builder = self.ivf_builder

        step_params = self.asmk.params.get("build_ivf")

        if topk_codes is None:
            qvecs_torch = torch.from_numpy(vecs).to(
                device=self.query_device, dtype=self.query_dtype
            )
            topk_inds = self.quantize_custom(qvecs_torch, step_params)
            topk_inds = topk_inds.cpu().numpy()
        else:
            # Reuse previously calculated! Only take top 1
            # NOTE: Assuming build params multiple assignment is less than query
            k = step_params["quantize"]["multiple_assignment"]
            topk_inds = topk_codes[:, :k]

        quantized = (vecs, topk_inds, imids)

        aggregated = ivf_builder.kernel.aggregate(
            *quantized, **ivf_builder.step_params["aggregate"]
        )
        ivf_builder.ivf.add(*aggregated)
