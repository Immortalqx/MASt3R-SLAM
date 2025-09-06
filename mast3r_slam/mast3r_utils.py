import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa  # 原注释：确保可定位 dust3r 路径；此行只为副作用导入
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching


def load_mast3r(path=None, device="cuda"):
    """
    加载 MASt3R 主模型（AsymmetricMASt3R）并移动到指定设备。

    参数：
    - path: 权重路径；为 None 时使用默认 checkpoint。
    - device: 设备字符串，如 "cuda" 或 "cpu"。

    返回：
    - model: 已载入权重的 AsymmetricMASt3R 模型。
    """
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model


def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    """
    加载检索数据库（RetrievalDatabase），供全局/局部回环或候选对选择时使用。

    参数：
    - mast3r_model: 已加载的 MASt3R 模型（用作特征主干）。
    - retriever_path: 检索权重；为 None 时用默认 checkpoint。
    - device: 设备。

    返回：
    - retriever: RetrievalDatabase 实例。
    """
    retriever_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever


@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    """
    仅执行 **解码+下游头**（不做图像编码）以获得两幅图的稠密预测。

    流程：
    1) `model._decoder(feat1, pos1, feat2, pos2)`：跨图交互的 Transformer 解码；
    2) `model._downstream_head(k, tokens, shape)`：将 token 恢复到目标分辨率，得到：
       - `pts3d`：每像素的 3D 规范点 `X ∈ R^{H×W×3}`；
       - `conf`：逐像素置信度 `C ∈ R^{H×W}`；
       - `desc`：逐像素描述子 `D ∈ R^{H×W×F}`；
       - `desc_conf`：描述子置信度 `Q ∈ R^{H×W}`。

    返回：
    - res1, res2：字典，各含上述键值。
    """
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    # autocast 关闭：下游头通常期待 float32，避免精度不一致
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    """
    按配置 `dataset.img_downsample` 对四类张量做**整型步长下采样**。

    形状约定：
    - C, Q：(... × H × W)
    - X, D：(... × H × W × F)

    说明：
    - 仅做子采样（等间隔抽取），不做插值；
    - `.contiguous()` 保证内存布局连续，便于后续 `view/rearrange`。

    返回：
    - 下采样后的 (X, C, D, Q)。
    """
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        # C 与 Q：最后两维是 H×W
        # X 与 D：最后三维是 H×W×F
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q


@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    """
    **对称推理**：对帧 i 与 j 双向解码，得到 `(i→j)` 与 `(j→i)` 两个方向的预测。

    步骤：
    - 若帧尚未编码，则调用 `model._encode_image(img, true_shape)` 得到 `(feat, pos)`；
    - 通过 `decoder` 两次：`(feat_i, feat_j)` 与 `(feat_j, feat_i)`；
    - 将 4 份输出堆叠为张量 `X, C, D, Q ∈ R^{4×H×W×...}`；
    - 调用 `downsample` 按配置做子采样。

    返回：
    - X, C, D, Q：形状均为 `4×H×W×(F)` 或 `4×H×W`。
    """
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res = [res11, res21, res22, res12]
    # 从字典中取出 3D 点、置信度、描述子及其置信度；仅取 batch=0（单图）
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
):
    """
    **批量对称解码**：一批 `(i_k, j_k)` 成对特征，逐对运行 `decoder`。

    约定/假设：输入批量内每对图像的目标形状 `shape_i[k], shape_j[k]` 已就绪。

    返回：
    - X, C, D, Q：形状为 `4×B×H×W×(F)` 或 `4×B×H×W` 的堆叠结果（先 4 再 B）。
    """
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


@torch.inference_mode
def mast3r_inference_mono(model, frame):
    """
    **单帧自解码**：将同一帧作为 (src, tgt) 输入 decoder，得到两份结果。

    返回：
    - Xii, Cii：分别为 3D 点与置信度的展平版（`(H×W)×3`, `(H×W)×1`）。
    说明：D/Q 在此函数中并未返回，便于轻量跟踪阶段的使用。
    """
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape

    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    # 展平为像素列表：b×H×W×3 -> b×(H*W)×3
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")

    return Xii, Cii


def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    """
    **对称匹配**（批量）：
    - 先调用 `mast3r_decode_symmetric_batch` 得到四份预测：
      `Xii (i→i), Xji (i→j), Xjj (j→j), Xij (j→i)` 以及对应 `D`/`Q`；
    - 将 `(Xii, Xjj)` 作为“源”堆叠，`(Xji, Xij)` 作为“目标”堆叠，调用 `matching.match`；
    - 把返回的索引/有效性拆回两半，得到 `idx_i2j / idx_j2i` 与 `valid_match_j / valid_match_i`；
    - 返回时还会把四份 `Q` 展平成 `(b, H*W, 1)` 的形状供后续加权。

    返回：
    - idx_i2j, idx_j2i: 源到目标的像素索引映射（整型）；
    - valid_match_j, valid_match_i: 双向有效匹配掩码；
    - Qii, Qjj, Qji, Qij: 对应方向的描述子置信度（展平）。
    """
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    # 源 = [Xii; Xjj]，目标 = [Xji; Xij]
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    # 调用匹配器（余弦/欧氏+描述子，具体见 matching.match 实现）
    # tic()
    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)
    # toc("Match")

    # TODO: Avoid this
    #  这里的拼接/拆分可以优化掉一次复制
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )


@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    """
    **非对称推理**：仅做一次 `(i→j)` 解码（相对对称模式更省时）。

    返回：
    - X, C, D, Q：形状为 `2×H×W×(F)` 或 `2×H×W`（分别对 i/j 一份）。
    """
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    """
    **非对称匹配**：只计算 i→j 方向的匹配（常用于在线跟踪）。

    步骤：
    1) 调用 `mast3r_asymmetric_inference` 得到 2 份输出（分别对应 i 与 j 的点/特征）；
    2) 取 `Xii, Xji` 与对应 `Dii, Dji` 调用 `matching.match`；
    3) 可选传入 `idx_i2j_init` 作为**匹配初始化**（加速/稳定）。

    返回：
    - idx_i2j, valid_match_j：从 i 到 j 的像素索引与有效掩码；
    - Xii, Cii, Qii, Xji, Cji, Qji：展平后的点/置信度/描述子置信度（供后续加权/筛选）。
    """
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    # 系统期望的展平形状（b×H×W×C -> b×(H*W)×C）
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")

    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji


def _resize_pil_image(img, long_edge_size):
    """
    根据**长边约束**调整 PIL 图像大小（保持纵横比不变），并选择合适插值器。

    记原图尺寸 `S = max(W, H)`，则新尺寸按下式缩放：
        \[ new\_size = (\operatorname{round}(W·L/S),\; \operatorname{round}(H·L/S)) \]
    其中 `L = long_edge_size`。

    插值策略：
    - 若 `S > L`（缩小）：`LANCZOS`；
    - 否则（放大或等大）：`BICUBIC`。

    返回：
    - 缩放后的 PIL.Image 对象。
    """
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def resize_img(img, size, square_ok=False, return_transformation=False):
    """
    将原始 `numpy` 图像缩放/裁剪到 MASt3R 期望的输入尺寸（224 或 512）。

    规则：
    - `size == 224`：先按**短边=224**等比缩放，再**中心正方形裁剪**到 224×224；
    - `size == 512`：先按**长边=512**等比缩放，再做**中心裁剪**到满足 16 的倍数；若 `square_ok=False` 且图像恰好正方形，则按 3:4 调整裁剪比例（使高=3/4×宽）。

    变量：
    - 返回的 `res` 包含：
      - `img`: 归一化张量（`ImgNorm` 预处理）形状 `[1,3,H,W]`；
      - `true_shape`: 原始 PIL 尺寸（H, W）；
      - `unnormalized_img`: 裁剪后的 `uint8` 图像副本（便于可视化）。
    - 若 `return_transformation=True`，额外返回几何参数 `(scale_w, scale_h, half_crop_w, half_crop_h)`，其中：
        \[
        \begin{aligned}
        &scale\_w = \frac{W_1}{W},\quad scale\_h = \frac{H_1}{H},\\
        &half\_crop\_w = \tfrac{1}{2}\big(W - W'\big),\quad
         half\_crop\_h = \tfrac{1}{2}\big(H - H'\big)
        \end{aligned}
        \]
      `W_1,H_1` 为缩放前尺寸；`W,H` 为等比缩放后的尺寸；`W',H'` 为最终裁剪的尺寸。

    参数：
    - img: `numpy` 的 RGB 图（0~1 浮点）。
    - size: 224 或 512。
    - square_ok: 对 512 模式是否允许保持正方形裁剪。
    - return_transformation: 是否返回几何参数。

    返回：
    - res: 包含标准化后的图像与形状信息的字典；
    - （可选）几何参数四元组。
    """
    assert size == 224 or size == 512
    # numpy -> PIL
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # 原注释：resize short side to 224 (then crop)
        # 译：先把短边缩放到 224（保持比例），再中心正方形裁剪
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # 原注释：resize long side to 512
        # 译：把长边缩放到 512（保持比例）
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        # 使裁剪后的边长是 16 的倍数（与 MASt3R patch/stride 对齐）
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        # 若不希望正方形且图像恰好正方形，则将高设为 3/4 宽（经验设置）
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res
