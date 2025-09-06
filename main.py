import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    """
    重定位流程（RELOC）：
    - 使用检索数据库（基于特征/编码）从历史关键帧中找近邻
    - 临时把当前帧加入关键帧集合用于因子构图尝试
    - 通过因子图匹配比例阈值判断是否成功重定位
    - 成功则触发一次全局优化（带标定或仅射线）
    - 注意：这里对 keyframes 的追加/弹出需要在 lock 下进行，避免与可视化等并发冲突
    """
    # 我们会临时 add 再 remove，因此要小心并发；用锁会稍慢但更安全
    with keyframes.lock:
        kf_idx = []
        # 在不修改DB的情况下进行检索，得到候选关键帧索引
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            # 先把当前帧加入关键帧集合，方便构图
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # 转为 list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            # 向因子图添加重定位相关的边（匹配比例阈值控制可靠性）
            if factor_graph.add_factors(
                    frame_idx,
                    kf_idx,
                    config["reloc"]["min_match_frac"],
                    is_reloc=config["reloc"]["strict"],
            ):
                # 若成功，才把当前帧真正写入检索数据库
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                # 将新增关键帧的位姿对齐到匹配到的参考关键帧（提升稳定性）
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                # 失败则撤销追加，回滚关键帧集合
                keyframes.pop_last()
                print("Failed to relocalize")

        # 成功后做一次全局优化（是否联合标定取决于配置）
        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    """
    后端优化进程：
    - 常驻循环，监听系统 Mode（INIT/TRACKING/RELOC/TERMINATED）与任务队列
    - 对新加入的关键帧：构建与近邻（顺序/检索）之间的因子，并进行一次GN优化
    - 把当前因子边（ii/jj）写回共享状态，便于可视化
    """
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        # INIT 或暂停时，后端空转等待
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        # 接到 RELOC 任务：对传入帧执行重定位
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        # 从全局优化任务队列中取第一个待优化关键帧索引
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # ========= 因子图构建（对 idx 关键帧） =========
        kf_idx = []
        # 与前一帧建立顺序连接：n_consec=1 表示仅连上一帧
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        # 召回环路候选（DB 检索），并允许把当前帧写入DB
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        # 仅用于打印：真实的环路边（排除与上一帧的顺序边）
        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        # 去重并移除自身索引，形成 (i,j) 边列表
        kf_idx = set(kf_idx)
        kf_idx.discard(idx)
        kf_idx = list(kf_idx)
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        # 把当前图的边索引写回共享状态，用于前端展示（ii/jj 是 COO 格式的端点）
        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        # 一次 GN 求解（带/不带联合标定）
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        # 任务完成：从队列中弹出
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    # ===== 多进程 & PyTorch 基本配置 =====
    mp.set_start_method("spawn")  # 跨平台安全的启动方式
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 以提升 Ampere+ 上的吞吐
    torch.set_grad_enabled(False)  # 推理/SLAM 不需要梯度
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    # ===== 命令行参数 =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    # ===== 载入配置 =====
    load_config(args.config)
    print(args.dataset)
    print(config)

    # ===== 与可视化进程通信的双向队列 =====
    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    # ===== 数据加载 & 采样 =====
    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    # ===== 可选：载入外部标定并写入数据集/全局配置 =====
    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    # ===== 共享内存中的关键帧容器与系统状态机 =====
    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    # ===== 可视化子进程（可选）=====
    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    # ===== 载入 MASt3R 模型并共享内存 =====
    model = load_mast3r(device=device)
    model.share_memory()  # 允许多进程共享权重，避免重复拷贝

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    # 若配置要求使用标定但数据集未提供，直接退出
    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        # 取出帧级内参矩阵（可能考虑金字塔/畸变已处理）
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # ===== 清理上一轮的轨迹/重建输出文件 =====
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    # ===== 前端跟踪器（位姿/是否建新关键帧/是否触发重定位）=====
    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    # ===== 后端优化进程（构图+GN）=====
    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()

    frames = []

    # ================== 主循环（数据驱动）==================
    while True:
        mode = states.get_mode()
        # 从可视化接收窗口交互消息（暂停/继续/单步/终止等）
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        # 暂停逻辑：若 paused 且不是“下一帧”，则主循环空转
        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        # 数据耗尽：终止
        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        # 取数据集一帧（timestamp, RGB[0..1]）
        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # 取上一帧位姿（或第一帧单位位姿），用于初始化当前帧位姿估计
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        # 构造 Frame 对象（包含图像、金字塔、初始位姿等）
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # 初始化：使用 MASt3R 单目推理获得稠密点图/特征（X:点云, C:特征/置信等）
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            # 将第一帧作为关键帧加入，并触发一次全局优化（后端异步执行）
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # 切换到 TRACKING 模式，并广播当前帧
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            # 追踪：返回是否建新关键帧、匹配信息、是否需要尝试重定位
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            # RELOC 模式下：对当前帧先跑一次单目点图推理，提交重定位请求
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # 单线程模式下，阻塞等待重定位完成（通过信号量）
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            # 条件满足则把当前帧升格为关键帧，并把它的索引提交给后端优化
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # 单线程模式：等待后端处理完该任务再继续
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)

        # 简单 FPS 统计
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    # ================== 收尾：保存结果 ==================
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        # 轨迹：TUM 格式 txt
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        # 稠密重建：PLY（用最后一次窗口中的置信度阈值）
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        # 导出关键帧图像（用于调试/论文可视化）
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        # 可选：把原始逐帧图像落盘
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
