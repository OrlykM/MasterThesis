r"""
Replays a recorded EuRoC-style dataset as a live UDP frame stream, using the
same VSLAM-pose + FDE-depth pairing logic as occupancy_3d_map.py's dataset
mode. This is the fastest way to exercise the streaming path end-to-end
without real live hardware: point it at a dataset run + VSLAM/FDE outputs
you already have, and it feeds occupancy_3d_map.py --mode udp exactly the
frames it would get from a live source.

Usage:
    python3 udp_replay_sender.py \
        --dataset_root ~/MasterThesis/datasets/euroc/2026-04-01_..._quadrotor \
        --vslam_dir ../../results/dpvo \
        --fde_dir ../../results/zoedepth \
        --host 127.0.0.1 --port 9000 --rate 10
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from euroc_io import load_gt_trajectory, load_camera_intrinsics, load_image_timestamps, pose_row_to_T
from compare_reconstruction_3d import (
    build_evo_trajectory, evo_traj_to_array, looks_like_frame_index, looks_like_raw_nanoseconds,
    remap_pseudo_timestamps, load_vslam_stride, load_tum, find_depth_file, load_depth_frame,
)
from evo.core import sync as evo_sync
import depth_calibration as depth_calib
from udp_streaming import FrameSender


def load_stream(dataset_root, vslam_dir, fde_dir, n_frames=80, max_time_diff=0.15,
                 pred_depth_scale=1.0, with_image=True, with_landmarks=True,
                 landmark_window=0, verbose=True):
    """Готує все потрібне, щоб програвати записаний набір як живий потік, і
    віддає генератор кадрів. Винесено окремо, бо цим користуються і
    `udp_replay_sender.py`, і `udp_demo.py`.

    -> (K, generator), де generator дає словники з полями frame_id, timestamp,
    depth, T_wc, pose_x, image, landmarks, name.
    """
    gt = load_gt_trajectory(os.path.join(dataset_root, "state_groundtruth_estimate0", "data.csv"))
    est = load_tum(os.path.join(vslam_dir, "poses.txt"))
    frames = load_image_timestamps(os.path.join(dataset_root, "cam0", "data.csv"))
    if looks_like_frame_index(est):
        est = remap_pseudo_timestamps(est, frames, load_vslam_stride(vslam_dir, default=1))
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9

    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(
        build_evo_trajectory(gt), build_evo_trajectory(est), max_diff=max_time_diff)
    # Перетворення Sim(3) беремо з ПЕРШОГО і єдиного вирівнювання: evo вирівнює
    # траєкторію на місці, тож повторний виклик дав би одиничне перетворення, і
    # розріджені точки лишились би в сирій системі VSLAM.
    r_a, t_a, s_a = traj_est_sync.align(traj_ref_sync, correct_scale=True)
    gt_m, est_m = evo_traj_to_array(traj_ref_sync), evo_traj_to_array(traj_est_sync)
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    intr = load_camera_intrinsics(os.path.join(dataset_root, "cam0", "sensor.yaml"))
    K, cam_w, cam_h = intr["K"], intr["width"], intr["height"]
    seg = np.linalg.norm(np.diff(gt_m[:, 1:4], axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg)])

    anchors = None
    if with_landmarks:
        anchors = depth_calib.SparseMapAnchors.from_vslam_dir(vslam_dir)
        if anchors is None:
            if verbose:
                print(f"[replay] у {vslam_dir} немає ні landmarks.txt, ні "
                      f"dpvo_sparse_map.ply: шлю кадри без розріджених точок")
        else:
            anchors.points = depth_calib.sim3_transform(anchors.points, r_a, t_a, s_a)
            if landmark_window > 0:
                anchors.attach_trajectory(est_m[:, 0], est_m[:, 1:4],
                                           window_frames=landmark_window)
            if verbose:
                print(f"[replay] {len(anchors.points)} розріджених точок VSLAM")

    step = max(1, len(frames) // n_frames)
    chosen = frames[::step][:n_frames]

    def gen():
        for frame_id, (frame_t, fname) in enumerate(chosen):
            j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
            j_est = int(np.argmin(np.abs(est_t - frame_t)))
            if (abs(gt_t[j_gt] - frame_t) > max_time_diff
                    or abs(est_t[j_est] - frame_t) > max_time_diff):
                continue
            pred_path = find_depth_file(fde_dir, fname)
            if pred_path is None:
                continue
            depth = load_depth_frame(pred_path) * pred_depth_scale
            if depth.shape != (cam_h, cam_w):
                depth = np.array(Image.fromarray(depth.astype(np.float32))
                                 .resize((cam_w, cam_h), Image.BILINEAR))
            image = None
            if with_image:
                ip = os.path.join(dataset_root, "cam0", "data", fname)
                if os.path.exists(ip):
                    im = Image.open(ip).convert("L")
                    if im.size != (cam_w, cam_h):
                        im = im.resize((cam_w, cam_h), Image.BILINEAR)
                    image = np.asarray(im, dtype=np.uint8)
            lm = anchors.for_frame(est_m[j_est, 0]) if anchors is not None else None
            yield {"frame_id": frame_id, "timestamp": frame_t, "name": fname,
                   "depth": depth, "T_wc": pose_row_to_T(est_m[j_est]),
                   "pose_x": float(cum_dist[j_gt]), "image": image, "landmarks": lm}

    return K, gen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--vslam_dir", required=True)
    ap.add_argument("--fde_dir", required=True)
    ap.add_argument("--host", required=True, help="occupancy_3d_map.py --listen_host")
    ap.add_argument("--port", type=int, required=True, help="occupancy_3d_map.py --listen_port")
    ap.add_argument("--max_time_diff", type=float, default=0.15)
    ap.add_argument("--n_frames", type=int, default=80)
    ap.add_argument("--pred_depth_scale", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=10.0, help="frames per second to send at "
                                                               "(0 = as fast as possible)")
    ap.add_argument("--send_intrinsics", action="store_true",
                     help="include K in every frame packet instead of relying on the "
                          "receiver's --sensor_yaml")
    ap.add_argument("--v1", action="store_true",
                     help="слати старий формат кадру, тобто лише глибину і позу, без "
                          "зображення і розріджених точок. Тоді приймач не зможе побудувати "
                          "карту довіри, лише карту зайнятості")
    ap.add_argument("--no_image", action="store_true",
                     help="не слати зображення. Без нього на приймачі не працює вето неба")
    ap.add_argument("--no_landmarks", action="store_true",
                     help="не слати розріджені точки VSLAM. Без них немає калібрування, тобто "
                          "немає і ваг довіри: саме цей випадок варто побачити для порівняння")
    ap.add_argument("--landmark_window", type=int, default=0,
                     help="слати лише точки, приписані до +/- N поз навколо поточної, а не всю "
                          "хмару. 0 означає всю хмару, як і в офлайновому режимі, де вужче "
                          "вікно було виміряно шкідливим")
    args = ap.parse_args()

    gt_traj_path = os.path.join(args.dataset_root, "state_groundtruth_estimate0", "data.csv")
    cam_csv_path = os.path.join(args.dataset_root, "cam0", "data.csv")
    sensor_yaml_path = os.path.join(args.dataset_root, "cam0", "sensor.yaml")
    poses_path = os.path.join(args.vslam_dir, "poses.txt")

    gt = load_gt_trajectory(gt_traj_path)
    est = load_tum(poses_path)
    frames = load_image_timestamps(cam_csv_path)
    if looks_like_frame_index(est):
        stride = load_vslam_stride(args.vslam_dir, default=1)
        est = remap_pseudo_timestamps(est, frames, stride)
    elif looks_like_raw_nanoseconds(est, gt):
        est = est.copy()
        est[:, 0] = est[:, 0] / 1e9

    traj_ref = build_evo_trajectory(gt)
    traj_est = build_evo_trajectory(est)
    traj_ref_sync, traj_est_sync = evo_sync.associate_trajectories(traj_ref, traj_est, max_diff=args.max_time_diff)
    r_a, t_a, s_a = traj_est_sync.align(traj_ref_sync, correct_scale=True)
    gt_m = evo_traj_to_array(traj_ref_sync)
    est_m = evo_traj_to_array(traj_est_sync)
    gt_t, est_t = gt_m[:, 0], est_m[:, 0]

    intr = load_camera_intrinsics(sensor_yaml_path)
    K, cam_w, cam_h = intr["K"], intr["width"], intr["height"]

    xyz = gt_m[:, 1:4]
    seg_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    idx_step = max(1, len(frames) // args.n_frames)
    chosen = frames[::idx_step][:args.n_frames]

    # Розріджені точки VSLAM у тій самій системі, що й пози: те саме
    # перетворення Sim(3), яким вирівняли траєкторію. На борту цю роль виконує
    # інерціальний модуль, тут це одноразове відновлення масштабу.
    anchors = None
    if not args.no_landmarks and not args.v1:
        anchors = depth_calib.SparseMapAnchors.from_vslam_dir(args.vslam_dir)
        if anchors is None:
            print(f"[replay] у {args.vslam_dir} немає ні landmarks.txt, ні "
                  f"dpvo_sparse_map.ply: шлю кадри без розріджених точок")
        else:
            # Перетворення взяте з того САМОГО вирівнювання, що й пози. Повторний
            # виклик align() тут давав би одиничне перетворення, бо evo вирівнює
            # траєкторію на місці, і точки лишались би в сирій системі VSLAM.
            anchors.points = depth_calib.sim3_transform(anchors.points, r_a, t_a, s_a)
            if args.landmark_window > 0:
                anchors.attach_trajectory(est_m[:, 0], est_m[:, 1:4],
                                           window_frames=args.landmark_window)
            print(f"[replay] {len(anchors.points)} розріджених точок VSLAM")

    sender = FrameSender(args.host, args.port)
    period = (1.0 / args.rate) if args.rate > 0 else 0.0
    sent = 0
    try:
        for frame_id, (frame_t, fname) in enumerate(chosen):
            j_gt = int(np.argmin(np.abs(gt_t - frame_t)))
            j_est = int(np.argmin(np.abs(est_t - frame_t)))
            if abs(gt_t[j_gt] - frame_t) > args.max_time_diff or abs(est_t[j_est] - frame_t) > args.max_time_diff:
                continue
            pred_path = find_depth_file(args.fde_dir, fname)
            if pred_path is None:
                continue
            pred_depth = load_depth_frame(pred_path) * args.pred_depth_scale
            if pred_depth.shape != (cam_h, cam_w):
                pred_depth = np.array(Image.fromarray(pred_depth.astype(np.float32)).resize((cam_w, cam_h), Image.BILINEAR))

            T_wc = pose_row_to_T(est_m[j_est])
            # pose_x is the pose error model's x-value (cumulative distance
            # travelled) -- occupancy_3d_map.py --mode udp evaluates the model
            # against it on arrival, same as dataset mode does locally. A real
            # live front-end would track its own cumulative distance the same
            # way instead of reading it out of ground truth.
            pose_x_val = float(cum_dist[j_gt])

            if args.v1:
                sender.send(frame_id, frame_t, pred_depth, T_wc, pose_x=pose_x_val,
                            K=K if args.send_intrinsics else None)
                extra = "без зображення і точок (v1)"
            else:
                image = None
                if not args.no_image:
                    ip = os.path.join(args.dataset_root, "cam0", "data", fname)
                    if os.path.exists(ip):
                        im = Image.open(ip).convert("L")
                        if im.size != (cam_w, cam_h):
                            im = im.resize((cam_w, cam_h), Image.BILINEAR)
                        image = np.asarray(im, dtype=np.uint8)
                lm = None
                if anchors is not None:
                    lm = anchors.for_frame(est_m[j_est, 0])
                sender.send_frame2(frame_id, frame_t, pred_depth, T_wc, pose_x=pose_x_val,
                                    K=K if args.send_intrinsics else None,
                                    image=image, landmarks=lm)
                extra = (f"зображення {'є' if image is not None else 'нема'}, "
                         f"точок {0 if lm is None else len(lm)}")
            sent += 1
            print(f"[replay] кадр {frame_id} ({fname}) -> {args.host}:{args.port}, {extra}")
            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        print("\n[replay] stopped (Ctrl+C)")
    finally:
        sender.close()

    print(f"[replay] done, sent {sent} frames")


if __name__ == "__main__":
    main()
