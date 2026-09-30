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
from udp_streaming import FrameSender


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
    traj_est_sync.align(traj_ref_sync, correct_scale=True)
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

            sender.send(frame_id, frame_t, pred_depth, T_wc, pose_x=pose_x_val,
                        K=K if args.send_intrinsics else None)
            sent += 1
            print(f"[replay] sent frame {frame_id} ({fname}) -> {args.host}:{args.port}")
            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        print("\n[replay] stopped (Ctrl+C)")
    finally:
        sender.close()

    print(f"[replay] done, sent {sent} frames")


if __name__ == "__main__":
    main()
