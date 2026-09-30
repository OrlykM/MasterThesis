r"""
Listens on the "separate address" occupancy_3d_map.py --mode udp forwards
each frame's voxel map to (its --send_host/--send_port), and writes every
received grid to disk as a .npz in the same {log_odds, origin, resolution}
format view_3d_occupancy_map.py already reads -- so you can point that
viewer at whatever lands here, or replace this script's body with your own
consumer (e.g. the safety layer) that reads straight from GridReceiver
instead of hitting the filesystem.

Usage:
    python3 udp_map_receiver.py --host 0.0.0.0 --port 9001 --output_dir live_maps
    python3 view_3d_occupancy_map.py live_maps/weighted_latest.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from udp_streaming import GridReceiver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--keep_every", type=int, default=None,
                     help="also archive every Nth received grid as "
                          "<kind>_frame<id>.npz, not just overwrite <kind>_latest.npz")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    receiver = GridReceiver(args.host, args.port)
    print(f"[map_receiver] listening on {args.host}:{args.port}, writing to {args.output_dir}")

    n = 0
    try:
        for grid in receiver.grids():
            n += 1
            latest_path = os.path.join(args.output_dir, f"{grid['kind']}_latest.npz")
            np.savez_compressed(latest_path, log_odds=grid["log_odds"], origin=grid["origin"],
                                 resolution=grid["resolution"])
            if args.keep_every and n % args.keep_every == 0:
                archive_path = os.path.join(args.output_dir, f"{grid['kind']}_frame{grid['frame_id']}.npz")
                np.savez_compressed(archive_path, log_odds=grid["log_odds"], origin=grid["origin"],
                                     resolution=grid["resolution"])
            print(f"[map_receiver] frame {grid['frame_id']} ({grid['kind']}) -> {latest_path}")
    except KeyboardInterrupt:
        print("\n[map_receiver] stopped (Ctrl+C)")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
