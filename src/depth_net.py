r"""
Small U-Net that refines a feed-forward depth map using sparse reference
depths, an image and its gradients (stage 2).

Why a network at all: error_budget.py attributes ~88% of the occupancy gap to
depth (IoU 0.11 with FDE depth and true poses, 0.91 with true depth), and the
non-parametric table in depth_refine.py -- a correction indexed only by
(depth bin, image row) -- can only capture a bias that is constant across a
whole row. A network can use the image itself.

Inputs (5 channels, 320x240), every depth already scale-normalised by
depth_net_data.py as x = log(depth / median(sparse depths of that frame)):
    0  log FDE depth
    1  log sparse reference depth (0 where absent)
    2  sparse-point mask
    3  greyscale image
    4  Sobel gradient magnitude
The image ROW INDEX is deliberately NOT an input: it would tie the model to one
camera pitch. `--with_row_channel` adds it as a comparison variant.

Outputs (3 channels):
    0  delta      -> refined depth = FDE depth * exp(delta)
    1  log_var    -> the model's own uncertainty on log-depth
    2  surface    -> logit of "a surface exists here at all" (rejects sky)

Loss: Laplace negative log-likelihood on log-depth over valid pixels, which
lets the model widen its uncertainty instead of being forced to guess, plus
binary cross-entropy for the surface channel.

    L = mean_valid[ |r| / b + log(b) ] + w_bce * BCE(surface, valid)
    where r = (log_fde + delta) - log_target,  b = exp(log_var / 2)

Usage:
    python3 depth_net.py train --fde zoedepth --holdout euroc --output ../nets/zoe_holdout_euroc.pt
    python3 depth_net.py eval  --model ../nets/zoe_holdout_euroc.pt --holdout euroc
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASETS = ["euroc"] + [f"run{i}" for i in range(1, 7)]
IN_CH = 5


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class DepthRefineNet(nn.Module):
    """Four-level U-Net, ~2.5M parameters at base=32."""

    def __init__(self, in_ch=IN_CH, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = Block(in_ch, c[0]), Block(c[0], c[1]), Block(c[1], c[2])
        self.bott = Block(c[2], c[3])
        self.d3 = Block(c[3] + c[2], c[2])
        self.d2 = Block(c[2] + c[1], c[1])
        self.d1 = Block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # start as the identity correction

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        b = self.bott(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, size=e3.shape[-2:], mode="nearest"), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[-2:], mode="nearest"), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        out = self.head(d1)
        delta, log_var, surface = out[:, 0:1], out[:, 1:2], out[:, 2:3]
        return delta, log_var.clamp(-8.0, 4.0), surface


CHANNELS = ["grey", "grad", "fde", "sparse", "smask", "target", "valid"]


class LazyCache:
    """Frames stay on disk. Each dataset is a directory of uncompressed .npy
    files opened with mmap_mode='r', so a batch reads only its own frames.
    Loading all seven datasets eagerly would be ~7 GB uncompressed against 8 GB
    of RAM here, and it is not needed: a batch is 16 frames.

    Indexing is global: `cache["fde"][idx]` works across dataset boundaries."""

    def __init__(self, cache_dir, fde, datasets):
        self.parts, self.owner, self.offsets = [], [], [0]
        for name in datasets:
            d = os.path.join(cache_dir, f"{name}_{fde}")
            npz = os.path.join(cache_dir, f"{name}_{fde}.npz")
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "valid.npy")):
                part = {k: np.load(os.path.join(d, f"{k}.npy"), mmap_mode="r") for k in CHANNELS}
            elif os.path.exists(npz):
                print(f"  {name}: only .npz present -- loading eagerly "
                      f"(run depth_net_data.py --to_memmap to avoid this)")
                z = np.load(npz)
                part = {k: z[k] for k in CHANNELS}
            else:
                print(f"  {name}: no cache -- skipped")
                continue
            n = len(part["valid"])
            self.parts.append(part)
            self.owner += [name] * n
            self.offsets.append(self.offsets[-1] + n)
            print(f"  {name}: {n} frames")
        if not self.parts:
            raise SystemExit("no caches found -- run depth_net_data.py first")
        self.owner = np.array(self.owner)

    def __len__(self):
        return self.offsets[-1]

    def gather(self, channel, idx):
        """idx: global frame indices -> (len(idx), H, W) array, read from disk."""
        idx = np.asarray(idx)
        out = None
        for p, lo, hi in zip(self.parts, self.offsets[:-1], self.offsets[1:]):
            sel = np.flatnonzero((idx >= lo) & (idx < hi))
            if sel.size == 0:
                continue
            block = np.asarray(p[channel][idx[sel] - lo])
            if out is None:
                out = np.empty((len(idx),) + block.shape[1:], dtype=block.dtype)
            out[sel] = block
        return out


def load_cache(cache_dir, fde, datasets):
    c = LazyCache(cache_dir, fde, datasets)
    return c, c.owner


def make_batch(data, idx, device, rng, augment, with_row):
    grey = data.gather("grey", idx).astype(np.float32)
    grad = data.gather("grad", idx).astype(np.float32)
    fde = data.gather("fde", idx).astype(np.float32)
    sparse = data.gather("sparse", idx).astype(np.float32)
    smask = data.gather("smask", idx).astype(np.float32)
    target = data.gather("target", idx).astype(np.float32)
    valid = data.gather("valid", idx).astype(np.float32)

    if augment:
        b = len(idx)
        # photometric: brightness, contrast, gamma, blur, haze
        grey = np.clip(grey * rng.uniform(0.7, 1.4, (b, 1, 1)) + rng.uniform(-0.1, 0.1, (b, 1, 1)), 0, 1)
        grey = np.clip(grey ** rng.uniform(0.7, 1.4, (b, 1, 1)), 0, 1)
        haze = rng.uniform(0.0, 0.3, (b, 1, 1))
        grey = grey * (1 - haze) + haze
        grad = grad * rng.uniform(0.7, 1.3, (b, 1, 1))
        # SCALE: multiply every depth in the frame by one factor. In log space
        # that is an additive shift, and it must leave the task unchanged --
        # this is the direct test that the model never learned absolute metres.
        shift = np.log(rng.uniform(0.3, 3.0, (b, 1, 1))).astype(np.float32)
        fde = np.where(fde != 0, fde + shift, fde)
        sparse = np.where(smask > 0, sparse + shift, sparse)
        target = np.where(valid > 0, target + shift, target)
        # drop some sparse points
        keep = rng.random(smask.shape) > rng.uniform(0.0, 0.5, (b, 1, 1))
        smask = smask * keep
        sparse = sparse * keep
        # extra noise on the sparse depths
        sparse = np.where(smask > 0, sparse + rng.normal(0, 0.05, sparse.shape), sparse)
        if rng.random() < 0.5:  # horizontal flip
            grey, grad, fde, sparse, smask, target, valid = [
                a[:, :, ::-1].copy() for a in (grey, grad, fde, sparse, smask, target, valid)]

    chans = [fde, sparse, smask, grey, grad]
    if with_row:
        h = grey.shape[1]
        row = np.broadcast_to((np.arange(h, dtype=np.float32) / h)[None, :, None], grey.shape)
        chans.append(np.ascontiguousarray(row))
    # The augmentation multiplies by float64 scalars from the generator, which
    # promotes the arrays to double; cast back explicitly or conv2d rejects them
    # against half-precision weights under AMP.
    x = torch.from_numpy(np.stack(chans, 1).astype(np.float32)).to(device)
    return (x,
            torch.from_numpy(fde.astype(np.float32)[:, None]).to(device),
            torch.from_numpy(target.astype(np.float32)[:, None]).to(device),
            torch.from_numpy(valid.astype(np.float32)[:, None]).to(device))


def laplace_nll(pred_log, target_log, log_var, valid):
    b = torch.exp(0.5 * log_var)
    r = (pred_log - target_log).abs()
    per_px = r / b + 0.5 * log_var
    denom = valid.sum().clamp(min=1.0)
    return (per_px * valid).sum() / denom


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_sets = [d for d in DATASETS if d != args.holdout]
    print(f"training on: {', '.join(train_sets)}   holdout: {args.holdout}")
    data, owner = load_cache(args.cache_dir, args.fde, train_sets)
    n = len(owner)

    # Validation split comes from the TRAINING sets, never the holdout.
    perm = rng.permutation(n)
    n_val = max(1, int(0.1 * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"frames: {len(tr_idx)} train, {len(val_idx)} val (from training sets)")

    in_ch = IN_CH + (1 if args.with_row_channel else 0)
    net = DepthRefineNet(in_ch=in_ch, base=args.base).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"parameters: {n_par/1e6:.2f}M")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    steps_per_epoch = max(1, len(tr_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps_per_epoch)

    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(tr_idx)
        tot = 0.0
        for s in tqdm(range(steps_per_epoch), desc=f"epoch {epoch+1}/{args.epochs}", leave=False):
            idx = np.sort(order[s * args.batch:(s + 1) * args.batch])
            x, fde, target, valid = make_batch(data, idx, device, rng, True, args.with_row_channel)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                delta, log_var, surface = net(x)
                nll = laplace_nll(fde + delta, target, log_var, valid)
                bce = F.binary_cross_entropy_with_logits(surface, valid)
                loss = nll + args.w_bce * bce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach())

        net.eval()
        vl = 0.0
        with torch.no_grad():
            for s in range(max(1, len(val_idx) // args.batch)):
                idx = np.sort(val_idx[s * args.batch:(s + 1) * args.batch])
                x, fde, target, valid = make_batch(data, idx, device, rng, False, args.with_row_channel)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    delta, log_var, surface = net(x)
                    vl += float(laplace_nll(fde + delta, target, log_var, valid)
                                + args.w_bce * F.binary_cross_entropy_with_logits(surface, valid))
        vl /= max(1, len(val_idx) // args.batch)
        print(f"  epoch {epoch+1:2d}: train {tot/steps_per_epoch:.4f}  val {vl:.4f}"
              f"  ({time.time()-t0:.0f}s)")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop (no improvement for {args.patience} epochs)")
                break

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"state_dict": best_state or net.state_dict(),
                "in_ch": in_ch, "base": args.base, "with_row": args.with_row_channel,
                "fde": args.fde, "holdout": args.holdout, "val_loss": best,
                "train_sets": train_sets, "params": n_par}, args.output)
    print(f"Saved: {args.output}  (best val {best:.4f}, {time.time()-t0:.0f}s)")


def load_net(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    net = DepthRefineNet(in_ch=ck["in_ch"], base=ck["base"]).to(device)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


@torch.no_grad()
def predict(net, ck, fde_log, sparse_log, smask, grey, grad, device="cuda"):
    """All inputs (H, W) float32 in the cache's log convention.
    -> (delta, sigma_log, p_surface), each (H, W)."""
    chans = [fde_log, sparse_log, smask.astype(np.float32), grey, grad]
    if ck["with_row"]:
        h = grey.shape[0]
        chans.append(np.broadcast_to((np.arange(h, dtype=np.float32) / h)[:, None], grey.shape))
    x = torch.from_numpy(np.stack(chans)[None].astype(np.float32)).to(device)
    delta, log_var, surface = net(x)
    return (delta[0, 0].cpu().numpy(),
            np.exp(0.5 * log_var[0, 0].cpu().numpy()),
            torch.sigmoid(surface)[0, 0].cpu().numpy())


def evaluate(args):
    """Depth metrics on every valid pixel of a cache, for raw FDE and for the
    network. Run on the holdout to measure generalisation, and on the training
    sets to see the gap."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ck = load_net(args.model, device)
    rows = []
    for split, sets in (("holdout", [ck["holdout"]]), ("train", ck["train_sets"])):
        data, owner = load_cache(args.cache_dir, ck["fde"], sets)
        n = len(data)
        pick = np.arange(0, n, max(1, n // args.eval_frames))
        agg = {k: [] for k in ["absrel_fde", "absrel_net", "rmse_fde", "rmse_net",
                                "d125_fde", "d125_net", "surf_acc"]}
        for i in tqdm(pick, desc=f"eval {split}", leave=False):
            one = np.array([i])
            valid = data.gather("valid", one)[0].astype(bool)
            if valid.sum() < 100:
                continue
            fde_l = data.gather("fde", one)[0].astype(np.float32)
            tgt_l = data.gather("target", one)[0].astype(np.float32)
            delta, _sig, p_surf = predict(net, ck, fde_l, data.gather("sparse", one)[0].astype(np.float32),
                                           data.gather("smask", one)[0], data.gather("grey", one)[0].astype(np.float32),
                                           data.gather("grad", one)[0].astype(np.float32), device)
            # ratios, so the frame's divisor cancels
            t = np.exp(tgt_l[valid])
            for tag, pred_l in (("fde", fde_l), ("net", fde_l + delta)):
                p = np.exp(pred_l[valid])
                agg[f"absrel_{tag}"].append(float(np.mean(np.abs(p - t) / t)))
                agg[f"rmse_{tag}"].append(float(np.sqrt(np.mean((p - t) ** 2))))
                agg[f"d125_{tag}"].append(float(np.mean(np.maximum(p / t, t / p) < 1.25)))
            agg["surf_acc"].append(float(np.mean((p_surf > 0.5) == valid)))
        rows.append((split, {k: float(np.mean(v)) for k, v in agg.items() if v}))

    print(f"\nmodel: {os.path.basename(args.model)}  fde={ck['fde']}  holdout={ck['holdout']}  "
          f"params={ck['params']/1e6:.2f}M")
    print(f"{'split':9s} {'AbsRel FDE':>11s} {'AbsRel net':>11s} {'RMSE FDE':>9s} {'RMSE net':>9s} "
          f"{'d<1.25 FDE':>11s} {'d<1.25 net':>11s} {'surface acc':>12s}")
    for split, m in rows:
        print(f"{split:9s} {m['absrel_fde']:11.4f} {m['absrel_net']:11.4f} {m['rmse_fde']:9.4f} "
              f"{m['rmse_net']:9.4f} {m['d125_fde']:11.4f} {m['d125_net']:11.4f} "
              f"{m['surf_acc']:12.4f}")
    if len(rows) == 2:
        h, t = rows[0][1], rows[1][1]
        print(f"\ngeneralisation gap (holdout - train): AbsRel {h['absrel_net']-t['absrel_net']:+.4f}, "
              f"d<1.25 {h['d125_net']-t['d125_net']:+.4f}")
    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump({s: m for s, m in rows}, f, indent=2)
        print(f"Saved: {args.metrics_out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--fde", required=True)
    t.add_argument("--holdout", required=True, choices=DATASETS)
    t.add_argument("--output", required=True)
    t.add_argument("--cache_dir", default="../cache")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--base", type=int, default=32)
    t.add_argument("--w_bce", type=float, default=0.5)
    t.add_argument("--patience", type=int, default=6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--with_row_channel", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--cache_dir", default="../cache")
    e.add_argument("--eval_frames", type=int, default=200)
    e.add_argument("--metrics_out")

    args = ap.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
