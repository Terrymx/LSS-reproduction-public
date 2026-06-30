import argparse
import csv
import os
from time import time

import matplotlib.pyplot as plt
import torch

from src.data import compile_data
from src.models import compile_model
from src.tools import SimpleLoss, get_batch_iou


def configs(args):
    grid_conf = {
        "xbound": args.xbound,
        "ybound": args.ybound,
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": args.dbound,
    }
    data_aug_conf = {
        "resize_lim": (args.resize, args.resize),
        "final_dim": (args.final_h, args.final_w),
        "rot_lim": (-5.4, 5.4) if args.aug else (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": args.aug,
        "bot_pct_lim": (0.0, 0.22) if args.aug else (0.0, 0.0),
        "cams": [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ],
        "Ncams": args.ncams,
    }
    return grid_conf, data_aug_conf


def binary_stats(probs, target, threshold):
    pred = probs > threshold
    tgt = target.bool()
    intersect = (pred & tgt).sum().float().item()
    union = (pred | tgt).sum().float().item()
    pred_pos = pred.sum().float().item()
    gt_pos = tgt.sum().float().item()
    return {
        f"val_iou_at_{threshold:g}": intersect / union if union > 0 else 1.0,
        f"val_precision_at_{threshold:g}": intersect / pred_pos if pred_pos > 0 else 0.0,
        f"val_recall_at_{threshold:g}": intersect / gt_pos if gt_pos > 0 else 0.0,
        f"val_pred_positive_at_{threshold:g}": int(pred_pos),
    }


def eval_model(model, valloader, loss_fn, device, max_batches):
    thresholds = [0.5, 0.25, 0.1]
    totals = {t: {"intersect": 0.0, "union": 0.0, "pred_pos": 0.0} for t in thresholds}
    total_loss = 0.0
    total_gt_pos = 0.0
    total_pixels = 0
    sigmoid_sum = 0.0
    sigmoid_min = float("inf")
    sigmoid_max = float("-inf")

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            if batch_idx >= max_batches:
                break
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            binimgs = binimgs.to(device)
            probs = preds.sigmoid()
            total_loss += loss_fn(preds, binimgs).item()
            total_gt_pos += binimgs.bool().sum().float().item()
            total_pixels += probs.numel()
            sigmoid_sum += probs.mean().item() * probs.numel()
            sigmoid_min = min(sigmoid_min, probs.min().item())
            sigmoid_max = max(sigmoid_max, probs.max().item())
            for threshold in thresholds:
                pred = probs > threshold
                tgt = binimgs.bool()
                totals[threshold]["intersect"] += (pred & tgt).sum().float().item()
                totals[threshold]["union"] += (pred | tgt).sum().float().item()
                totals[threshold]["pred_pos"] += pred.sum().float().item()
    if was_training:
        model.train()

    out = {
        "val_loss": total_loss / max_batches,
        "val_sigmoid_min": sigmoid_min,
        "val_sigmoid_mean": sigmoid_sum / total_pixels,
        "val_sigmoid_max": sigmoid_max,
        "val_gt_positive": int(total_gt_pos),
    }
    for threshold in thresholds:
        key = f"{threshold:g}"
        intersect = totals[threshold]["intersect"]
        union = totals[threshold]["union"]
        pred_pos = totals[threshold]["pred_pos"]
        out[f"val_iou_at_{key}"] = intersect / union if union > 0 else 1.0
        out[f"val_precision_at_{key}"] = intersect / pred_pos if pred_pos > 0 else 0.0
        out[f"val_recall_at_{key}"] = intersect / total_gt_pos if total_gt_pos > 0 else 0.0
        out[f"val_pred_positive_at_{key}"] = int(pred_pos)
    return out


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Tiny nuScenes mini training run for LSS reproduction.")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    parser.add_argument("--outdir", default="outputs/debug_train")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float, default=2.13)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--final-h", type=int, default=128)
    parser.add_argument("--final-w", type=int, default=352)
    parser.add_argument("--resize", type=float, default=0.22)
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    parser.add_argument("--aug", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-batches", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    grid_conf, data_aug_conf = configs(args)
    trainloader, valloader = compile_data(
        args.version, args.dataroot, data_aug_conf, grid_conf,
        bsz=args.batch_size, nworkers=args.nworkers, parser_name="segmentationdata",
    )
    model = compile_model(grid_conf, data_aug_conf, outC=1).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-7)
    loss_fn = SimpleLoss(args.pos_weight).to(device)

    rows = []
    eval_rows = []
    step = 0
    while step < args.steps:
        for batch in trainloader:
            t0 = time()
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            opt.zero_grad()
            preds = model(
                imgs.to(device), rots.to(device), trans.to(device), intrins.to(device),
                post_rots.to(device), post_trans.to(device),
            )
            binimgs = binimgs.to(device)
            loss = loss_fn(preds, binimgs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            _, _, iou = get_batch_iou(preds.detach(), binimgs)
            elapsed = time() - t0
            step += 1
            row = {"step": step, "loss": loss.item(), "iou": iou, "seconds": elapsed}
            print(row)
            rows.append(row)

            if args.save_every and step % args.save_every == 0:
                ckpt = os.path.join(args.outdir, f"debug_model_step{step}.pt")
                torch.save(model.state_dict(), ckpt)
                print("saved", ckpt)

            if args.eval_every and step % args.eval_every == 0:
                eval_row = {"step": step}
                eval_row.update(eval_model(model, valloader, loss_fn, device, args.eval_batches))
                print("EVAL", eval_row)
                eval_rows.append(eval_row)
                write_csv(os.path.join(args.outdir, "eval_log.csv"), eval_rows)

            if step >= args.steps:
                break

    write_csv(os.path.join(args.outdir, "loss_log.csv"), rows)
    plt.figure(figsize=(6, 3))
    plt.plot([r["step"] for r in rows], [r["loss"] for r in rows])
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curve.png"), dpi=160)
    plt.close()

    ckpt = os.path.join(args.outdir, "debug_model.pt")
    torch.save(model.state_dict(), ckpt)
    print("saved", ckpt)


if __name__ == "__main__":
    main()
