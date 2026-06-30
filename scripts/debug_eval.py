import argparse
import csv
import os

import torch

from src.data import compile_data
from src.models import compile_model
from src.tools import SimpleLoss


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
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": False,
        "bot_pct_lim": (0.0, 0.0),
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
        f"iou_at_{threshold:g}": intersect / union if union > 0 else 1.0,
        f"precision_at_{threshold:g}": intersect / pred_pos if pred_pos > 0 else 0.0,
        f"recall_at_{threshold:g}": intersect / gt_pos if gt_pos > 0 else 0.0,
        f"pred_positive_at_{threshold:g}": int(pred_pos),
        f"intersect_at_{threshold:g}": int(intersect),
        f"union_at_{threshold:g}": int(union),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a debug LSS checkpoint.")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--outdir", default="outputs/debug_eval")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--pos-weight", type=float, default=2.13)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--final-h", type=int, default=96)
    parser.add_argument("--final-w", type=int, default=256)
    parser.add_argument("--resize", type=float, default=0.22)
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 44.0, 2.0])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    grid_conf, data_aug_conf = configs(args)
    _, valloader = compile_data(
        args.version, args.dataroot, data_aug_conf, grid_conf,
        bsz=args.batch_size, nworkers=args.nworkers, parser_name="segmentationdata",
    )
    model = compile_model(grid_conf, data_aug_conf, outC=1).to(device)
    model.load_state_dict(torch.load(args.modelf, map_location=device))
    model.eval()
    loss_fn = SimpleLoss(args.pos_weight).to(device)

    thresholds = [0.5, 0.25, 0.1]
    totals = {t: {"intersect": 0.0, "union": 0.0, "pred_positive": 0.0} for t in thresholds}
    rows = []
    total_loss = 0.0
    total_gt_positive = 0
    total_pixels = 0
    sigmoid_min = float("inf")
    sigmoid_max = float("-inf")
    sigmoid_sum = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            if batch_idx >= args.max_batches:
                break
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(
                imgs.to(device), rots.to(device), trans.to(device), intrins.to(device),
                post_rots.to(device), post_trans.to(device),
            )
            binimgs = binimgs.to(device)
            probs = preds.sigmoid()
            loss = loss_fn(preds, binimgs)
            row = {
                "batch": batch_idx,
                "loss": loss.item(),
                "sigmoid_min": probs.min().item(),
                "sigmoid_mean": probs.mean().item(),
                "sigmoid_max": probs.max().item(),
                "gt_positive": int(binimgs.bool().sum().item()),
                "pixels": probs.numel(),
            }
            for threshold in thresholds:
                row.update(binary_stats(probs, binimgs, threshold))
                key = f"{threshold:g}"
                totals[threshold]["intersect"] += row[f"intersect_at_{key}"]
                totals[threshold]["union"] += row[f"union_at_{key}"]
                totals[threshold]["pred_positive"] += row[f"pred_positive_at_{key}"]
            print(row)
            rows.append(row)
            total_loss += loss.item()
            total_gt_positive += row["gt_positive"]
            total_pixels += row["pixels"]
            sigmoid_min = min(sigmoid_min, row["sigmoid_min"])
            sigmoid_max = max(sigmoid_max, row["sigmoid_max"])
            sigmoid_sum += row["sigmoid_mean"] * row["pixels"]

    summary = {
        "batches": len(rows),
        "avg_loss_per_batch": total_loss / len(rows),
        "sigmoid_min": sigmoid_min,
        "sigmoid_mean": sigmoid_sum / total_pixels,
        "sigmoid_max": sigmoid_max,
        "gt_positive": total_gt_positive,
        "pixels": total_pixels,
    }
    for threshold in thresholds:
        key = f"{threshold:g}"
        intersect = totals[threshold]["intersect"]
        union = totals[threshold]["union"]
        pred_positive = totals[threshold]["pred_positive"]
        summary[f"iou_at_{key}"] = intersect / union if union > 0 else 1.0
        summary[f"precision_at_{key}"] = intersect / pred_positive if pred_positive > 0 else 0.0
        summary[f"recall_at_{key}"] = intersect / total_gt_positive if total_gt_positive > 0 else 0.0
        summary[f"pred_positive_at_{key}"] = int(pred_positive)
    print("SUMMARY", summary)

    with open(os.path.join(args.outdir, "eval_batches.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.outdir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
