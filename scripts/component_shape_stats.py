import argparse
import csv
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import compile_data
from src.models import compile_model


def make_configs(args):
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
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ],
        "Ncams": args.ncams,
    }
    return grid_conf, data_aug_conf


def component_rows(mask, source, sample_index, grid_size_m, min_area):
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    rows = []
    for label_id in range(1, num):
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        if area_px < min_area:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / max(1, min(w, h))

        ys, xs = np.where(labels == label_id)
        oriented_w = 0.0
        oriented_h = 0.0
        oriented_aspect = 0.0
        if len(xs) >= 3:
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            rect = cv2.minAreaRect(pts)
            side_a, side_b = rect[1]
            oriented_w = float(max(side_a, side_b))
            oriented_h = float(min(side_a, side_b))
            oriented_aspect = oriented_w / max(oriented_h, 1e-6)

        rows.append(
            {
                "source": source,
                "sample_index": sample_index,
                "component_id": label_id,
                "area_px": area_px,
                "area_m2": area_px * grid_size_m * grid_size_m,
                "bbox_w_px": w,
                "bbox_h_px": h,
                "bbox_w_m": w * grid_size_m,
                "bbox_h_m": h * grid_size_m,
                "bbox_aspect": aspect,
                "oriented_w_px": oriented_w,
                "oriented_h_px": oriented_h,
                "oriented_w_m": oriented_w * grid_size_m,
                "oriented_h_m": oriented_h * grid_size_m,
                "oriented_aspect": oriented_aspect,
            }
        )
    return rows


def summarize(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def source_summary(rows, source):
    subset = [row for row in rows if row["source"] == source]
    return {
        "components": len(subset),
        "area_px": summarize([row["area_px"] for row in subset]),
        "area_m2": summarize([row["area_m2"] for row in subset]),
        "bbox_w_m": summarize([row["bbox_w_m"] for row in subset]),
        "bbox_h_m": summarize([row["bbox_h_m"] for row in subset]),
        "bbox_aspect": summarize([row["bbox_aspect"] for row in subset]),
        "oriented_w_m": summarize([row["oriented_w_m"] for row in subset]),
        "oriented_h_m": summarize([row["oriented_h_m"] for row in subset]),
        "oriented_aspect": summarize([row["oriented_aspect"] for row in subset]),
    }


def save_hist(rows, outpath):
    gt = [row for row in rows if row["source"] == "gt"]
    pred = [row for row in rows if row["source"] == "pred"]
    fields = [
        ("area_m2", "Area (m^2)", (0, 80)),
        ("bbox_aspect", "Axis-aligned aspect ratio", (1, 15)),
        ("oriented_w_m", "Oriented long side (m)", (0, 30)),
        ("oriented_aspect", "Oriented aspect ratio", (1, 15)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (field, title, value_range) in zip(axes.flat, fields):
        gt_values = [row[field] for row in gt if row[field] > 0]
        pred_values = [row[field] for row in pred if row[field] > 0]
        ax.hist(gt_values, bins=40, range=value_range, alpha=0.6, label="GT")
        ax.hist(pred_values, bins=40, range=value_range, alpha=0.6, label="Pred")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare connected-component shape statistics for LSS GT and predictions."
    )
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="trainval", choices=["mini", "trainval"])
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--outdir", default="outputs/component_shape_stats")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--final-h", type=int, default=256)
    parser.add_argument("--final-w", type=int, default=704)
    parser.add_argument("--resize", type=float, default=0.44)
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    grid_size_m = float(args.xbound[2])
    grid_conf, data_aug_conf = make_configs(args)
    _, valloader = compile_data(
        args.version,
        args.dataroot,
        data_aug_conf,
        grid_conf,
        bsz=args.batch_size,
        nworkers=args.nworkers,
        parser_name="segmentationdata",
    )

    model = compile_model(grid_conf, data_aug_conf, outC=1).to(device)
    model.load_state_dict(torch.load(args.modelf, map_location=device))
    model.eval()

    rows = []
    sample_base = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(valloader):
            if batch_idx >= args.max_batches:
                break
            imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            logits = model(
                imgs.to(device),
                rots.to(device),
                trans.to(device),
                intrins.to(device),
                post_rots.to(device),
                post_trans.to(device),
            )
            probs = logits.sigmoid().detach().cpu().numpy()
            targets = binimgs.detach().cpu().numpy()
            for item_idx in range(probs.shape[0]):
                sample_index = sample_base + item_idx
                pred = probs[item_idx, 0] > args.threshold
                gt = targets[item_idx, 0] > 0.5
                rows.extend(component_rows(gt, "gt", sample_index, grid_size_m, args.min_area))
                rows.extend(component_rows(pred, "pred", sample_index, grid_size_m, args.min_area))
            sample_base += probs.shape[0]

    csv_path = os.path.join(args.outdir, "component_shape_stats.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "checkpoint": args.modelf,
        "threshold": args.threshold,
        "max_batches": args.max_batches,
        "min_area": args.min_area,
        "grid_size_m": grid_size_m,
        "gt": source_summary(rows, "gt"),
        "pred": source_summary(rows, "pred"),
    }
    summary_path = os.path.join(args.outdir, "component_shape_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    hist_path = os.path.join(args.outdir, "component_shape_hist.png")
    save_hist(rows, hist_path)

    print("COMPONENT SHAPE SUMMARY")
    print(json.dumps(summary, indent=2))
    print("saved", csv_path)
    print("saved", summary_path)
    print("saved", hist_path)


if __name__ == "__main__":
    main()
