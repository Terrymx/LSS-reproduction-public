import argparse
import csv
import os

import cv2
import numpy as np
import torch

from scripts.debug_eval import configs
from src.data import compile_data
from src.models import compile_model


def component_rows(gt_mask, prob, batch_idx, sample_idx, thresholds):
    gt_u8 = gt_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gt_u8, connectivity=8)
    rows = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        comp = labels == label
        row = {
            "batch": batch_idx,
            "sample": sample_idx,
            "component": label,
            "area_cells": area,
            "bbox_x": x,
            "bbox_y": y,
            "bbox_w": w,
            "bbox_h": h,
            "max_prob": float(prob[comp].max()) if area else 0.0,
            "mean_prob": float(prob[comp].mean()) if area else 0.0,
        }
        for threshold in thresholds:
            pred = prob > threshold
            hit = int((pred & comp).sum())
            row[f"covered_cells_at_{threshold:g}"] = hit
            row[f"coverage_at_{threshold:g}"] = hit / area if area else 0.0
            row[f"detected_at_{threshold:g}"] = int(hit > 0)
        rows.append(row)
    return rows


def summarize(rows, thresholds, tiny_area):
    summary = []
    groups = {
        "all": rows,
        f"tiny_area_lt_{tiny_area}": [r for r in rows if r["area_cells"] < tiny_area],
        f"large_area_ge_{tiny_area}": [r for r in rows if r["area_cells"] >= tiny_area],
    }
    for name, group in groups.items():
        out = {
            "group": name,
            "components": len(group),
            "area_cells": sum(r["area_cells"] for r in group),
            "mean_area_cells": (
                sum(r["area_cells"] for r in group) / len(group) if group else 0.0
            ),
            "mean_max_prob": (
                sum(r["max_prob"] for r in group) / len(group) if group else 0.0
            ),
        }
        for threshold in thresholds:
            detected = sum(r[f"detected_at_{threshold:g}"] for r in group)
            covered = sum(r[f"covered_cells_at_{threshold:g}"] for r in group)
            area = sum(r["area_cells"] for r in group)
            out[f"detected_components_at_{threshold:g}"] = detected
            out[f"component_recall_at_{threshold:g}"] = detected / len(group) if group else 0.0
            out[f"cell_coverage_at_{threshold:g}"] = covered / area if area else 0.0
        summary.append(out)
    return summary


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
    parser = argparse.ArgumentParser(description="Evaluate GT vehicle blobs against BEV predictions.")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--outdir", default="outputs/component_eval")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=0)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.1, 0.25, 0.5])
    parser.add_argument("--tiny-area", type=int, default=20)
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
            probs = logits.sigmoid().cpu().numpy()
            gts = binimgs.cpu().numpy()
            for sample_idx in range(gts.shape[0]):
                rows.extend(
                    component_rows(
                        gts[sample_idx, 0] > 0.5,
                        probs[sample_idx, 0],
                        batch_idx,
                        sample_idx,
                        args.thresholds,
                    )
                )

    summary = summarize(rows, args.thresholds, args.tiny_area)
    write_csv(os.path.join(args.outdir, "components.csv"), rows)
    write_csv(os.path.join(args.outdir, "summary.csv"), summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
