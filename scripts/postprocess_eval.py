import argparse

import cv2
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


def area_filter(mask, min_area=10, max_area=10000):
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            out[labels == i] = 1
    return out.astype(bool)


def shape_filter(mask, min_area=10, max_area=10000, max_long_side=18.0, max_aspect=6.0, grid_size_m=0.5):
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue

        ys, xs = np.where(labels == i)
        if len(xs) < 3:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        side_a, side_b = rect[1]
        long_side = max(side_a, side_b) * grid_size_m
        short_side = max(min(side_a, side_b) * grid_size_m, 1e-6)
        aspect = long_side / short_side
        if long_side <= max_long_side and aspect <= max_aspect:
            out[labels == i] = 1
    return out.astype(bool)


def rect_prior(mask, min_area=10, max_area=10000):
    mask_u8 = mask.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    out = np.zeros_like(mask_u8)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < 3:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(out, box, 1)
    return out.astype(bool)


def postprocess(mask, variant, min_area, max_long_side=18.0, max_aspect=6.0, grid_size_m=0.5):
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    if variant == "raw":
        return mask_u8.astype(bool)
    if variant == "erode1":
        return cv2.erode(mask_u8, kernel, iterations=1).astype(bool)
    if variant == "open1":
        return cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    if variant == "area":
        return area_filter(mask_u8, min_area=min_area)
    if variant == "open_area":
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        return area_filter(opened, min_area=min_area)
    if variant == "shape":
        return shape_filter(
            mask_u8,
            min_area=min_area,
            max_long_side=max_long_side,
            max_aspect=max_aspect,
            grid_size_m=grid_size_m,
        )
    if variant == "open_shape":
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        return shape_filter(
            opened,
            min_area=min_area,
            max_long_side=max_long_side,
            max_aspect=max_aspect,
            grid_size_m=grid_size_m,
        )
    if variant == "rect":
        return rect_prior(mask_u8, min_area=min_area)
    raise ValueError(f"Unknown postprocess variant: {variant}")


def update_stats(stats, pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    pred_pos = pred.sum()
    gt_pos = target.sum()
    stats["inter"] += int(inter)
    stats["union"] += int(union)
    stats["pred_pos"] += int(pred_pos)
    stats["gt_pos"] += int(gt_pos)


def summarize_stats(name, stats):
    precision = stats["inter"] / stats["pred_pos"] if stats["pred_pos"] else 0.0
    recall = stats["inter"] / stats["gt_pos"] if stats["gt_pos"] else 0.0
    iou = stats["inter"] / stats["union"] if stats["union"] else 1.0
    return {
        "variant": name,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "pred_positive": stats["pred_pos"],
        "gt_positive": stats["gt_pos"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate simple BEV mask post-processing for LSS predictions."
    )
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="trainval", choices=["mini", "trainval"])
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=10)
    parser.add_argument("--max-long-side", type=float, default=18.0)
    parser.add_argument("--max-aspect", type=float, default=6.0)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--final-h", type=int, default=256)
    parser.add_argument("--final-w", type=int, default=704)
    parser.add_argument("--resize", type=float, default=0.44)
    parser.add_argument("--dbound", nargs=3, type=float, default=[4.0, 45.0, 1.0])
    parser.add_argument("--xbound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-50.0, 50.0, 0.5])
    args = parser.parse_args()

    device = torch.device(args.device)
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

    grid_size_m = float(args.xbound[2])
    variants = ["raw", "erode1", "open1", "area", "open_area", "shape", "open_shape", "rect"]
    stats = {v: {"inter": 0, "union": 0, "pred_pos": 0, "gt_pos": 0} for v in variants}

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

            for batch_item in range(probs.shape[0]):
                raw_mask = probs[batch_item, 0] > args.threshold
                target = targets[batch_item, 0] > 0.5
                for variant in variants:
                    pred = postprocess(
                        raw_mask,
                        variant,
                        args.min_area,
                        max_long_side=args.max_long_side,
                        max_aspect=args.max_aspect,
                        grid_size_m=grid_size_m,
                    )
                    update_stats(stats[variant], pred, target)

    print("POSTPROCESS SUMMARY")
    print(
        {
            "checkpoint": args.modelf,
            "threshold": args.threshold,
            "max_batches": args.max_batches,
            "min_area": args.min_area,
            "max_long_side": args.max_long_side,
            "max_aspect": args.max_aspect,
        }
    )
    for variant in variants:
        print(summarize_stats(variant, stats[variant]))


if __name__ == "__main__":
    main()
