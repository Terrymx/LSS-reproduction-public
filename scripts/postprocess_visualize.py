import argparse
import os

import matplotlib.pyplot as plt
import torch

from scripts.postprocess_eval import make_configs, postprocess
from src.data import compile_data
from src.models import compile_model


def bev_extent(grid_conf):
    return [
        grid_conf["ybound"][0],
        grid_conf["ybound"][1],
        grid_conf["xbound"][1],
        grid_conf["xbound"][0],
    ]


def mask_stats(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = (pred & target).sum()
    union = (pred | target).sum()
    pred_pos = pred.sum()
    gt_pos = target.sum()
    return {
        "iou": inter / union if union else 1.0,
        "precision": inter / pred_pos if pred_pos else 0.0,
        "recall": inter / gt_pos if gt_pos else 0.0,
    }


def plot_bev(ax, image, title, grid_conf, cmap="gray", vmin=0, vmax=1):
    ax.imshow(
        image,
        cmap=cmap,
        extent=bev_extent(grid_conf),
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("left/right y (m)")
    ax.set_ylabel("front/back x (m)")
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    ax.grid(color="white", linewidth=0.3, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize LSS BEV masks before and after simple post-processing."
    )
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="trainval", choices=["mini", "trainval"])
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--outdir", default="outputs/postprocess_visualizations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=10)
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
    grid_conf, data_aug_conf = make_configs(args)
    _, valloader = compile_data(
        args.version,
        args.dataroot,
        data_aug_conf,
        grid_conf,
        bsz=1,
        nworkers=0,
        parser_name="segmentationdata",
    )

    model = compile_model(grid_conf, data_aug_conf, outC=1).to(device)
    model.load_state_dict(torch.load(args.modelf, map_location=device))
    model.eval()

    loader_iter = iter(valloader)
    batch = next(loader_iter)
    for _ in range(args.sample_idx):
        batch = next(loader_iter)

    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
    with torch.no_grad():
        logits = model(
            imgs.to(device),
            rots.to(device),
            trans.to(device),
            intrins.to(device),
            post_rots.to(device),
            post_trans.to(device),
        )
        probs = logits.sigmoid().cpu().numpy()

    prob = probs[0, 0]
    target = (binimgs[0, 0].cpu().numpy() > 0.5)
    raw = prob > args.threshold
    variants = ["raw", "open1", "area", "open_area", "erode1", "rect"]
    masks = {name: postprocess(raw, name, args.min_area) for name in variants}

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flat
    plot_bev(axes[0], target.astype(float), "GT vehicle mask", grid_conf)
    plot_bev(axes[1], prob, "Prediction probability", grid_conf, cmap="magma")

    for ax, name in zip(axes[2:], variants):
        stats = mask_stats(masks[name], target)
        title = (
            f"{name}\n"
            f"IoU {stats['iou']:.3f} "
            f"P {stats['precision']:.3f} "
            f"R {stats['recall']:.3f}"
        )
        plot_bev(ax, masks[name].astype(float), title, grid_conf)

    fig.tight_layout()
    outpath = os.path.join(args.outdir, "bev_postprocess_compare.png")
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    print("saved", outpath)


if __name__ == "__main__":
    main()
