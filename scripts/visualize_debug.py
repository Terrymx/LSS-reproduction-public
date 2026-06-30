import argparse
import os

import matplotlib.pyplot as plt
import torch

from src.data import compile_data
from src.models import compile_model
from scripts.debug_eval import configs


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize(img):
    return (img.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1).permute(1, 2, 0)


def bev_extent(grid_conf):
    return [
        grid_conf["ybound"][0],
        grid_conf["ybound"][1],
        grid_conf["xbound"][1],
        grid_conf["xbound"][0],
    ]


def plot_bev(ax, image, title, grid_conf, cmap="magma", vmin=None, vmax=None):
    ax.imshow(image, cmap=cmap, extent=bev_extent(grid_conf), origin="upper", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("left/right y (m)")
    ax.set_ylabel("front/back x (m)")
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.7)
    ax.grid(color="white", linewidth=0.3, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description="Visualize a debug LSS checkpoint on nuScenes mini.")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    parser.add_argument("--modelf", required=True)
    parser.add_argument("--outdir", default="outputs/visualizations_debug")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.1, 0.25, 0.5])
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
        bsz=1, nworkers=0, parser_name="segmentationdata",
    )
    model = compile_model(grid_conf, data_aug_conf, outC=1).to(device)
    model.load_state_dict(torch.load(args.modelf, map_location=device))
    model.eval()

    batch = next(iter(valloader))
    for _ in range(args.sample_idx):
        batch = next(iter(valloader))

    imgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
    with torch.no_grad():
        logits = model(
            imgs.to(device), rots.to(device), trans.to(device), intrins.to(device),
            post_rots.to(device), post_trans.to(device),
        )
        probs = logits.sigmoid().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for cam_idx, ax in enumerate(axes.flat):
        ax.imshow(denormalize(imgs[0, cam_idx]))
        ax.set_title(data_aug_conf["cams"][cam_idx])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "multi_camera_inputs.png"), dpi=160)
    plt.close(fig)

    gt = binimgs[0, 0].cpu().numpy()
    prob = probs[0, 0].numpy()
    pred = (prob > args.threshold).astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    plot_bev(axes[0], gt, "GT vehicle mask", grid_conf, cmap="gray", vmin=0, vmax=1)
    plot_bev(axes[1], prob, "Prediction probability", grid_conf, cmap="magma", vmin=0, vmax=1)
    plot_bev(axes[2], pred, f"Prediction > {args.threshold}", grid_conf, cmap="gray", vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "bev_gt_vs_pred.png"), dpi=180)
    plt.close(fig)

    ncols = 1 + len(args.thresholds)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    plot_bev(axes[0], gt, "GT vehicle mask", grid_conf, cmap="gray", vmin=0, vmax=1)
    for ax, threshold in zip(axes[1:], args.thresholds):
        plot_bev(
            ax,
            (prob > threshold).astype(float),
            f"Prediction > {threshold:g}",
            grid_conf,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "bev_threshold_sweep.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
