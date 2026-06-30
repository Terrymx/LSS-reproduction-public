import argparse
import os

import matplotlib.pyplot as plt
import torch

from scripts.debug_eval import configs
from scripts.visualize_debug import denormalize, plot_bev
from src.data import compile_data
from src.models import compile_model


def predict(model_path, grid_conf, data_aug_conf, batch, device):
    model = compile_model(grid_conf, data_aug_conf, outC=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    imgs, rots, trans, intrins, post_rots, post_trans, _ = batch
    with torch.no_grad():
        logits = model(
            imgs.to(device), rots.to(device), trans.to(device), intrins.to(device),
            post_rots.to(device), post_trans.to(device),
        )
    return logits.sigmoid().cpu()[0, 0].numpy()


def main():
    parser = argparse.ArgumentParser(description="Compare two debug LSS checkpoints on one val sample.")
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="mini", choices=["mini", "trainval"])
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--outdir", default="outputs/checkpoint_compare")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.25)
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
    batch = next(iter(valloader))
    imgs, _, _, _, _, _, binimgs = batch

    pred_a = predict(args.model_a, grid_conf, data_aug_conf, batch, device)
    pred_b = predict(args.model_b, grid_conf, data_aug_conf, batch, device)
    gt = binimgs[0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    plot_bev(axes[0], gt, "GT vehicle mask", grid_conf, cmap="gray", vmin=0, vmax=1)
    plot_bev(axes[1], pred_a, args.label_a, grid_conf, cmap="magma", vmin=0, vmax=1)
    plot_bev(axes[2], pred_b, args.label_b, grid_conf, cmap="magma", vmin=0, vmax=1)
    plot_bev(axes[3], pred_b - pred_a, f"{args.label_b} - {args.label_a}", grid_conf, cmap="coolwarm", vmin=-1, vmax=1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_000_bev_compare.png"), dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for cam_idx, ax in enumerate(axes.flat):
        ax.imshow(denormalize(imgs[0, cam_idx]))
        ax.set_title(data_aug_conf["cams"][cam_idx])
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sample_000_inputs.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
