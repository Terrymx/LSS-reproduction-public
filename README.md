# Lift-Splat-Shoot Reproduction Notes

This repository is a reproduction and study fork of NVIDIA's **Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D**. The goal of this fork is not to replace the official implementation, but to record a runnable BEV perception pipeline, debugging scripts, ablation experiments, visualizations, and a reproduction report.

Main additions in this fork:

- debug training / evaluation / visualization scripts under `scripts/`;
- full nuScenes trainval experiments with resolution, depth-bin, augmentation, and learning-rate ablations;
- post-processing and component-shape analysis scripts for BEV mask error analysis;
- a Chinese reproduction report: [`reproduction_report.md`](reproduction_report.md);
- report figures under [`report_assets/`](report_assets/).

Large local artifacts are intentionally not tracked:

- nuScenes data: `data/`
- checkpoints: `*.pt`, `*.pth`
- generated training outputs: `outputs/`
- local caches: `.cache/`, `.matplotlib/`

## Quick Reproduction Commands

Example environment setup on Linux:

```bash
conda env create -f environment.yml
```

```bash
conda activate lss-repro
export PYTHONPATH="$PWD"
export TORCH_HOME="$PWD/.cache/torch"
export MPLCONFIGDIR="$PWD/.matplotlib"
export PYTHONIOENCODING=utf-8
```

Expected nuScenes layout:

```text
data/nuscenes/trainval/
  samples/
  sweeps/
  maps/
  v1.0-trainval/
```

Run a debug evaluation:

```bash
python scripts/debug_eval.py \
  --dataroot=data/nuscenes \
  --version=trainval \
  --modelf=checkpoints/model.pt \
  --device=cuda:0 \
  --final-h=256 \
  --final-w=704 \
  --resize=0.44 \
  --dbound 4 45 1
```

Run visualization:

```bash
python scripts/visualize_debug.py \
  --dataroot=data/nuscenes \
  --version=trainval \
  --modelf=checkpoints/model.pt \
  --outdir=outputs/visualizations \
  --device=cuda:0 \
  --final-h=256 \
  --final-w=704 \
  --resize=0.44 \
  --dbound 4 45 1
```

For detailed experiment settings, results, and analysis, see [`reproduction_report.md`](reproduction_report.md).

---

# Original Project: Lift, Splat, Shoot

PyTorch code for Lift-Splat-Shoot (ECCV 2020).

**Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D**  
Jonah Philion, [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)\
ECCV, 2020 (Poster)\
**[[Paper](https://arxiv.org/abs/2008.05711)] [[Project Page](https://nv-tlabs.github.io/lift-splat-shoot/)] [[10-min video](https://youtu.be/oL5ISk6BnDE)] [[1-min video](https://youtu.be/ypQQUG4nFJY)]**

**Abstract:**
The goal of perception for autonomous vehicles is to extract semantic representations from multiple sensors and fuse these representations into a single "bird's-eye-view" coordinate frame for consumption by motion planning. We propose a new end-to-end architecture that directly extracts a bird's-eye-view representation of a scene given image data from an arbitrary number of cameras. The core idea behind our approach is to "lift" each image individually into a frustum of features for each camera, then "splat" all frustums into a rasterized bird's-eye-view grid. By training on the entire camera rig, we provide evidence that our model is able to learn not only how to represent images but how to fuse predictions from all cameras into a single cohesive representation of the scene while being robust to calibration error. On standard bird's-eye-view tasks such as object segmentation and map segmentation, our model outperforms all baselines and prior work. In pursuit of the goal of learning dense representations for motion planning, we show that the representations inferred by our model enable interpretable end-to-end motion planning by "shooting" template trajectories into a bird's-eye-view cost map output by our network. We benchmark our approach against models that use oracle depth from lidar. Project page: [https://nv-tlabs.github.io/lift-splat-shoot/](https://nv-tlabs.github.io/lift-splat-shoot/).

**Questions/Requests:** Please file an [issue](https://github.com/nv-tlabs/lift-splat-shoot/issues) if you have any questions or requests about the code or the [paper](https://arxiv.org/abs/2008.05711). If you prefer your question to be private, you can alternatively email me at jphilion@nvidia.com.

### Citation
If you found this codebase useful in your research, please consider citing
```
@inproceedings{philion2020lift,
    title={Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D},
    author={Jonah Philion and Sanja Fidler},
    booktitle={Proceedings of the European Conference on Computer Vision},
    year={2020},
}
```

### Preparation
Download nuscenes data from [https://www.nuscenes.org/](https://www.nuscenes.org/). Install dependencies.

```
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

### Pre-trained Model
Download a pre-trained BEV vehicle segmentation model from here: [https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing](https://drive.google.com/file/d/1bsUYveW_eOqa4lglryyGQNeC4fyQWvQQ/view?usp=sharing)

| Vehicle IOU (reported in paper)        | Vehicle IOU (this repository)         |
|:-------------:|:-------------:| 
| 32.07      | 33.03 |

### Evaluate a model
Evaluate the IOU of a model on the nuScenes validation set. To evaluate on the "mini" split, pass `mini`. To evaluate on the "trainval" split, pass `trainval`.

```
python main.py eval_model_iou mini/trainval --modelf=MODEL_LOCATION --dataroot=NUSCENES_ROOT
```

### Visualize Predictions
Visualize the BEV segmentation output by a model:

```
python main.py viz_model_preds mini/trainval --modelf=MODEL_LOCATION --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
```
<img src="./imgs/eval.gif">

### Visualize Input/Output Data (optional)
Run a visual check to make sure extrinsics/intrinsics are being parsed correctly. Left: input images with LiDAR scans projected using the extrinsics and intrinsics. Middle: the LiDAR scan that is projected. Right: X-Y projection of the point cloud generated by the lift-splat model. Pass `--viz_train=True` to view data augmentation.

```
python main.py lidar_check mini/trainval --dataroot=NUSCENES_ROOT --viz_train=False
```
<img src="./imgs/check.gif">

### Train a model (optional)
Train a model. Monitor with tensorboard.

```
python main.py train mini/trainval --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0
tensorboard --logdir=./runs --bind_all
```

### Acknowledgements
Thank you to Sanja Fidler, as well as David Acuna, Daiqing Li, Amlan Kar, Jun Gao, Kevin, Xie, Karan Sapra, the NVIDIA AV Team, and NVIDIA Research for their help in making this research possible.
