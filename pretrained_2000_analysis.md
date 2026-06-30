# Pretrained 2000-Step Debug Run

This run fixes the previous EfficientNet cache-path issue by setting `TORCH_HOME` to the project-local cache. The model loaded ImageNet-pretrained EfficientNet-B0 weights successfully:

```text
Loaded pretrained weights for efficientnet-b0
```

## Command

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .matplotlib).Path
$env:TORCH_HOME=(Resolve-Path .cache\torch).Path
$env:PYTHONPATH=(Resolve-Path .).Path
$env:PYTHONIOENCODING='utf-8'

python scripts\debug_train.py `
  --dataroot=data\nuscenes `
  --version=mini `
  --device=cuda:0 `
  --batch-size=1 `
  --steps=2000 `
  --nworkers=0 `
  --outdir=outputs\debug_train_mini_2000_pretrained_staged `
  --final-h=96 `
  --final-w=256 `
  --dbound 4.0 44.0 2.0 `
  --save-every=200 `
  --eval-every=200 `
  --eval-batches=20
```

## Outputs

- Training log: `outputs/debug_train_mini_2000_pretrained_staged/loss_log.csv`
- Validation log: `outputs/debug_train_mini_2000_pretrained_staged/eval_log.csv`
- Loss curve: `outputs/debug_train_mini_2000_pretrained_staged/loss_curve.png`
- Validation metric curve: `outputs/debug_train_mini_2000_pretrained_staged/eval_metrics.png`
- Best qualitative checkpoint so far: `outputs/debug_train_mini_2000_pretrained_staged/debug_model_step1400.pt`
- GT vs prediction: `outputs/visualizations_mini_1400_pretrained_axes/bev_gt_vs_pred.png`
- Threshold sweep: `outputs/visualizations_mini_1400_pretrained_axes/bev_threshold_sweep.png`
- Checkpoint comparison: `outputs/checkpoint_compare_400_1400_pretrained_axes/sample_000_bev_compare.png`

The `.pt` checkpoint files are intentionally not committed to git.

## Validation Metrics

| step | val loss | sigmoid max | IoU@0.5 | precision@0.5 | recall@0.5 | IoU@0.25 | precision@0.25 | recall@0.25 | IoU@0.1 | recall@0.1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.3188 | 0.3442 | 0.0000 | 0.0000 | 0.0000 | 0.0091 | 0.0836 | 0.0101 | 0.0533 | 0.1628 |
| 400 | 0.3532 | 0.9357 | 0.0653 | 0.2140 | 0.0859 | 0.0963 | 0.1584 | 0.1971 | 0.1054 | 0.3926 |
| 600 | 0.3285 | 0.9156 | 0.0380 | 0.3676 | 0.0406 | 0.0856 | 0.2623 | 0.1128 | 0.1027 | 0.2852 |
| 800 | 0.3444 | 0.9264 | 0.0582 | 0.3556 | 0.0651 | 0.1112 | 0.2335 | 0.1750 | 0.1082 | 0.2759 |
| 1000 | 0.4131 | 0.8097 | 0.0241 | 0.3621 | 0.0252 | 0.0621 | 0.2197 | 0.0796 | 0.0948 | 0.1718 |
| 1200 | 0.3682 | 0.8997 | 0.0213 | 0.4741 | 0.0218 | 0.0799 | 0.3810 | 0.0918 | 0.1035 | 0.1713 |
| 1400 | 0.3084 | 0.9662 | 0.0501 | 0.2629 | 0.0583 | 0.1245 | 0.2133 | 0.2301 | 0.1228 | 0.4412 |
| 1600 | 0.3427 | 1.0000 | 0.0172 | 0.2844 | 0.0180 | 0.0814 | 0.2509 | 0.1075 | 0.1212 | 0.3122 |
| 1800 | 0.3240 | 0.9987 | 0.0436 | 0.4852 | 0.0458 | 0.1036 | 0.3254 | 0.1319 | 0.1167 | 0.3184 |
| 2000 | 0.4173 | 0.9775 | 0.0438 | 0.4115 | 0.0467 | 0.0994 | 0.3082 | 0.1279 | 0.1211 | 0.2102 |

## Interpretation

Compared with the earlier random-initialized 500-step run, the pretrained run learns the training samples much more strongly: late training batch IoU often reaches 0.3 to 0.5. However, validation IoU remains modest and does not improve monotonically.

The best current checkpoint is `step1400`:

- best validation loss in this run;
- best `IoU@0.25`;
- best `IoU@0.1`;
- highest low-threshold recall.

The final `step2000` checkpoint has stronger confidence and higher precision at some thresholds, but validation loss is worse and recall drops. This suggests overfitting on nuScenes mini rather than reliable generalization.

## Recommendation

For the current mini-dataset reproduction, use `debug_model_step1400.pt` for qualitative visualization and report figures.

To improve IoU meaningfully, the next useful move is not simply more mini training. The better options are:

1. train on full nuScenes trainval;
2. increase input resolution once GPU memory allows;
3. evaluate on more validation batches or the full validation split;
4. tune `pos_weight`, threshold selection, and augmentation;
5. keep staged checkpoint selection instead of trusting the last checkpoint.

## Component-Level GT Analysis

To check whether small GT blobs disproportionately affect IoU, I added `scripts/component_eval.py` and evaluated `debug_model_step1400.pt` on 20 validation batches.

Command:

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .matplotlib).Path
$env:TORCH_HOME=(Resolve-Path .cache\torch).Path
$env:PYTHONPATH=(Resolve-Path .).Path
$env:PYTHONIOENCODING='utf-8'

python scripts\component_eval.py `
  --dataroot=data\nuscenes `
  --version=mini `
  --modelf=outputs\debug_train_mini_2000_pretrained_staged\debug_model_step1400.pt `
  --outdir=outputs\component_eval_step1400_pretrained `
  --device=cuda:0 `
  --max-batches=20 `
  --thresholds 0.1 0.25 0.5 `
  --tiny-area=20
```

Outputs:

- `outputs/component_eval_step1400_pretrained/components.csv`
- `outputs/component_eval_step1400_pretrained/summary.csv`

Summary:

| group | components | area cells | component recall@0.1 | cell coverage@0.1 | component recall@0.25 | cell coverage@0.25 | component recall@0.5 | cell coverage@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 447 | 33977 | 0.5817 | 0.4412 | 0.3714 | 0.2301 | 0.1499 | 0.0583 |
| tiny area < 20 | 64 | 873 | 0.4062 | 0.2646 | 0.0938 | 0.0710 | 0.0156 | 0.0092 |
| large area >= 20 | 383 | 33104 | 0.6110 | 0.4458 | 0.4178 | 0.2343 | 0.1723 | 0.0596 |

This supports the visual observation: tiny GT blobs are much harder for the current model to hit. At threshold 0.25, large components have 41.8% component recall, while tiny components have only 9.4%. A right-side small GT point can therefore lower recall and IoU noticeably, especially on a small validation sample.
