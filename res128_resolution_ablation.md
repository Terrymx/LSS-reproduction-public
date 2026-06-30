# Resolution Ablation: 128 x 352 Input

This experiment tests solution 2: improve image input resolution while keeping the rest of the mini-dataset debug setup unchanged.

Compared configuration:

| item | previous run | this run |
| --- | --- | --- |
| dataset | nuScenes mini | nuScenes mini |
| backbone | EfficientNet-B0 pretrained | EfficientNet-B0 pretrained |
| input final dim | 96 x 256 | 128 x 352 |
| depth bins | `dbound 4 44 2` | `dbound 4 44 2` |
| batch size | 1 | 1 |
| train steps | 2000 | 2000 |
| save/eval interval | 200 | 200 |

## Smoke Test

The 20-step smoke run passed without CUDA OOM:

```powershell
python scripts\debug_train.py `
  --dataroot=data\nuscenes `
  --version=mini `
  --device=cuda:0 `
  --batch-size=1 `
  --steps=20 `
  --nworkers=0 `
  --outdir=outputs\debug_train_mini_res128_smoke `
  --final-h=128 `
  --final-w=352 `
  --dbound 4.0 44.0 2.0 `
  --save-every=20 `
  --eval-every=20 `
  --eval-batches=5
```

## Full Debug Run

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
  --outdir=outputs\debug_train_mini_res128_2000_pretrained_staged `
  --final-h=128 `
  --final-w=352 `
  --dbound 4.0 44.0 2.0 `
  --save-every=200 `
  --eval-every=200 `
  --eval-batches=20
```

Outputs:

- Training log: `outputs/debug_train_mini_res128_2000_pretrained_staged/loss_log.csv`
- Validation log: `outputs/debug_train_mini_res128_2000_pretrained_staged/eval_log.csv`
- Metric curve: `outputs/debug_train_mini_res128_2000_pretrained_staged/eval_metrics.png`
- Recommended checkpoint: `outputs/debug_train_mini_res128_2000_pretrained_staged/debug_model_step1800.pt`
- Visualization: `outputs/visualizations_mini_res128_step1800_axes/bev_gt_vs_pred.png`
- Threshold sweep: `outputs/visualizations_mini_res128_step1800_axes/bev_threshold_sweep.png`

The `.pt` checkpoint files remain local and are not committed to git.

## Validation Metrics

| step | val loss | IoU@0.5 | precision@0.5 | recall@0.5 | IoU@0.25 | precision@0.25 | recall@0.25 | IoU@0.1 | recall@0.1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.2606 | 0.0151 | 0.1483 | 0.0166 | 0.1254 | 0.1907 | 0.2679 | 0.0945 | 0.7763 |
| 400 | 0.3046 | 0.0298 | 0.3716 | 0.0314 | 0.1313 | 0.2242 | 0.2406 | 0.1203 | 0.3896 |
| 600 | 0.2735 | 0.0754 | 0.3599 | 0.0871 | 0.1491 | 0.2559 | 0.2632 | 0.1346 | 0.5223 |
| 800 | 0.2380 | 0.1200 | 0.3999 | 0.1464 | 0.1845 | 0.2549 | 0.4005 | 0.1531 | 0.6539 |
| 1000 | 0.3461 | 0.0396 | 0.4710 | 0.0414 | 0.1618 | 0.3840 | 0.2186 | 0.1852 | 0.3183 |
| 1200 | 0.2845 | 0.0493 | 0.6128 | 0.0509 | 0.1617 | 0.3506 | 0.2309 | 0.1711 | 0.4501 |
| 1400 | 0.3438 | 0.1058 | 0.4132 | 0.1245 | 0.1424 | 0.2632 | 0.2368 | 0.1460 | 0.3810 |
| 1600 | 0.3384 | 0.0811 | 0.5189 | 0.0877 | 0.1719 | 0.4275 | 0.2234 | 0.2100 | 0.3519 |
| 1800 | 0.2872 | 0.1262 | 0.5010 | 0.1443 | 0.2034 | 0.3852 | 0.3012 | 0.1984 | 0.4632 |
| 2000 | 0.3151 | 0.1230 | 0.4511 | 0.1447 | 0.1842 | 0.3341 | 0.2912 | 0.2015 | 0.4318 |

## Comparison With 96 x 256

Best 96 x 256 checkpoint from the previous run was `step1400`:

| config | checkpoint | IoU@0.5 | IoU@0.25 | IoU@0.1 |
| --- | ---: | ---: | ---: | ---: |
| 96 x 256 | 1400 | 0.0501 | 0.1245 | 0.1228 |
| 128 x 352 | 1800 | 0.1262 | 0.2034 | 0.1984 |

Increasing input resolution clearly improves BEV prediction quality on this mini debug setup. The model no longer needs to rely as much on broad low-threshold coverage: precision improves substantially at 0.25 and 0.5 while IoU also improves.

## Component-Level Analysis

For `128 x 352 step1800`:

| group | components | component recall@0.25 | cell coverage@0.25 | component recall@0.5 | cell coverage@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 447 | 0.3669 | 0.3012 | 0.2483 | 0.1443 |
| tiny area < 20 | 64 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| large area >= 20 | 383 | 0.4282 | 0.3091 | 0.2898 | 0.1481 |

The resolution increase helps the main vehicle regions a lot, but it does not solve tiny GT blobs. Tiny components are still missed almost completely at practical thresholds. This means higher image resolution improves localization for medium/large targets, but small-object recovery likely needs more data, stronger supervision, better thresholds/loss tuning, or full nuScenes training.

## Conclusion

Solution 2 is effective. The best mini-run IoU improved from:

- `IoU@0.5`: 0.0501 to 0.1262
- `IoU@0.25`: 0.1245 to 0.2034
- `IoU@0.1`: 0.1228 to 0.1984

This supports the idea that better feature resolution lets the model predict more accurately instead of only increasing coverage area. However, mini is still too small to fix generalization and tiny-object recall. The next meaningful step is to move this same configuration to full nuScenes trainval once the dataset is ready.
