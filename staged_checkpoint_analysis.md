# Staged Checkpoint Analysis

This note records the mini nuScenes debug run that saved and evaluated checkpoints every 100 training steps. The goal was to verify whether the 500-step model was genuinely better than the earlier 100-step model, or whether it only became more confident in a smaller region.

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
  --steps=500 `
  --nworkers=0 `
  --outdir=outputs\debug_train_mini_500_staged `
  --final-h=96 `
  --final-w=256 `
  --dbound 4.0 44.0 2.0 `
  --save-every=100 `
  --eval-every=100 `
  --eval-batches=20
```

## Saved Outputs

- Training log: `outputs/debug_train_mini_500_staged/loss_log.csv`
- Validation log: `outputs/debug_train_mini_500_staged/eval_log.csv`
- Loss curve: `outputs/debug_train_mini_500_staged/loss_curve.png`
- Validation metric curve: `outputs/debug_train_mini_500_staged/eval_metrics.png`
- Checkpoints: `debug_model_step100.pt`, `debug_model_step200.pt`, `debug_model_step300.pt`, `debug_model_step400.pt`, `debug_model_step500.pt`, `debug_model.pt`

The checkpoint files are intentionally kept out of git because they are large generated artifacts.

## Validation Metrics

| step | val loss | sigmoid max | IoU@0.5 | precision@0.5 | recall@0.5 | IoU@0.25 | precision@0.25 | recall@0.25 | IoU@0.1 | recall@0.1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.2880 | 0.4312 | 0.0000 | 0.0000 | 0.0000 | 0.0472 | 0.1701 | 0.0614 | 0.1126 | 0.3512 |
| 200 | 0.2582 | 0.6292 | 0.0147 | 0.2424 | 0.0155 | 0.1406 | 0.2049 | 0.3095 | 0.1429 | 0.5946 |
| 300 | 0.2999 | 0.9016 | 0.0302 | 0.2859 | 0.0327 | 0.0632 | 0.1291 | 0.1102 | 0.0989 | 0.3914 |
| 400 | 0.2850 | 0.9168 | 0.0735 | 0.3316 | 0.0864 | 0.1277 | 0.1612 | 0.3804 | 0.0985 | 0.5847 |
| 500 | 0.2903 | 0.8265 | 0.0459 | 0.3682 | 0.0498 | 0.1293 | 0.2327 | 0.2254 | 0.0990 | 0.5332 |

## Interpretation

The staged run confirms that the model behavior is not monotonic over this very small debug training setup.

- Step 100 is under-confident: the maximum sigmoid score is only 0.431, so IoU@0.5 is zero. At low threshold it still covers some GT regions, which is why the visual output can look broadly reasonable.
- Step 200 gives the best validation loss and the best IoU@0.1 / IoU@0.25 tradeoff among early checkpoints. It predicts many more positive cells, so recall improves.
- Step 300 becomes much more confident but spatial alignment degrades at lower thresholds. This is a warning sign that confidence alone is not enough.
- Step 400 is the best high-threshold checkpoint in this run. It has the highest IoU@0.5 and better recall at 0.5 than the others.
- Step 500 becomes more conservative at 0.5 but improves precision and slightly edges out step 400 at IoU@0.25. This matches the observation that it may look better in one local region while losing some broad coverage.

## Current Recommendation

Use `debug_model_step400.pt` when inspecting high-confidence predictions, and compare it with `debug_model_step200.pt` or `debug_model_step500.pt` for lower-threshold qualitative analysis.

For the next round, the most useful improvements are:

1. keep periodic checkpoint evaluation enabled;
2. add checkpoint comparison visualizations for steps 100/200/300/400/500;
3. reduce random fluctuation by evaluating more validation batches;
4. train longer only after confirming that validation IoU and visualization quality improve together.

## Reproduction Scripts

Single-checkpoint visualization with BEV axes:

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .matplotlib).Path
$env:PYTHONPATH=(Resolve-Path .).Path

python scripts\visualize_debug.py `
  --dataroot=data\nuscenes `
  --version=mini `
  --modelf=outputs\debug_train_mini_500_staged\debug_model_step400.pt `
  --outdir=outputs\visualizations_mini_400_axes `
  --device=cuda:0 `
  --threshold=0.25
```

Checkpoint comparison:

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .matplotlib).Path
$env:PYTHONPATH=(Resolve-Path .).Path

python scripts\compare_checkpoints.py `
  --dataroot=data\nuscenes `
  --version=mini `
  --model-a=outputs\debug_train_mini_500_staged\debug_model_step200.pt `
  --model-b=outputs\debug_train_mini_500_staged\debug_model_step400.pt `
  --label-a=step200 `
  --label-b=step400 `
  --outdir=outputs\checkpoint_compare_200_400_axes `
  --device=cuda:0 `
  --threshold=0.25
```
