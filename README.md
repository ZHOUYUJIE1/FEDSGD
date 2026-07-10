# FEDSGD Reviewer5 Experiment Implementation

This repository archives the reviewer5 experiment implementation for federated SGD compression, adaptive Top-k communication, CIFAR-10 backdoor evaluation, and corrected retain-protected unlearning experiments.

## Environment

Use the existing Conda environment:

```bash
conda activate zlf_2
```

Data files, downloaded datasets, caches, model checkpoints, and large logs are not included in this repository.

## Main Entry Points

- Federated training baseline and original workflow: `main.py`, `main_1.py`
- Aligned CIFAR-10 adaptive Top-k Table IV runner: `table_iv_ours_main1_aligned.py`
- Backdoor attack and ASR runner: `table_iv_ours_backdoor_runner.py`
- Shared backdoor evaluation utilities: `table_iv_backdoor_utils.py`
- Corrected retain-protected unlearning confirmation: `table_iv_ours_revised_protocol_confirm.py`
- Local calibration for the revised unlearning protocol: `table_iv_ours_revised_protocol_local_calib.py`
- Offline unlearning diagnostics: `table_iv_ours_offline_unlearn_diagnostic.py`
- Stability, LR, and retain-grid sweeps: `table_iv_ours_unlearn_stability_grid.py`, `table_iv_ours_unlearn_lr_fine_clip1.py`, `table_iv_ours_unlearn_retain_grid.py`

## Adaptive Top-k and Communication

- Adaptive and fixed Top-k compressor implementation: `topk.py`
- Compressor base utilities: `base_compressor.py`
- Fixed-Avg fair communication comparison: `run_fixed_avg_experiment.py`
- Communication summaries are stored as small CSV/JSON/Markdown files under `results/fixed_avg/`.

## Measurement Matrix / Gradient Artifacts

Measurement-matrix and gradient-cache artifacts are generated during experiments in workspace/cache directories and are intentionally excluded from Git. Related generation and inspection logic lives in:

- `topk.py`
- `base_compressor.py`
- `inspect_grads.py`
- `model/client.py`
- `model/server.py`
- `model/server1.py`

## Checkpoints and Data

The repository intentionally excludes:

- CIFAR-10 and other datasets
- Training and unlearning checkpoints
- Cached gradients and pickled workspaces
- Large log files

Reproduce experiments by providing the required local data/checkpoint paths and running the entry points above in the `zlf_2` Conda environment.
