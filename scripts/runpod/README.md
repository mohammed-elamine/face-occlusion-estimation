# RunPod Scripts

Helpers for preparing a RunPod instance and launching experiments.

```bash
# First launch: provision the pod and sync the env for a specific branch.
# The dependency env (uv.lock) is branch-specific, so pass the branch you want.
bash scripts/runpod/setup_pod.sh --branch feat/lora
bash scripts/runpod/setup_pod.sh -b main --extra synthetic   # with an optional extra

bash scripts/runpod/run_experiment.sh configs/baseline.yaml
bash scripts/runpod/run_experiment_tmux.sh configs/baseline.yaml
```

`setup_pod.sh` is the one-time pod bootstrap (SSH, clone, persistent symlinks,
system packages, uv install, tmux) plus a branch-aware `uv sync`. Re-running it to
switch branches is safe: pass a different `--branch` (see `--help` for all flags).

RunPod logs are written under `outputs/runpod_logs/`.
