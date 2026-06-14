# 08 — Cluster & Remote Execution

Training runs the same way locally, on a SLURM cluster, or on a RunPod GPU pod — only the
provisioning differs. Everything funnels into `python -m scripts.training.train --config`.

## Environment & CUDA wheels

The project uses `uv` (env `.venv`, Python 3.11–3.12). The PyTorch stack is pinned to
**CUDA 12.8 (cu128) wheels** via an explicit uv index in `pyproject.toml`
(`torch` / `torchvision` under `[tool.uv.sources]`, marker `linux x86_64`). cu128 ships
kernels for **both Ada (sm_89, e.g. RTX 4090) and Blackwell (sm_120, e.g. RTX 5090)**, so a
single environment runs on both architectures with a bare `uv sync` (needs an NVIDIA driver
recent enough for CUDA 12.8). Optional extras: `synthetic` (MediaPipe), `wandb`. On
non-linux dev machines the markers fall back to PyPI CPU/MPS wheels.

## SLURM — `jobs/train.slurm`

A single-GPU batch job. Override the config with an env var:

```bash
CONFIG_PATH=configs/baseline.yaml sbatch jobs/train.slurm
CONFIG_PATH=configs/x.yaml RUN_NAME=my_run sbatch jobs/train.slurm
```

It returns to the submit dir, loads cluster modules (`python/3.11`, `cuda/12.4`, `uv`),
activates `.venv` (created once via `scripts/setup/setup_cluster_env.sh`, which runs
`uv sync --frozen --extra wandb`), checks CUDA, and runs the trainer, tee-ing logs to
`outputs/slurm_logs/`. Note the module-loaded CUDA toolkit is only used for building
extensions; the torch wheels bundle their own runtime, so the host **driver** must support
CUDA 12.8 for the cu128 wheels to run.

## RunPod — `scripts/runpod/`

A pod has a persistent `/workspace` volume; the helpers keep the repo, datasets, outputs,
uv cache, and venv there so they survive container restarts.

- `setup_pod.sh [--branch <name>] [--extra <name>...] [--no-interactive]` — **one-time pod
  bootstrap**: SSH bootstrap, clone/fetch the repo, **interactive branch picker** (fzf, else
  a numbered menu) when no `--branch` is given on a TTY, create persistent symlinks
  (`data/raw`, `outputs/experiments`, `outputs/runpod_logs`), install system packages,
  `uv sync [--extra ...]`, and a CUDA check. The env is **branch-specific** (each branch
  carries its own `uv.lock`), so the branch is an explicit input.
- `sync_repo_to_remote.sh [branch]` (or interactive / `NONINTERACTIVE=1`) — make the pod
  repo exactly match `origin/<branch>` (fetch → optional picker → confirm-then-`reset
  --hard` → switch → `uv sync`), reusing the persistent cache/venv.
- `run_experiment.sh <config>` / `run_experiment_tmux.sh <config> [session]` — launch one
  training run (tee logs to `outputs/runpod_logs/`), optionally inside tmux so it survives
  disconnects.

### Shared persistent uv cache/venv

`setup_pod.sh`, `sync_repo_to_remote.sh`, and `run_experiment.sh` all export the **same**:

```
UV_LINK_MODE=copy
UV_CACHE_DIR=/workspace/cache/uv
UV_PROJECT_ENVIRONMENT=/workspace/venvs/face-occlusion-estimation
```

so the heavy install happens once into the persistent volume and every later `uv sync` /
`uv run` reuses it. (Keeping these in sync across the three scripts is what makes
first-launch work warm the cache the others read.)

### Running two experiments, one per GPU

Pin each process to a single device so Lightning's `devices="auto"` resolves to one GPU (no
DDP):

```bash
uv sync   # warm the shared venv once first
tmux new-session -d -s p0 "cd <repo> && CUDA_VISIBLE_DEVICES=0 bash scripts/runpod/run_experiment.sh configs/<a>.yaml; exec bash"
tmux new-session -d -s p1 "cd <repo> && CUDA_VISIBLE_DEVICES=1 bash scripts/runpod/run_experiment.sh configs/<b>.yaml; exec bash"
```

Put `CUDA_VISIBLE_DEVICES` **inside** the tmux command (a `tmux new-session` inherits the
server's environment, not the current shell's).
