from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch before data loading or model creation.

    When ``deterministic`` is true, also force cuDNN into a deterministic mode.
    This trades some GPU throughput for bit-reproducibility, which matters when
    comparing small ablation deltas (most of the roadmap is ablations).
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``DataLoader`` ``worker_init_fn`` that seeds NumPy/Python per worker.

    PyTorch seeds each worker's ``torch`` RNG from the base generator, but NumPy
    and the ``random`` module are left unseeded in forked workers. Derive their
    seeds from ``torch.initial_seed()`` so augmentations are reproducible and not
    correlated across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_generator(seed: int) -> torch.Generator:
    """Return a seeded ``torch.Generator`` for ``DataLoader`` shuffling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
