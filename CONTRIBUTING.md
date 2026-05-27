# Contributing

## Getting started

1. Clone the repository and switch to a feature branch:

   ```bash
   git clone https://github.com/mohammed-elamine/face-occlusion-estimation.git
   cd face-occlusion-estimation
   git checkout -b feat/your-feature
   ```

2. Run the setup check and install dependencies:

   ```bash
   make install
   ```

   This verifies prerequisites (`uv`, Python 3.11–3.12, etc.), installs all
   dependencies, and sets up the pre-commit hooks.

## Development workflow

- **Run all checks** before committing:

  ```bash
  make check
  ```

- **Format code** automatically:

  ```bash
  make format
  ```

- **Run pre-commit hooks** on all files:

  ```bash
  make pre-commit
  ```

- Run `make help` to see every available command.

## Dependencies

`uv.lock` is tracked in git to ensure everyone uses the same package versions.
When you add or update a dependency, commit the updated lock file along with `pyproject.toml`.
If you hit a merge conflict in `uv.lock`, accept either side and run `uv sync` to regenerate it.

## Code style

- [Ruff](https://docs.astral.sh/ruff/) handles both linting and formatting.
- Configuration lives in `pyproject.toml` — no extra config files needed.
- Pre-commit hooks enforce style on every commit automatically.

## Branch naming

| Prefix | Purpose |
|--------|---------|
| `feat/` | New feature |
| `fix/` | Bug fix |
| `chore/` | Maintenance / tooling |
| `exp/` | Experiment or exploration |

## Pull requests

- Keep PRs focused on a single change.
- Make sure `make check` passes before opening a PR.
- CI runs the same checks automatically.

---

<p align="center">
  <img src="assets/illustrations/challenge-accepted-meme.png" alt="Challenge accepted" width="180"/>
  <br/>
  <sub><em>Your first PR passed CI on the first try? Sure it did.</em></sub>
</p>
