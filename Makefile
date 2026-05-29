.DEFAULT_GOAL := help

# -------------------------------------------------------------------
# Makefile – local dev commands for face-occlusion-estimation
# -------------------------------------------------------------------

.PHONY: help check-setup install setup-cluster lint format format-check pre-commit check clean

help: ## Show this help message
	@printf '\nUsage: make <target>\n\n'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# -- Setup --------------------------------------------------------------

check-setup: ## Verify prerequisites (uv, Python, etc.)
	@bash scripts/check_setup.sh

install: check-setup ## Install project + dev dependencies with uv
	uv sync --group dev
	uv run pre-commit install

setup-cluster: ## Set up the Python environment on a Slurm/GPU cluster
	@bash scripts/setup_cluster_env.sh

# -- Code quality --------------------------------------------------------

lint: ## Run ruff linter
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

pre-commit: ## Run all pre-commit hooks on every file
	uv run pre-commit run --all-files

check: lint format-check ## Run all checks (lint + format)

# -- Cleanup -------------------------------------------------------------

clean: ## Remove build artifacts and caches
	rm -rf .ruff_cache __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
