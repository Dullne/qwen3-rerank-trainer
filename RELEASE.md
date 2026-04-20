# Release Workflow

This package is released from this directory: `/data/train_res/code/rerank/qwen3-rerank-trainer`.

## Source of Truth

- TrainFactory development source:
  - `/data/train_res/code/TrainFactory/train_factory/qwen3_rerank_core/losses`
  - `/data/train_res/code/TrainFactory/train_factory/qwen3_rerank_core/rl`
- Publish source:
  - `src/qwen3_rerank_trainer/losses`
  - `src/qwen3_rerank_trainer/rl`

Always sync from TrainFactory before cutting a release.

## Sync

```bash
python scripts/sync_from_trainfactory.py
```

Dry-run:

```bash
python scripts/sync_from_trainfactory.py --check
```

## Test

```bash
python -m pytest -q tests/test_losses.py tests/test_dataset_collator.py tests/test_metrics.py
```

## Version Bump

Update:

- `pyproject.toml` -> `[project].version`
- `CHANGELOG.md`

Recommended policy:

- Patch: bug fix or internal cleanup
- Minor: new loss / reward / training behavior
- Major: breaking API or CLI changes

## Build

```bash
rm -rf build dist src/qwen3_rerank_trainer.egg-info
python -m build
```

## Git Release

```bash
git status
git add .
git commit -m "Release qwen3-rerank-trainer X.Y.Z"
git tag vX.Y.Z
git push origin master
git push origin vX.Y.Z
```

## Publish Package

If PyPI publishing is configured:

```bash
python -m twine upload dist/*
```

If using another registry, replace the upload command accordingly.
