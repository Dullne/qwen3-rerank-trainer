# Release Workflow

`/data/train_res/code/rerank/qwen3-rerank-trainer` is the single source of truth for the
`qwen3-rerank-trainer` package.

All new loss functions, RL logic, training changes, evaluation changes, and packaging updates
should be developed and released directly from this repository.

## Development Rule

- Do not sync code from `TrainFactory`.
- Do not maintain a second copy of `qwen3_rerank_trainer` elsewhere.
- If `TrainFactory` needs new functionality, update this repository first and consume it as a
  dependency.

## Pre-release Checklist

Before cutting a release:

1. Update package code in `src/qwen3_rerank_trainer/`.
2. Run tests locally.
3. Update the version in `pyproject.toml`.
4. Update `CHANGELOG.md`.
5. Rebuild the package artifacts.

## Test

Run the full test suite:

```bash
python -m pytest -q
```

## Version Bump

Update:

- `pyproject.toml` -> `[project].version`
- `src/qwen3_rerank_trainer/__init__.py` -> `__version__`
- `CHANGELOG.md`

Suggested versioning policy:

- Patch: bug fix, packaging fix, docs fix, small internal cleanup
- Minor: new loss, reward, CLI option, trainer behavior, or new public API
- Major: breaking API, CLI, data format, or training behavior change

## Build

Clean old artifacts and rebuild:

```bash
rm -rf build dist src/qwen3_rerank_trainer.egg-info
python -m build
```

Expected artifacts:

- `dist/qwen3_rerank_trainer-X.Y.Z.tar.gz`
- `dist/qwen3_rerank_trainer-X.Y.Z-py3-none-any.whl`

## Git Release

```bash
git status
git add .
git commit -m "Release qwen3-rerank-trainer X.Y.Z"
git tag vX.Y.Z
git push origin master --tags
```

If your default branch changes in the future, replace `master` with the actual default branch.

## Publish Package

If publishing to PyPI:

```bash
python -m twine upload dist/*
```

If publishing to another registry, replace the upload target accordingly.

## Post-release Check

After release:

1. Confirm the Git tag exists on GitHub.
2. Confirm the uploaded package version matches the Git tag.
3. In downstream repositories such as `TrainFactory`, upgrade to the released package version
   instead of copying source code.
