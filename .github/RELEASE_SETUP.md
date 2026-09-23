# GitHub Actions Auto-Release Setup

This document explains how to set up automatic PyPI publishing using the new `release.yml` workflow.

## Overview

The workflow automatically:

1. ✅ Runs quality checks (ruff format, ruff check, mypy)
2. ✅ Tests on Python 3.10 through 3.14, and at the dependency floors
3. ✅ Builds distributions (wheel + sdist)
4. ✅ Tests both distribution formats
5. 🚀 **Publishes to PyPI when you push a tag** (e.g., `v1.6.0`)
6. 📝 **Creates a GitHub Release with CHANGELOG notes**

## Setup Instructions

### 1. Configure PyPI Trusted Publishing (Recommended - No Tokens!)

PyPI now supports trusted publishing from GitHub Actions without API tokens:

1. Go to <https://pypi.org/manage/account/publishing/>
2. Click "Add a new pending publisher"
3. Fill in:
   - **PyPI Project Name**: `ripple-detection`
   - **Owner**: `Eden-Kramer-Lab` (your GitHub org/user)
   - **Repository**: `ripple_detection`
   - **Workflow name**: `release.yml`
   - **Environment name**: `pypi`
4. Save

That's it! No tokens needed. GitHub Actions will authenticate automatically.

### 2. Alternative: Use API Token (Legacy Method)

If you prefer the traditional method:

1. Go to <https://pypi.org/manage/account/token/>
2. Create an API token with scope limited to `ripple-detection` project
3. Copy the token (starts with `pypi-`)
4. Go to your GitHub repo → Settings → Secrets and variables → Actions
5. Create a new secret:
   - Name: `PYPI_API_TOKEN`
   - Value: paste your PyPI token
6. Update `.github/workflows/release.yml`:

   ```yaml
   # Add a `with` block to the pinned publish step:
   - name: Publish to PyPI
     uses: pypa/gh-action-pypi-publish@<the pinned commit>  # keep the pin
     with:
       user: __token__
       password: ${{ secrets.PYPI_API_TOKEN }}
   ```

### 3. Set Up CodeCov (Optional but Recommended)

For coverage reporting:

1. Go to <https://codecov.io/> and sign in with GitHub
2. Add the `ripple_detection` repository
3. Copy the upload token
4. Add as GitHub secret: `CODECOV_TOKEN`

## How to Release

### Simple Method: Tag and Push

```bash
# Make sure you're on master with all changes committed
git checkout master
git pull

# Create an annotated tag
git tag -a v1.6.0 -m "Release v1.6.0: Description here"

# Push the tag (this triggers the release!)
git push origin v1.6.0
```

The workflow will automatically:

- Run all tests
- Build the package
- Publish to PyPI
- Create a GitHub Release

### Monitoring the Release

1. Go to: <https://github.com/Eden-Kramer-Lab/ripple_detection/actions>
2. Find the "Test, Build, and Publish" workflow for your tag
3. Watch the progress through each job
4. When complete, check:
   - PyPI: <https://pypi.org/project/ripple-detection/>
   - GitHub Releases: <https://github.com/Eden-Kramer-Lab/ripple_detection/releases>

## Release Checklist

Before pushing a release tag:

- [ ] All changes committed and pushed to `master`
- [ ] `CHANGELOG.md` updated with version number and date
- [ ] Version tag follows semantic versioning (e.g., `v1.6.0`)
- [ ] All tests pass locally, docstring examples included: `pytest`
- [ ] Quality checks pass: `ruff format --check src/ tests/`, `ruff check src/ tests/`, and `mypy src/`
- [ ] Built and tested locally: `python -m build` works

## Troubleshooting

### Release fails at "Publish to PyPI"

**Error: "The user 'token' isn't allowed to upload"**

- Solution: Set up trusted publishing (see Setup section)

**Error: "File already exists"**

- Problem: Version already published
- Solution: Delete the tag, increment version, create new tag

### Tests fail on Python 3.X

- Check the Actions log for details
- Reproduce locally: `uv run --python 3.X pytest`
- Fix the issue, commit, and push

### Quality checks fail

- Run locally to see errors:

  ```bash
  ruff format --check src/ tests/
  ruff check src/ tests/
  mypy src/
  ```

- Fix issues, commit, and push

## Workflow Files

- `.github/workflows/release.yml` - Quality checks and tests on pushes and PRs to `master`; release on tags

## Comparison with Manual Release

| Task | Manual | Automated |
|------|--------|-----------|
| Run tests | `pytest` | ✅ Automatic |
| Check formatting | `ruff format --check` | ✅ Automatic |
| Check linting | `ruff check` | ✅ Automatic |
| Build package | `python -m build` | ✅ Automatic |
| Test install | Manual | ✅ Automatic (wheel & sdist) |
| Upload to PyPI | `twine upload` | ✅ Automatic |
| Create GitHub Release | Manual | ✅ Automatic |
| Extract CHANGELOG | Manual | ✅ Automatic |

## Best Practices

1. **Always test locally first** before pushing tags
2. **Use semantic versioning**:
   - `v1.0.0` - Major (breaking changes)
   - `v1.1.0` - Minor (new features, backward compatible)
   - `v1.0.1` - Patch (bug fixes)
3. **Update CHANGELOG.md** before creating release
4. **Use annotated tags** (not lightweight tags)
5. **Never force-push tags** to production

## Rollback

If you need to rollback a release:

1. **PyPI**: You cannot delete releases, but you can yank them:

   ```bash
   pip install --upgrade twine
   twine yank ripple-detection 1.6.0 -r pypi
   ```

2. **GitHub**: Delete the release and tag:

   ```bash
   # Delete GitHub release (via web UI)
   git tag -d v1.6.0                    # Delete local tag
   git push origin :refs/tags/v1.6.0    # Delete remote tag
   ```

## Questions?

- GitHub Actions docs: <https://docs.github.com/en/actions>
- PyPI Trusted Publishing: <https://docs.pypi.org/trusted-publishers/>
- Issues: File at <https://github.com/Eden-Kramer-Lab/ripple_detection/issues>
