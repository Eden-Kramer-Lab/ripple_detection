---
name: release
description: Use when preparing or cutting a ripple_detection release - updating CHANGELOG.md (and MIGRATING.md for a major version), tagging vX.Y.Z, and pushing the tag that triggers the PyPI release workflow.
---

# Releasing ripple_detection

When preparing a new release:

```bash
# 1. Run all tests to ensure everything passes
pytest    # tests/ and the docstring examples in src/

# 2. Run code quality checks
ruff format --check src/ tests/
ruff check src/ tests/
mypy src/

# 3. Update CHANGELOG.md (and MIGRATING.md for a major version: the calls to
#    change and why results differ; the changelog links it rather than repeating it)
# - Add new version section with date: ## [X.Y.Z] - YYYY-MM-DD
# - Document all changes under appropriate headers:
#   - Added (new features)
#   - Changed (changes to existing functionality)
#   - Deprecated (soon-to-be removed features)
#   - Removed (removed features)
#   - Fixed (bug fixes)
#   - Security (security fixes)
# - List closed issues: "Closes #N"
# - Update comparison links at bottom of file

# 4. Commit the changelog
git add CHANGELOG.md
git commit -m "Update CHANGELOG for vX.Y.Z release"
git push origin master

# 5. Create and push annotated git tag
git tag -a vX.Y.Z -m "Release vX.Y.Z

## New Features
- Feature description

## Improvements
- Improvement description

Closes #N"

git push origin vX.Y.Z

# The tag push triggers the automated GitHub Actions release workflow:
# - Runs tests on Python 3.10 through 3.14, and at the dependency floors
# - Builds source distribution and wheels
# - Publishes to PyPI
# - Creates GitHub release with auto-generated notes
```

**Important Notes:**
- Always update CHANGELOG.md BEFORE creating the tag
- The tag must be an annotated tag (use `-a` flag) with a meaningful message
- Version follows semantic versioning (MAJOR.MINOR.PATCH)
- The version in `src/ripple_detection/_version.py` is auto-generated from the git tag by hatch-vcs
- Monitor the release workflow at: https://github.com/Eden-Kramer-Lab/ripple_detection/actions
