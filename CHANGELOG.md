# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.0] - 2025-10-16

### Added

- **Critical UX Improvements**: Comprehensive input validation to prevent common user errors
  - Speed units validation: Warns when speed appears to be in m/s instead of cm/s, preventing silent data loss
  - Time units validation: Errors when time array appears to be in samples instead of seconds
  - Sampling frequency validation: Warns when time step doesn't match expected sampling rate
  - Array shape validation: Clear error messages for wrong LFP dimensions (1D, 0D, >2D arrays)
  - Array length validation: Detailed error showing exact sample counts when arrays don't match
  - Data length validation: Checks minimum data length required for filtering (954 samples)
  - Filter sampling frequency validation: Prevents crashes with incompatible sampling rates (<1200 Hz)

- **Enhanced Documentation**
  - Added comprehensive Troubleshooting section to README with solutions for common errors
  - Added Parameter Selection Guide table with recommended adjustment strategies
  - Added complete workflow example to `Kay_ripple_detector` docstring showing filtering → detection steps
  - Clarified that `multiunit_HSE_detector` accepts both binary (0/1) and spike count formats
  - Added amplitude units documentation to `simulate_LFP` (arbitrary units with SNR guidance)
  - Bolded all unit specifications in docstrings (**seconds**, **cm/s**, **Hz**) for clarity

- **Improved Error Messages**
  - All new error messages follow WHAT/WHY/HOW pattern with actionable solutions
  - Errors include working code examples for fixing issues
  - Validation errors provide specific details (array shapes, sample counts, expected values)

### Changed

- **BREAKING**: `_preprocess_detector_inputs()` now requires `sampling_frequency` and `speed_threshold` parameters for validation (internal function, unlikely to affect users)
- Enhanced parameter descriptions across all detector functions with clearer units and adjustment guidance
- Improved `filter_ripple_band()` error handling to catch issues before scipy crashes

### Fixed

- Prevented cryptic numpy errors (`axis 1 is out of bounds`, `operands could not be broadcast`) with clear validation errors
- Fixed misleading filter warning-then-crash behavior for incompatible sampling frequencies
- Improved empty result guidance - docstrings now suggest specific parameter adjustments

## [1.5.1] - Previous Release

### Added
- Improved code formatting and organization
- Enhanced test coverage to 98%
- Added snapshot tests for detector outputs
- Improved type annotations with modern Python 3.10+ syntax

### Changed
- Refactored test suite for better organization
- Updated CI/CD for ruff and mypy integration
- Modernized type hints (using `|` instead of `Union`)

---

## Release Notes for Upcoming Version

This release focuses on **user experience improvements** to make ripple_detection more accessible to neuroscientists with varying Python expertise. The changes eliminate common confusion points and provide clear, actionable guidance when issues occur.

### Key Improvements

1. **Prevents Silent Failures**: Validates speed/time units to catch mistakes before they cause incorrect results
2. **Better Error Messages**: All errors explain what went wrong, why, and exactly how to fix it
3. **Comprehensive Troubleshooting**: README now includes solutions for every common error scenario
4. **Clearer Workflow**: Documentation emphasizes the filter → detect workflow with complete examples
5. **No Breaking Changes**: All improvements are backward-compatible; existing code continues to work

### Upgrade Recommendation

**Strongly recommended** for all users. This release prevents common mistakes that could lead to:
- Zero detections due to wrong speed units
- Incorrect results due to time in samples instead of seconds
- Confusion about parameter meanings and units

All improvements are non-breaking and add safety guardrails without changing functionality.

### Migration Guide

No migration needed! All changes are backward-compatible. However, you may now see helpful warnings if:
- Your speed data appears to be in m/s (should be cm/s)
- Your time array doesn't match the specified sampling frequency
- Your sampling rate is incompatible with the pre-computed filter

These warnings help catch real issues - follow the suggested fixes in the warning messages.
