# flake8: noqa
from ripple_detection.core import (
    filter_ripple_band,
    get_multiunit_population_firing_rate,
    normalize_signal,
    ripple_bandpass_filter,
)
from ripple_detection.detectors import (
    Carey_candidate_detector,
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Roumis_ripple_detector,
    Shvartsman_ripple_detector,
    Yu_ripple_detector,
    multiunit_HSE_detector,
)

try:
    from ripple_detection._version import __version__
except ImportError:
    __version__ = "unknown"
