# flake8: noqa
from ripple_detection.core import (
    filter_ripple_band,
    get_multiunit_population_firing_rate,
    normalize_signal,
)
from ripple_detection.detectors import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Roumis_ripple_detector,
    multiunit_HSE_detector,
)

try:
    from ripple_detection._version import __version__
except ImportError:
    __version__ = "unknown"
