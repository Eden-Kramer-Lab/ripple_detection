"""High-level detectors for sharp-wave ripple events and multiunit synchrony events."""

from ripple_detection.detectors._carey import Carey_candidate_detector
from ripple_detection.detectors._hse import multiunit_HSE_detector
from ripple_detection.detectors._lfp import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Roumis_ripple_detector,
    Shvartsman_ripple_detector,
    Yu_ripple_detector,
    get_Kay_ripple_consensus_trace,
    get_Yu_ripple_consensus_trace,
)
from ripple_detection.detectors._long import Long_sharp_wave_ripple_detector
from ripple_detection.detectors._state import state_intervals, theta_delta_ratio
from ripple_detection.detectors._trace import detect_events_from_trace
from ripple_detection.detectors._units import count_spikes_in_events, require_active_units
from ripple_detection.detectors._zugaro import Zugaro_ripple_detector

__all__ = [
    "Carey_candidate_detector",
    "Karlsson_ripple_detector",
    "Kay_ripple_detector",
    "Long_sharp_wave_ripple_detector",
    "Roumis_ripple_detector",
    "Shvartsman_ripple_detector",
    "Yu_ripple_detector",
    "Zugaro_ripple_detector",
    "count_spikes_in_events",
    "detect_events_from_trace",
    "get_Kay_ripple_consensus_trace",
    "get_Yu_ripple_consensus_trace",
    "multiunit_HSE_detector",
    "require_active_units",
    "state_intervals",
    "theta_delta_ratio",
]
