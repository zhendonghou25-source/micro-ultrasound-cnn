"""Ultrasound simulation: phantom generation and RF data acquisition."""

from .acquire import TransducerArray, array_engineering_metrics
from .micro_array import (
    MICRO_ARRAY_TEMPLATES,
    MicroArrayConfig,
    apply_receive_chain,
    generate_parameterized_multibeam_data,
    get_micro_array_config,
    micro_array_metrics,
    parameterized_plane_wave_rf,
    sample_element_response,
)

__all__ = [
    "MICRO_ARRAY_TEMPLATES",
    "CompleteSimulationConfig",
    "MicroArrayConfig",
    "TransducerArray",
    "apply_receive_chain",
    "array_engineering_metrics",
    "generate_parameterized_multibeam_data",
    "get_micro_array_config",
    "micro_array_metrics",
    "parameterized_plane_wave_rf",
    "sample_element_response",
    "run_complete_simulation",
    "save_complete_simulation_report",
]


def __getattr__(name):
    if name in {
        "CompleteSimulationConfig",
        "run_complete_simulation",
        "save_complete_simulation_report",
    }:
        from . import complete_simulation

        return getattr(complete_simulation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
