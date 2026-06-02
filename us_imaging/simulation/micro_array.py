"""Parameterized micro-ultrasound array response models.

The functions here bridge the current lightweight Gaussian-pulse RF simulator
with micro-array engineering parameters such as Q, bandwidth, element mismatch,
frequency drift, element dropout, and receive-chain noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from us_imaging.simulation.acquire import (
    TransducerArray,
    array_engineering_metrics,
    plane_wave_rf,
)
from us_imaging.simulation.multibeam_data import (
    extract_multibeam_patches,
    generate_multibeam_phantom,
)
from us_imaging.simulation.phantom import Phantom


@dataclass(frozen=True)
class MicroArrayConfig:
    array_type: str
    n_elements: int = 64
    pitch_m: float = 0.3e-3
    center_freq_hz: float = 5e6
    fractional_bandwidth: float = 0.7
    q_factor: float = 2.0
    sound_speed_m_s: float = 1540.0
    sampling_freq_hz: float = 40e6
    element_amp_std: float = 0.05
    element_phase_std_deg: float = 2.0
    element_fc_std_fraction: float = 0.01
    element_failure_rate: float = 0.0
    receive_noise_snr_db: float | None = 35.0

    def to_transducer_array(self) -> TransducerArray:
        return TransducerArray(
            n_elements=self.n_elements,
            pitch=self.pitch_m,
            center_freq=self.center_freq_hz,
            bandwidth=self.fractional_bandwidth,
            sound_speed=self.sound_speed_m_s,
            sampling_freq=self.sampling_freq_hz,
        )

    def with_overrides(self, **kwargs) -> "MicroArrayConfig":
        return replace(self, **kwargs)


MICRO_ARRAY_TEMPLATES = {
    "default": MicroArrayConfig(array_type="default"),
    "pmut_like": MicroArrayConfig(
        array_type="pmut_like",
        n_elements=16,
        pitch_m=100e-6,
        center_freq_hz=16e6,
        fractional_bandwidth=0.45,
        q_factor=2.2,
        sampling_freq_hz=80e6,
        element_amp_std=0.08,
        element_phase_std_deg=4.0,
        element_fc_std_fraction=0.02,
        receive_noise_snr_db=32.0,
    ),
    "cmut_like": MicroArrayConfig(
        array_type="cmut_like",
        n_elements=32,
        pitch_m=75e-6,
        center_freq_hz=15e6,
        fractional_bandwidth=0.9,
        q_factor=1.2,
        sampling_freq_hz=100e6,
        element_amp_std=0.04,
        element_phase_std_deg=2.5,
        element_fc_std_fraction=0.015,
        receive_noise_snr_db=35.0,
    ),
}


def get_micro_array_config(template: str = "default", **overrides) -> MicroArrayConfig:
    if template not in MICRO_ARRAY_TEMPLATES:
        choices = ", ".join(sorted(MICRO_ARRAY_TEMPLATES))
        raise ValueError(f"Unknown micro-array template {template!r}. Choose one of: {choices}")
    return MICRO_ARRAY_TEMPLATES[template].with_overrides(**overrides)


def sample_element_response(config: MicroArrayConfig,
                            seed: int | None = None) -> dict[str, np.ndarray]:
    """Sample per-element response parameters for one simulated array instance."""
    rng = np.random.RandomState(seed)
    amp = 1.0 + rng.normal(0.0, config.element_amp_std, config.n_elements)
    phase_rad = np.radians(rng.normal(0.0, config.element_phase_std_deg, config.n_elements))
    fc_hz = config.center_freq_hz * (
        1.0 + rng.normal(0.0, config.element_fc_std_fraction, config.n_elements)
    )
    active = rng.rand(config.n_elements) >= config.element_failure_rate

    return {
        "amp_factors": amp.astype(np.float32),
        "phase_offsets_rad": phase_rad.astype(np.float32),
        "center_freq_offsets_hz": fc_hz.astype(np.float32),
        "active_mask": active.astype(bool),
    }


def _fractional_sample_shift(signal: np.ndarray, shift_samples: float) -> np.ndarray:
    x = np.arange(signal.shape[-1], dtype=np.float32)
    return np.interp(x - shift_samples, x, signal, left=0.0, right=0.0).astype(np.float32)


def apply_receive_chain(rf_data: np.ndarray,
                        config: MicroArrayConfig,
                        seed: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply element mismatch, phase offsets, dropout, and AWGN to RF data."""
    response = sample_element_response(config, seed=seed)
    out = np.asarray(rf_data, dtype=np.float32).copy()

    if out.ndim != 3:
        raise ValueError(f"Expected RF data [events, elements, samples], got shape {out.shape}")
    if out.shape[1] != config.n_elements:
        raise ValueError(f"Expected {config.n_elements} elements, got {out.shape[1]}")

    phase_to_samples = config.sampling_freq_hz / (2 * np.pi * config.center_freq_hz)
    for elem_idx in range(config.n_elements):
        if not response["active_mask"][elem_idx]:
            out[:, elem_idx, :] = 0.0
            continue

        out[:, elem_idx, :] *= response["amp_factors"][elem_idx]
        shift = response["phase_offsets_rad"][elem_idx] * phase_to_samples
        if abs(float(shift)) > 1e-6:
            for event_idx in range(out.shape[0]):
                out[event_idx, elem_idx, :] = _fractional_sample_shift(
                    out[event_idx, elem_idx, :],
                    float(shift),
                )

    if config.receive_noise_snr_db is not None:
        signal_power = float(np.mean(out ** 2)) + 1e-12
        noise_power = signal_power / (10 ** (config.receive_noise_snr_db / 10))
        rng = np.random.RandomState(None if seed is None else seed + 100_003)
        noise = rng.randn(*out.shape).astype(np.float32) * np.sqrt(noise_power)
        out = out + noise

    return out.astype(np.float32), response


def micro_array_metrics(config: MicroArrayConfig,
                        patches: np.ndarray | None = None,
                        response: dict[str, np.ndarray] | None = None) -> dict:
    metrics = array_engineering_metrics(config.to_transducer_array(), patches=patches)
    metrics.update({
        "array_type": config.array_type,
        "q_factor": float(config.q_factor),
        "element_amp_std": float(config.element_amp_std),
        "element_phase_std_deg": float(config.element_phase_std_deg),
        "element_fc_std_fraction": float(config.element_fc_std_fraction),
        "element_failure_rate": float(config.element_failure_rate),
        "receive_noise_snr_db": (
            None if config.receive_noise_snr_db is None else float(config.receive_noise_snr_db)
        ),
    })
    if response is not None:
        metrics.update({
            "active_elements": int(np.sum(response["active_mask"])),
            "failed_elements": int(config.n_elements - np.sum(response["active_mask"])),
            "amp_factor_mean": float(np.mean(response["amp_factors"])),
            "amp_factor_std": float(np.std(response["amp_factors"])),
            "phase_offset_std_deg": float(np.degrees(np.std(response["phase_offsets_rad"]))),
            "center_freq_mean_mhz": float(np.mean(response["center_freq_offsets_hz"]) / 1e6),
        })
    return metrics


def parameterized_plane_wave_rf(phantom: Phantom,
                                config: MicroArrayConfig,
                                angles: np.ndarray | None = None,
                                seed: int | None = None) -> tuple[np.ndarray, dict]:
    """Generate plane-wave RF data with a parameterized receive chain."""
    array = config.to_transducer_array()
    rf = plane_wave_rf(phantom, array, angles=angles)
    rf, response = apply_receive_chain(rf, config, seed=seed)
    return rf, micro_array_metrics(config, response=response)


def generate_parameterized_multibeam_data(
    n_phantoms: int,
    config: MicroArrayConfig,
    n_point_per_phantom: int = 3,
    n_dense: int = 200,
    n_beams: int = 3,
    patch_len: int = 256,
    base_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate balanced multi-beam patches using parameterized array effects."""
    array = config.to_transducer_array()
    all_patches, all_labels = [], []
    last_response = None

    for idx in range(n_phantoms):
        rng = np.random.RandomState(base_seed + idx * 1000)
        half_beams = n_beams // 2
        x_range = (
            float(array.element_x[half_beams]),
            float(array.element_x[-half_beams - 1]),
        )
        phantom, point_xs, point_zs = generate_multibeam_phantom(
            n_point=n_point_per_phantom,
            n_dense=n_dense,
            x_range=x_range,
            rng=rng,
            sound_speed=config.sound_speed_m_s,
        )
        rf = plane_wave_rf(phantom, array, angles=np.array([0.0]))
        rf, last_response = apply_receive_chain(rf, config, seed=base_seed + idx)
        patches, labels = extract_multibeam_patches(
            rf,
            array,
            point_xs,
            point_zs,
            n_beams=n_beams,
            patch_len=patch_len,
            rng=rng,
        )
        if len(patches) > 0:
            all_patches.append(patches)
            all_labels.append(labels)

    if not all_patches:
        raise RuntimeError("No parameterized multibeam patches were generated")

    patches_out = np.concatenate(all_patches, axis=0).astype(np.float32)
    labels_out = np.concatenate(all_labels, axis=0).astype(np.int64)
    metrics = micro_array_metrics(config, patches=patches_out, response=last_response)
    return patches_out, labels_out, metrics
