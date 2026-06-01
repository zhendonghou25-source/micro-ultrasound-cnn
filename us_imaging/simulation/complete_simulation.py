"""Complete lightweight simulation loop for micro-ultrasound experiments.

The scaffold explicitly connects four layers:
1. electroacoustic array response parameters,
2. propagation and scatterer RF generation,
3. receive-chain perturbations,
4. beamforming and B-mode reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

from us_imaging.beamforming.das import das_beamform_plane_wave
from us_imaging.reconstruction.compress import log_compress
from us_imaging.reconstruction.envelope import envelope_hilbert
from us_imaging.simulation.micro_array import (
    MicroArrayConfig,
    get_micro_array_config,
    micro_array_metrics,
    parameterized_plane_wave_rf,
)
from us_imaging.simulation.multibeam_data import (
    extract_multibeam_patches,
    generate_multibeam_phantom,
)


@dataclass(frozen=True)
class CompleteSimulationConfig:
    micro_array_template: str = "default"
    micro_array_config: MicroArrayConfig | None = None
    n_point: int = 3
    n_dense: int = 100
    n_beams: int = 3
    patch_len: int = 256
    base_seed: int = 42
    x_pixels: int = 32
    z_pixels: int = 64
    z_min_m: float = 0.005
    z_max_m: float = 0.04
    dynamic_range_db: float = 60.0
    f_number: float = 1.5
    apodization: str = "hanning"

    def resolved_array_config(self) -> MicroArrayConfig:
        if self.micro_array_config is not None:
            return self.micro_array_config
        return get_micro_array_config(self.micro_array_template)


def _grid_for_array(array, config: CompleteSimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    x_min = float(array.element_x[config.n_beams // 2])
    x_max = float(array.element_x[-config.n_beams // 2 - 1])
    x_grid = np.linspace(x_min, x_max, config.x_pixels, dtype=np.float32)
    z_grid = np.linspace(config.z_min_m, config.z_max_m, config.z_pixels, dtype=np.float32)
    return x_grid, z_grid


def run_complete_simulation(config: CompleteSimulationConfig) -> dict:
    """Run the full lightweight micro-array simulation loop."""
    array_config = config.resolved_array_config()
    array = array_config.to_transducer_array()
    x_grid, z_grid = _grid_for_array(array, config)
    rng = np.random.RandomState(config.base_seed)

    phantom, point_xs, point_zs = generate_multibeam_phantom(
        n_point=config.n_point,
        n_dense=config.n_dense,
        x_range=(float(x_grid[0]), float(x_grid[-1])),
        z_range=(config.z_min_m, config.z_max_m),
        rng=rng,
        sound_speed=array_config.sound_speed_m_s,
    )

    rf_data, response_metrics = parameterized_plane_wave_rf(
        phantom,
        array_config,
        angles=np.array([0.0]),
        seed=config.base_seed,
    )

    patches, labels = extract_multibeam_patches(
        rf_data,
        array,
        point_xs,
        point_zs,
        n_beams=config.n_beams,
        patch_len=config.patch_len,
        rng=rng,
    )

    beamformed = das_beamform_plane_wave(
        rf_data[0],
        array,
        x_grid,
        z_grid,
        angle=0.0,
        apodization=config.apodization,
        f_number=config.f_number,
    )
    envelope = envelope_hilbert(beamformed, axis=0)
    bmode = log_compress(envelope, dynamic_range=config.dynamic_range_db)

    metrics = micro_array_metrics(array_config, patches=patches)
    metrics.update(response_metrics)
    metrics.update({
        "simulation_layers": "electroacoustic,propagation,receive_chain,beamforming",
        "rf_events": int(rf_data.shape[0]),
        "rf_elements": int(rf_data.shape[1]),
        "rf_samples": int(rf_data.shape[2]),
        "n_patches": int(len(patches)),
        "n_positive_patches": int(np.sum(labels)),
        "n_negative_patches": int(len(labels) - np.sum(labels)),
        "beamformed_z_pixels": int(beamformed.shape[0]),
        "beamformed_x_pixels": int(beamformed.shape[1]),
        "bmode_min": float(np.min(bmode)),
        "bmode_max": float(np.max(bmode)),
        "dynamic_range_db": float(config.dynamic_range_db),
    })

    return {
        "phantom": phantom,
        "rf_data": rf_data,
        "patches": patches,
        "labels": labels,
        "beamformed": beamformed.astype(np.float32),
        "envelope": envelope.astype(np.float32),
        "bmode": bmode.astype(np.float32),
        "x_grid": x_grid,
        "z_grid": z_grid,
        "metrics": metrics,
    }


def save_complete_simulation_report(result: dict, output_dir: str) -> None:
    """Save metrics and core arrays for a complete simulation run."""
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "complete_simulation_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in result["metrics"].items():
            writer.writerow({"metric": key, "value": value})

    np.save(os.path.join(output_dir, "rf_data.npy"), result["rf_data"])
    np.save(os.path.join(output_dir, "patches.npy"), result["patches"])
    np.save(os.path.join(output_dir, "labels.npy"), result["labels"])
    np.save(os.path.join(output_dir, "bmode.npy"), result["bmode"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="default", choices=["default", "pmut_like", "cmut_like"])
    parser.add_argument("--output-dir", default="evaluation_results/complete_simulation")
    parser.add_argument("--n-point", type=int, default=3)
    parser.add_argument("--n-dense", type=int, default=100)
    parser.add_argument("--n-beams", type=int, default=3)
    parser.add_argument("--x-pixels", type=int, default=32)
    parser.add_argument("--z-pixels", type=int, default=64)
    args = parser.parse_args()

    result = run_complete_simulation(
        CompleteSimulationConfig(
            micro_array_template=args.template,
            n_point=args.n_point,
            n_dense=args.n_dense,
            n_beams=args.n_beams,
            x_pixels=args.x_pixels,
            z_pixels=args.z_pixels,
        )
    )
    save_complete_simulation_report(result, args.output_dir)
    print(f"Saved complete simulation report to {args.output_dir}")


if __name__ == "__main__":
    main()
