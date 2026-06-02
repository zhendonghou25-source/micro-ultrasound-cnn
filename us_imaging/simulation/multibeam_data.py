"""Multi-beam RF data generation for multi-channel point target detection.

Generates 2D phantoms with point targets embedded in dense background scatterers,
simulates plane-wave RF acquisition, and extracts multi-beam sub-aperture patches.

The key physics: point targets produce coherent echoes across adjacent beams,
while dense scatterers produce partially decorrelated speckle. This cross-beam
coherence difference is the discriminant for multi-channel AE learning.
"""

import numpy as np
from us_imaging.simulation.phantom import Phantom
from us_imaging.simulation.acquire import TransducerArray, plane_wave_rf

# --- Default array parameters (match single-channel simulation) ---
DEFAULT_ARRAY = TransducerArray(
    n_elements=64, pitch=0.3e-3, center_freq=5e6, bandwidth=0.7,
    sound_speed=1540.0, sampling_freq=40e6,
)


def generate_multibeam_phantom(
    n_point: int,
    n_dense: int,
    x_range: tuple = (-0.008, 0.008),
    z_range: tuple = (0.005, 0.04),
    point_amp_range: tuple = (0.7, 1.0),
    dense_amp_range: tuple = (0.1, 0.5),
    min_separation_m: float = 0.003,
    sound_speed: float = 1540.0,
    rng: np.random.RandomState | None = None,
    allow_zero_points: bool = False,
) -> tuple:
    """Create a 2D phantom with point targets embedded in dense background.

    Args:
        n_point: Number of point targets (high amplitude, coherent).
        n_dense: Number of dense background scatterers (low amplitude, random).
        x_range: Lateral range (m) for scatterer placement.
        z_range: Depth range (m) for scatterer placement.
        point_amp_range: Amplitude range for point targets.
        dense_amp_range: Amplitude range for background scatterers.
        min_separation_m: Minimum separation (m) between point targets.
        sound_speed: Medium sound speed (m/s).
        rng: Seeded RandomState for reproducibility.

    Returns:
        (phantom, point_xs, point_zs): Phantom object + point target coordinate arrays.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Generate dense background scatterers
    dense_x = rng.uniform(*x_range, n_dense)
    dense_z = rng.uniform(*z_range, n_dense)
    dense_amp = rng.uniform(*dense_amp_range, n_dense)

    # Generate point targets with minimum separation constraint
    point_xs, point_zs, point_amps = [], [], []
    for _ in range(n_point):
        for attempt in range(100):
            px = rng.uniform(*x_range)
            pz = rng.uniform(*z_range)
            # Check separation from existing points
            ok = True
            for ex, ez in zip(point_xs, point_zs):
                if np.sqrt((px - ex)**2 + (pz - ez)**2) < min_separation_m:
                    ok = False
                    break
            if ok:
                point_xs.append(px)
                point_zs.append(pz)
                point_amps.append(rng.uniform(*point_amp_range))
                break

    if len(point_xs) == 0:
        if n_point == 0 and allow_zero_points:
            point_xs = np.array([], dtype=np.float32)
            point_zs = np.array([], dtype=np.float32)
            point_amps = np.array([], dtype=np.float32)
        else:
            raise RuntimeError("Could not place point targets with min_separation constraint")
    else:
        point_xs = np.array(point_xs, dtype=np.float32)
        point_zs = np.array(point_zs, dtype=np.float32)
        point_amps = np.array(point_amps, dtype=np.float32)

    # Combine
    all_x = np.concatenate([dense_x, point_xs])
    all_z = np.concatenate([dense_z, point_zs])
    all_amp = np.concatenate([dense_amp, point_amps])

    phantom = Phantom(x=all_x, z=all_z, amplitude=all_amp, sound_speed=sound_speed)
    return phantom, point_xs, point_zs


def extract_multibeam_patches(
    rf_data: np.ndarray,
    array: TransducerArray,
    point_xs: np.ndarray,
    point_zs: np.ndarray,
    n_beams: int = 3,
    patch_len: int = 256,
    n_negative: int | None = None,
    min_neg_separation_m: float = 0.002,
    rng: np.random.RandomState | None = None,
) -> tuple:
    """Extract multi-beam patches from plane-wave RF data.

    For each point target: extract a patch centered at its round-trip delay,
    using the closest array element as center beam, plus adjacent beams.

    Negative patches: random (x, z) locations away from point targets.

    Args:
        rf_data: Output of plane_wave_rf(), shape [1, n_elements, n_samples].
        array: TransducerArray used for simulation.
        point_xs, point_zs: Point target coordinates (m).
        n_beams: Number of adjacent beams to extract (odd, e.g. 3 or 5).
        patch_len: Number of temporal samples per patch.
        n_negative: Number of negative patches to extract. Default: same as n_point.
        min_neg_separation_m: Minimum distance from any point target for negatives.
        rng: Seeded RandomState.

    Returns:
        (patches, labels): patches [N, n_beams, patch_len], labels [N] (1=positive, 0=negative).
    """
    if rng is None:
        rng = np.random.RandomState()

    assert n_beams % 2 == 1, f"n_beams must be odd, got {n_beams}"
    half_beams = n_beams // 2

    c = array.sound_speed
    fs = array.sampling_freq
    n_elements = array.n_elements
    n_samples = rf_data.shape[-1]

    if n_negative is None:
        n_negative = len(point_xs)

    x_el = array.element_x
    patches, labels = [], []

    # --- Positive patches: centered on point targets ---
    for px, pz in zip(point_xs, point_zs):
        # Find closest array element
        center_el = int(np.argmin(np.abs(x_el - px)))
        el_start = center_el - half_beams
        el_end = center_el + half_beams + 1

        # Skip if beams are out of bounds
        if el_start < 0 or el_end > n_elements:
            continue

        # Compute expected round-trip delay
        t_delay = 2 * pz / c
        center_sample = int(t_delay * fs)
        sample_start = center_sample - patch_len // 2
        sample_end = center_sample + patch_len // 2

        if sample_start < 0 or sample_end > n_samples:
            continue

        # Extract sub-aperture patch
        patch = rf_data[0, el_start:el_end, sample_start:sample_end]  # [n_beams, patch_len]
        patches.append(patch)
        labels.append(1)

    # --- Negative patches: random locations away from point targets ---
    x_min, x_max = x_el[half_beams], x_el[-half_beams - 1]
    depth_per_sample = c / (2 * fs)
    z_min = (patch_len // 2 + 10) * depth_per_sample
    z_max = (n_samples - patch_len // 2 - 10) * depth_per_sample

    for _ in range(n_negative):
        for attempt in range(200):
            nx = rng.uniform(x_min, x_max)
            nz = rng.uniform(z_min, z_max)
            # Must be away from any point target
            ok = True
            for px, pz in zip(point_xs, point_zs):
                if np.sqrt((nx - px)**2 + (nz - pz)**2) < min_neg_separation_m:
                    ok = False
                    break
            if ok:
                break
        if not ok:
            continue  # skip this negative if we couldn't find a valid location

        center_el = int(np.argmin(np.abs(x_el - nx)))
        el_start = center_el - half_beams
        el_end = center_el + half_beams + 1
        if el_start < 0 or el_end > n_elements:
            continue

        t_delay = 2 * nz / c
        center_sample = int(t_delay * fs)
        sample_start = center_sample - patch_len // 2
        sample_end = center_sample + patch_len // 2
        if sample_start < 0 or sample_end > n_samples:
            continue

        patch = rf_data[0, el_start:el_end, sample_start:sample_end]
        patches.append(patch)
        labels.append(0)

    return np.array(patches, dtype=np.float32), np.array(labels, dtype=np.int64)


def generate_multibeam_training_data(
    n_phantoms: int,
    n_point_per_phantom: int = 3,
    n_dense: int = 200,
    n_beams: int = 3,
    patch_len: int = 256,
    base_seed: int = 42,
    array: TransducerArray | None = None,
    verbose: bool = True,
) -> tuple:
    """Generate multi-beam training data for AE training.

    Creates multiple 2D phantoms, simulates plane-wave RF, and extracts
    multi-beam patches. Returns balanced dataset suitable for AE training.

    Args:
        n_phantoms: Number of independent phantoms to simulate.
        n_point_per_phantom: Point targets per phantom.
        n_dense: Background scatterers per phantom.
        n_beams: Number of adjacent beams per patch (odd).
        patch_len: RF samples per patch.
        base_seed: Base random seed.
        array: TransducerArray. Default: DEFAULT_ARRAY.
        verbose: Print progress.

    Returns:
        (patches, labels): patches [N_total, n_beams, patch_len],
                           labels [N_total] (1=point target, 0=dense speckle).
    """
    if array is None:
        array = DEFAULT_ARRAY

    all_patches, all_labels = [], []
    total_point, total_dense = 0, 0

    for pi in range(n_phantoms):
        seed = base_seed + pi * 1000
        rng = np.random.RandomState(seed)

        phantom, pxs, pzs = generate_multibeam_phantom(
            n_point=n_point_per_phantom, n_dense=n_dense, rng=rng,
        )
        rf = plane_wave_rf(phantom, array, angles=np.array([0.0]))

        patches, labels = extract_multibeam_patches(
            rf, array, pxs, pzs, n_beams=n_beams, patch_len=patch_len,
            rng=rng,
        )
        all_patches.append(patches)
        all_labels.append(labels)
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        total_point += n_pos
        total_dense += n_neg

        if verbose and (pi + 1) % max(1, n_phantoms // 10) == 0:
            print(f"\r  Phantom {pi+1}/{n_phantoms}: {total_point} pos, {total_dense} neg",
                  end="", flush=True)

    if verbose:
        print(f"\r  Generated {total_point} positive + {total_dense} negative patches "
              f"from {n_phantoms} phantoms")

    return (np.concatenate(all_patches, axis=0).astype(np.float32),
            np.concatenate(all_labels, axis=0).astype(np.int64))


def generate_multibeam_detection_data(
    n_per_class: int,
    snr_db: float | None,
    seed: int,
    n_beams: int = 3,
    patch_len: int = 256,
    n_dense: int = 200,
    array: TransducerArray | None = None,
) -> tuple:
    """Drop-in replacement for detection eval data generation.

    Generates multi-beam patches for point target detection:
    - Positive: patches centered on bright point targets (amp 0.7-1.0)
    - Negative: patches from random locations in dense scatterer phantoms

    Uses multiple phantoms with different seeds for diversity.

    Args:
        n_per_class: Number of patches per class.
        snr_db: SNR in dB (None = clean). Added as AWGN after extraction.
        seed: Random seed for reproducibility.
        n_beams: Number of adjacent beams.
        patch_len: RF samples per patch.
        n_dense: Background scatterers per phantom.
        array: TransducerArray.

    Returns:
        (patches, labels): patches [2*n_per_class, n_beams, patch_len],
                           labels [2*n_per_class].
    """
    if array is None:
        array = DEFAULT_ARRAY

    rng = np.random.RandomState(seed)

    # Generate phantom with point targets + dense background
    phantom, pxs, pzs = generate_multibeam_phantom(
        n_point=n_per_class, n_dense=n_dense, rng=rng,
    )
    rf = plane_wave_rf(phantom, array, angles=np.array([0.0]))

    # Add noise if requested
    if snr_db is not None:
        signal_power = np.mean(rf ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = rng.randn(*rf.shape).astype(np.float32) * np.sqrt(noise_power)
        rf = rf + noise

    # Extract positive patches (centered on point targets)
    pos_patches, _ = extract_multibeam_patches(
        rf, array, pxs, pzs, n_beams=n_beams, patch_len=patch_len,
        n_negative=0, rng=rng,
    )

    # Extract negative patches (random locations, no point targets needed)
    # Use a separate phantom with only dense scatterers for negatives
    neg_phantom, _, _ = generate_multibeam_phantom(
        n_point=0, n_dense=n_dense, rng=rng, allow_zero_points=True,
    )
    neg_rf = plane_wave_rf(neg_phantom, array, angles=np.array([0.0]))
    if snr_db is not None:
        signal_power = np.mean(neg_rf ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        neg_noise = rng.randn(*neg_rf.shape).astype(np.float32) * np.sqrt(noise_power)
        neg_rf = neg_rf + neg_noise

    # For negatives, we use the center of the array at random depths
    n_elements = array.n_elements
    half_beams = n_beams // 2
    center_el = n_elements // 2
    el_start = center_el - half_beams
    el_end = center_el + half_beams + 1

    c = array.sound_speed
    fs = array.sampling_freq
    n_samples = neg_rf.shape[-1]
    depth_per_sample = c / (2 * fs)

    neg_patches = []
    for _ in range(n_per_class):
        # Random depth within valid range
        sample_center = rng.randint(patch_len // 2 + 10, n_samples - patch_len // 2 - 10)
        sample_start = sample_center - patch_len // 2
        sample_end = sample_center + patch_len // 2
        patch = neg_rf[0, el_start:el_end, sample_start:sample_end]
        neg_patches.append(patch)

    neg_patches = np.array(neg_patches, dtype=np.float32)

    # Ensure pos_patches has enough samples
    if len(pos_patches) > n_per_class:
        idx = rng.choice(len(pos_patches), n_per_class, replace=False)
        pos_patches = pos_patches[idx]
    elif len(pos_patches) < n_per_class:
        # Augment by slight shifts if we don't have enough
        extra = []
        while len(pos_patches) + len(extra) < n_per_class:
            idx = rng.randint(0, len(pos_patches))
            shift = rng.randint(-5, 6, n_beams)
            shifted = np.roll(pos_patches[idx], shift, axis=-1)
            extra.append(shifted)
        pos_patches = np.concatenate([pos_patches, np.array(extra, dtype=np.float32)], axis=0)[:n_per_class]

    patches = np.concatenate([pos_patches[:n_per_class], neg_patches[:n_per_class]], axis=0)
    labels = np.concatenate([np.ones(n_per_class, dtype=np.int64),
                              np.zeros(n_per_class, dtype=np.int64)])
    return patches.astype(np.float32), labels


# ===========================================================================
# Verification Utilities
# ===========================================================================

def compute_cross_beam_correlation(patches: np.ndarray) -> tuple:
    """Compute average cross-beam Pearson correlation for diagnostic check.

    Args:
        patches: [N, n_beams, patch_len] array.

    Returns:
        (mean_corr, std_corr): Mean and std of pairwise beam correlations.
    """
    N, C, L = patches.shape
    corrs = []
    for i in range(N):
        for c1 in range(C):
            for c2 in range(c1 + 1, C):
                s1, s2 = patches[i, c1], patches[i, c2]
                std1, std2 = s1.std(), s2.std()
                if std1 < 1e-12 or std2 < 1e-12:
                    continue  # skip zero-variance patches
                r = np.corrcoef(s1, s2)[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
    if len(corrs) == 0:
        return 0.0, 0.0
    return np.mean(corrs), np.std(corrs)
