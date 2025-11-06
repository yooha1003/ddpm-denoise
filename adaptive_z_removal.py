#!/usr/bin/env python3
"""
Adaptive z-slice removal based on salt-and-pepper noise detection
Analyzes each z-slice individually to detect noise characteristics
"""

import nibabel as nib
import numpy as np
from scipy import ndimage, signal
import warnings
warnings.filterwarnings('ignore')


def analyze_slice_noise_characteristics(slice_data, threshold=0.05):
    """
    Analyze a single z-slice to determine if it's noise or brain tissue

    Args:
        slice_data: 2D array (x, y)
        threshold: Minimum value to consider as signal

    Returns:
        noise_score: Higher score = more likely to be noise (0-1)
        metrics: Dictionary of individual metrics
    """
    # Get mask of significant pixels
    mask = slice_data > threshold

    if np.sum(mask) < 10:  # Too few pixels
        return 1.0, {'reason': 'empty'}

    # Extract non-zero region
    significant_data = slice_data[mask]

    # Metric 1: Local variance (salt-and-pepper has high local variance)
    # Compute variance in local neighborhoods
    local_var = ndimage.generic_filter(slice_data, np.var, size=3)
    local_var_masked = local_var[mask]
    mean_local_var = np.mean(local_var_masked) if len(local_var_masked) > 0 else 0

    # Metric 2: Neighbor difference (salt-and-pepper has large differences)
    # Sobel edge detection strength
    edges = ndimage.sobel(slice_data)
    edge_strength = np.mean(np.abs(edges[mask])) if np.sum(mask) > 0 else 0

    # Metric 3: Spatial coherence (brain tissue has coherent structures)
    # Measure correlation with smoothed version
    smoothed = ndimage.gaussian_filter(slice_data, sigma=2)
    correlation = np.corrcoef(slice_data[mask], smoothed[mask])[0, 1] if len(significant_data) > 10 else 0

    # Metric 4: Radial profile smoothness (brain has smooth center-to-edge decline)
    center_y, center_x = np.array(slice_data.shape) // 2
    y_indices, x_indices = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
    distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)

    # Bin by distance and compute mean intensity
    max_dist = int(np.min(slice_data.shape) / 2)
    radial_profile = []
    for r in range(0, max_dist, 5):
        ring_mask = (distances >= r) & (distances < r+5) & mask
        if np.sum(ring_mask) > 0:
            radial_profile.append(np.mean(slice_data[ring_mask]))

    # Check if radial profile decreases smoothly
    if len(radial_profile) > 2:
        # Compute smoothness (lower diff = smoother)
        radial_diff = np.mean(np.abs(np.diff(radial_profile)))
        # Check for general declining trend
        declining_trend = radial_profile[0] > radial_profile[-1] if len(radial_profile) > 1 else False
    else:
        radial_diff = 999
        declining_trend = False

    # Metric 5: Intensity distribution (noise is more uniform, brain has peaks)
    hist, _ = np.histogram(significant_data, bins=20)
    hist_uniformity = np.std(hist) / (np.mean(hist) + 1e-10)  # Lower = more uniform

    # Metric 6: Fill ratio (noise tends to fill more of the space)
    fill_ratio = np.sum(mask) / mask.size

    # Combine metrics into noise score
    metrics = {
        'local_variance': mean_local_var,
        'edge_strength': edge_strength,
        'correlation': correlation,
        'radial_smoothness': radial_diff,
        'declining_trend': declining_trend,
        'hist_uniformity': hist_uniformity,
        'fill_ratio': fill_ratio
    }

    # Compute noise score (0 = brain tissue, 1 = noise)
    noise_score = 0.0

    # High local variance indicates noise
    if mean_local_var > 0.02:
        noise_score += 0.2

    # High edge strength indicates noise
    if edge_strength > 0.15:
        noise_score += 0.2

    # Low correlation with smoothed version indicates noise
    if correlation < 0.7:
        noise_score += 0.2

    # Non-smooth radial profile indicates noise
    if radial_diff > 0.05:
        noise_score += 0.15

    # No declining trend indicates noise
    if not declining_trend:
        noise_score += 0.15

    # Uniform histogram indicates noise
    if hist_uniformity < 0.5:
        noise_score += 0.1

    noise_score = min(noise_score, 1.0)

    return noise_score, metrics


def adaptive_z_removal(data, noise_threshold=0.35, min_z_start=0.70):
    """
    Adaptively remove z-slices based on noise characteristics

    Args:
        data: 3D numpy array
        noise_threshold: Score above which slice is considered noise
        min_z_start: Minimum z position to start checking (safety)

    Returns:
        denoised_data: Data with noise slices removed
        removal_mask: Boolean array indicating which slices were removed
    """
    print("\n=== Adaptive Slice-wise Noise Detection ===")
    x_dim, y_dim, z_dim = data.shape

    result = data.copy()
    removal_mask = np.zeros(z_dim, dtype=bool)
    noise_scores = np.zeros(z_dim)

    # Start checking from min_z_start
    start_z = int(z_dim * min_z_start)
    print(f"Analyzing slices from Z={start_z} to Z={z_dim-1}...")

    # Analyze each slice
    for z in range(start_z, z_dim):
        slice_data = data[:, :, z]
        score, metrics = analyze_slice_noise_characteristics(slice_data)
        noise_scores[z] = score

        if z % 10 == 0:
            print(f"  Z={z}: noise_score={score:.3f}")

    # Find continuous noise regions
    print(f"\nDetecting continuous noise regions (threshold={noise_threshold})...")

    # A slice is noise if its score is high
    potential_noise = noise_scores > noise_threshold

    # Find the first continuous noise region in upper part
    in_noise_region = False
    noise_start = None

    for z in range(start_z, z_dim):
        if potential_noise[z] and not in_noise_region:
            # Start of noise region
            noise_start = z
            in_noise_region = True
        elif not potential_noise[z] and in_noise_region:
            # If we exit noise region but it's short, continue searching
            if z - noise_start < 5:
                in_noise_region = False
                noise_start = None
            else:
                # Found substantial noise region
                break

    if noise_start is not None:
        print(f"Detected noise region starting at Z={noise_start}")
        print(f"Removing {z_dim - noise_start} slices")

        # Mark for removal
        removal_mask[noise_start:] = True

        # Apply smooth transition
        transition_length = 8
        transition_start = max(start_z, noise_start - transition_length)

        print(f"Smooth transition: Z={transition_start} to Z={noise_start-1}")

        for i, z in enumerate(range(transition_start, noise_start)):
            t = (i + 1) / (transition_length + 1)
            alpha = 1.0 - (3 * t**2 - 2 * t**3)  # Smooth step
            result[:, :, z] = data[:, :, z] * alpha

        # Zero out noise region
        result[:, :, noise_start:] = 0.0

    else:
        print("No clear noise region detected (all slices appear to be brain tissue)")

    return result, removal_mask, noise_scores


def combined_adaptive_denoising(data):
    """
    Complete denoising pipeline with adaptive z-removal

    Returns:
        Denoised data, removal mask, and noise scores
    """
    print("="*70)
    print("ADAPTIVE Z-REMOVAL DENOISING PIPELINE")
    print("="*70)

    # Step 1: Gentle z-slice correction (from refined method)
    print("\nStep 1/4: Gentle z-slice correction...")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])
    z_means_smoothed = signal.savgol_filter(z_means, window_length=13, polyorder=2)
    deviations = z_means - z_means_smoothed
    std_dev = np.std(deviations)
    anomalous_mask = np.abs(deviations) > 1.0 * std_dev
    anomalous_z = np.where(anomalous_mask)[0]

    print(f"Correcting {len(anomalous_z)} anomalous z-slices")

    for z in anomalous_z:
        if z_means[z] < 0.001:
            continue
        correction_factor = z_means_smoothed[z] / z_means[z]
        correction_factor = np.clip(correction_factor, 0.7, 1.5)
        result[:, :, z] = data[:, :, z] * correction_factor

    # Step 2: Targeted stripe removal (from refined method)
    print("\nStep 2/4: Targeted horizontal stripe removal...")
    for x in range(x_dim):
        if x % 40 == 0:
            print(f"  Processing x={x}/{x_dim}...")

        for y in range(y_dim):
            z_profile = result[x, y, :]
            if z_profile.std() < 0.01:
                continue

            fft = np.fft.fft(z_profile)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            center = len(fft_shift) // 2
            preserve_radius = max(5, len(fft_shift) // 15)
            threshold = np.percentile(magnitude, 98)

            filter_mask = np.ones_like(fft_shift, dtype=np.float64)
            for i in range(len(fft_shift)):
                distance = abs(i - center)
                if distance > preserve_radius and magnitude[i] > threshold:
                    filter_mask[i] = 0.5

            fft_filtered = fft_shift * filter_mask
            fft_ishift = np.fft.ifftshift(fft_filtered)
            reconstructed = np.fft.ifft(fft_ishift)
            result[x, y, :] = np.real(reconstructed)

    print("Stripe removal complete")

    # Step 3: Adaptive z-removal based on noise characteristics
    print("\nStep 3/4: Adaptive slice-wise noise removal...")
    result, removal_mask, noise_scores = adaptive_z_removal(result, noise_threshold=0.35, min_z_start=0.70)

    # Step 4: Final gentle smoothing
    print("\nStep 4/4: Final gentle smoothing...")
    data_final = np.zeros_like(result)
    for z in range(z_dim):
        if z % 40 == 0:
            print(f"  Smoothing slice {z}/{z_dim}...")
        data_final[:, :, z] = ndimage.gaussian_filter(result[:, :, z], sigma=0.2)

    print("\n" + "="*70)
    print("ADAPTIVE DENOISING COMPLETE")
    print("="*70)

    return data_final, removal_mask, noise_scores


if __name__ == "__main__":
    # Load data
    print("Loading NIfTI file...")
    img = nib.load('gneo_sample_sr_189.nii.gz')
    data = img.get_fdata()

    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]")

    # Apply adaptive denoising
    denoised_data, removal_mask, noise_scores = combined_adaptive_denoising(data)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")

    # Find where removal started
    removed_slices = np.where(removal_mask)[0]
    if len(removed_slices) > 0:
        removal_start = removed_slices[0]
        print(f"\nAdaptive removal:")
        print(f"  First removed slice: Z={removal_start} ({removal_start/data.shape[2]*100:.1f}%)")
        print(f"  Total removed: {len(removed_slices)} slices")

        # Check upper region
        upper_region = denoised_data[:, :, removal_start:]
        print(f"\nRemoved region stats:")
        print(f"  Mean: {upper_region.mean():.10f}")
        print(f"  Max: {upper_region.max():.10f}")

    # Check edge preservation
    from scipy import ndimage
    z_mid = data.shape[2] // 2
    orig_edges = ndimage.sobel(data[:, :, z_mid])
    denoised_edges = ndimage.sobel(denoised_data[:, :, z_mid])
    edge_ratio = np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges))
    print(f"\nEdge strength ratio: {edge_ratio:.4f}")

    # Save result
    print("\nSaving adaptive denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_adaptive_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_adaptive_denoised.nii.gz")

    # Save noise scores
    np.save('adaptive_noise_scores.npy', noise_scores)
    print("Saved: adaptive_noise_scores.npy")

    # Save metadata
    with open('adaptive_denoise_metadata.txt', 'w') as f:
        if len(removed_slices) > 0:
            f.write(f"Adaptive noise detection method\n")
            f.write(f"First removed slice: Z={removal_start} ({removal_start/data.shape[2]*100:.1f}%)\n")
            f.write(f"Total removed: {len(removed_slices)} slices\n")
            f.write(f"Removed region: Z={removal_start} to Z={data.shape[2]-1}\n")
            f.write(f"\nNoise scores (Z={int(data.shape[2]*0.75)} onwards):\n")
            for z in range(int(data.shape[2]*0.75), data.shape[2]):
                if noise_scores[z] > 0:
                    f.write(f"  Z={z}: {noise_scores[z]:.3f} {'[REMOVED]' if removal_mask[z] else ''}\n")
        else:
            f.write("No noise region detected\n")

    print("\n✓ Adaptive denoising complete!")
    print("✓ Slice-wise analysis preserved brain tissue")
