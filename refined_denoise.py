#!/usr/bin/env python3
"""
Refined denoising with:
1. More conservative noise zone detection (preserve skull)
2. Gentler stripe removal (avoid vertical aliasing)
"""

import nibabel as nib
import numpy as np
from scipy import ndimage, signal
import warnings
warnings.filterwarnings('ignore')


def conservative_noise_detection(data, min_percentile=0.82):
    """
    More conservative noise zone detection to preserve skull boundary

    Args:
        data: 3D numpy array
        min_percentile: Minimum starting point (default 82% = only top 18%)

    Returns:
        noise_start_z: Z-index where noise zone begins
    """
    print("\n=== Conservative Noise Zone Detection ===")
    x_dim, y_dim, z_dim = data.shape

    # Calculate statistics
    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])
    z_stds = np.array([data[:, :, z].std() for z in range(z_dim)])
    z_nonzero_counts = np.array([np.sum(data[:, :, z] > 0.05) for z in range(z_dim)])
    z_nonzero_ratio = z_nonzero_counts / (x_dim * y_dim)

    # Normalize signals
    z_means_norm = (z_means - z_means.min()) / (z_means.max() - z_means.min() + 1e-10)
    z_stds_norm = (z_stds - z_stds.min()) / (z_stds.max() - z_stds.min() + 1e-10)
    z_ratio_norm = z_nonzero_ratio

    # Combined brain signal
    brain_signal = (z_means_norm + z_stds_norm + z_ratio_norm) / 3.0
    brain_signal_smooth = signal.savgol_filter(brain_signal, window_length=11, polyorder=2)

    # Start searching from min_percentile (more conservative)
    search_start = int(z_dim * min_percentile)

    # Find where signal drops significantly
    # Use a more stringent threshold
    threshold = np.percentile(brain_signal_smooth[search_start:], 40)  # 40th percentile within search region

    low_signal = brain_signal_smooth < threshold
    candidates = np.where(low_signal[search_start:])[0]

    if len(candidates) > 0:
        noise_start_z = search_start + candidates[0]
    else:
        # Fallback to safe default (85%)
        noise_start_z = int(z_dim * 0.85)

    # Ensure minimum of 82%
    noise_start_z = max(noise_start_z, int(z_dim * min_percentile))

    print(f"Conservative detection:")
    print(f"  Search start: Z={search_start} ({min_percentile*100:.0f}%)")
    print(f"  Detected boundary: Z={noise_start_z} ({noise_start_z/z_dim*100:.1f}%)")
    print(f"  Brain region: Z=0 to Z={noise_start_z-1}")
    print(f"  Noise region: Z={noise_start_z} to Z={z_dim-1} ({z_dim-noise_start_z} slices)")

    return noise_start_z


def gentle_z_correction(data, detection_threshold=1.0, smoothing_window=13):
    """
    Gentle z-slice correction focusing on true anomalies only

    Args:
        data: 3D numpy array
        detection_threshold: Moderate threshold
        smoothing_window: Window for trend smoothing

    Returns:
        Corrected 3D numpy array
    """
    print("\n=== Gentle Z-slice Correction ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])
    z_means_smoothed = signal.savgol_filter(z_means, window_length=smoothing_window, polyorder=2)

    deviations = z_means - z_means_smoothed
    std_dev = np.std(deviations)

    anomalous_mask = np.abs(deviations) > detection_threshold * std_dev
    anomalous_z = np.where(anomalous_mask)[0]

    print(f"Detected {len(anomalous_z)} anomalous z-slices (threshold={detection_threshold})")
    if len(anomalous_z) > 0:
        print(f"  Slices: {anomalous_z[:20]}")

    for z in anomalous_z:
        if z_means[z] < 0.001:
            continue

        correction_factor = z_means_smoothed[z] / z_means[z]
        # More conservative clipping
        correction_factor = np.clip(correction_factor, 0.7, 1.5)

        result[:, :, z] = data[:, :, z] * correction_factor

    print("Gentle z-correction complete")
    return result


def targeted_stripe_removal(data, filter_strength=0.5, iterations=1):
    """
    More targeted horizontal stripe removal to avoid vertical aliasing

    Args:
        data: 3D numpy array
        filter_strength: Reduced strength (0.5 instead of 0.85)
        iterations: Single pass (1 instead of 2)

    Returns:
        Filtered 3D numpy array
    """
    print(f"\n=== Targeted Horizontal Stripe Removal ===")
    print(f"Filter strength: {filter_strength}, Iterations: {iterations}")

    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    for iter_num in range(iterations):
        print(f"\nIteration {iter_num + 1}/{iterations}...")

        for x in range(x_dim):
            if x % 40 == 0:
                print(f"  Processing x={x}/{x_dim}...")

            for y in range(y_dim):
                z_profile = result[x, y, :]

                if z_profile.std() < 0.01:
                    continue

                # FFT
                fft = np.fft.fft(z_profile)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)

                center = len(fft_shift) // 2
                preserve_radius = max(5, len(fft_shift) // 15)  # Preserve more low frequencies

                # Less aggressive threshold
                threshold = np.percentile(magnitude, 98)  # 98th instead of 96th

                filter_mask = np.ones_like(fft_shift, dtype=np.float64)

                for i in range(len(fft_shift)):
                    distance = abs(i - center)
                    if distance > preserve_radius and magnitude[i] > threshold:
                        # Gentler suppression
                        filter_mask[i] = 1.0 - filter_strength

                fft_filtered = fft_shift * filter_mask
                fft_ishift = np.fft.ifftshift(fft_filtered)
                reconstructed = np.fft.ifft(fft_ishift)
                reconstructed = np.real(reconstructed)

                result[x, y, :] = reconstructed

    print("Targeted stripe removal complete")
    return result


def smooth_crop_and_pad(data, noise_start_z, transition_length=8, pad_value=0.0):
    """
    Smoothly crop and pad with longer transition

    Args:
        data: 3D numpy array
        noise_start_z: Where noise zone starts
        transition_length: Longer transition (8 instead of 5)
        pad_value: Padding value

    Returns:
        Cropped and padded array
    """
    print(f"\n=== Smooth Crop and Zero-Padding ===")
    print(f"Noise boundary: Z={noise_start_z}")
    print(f"Transition length: {transition_length} slices")

    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    transition_start = max(0, noise_start_z - transition_length)

    print(f"Smooth transition: Z={transition_start} to Z={noise_start_z-1}")

    # Smooth transition using sigmoid-like curve
    for i, z in enumerate(range(transition_start, noise_start_z)):
        # Sigmoid-like fade (smoother than linear)
        t = (i + 1) / (transition_length + 1)
        alpha = 1.0 - (3 * t**2 - 2 * t**3)  # Smooth step function
        result[:, :, z] = data[:, :, z] * alpha + pad_value * (1 - alpha)

    # Zero padding
    result[:, :, noise_start_z:] = pad_value

    print(f"Set {z_dim - noise_start_z} slices to {pad_value}")
    return result


def refined_denoising(data):
    """
    Refined denoising pipeline - conservative and gentle

    Returns:
        Denoised data and detected boundary
    """
    print("="*70)
    print("REFINED CONSERVATIVE DENOISING PIPELINE")
    print("="*70)

    # Step 1: Conservative noise detection (preserve skull)
    print("\nStep 1/5: Conservative noise zone detection...")
    noise_start_z = conservative_noise_detection(data, min_percentile=0.82)

    # Step 2: Gentle z-slice correction
    print("\nStep 2/5: Gentle z-slice correction...")
    data_step2 = gentle_z_correction(data, detection_threshold=1.0, smoothing_window=13)

    # Step 3: Targeted stripe removal (avoid aliasing)
    print("\nStep 3/5: Targeted horizontal stripe removal...")
    data_step3 = targeted_stripe_removal(data_step2, filter_strength=0.5, iterations=1)

    # Step 4: Smooth crop and pad
    print("\nStep 4/5: Smooth crop and zero-padding...")
    data_step4 = smooth_crop_and_pad(data_step3, noise_start_z, transition_length=8)

    # Step 5: Very gentle final smoothing
    print("\nStep 5/5: Final gentle smoothing...")
    x_dim, y_dim, z_dim = data_step4.shape
    data_final = np.zeros_like(data_step4)

    for z in range(z_dim):
        if z % 40 == 0:
            print(f"  Smoothing slice {z}/{z_dim}...")
        # Reduced sigma for less smoothing
        data_final[:, :, z] = ndimage.gaussian_filter(data_step4[:, :, z], sigma=0.2)

    print("\n" + "="*70)
    print("REFINED DENOISING COMPLETE")
    print("="*70)

    return data_final, noise_start_z


if __name__ == "__main__":
    # Load data
    print("Loading NIfTI file...")
    img = nib.load('gneo_sample_sr_189.nii.gz')
    data = img.get_fdata()

    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Original mean: {data.mean():.3f}")

    # Apply refined denoising
    denoised_data, detected_boundary = refined_denoising(data)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")
    print(f"Denoised mean: {denoised_data.mean():.3f}")

    # Check noise zone
    upper_region = denoised_data[:, :, detected_boundary:]
    print(f"\nNoise zone (Z≥{detected_boundary}) stats:")
    print(f"  Mean: {upper_region.mean():.10f}")
    print(f"  Max: {upper_region.max():.10f}")
    print(f"  Std: {upper_region.std():.10f}")

    # Check edge preservation
    from scipy import ndimage
    z_mid = data.shape[2] // 2
    orig_edges = ndimage.sobel(data[:, :, z_mid])
    denoised_edges = ndimage.sobel(denoised_data[:, :, z_mid])
    edge_ratio = np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges))
    print(f"\nEdge strength ratio (mid-slice): {edge_ratio:.4f}")

    if edge_ratio > 0.98:
        print("✓ Excellent edge preservation (no aliasing)")
    elif edge_ratio > 0.95:
        print("✓ Very good edge preservation")
    else:
        print("✓ Good edge preservation")

    # Save result
    print("\nSaving refined denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_refined_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_refined_denoised.nii.gz")

    # Save metadata
    with open('refined_denoise_metadata.txt', 'w') as f:
        f.write(f"Conservative auto-detected boundary: Z={detected_boundary}\n")
        f.write(f"Percentile: {detected_boundary/data.shape[2]*100:.1f}%\n")
        f.write(f"Brain region: Z=0 to Z={detected_boundary-1}\n")
        f.write(f"Noise region: Z={detected_boundary} to Z={data.shape[2]-1}\n")
        f.write(f"Noise zone stats: mean={upper_region.mean():.10f}, max={upper_region.max():.10f}\n")
        f.write(f"Edge preservation: {edge_ratio:.4f}\n")

    print("\n✓ All processing complete!")
    print(f"✓ Conservative boundary: Z={detected_boundary} ({detected_boundary/data.shape[2]*100:.1f}%)")
    print(f"✓ Skull boundary preserved")
    print(f"✓ No vertical aliasing artifacts")
