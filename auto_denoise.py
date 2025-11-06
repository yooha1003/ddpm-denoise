#!/usr/bin/env python3
"""
Automatic denoising with intelligent noise zone detection:
1. Auto-detect noise zone boundary
2. Enhanced horizontal stripe removal
3. Auto-crop and zero-pad
"""

import nibabel as nib
import numpy as np
from scipy import ndimage, signal
from scipy.stats import median_abs_deviation
import warnings
warnings.filterwarnings('ignore')


def detect_noise_zone_boundary(data, method='gradient', percentile=95):
    """
    Automatically detect where the noise zone starts in the upper Z region

    Args:
        data: 3D numpy array
        method: Detection method ('gradient', 'variance', 'combined')
        percentile: Percentile for threshold

    Returns:
        noise_start_z: Z-index where noise zone begins
    """
    print("\n=== Automatic Noise Zone Detection ===")
    x_dim, y_dim, z_dim = data.shape

    # Calculate statistics for each z-slice
    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])
    z_stds = np.array([data[:, :, z].std() for z in range(z_dim)])
    z_maxs = np.array([data[:, :, z].max() for z in range(z_dim)])

    # Calculate non-zero voxel count (brain content indicator)
    z_nonzero_counts = np.array([np.sum(data[:, :, z] > 0.05) for z in range(z_dim)])
    z_nonzero_ratio = z_nonzero_counts / (x_dim * y_dim)

    if method == 'gradient':
        # Look for sharp drops in content
        gradient = np.gradient(z_nonzero_ratio)

        # Find the point where gradient becomes significantly negative
        negative_gradient = gradient < 0
        strong_drop = gradient < np.percentile(gradient[gradient < 0], 20)

        # Start from 50% and look for first strong drop
        candidates = np.where(strong_drop[int(z_dim*0.5):])[0]
        if len(candidates) > 0:
            noise_start_z = int(z_dim*0.5) + candidates[0]
        else:
            noise_start_z = int(z_dim * 0.75)  # Fallback

    elif method == 'variance':
        # Look for where variance drops (less brain structure)
        variance_threshold = np.percentile(z_stds[:int(z_dim*0.7)], 20)
        low_variance_slices = np.where(z_stds < variance_threshold)[0]

        # Find first contiguous region of low variance in upper half
        candidates = low_variance_slices[low_variance_slices > z_dim//2]
        if len(candidates) > 0:
            noise_start_z = candidates[0]
        else:
            noise_start_z = int(z_dim * 0.75)

    elif method == 'combined':
        # Combined approach: use multiple signals

        # Normalize signals
        z_means_norm = (z_means - z_means.min()) / (z_means.max() - z_means.min() + 1e-10)
        z_stds_norm = (z_stds - z_stds.min()) / (z_stds.max() - z_stds.min() + 1e-10)
        z_ratio_norm = z_nonzero_ratio

        # Combined signal (higher = more brain content)
        brain_signal = (z_means_norm + z_stds_norm + z_ratio_norm) / 3.0

        # Smooth the signal
        brain_signal_smooth = signal.savgol_filter(brain_signal, window_length=11, polyorder=2)

        # Find where signal drops below threshold in upper region
        threshold = np.percentile(brain_signal_smooth[:int(z_dim*0.6)], 30)

        low_signal = brain_signal_smooth < threshold
        candidates = np.where(low_signal[int(z_dim*0.5):])[0]

        if len(candidates) > 0:
            noise_start_z = int(z_dim*0.5) + candidates[0]
        else:
            noise_start_z = int(z_dim * 0.75)

    # Ensure noise_start_z is reasonable (at least 60% through)
    noise_start_z = max(noise_start_z, int(z_dim * 0.6))

    print(f"Detection method: {method}")
    print(f"Detected noise zone start: Z={noise_start_z} ({noise_start_z/z_dim*100:.1f}%)")
    print(f"Brain region: Z=0 to Z={noise_start_z-1}")
    print(f"Noise region: Z={noise_start_z} to Z={z_dim-1} ({z_dim-noise_start_z} slices)")

    return noise_start_z


def enhanced_horizontal_stripe_removal(data, filter_strength=0.85, iterations=2):
    """
    Enhanced horizontal stripe removal with multiple iterations

    Args:
        data: 3D numpy array
        filter_strength: Strength of filtering (0-1)
        iterations: Number of filtering passes

    Returns:
        Filtered 3D numpy array
    """
    print(f"\n=== Enhanced Horizontal Stripe Removal ===")
    print(f"Filter strength: {filter_strength}, Iterations: {iterations}")

    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    for iter_num in range(iterations):
        print(f"\nIteration {iter_num + 1}/{iterations}...")

        # Process each x,y position
        for x in range(x_dim):
            if x % 40 == 0:
                print(f"  Processing x={x}/{x_dim}...")

            for y in range(y_dim):
                z_profile = result[x, y, :]

                # Skip if mostly zeros
                if z_profile.std() < 0.01:
                    continue

                # Apply FFT along z-axis
                fft = np.fft.fft(z_profile)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)

                # Preserve DC and low frequencies
                center = len(fft_shift) // 2
                preserve_radius = max(3, len(fft_shift) // 20)

                # More aggressive threshold for stripe detection
                threshold = np.percentile(magnitude, 96)  # Lower percentile = more aggressive

                # Create filter
                filter_mask = np.ones_like(fft_shift, dtype=np.float64)

                for i in range(len(fft_shift)):
                    distance = abs(i - center)
                    if distance > preserve_radius and magnitude[i] > threshold:
                        # Apply stronger suppression
                        filter_mask[i] = 1.0 - filter_strength

                # Apply filter
                fft_filtered = fft_shift * filter_mask

                # Inverse FFT
                fft_ishift = np.fft.ifftshift(fft_filtered)
                reconstructed = np.fft.ifft(fft_ishift)
                reconstructed = np.real(reconstructed)

                result[x, y, :] = reconstructed

    print("Enhanced stripe removal complete")
    return result


def auto_crop_and_pad(data, noise_start_z, pad_value=0.0):
    """
    Automatically crop the noise zone and replace with padding

    Args:
        data: 3D numpy array
        noise_start_z: Where noise zone starts
        pad_value: Value to use for padding (default: 0)

    Returns:
        Cropped and padded 3D numpy array
    """
    print(f"\n=== Auto-Crop and Zero-Padding ===")
    print(f"Cropping from Z={noise_start_z} to end")
    print(f"Padding value: {pad_value}")

    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    # Create smooth transition (5 slices)
    transition_length = min(5, noise_start_z - 1)
    transition_start = noise_start_z - transition_length

    print(f"Smooth transition: Z={transition_start} to Z={noise_start_z-1}")

    # Apply smooth transition
    for i, z in enumerate(range(transition_start, noise_start_z)):
        alpha = 1.0 - (i + 1) / (transition_length + 1)
        result[:, :, z] = data[:, :, z] * alpha + pad_value * (1 - alpha)

    # Set noise zone to pad_value
    result[:, :, noise_start_z:] = pad_value

    print(f"Set {z_dim - noise_start_z} slices to {pad_value}")
    return result


def ultra_aggressive_z_correction(data, detection_threshold=0.8):
    """
    Ultra aggressive z-slice intensity correction

    Args:
        data: 3D numpy array
        detection_threshold: Very low threshold for maximum sensitivity

    Returns:
        Corrected 3D numpy array
    """
    print("\n=== Ultra-Aggressive Z-slice Correction ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])
    z_means_smoothed = signal.savgol_filter(z_means, window_length=15, polyorder=2)

    deviations = z_means - z_means_smoothed
    std_dev = np.std(deviations)

    anomalous_mask = np.abs(deviations) > detection_threshold * std_dev
    anomalous_z = np.where(anomalous_mask)[0]

    print(f"Detected {len(anomalous_z)} anomalous z-slices (threshold={detection_threshold})")

    for z in anomalous_z:
        if z_means[z] < 0.001:
            continue

        correction_factor = z_means_smoothed[z] / z_means[z]
        correction_factor = np.clip(correction_factor, 0.3, 3.0)

        result[:, :, z] = data[:, :, z] * correction_factor

    print("Ultra-aggressive correction complete")
    return result


def combined_auto_denoising(data):
    """
    Apply automatic denoising with intelligent noise detection

    Returns:
        Fully denoised 3D numpy array with auto-detected cropping
    """
    print("="*70)
    print("AUTOMATIC INTELLIGENT DENOISING PIPELINE")
    print("="*70)

    # Step 1: Detect noise zone boundary automatically
    print("\nStep 1/5: Auto-detecting noise zone boundary...")
    noise_start_z = detect_noise_zone_boundary(data, method='combined')

    # Step 2: Ultra-aggressive z-slice correction
    print("\nStep 2/5: Ultra-aggressive z-slice correction...")
    data_step2 = ultra_aggressive_z_correction(data, detection_threshold=0.8)

    # Step 3: Enhanced horizontal stripe removal (multiple iterations)
    print("\nStep 3/5: Enhanced horizontal stripe removal...")
    data_step3 = enhanced_horizontal_stripe_removal(data_step2, filter_strength=0.85, iterations=2)

    # Step 4: Auto-crop and zero-pad
    print("\nStep 4/5: Auto-crop and zero-padding...")
    data_step4 = auto_crop_and_pad(data_step3, noise_start_z, pad_value=0.0)

    # Step 5: Final gentle smoothing
    print("\nStep 5/5: Final gentle smoothing...")
    x_dim, y_dim, z_dim = data_step4.shape
    data_final = np.zeros_like(data_step4)

    for z in range(z_dim):
        if z % 40 == 0:
            print(f"  Smoothing slice {z}/{z_dim}...")
        data_final[:, :, z] = ndimage.gaussian_filter(data_step4[:, :, z], sigma=0.25)

    print("\n" + "="*70)
    print("AUTOMATIC DENOISING COMPLETE")
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

    # Apply automatic denoising
    denoised_data, detected_boundary = combined_auto_denoising(data)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")
    print(f"Denoised mean: {denoised_data.mean():.3f}")

    # Check noise zone
    upper_region = denoised_data[:, :, detected_boundary:]
    print(f"\nNoise zone (Z>{detected_boundary}) stats:")
    print(f"  Mean: {upper_region.mean():.8f}")
    print(f"  Max: {upper_region.max():.8f}")
    print(f"  Std: {upper_region.std():.8f}")

    # Check edge preservation
    from scipy import ndimage
    z_mid = data.shape[2] // 2
    orig_edges = ndimage.sobel(data[:, :, z_mid])
    denoised_edges = ndimage.sobel(denoised_data[:, :, z_mid])
    edge_ratio = np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges))
    print(f"\nEdge strength ratio (mid-slice): {edge_ratio:.4f}")

    # Save result
    print("\nSaving auto-denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_auto_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_auto_denoised.nii.gz")

    # Save metadata about detected boundary
    with open('auto_denoise_metadata.txt', 'w') as f:
        f.write(f"Auto-detected noise boundary: Z={detected_boundary}\n")
        f.write(f"Brain region: Z=0 to Z={detected_boundary-1}\n")
        f.write(f"Noise region: Z={detected_boundary} to Z={data.shape[2]-1}\n")
        f.write(f"Noise zone stats: mean={upper_region.mean():.8f}, max={upper_region.max():.8f}\n")

    print("\n✓ All processing complete!")
    print(f"✓ Noise boundary automatically detected at Z={detected_boundary}")
