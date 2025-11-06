#!/usr/bin/env python3
"""
Aggressive denoising:
1. Stronger z-slice intensity correction for horizontal stripes
2. More aggressive upper z region removal (near zero)
"""

import nibabel as nib
import numpy as np
from scipy import ndimage, signal
import warnings
warnings.filterwarnings('ignore')


def aggressive_z_slice_correction(data, detection_threshold=1.2, smoothing_window=11):
    """
    More aggressive z-slice intensity correction

    Args:
        data: 3D numpy array
        detection_threshold: Lower threshold = more slices corrected
        smoothing_window: Larger window = smoother trend

    Returns:
        Corrected 3D numpy array
    """
    print("\n=== Aggressive Z-slice Intensity Correction ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    # Calculate mean intensity for each z-slice
    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])

    # Smooth the profile with larger window for better trend
    z_means_smoothed = signal.savgol_filter(z_means, window_length=smoothing_window, polyorder=2)

    # Detect anomalies with lower threshold (more aggressive)
    deviations = z_means - z_means_smoothed
    std_dev = np.std(deviations)

    anomalous_mask = np.abs(deviations) > detection_threshold * std_dev
    anomalous_z = np.where(anomalous_mask)[0]

    print(f"Detected {len(anomalous_z)} anomalous z-slices (threshold={detection_threshold})")
    print(f"Anomalous slices: {anomalous_z[:30]}")

    # Correct each anomalous slice
    for z in anomalous_z:
        if z_means[z] < 0.001:  # Skip background slices
            continue

        # Calculate correction factor
        correction_factor = z_means_smoothed[z] / z_means[z]

        # Apply correction with safety bounds
        correction_factor = np.clip(correction_factor, 0.5, 2.0)

        result[:, :, z] = data[:, :, z] * correction_factor

    print("Aggressive z-slice correction complete")
    return result


def aggressive_upper_z_removal(data, noise_start_percentile=0.75, fade_length=10):
    """
    Aggressively remove upper z noise - make it nearly zero

    Args:
        data: 3D numpy array
        noise_start_percentile: Start removing from this percentile (0.75 = top 25%)
        fade_length: Number of slices for transition zone

    Returns:
        Corrected 3D numpy array
    """
    print(f"\n=== Aggressive Upper Z Removal (start at {noise_start_percentile*100:.0f}%) ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    noise_start_z = int(z_dim * noise_start_percentile)
    fade_end_z = noise_start_z + fade_length

    print(f"Transition zone: Z={noise_start_z} to Z={fade_end_z}")
    print(f"Zero zone: Z={fade_end_z} to Z={z_dim-1}")

    # Transition zone: linear fade to zero
    for z in range(noise_start_z, min(fade_end_z, z_dim)):
        alpha = 1.0 - (z - noise_start_z) / fade_length
        result[:, :, z] = data[:, :, z] * alpha

    # Beyond transition: set to zero
    if fade_end_z < z_dim:
        result[:, :, fade_end_z:] = 0.0

    print(f"Set {z_dim - fade_end_z} slices to zero")
    return result


def remove_horizontal_stripes_frequency(data, axis=2, filter_strength=0.7):
    """
    Remove horizontal stripes in sagittal view using frequency filtering along z-axis

    Args:
        data: 3D numpy array
        axis: Axis along which stripes appear (2 = z-axis)
        filter_strength: Strength of filtering (0-1)

    Returns:
        Filtered 3D numpy array
    """
    print(f"\n=== Horizontal Stripe Removal (FFT along z-axis) ===")
    print(f"Filter strength: {filter_strength}")

    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    # Process each x,y position
    for x in range(x_dim):
        if x % 40 == 0:
            print(f"Processing x={x}/{x_dim}...")

        for y in range(y_dim):
            # Get z-profile at this x,y position
            z_profile = data[x, y, :]

            # Skip if mostly zeros
            if z_profile.std() < 0.01:
                continue

            # Apply FFT along z-axis
            fft = np.fft.fft(z_profile)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)

            # Preserve DC and low frequencies
            center = len(fft_shift) // 2
            preserve_radius = max(3, len(fft_shift) // 20)  # Preserve lowest 5%

            # Detect high-frequency components (stripes)
            threshold = np.percentile(magnitude, 98)

            # Create filter
            filter_mask = np.ones_like(fft_shift, dtype=np.float64)

            for i in range(len(fft_shift)):
                distance = abs(i - center)
                if distance > preserve_radius and magnitude[i] > threshold:
                    # Suppress high-frequency peaks
                    filter_mask[i] = 1.0 - filter_strength

            # Apply filter
            fft_filtered = fft_shift * filter_mask

            # Inverse FFT
            fft_ishift = np.fft.ifftshift(fft_filtered)
            reconstructed = np.fft.ifft(fft_ishift)
            reconstructed = np.real(reconstructed)

            result[x, y, :] = reconstructed

    print("Horizontal stripe removal complete")
    return result


def combined_aggressive_denoising(data):
    """
    Apply aggressive combined denoising pipeline

    Returns:
        Aggressively denoised 3D numpy array
    """
    print("="*70)
    print("AGGRESSIVE DENOISING PIPELINE")
    print("="*70)

    # Step 1: Aggressive z-slice intensity correction
    print("\nStep 1/4: Aggressive z-slice intensity correction...")
    data_step1 = aggressive_z_slice_correction(data, detection_threshold=1.2, smoothing_window=11)

    # Step 2: Remove horizontal stripes using frequency filtering
    print("\nStep 2/4: Removing horizontal stripes (frequency domain)...")
    data_step2 = remove_horizontal_stripes_frequency(data_step1, filter_strength=0.7)

    # Step 3: Aggressive upper z removal
    print("\nStep 3/4: Aggressive upper z removal...")
    data_step3 = aggressive_upper_z_removal(data_step2, noise_start_percentile=0.75, fade_length=10)

    # Step 4: Very gentle final smoothing
    print("\nStep 4/4: Final gentle smoothing...")
    x_dim, y_dim, z_dim = data_step3.shape
    data_final = np.zeros_like(data_step3)

    for z in range(z_dim):
        if z % 40 == 0:
            print(f"  Smoothing slice {z}/{z_dim}...")
        data_final[:, :, z] = ndimage.gaussian_filter(data_step3[:, :, z], sigma=0.3)

    print("\n" + "="*70)
    print("AGGRESSIVE DENOISING COMPLETE")
    print("="*70)

    return data_final


if __name__ == "__main__":
    # Load data
    print("Loading NIfTI file...")
    img = nib.load('gneo_sample_sr_189.nii.gz')
    data = img.get_fdata()

    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Original mean: {data.mean():.3f}")

    # Apply aggressive denoising
    denoised_data = combined_aggressive_denoising(data)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")
    print(f"Denoised mean: {denoised_data.mean():.3f}")

    # Check upper z region
    upper_start = int(data.shape[2] * 0.75)
    upper_region = denoised_data[:, :, upper_start:]
    print(f"\nUpper region (Z>{upper_start}) stats:")
    print(f"  Mean: {upper_region.mean():.6f}")
    print(f"  Max: {upper_region.max():.6f}")
    print(f"  Std: {upper_region.std():.6f}")

    # Check edge preservation
    from scipy import ndimage
    z_mid = data.shape[2] // 2
    orig_edges = ndimage.sobel(data[:, :, z_mid])
    denoised_edges = ndimage.sobel(denoised_data[:, :, z_mid])
    edge_ratio = np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges))
    print(f"\nEdge strength ratio (mid-slice): {edge_ratio:.4f}")

    if edge_ratio > 0.90:
        print("✓ Good edge preservation")
    elif edge_ratio > 0.80:
        print("✓ Acceptable edge preservation")
    else:
        print("⚠ Some edge loss (expected with aggressive denoising)")

    # Save result
    print("\nSaving aggressively denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_aggressive_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_aggressive_denoised.nii.gz")

    # Save intermediate results
    print("\nSaving intermediate results...")

    # After step 1
    step1 = aggressive_z_slice_correction(data)
    step1_img = nib.Nifti1Image(step1, img.affine, img.header)
    nib.save(step1_img, 'gneo_sample_sr_189_aggr_step1.nii.gz')
    print("Saved: gneo_sample_sr_189_aggr_step1.nii.gz")

    # After step 2
    step2 = remove_horizontal_stripes_frequency(step1)
    step2_img = nib.Nifti1Image(step2, img.affine, img.header)
    nib.save(step2_img, 'gneo_sample_sr_189_aggr_step2.nii.gz')
    print("Saved: gneo_sample_sr_189_aggr_step2.nii.gz")

    # After step 3
    step3 = aggressive_upper_z_removal(step2)
    step3_img = nib.Nifti1Image(step3, img.affine, img.header)
    nib.save(step3_img, 'gneo_sample_sr_189_aggr_step3.nii.gz')
    print("Saved: gneo_sample_sr_189_aggr_step3.nii.gz")

    print("\n✓ All processing complete!")
