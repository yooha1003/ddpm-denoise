#!/usr/bin/env python3
"""
Improved noise removal methods
"""

import nibabel as nib
import numpy as np
from scipy import ndimage, interpolate
import warnings
warnings.filterwarnings('ignore')


def remove_z_noise_interpolation(data, noise_threshold_z=0.8, method='linear'):
    """
    Remove z-direction noise using interpolation from clean slices

    This is more aggressive - it replaces noisy slices with interpolated values
    from clean slices below

    Args:
        data: 3D numpy array
        noise_threshold_z: Fraction of z-axis where noise starts
        method: Interpolation method ('linear' or 'cubic')

    Returns:
        Denoised 3D numpy array
    """
    print("\n=== Z-direction Interpolation (Improved) ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    noise_start_z = int(z_dim * noise_threshold_z)
    print(f"Noise zone: Z slices {noise_start_z} to {z_dim-1}")
    print(f"Method: Interpolating from clean slices (with bounds checking)")

    # Get global min/max for clipping
    data_min, data_max = data.min(), data.max()

    # For each x,y position, interpolate z values
    # Use clean slices (0 to noise_start_z-1) to extrapolate/interpolate upper slices

    clean_z_indices = np.arange(max(0, noise_start_z-50), noise_start_z)
    noisy_z_indices = np.arange(noise_start_z, z_dim)

    # Process in chunks to save memory
    print("Interpolating...")
    for x in range(x_dim):
        if x % 20 == 0:
            print(f"  Processing x={x}/{x_dim}...")

        for y in range(y_dim):
            # Get clean z-profile (use more recent slices for better extrapolation)
            clean_profile = data[x, y, clean_z_indices]

            # Skip if this is background (low variance)
            if clean_profile.std() < 0.01:
                continue

            # Use linear extrapolation (safer than cubic)
            # Fit a line to the last few points and extrapolate
            if len(clean_z_indices) >= 2:
                # Use last 10 points for trend
                n_points = min(10, len(clean_z_indices))
                fit_z = clean_z_indices[-n_points:]
                fit_values = clean_profile[-n_points:]

                # Linear fit
                coeffs = np.polyfit(fit_z, fit_values, 1)

                # Extrapolate
                new_values = np.polyval(coeffs, noisy_z_indices)

                # Clip to reasonable range (original data range)
                new_values = np.clip(new_values, data_min, data_max)

                result[x, y, noise_start_z:] = new_values

    print("Z-direction interpolation complete")
    return result


def remove_xy_stripes_fft(data, percentile_threshold=99.9, filter_strength=0.5):
    """
    Remove x-y plane stripe artifacts using FFT filtering

    This detects periodic patterns in frequency domain and removes them

    Args:
        data: 3D numpy array
        percentile_threshold: Percentile for detecting high-frequency peaks (stripes)
        filter_strength: Strength of filtering (0=none, 1=complete removal)

    Returns:
        Denoised 3D numpy array
    """
    print("\n=== FFT-based X-Y Stripe Removal (Improved) ===")
    print(f"Threshold: {percentile_threshold}th percentile")
    print(f"Filter strength: {filter_strength}")

    x_dim, y_dim, z_dim = data.shape
    result = np.zeros_like(data)

    # Get global data range for clipping
    data_min, data_max = data.min(), data.max()

    for z in range(z_dim):
        if z % 20 == 0:
            print(f"Processing slice {z}/{z_dim}...")

        slice_data = data[:, :, z]

        # Skip if slice is mostly zeros/background
        if slice_data.std() < 0.001:
            result[:, :, z] = slice_data
            continue

        # Apply 2D FFT
        fft = np.fft.fft2(slice_data)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Detect outlier frequencies (likely stripe artifacts)
        # More conservative threshold
        threshold = np.percentile(magnitude, percentile_threshold)

        # Create a mask, but preserve DC component and low frequencies
        cy, cx = magnitude.shape[0] // 2, magnitude.shape[1] // 2

        # Create frequency distance map
        y_freq, x_freq = np.ogrid[0:magnitude.shape[0], 0:magnitude.shape[1]]
        distance_from_center = np.sqrt((y_freq - cy)**2 + (x_freq - cx)**2)

        # Preserve low frequencies (within 15% of image size from center)
        preserve_radius = min(magnitude.shape) * 0.15

        # Find high magnitude points that are NOT in low-frequency region
        high_freq_mask = (magnitude > threshold) & (distance_from_center > preserve_radius)

        # Create filter: start with all ones
        filter_mask = np.ones_like(fft_shift, dtype=np.float64)

        # For detected stripe frequencies, apply gentle suppression
        if np.sum(high_freq_mask) > 0:
            # Apply gentle suppression based on filter_strength
            filter_mask[high_freq_mask] = 1.0 - filter_strength

        # Apply filter
        fft_filtered = fft_shift * filter_mask

        # Inverse FFT
        fft_ishift = np.fft.ifftshift(fft_filtered)
        reconstructed = np.fft.ifft2(fft_ishift)
        reconstructed = np.real(reconstructed)

        # Clip to original data range to prevent artifacts
        reconstructed = np.clip(reconstructed, data_min, data_max)

        result[:, :, z] = reconstructed

    print("FFT-based stripe removal complete")
    return result


def combined_denoising_improved(data, z_threshold=0.8):
    """
    Apply improved denoising methods

    Args:
        data: 3D numpy array
        z_threshold: Z-axis threshold for noise zone

    Returns:
        Fully denoised 3D numpy array
    """
    print("=== Improved Combined Denoising Pipeline ===")

    # Step 1: Remove x-y stripes FIRST (to avoid blurring the interpolation)
    print("\nStep 1/2: Removing X-Y stripe artifacts with FFT...")
    data_xy_clean = remove_xy_stripes_fft(data, percentile_threshold=99.9, filter_strength=0.5)

    # Step 2: Remove z-direction noise with interpolation
    print("\nStep 2/2: Removing Z-direction noise with interpolation...")
    data_fully_clean = remove_z_noise_interpolation(data_xy_clean, noise_threshold_z=z_threshold)

    print("\n=== Improved Denoising Complete ===")

    return data_fully_clean


if __name__ == "__main__":
    # Load data
    print("Loading NIfTI file...")
    img = nib.load('gneo_sample_sr_189.nii.gz')
    data = img.get_fdata()

    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]")

    # Apply improved denoising
    denoised_data = combined_denoising_improved(data, z_threshold=0.8)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")

    # Check for blur
    from scipy import ndimage
    z_mid = data.shape[2] // 2
    orig_edges = ndimage.sobel(data[:, :, z_mid])
    denoised_edges = ndimage.sobel(denoised_data[:, :, z_mid])
    edge_ratio = np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges))
    print(f"Edge strength ratio: {edge_ratio:.4f}")
    if edge_ratio > 0.95:
        print("✓ Minimal blur (edge strength preserved)")
    elif edge_ratio > 0.85:
        print("⚠ Slight blur detected")
    else:
        print("⚠️ Significant blur detected")

    # Save result
    print("\nSaving improved denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_improved_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_improved_denoised.nii.gz")

    # Save intermediate results
    print("\nSaving intermediate results...")

    # XY-only (FFT)
    xy_only = remove_xy_stripes_fft(data)
    xy_img = nib.Nifti1Image(xy_only, img.affine, img.header)
    nib.save(xy_img, 'gneo_sample_sr_189_fft_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_fft_denoised.nii.gz")

    # Z-only (interpolation)
    z_only = remove_z_noise_interpolation(data)
    z_img = nib.Nifti1Image(z_only, img.affine, img.header)
    nib.save(z_img, 'gneo_sample_sr_189_interp_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_interp_denoised.nii.gz")

    print("\n✓ All processing complete!")
