#!/usr/bin/env python3
"""
Final denoising approach based on accurate noise identification:
1. Z-slice intensity correction (removes horizontal stripes in sagittal view)
2. Upper z region handling
"""

import nibabel as nib
import numpy as np
from scipy import ndimage, signal
import warnings
warnings.filterwarnings('ignore')


def correct_z_slice_intensity(data, detection_threshold=2.0, smoothing_window=5):
    """
    Correct intensity anomalies in specific z-slices that appear as horizontal
    stripes in sagittal view

    Args:
        data: 3D numpy array
        detection_threshold: Std multiplier for anomaly detection
        smoothing_window: Window size for smoothing the mean profile

    Returns:
        Corrected 3D numpy array
    """
    print("\n=== Z-slice Intensity Correction ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    # Calculate mean intensity for each z-slice
    z_means = np.array([data[:, :, z].mean() for z in range(z_dim)])

    # Smooth the profile to get expected trend
    z_means_smoothed = signal.savgol_filter(z_means, window_length=smoothing_window, polyorder=2)

    # Detect anomalies: slices where actual mean deviates significantly from smoothed trend
    deviations = z_means - z_means_smoothed
    std_dev = np.std(deviations)

    anomalous_mask = np.abs(deviations) > detection_threshold * std_dev
    anomalous_z = np.where(anomalous_mask)[0]

    print(f"Detected {len(anomalous_z)} anomalous z-slices: {anomalous_z}")

    # Correct each anomalous slice
    for z in anomalous_z:
        if z_means[z] < 0.001:  # Skip background slices
            continue

        # Calculate correction factor (multiplicative)
        correction_factor = z_means_smoothed[z] / z_means[z]

        # Apply correction
        result[:, :, z] = data[:, :, z] * correction_factor

        print(f"  Z={z}: mean {z_means[z]:.4f} -> {z_means_smoothed[z]:.4f} "
              f"(factor: {correction_factor:.4f})")

    print("Z-slice intensity correction complete")
    return result


def handle_upper_z_noise(data, noise_start_percentile=0.85, method='crop'):
    """
    Handle noise in upper z region

    Args:
        data: 3D numpy array
        noise_start_percentile: Where noise starts (default: 85%)
        method: 'crop' or 'smooth_transition'

    Returns:
        Corrected 3D numpy array
    """
    print(f"\n=== Upper Z Region Handling (method: {method}) ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    noise_start_z = int(z_dim * noise_start_percentile)
    print(f"Noise zone: Z slices {noise_start_z} to {z_dim-1}")

    if method == 'crop':
        # Simple cropping: replace with zeros or last good slice
        print("Method: Replacing noisy region with extrapolation from clean region")

        # Use last clean slice as reference
        reference_z = noise_start_z - 1

        # For each noisy slice, use decreasing intensity based on distance
        for z in range(noise_start_z, z_dim):
            distance = z - noise_start_z + 1
            # Exponential decay
            decay = np.exp(-distance / 10.0)

            # Use reference slice with decay
            result[:, :, z] = data[:, :, reference_z] * decay

    elif method == 'smooth_transition':
        # Create smooth transition from clean to zero
        transition_length = z_dim - noise_start_z

        for z in range(noise_start_z, z_dim):
            # Linear fade to zero
            alpha = 1.0 - (z - noise_start_z) / transition_length
            result[:, :, z] = data[:, :, z] * alpha

    print("Upper z region handling complete")
    return result


def remove_subtle_xy_artifacts(data, sigma=0.5):
    """
    Remove very subtle x-y plane artifacts using gentle smoothing

    Args:
        data: 3D numpy array
        sigma: Gaussian smoothing sigma (very small to preserve details)

    Returns:
        Smoothed 3D numpy array
    """
    print(f"\n=== Subtle X-Y Artifact Removal (sigma={sigma}) ===")

    # Very gentle 2D Gaussian smoothing on each slice
    x_dim, y_dim, z_dim = data.shape
    result = np.zeros_like(data)

    for z in range(z_dim):
        if z % 40 == 0:
            print(f"Processing slice {z}/{z_dim}...")

        # Apply very gentle Gaussian filter
        result[:, :, z] = ndimage.gaussian_filter(data[:, :, z], sigma=sigma)

    print("Subtle artifact removal complete")
    return result


def combined_final_denoising(data):
    """
    Apply final combined denoising pipeline

    Returns:
        Fully denoised 3D numpy array
    """
    print("="*70)
    print("FINAL DENOISING PIPELINE")
    print("="*70)

    # Step 1: Correct z-slice intensity (removes horizontal stripes in sagittal)
    print("\nStep 1/3: Correcting Z-slice intensity anomalies...")
    data_step1 = correct_z_slice_intensity(data, detection_threshold=2.0, smoothing_window=9)

    # Step 2: Handle upper z noise
    print("\nStep 2/3: Handling upper Z region...")
    data_step2 = handle_upper_z_noise(data_step1, noise_start_percentile=0.85, method='crop')

    # Step 3: Very gentle smoothing for any remaining subtle artifacts
    print("\nStep 3/3: Gentle smoothing for subtle artifacts...")
    data_final = remove_subtle_xy_artifacts(data_step2, sigma=0.3)

    print("\n" + "="*70)
    print("DENOISING COMPLETE")
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

    # Apply final denoising
    denoised_data = combined_final_denoising(data)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")
    print(f"Denoised mean: {denoised_data.mean():.3f}")

    # Check edge preservation
    from scipy import ndimage
    z_mid = data.shape[2] // 2
    orig_edges = ndimage.sobel(data[:, :, z_mid])
    denoised_edges = ndimage.sobel(denoised_data[:, :, z_mid])
    edge_ratio = np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges))
    print(f"Edge strength ratio: {edge_ratio:.4f}")

    if edge_ratio > 0.90:
        print("✓ Excellent edge preservation")
    elif edge_ratio > 0.80:
        print("✓ Good edge preservation")
    else:
        print("⚠ Some edge loss")

    # Save result
    print("\nSaving final denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_final_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_final_denoised.nii.gz")

    # Save intermediate results for debugging
    print("\nSaving intermediate results...")

    # After step 1
    step1_only = correct_z_slice_intensity(data)
    step1_img = nib.Nifti1Image(step1_only, img.affine, img.header)
    nib.save(step1_img, 'gneo_sample_sr_189_step1_z_corrected.nii.gz')
    print("Saved: gneo_sample_sr_189_step1_z_corrected.nii.gz")

    # After step 2
    step2_only = handle_upper_z_noise(step1_only)
    step2_img = nib.Nifti1Image(step2_only, img.affine, img.header)
    nib.save(step2_img, 'gneo_sample_sr_189_step2_upper_handled.nii.gz')
    print("Saved: gneo_sample_sr_189_step2_upper_handled.nii.gz")

    print("\n✓ All processing complete!")
