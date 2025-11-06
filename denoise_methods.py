#!/usr/bin/env python3
"""
Noise removal methods for NIfTI medical imaging data
"""

import nibabel as nib
import numpy as np
import pywt
from scipy import ndimage
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')

def remove_z_noise_inpainting(data, noise_threshold_z=0.8, transition_width=10):
    """
    Remove z-direction noise in upper slices using 3D inpainting

    Args:
        data: 3D numpy array
        noise_threshold_z: Fraction of z-axis where noise starts (default: 0.8 = top 20%)
        transition_width: Number of slices for smooth transition

    Returns:
        Denoised 3D numpy array
    """
    print("\n=== Z-direction 3D Inpainting ===")
    x_dim, y_dim, z_dim = data.shape
    result = data.copy()

    # Identify noise zone
    noise_start_z = int(z_dim * noise_threshold_z)
    print(f"Noise zone: Z slices {noise_start_z} to {z_dim-1}")

    # Strategy: Use information from clean slices to reconstruct noisy slices
    # We'll use a weighted average and interpolation approach

    # Get reference slices (clean region just before noise zone)
    ref_start = max(0, noise_start_z - 20)
    ref_end = noise_start_z
    ref_slices = data[:, :, ref_start:ref_end]

    print(f"Using reference slices Z={ref_start} to {ref_end-1}")

    # For each noisy slice, interpolate based on clean data
    for z in range(noise_start_z, z_dim):
        # Distance from noise boundary
        distance = z - noise_start_z

        # Create inpainted slice based on reference statistics
        # Method 1: Use the pattern from reference slices with decay

        # Find the corresponding reference slice (mirror or last clean slice)
        ref_idx = min(ref_end - 1 - distance, ref_end - 1)
        ref_idx = max(ref_start, ref_idx)

        # Get reference slice
        if ref_idx >= ref_start and ref_idx < ref_end:
            ref_slice = data[:, :, ref_idx]
        else:
            ref_slice = data[:, :, ref_end - 1]

        # Blend with local structure using gradual decay
        decay_factor = np.exp(-distance / 5.0)  # Exponential decay

        # Apply Gaussian smoothing to reduce sharp transitions
        smoothed_ref = ndimage.gaussian_filter(ref_slice, sigma=2.0)

        # Weighted combination
        result[:, :, z] = decay_factor * smoothed_ref + (1 - decay_factor) * smoothed_ref * 0.5

    # Apply smooth transition at the boundary
    for i in range(transition_width):
        z = noise_start_z + i
        if z < z_dim:
            weight = i / transition_width
            result[:, :, z] = (1 - weight) * data[:, :, z] + weight * result[:, :, z]

    print(f"Inpainting complete. Processed {z_dim - noise_start_z} slices")

    return result


def remove_xy_stripes_wavelet(data, wavelet='db4', level=3, threshold_scale=1.0):
    """
    Remove x-y plane stripe artifacts using wavelet denoising

    Args:
        data: 3D numpy array
        wavelet: Wavelet type (default: 'db4' - Daubechies 4)
        level: Decomposition level (default: 3)
        threshold_scale: Threshold multiplier for noise removal (default: 1.0)

    Returns:
        Denoised 3D numpy array
    """
    print("\n=== Wavelet Denoising for X-Y Stripes ===")
    print(f"Wavelet: {wavelet}, Level: {level}, Threshold scale: {threshold_scale}")

    x_dim, y_dim, z_dim = data.shape
    result = np.zeros_like(data)

    # Process each z-slice independently
    for z in range(z_dim):
        if z % 20 == 0:
            print(f"Processing slice {z}/{z_dim}...")

        slice_data = data[:, :, z]

        # Perform 2D wavelet decomposition
        coeffs = pywt.wavedec2(slice_data, wavelet, level=level)

        # coeffs structure: [cAn, (cHn, cVn, cDn), ..., (cH1, cV1, cD1)]
        # cA: approximation, cH: horizontal, cV: vertical, cD: diagonal

        # Estimate noise level from finest detail coefficients
        detail_coeffs = coeffs[-1]  # Finest level (cH1, cV1, cD1)
        sigma = np.median(np.abs(detail_coeffs[0])) / 0.6745  # MAD estimator

        # Apply soft thresholding to detail coefficients
        new_coeffs = [coeffs[0]]  # Keep approximation

        for i in range(1, len(coeffs)):
            # Get detail coefficients at this level
            cH, cV, cD = coeffs[i]

            # Calculate threshold using BayesShrink or VisuShrink
            threshold = threshold_scale * sigma * np.sqrt(2 * np.log(slice_data.size))

            # Apply soft thresholding
            cH_thresh = pywt.threshold(cH, threshold, mode='soft')
            cV_thresh = pywt.threshold(cV, threshold, mode='soft')
            cD_thresh = pywt.threshold(cD, threshold, mode='soft')

            # For stripe removal, we can be more aggressive on horizontal/vertical
            # components which typically contain stripe artifacts
            stripe_threshold = threshold * 1.2
            cH_thresh = pywt.threshold(cH, stripe_threshold, mode='soft')
            cV_thresh = pywt.threshold(cV, stripe_threshold, mode='soft')

            new_coeffs.append((cH_thresh, cV_thresh, cD_thresh))

        # Reconstruct the denoised slice
        denoised_slice = pywt.waverec2(new_coeffs, wavelet)

        # Handle size mismatch due to wavelet decomposition
        if denoised_slice.shape != slice_data.shape:
            denoised_slice = denoised_slice[:slice_data.shape[0], :slice_data.shape[1]]

        result[:, :, z] = denoised_slice

    print("Wavelet denoising complete")

    return result


def combined_denoising(data, z_threshold=0.8, wavelet='db4', wavelet_level=3):
    """
    Apply both denoising methods sequentially

    Args:
        data: 3D numpy array
        z_threshold: Z-axis threshold for noise zone
        wavelet: Wavelet type for stripe removal
        wavelet_level: Wavelet decomposition level

    Returns:
        Fully denoised 3D numpy array
    """
    print("=== Combined Denoising Pipeline ===")

    # Step 1: Remove z-direction noise
    print("\nStep 1/2: Removing Z-direction noise...")
    data_z_clean = remove_z_noise_inpainting(data, noise_threshold_z=z_threshold)

    # Step 2: Remove x-y stripes
    print("\nStep 2/2: Removing X-Y stripe artifacts...")
    data_fully_clean = remove_xy_stripes_wavelet(data_z_clean, wavelet=wavelet, level=wavelet_level)

    print("\n=== Denoising Complete ===")

    return data_fully_clean


if __name__ == "__main__":
    # Load data
    print("Loading NIfTI file...")
    img = nib.load('gneo_sample_sr_189.nii.gz')
    data = img.get_fdata()

    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.3f}, {data.max():.3f}]")

    # Apply combined denoising
    denoised_data = combined_denoising(data, z_threshold=0.8, wavelet='db4', wavelet_level=3)

    print(f"\nDenoised data range: [{denoised_data.min():.3f}, {denoised_data.max():.3f}]")

    # Save result
    print("\nSaving denoised NIfTI file...")
    denoised_img = nib.Nifti1Image(denoised_data, img.affine, img.header)
    nib.save(denoised_img, 'gneo_sample_sr_189_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_denoised.nii.gz")

    # Save intermediate results for comparison
    print("\nSaving intermediate results...")
    z_only_clean = remove_z_noise_inpainting(data)
    z_only_img = nib.Nifti1Image(z_only_clean, img.affine, img.header)
    nib.save(z_only_img, 'gneo_sample_sr_189_z_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_z_denoised.nii.gz")

    xy_only_clean = remove_xy_stripes_wavelet(data)
    xy_only_img = nib.Nifti1Image(xy_only_clean, img.affine, img.header)
    nib.save(xy_only_img, 'gneo_sample_sr_189_xy_denoised.nii.gz')
    print("Saved: gneo_sample_sr_189_xy_denoised.nii.gz")
