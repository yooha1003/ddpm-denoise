#!/usr/bin/env python3
"""
Analyze why current denoising methods failed
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
print("Loading data...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
denoised = nib.load('gneo_sample_sr_189_denoised.nii.gz').get_fdata()

x_dim, y_dim, z_dim = original.shape

print(f"Analyzing denoising results...")

# Focus on problem areas
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle('Denoising Failure Analysis', fontsize=18, fontweight='bold')

# Check upper z slices (where inpainting was applied)
upper_z_samples = [z_dim-5, z_dim-10, z_dim-15, z_dim-20]

for idx, z in enumerate(upper_z_samples):
    # Original
    axes[idx, 0].imshow(original[:, :, z], cmap='gray', vmin=original.min(), vmax=original.max())
    axes[idx, 0].set_title(f'Original Z={z}')
    axes[idx, 0].axis('off')

    # Denoised
    axes[idx, 1].imshow(denoised[:, :, z], cmap='gray', vmin=original.min(), vmax=original.max())
    axes[idx, 1].set_title(f'Denoised Z={z}')
    axes[idx, 1].axis('off')

    # Difference
    diff = np.abs(original[:, :, z] - denoised[:, :, z])
    axes[idx, 2].imshow(diff, cmap='hot')
    axes[idx, 2].set_title(f'Abs Difference Z={z}')
    axes[idx, 2].axis('off')

    # Enhanced to see stripes
    axes[idx, 3].imshow(denoised[:, :, z], cmap='gray', vmin=np.percentile(denoised, 1),
                        vmax=np.percentile(denoised, 99))
    axes[idx, 3].set_title(f'Enhanced Contrast Z={z}')
    axes[idx, 3].axis('off')

plt.tight_layout()
plt.savefig('denoising_failure_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: denoising_failure_analysis.png")

# Check for stripes in middle slices
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('X-Y Stripe Analysis (Middle Slices)', fontsize=16, fontweight='bold')

mid_z = z_dim // 2
for i, z_offset in enumerate([0, 10, 20, 30]):
    z = mid_z + z_offset

    # Original
    axes[0, i].imshow(original[:, :, z], cmap='gray')
    axes[0, i].set_title(f'Original Z={z}')
    axes[0, i].axis('off')

    # Denoised
    axes[1, i].imshow(denoised[:, :, z], cmap='gray')
    axes[1, i].set_title(f'Denoised Z={z}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('stripe_check.png', dpi=150, bbox_inches='tight')
print("Saved: stripe_check.png")

# Check blurriness
print("\nBlur analysis:")
from scipy import ndimage

mid_z = z_dim // 2
orig_slice = original[:, :, mid_z]
denoised_slice = denoised[:, :, mid_z]

# Compute edge strength (high-pass filter)
orig_edges = ndimage.sobel(orig_slice)
denoised_edges = ndimage.sobel(denoised_slice)

print(f"Edge strength (original): {np.mean(np.abs(orig_edges)):.4f}")
print(f"Edge strength (denoised): {np.mean(np.abs(denoised_edges)):.4f}")
print(f"Edge strength ratio: {np.mean(np.abs(denoised_edges)) / np.mean(np.abs(orig_edges)):.4f}")

if np.mean(np.abs(denoised_edges)) < np.mean(np.abs(orig_edges)) * 0.8:
    print("⚠️ WARNING: Significant blur detected (edge strength reduced by >20%)")

print("\nAnalysis complete.")
