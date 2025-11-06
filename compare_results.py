#!/usr/bin/env python3
"""
Compare original and denoised results
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load all versions
print("Loading data files...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
z_denoised = nib.load('gneo_sample_sr_189_z_denoised.nii.gz').get_fdata()
xy_denoised = nib.load('gneo_sample_sr_189_xy_denoised.nii.gz').get_fdata()
fully_denoised = nib.load('gneo_sample_sr_189_denoised.nii.gz').get_fdata()

x_dim, y_dim, z_dim = original.shape
print(f"Data shape: {original.shape}")

# 1. Compare Z-direction noise removal
print("\n1. Creating Z-direction comparison...")
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle('Z-direction Noise Removal Comparison', fontsize=18, fontweight='bold')

# Show top slices where noise was present
z_positions = [z_dim-5, z_dim-10, z_dim-15, z_dim-20, z_dim-30]

for idx, z in enumerate(z_positions):
    # Original
    axes[0, idx].imshow(original[:, :, z], cmap='gray', vmin=original.min(), vmax=original.max())
    axes[0, idx].set_title(f'Original Z={z}')
    axes[0, idx].axis('off')

    # Z-denoised
    axes[1, idx].imshow(z_denoised[:, :, z], cmap='gray', vmin=original.min(), vmax=original.max())
    axes[1, idx].set_title(f'Z-Inpainted Z={z}')
    axes[1, idx].axis('off')

    # Difference
    diff = np.abs(original[:, :, z] - z_denoised[:, :, z])
    axes[2, idx].imshow(diff, cmap='hot', vmin=0, vmax=diff.max())
    axes[2, idx].set_title(f'Difference Z={z}')
    axes[2, idx].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Z-Inpainted', fontsize=14, fontweight='bold')
axes[2, 0].set_ylabel('Difference', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_z_denoising.png', dpi=150, bbox_inches='tight')
print("Saved: comparison_z_denoising.png")

# 2. Compare X-Y stripe removal
print("\n2. Creating X-Y stripe removal comparison...")
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle('X-Y Stripe Artifact Removal Comparison', fontsize=18, fontweight='bold')

# Show slices at different z positions
z_samples = [z_dim//6, z_dim//4, z_dim//2, 2*z_dim//3]

for idx, z in enumerate(z_samples):
    # Original
    axes[idx, 0].imshow(original[:, :, z], cmap='gray')
    axes[idx, 0].set_title(f'Original Z={z}')
    axes[idx, 0].axis('off')

    # XY-denoised
    axes[idx, 1].imshow(xy_denoised[:, :, z], cmap='gray')
    axes[idx, 1].set_title(f'Wavelet Denoised Z={z}')
    axes[idx, 1].axis('off')

    # Zoomed region to see stripes better
    zoom_size = 80
    center_x, center_y = x_dim // 2, y_dim // 2
    zoom_orig = original[center_y-zoom_size//2:center_y+zoom_size//2,
                        center_x-zoom_size//2:center_x+zoom_size//2, z]
    zoom_denoised = xy_denoised[center_y-zoom_size//2:center_y+zoom_size//2,
                                center_x-zoom_size//2:center_x+zoom_size//2, z]

    axes[idx, 2].imshow(zoom_orig, cmap='gray', interpolation='nearest')
    axes[idx, 2].set_title(f'Original (Zoom) Z={z}')
    axes[idx, 2].axis('off')

    axes[idx, 3].imshow(zoom_denoised, cmap='gray', interpolation='nearest')
    axes[idx, 3].set_title(f'Denoised (Zoom) Z={z}')
    axes[idx, 3].axis('off')

plt.tight_layout()
plt.savefig('comparison_xy_denoising.png', dpi=150, bbox_inches='tight')
print("Saved: comparison_xy_denoising.png")

# 3. Complete comparison (all methods)
print("\n3. Creating complete comparison...")
fig, axes = plt.subplots(4, 4, figsize=(20, 20))
fig.suptitle('Complete Denoising Pipeline Comparison', fontsize=18, fontweight='bold')

# Select representative slices
z_samples = [z_dim//4, z_dim//3, z_dim//2, z_dim-15]

for idx, z in enumerate(z_samples):
    # Original
    axes[idx, 0].imshow(original[:, :, z], cmap='gray')
    axes[idx, 0].set_title(f'Original Z={z}')
    axes[idx, 0].axis('off')

    # Z-only denoised
    axes[idx, 1].imshow(z_denoised[:, :, z], cmap='gray')
    axes[idx, 1].set_title(f'Z-Inpainted Z={z}')
    axes[idx, 1].axis('off')

    # XY-only denoised
    axes[idx, 2].imshow(xy_denoised[:, :, z], cmap='gray')
    axes[idx, 2].set_title(f'Wavelet Denoised Z={z}')
    axes[idx, 2].axis('off')

    # Fully denoised
    axes[idx, 3].imshow(fully_denoised[:, :, z], cmap='gray')
    axes[idx, 3].set_title(f'Both Methods Z={z}')
    axes[idx, 3].axis('off')

# Add column labels
axes[0, 0].text(0.5, 1.15, 'Original', transform=axes[0, 0].transAxes,
                ha='center', fontsize=14, fontweight='bold')
axes[0, 1].text(0.5, 1.15, 'Z-Inpainting Only', transform=axes[0, 1].transAxes,
                ha='center', fontsize=14, fontweight='bold')
axes[0, 2].text(0.5, 1.15, 'Wavelet Only', transform=axes[0, 2].transAxes,
                ha='center', fontsize=14, fontweight='bold')
axes[0, 3].text(0.5, 1.15, 'Combined', transform=axes[0, 3].transAxes,
                ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_complete.png', dpi=150, bbox_inches='tight')
print("Saved: comparison_complete.png")

# 4. Statistical comparison
print("\n4. Creating statistical comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Statistical Analysis of Denoising Results', fontsize=16, fontweight='bold')

# Mean intensity along z-axis
z_means_orig = [original[:, :, z].mean() for z in range(z_dim)]
z_means_z = [z_denoised[:, :, z].mean() for z in range(z_dim)]
z_means_xy = [xy_denoised[:, :, z].mean() for z in range(z_dim)]
z_means_full = [fully_denoised[:, :, z].mean() for z in range(z_dim)]

axes[0, 0].plot(z_means_orig, label='Original', alpha=0.7)
axes[0, 0].plot(z_means_z, label='Z-Inpainted', alpha=0.7)
axes[0, 0].plot(z_means_full, label='Fully Denoised', alpha=0.7)
axes[0, 0].set_title('Mean Intensity along Z-axis')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Mean intensity')
axes[0, 0].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.3, label='Noise zone')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Std deviation along z-axis
z_stds_orig = [original[:, :, z].std() for z in range(z_dim)]
z_stds_z = [z_denoised[:, :, z].std() for z in range(z_dim)]
z_stds_xy = [xy_denoised[:, :, z].std() for z in range(z_dim)]
z_stds_full = [fully_denoised[:, :, z].std() for z in range(z_dim)]

axes[0, 1].plot(z_stds_orig, label='Original', alpha=0.7)
axes[0, 1].plot(z_stds_z, label='Z-Inpainted', alpha=0.7)
axes[0, 1].plot(z_stds_full, label='Fully Denoised', alpha=0.7)
axes[0, 1].set_title('Std Deviation along Z-axis')
axes[0, 1].set_xlabel('Z slice')
axes[0, 1].set_ylabel('Std deviation')
axes[0, 1].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.3)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histogram comparison
axes[0, 2].hist(original.flatten(), bins=50, alpha=0.5, label='Original', density=True)
axes[0, 2].hist(fully_denoised.flatten(), bins=50, alpha=0.5, label='Denoised', density=True)
axes[0, 2].set_title('Intensity Distribution')
axes[0, 2].set_xlabel('Intensity')
axes[0, 2].set_ylabel('Density')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# SNR improvement (simple estimate)
# Using std of background as noise estimate
mid_z = z_dim // 2
mid_slice_orig = original[:, :, mid_z]
mid_slice_denoised = fully_denoised[:, :, mid_z]

# Row variance comparison
row_var_orig = np.var(mid_slice_orig, axis=1)
row_var_denoised = np.var(mid_slice_denoised, axis=1)

axes[1, 0].plot(row_var_orig, label='Original', alpha=0.7)
axes[1, 0].plot(row_var_denoised, label='Denoised', alpha=0.7)
axes[1, 0].set_title(f'Row Variance (Z={mid_z})')
axes[1, 0].set_xlabel('Y position')
axes[1, 0].set_ylabel('Variance')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Column variance comparison
col_var_orig = np.var(mid_slice_orig, axis=0)
col_var_denoised = np.var(mid_slice_denoised, axis=0)

axes[1, 1].plot(col_var_orig, label='Original', alpha=0.7)
axes[1, 1].plot(col_var_denoised, label='Denoised', alpha=0.7)
axes[1, 1].set_title(f'Column Variance (Z={mid_z})')
axes[1, 1].set_xlabel('X position')
axes[1, 1].set_ylabel('Variance')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Difference map
diff_total = np.abs(original - fully_denoised)
diff_mean_per_z = [diff_total[:, :, z].mean() for z in range(z_dim)]

axes[1, 2].plot(diff_mean_per_z)
axes[1, 2].set_title('Mean Absolute Difference per Slice')
axes[1, 2].set_xlabel('Z slice')
axes[1, 2].set_ylabel('Mean |Original - Denoised|')
axes[1, 2].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.3, label='Noise zone')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comparison_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: comparison_statistics.png")

# 5. Coronal view comparison for z-direction
print("\n5. Creating coronal view comparison...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Coronal View: Z-direction Noise Removal', fontsize=16, fontweight='bold')

y_samples = [y_dim//4, y_dim//3, y_dim//2, 2*y_dim//3]

for idx, y in enumerate(y_samples):
    # Original coronal
    coronal_orig = original[:, y, :].T
    axes[0, idx].imshow(coronal_orig, cmap='gray', aspect='auto')
    axes[0, idx].set_title(f'Original Y={y}')
    axes[0, idx].set_xlabel('X')
    axes[0, idx].set_ylabel('Z')
    axes[0, idx].axhline(y=int(z_dim*0.8), color='r', linestyle='--', linewidth=1, alpha=0.5)

    # Denoised coronal
    coronal_denoised = fully_denoised[:, y, :].T
    axes[1, idx].imshow(coronal_denoised, cmap='gray', aspect='auto')
    axes[1, idx].set_title(f'Denoised Y={y}')
    axes[1, idx].set_xlabel('X')
    axes[1, idx].set_ylabel('Z')
    axes[1, idx].axhline(y=int(z_dim*0.8), color='r', linestyle='--', linewidth=1, alpha=0.5)

axes[0, 0].text(-0.1, 0.5, 'Original', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold')
axes[1, 0].text(-0.1, 0.5, 'Denoised', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('comparison_coronal.png', dpi=150, bbox_inches='tight')
print("Saved: comparison_coronal.png")

# Print summary statistics
print("\n" + "="*60)
print("DENOISING SUMMARY STATISTICS")
print("="*60)
print(f"\nOriginal data:")
print(f"  Range: [{original.min():.4f}, {original.max():.4f}]")
print(f"  Mean: {original.mean():.4f}")
print(f"  Std: {original.std():.4f}")

print(f"\nZ-inpainted data:")
print(f"  Range: [{z_denoised.min():.4f}, {z_denoised.max():.4f}]")
print(f"  Mean: {z_denoised.mean():.4f}")
print(f"  Std: {z_denoised.std():.4f}")

print(f"\nWavelet denoised data:")
print(f"  Range: [{xy_denoised.min():.4f}, {xy_denoised.max():.4f}]")
print(f"  Mean: {xy_denoised.mean():.4f}")
print(f"  Std: {xy_denoised.std():.4f}")

print(f"\nFully denoised data:")
print(f"  Range: [{fully_denoised.min():.4f}, {fully_denoised.max():.4f}]")
print(f"  Mean: {fully_denoised.mean():.4f}")
print(f"  Std: {fully_denoised.std():.4f}")

print(f"\nMean absolute difference:")
print(f"  Original vs Z-inpainted: {np.abs(original - z_denoised).mean():.4f}")
print(f"  Original vs Wavelet: {np.abs(original - xy_denoised).mean():.4f}")
print(f"  Original vs Fully denoised: {np.abs(original - fully_denoised).mean():.4f}")

print("\n" + "="*60)
print("Comparison visualizations complete!")
print("="*60)
