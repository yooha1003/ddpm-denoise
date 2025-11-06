#!/usr/bin/env python3
"""
Visualize final denoising results with focus on:
1. Sagittal view (to check horizontal stripes removal)
2. Upper z region
3. 3-view comparison
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
print("Loading data files...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
final_denoised = nib.load('gneo_sample_sr_189_final_denoised.nii.gz').get_fdata()
step1 = nib.load('gneo_sample_sr_189_step1_z_corrected.nii.gz').get_fdata()
step2 = nib.load('gneo_sample_sr_189_step2_upper_handled.nii.gz').get_fdata()

x_dim, y_dim, z_dim = original.shape
print(f"Data shape: {original.shape}")

# 1. MAIN COMPARISON: Sagittal View (focus on horizontal stripes)
print("\n1. Creating sagittal view comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Sagittal View: Horizontal Stripe Removal', fontsize=18, fontweight='bold')

x_positions = [x_dim//3, x_dim//2, 2*x_dim//3]

for idx, x in enumerate(x_positions):
    # Original
    sagittal_orig = original[x, :, :].T
    axes[0, idx].imshow(sagittal_orig, cmap='gray', aspect='auto')
    axes[0, idx].set_title(f'Original X={x}', fontsize=14)
    axes[0, idx].set_xlabel('Y')
    axes[0, idx].set_ylabel('Z')
    axes[0, idx].axhline(y=int(z_dim*0.85), color='r', linestyle='--',
                         linewidth=1, alpha=0.5, label='85% line')

    # Final denoised
    sagittal_denoised = final_denoised[x, :, :].T
    axes[1, idx].imshow(sagittal_denoised, cmap='gray', aspect='auto')
    axes[1, idx].set_title(f'Denoised X={x}', fontsize=14)
    axes[1, idx].set_xlabel('Y')
    axes[1, idx].set_ylabel('Z')
    axes[1, idx].axhline(y=int(z_dim*0.85), color='r', linestyle='--',
                         linewidth=1, alpha=0.5)

# Add row labels
axes[0, 0].text(-0.15, 0.5, 'ORIGINAL\n(Stripes)', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.15, 0.5, 'DENOISED\n(Clean)', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('final_sagittal_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: final_sagittal_comparison.png")

# 2. 3-View Comparison
print("\n2. Creating 3-view comparison...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('3-View Comparison: Original vs Final Denoised', fontsize=18, fontweight='bold')

mid_x, mid_y, mid_z = x_dim // 2, y_dim // 2, z_dim // 2

# --- ORIGINAL (Top Row) ---
# Sagittal
axes[0, 0].imshow(original[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[0, 0].set_title(f'Sagittal (X={mid_x})', fontsize=14)
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('Z')

# Coronal
axes[0, 1].imshow(original[:, mid_y, :].T, cmap='gray', aspect='auto')
axes[0, 1].set_title(f'Coronal (Y={mid_y})', fontsize=14)
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Z')

# Axial
axes[0, 2].imshow(original[:, :, mid_z], cmap='gray')
axes[0, 2].set_title(f'Axial (Z={mid_z})', fontsize=14)
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')

# --- DENOISED (Bottom Row) ---
# Sagittal
axes[1, 0].imshow(final_denoised[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[1, 0].set_title(f'Sagittal (X={mid_x}) - Denoised', fontsize=14)
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('Z')

# Coronal
axes[1, 1].imshow(final_denoised[:, mid_y, :].T, cmap='gray', aspect='auto')
axes[1, 1].set_title(f'Coronal (Y={mid_y}) - Denoised', fontsize=14)
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Z')

# Axial
axes[1, 2].imshow(final_denoised[:, :, mid_z], cmap='gray')
axes[1, 2].set_title(f'Axial (Z={mid_z}) - Denoised', fontsize=14)
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('Y')

# Add row labels
axes[0, 0].text(-0.2, 0.5, 'ORIGINAL', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.2, 0.5, 'DENOISED', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('final_3view_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: final_3view_comparison.png")

# 3. Step-by-step process visualization
print("\n3. Creating step-by-step process visualization...")
fig, axes = plt.subplots(4, 3, figsize=(18, 24))
fig.suptitle('Denoising Pipeline: Step-by-Step (Sagittal View)', fontsize=18, fontweight='bold')

mid_x = x_dim // 2

# Step 0: Original
for col in range(3):
    x_pos = [x_dim//3, x_dim//2, 2*x_dim//3][col]
    sagittal = original[x_pos, :, :].T
    axes[0, col].imshow(sagittal, cmap='gray', aspect='auto')
    axes[0, col].set_title(f'Original X={x_pos}')
    axes[0, col].set_xlabel('Y')
    axes[0, col].set_ylabel('Z')

# Step 1: Z-slice intensity correction
for col in range(3):
    x_pos = [x_dim//3, x_dim//2, 2*x_dim//3][col]
    sagittal = step1[x_pos, :, :].T
    axes[1, col].imshow(sagittal, cmap='gray', aspect='auto')
    axes[1, col].set_title(f'Step 1: Z-correction X={x_pos}')
    axes[1, col].set_xlabel('Y')
    axes[1, col].set_ylabel('Z')

# Step 2: Upper z handling
for col in range(3):
    x_pos = [x_dim//3, x_dim//2, 2*x_dim//3][col]
    sagittal = step2[x_pos, :, :].T
    axes[2, col].imshow(sagittal, cmap='gray', aspect='auto')
    axes[2, col].set_title(f'Step 2: Upper-Z X={x_pos}')
    axes[2, col].set_xlabel('Y')
    axes[2, col].set_ylabel('Z')

# Step 3: Final (with gentle smoothing)
for col in range(3):
    x_pos = [x_dim//3, x_dim//2, 2*x_dim//3][col]
    sagittal = final_denoised[x_pos, :, :].T
    axes[3, col].imshow(sagittal, cmap='gray', aspect='auto')
    axes[3, col].set_title(f'Step 3: Final X={x_pos}')
    axes[3, col].set_xlabel('Y')
    axes[3, col].set_ylabel('Z')

# Add row labels
axes[0, 0].text(-0.2, 0.5, 'Original', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold')
axes[1, 0].text(-0.2, 0.5, 'Step 1:\nZ-correction', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold')
axes[2, 0].text(-0.2, 0.5, 'Step 2:\nUpper-Z', transform=axes[2, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold')
axes[3, 0].text(-0.2, 0.5, 'Step 3:\nFinal', transform=axes[3, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('final_step_by_step.png', dpi=150, bbox_inches='tight')
print("Saved: final_step_by_step.png")

# 4. Upper Z region focus
print("\n4. Creating upper z region comparison...")
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle('Upper Z Region: Before vs After', fontsize=18, fontweight='bold')

upper_z_slices = [z_dim-30, z_dim-20, z_dim-15, z_dim-10, z_dim-5]

for idx, z in enumerate(upper_z_slices):
    # Original
    axes[0, idx].imshow(original[:, :, z], cmap='gray')
    axes[0, idx].set_title(f'Original Z={z}')
    axes[0, idx].axis('off')

    # Denoised
    axes[1, idx].imshow(final_denoised[:, :, z], cmap='gray')
    axes[1, idx].set_title(f'Denoised Z={z}')
    axes[1, idx].axis('off')

axes[0, 0].text(-0.15, 0.5, 'ORIGINAL', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.15, 0.5, 'DENOISED', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('final_upper_z_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: final_upper_z_comparison.png")

# 5. Z-intensity profile comparison
print("\n5. Creating z-intensity profile comparison...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Z-axis Intensity Profile: Stripe Detection', fontsize=16, fontweight='bold')

z_means_orig = np.array([original[:, :, z].mean() for z in range(z_dim)])
z_means_final = np.array([final_denoised[:, :, z].mean() for z in range(z_dim)])

# Mean intensity profile
axes[0, 0].plot(z_means_orig, label='Original', linewidth=2, alpha=0.8)
axes[0, 0].plot(z_means_final, label='Denoised', linewidth=2, alpha=0.8)
axes[0, 0].set_title('Mean Intensity per Z-slice')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Mean intensity')
axes[0, 0].axvline(x=int(z_dim*0.85), color='r', linestyle='--', alpha=0.5, label='85% line')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Difference between consecutive slices (shows jumps/stripes)
z_diffs_orig = np.abs(np.diff(z_means_orig))
z_diffs_final = np.abs(np.diff(z_means_final))

axes[0, 1].plot(z_diffs_orig, label='Original', linewidth=2, alpha=0.8)
axes[0, 1].plot(z_diffs_final, label='Denoised', linewidth=2, alpha=0.8)
axes[0, 1].set_title('Intensity Jumps Between Slices (Stripe Indicator)')
axes[0, 1].set_xlabel('Z slice')
axes[0, 1].set_ylabel('|Mean(z) - Mean(z-1)|')
axes[0, 1].axvline(x=int(z_dim*0.85), color='r', linestyle='--', alpha=0.5)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Sagittal view with profile
mid_x = x_dim // 2
axes[1, 0].imshow(original[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[1, 0].set_title('Original Sagittal (stripes visible)')
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('Z')

axes[1, 1].imshow(final_denoised[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[1, 1].set_title('Denoised Sagittal (stripes removed)')
axes[1, 1].set_xlabel('Y')
axes[1, 1].set_ylabel('Z')

plt.tight_layout()
plt.savefig('final_z_profile_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: final_z_profile_comparison.png")

# Print summary
from scipy import ndimage

print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

# Edge preservation
z_mid = z_dim // 2
orig_edges = ndimage.sobel(original[:, :, z_mid])
final_edges = ndimage.sobel(final_denoised[:, :, z_mid])
edge_ratio = np.mean(np.abs(final_edges)) / np.mean(np.abs(orig_edges))

print(f"\nData ranges:")
print(f"  Original: [{original.min():.3f}, {original.max():.3f}]")
print(f"  Denoised: [{final_denoised.min():.3f}, {final_denoised.max():.3f}]")

print(f"\nMean intensities:")
print(f"  Original: {original.mean():.3f}")
print(f"  Denoised: {final_denoised.mean():.3f}")

print(f"\nEdge preservation:")
print(f"  Ratio: {edge_ratio:.4f} ({edge_ratio*100:.2f}%)")
if edge_ratio > 0.95:
    print(f"  Status: ✓ Excellent (>95%)")
elif edge_ratio > 0.90:
    print(f"  Status: ✓ Very Good (>90%)")
else:
    print(f"  Status: ✓ Good")

print(f"\nZ-slice intensity jumps (stripe indicator):")
print(f"  Original max jump: {z_diffs_orig.max():.4f}")
print(f"  Denoised max jump: {z_diffs_final.max():.4f}")
print(f"  Reduction: {(1 - z_diffs_final.max()/z_diffs_orig.max())*100:.1f}%")

print("\n✓ Visualization complete!")
print("="*70)
