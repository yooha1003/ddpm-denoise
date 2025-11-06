#!/usr/bin/env python3
"""
Create 3-view comparison (Sagittal, Coronal, Axial)
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
print("Loading data files...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
improved = nib.load('gneo_sample_sr_189_improved_denoised.nii.gz').get_fdata()

x_dim, y_dim, z_dim = original.shape
print(f"Data shape: {original.shape}")

# Take middle slices for each view
mid_x = x_dim // 2
mid_y = y_dim // 2
mid_z = z_dim // 2

print(f"Middle positions: X={mid_x}, Y={mid_y}, Z={mid_z}")

# Create 2x3 comparison (Original vs Improved)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('3-View Comparison: Original vs Improved Denoising', fontsize=18, fontweight='bold')

# --- ORIGINAL (Top Row) ---
# Sagittal view (YZ plane) - slice through X
sagittal_orig = original[mid_x, :, :].T
axes[0, 0].imshow(sagittal_orig, cmap='gray', aspect='auto', vmin=original.min(), vmax=original.max())
axes[0, 0].set_title(f'Sagittal View (X={mid_x})', fontsize=14)
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('Z')
axes[0, 0].axhline(y=int(z_dim*0.8), color='r', linestyle='--', linewidth=1, alpha=0.5, label='Noise zone')

# Coronal view (XZ plane) - slice through Y
coronal_orig = original[:, mid_y, :].T
axes[0, 1].imshow(coronal_orig, cmap='gray', aspect='auto', vmin=original.min(), vmax=original.max())
axes[0, 1].set_title(f'Coronal View (Y={mid_y})', fontsize=14)
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Z')
axes[0, 1].axhline(y=int(z_dim*0.8), color='r', linestyle='--', linewidth=1, alpha=0.5, label='Noise zone')

# Axial view (XY plane) - slice through Z
axial_orig = original[:, :, mid_z]
axes[0, 2].imshow(axial_orig, cmap='gray', vmin=original.min(), vmax=original.max())
axes[0, 2].set_title(f'Axial View (Z={mid_z})', fontsize=14)
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')

# --- IMPROVED (Bottom Row) ---
# Sagittal view (YZ plane)
sagittal_improved = improved[mid_x, :, :].T
axes[1, 0].imshow(sagittal_improved, cmap='gray', aspect='auto', vmin=original.min(), vmax=original.max())
axes[1, 0].set_title(f'Sagittal View (X={mid_x}) - Denoised', fontsize=14)
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('Z')
axes[1, 0].axhline(y=int(z_dim*0.8), color='r', linestyle='--', linewidth=1, alpha=0.5, label='Noise zone')

# Coronal view (XZ plane)
coronal_improved = improved[:, mid_y, :].T
axes[1, 1].imshow(coronal_improved, cmap='gray', aspect='auto', vmin=original.min(), vmax=original.max())
axes[1, 1].set_title(f'Coronal View (Y={mid_y}) - Denoised', fontsize=14)
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Z')
axes[1, 1].axhline(y=int(z_dim*0.8), color='r', linestyle='--', linewidth=1, alpha=0.5, label='Noise zone')

# Axial view (XY plane)
axial_improved = improved[:, :, mid_z]
axes[1, 2].imshow(axial_improved, cmap='gray', vmin=original.min(), vmax=original.max())
axes[1, 2].set_title(f'Axial View (Z={mid_z}) - Denoised', fontsize=14)
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('Y')

# Add row labels
axes[0, 0].text(-0.15, 0.5, 'ORIGINAL', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
axes[1, 0].text(-0.15, 0.5, 'IMPROVED', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('3view_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: 3view_comparison.png")

# Create additional view showing upper z region (where noise was)
print("\nCreating upper z region view...")
upper_z = z_dim - 15  # Near the top where noise is prominent

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'3-View Comparison at Upper Z Region (Z={upper_z} - Noise Zone)',
             fontsize=18, fontweight='bold')

# --- ORIGINAL (Top Row) ---
sagittal_orig_upper = original[mid_x, :, :].T
axes[0, 0].imshow(sagittal_orig_upper, cmap='gray', aspect='auto')
axes[0, 0].set_title('Sagittal View', fontsize=14)
axes[0, 0].axhline(y=upper_z, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label=f'Z={upper_z}')
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('Z')
axes[0, 0].legend()

coronal_orig_upper = original[:, mid_y, :].T
axes[0, 1].imshow(coronal_orig_upper, cmap='gray', aspect='auto')
axes[0, 1].set_title('Coronal View', fontsize=14)
axes[0, 1].axhline(y=upper_z, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label=f'Z={upper_z}')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Z')
axes[0, 1].legend()

axial_orig_upper = original[:, :, upper_z]
axes[0, 2].imshow(axial_orig_upper, cmap='gray')
axes[0, 2].set_title(f'Axial View (Z={upper_z})', fontsize=14)
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')

# --- IMPROVED (Bottom Row) ---
sagittal_improved_upper = improved[mid_x, :, :].T
axes[1, 0].imshow(sagittal_improved_upper, cmap='gray', aspect='auto')
axes[1, 0].set_title('Sagittal View - Denoised', fontsize=14)
axes[1, 0].axhline(y=upper_z, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label=f'Z={upper_z}')
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('Z')
axes[1, 0].legend()

coronal_improved_upper = improved[:, mid_y, :].T
axes[1, 1].imshow(coronal_improved_upper, cmap='gray', aspect='auto')
axes[1, 1].set_title('Coronal View - Denoised', fontsize=14)
axes[1, 1].axhline(y=upper_z, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label=f'Z={upper_z}')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Z')
axes[1, 1].legend()

axial_improved_upper = improved[:, :, upper_z]
axes[1, 2].imshow(axial_improved_upper, cmap='gray')
axes[1, 2].set_title(f'Axial View (Z={upper_z}) - Denoised', fontsize=14)
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('Y')

# Add row labels
axes[0, 0].text(-0.15, 0.5, 'ORIGINAL\n(Noisy)', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
axes[1, 0].text(-0.15, 0.5, 'IMPROVED\n(Clean)', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.tight_layout()
plt.savefig('3view_upper_z.png', dpi=150, bbox_inches='tight')
print("Saved: 3view_upper_z.png")

print("\n✓ 3-view comparisons created successfully!")
