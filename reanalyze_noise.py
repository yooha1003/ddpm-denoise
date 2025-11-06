#!/usr/bin/env python3
"""
Re-analyze noise based on user feedback:
1. Z-direction noise in upper region (still present, affecting brain)
2. Sagittal view shows horizontal stripes along z-direction
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load original data
print("Loading original data...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
x_dim, y_dim, z_dim = original.shape
print(f"Data shape: {original.shape}")

# 1. Analyze sagittal view for horizontal stripes
print("\n=== Analyzing Sagittal View (YZ plane) ===")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Sagittal View Analysis: Horizontal Stripes in Z-direction', fontsize=16, fontweight='bold')

# Show multiple sagittal slices
x_positions = [x_dim//4, x_dim//3, x_dim//2, 2*x_dim//3, 3*x_dim//4]

for idx in range(min(5, len(x_positions))):
    ax_idx = idx if idx < 3 else idx - 3
    row_idx = 0 if idx < 3 else 1

    x = x_positions[idx]
    sagittal = original[x, :, :].T  # YZ plane, transposed so Z is vertical

    axes[row_idx, ax_idx].imshow(sagittal, cmap='gray', aspect='auto')
    axes[row_idx, ax_idx].set_title(f'Sagittal X={x}')
    axes[row_idx, ax_idx].set_xlabel('Y')
    axes[row_idx, ax_idx].set_ylabel('Z (top is high)')

    # Mark upper region
    axes[row_idx, ax_idx].axhline(y=int(z_dim*0.8), color='r', linestyle='--',
                                   linewidth=1, alpha=0.5, label='80% line')

# Hide empty subplot
if len(x_positions) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('sagittal_stripe_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: sagittal_stripe_analysis.png")

# 2. Analyze intensity profile along Z axis
print("\n=== Analyzing Z-axis Intensity Profile ===")
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Z-axis Intensity Analysis', fontsize=16, fontweight='bold')

# Mean intensity per z-slice
z_means = np.array([original[:, :, z].mean() for z in range(z_dim)])
z_stds = np.array([original[:, :, z].std() for z in range(z_dim)])
z_maxs = np.array([original[:, :, z].max() for z in range(z_dim)])

axes[0, 0].plot(z_means, linewidth=2)
axes[0, 0].set_title('Mean Intensity per Z-slice')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Mean intensity')
axes[0, 0].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[0, 0].grid(True, alpha=0.3)

# Detect anomalous z-slices
mean_of_means = np.mean(z_means)
std_of_means = np.std(z_means)
anomalous_z = np.where(np.abs(z_means - mean_of_means) > 2 * std_of_means)[0]
print(f"Anomalous Z slices (>2 std from mean): {len(anomalous_z)}")
if len(anomalous_z) > 0:
    print(f"  Indices: {anomalous_z[:20]}...")

axes[0, 0].scatter(anomalous_z, z_means[anomalous_z], color='red', s=50,
                  alpha=0.7, label='Anomalous slices')
axes[0, 0].legend()

# Std per slice
axes[0, 1].plot(z_stds, linewidth=2)
axes[0, 1].set_title('Std Deviation per Z-slice')
axes[0, 1].set_xlabel('Z slice')
axes[0, 1].set_ylabel('Std deviation')
axes[0, 1].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[0, 1].grid(True, alpha=0.3)

# Difference between consecutive slices
z_diffs = np.abs(np.diff(z_means))
axes[1, 0].plot(z_diffs, linewidth=2)
axes[1, 0].set_title('Absolute Difference Between Consecutive Z-slices')
axes[1, 0].set_xlabel('Z slice')
axes[1, 0].set_ylabel('|Mean(z) - Mean(z-1)|')
axes[1, 0].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[1, 0].grid(True, alpha=0.3)

# Find jumps (large differences)
jump_threshold = np.mean(z_diffs) + 2 * np.std(z_diffs)
jumps = np.where(z_diffs > jump_threshold)[0]
print(f"\nLarge intensity jumps between slices: {len(jumps)}")
if len(jumps) > 0:
    print(f"  At Z positions: {jumps[:20]}...")

axes[1, 0].scatter(jumps, z_diffs[jumps], color='red', s=50,
                  alpha=0.7, label='Large jumps')
axes[1, 0].axhline(y=jump_threshold, color='orange', linestyle='--',
                   alpha=0.5, label='Threshold')
axes[1, 0].legend()

# Max intensity per slice
axes[1, 1].plot(z_maxs, linewidth=2)
axes[1, 1].set_title('Max Intensity per Z-slice')
axes[1, 1].set_xlabel('Z slice')
axes[1, 1].set_ylabel('Max intensity')
axes[1, 1].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[1, 1].grid(True, alpha=0.3)

# 3. Check for stripe pattern: look at intensity along single lines
# Sample a line through the brain in sagittal view
mid_x = x_dim // 2
mid_y = y_dim // 2

# Get z-profile at center of brain
z_profile_center = original[mid_x, mid_y, :]

axes[2, 0].plot(z_profile_center, linewidth=2)
axes[2, 0].set_title(f'Intensity Profile at Center (X={mid_x}, Y={mid_y})')
axes[2, 0].set_xlabel('Z slice')
axes[2, 0].set_ylabel('Intensity')
axes[2, 0].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[2, 0].grid(True, alpha=0.3)

# Show sagittal slice with profile line marked
sagittal_mid = original[mid_x, :, :].T
axes[2, 1].imshow(sagittal_mid, cmap='gray', aspect='auto')
axes[2, 1].axvline(x=mid_y, color='yellow', linewidth=2, label=f'Profile line Y={mid_y}')
axes[2, 1].axhline(y=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[2, 1].set_title(f'Sagittal View X={mid_x}')
axes[2, 1].set_xlabel('Y')
axes[2, 1].set_ylabel('Z')
axes[2, 1].legend()

plt.tight_layout()
plt.savefig('z_intensity_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: z_intensity_analysis.png")

# 4. Look for specific z-slices with artifacts
print("\n=== Identifying Problem Z-slices ===")
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Axial Slices: Looking for Artifacts', fontsize=18, fontweight='bold')

# Check slices throughout volume
z_samples = np.linspace(0, z_dim-1, 20, dtype=int)

for idx, z in enumerate(z_samples):
    ax = axes[idx // 5, idx % 5]
    slice_data = original[:, :, z]

    ax.imshow(slice_data, cmap='gray')

    # Highlight if this is an anomalous slice
    is_anomalous = z in anomalous_z
    is_jump = z in jumps or (z-1) in jumps

    title = f'Z={z}'
    if is_anomalous:
        title += ' [ANOM]'
        ax.set_facecolor('#ffcccc')
    if is_jump:
        title += ' [JUMP]'
        ax.set_facecolor('#ffffcc')

    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('z_slices_overview.png', dpi=150, bbox_inches='tight')
print("Saved: z_slices_overview.png")

# 5. Create detailed view showing the stripe pattern in sagittal
print("\n=== Creating Detailed Sagittal Stripe View ===")
fig, axes = plt.subplots(2, 2, figsize=(16, 16))
fig.suptitle('Detailed Sagittal View: Stripe Pattern', fontsize=16, fontweight='bold')

# Show sagittal with enhanced contrast
mid_x = x_dim // 2
sagittal = original[mid_x, :, :].T

axes[0, 0].imshow(sagittal, cmap='gray', aspect='auto')
axes[0, 0].set_title(f'Sagittal X={mid_x} - Normal')
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('Z')

# Enhanced contrast
axes[0, 1].imshow(sagittal, cmap='gray', aspect='auto',
                  vmin=np.percentile(sagittal, 1), vmax=np.percentile(sagittal, 99))
axes[0, 1].set_title(f'Sagittal X={mid_x} - Enhanced Contrast')
axes[0, 1].set_xlabel('Y')
axes[0, 1].set_ylabel('Z')

# Show mean intensity across Y for each Z (this will show stripes as variations)
mean_across_y = np.mean(sagittal, axis=1)  # Average across Y for each Z
axes[1, 0].plot(mean_across_y, np.arange(z_dim), linewidth=2)
axes[1, 0].set_title('Mean Intensity Across Y (for each Z)')
axes[1, 0].set_xlabel('Mean intensity')
axes[1, 0].set_ylabel('Z')
axes[1, 0].axhline(y=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3)

# Heatmap showing variance in Y direction for each Z
variance_across_y = np.var(sagittal, axis=1)
axes[1, 1].plot(variance_across_y, np.arange(z_dim), linewidth=2)
axes[1, 1].set_title('Variance Across Y (for each Z)')
axes[1, 1].set_xlabel('Variance')
axes[1, 1].set_ylabel('Z')
axes[1, 1].axhline(y=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5)
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sagittal_detailed.png', dpi=150, bbox_inches='tight')
print("Saved: sagittal_detailed.png")

print("\n=== Analysis Complete ===")
print(f"\nSummary:")
print(f"  Total Z slices: {z_dim}")
print(f"  Anomalous slices: {len(anomalous_z)}")
print(f"  Large intensity jumps: {len(jumps)}")
print(f"  Upper region start (80%): Z={int(z_dim*0.8)}")
