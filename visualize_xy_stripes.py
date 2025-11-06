#!/usr/bin/env python3
"""
Visualize x-y plane stripe artifacts
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy import signal

# Load the NIfTI file
print("Loading NIfTI file...")
img = nib.load('gneo_sample_sr_189.nii.gz')
data = img.get_fdata()

x_dim, y_dim, z_dim = data.shape
print(f"Dimensions: {data.shape}")

# Create visualization for x-y plane stripes
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('X-Y Plane Analysis: Stripe Artifacts', fontsize=16)

# Show axial slices (x-y planes) at different z positions
z_positions = [
    z_dim // 8,
    z_dim // 6,
    z_dim // 4,
    z_dim // 3,
    z_dim // 2,
    2*z_dim // 3,
    3*z_dim // 4,
    5*z_dim // 6,
    7*z_dim // 8,
    z_dim - 30,
    z_dim - 15,
    z_dim - 5
]

for idx, z in enumerate(z_positions):
    ax = axes[idx // 4, idx % 4]
    slice_data = data[:, :, z]

    im = ax.imshow(slice_data, cmap='gray', vmin=data.min(), vmax=data.max())
    ax.set_title(f'Axial slice Z={z}\nMean: {slice_data.mean():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('xy_plane_slices.png', dpi=150, bbox_inches='tight')
print("Saved: xy_plane_slices.png")

# Analyze row and column patterns
print("\n=== Analyzing stripe patterns ===")
mid_z = z_dim // 2
mid_slice = data[:, :, mid_z]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'X-Y Plane Stripe Analysis (Z={mid_z})', fontsize=16)

# Show the slice
axes[0, 0].imshow(mid_slice, cmap='gray')
axes[0, 0].set_title('Axial slice')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')

# Enhanced contrast to see stripes better
axes[0, 1].imshow(mid_slice, cmap='gray', vmin=np.percentile(mid_slice, 1),
                  vmax=np.percentile(mid_slice, 99))
axes[0, 1].set_title('Enhanced contrast (1-99 percentile)')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')

# Show zoomed region to see stripes better
zoom_size = 64
center_x, center_y = x_dim // 2, y_dim // 2
zoom_region = mid_slice[center_y-zoom_size//2:center_y+zoom_size//2,
                        center_x-zoom_size//2:center_x+zoom_size//2]
axes[0, 2].imshow(zoom_region, cmap='gray', interpolation='nearest')
axes[0, 2].set_title(f'Zoomed center region ({zoom_size}x{zoom_size})')
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')

# Row-wise mean (average across X for each Y)
row_means = np.mean(mid_slice, axis=1)
axes[1, 0].plot(row_means)
axes[1, 0].set_title('Row-wise mean (averaged across X)')
axes[1, 0].set_xlabel('Y position')
axes[1, 0].set_ylabel('Mean intensity')
axes[1, 0].grid(True)

# Column-wise mean (average across Y for each X)
col_means = np.mean(mid_slice, axis=0)
axes[1, 1].plot(col_means)
axes[1, 1].set_title('Column-wise mean (averaged across Y)')
axes[1, 1].set_xlabel('X position')
axes[1, 1].set_ylabel('Mean intensity')
axes[1, 1].grid(True)

# 2D FFT to detect periodic patterns
axes[1, 2].set_title('Frequency analysis')
# Take FFT of a region with tissue (not background)
# Find region with highest variance
region_size = 128
max_var = 0
best_region = None
for cy in range(region_size//2, y_dim - region_size//2, 32):
    for cx in range(region_size//2, x_dim - region_size//2, 32):
        region = mid_slice[cy-region_size//2:cy+region_size//2,
                          cx-region_size//2:cx+region_size//2]
        if region.var() > max_var:
            max_var = region.var()
            best_region = region.copy()

if best_region is not None:
    fft = np.fft.fft2(best_region)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    magnitude_log = np.log1p(magnitude)  # log scale for visualization

    axes[1, 2].imshow(magnitude_log, cmap='hot')
    axes[1, 2].set_title('2D FFT magnitude (log scale)\nHigh values indicate periodic patterns')
axes[1, 2].set_xlabel('Frequency X')
axes[1, 2].set_ylabel('Frequency Y')

plt.tight_layout()
plt.savefig('xy_stripe_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: xy_stripe_analysis.png")

# Create detailed stripe detection
print("\n=== Detailed stripe detection ===")
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Stripe Detection: Multiple Z-slices', fontsize=16)

z_samples = [z_dim//6, z_dim//4, z_dim//3, z_dim//2, 2*z_dim//3,
             3*z_dim//4, 5*z_dim//6, z_dim-30, z_dim-10]

for idx, z in enumerate(z_samples):
    ax = axes[idx // 3, idx % 3]
    slice_data = data[:, :, z]

    # Apply edge detection to highlight stripes
    # Using Sobel filter
    from scipy import ndimage
    sx = ndimage.sobel(slice_data, axis=0)
    sy = ndimage.sobel(slice_data, axis=1)
    sobel = np.hypot(sx, sy)

    # Show edges
    im = ax.imshow(sobel, cmap='hot', vmin=0, vmax=np.percentile(sobel, 99))
    ax.set_title(f'Edge detection Z={z}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('xy_stripe_edges.png', dpi=150, bbox_inches='tight')
print("Saved: xy_stripe_edges.png")

# Analyze specific rows and columns for stripe patterns
print("\n=== Row/Column intensity profiles ===")
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle(f'Intensity Profiles at Z={mid_z}', fontsize=16)

# Plot multiple row profiles
axes[0, 0].set_title('Row intensity profiles (samples across slice)')
for y in range(0, y_dim, y_dim // 10):
    axes[0, 0].plot(mid_slice[y, :], alpha=0.5, label=f'Y={y}')
axes[0, 0].set_xlabel('X position')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True)

# Plot multiple column profiles
axes[0, 1].set_title('Column intensity profiles (samples across slice)')
for x in range(0, x_dim, x_dim // 10):
    axes[0, 1].plot(mid_slice[:, x], alpha=0.5, label=f'X={x}')
axes[0, 1].set_xlabel('Y position')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(True)

# Show variance across rows and columns
row_var = np.var(mid_slice, axis=1)
col_var = np.var(mid_slice, axis=0)

axes[1, 0].plot(row_var)
axes[1, 0].set_title('Variance of each row')
axes[1, 0].set_xlabel('Y position')
axes[1, 0].set_ylabel('Variance')
axes[1, 0].grid(True)

axes[1, 1].plot(col_var)
axes[1, 1].set_title('Variance of each column')
axes[1, 1].set_xlabel('X position')
axes[1, 1].set_ylabel('Variance')
axes[1, 1].grid(True)

# Difference from median for each row/column
row_med_diff = row_means - np.median(row_means)
col_med_diff = col_means - np.median(col_means)

axes[2, 0].plot(row_med_diff)
axes[2, 0].set_title('Row mean deviation from median')
axes[2, 0].set_xlabel('Y position')
axes[2, 0].set_ylabel('Deviation')
axes[2, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[2, 0].grid(True)

axes[2, 1].plot(col_med_diff)
axes[2, 1].set_title('Column mean deviation from median')
axes[2, 1].set_xlabel('X position')
axes[2, 1].set_ylabel('Deviation')
axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[2, 1].grid(True)

plt.tight_layout()
plt.savefig('xy_intensity_profiles.png', dpi=150, bbox_inches='tight')
print("Saved: xy_intensity_profiles.png")

print("\nX-Y plane stripe visualization complete!")
