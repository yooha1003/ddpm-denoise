#!/usr/bin/env python3
"""
Visualize z-direction noise in upper slices
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load the NIfTI file
print("Loading NIfTI file...")
img = nib.load('gneo_sample_sr_189.nii.gz')
data = img.get_fdata()

x_dim, y_dim, z_dim = data.shape
print(f"Dimensions: {data.shape}")

# Create visualization for z-direction noise
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Z-direction Analysis: Upper Slices (DDPM Artifact Zone)', fontsize=16)

# Show slices from different z positions, focusing on upper region
z_positions = [
    z_dim - 1,      # Top slice
    z_dim - 5,
    z_dim - 10,
    z_dim - 15,
    z_dim - 20,
    z_dim - 30,
    z_dim - 40,
    z_dim - 50,
    z_dim // 2,     # Middle for comparison
    z_dim // 4,
    z_dim // 8,
    z_dim // 16
]

for idx, z in enumerate(z_positions):
    ax = axes[idx // 4, idx % 4]
    slice_data = data[:, :, z]

    im = ax.imshow(slice_data, cmap='gray', vmin=data.min(), vmax=data.max())
    ax.set_title(f'Z={z} (from top: {z_dim-1-z})\nMean: {slice_data.mean():.3f}, Std: {slice_data.std():.3f}')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('z_direction_noise.png', dpi=150, bbox_inches='tight')
print("Saved: z_direction_noise.png")

# Create a montage showing coronal view (x-z plane) to see z-direction patterns
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Coronal View (X-Z plane): Z-direction Noise Pattern', fontsize=16)

y_positions = [y_dim//6, y_dim//4, y_dim//3, y_dim//2, 2*y_dim//3, 5*y_dim//6]

for idx, y in enumerate(y_positions):
    ax = axes[idx // 3, idx % 3]
    coronal_slice = data[:, y, :]

    im = ax.imshow(coronal_slice.T, cmap='gray', aspect='auto', vmin=data.min(), vmax=data.max())
    ax.set_title(f'Coronal slice at Y={y}')
    ax.set_xlabel('X')
    ax.set_ylabel('Z (upper = top of brain)')

    # Add horizontal line to mark upper region
    upper_z_threshold = int(z_dim * 0.8)
    ax.axhline(y=upper_z_threshold, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Upper 20%')
    ax.legend()

    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig('coronal_z_noise.png', dpi=150, bbox_inches='tight')
print("Saved: coronal_z_noise.png")

# Analyze intensity profile along z-axis
print("\n=== Z-axis intensity profile ===")
z_means = [data[:, :, z].mean() for z in range(z_dim)]
z_stds = [data[:, :, z].std() for z in range(z_dim)]
z_maxs = [data[:, :, z].max() for z in range(z_dim)]
z_mins = [data[:, :, z].min() for z in range(z_dim)]

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Z-axis Intensity Statistics', fontsize=16)

axes[0, 0].plot(z_means)
axes[0, 0].set_title('Mean intensity along Z')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Mean intensity')
axes[0, 0].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5, label='Upper 20%')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(z_stds)
axes[0, 1].set_title('Standard deviation along Z')
axes[0, 1].set_xlabel('Z slice')
axes[0, 1].set_ylabel('Std dev')
axes[0, 1].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5, label='Upper 20%')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(z_maxs, label='Max')
axes[1, 0].plot(z_mins, label='Min')
axes[1, 0].set_title('Max/Min intensity along Z')
axes[1, 0].set_xlabel('Z slice')
axes[1, 0].set_ylabel('Intensity')
axes[1, 0].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5, label='Upper 20%')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Compute difference between consecutive slices
z_diffs = [np.abs(data[:, :, z] - data[:, :, z-1]).mean() for z in range(1, z_dim)]
axes[1, 1].plot(z_diffs)
axes[1, 1].set_title('Mean absolute difference between consecutive Z slices')
axes[1, 1].set_xlabel('Z slice')
axes[1, 1].set_ylabel('Mean abs diff')
axes[1, 1].axvline(x=int(z_dim*0.8), color='r', linestyle='--', alpha=0.5, label='Upper 20%')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('z_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: z_statistics.png")

print("\nZ-direction visualization complete!")
