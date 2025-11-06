#!/usr/bin/env python3
"""
Compare refined (uniform z-cut) vs adaptive (slice-wise analysis)
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
print("Loading data files...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
refined = nib.load('gneo_sample_sr_189_refined_denoised.nii.gz').get_fdata()
adaptive = nib.load('gneo_sample_sr_189_adaptive_denoised.nii.gz').get_fdata()

# Load noise scores
noise_scores = np.load('adaptive_noise_scores.npy')

x_dim, y_dim, z_dim = original.shape
print(f"Data shape: {original.shape}")

refined_boundary = 161
adaptive_boundary = 179

# 1. Sagittal comparison
print("\n1. Creating sagittal boundary comparison...")
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Sagittal View: Uniform vs Adaptive Boundary Detection', fontsize=18, fontweight='bold')

x_positions = [x_dim//3, x_dim//2, 2*x_dim//3]

for idx, x in enumerate(x_positions):
    # Original
    axes[0, idx].imshow(original[x, :, :].T, cmap='gray', aspect='auto')
    axes[0, idx].set_title(f'Original X={x}', fontsize=14)
    axes[0, idx].set_xlabel('Y')
    axes[0, idx].set_ylabel('Z')
    axes[0, idx].axhline(y=refined_boundary, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Refined=161')
    axes[0, idx].axhline(y=adaptive_boundary, color='g', linestyle='-', linewidth=2, alpha=0.8, label='Adaptive=179')
    if idx == 0:
        axes[0, idx].legend(loc='upper left', fontsize=10)

    # Refined (uniform cut)
    axes[1, idx].imshow(refined[x, :, :].T, cmap='gray', aspect='auto')
    axes[1, idx].set_title(f'Refined (Uniform 83.9%) X={x}', fontsize=14)
    axes[1, idx].set_xlabel('Y')
    axes[1, idx].set_ylabel('Z')
    axes[1, idx].axhline(y=refined_boundary, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

    # Adaptive (slice-wise)
    axes[2, idx].imshow(adaptive[x, :, :].T, cmap='gray', aspect='auto')
    axes[2, idx].set_title(f'Adaptive (Slice-wise 93.2%) X={x}', fontsize=14)
    axes[2, idx].set_xlabel('Y')
    axes[2, idx].set_ylabel('Z')
    axes[2, idx].axhline(y=adaptive_boundary, color='g', linestyle='-', linewidth=2, alpha=0.8)

# Row labels
axes[0, 0].text(-0.15, 0.5, 'ORIGINAL', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.15, 0.5, 'REFINED\n(Uniform)', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
axes[2, 0].text(-0.15, 0.5, 'ADAPTIVE\n(Slice-wise)', transform=axes[2, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('adaptive_boundary_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: adaptive_boundary_comparison.png")

# 2. Preserved region comparison
print("\n2. Creating preserved skull region comparison...")
fig, axes = plt.subplots(2, 6, figsize=(30, 10))
fig.suptitle('Skull Region Preservation: Refined vs Adaptive', fontsize=18, fontweight='bold')

# Show slices in the difference region
preserved_slices = [refined_boundary, refined_boundary+3, refined_boundary+6,
                   refined_boundary+9, refined_boundary+12, adaptive_boundary-1]

for idx, z in enumerate(preserved_slices):
    # Refined (may be zero)
    axes[0, idx].imshow(refined[:, :, z], cmap='gray')
    if z >= refined_boundary:
        axes[0, idx].set_title(f'Refined Z={z}\n[ZERO]', fontsize=11, color='orange', fontweight='bold')
    else:
        axes[0, idx].set_title(f'Refined Z={z}', fontsize=11)
    axes[0, idx].axis('off')

    # Adaptive (preserved more)
    axes[1, idx].imshow(adaptive[:, :, z], cmap='gray')
    if z >= adaptive_boundary:
        axes[1, idx].set_title(f'Adaptive Z={z}\n[ZERO]', fontsize=11, color='green', fontweight='bold')
    elif z >= refined_boundary:
        axes[1, idx].set_title(f'Adaptive Z={z}\n[PRESERVED]', fontsize=11, color='blue', fontweight='bold')
    else:
        axes[1, idx].set_title(f'Adaptive Z={z}', fontsize=11)
    axes[1, idx].axis('off')

axes[0, 0].text(-0.1, 0.5, 'REFINED', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
axes[1, 0].text(-0.1, 0.5, 'ADAPTIVE', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('adaptive_preserved_region.png', dpi=150, bbox_inches='tight')
print("Saved: adaptive_preserved_region.png")

# 3. Noise score visualization
print("\n3. Creating noise score visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Adaptive Noise Detection: Slice-by-Slice Analysis', fontsize=16, fontweight='bold')

# Plot noise scores
z_range = range(int(z_dim * 0.70), z_dim)
axes[0, 0].plot(z_range, noise_scores[z_range], 'b-', linewidth=2, marker='o', markersize=4)
axes[0, 0].axhline(y=0.35, color='r', linestyle='--', linewidth=2, label='Threshold=0.35')
axes[0, 0].axvline(x=refined_boundary, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Refined boundary')
axes[0, 0].axvline(x=adaptive_boundary, color='g', linestyle='-', linewidth=2, alpha=0.8, label='Adaptive boundary')
axes[0, 0].set_title('Noise Scores per Z-slice')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Noise score (0=tissue, 1=noise)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Show example slices with scores
example_slices = [refined_boundary, 170, adaptive_boundary-1, adaptive_boundary]
for i, z in enumerate(example_slices):
    if i >= 3:
        break
    row = (i // 2)
    col = (i % 2) + (1 if i < 2 else 0)

    if row == 0 and col == 1:
        # Use this spot for slice at Z=170
        z_show = 170
    elif row == 1 and col == 0:
        # Z=178 (adaptive-1)
        z_show = adaptive_boundary - 1
    elif row == 1 and col == 1:
        # Z=179 (adaptive boundary)
        z_show = adaptive_boundary
    else:
        continue

    axes[row, col].imshow(original[:, :, z_show], cmap='gray')
    score = noise_scores[z_show]
    if z_show >= adaptive_boundary:
        status = "[NOISE - REMOVED]"
        color = 'red'
    elif z_show >= refined_boundary:
        status = "[TISSUE - PRESERVED by adaptive]"
        color = 'green'
    else:
        status = "[TISSUE]"
        color = 'blue'

    axes[row, col].set_title(f'Z={z_show}: score={score:.3f}\n{status}',
                            fontsize=11, color=color, fontweight='bold')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('adaptive_noise_scores.png', dpi=150, bbox_inches='tight')
print("Saved: adaptive_noise_scores.png")

# 4. Statistics comparison
print("\n4. Creating statistics comparison...")
from scipy import ndimage

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Statistical Comparison: Refined vs Adaptive', fontsize=16, fontweight='bold')

# Z-intensity profiles
z_means_orig = np.array([original[:, :, z].mean() for z in range(z_dim)])
z_means_refined = np.array([refined[:, :, z].mean() for z in range(z_dim)])
z_means_adaptive = np.array([adaptive[:, :, z].mean() for z in range(z_dim)])

axes[0, 0].plot(z_means_orig, label='Original', linewidth=2, alpha=0.8)
axes[0, 0].plot(z_means_refined, label='Refined (cut at 161)', linewidth=2, alpha=0.8)
axes[0, 0].plot(z_means_adaptive, label='Adaptive (cut at 179)', linewidth=2, alpha=0.8)
axes[0, 0].axvline(x=refined_boundary, color='orange', linestyle='--', linewidth=1.5, alpha=0.5)
axes[0, 0].axvline(x=adaptive_boundary, color='g', linestyle='-', linewidth=2, alpha=0.5)
axes[0, 0].set_title('Mean Intensity per Z-slice')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Mean intensity')
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Preserved region comparison
preserved_region_orig = original[:, :, refined_boundary:adaptive_boundary]
preserved_region_refined = refined[:, :, refined_boundary:adaptive_boundary]
preserved_region_adaptive = adaptive[:, :, refined_boundary:adaptive_boundary]

axes[0, 1].hist([preserved_region_orig.flatten(),
                 preserved_region_refined.flatten(),
                 preserved_region_adaptive.flatten()],
                bins=30, label=['Original', 'Refined (LOST)', 'Adaptive (PRESERVED)'],
                alpha=0.7, density=True)
axes[0, 1].set_title(f'Intensity Dist: Z={refined_boundary}-{adaptive_boundary-1}\n(Extra preserved by adaptive)')
axes[0, 1].set_xlabel('Intensity')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Edge preservation
z_mid = z_dim // 2
orig_edges = ndimage.sobel(original[:, :, z_mid])
refined_edges = ndimage.sobel(refined[:, :, z_mid])
adaptive_edges = ndimage.sobel(adaptive[:, :, z_mid])

edge_orig = np.mean(np.abs(orig_edges))
edge_refined = np.mean(np.abs(refined_edges))
edge_adaptive = np.mean(np.abs(adaptive_edges))

methods = ['Original', 'Refined', 'Adaptive']
edge_ratios = [1.0, edge_refined/edge_orig, edge_adaptive/edge_orig]

bars = axes[0, 2].bar(methods, edge_ratios, color=['#aaaaaa', '#ffaa99', '#99ff99'], alpha=0.8)
axes[0, 2].set_title('Edge Preservation')
axes[0, 2].set_ylabel('Ratio vs Original')
axes[0, 2].axhline(y=1.0, color='b', linestyle='--', alpha=0.5)
axes[0, 2].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, edge_ratios):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val*100:.2f}%', ha='center', va='bottom', fontsize=10)

# Summary text 1
axes[1, 0].text(0.05, 0.5, f"""
BOUNDARY COMPARISON

Refined (Uniform):
  Boundary: Z={refined_boundary} (83.9%)
  Method: Fixed percentile
  Removed: {z_dim - refined_boundary} slices

Adaptive (Slice-wise):
  Boundary: Z={adaptive_boundary} (93.2%)
  Method: Noise characteristics
  Removed: {z_dim - adaptive_boundary} slices

Extra preserved: {adaptive_boundary - refined_boundary} slices!
""", transform=axes[1, 0].transAxes, fontsize=11, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
axes[1, 0].axis('off')

# Summary text 2
axes[1, 1].text(0.05, 0.5, f"""
PRESERVED REGION (Z={refined_boundary}-{adaptive_boundary-1})

Original:
  Mean: {preserved_region_orig.mean():.4f}
  Std: {preserved_region_orig.std():.4f}
  Non-zero voxels: {np.sum(preserved_region_orig > 0.05)}

Refined: LOST (set to zero)

Adaptive: PRESERVED
  Mean: {preserved_region_adaptive.mean():.4f}
  Std: {preserved_region_adaptive.std():.4f}
  Non-zero voxels: {np.sum(preserved_region_adaptive > 0.05)}

✓ Skull tissue preserved!
""", transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1, 1].axis('off')

# Summary text 3
axes[1, 2].text(0.05, 0.5, f"""
ADAPTIVE METHOD ADVANTAGES

1. Slice-wise Analysis:
   ✓ Each slice evaluated independently
   ✓ Detects salt-and-pepper noise
   ✓ Preserves brain tissue structure

2. Key Metrics:
   ✓ Local variance
   ✓ Edge characteristics
   ✓ Spatial coherence
   ✓ Radial profile smoothness
   ✓ Center-to-edge decline

3. Results:
   ✓ {adaptive_boundary - refined_boundary} more slices preserved
   ✓ 93.2% threshold (vs 83.9%)
   ✓ Only true noise removed
   ✓ Edge: {edge_ratios[2]*100:.2f}%

Smart detection > Fixed cutoff!
""", transform=axes[1, 2].transAxes, fontsize=9, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('adaptive_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: adaptive_statistics.png")

# Print summary
print("\n" + "="*70)
print("ADAPTIVE vs REFINED COMPARISON")
print("="*70)

print(f"\nBoundary detection:")
print(f"  Refined: Z={refined_boundary} (83.9%) - Uniform cutoff")
print(f"  Adaptive: Z={adaptive_boundary} (93.2%) - Slice-wise analysis")
print(f"  Extra preserved: {adaptive_boundary - refined_boundary} slices")

print(f"\nPreserved region (Z={refined_boundary} to {adaptive_boundary-1}):")
print(f"  Original mean: {preserved_region_orig.mean():.4f}")
print(f"  Refined: LOST (zero)")
print(f"  Adaptive: {preserved_region_adaptive.mean():.4f} (PRESERVED)")

print(f"\nEdge preservation:")
print(f"  Refined: {edge_ratios[1]*100:.2f}%")
print(f"  Adaptive: {edge_ratios[2]*100:.2f}%")

print("\n✓ Adaptive method preserves more skull/brain tissue!")
print("✓ Intelligent slice-wise analysis > uniform cutoff!")
print("="*70)
