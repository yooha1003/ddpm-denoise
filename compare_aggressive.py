#!/usr/bin/env python3
"""
Compare original, previous (final), and aggressive denoising results
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
print("Loading data files...")
original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
previous = nib.load('gneo_sample_sr_189_final_denoised.nii.gz').get_fdata()
aggressive = nib.load('gneo_sample_sr_189_aggressive_denoised.nii.gz').get_fdata()

x_dim, y_dim, z_dim = original.shape
print(f"Data shape: {original.shape}")

# 1. Sagittal view comparison (horizontal stripes)
print("\n1. Creating sagittal view comparison...")
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('Sagittal View: Progressive Stripe Removal', fontsize=18, fontweight='bold')

x_positions = [x_dim//3, x_dim//2, 2*x_dim//3]

for idx, x in enumerate(x_positions):
    # Original
    axes[0, idx].imshow(original[x, :, :].T, cmap='gray', aspect='auto')
    axes[0, idx].set_title(f'Original X={x}', fontsize=14)
    axes[0, idx].set_xlabel('Y')
    axes[0, idx].set_ylabel('Z')
    axes[0, idx].axhline(y=144, color='r', linestyle='--', linewidth=1, alpha=0.5, label='75% line')

    # Previous (final)
    axes[1, idx].imshow(previous[x, :, :].T, cmap='gray', aspect='auto')
    axes[1, idx].set_title(f'Previous Denoising X={x}', fontsize=14)
    axes[1, idx].set_xlabel('Y')
    axes[1, idx].set_ylabel('Z')
    axes[1, idx].axhline(y=144, color='r', linestyle='--', linewidth=1, alpha=0.5)

    # Aggressive
    axes[2, idx].imshow(aggressive[x, :, :].T, cmap='gray', aspect='auto')
    axes[2, idx].set_title(f'Aggressive Denoising X={x}', fontsize=14)
    axes[2, idx].set_xlabel('Y')
    axes[2, idx].set_ylabel('Z')
    axes[2, idx].axhline(y=144, color='r', linestyle='--', linewidth=1, alpha=0.5)

# Add row labels
axes[0, 0].text(-0.15, 0.5, 'ORIGINAL\n(Stripes)', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.15, 0.5, 'PREVIOUS\n(Mild)', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
axes[2, 0].text(-0.15, 0.5, 'AGGRESSIVE\n(Strong)', transform=axes[2, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('aggressive_sagittal_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: aggressive_sagittal_comparison.png")

# 2. Upper Z region comparison
print("\n2. Creating upper z region comparison...")
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle('Upper Z Region: Near-Zero Requirement', fontsize=18, fontweight='bold')

upper_z_slices = [z_dim-40, z_dim-30, z_dim-20, z_dim-10, z_dim-5]

for idx, z in enumerate(upper_z_slices):
    # Original
    axes[0, idx].imshow(original[:, :, z], cmap='gray')
    axes[0, idx].set_title(f'Original Z={z}')
    axes[0, idx].axis('off')

    # Previous
    axes[1, idx].imshow(previous[:, :, z], cmap='gray')
    axes[1, idx].set_title(f'Previous Z={z}')
    axes[1, idx].axis('off')

    # Aggressive
    axes[2, idx].imshow(aggressive[:, :, z], cmap='gray')
    axes[2, idx].set_title(f'Aggressive Z={z}')
    axes[2, idx].axis('off')

axes[0, 0].text(-0.15, 0.5, 'ORIGINAL', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.15, 0.5, 'PREVIOUS', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
axes[2, 0].text(-0.15, 0.5, 'AGGRESSIVE\n(Near Zero)', transform=axes[2, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('aggressive_upper_z_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: aggressive_upper_z_comparison.png")

# 3. 3-view comparison
print("\n3. Creating 3-view comparison...")
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle('3-View Comparison: Aggressive Denoising', fontsize=18, fontweight='bold')

mid_x, mid_y, mid_z = x_dim // 2, y_dim // 2, z_dim // 2

# Original
axes[0, 0].imshow(original[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[0, 0].set_title(f'Sagittal (X={mid_x})')
axes[0, 0].set_xlabel('Y')
axes[0, 0].set_ylabel('Z')

axes[0, 1].imshow(original[:, mid_y, :].T, cmap='gray', aspect='auto')
axes[0, 1].set_title(f'Coronal (Y={mid_y})')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Z')

axes[0, 2].imshow(original[:, :, mid_z], cmap='gray')
axes[0, 2].set_title(f'Axial (Z={mid_z})')
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('Y')

# Previous
axes[1, 0].imshow(previous[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[1, 0].set_title(f'Sagittal (X={mid_x})')
axes[1, 0].set_xlabel('Y')
axes[1, 0].set_ylabel('Z')

axes[1, 1].imshow(previous[:, mid_y, :].T, cmap='gray', aspect='auto')
axes[1, 1].set_title(f'Coronal (Y={mid_y})')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Z')

axes[1, 2].imshow(previous[:, :, mid_z], cmap='gray')
axes[1, 2].set_title(f'Axial (Z={mid_z})')
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('Y')

# Aggressive
axes[2, 0].imshow(aggressive[mid_x, :, :].T, cmap='gray', aspect='auto')
axes[2, 0].set_title(f'Sagittal (X={mid_x})')
axes[2, 0].set_xlabel('Y')
axes[2, 0].set_ylabel('Z')

axes[2, 1].imshow(aggressive[:, mid_y, :].T, cmap='gray', aspect='auto')
axes[2, 1].set_title(f'Coronal (Y={mid_y})')
axes[2, 1].set_xlabel('X')
axes[2, 1].set_ylabel('Z')

axes[2, 2].imshow(aggressive[:, :, mid_z], cmap='gray')
axes[2, 2].set_title(f'Axial (Z={mid_z})')
axes[2, 2].set_xlabel('X')
axes[2, 2].set_ylabel('Y')

# Row labels
axes[0, 0].text(-0.2, 0.5, 'ORIGINAL', transform=axes[0, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))
axes[1, 0].text(-0.2, 0.5, 'PREVIOUS', transform=axes[1, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))
axes[2, 0].text(-0.2, 0.5, 'AGGRESSIVE', transform=axes[2, 0].transAxes,
                rotation=90, va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

plt.tight_layout()
plt.savefig('aggressive_3view_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: aggressive_3view_comparison.png")

# 4. Statistics comparison
print("\n4. Creating statistics comparison...")
from scipy import ndimage

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Statistical Comparison: Stripe Removal Analysis', fontsize=16, fontweight='bold')

# Z-intensity profiles
z_means_orig = np.array([original[:, :, z].mean() for z in range(z_dim)])
z_means_prev = np.array([previous[:, :, z].mean() for z in range(z_dim)])
z_means_aggr = np.array([aggressive[:, :, z].mean() for z in range(z_dim)])

axes[0, 0].plot(z_means_orig, label='Original', linewidth=2, alpha=0.8)
axes[0, 0].plot(z_means_prev, label='Previous', linewidth=2, alpha=0.8)
axes[0, 0].plot(z_means_aggr, label='Aggressive', linewidth=2, alpha=0.8)
axes[0, 0].set_title('Mean Intensity per Z-slice')
axes[0, 0].set_xlabel('Z slice')
axes[0, 0].set_ylabel('Mean intensity')
axes[0, 0].axvline(x=144, color='r', linestyle='--', alpha=0.5, label='75% line')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Intensity jumps
z_diffs_orig = np.abs(np.diff(z_means_orig))
z_diffs_prev = np.abs(np.diff(z_means_prev))
z_diffs_aggr = np.abs(np.diff(z_means_aggr))

axes[0, 1].plot(z_diffs_orig, label='Original', linewidth=2, alpha=0.8)
axes[0, 1].plot(z_diffs_prev, label='Previous', linewidth=2, alpha=0.8)
axes[0, 1].plot(z_diffs_aggr, label='Aggressive', linewidth=2, alpha=0.8)
axes[0, 1].set_title('Intensity Jumps (Stripe Indicator)')
axes[0, 1].set_xlabel('Z slice')
axes[0, 1].set_ylabel('|Mean(z) - Mean(z-1)|')
axes[0, 1].axvline(x=144, color='r', linestyle='--', alpha=0.5)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Upper region statistics
upper_start = 144
upper_means_orig = [original[:, :, z].mean() for z in range(upper_start, z_dim)]
upper_means_prev = [previous[:, :, z].mean() for z in range(upper_start, z_dim)]
upper_means_aggr = [aggressive[:, :, z].mean() for z in range(upper_start, z_dim)]

axes[0, 2].plot(range(upper_start, z_dim), upper_means_orig, label='Original', linewidth=2, alpha=0.8)
axes[0, 2].plot(range(upper_start, z_dim), upper_means_prev, label='Previous', linewidth=2, alpha=0.8)
axes[0, 2].plot(range(upper_start, z_dim), upper_means_aggr, label='Aggressive', linewidth=2, alpha=0.8)
axes[0, 2].set_title('Upper Region Mean Intensity (Z>144)')
axes[0, 2].set_xlabel('Z slice')
axes[0, 2].set_ylabel('Mean intensity')
axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Edge preservation
z_mid = z_dim // 2
orig_edges = ndimage.sobel(original[:, :, z_mid])
prev_edges = ndimage.sobel(previous[:, :, z_mid])
aggr_edges = ndimage.sobel(aggressive[:, :, z_mid])

edge_orig_strength = np.mean(np.abs(orig_edges))
edge_prev_strength = np.mean(np.abs(prev_edges))
edge_aggr_strength = np.mean(np.abs(aggr_edges))

methods = ['Original', 'Previous', 'Aggressive']
edge_strengths = [edge_orig_strength, edge_prev_strength, edge_aggr_strength]
edge_ratios = [1.0, edge_prev_strength/edge_orig_strength, edge_aggr_strength/edge_orig_strength]

axes[1, 0].bar(methods, edge_ratios, color=['#ff9999', '#ffff99', '#99ff99'], alpha=0.8)
axes[1, 0].set_title('Edge Preservation Ratio')
axes[1, 0].set_ylabel('Ratio vs Original')
axes[1, 0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% threshold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Upper region statistics summary
upper_region_orig = original[:, :, upper_start:]
upper_region_prev = previous[:, :, upper_start:]
upper_region_aggr = aggressive[:, :, upper_start:]

upper_stats = {
    'Original': [upper_region_orig.mean(), upper_region_orig.max(), upper_region_orig.std()],
    'Previous': [upper_region_prev.mean(), upper_region_prev.max(), upper_region_prev.std()],
    'Aggressive': [upper_region_aggr.mean(), upper_region_aggr.max(), upper_region_aggr.std()]
}

x_pos = np.arange(3)
width = 0.25

axes[1, 1].bar(x_pos - width, [v[0] for v in upper_stats.values()], width, label='Mean', alpha=0.8)
axes[1, 1].bar(x_pos, [v[1] for v in upper_stats.values()], width, label='Max', alpha=0.8)
axes[1, 1].bar(x_pos + width, [v[2] for v in upper_stats.values()], width, label='Std', alpha=0.8)
axes[1, 1].set_title('Upper Region Statistics (Z>144)')
axes[1, 1].set_ylabel('Intensity')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(upper_stats.keys())
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Summary text
summary_text = f"""
COMPARISON SUMMARY

Original:
  Range: [{original.min():.3f}, {original.max():.3f}]
  Mean: {original.mean():.3f}
  Upper mean: {upper_region_orig.mean():.4f}

Previous:
  Range: [{previous.min():.3f}, {previous.max():.3f}]
  Mean: {previous.mean():.3f}
  Upper mean: {upper_region_prev.mean():.4f}
  Edge: {edge_ratios[1]*100:.1f}%

Aggressive:
  Range: [{aggressive.min():.3f}, {aggressive.max():.3f}]
  Mean: {aggressive.mean():.3f}
  Upper mean: {upper_region_aggr.mean():.4f}
  Edge: {edge_ratios[2]*100:.1f}%

Stripe Reduction:
  Original jumps: {z_diffs_orig.max():.4f}
  Previous jumps: {z_diffs_prev.max():.4f}
  Aggressive jumps: {z_diffs_aggr.max():.4f}

✓ Aggressive: {(1-z_diffs_aggr.max()/z_diffs_orig.max())*100:.1f}% reduction
"""

axes[1, 2].text(0.05, 0.5, summary_text, transform=axes[1, 2].transAxes,
               fontsize=9, verticalalignment='center', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('aggressive_statistics.png', dpi=150, bbox_inches='tight')
print("Saved: aggressive_statistics.png")

# Print summary
print("\n" + "="*70)
print("AGGRESSIVE DENOISING RESULTS")
print("="*70)

print(f"\nUpper region (Z>{upper_start}) intensity:")
print(f"  Original: mean={upper_region_orig.mean():.4f}, max={upper_region_orig.max():.4f}")
print(f"  Previous: mean={upper_region_prev.mean():.4f}, max={upper_region_prev.max():.4f}")
print(f"  Aggressive: mean={upper_region_aggr.mean():.4f}, max={upper_region_aggr.max():.4f}")

print(f"\nStripe indicator (max intensity jump):")
print(f"  Original: {z_diffs_orig.max():.4f}")
print(f"  Previous: {z_diffs_prev.max():.4f} ({(1-z_diffs_prev.max()/z_diffs_orig.max())*100:.1f}% reduction)")
print(f"  Aggressive: {z_diffs_aggr.max():.4f} ({(1-z_diffs_aggr.max()/z_diffs_orig.max())*100:.1f}% reduction)")

print(f"\nEdge preservation:")
print(f"  Previous: {edge_ratios[1]*100:.2f}%")
print(f"  Aggressive: {edge_ratios[2]*100:.2f}%")

print("\n✓ Comparison complete!")
print("="*70)
