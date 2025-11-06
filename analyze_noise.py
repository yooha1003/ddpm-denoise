#!/usr/bin/env python3
"""
Analyze noise in NIfTI medical imaging data
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load the NIfTI file
print("Loading NIfTI file...")
img = nib.load('gneo_sample_sr_189.nii.gz')
data = img.get_fdata()

print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
print(f"Data mean: {data.mean():.2f}")
print(f"Data std: {data.std():.2f}")

# Get dimensions
x_dim, y_dim, z_dim = data.shape
print(f"\nDimensions: X={x_dim}, Y={y_dim}, Z={z_dim}")

# Analyze z-direction noise (upper part)
print("\n=== Analyzing Z-direction (upper part) ===")
upper_z_start = int(z_dim * 0.8)  # Top 20%
print(f"Checking upper z slices from {upper_z_start} to {z_dim-1}")

# Check statistics for upper z slices
upper_slices_mean = []
upper_slices_std = []
for z in range(upper_z_start, z_dim):
    slice_data = data[:, :, z]
    upper_slices_mean.append(slice_data.mean())
    upper_slices_std.append(slice_data.std())

print(f"Upper z-slices mean range: [{min(upper_slices_mean):.2f}, {max(upper_slices_mean):.2f}]")
print(f"Upper z-slices std range: [{min(upper_slices_std):.2f}, {max(upper_slices_std):.2f}]")

# Analyze x-y plane stripe artifacts
print("\n=== Analyzing X-Y plane stripe artifacts ===")
# Take a middle z slice
mid_z = z_dim // 2
mid_slice = data[:, :, mid_z]

# Check for periodic patterns in rows and columns
# Compute variance along rows and columns
row_variance = np.var(mid_slice, axis=1)  # Variance along each row
col_variance = np.var(mid_slice, axis=0)  # Variance along each column

print(f"Row variance range: [{row_variance.min():.2f}, {row_variance.max():.2f}]")
print(f"Col variance range: [{col_variance.min():.2f}, {col_variance.max():.2f}]")

# Look for stripes by checking if certain rows/columns have anomalous values
row_mean = np.mean(mid_slice, axis=1)
col_mean = np.mean(mid_slice, axis=0)

# Find anomalous rows/columns (outliers)
row_mean_median = np.median(row_mean)
col_mean_median = np.median(col_mean)
row_mean_std = np.std(row_mean)
col_mean_std = np.std(col_mean)

anomalous_rows = np.where(np.abs(row_mean - row_mean_median) > 3 * row_mean_std)[0]
anomalous_cols = np.where(np.abs(col_mean - col_mean_median) > 3 * col_mean_std)[0]

print(f"\nAnomalous rows detected: {len(anomalous_rows)}")
if len(anomalous_rows) > 0:
    print(f"  Indices: {anomalous_rows[:10]}...")  # Show first 10

print(f"Anomalous columns detected: {len(anomalous_cols)}")
if len(anomalous_cols) > 0:
    print(f"  Indices: {anomalous_cols[:10]}...")  # Show first 10

print("\nAnalysis complete. Generating visualizations...")
