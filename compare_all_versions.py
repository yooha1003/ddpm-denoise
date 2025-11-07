#!/usr/bin/env python3
"""
세 버전의 boundary masking 비교
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def compare_all_versions():
    """세 버전 비교"""

    print("데이터 로딩 중...")
    original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
    v1 = nib.load('gneo_sample_sr_189_unified_with_masking.nii.gz').get_fdata()
    v2 = nib.load('gneo_sample_sr_189_unified_with_masking_v2.nii.gz').get_fdata()
    v3 = nib.load('gneo_sample_sr_189_unified_with_masking_v3.nii.gz').get_fdata()

    # 관심 있는 X 슬라이스 (왼쪽, 왼쪽중앙, 중앙, 오른쪽중앙, 오른쪽)
    x_slices = [60, 80, 96, 120, 150]

    # 비교 시각화
    fig, axes = plt.subplots(5, len(x_slices), figsize=(20, 18))
    fig.suptitle('Boundary Masking: V1 (15%) vs V2 (77%) vs V3 (67%)', fontsize=16, y=0.995)

    for col, x in enumerate(x_slices):
        # 원본
        axes[0, col].imshow(original[x, :, :].T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[0, col].set_title(f'Original X={x}', fontsize=10)
        axes[0, col].axis('off')

        # V1 (15.23% masked - edge detection 기반)
        axes[1, col].imshow(v1[x, :, :].T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[1, col].set_title(f'V1: Edge-based (15%)', fontsize=10)
        axes[1, col].axis('off')

        # V2 (77.55% masked - threshold 1.2x)
        axes[2, col].imshow(v2[x, :, :].T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[2, col].set_title(f'V2: Thresh 1.2x (77%)', fontsize=10)
        axes[2, col].axis('off')

        # V3 (66.79% masked - threshold 1.1x)
        axes[3, col].imshow(v3[x, :, :].T, cmap='gray', origin='lower', vmin=0, vmax=1)
        axes[3, col].set_title(f'V3: Thresh 1.1x (67%)', fontsize=10)
        axes[3, col].axis('off')

        # 차이 (V1 vs V3)
        diff = np.abs(v1[x, :, :] - v3[x, :, :])
        axes[4, col].imshow(diff.T, cmap='hot', origin='lower')
        axes[4, col].set_title(f'Diff (V1 vs V3)', fontsize=10)
        axes[4, col].axis('off')

    # Row labels
    row_labels = ['Original', 'V1 (Edge-based)', 'V2 (Thresh 1.2x)', 'V3 (Thresh 1.1x)', 'Difference']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.15, 0.5, label, transform=axes[i, 0].transAxes,
                       fontsize=12, va='center', ha='right', rotation=90, weight='bold')

    plt.tight_layout()
    output_path = 'masking_all_versions_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장됨: {output_path}")
    plt.close()

    # 통계
    print("\n=== 통계 비교 ===")
    versions = [
        ("Original", original),
        ("V1 (Edge-based)", v1),
        ("V2 (Threshold 1.2x)", v2),
        ("V3 (Threshold 1.1x)", v3)
    ]

    for name, data in versions:
        zero_voxels = np.sum(data == 0)
        zero_percent = zero_voxels / data.size * 100
        print(f"\n{name}:")
        print(f"  범위: [{data.min():.3f}, {data.max():.3f}]")
        print(f"  평균: {data.mean():.3f}")
        print(f"  0인 복셀: {zero_voxels:,} ({zero_percent:.2f}%)")

if __name__ == '__main__':
    compare_all_versions()
