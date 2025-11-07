#!/usr/bin/env python3
"""
두 버전의 boundary masking 비교
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def compare_versions():
    """두 버전 비교"""

    print("데이터 로딩 중...")
    original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
    v1 = nib.load('gneo_sample_sr_189_unified_with_masking.nii.gz').get_fdata()
    v2 = nib.load('gneo_sample_sr_189_unified_with_masking_v2.nii.gz').get_fdata()

    # 관심 있는 X 슬라이스
    x_slices = [80, 96, 120, 140]

    # 비교 시각화
    fig, axes = plt.subplots(4, len(x_slices), figsize=(18, 16))
    fig.suptitle('Boundary Masking Comparison: v1 (15%) vs v2 (77%)', fontsize=16, y=0.98)

    for col, x in enumerate(x_slices):
        # 원본
        axes[0, col].imshow(original[x, :, :].T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'Original X={x}')
        axes[0, col].axis('off')

        # V1 (15.23% masked)
        axes[1, col].imshow(v1[x, :, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title(f'V1 (15% masked) X={x}')
        axes[1, col].axis('off')

        # V2 (77.55% masked)
        axes[2, col].imshow(v2[x, :, :].T, cmap='gray', origin='lower')
        axes[2, col].set_title(f'V2 (77% masked) X={x}')
        axes[2, col].axis('off')

        # 차이 (V1 vs V2)
        diff = np.abs(v1[x, :, :] - v2[x, :, :])
        axes[3, col].imshow(diff.T, cmap='hot', origin='lower')
        axes[3, col].set_title(f'Difference X={x}')
        axes[3, col].axis('off')

    plt.tight_layout()
    output_path = 'masking_version_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장됨: {output_path}")
    plt.close()

    # 통계
    print("\n=== 통계 비교 ===")
    print(f"\n원본:")
    print(f"  범위: [{original.min():.3f}, {original.max():.3f}]")
    print(f"  평균: {original.mean():.3f}")
    print(f"  0인 복셀: {np.sum(original == 0):,}")

    print(f"\nV1 (15.23% masked):")
    print(f"  범위: [{v1.min():.3f}, {v1.max():.3f}]")
    print(f"  평균: {v1.mean():.3f}")
    print(f"  0인 복셀: {np.sum(v1 == 0):,} ({np.sum(v1 == 0)/v1.size*100:.2f}%)")

    print(f"\nV2 (77.55% masked):")
    print(f"  범위: [{v2.min():.3f}, {v2.max():.3f}]")
    print(f"  평균: {v2.mean():.3f}")
    print(f"  0인 복셀: {np.sum(v2 == 0):,} ({np.sum(v2 == 0)/v2.size*100:.2f}%)")

if __name__ == '__main__':
    compare_versions()
