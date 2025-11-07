#!/usr/bin/env python3
"""
Boundary Masking 시각화 스크립트
여러 sagittal 슬라이스에서 boundary masking 효과를 보여줍니다.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def visualize_boundary_masking():
    """Boundary masking 결과 시각화"""

    # 원본과 마스킹된 데이터 로드
    print("데이터 로딩 중...")
    original = nib.load('gneo_sample_sr_189.nii.gz').get_fdata()
    masked = nib.load('gneo_sample_sr_189_unified_with_masking.nii.gz').get_fdata()

    # 관심 있는 X 슬라이스 선택 (왼쪽, 중앙, 오른쪽)
    x_slices = [60, 96, 120, 150]

    # 시각화
    fig, axes = plt.subplots(3, len(x_slices), figsize=(18, 12))
    fig.suptitle('Boundary Masking 효과 - Sagittal Views', fontsize=16, y=0.98)

    for col, x in enumerate(x_slices):
        # 원본
        axes[0, col].imshow(original[x, :, :].T, cmap='gray', origin='lower')
        axes[0, col].set_title(f'원본 X={x}')
        axes[0, col].axis('off')

        # 마스킹 결과
        axes[1, col].imshow(masked[x, :, :].T, cmap='gray', origin='lower')
        axes[1, col].set_title(f'마스킹 후 X={x}')
        axes[1, col].axis('off')

        # 차이 (제거된 부분)
        diff = original[x, :, :] - masked[x, :, :]
        axes[2, col].imshow(diff.T, cmap='hot', origin='lower')
        axes[2, col].set_title(f'제거된 영역 X={x}')
        axes[2, col].axis('off')

    plt.tight_layout()
    output_path = 'gneo_sample_sr_189_boundary_masking_effect.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"저장됨: {output_path}")
    plt.close()

    # 통계 비교
    print("\n=== Boundary Masking 통계 ===")
    print(f"원본 범위: [{original.min():.3f}, {original.max():.3f}]")
    print(f"원본 평균: {original.mean():.3f}")
    print(f"마스킹 후 범위: [{masked.min():.3f}, {masked.max():.3f}]")
    print(f"마스킹 후 평균: {masked.mean():.3f}")

    # 마스킹된 복셀 수 계산
    masked_voxels = np.sum(masked == 0)
    total_voxels = masked.size
    masked_percent = (masked_voxels / total_voxels) * 100
    print(f"마스킹된 복셀: {masked_voxels:,} / {total_voxels:,} ({masked_percent:.2f}%)")

if __name__ == '__main__':
    visualize_boundary_masking()
