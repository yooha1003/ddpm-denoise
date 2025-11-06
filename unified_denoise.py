#!/usr/bin/env python3
"""
통합 의료 영상 디노이징 파이프라인
Unified Medical Image Denoising Pipeline

이 스크립트는 DDPM(Denoising Diffusion Probabilistic Models) 생성 과정에서
발생하는 아티팩트를 제거하기 위한 통합 디노이징 파이프라인입니다.

주요 기능:
1. Z-슬라이스 강도 보정 (Savitzky-Golay 필터링)
2. 적응형 Z-영역 노이즈 제거 (Slice-wise 분석)
3. 수평 스트라이프 제거 (FFT 기반)

Author: Claude
Date: 2025-11-06
"""

import numpy as np
import nibabel as nib
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from pathlib import Path
import json


class UnifiedDenoiser:
    """통합 디노이징 클래스"""

    def __init__(self, input_path, output_path='denoised_output.nii.gz'):
        """
        Args:
            input_path: 입력 NIfTI 파일 경로
            output_path: 출력 NIfTI 파일 경로
        """
        self.input_path = input_path
        self.output_path = output_path
        self.nii = nib.load(input_path)
        self.data = self.nii.get_fdata().copy()
        self.original_data = self.data.copy()
        self.data_min = self.data.min()
        self.data_max = self.data.max()
        self.shape = self.data.shape
        self.metadata = {
            'input_file': str(input_path),
            'shape': self.shape,
            'original_range': (float(self.data_min), float(self.data_max)),
            'steps': []
        }

    def correct_z_slice_intensity(self, detection_threshold=2.0, smoothing_window=5):
        """
        Z-슬라이스 강도 보정

        각 z-슬라이스의 평균 강도를 분석하여 이상값을 감지하고 보정합니다.
        Savitzky-Golay 필터를 사용하여 부드러운 참조 곡선을 생성하고,
        이 곡선에서 크게 벗어난 슬라이스를 보정합니다.

        Args:
            detection_threshold: 이상 슬라이스 감지 임계값 (표준편차 배수)
            smoothing_window: Savitzky-Golay 필터 윈도우 크기

        Returns:
            corrected_slices: 보정된 슬라이스 인덱스 리스트
        """
        print("\n[1/3] Z-슬라이스 강도 보정 중...")

        z_dim = self.shape[2]
        z_means = np.array([self.data[:, :, z].mean() for z in range(z_dim)])

        # Savitzky-Golay 필터로 부드러운 참조 곡선 생성
        z_means_smoothed = signal.savgol_filter(
            z_means,
            window_length=smoothing_window,
            polyorder=2
        )

        # 편차 계산 및 이상값 감지
        deviations = z_means - z_means_smoothed
        std_dev = np.std(deviations)
        anomalous_slices = np.where(np.abs(deviations) > detection_threshold * std_dev)[0]

        print(f"  감지된 이상 슬라이스: {len(anomalous_slices)}개")

        # 이상 슬라이스 보정
        for z in anomalous_slices:
            if z_means[z] > 0:  # 0이 아닌 슬라이스만 보정
                correction_factor = z_means_smoothed[z] / z_means[z]
                self.data[:, :, z] *= correction_factor

        self.metadata['steps'].append({
            'step': 'z_slice_intensity_correction',
            'anomalous_slices': anomalous_slices.tolist(),
            'count': len(anomalous_slices),
            'threshold': detection_threshold
        })

        return anomalous_slices

    def conservative_z_removal(self, min_percentile=0.82):
        """
        보수적 Z-영역 노이즈 제거

        통계적 신호 분석을 사용하여 노이즈 경계를 감지합니다.
        두개골 경계를 보존하기 위해 보수적인 임계값을 사용합니다.

        Args:
            min_percentile: 검색 시작 위치 (기본값: 0.82 = 상위 18%만 검색)

        Returns:
            detected_boundary: 감지된 노이즈 경계 z-인덱스
        """
        print("\n[3/3] 보수적 Z-영역 노이즈 제거 중...")

        z_dim = self.shape[2]
        x_dim, y_dim = self.shape[0], self.shape[1]

        # 통계 계산
        z_means = np.array([self.data[:, :, z].mean() for z in range(z_dim)])
        z_stds = np.array([self.data[:, :, z].std() for z in range(z_dim)])
        z_nonzero_counts = np.array([np.sum(self.data[:, :, z] > 0.05) for z in range(z_dim)])
        z_nonzero_ratio = z_nonzero_counts / (x_dim * y_dim)

        # 신호 정규화
        z_means_norm = (z_means - z_means.min()) / (z_means.max() - z_means.min() + 1e-10)
        z_stds_norm = (z_stds - z_stds.min()) / (z_stds.max() - z_stds.min() + 1e-10)
        z_ratio_norm = z_nonzero_ratio

        # 뇌 신호 조합
        brain_signal = (z_means_norm + z_stds_norm + z_ratio_norm) / 3.0
        brain_signal_smooth = signal.savgol_filter(brain_signal, window_length=11, polyorder=2)

        # 검색 시작점 설정
        search_start = int(z_dim * min_percentile)

        # 신호가 크게 떨어지는 지점 찾기
        threshold = np.percentile(brain_signal_smooth[search_start:], 40)

        low_signal = brain_signal_smooth < threshold
        candidates = np.where(low_signal[search_start:])[0]

        if len(candidates) > 0:
            detected_boundary = search_start + candidates[0]
        else:
            # 안전한 기본값 (85%)
            detected_boundary = int(z_dim * 0.85)

        # 최소값 보장
        detected_boundary = max(detected_boundary, int(z_dim * min_percentile))

        print(f"  검색 시작: Z={search_start} ({min_percentile*100:.0f}%)")
        print(f"  노이즈 경계 감지: Z={detected_boundary} ({detected_boundary/z_dim*100:.1f}%)")
        print(f"  제거할 슬라이스: {z_dim - detected_boundary}개")

        # 노이즈 영역 제거
        for z in range(detected_boundary, z_dim):
            self.data[:, :, z] = 0.0

        self.metadata['steps'].append({
            'step': 'conservative_z_removal',
            'boundary': int(detected_boundary),
            'removed_slices': int(z_dim - detected_boundary),
            'method': 'statistical_signal_analysis'
        })

        return detected_boundary

    def adaptive_z_removal(self, noise_threshold=0.35, min_z_start=0.80):
        """
        적응형 Z-영역 노이즈 제거 (고급)

        각 z-슬라이스를 개별적으로 분석하여 salt-and-pepper 노이즈 특성을 감지합니다.
        6가지 메트릭을 사용하여 노이즈 점수를 계산하고, 임계값을 초과하는
        슬라이스부터 제거합니다.

        분석 메트릭:
        1. 국소 분산 (Local Variance) - salt-and-pepper 노이즈는 높은 국소 분산
        2. 엣지 강도 (Edge Strength) - 노이즈는 날카로운 랜덤 엣지 생성
        3. 공간 일관성 (Spatial Coherence) - 뇌 조직은 평활화 버전과 상관관계 높음
        4. 방사형 프로파일 (Radial Profile) - 뇌 조직은 중심에서 가장자리로 부드럽게 감소
        5. 강도 분포 균일성 (Intensity Uniformity) - 노이즈는 균일한 히스토그램
        6. 채움 비율 (Fill Ratio) - 노이즈는 더 많은 공간을 채움

        Args:
            noise_threshold: 노이즈 감지 임계값 (0=조직, 1=노이즈)
            min_z_start: 분석 시작 위치 (전체 z-축의 백분율)

        Returns:
            detected_boundary: 감지된 노이즈 경계 z-인덱스
        """
        print("\n[2/3] 적응형 Z-영역 노이즈 제거 중...")

        z_dim = self.shape[2]
        search_start = int(z_dim * min_z_start)
        noise_scores = np.zeros(z_dim)

        # 각 슬라이스 분석
        for z in range(search_start, z_dim):
            slice_data = self.data[:, :, z]
            noise_scores[z] = self._analyze_slice_noise(slice_data)

        # 노이즈 경계 감지
        noisy_slices = np.where(noise_scores >= noise_threshold)[0]

        if len(noisy_slices) > 0:
            detected_boundary = noisy_slices[0]
            print(f"  노이즈 경계 감지: Z={detected_boundary} ({detected_boundary/z_dim*100:.1f}%)")
            print(f"  제거할 슬라이스: {z_dim - detected_boundary}개")

            # 노이즈 영역 제거
            for z in range(detected_boundary, z_dim):
                self.data[:, :, z] = 0.0

            self.metadata['steps'].append({
                'step': 'adaptive_z_removal',
                'boundary': int(detected_boundary),
                'removed_slices': int(z_dim - detected_boundary),
                'threshold': noise_threshold
            })

            return detected_boundary
        else:
            print("  노이즈 경계를 감지하지 못했습니다.")
            return None

    def _analyze_slice_noise(self, slice_data, threshold=0.05):
        """
        단일 슬라이스의 노이즈 특성 분석

        Args:
            slice_data: 분석할 2D 슬라이스
            threshold: 배경/전경 구분 임계값

        Returns:
            noise_score: 노이즈 점수 (0=조직, 1=노이즈)
        """
        mask = slice_data > threshold

        if np.sum(mask) < 10:
            return 1.0  # 거의 비어있으면 노이즈로 간주

        # 메트릭 1: 국소 분산
        local_var = ndimage.generic_filter(slice_data, np.var, size=3)
        avg_local_var = np.mean(local_var[mask])
        local_var_score = min(avg_local_var / 0.01, 1.0)

        # 메트릭 2: 엣지 강도
        edges = ndimage.sobel(slice_data)
        edge_strength = np.mean(np.abs(edges[mask]))
        edge_score = min(edge_strength / 0.1, 1.0)

        # 메트릭 3: 공간 일관성
        smoothed = ndimage.gaussian_filter(slice_data, sigma=2)
        if np.std(slice_data[mask]) > 0 and np.std(smoothed[mask]) > 0:
            correlation = np.corrcoef(slice_data[mask], smoothed[mask])[0, 1]
            coherence_score = 1.0 - max(0, correlation)
        else:
            coherence_score = 1.0

        # 메트릭 4: 방사형 프로파일 부드러움
        cy, cx = np.array(slice_data.shape) // 2
        y, x = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_r = int(np.max(r[mask])) if np.sum(mask) > 0 else 1

        radial_profile = []
        for i in range(0, max_r, max(1, max_r // 20)):
            ring_mask = (r >= i) & (r < i + max(1, max_r // 20)) & mask
            if np.sum(ring_mask) > 0:
                radial_profile.append(np.mean(slice_data[ring_mask]))

        if len(radial_profile) > 2:
            radial_smoothness = np.std(np.diff(radial_profile))
            radial_score = min(radial_smoothness / 0.05, 1.0)
        else:
            radial_score = 1.0

        # 메트릭 5: 강도 분포 균일성
        hist, _ = np.histogram(slice_data[mask], bins=30)
        hist_uniformity = np.std(hist) / (np.mean(hist) + 1e-10)
        uniformity_score = 1.0 - min(hist_uniformity / 2.0, 1.0)

        # 메트릭 6: 채움 비율
        fill_ratio = np.sum(mask) / mask.size
        fill_score = fill_ratio

        # 종합 노이즈 점수 계산 (가중 평균)
        weights = [0.25, 0.20, 0.25, 0.15, 0.10, 0.05]
        scores = [local_var_score, edge_score, coherence_score,
                  radial_score, uniformity_score, fill_score]

        noise_score = np.average(scores, weights=weights)
        return noise_score

    def remove_horizontal_stripes(self, filter_strength=0.5, iterations=1):
        """
        수평 스트라이프 제거 (FFT 기반)

        FFT(Fast Fourier Transform)를 사용하여 주파수 도메인에서
        z-축 방향의 수평 스트라이프를 제거합니다.

        각 (x, y) 위치에서 z-축을 따라 1D FFT를 수행하고,
        고주파 성분을 선택적으로 억제하여 스트라이프를 제거합니다.

        Args:
            filter_strength: 필터 강도 (0.0~1.0, 높을수록 강하게 제거)
            iterations: 반복 횟수

        Returns:
            stripe_reduction: 스트라이프 감소율 (%)
        """
        print("\n[2/3] 수평 스트라이프 제거 중...")

        # Z-축 방향 분산 계산 (스트라이프 측정)
        z_profiles_before = []
        for i in range(0, self.shape[0], 10):  # 샘플링
            for j in range(0, self.shape[1], 10):
                profile = self.data[i, j, :]
                if profile.std() > 0.01:
                    z_profiles_before.append(profile.std())
        variance_before = np.mean(z_profiles_before) if z_profiles_before else 0

        for iter_num in range(iterations):
            processed = 0
            skipped = 0
            for i in range(self.shape[0]):
                if i % 40 == 0:
                    print(f"  처리 중 x={i}/{self.shape[0]}...")

                for j in range(self.shape[1]):
                    z_line = self.data[i, j, :]

                    # 변화가 거의 없는 라인은 스킵
                    if z_line.std() < 0.01:
                        skipped += 1
                        continue

                    processed += 1

                    # FFT 수행
                    fft = np.fft.fft(z_line)
                    fft_shift = np.fft.fftshift(fft)
                    magnitude = np.abs(fft_shift)

                    # 저주파 보존 영역 설정
                    center = len(fft_shift) // 2
                    preserve_radius = max(5, len(fft_shift) // 15)

                    # 고주파 억제 (98th 백분위수 이상)
                    threshold = np.percentile(magnitude, 98)
                    suppress_mask = magnitude > threshold
                    suppress_mask[center - preserve_radius:center + preserve_radius] = False

                    # 필터 적용
                    fft_shift[suppress_mask] *= (1.0 - filter_strength)

                    # 역변환
                    fft_back = np.fft.ifftshift(fft_shift)
                    filtered = np.fft.ifft(fft_back).real

                    self.data[i, j, :] = filtered

            print(f"  반복 {iter_num + 1}/{iterations} 완료 (처리: {processed}, 스킵: {skipped})")

        # 원본 데이터 범위로 클리핑
        self.data = np.clip(self.data, self.data_min, self.data_max)

        # Z-축 방향 분산 재계산
        z_profiles_after = []
        for i in range(0, self.shape[0], 10):
            for j in range(0, self.shape[1], 10):
                profile = self.data[i, j, :]
                if profile.std() > 0.01:
                    z_profiles_after.append(profile.std())
        variance_after = np.mean(z_profiles_after) if z_profiles_after else 0

        # 스트라이프 감소율 계산
        if variance_before > 0:
            stripe_reduction = (1 - variance_after / variance_before) * 100
        else:
            stripe_reduction = 0.0

        print(f"  스트라이프 감소율: {stripe_reduction:.1f}%")

        self.metadata['steps'].append({
            'step': 'horizontal_stripe_removal',
            'filter_strength': filter_strength,
            'iterations': iterations,
            'stripe_reduction_percent': float(stripe_reduction)
        })

        return stripe_reduction

    def compute_edge_preservation(self):
        """
        엣지 보존율 계산

        Sobel 필터를 사용하여 원본과 디노이즈된 이미지의 엣지를 비교합니다.
        뇌 영역(중간 슬라이스)에서 측정하여 노이즈 제거 영향 제외

        Returns:
            edge_preservation: 엣지 보존율 (%)
        """
        # 중간 슬라이스로 측정 (노이즈 영역 제외)
        z_mid = self.shape[2] // 2

        original_slice = self.original_data[:, :, z_mid]
        denoised_slice = self.data[:, :, z_mid]

        original_edges = ndimage.sobel(original_slice)
        denoised_edges = ndimage.sobel(denoised_slice)

        original_edge_strength = np.mean(np.abs(original_edges))
        denoised_edge_strength = np.mean(np.abs(denoised_edges))

        edge_preservation = (denoised_edge_strength / original_edge_strength) * 100

        self.metadata['edge_preservation_percent'] = float(edge_preservation)
        self.metadata['edge_measurement_method'] = 'mid_slice_only'

        return edge_preservation

    def save_output(self):
        """디노이즈된 데이터 저장"""
        print("\n결과 저장 중...")

        # NIfTI 파일 저장
        denoised_nii = nib.Nifti1Image(self.data, self.nii.affine, self.nii.header)
        nib.save(denoised_nii, self.output_path)
        print(f"  저장됨: {self.output_path}")

        # 메타데이터 저장
        metadata_path = self.output_path.replace('.nii.gz', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"  메타데이터: {metadata_path}")

    def generate_comparison_images(self):
        """비교 시각화 생성"""
        print("\n비교 이미지 생성 중...")

        # 3뷰 비교
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Original vs Denoised - 3 View Comparison', fontsize=16, y=0.98)

        mid_x, mid_y, mid_z = [s // 2 for s in self.shape]

        # 원본
        axes[0, 0].imshow(self.original_data[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[0, 0].set_title('Original - Sagittal')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.original_data[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[0, 1].set_title('Original - Coronal')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(self.original_data[:, :, mid_z].T, cmap='gray', origin='lower')
        axes[0, 2].set_title('Original - Axial')
        axes[0, 2].axis('off')

        # 디노이즈
        axes[1, 0].imshow(self.data[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[1, 0].set_title('Denoised - Sagittal')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(self.data[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[1, 1].set_title('Denoised - Coronal')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(self.data[:, :, mid_z].T, cmap='gray', origin='lower')
        axes[1, 2].set_title('Denoised - Axial')
        axes[1, 2].axis('off')

        plt.tight_layout()
        comparison_path = self.output_path.replace('.nii.gz', '_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  저장됨: {comparison_path}")

        # Z-축 프로파일 비교
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        z_means_orig = [self.original_data[:, :, z].mean() for z in range(self.shape[2])]
        z_means_denoised = [self.data[:, :, z].mean() for z in range(self.shape[2])]

        axes[0].plot(z_means_orig, label='Original', alpha=0.7)
        axes[0].plot(z_means_denoised, label='Denoised', alpha=0.7)
        axes[0].set_xlabel('Z-slice index')
        axes[0].set_ylabel('Mean intensity')
        axes[0].set_title('Z-axis Intensity Profile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        z_stds_orig = [self.original_data[:, :, z].std() for z in range(self.shape[2])]
        z_stds_denoised = [self.data[:, :, z].std() for z in range(self.shape[2])]

        axes[1].plot(z_stds_orig, label='Original', alpha=0.7)
        axes[1].plot(z_stds_denoised, label='Denoised', alpha=0.7)
        axes[1].set_xlabel('Z-slice index')
        axes[1].set_ylabel('Std deviation')
        axes[1].set_title('Z-axis Variability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        profile_path = self.output_path.replace('.nii.gz', '_z_profile.png')
        plt.savefig(profile_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  저장됨: {profile_path}")

    def run_full_pipeline(self):
        """전체 디노이징 파이프라인 실행"""
        print("=" * 70)
        print("통합 의료 영상 디노이징 파이프라인")
        print("Unified Medical Image Denoising Pipeline")
        print("=" * 70)
        print(f"\n입력 파일: {self.input_path}")
        print(f"데이터 형상: {self.shape}")
        print(f"데이터 범위: [{self.data_min:.3f}, {self.data_max:.3f}]")

        # 1. Z-슬라이스 강도 보정
        self.correct_z_slice_intensity(detection_threshold=2.0, smoothing_window=5)

        # 2. 수평 스트라이프 제거 (먼저 수행 - 엣지 보존을 위해)
        self.remove_horizontal_stripes(filter_strength=0.5, iterations=1)

        # 3. 보수적 Z-영역 노이즈 제거 (나중에 수행)
        # 두개골 경계를 보존하는 안전한 방법
        self.conservative_z_removal(min_percentile=0.82)

        # 4. 엣지 보존율 계산
        edge_preservation = self.compute_edge_preservation()

        print("\n" + "=" * 70)
        print("처리 완료!")
        print("=" * 70)
        print(f"엣지 보존율: {edge_preservation:.2f}%")
        print(f"최종 데이터 범위: [{self.data.min():.3f}, {self.data.max():.3f}]")

        # 5. 결과 저장
        self.save_output()

        # 6. 비교 이미지 생성
        self.generate_comparison_images()

        print("\n모든 작업이 완료되었습니다!\n")

        return self.data, self.metadata


def main():
    """메인 실행 함수"""
    import sys

    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'gneo_sample_sr_189.nii.gz'

    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        output_file = 'gneo_sample_sr_189_unified_denoised.nii.gz'

    # 디노이저 생성 및 실행
    denoiser = UnifiedDenoiser(input_file, output_file)
    denoised_data, metadata = denoiser.run_full_pipeline()

    return denoised_data, metadata


if __name__ == '__main__':
    main()
