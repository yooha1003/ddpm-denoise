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
        print("\n[1/4] Z-슬라이스 강도 보정 중...")

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
            if z_means[z] > 0.001:  # 거의 0이 아닌 슬라이스만 보정
                correction_factor = z_means_smoothed[z] / z_means[z]
                # adaptive 방법과 동일하게 clipping 적용
                correction_factor = np.clip(correction_factor, 0.7, 1.5)
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
        print("\n[3/4] 보수적 Z-영역 노이즈 제거 중...")

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

    def intensity_gradient_z_removal(self, min_z_start=0.70, gradient_threshold=2.0):
        """
        Mean Intensity Gradient 기반 Z-영역 노이즈 제거

        Mean intensity가 급격히 증가하는 지점을 자동 감지합니다.
        이 방법은 다양한 데이터셋에 더 잘 일반화됩니다.

        Args:
            min_z_start: 검색 시작 위치 (기본값: 0.70 = 70%부터)
            gradient_threshold: Gradient 임계값 (표준편차 배수)

        Returns:
            detected_boundary: 감지된 노이즈 경계 z-인덱스
        """
        print("\n[3/4] Mean Intensity Gradient 기반 Z-영역 노이즈 제거 중...")

        z_dim = self.shape[2]
        search_start = int(z_dim * min_z_start)

        # Z-축 mean intensity 계산
        z_means = np.array([self.data[:, :, z].mean() for z in range(z_dim)])

        # Savitzky-Golay 필터로 평활화 (노이즈 제거)
        z_means_smooth = signal.savgol_filter(z_means, window_length=11, polyorder=2)

        # Gradient (1차 미분) 계산 - 증가율 측정
        gradient = np.gradient(z_means_smooth)

        # 검색 영역에서만 분석
        gradient_search = gradient[search_start:]

        # Gradient 통계
        grad_mean = np.mean(gradient_search)
        grad_std = np.std(gradient_search)

        print(f"  검색 영역: Z={search_start}~{z_dim-1}")
        print(f"  Gradient 평균: {grad_mean:.6f}, 표준편차: {grad_std:.6f}")

        # 급격한 증가 (양수 gradient가 threshold 이상) 감지
        threshold_value = grad_mean + gradient_threshold * grad_std

        detected_boundary = None
        for z in range(search_start, z_dim):
            if gradient[z] > threshold_value:
                detected_boundary = z
                print(f"  급격한 intensity 증가 감지: Z={z}, gradient={gradient[z]:.6f}")
                break

        if detected_boundary is None:
            # 대안: mean intensity가 급격히 상승하는 지점 찾기
            # z_means_smooth의 90th percentile 이상인 첫 지점
            intensity_threshold = np.percentile(z_means_smooth, 90)
            for z in range(search_start, z_dim):
                if z_means_smooth[z] > intensity_threshold:
                    detected_boundary = z
                    print(f"  높은 intensity 영역 감지: Z={z}, intensity={z_means_smooth[z]:.6f}")
                    break

        if detected_boundary is None:
            # 최후의 수단: 85% 지점
            detected_boundary = int(z_dim * 0.85)
            print(f"  기본값 사용: Z={detected_boundary}")

        print(f"  노이즈 경계: Z={detected_boundary} ({detected_boundary/z_dim*100:.1f}%)")
        print(f"  제거할 슬라이스: {z_dim - detected_boundary}개")

        # Smooth transition 적용
        transition_length = 8
        transition_start = max(search_start, detected_boundary - transition_length)

        print(f"  Smooth transition: Z={transition_start} to Z={detected_boundary-1}")

        for i, z in enumerate(range(transition_start, detected_boundary)):
            t = (i + 1) / (transition_length + 1)
            alpha = 1.0 - (3 * t**2 - 2 * t**3)  # Smooth step function
            self.data[:, :, z] *= alpha

        # 노이즈 영역 완전 제거
        self.data[:, :, detected_boundary:] = 0.0

        self.metadata['steps'].append({
            'step': 'intensity_gradient_z_removal',
            'boundary': int(detected_boundary),
            'removed_slices': int(z_dim - detected_boundary),
            'transition_start': int(transition_start),
            'transition_length': transition_length,
            'gradient_threshold': gradient_threshold,
            'method': 'mean_intensity_gradient_detection'
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
        print("\n[3/4] 적응형 Z-영역 노이즈 제거 중...")

        z_dim = self.shape[2]
        search_start = int(z_dim * min_z_start)
        noise_scores = np.zeros(z_dim)

        # 각 슬라이스 분석
        for z in range(search_start, z_dim):
            slice_data = self.data[:, :, z]
            noise_scores[z] = self._analyze_slice_noise(slice_data)

        # 노이즈 경계 감지 (최소 5개 이상 연속되는 노이즈 영역 찾기)
        potential_noise = noise_scores >= noise_threshold

        in_noise_region = False
        noise_start = None

        for z in range(search_start, z_dim):
            if potential_noise[z] and not in_noise_region:
                # 노이즈 영역 시작
                noise_start = z
                in_noise_region = True
            elif not potential_noise[z] and in_noise_region:
                # 노이즈 영역 종료 - 짧으면 계속 탐색
                if z - noise_start < 5:
                    in_noise_region = False
                    noise_start = None
                else:
                    # 실질적인 노이즈 영역 발견
                    break

        if noise_start is not None:
            detected_boundary = noise_start
            print(f"  노이즈 경계 감지: Z={detected_boundary} ({detected_boundary/z_dim*100:.1f}%)")
            print(f"  제거할 슬라이스: {z_dim - detected_boundary}개")

            # Smooth transition 적용 (adaptive 방법과 동일)
            transition_length = 8
            transition_start = max(search_start, detected_boundary - transition_length)

            print(f"  Smooth transition: Z={transition_start} to Z={detected_boundary-1}")

            for i, z in enumerate(range(transition_start, detected_boundary)):
                t = (i + 1) / (transition_length + 1)
                alpha = 1.0 - (3 * t**2 - 2 * t**3)  # Smooth step function
                self.data[:, :, z] *= alpha

            # 노이즈 영역 완전 제거
            self.data[:, :, detected_boundary:] = 0.0

            self.metadata['steps'].append({
                'step': 'adaptive_z_removal',
                'boundary': int(detected_boundary),
                'removed_slices': int(z_dim - detected_boundary),
                'transition_start': int(transition_start),
                'transition_length': transition_length,
                'threshold': noise_threshold,
                'method': 'continuous_noise_region_with_smooth_transition'
            })

            return detected_boundary
        else:
            print("  노이즈 경계를 감지하지 못했습니다.")
            return None

    def _analyze_slice_noise(self, slice_data, threshold=0.05):
        """
        단일 슬라이스의 노이즈 특성 분석 (adaptive 방법과 동일)

        Args:
            slice_data: 분석할 2D 슬라이스
            threshold: 배경/전경 구분 임계값

        Returns:
            noise_score: 노이즈 점수 (0=조직, 1=노이즈)
        """
        mask = slice_data > threshold

        if np.sum(mask) < 10:
            return 1.0  # 거의 비어있으면 노이즈로 간주

        significant_data = slice_data[mask]

        # Metric 1: Local variance (salt-and-pepper has high local variance)
        local_var = ndimage.generic_filter(slice_data, np.var, size=3)
        local_var_masked = local_var[mask]
        mean_local_var = np.mean(local_var_masked) if len(local_var_masked) > 0 else 0

        # Metric 2: Neighbor difference (salt-and-pepper has large differences)
        edges = ndimage.sobel(slice_data)
        edge_strength = np.mean(np.abs(edges[mask])) if np.sum(mask) > 0 else 0

        # Metric 3: Spatial coherence (brain tissue has coherent structures)
        smoothed = ndimage.gaussian_filter(slice_data, sigma=2)
        correlation = np.corrcoef(slice_data[mask], smoothed[mask])[0, 1] if len(significant_data) > 10 else 0

        # Metric 4: Radial profile smoothness (brain has smooth center-to-edge decline)
        center_y, center_x = np.array(slice_data.shape) // 2
        y_indices, x_indices = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
        distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)

        # Bin by distance and compute mean intensity
        max_dist = int(np.min(slice_data.shape) / 2)
        radial_profile = []
        for r in range(0, max_dist, 5):
            ring_mask = (distances >= r) & (distances < r+5) & mask
            if np.sum(ring_mask) > 0:
                radial_profile.append(np.mean(slice_data[ring_mask]))

        # Check if radial profile decreases smoothly
        if len(radial_profile) > 2:
            radial_diff = np.mean(np.abs(np.diff(radial_profile)))
            declining_trend = radial_profile[0] > radial_profile[-1] if len(radial_profile) > 1 else False
        else:
            radial_diff = 999
            declining_trend = False

        # Metric 5: Intensity distribution (noise is more uniform, brain has peaks)
        hist, _ = np.histogram(significant_data, bins=20)
        hist_uniformity = np.std(hist) / (np.mean(hist) + 1e-10)

        # Metric 6: Fill ratio (noise tends to fill more of the space)
        fill_ratio = np.sum(mask) / mask.size

        # Compute noise score (0 = brain tissue, 1 = noise) - conditional accumulation
        noise_score = 0.0

        # High local variance indicates noise
        if mean_local_var > 0.02:
            noise_score += 0.2

        # High edge strength indicates noise
        if edge_strength > 0.15:
            noise_score += 0.2

        # Low correlation with smoothed version indicates noise
        if correlation < 0.7:
            noise_score += 0.2

        # Non-smooth radial profile indicates noise
        if radial_diff > 0.05:
            noise_score += 0.15

        # No declining trend indicates noise
        if not declining_trend:
            noise_score += 0.15

        # Uniform histogram indicates noise
        if hist_uniformity < 0.5:
            noise_score += 0.1

        noise_score = min(noise_score, 1.0)

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
        print("\n[2/4] 수평 스트라이프 제거 중...")

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

                    # FFT 수행 (adaptive 방법과 동일)
                    fft = np.fft.fft(z_line)
                    fft_shift = np.fft.fftshift(fft)
                    magnitude = np.abs(fft_shift)

                    # 저주파 보존 영역 설정
                    center = len(fft_shift) // 2
                    preserve_radius = max(5, len(fft_shift) // 15)

                    # 고주파 억제 (98th 백분위수 이상)
                    threshold = np.percentile(magnitude, 98)

                    # adaptive 방법과 동일한 필터 마스크 생성 (하드코딩 0.5)
                    filter_mask = np.ones_like(fft_shift, dtype=np.float64)
                    for idx in range(len(fft_shift)):
                        distance = abs(idx - center)
                        if distance > preserve_radius and magnitude[idx] > threshold:
                            filter_mask[idx] = 0.5

                    # 필터 적용
                    fft_filtered = fft_shift * filter_mask

                    # 역변환
                    fft_back = np.fft.ifftshift(fft_filtered)
                    filtered = np.fft.ifft(fft_back).real

                    self.data[i, j, :] = filtered

            print(f"  반복 {iter_num + 1}/{iterations} 완료 (처리: {processed}, 스킵: {skipped})")

        # 원본 데이터 범위로 클리핑 - adaptive 방법에서는 하지 않음!
        # self.data = np.clip(self.data, self.data_min, self.data_max)

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

    def gentle_final_smoothing(self, sigma=0.2):
        """
        최종 부드러운 평활화

        각 슬라이스에 부드러운 Gaussian 필터를 적용하여
        디노이징 과정에서 생긴 미세한 아티팩트를 제거합니다.

        Args:
            sigma: Gaussian 필터의 표준편차 (작을수록 부드러움)
        """
        print("\n[4/4] 최종 평활화 중...")

        for z in range(self.shape[2]):
            if z % 40 == 0:
                print(f"  처리 중 z={z}/{self.shape[2]}...")
            self.data[:, :, z] = ndimage.gaussian_filter(self.data[:, :, z], sigma=sigma)

        print(f"  평활화 완료 (sigma={sigma})")

        self.metadata['steps'].append({
            'step': 'gentle_final_smoothing',
            'sigma': sigma
        })

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
        # adaptive 방법과 동일한 파라미터 사용
        self.correct_z_slice_intensity(detection_threshold=1.0, smoothing_window=13)

        # 2. 수평 스트라이프 제거 (먼저 수행 - 엣지 보존을 위해)
        self.remove_horizontal_stripes(filter_strength=0.5, iterations=1)

        # 3. Mean Intensity Gradient 기반 Z-영역 노이즈 제거
        # Mean intensity 급증 지점을 자동 감지 (더 일반화된 방법)
        self.intensity_gradient_z_removal(min_z_start=0.70, gradient_threshold=2.0)

        # 4. 최종 부드러운 평활화 (adaptive 방법 추가)
        self.gentle_final_smoothing()

        # 5. 엣지 보존율 계산
        edge_preservation = self.compute_edge_preservation()

        print("\n" + "=" * 70)
        print("처리 완료!")
        print("=" * 70)
        print(f"엣지 보존율: {edge_preservation:.2f}%")
        print(f"최종 데이터 범위: [{self.data.min():.3f}, {self.data.max():.3f}]")

        # 6. 결과 저장
        self.save_output()

        # 7. 비교 이미지 생성
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
