#!/usr/bin/env python3
"""
경계선 기반 ROI 마스킹 및 패딩
Boundary-based ROI Masking and Padding

3D 의료 영상에서 흰색 경계선을 검출하여 내부 ROI만 보존하고
경계선 바깥의 배경(노란색 영역)을 0으로 채우는 전처리 도구입니다.

주요 기능:
1. 전처리: 스무딩 & 배경 균질화
2. 경계선 검출 및 마스크 생성
3. Sagittal 슬라이스 단위 3D 볼륨 처리
4. 검증 시각화

Author: Claude
Date: 2025-11-06
"""

import numpy as np
import nibabel as nib
import cv2
from scipy import signal, ndimage
from skimage import restoration, morphology, filters
import matplotlib.pyplot as plt
from pathlib import Path
import json


class BoundaryMaskProcessor:
    """경계선 기반 마스킹 프로세서"""

    def __init__(self, input_path, output_path='boundary_masked_output.nii.gz'):
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
        self.shape = self.data.shape
        self.mask_3d = np.ones(self.shape, dtype=bool)  # 초기에는 모두 True

        self.metadata = {
            'input_file': str(input_path),
            'shape': self.shape,
            'original_range': (float(self.data.min()), float(self.data.max())),
            'processing_axis': 'sagittal',  # x-axis
            'steps': []
        }

    def preprocess_slice(self, slice_data, method='gaussian',
                         smooth_sigma=2.0, bg_removal='morphological'):
        """
        슬라이스 전처리: 스무딩 & 배경 균질화

        Args:
            slice_data: 2D 슬라이스 데이터
            method: 스무딩 방법 ('gaussian', 'median', 'nlm')
            smooth_sigma: 스무딩 강도
            bg_removal: 배경 제거 방법 ('gaussian', 'morphological', 'tophat')

        Returns:
            preprocessed: 전처리된 슬라이스
            debug_info: 디버깅 정보
        """
        # 1. 정규화 [0, 1]
        sl_min, sl_max = slice_data.min(), slice_data.max()
        if sl_max - sl_min < 1e-6:
            return np.zeros_like(slice_data), {}

        sl_norm = (slice_data - sl_min) / (sl_max - sl_min)

        # 2. 스무딩
        if method == 'gaussian':
            sl_smooth = ndimage.gaussian_filter(sl_norm, sigma=smooth_sigma)
        elif method == 'median':
            ksize = int(smooth_sigma * 2) * 2 + 1  # 홀수로 변환
            sl_smooth = ndimage.median_filter(sl_norm, size=ksize)
        elif method == 'nlm':
            # Non-Local Means (더 느리지만 경계 보존 우수)
            sl_smooth = restoration.denoise_nl_means(
                sl_norm,
                patch_size=5,
                patch_distance=7,
                h=0.1 * sl_norm.std()
            )
        else:
            sl_smooth = sl_norm

        # 3. 배경 추정 & 제거
        if bg_removal == 'gaussian':
            # Large-kernel Gaussian으로 배경 추정
            bg_sigma = 20.0
            background = ndimage.gaussian_filter(sl_smooth, sigma=bg_sigma)
            sl_corrected = sl_smooth - background

        elif bg_removal == 'morphological':
            # Morphological Opening (rolling-ball 대용)
            # 큰 구조 요소로 배경 추정
            radius = 15
            selem = morphology.disk(radius)
            # Convert to uint8 for cv2 operations
            sl_uint8 = (sl_smooth * 255).astype(np.uint8)
            background = cv2.morphologyEx(sl_uint8, cv2.MORPH_OPEN, selem)
            background = background.astype(np.float32) / 255.0
            sl_corrected = sl_smooth - background

        elif bg_removal == 'tophat':
            # White Top-hat: 구조 요소보다 작은 밝은 구조 추출
            radius = 15
            selem = morphology.disk(radius)
            sl_uint8 = (sl_smooth * 255).astype(np.uint8)
            tophat = cv2.morphologyEx(sl_uint8, cv2.MORPH_TOPHAT, selem)
            sl_corrected = tophat.astype(np.float32) / 255.0
        else:
            sl_corrected = sl_smooth

        # 4. 재정규화 및 클리핑
        sl_corrected = np.clip(sl_corrected, 0, None)
        if sl_corrected.max() > 1e-6:
            sl_corrected = sl_corrected / sl_corrected.max()

        # 5. (선택) 경계 강화 - Unsharp masking
        gaussian_blurred = ndimage.gaussian_filter(sl_corrected, sigma=1.0)
        sl_enhanced = sl_corrected + 0.5 * (sl_corrected - gaussian_blurred)
        sl_enhanced = np.clip(sl_enhanced, 0, 1)

        debug_info = {
            'original_range': (float(sl_min), float(sl_max)),
            'normalized_range': (float(sl_norm.min()), float(sl_norm.max())),
            'corrected_range': (float(sl_corrected.min()), float(sl_corrected.max())),
            'enhanced_range': (float(sl_enhanced.min()), float(sl_enhanced.max()))
        }

        return sl_enhanced, debug_info

    def detect_boundary_contour(self, preprocessed_slice,
                                 threshold_method='edge',
                                 min_contour_area=100):
        """
        경계선(흰색 라인) 검출 및 내부 마스크 생성

        전략: Edge detection으로 뇌의 외곽 경계선만 찾고,
        그 내부는 값과 관계없이 모두 보존

        Args:
            preprocessed_slice: 전처리된 슬라이스
            threshold_method: 경계 검출 방법 ('edge', 'otsu', 'adaptive', 'percentile')
            min_contour_area: 최소 컨투어 면적

        Returns:
            mask: 내부 영역 마스크 (True=보존, False=제거)
            contours: 검출된 컨투어 리스트
        """
        sl_uint8 = (preprocessed_slice * 255).astype(np.uint8)

        if threshold_method == 'edge':
            # Edge detection 전략: 뇌의 외곽 경계선만 검출
            # 1. Canny edge detection
            edges = cv2.Canny(sl_uint8, threshold1=30, threshold2=100)

            # 2. Morphological closing으로 edge 연결
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 3. 구멍 채우기 (flood fill의 역)
            # 작은 구멍들을 채워서 연속된 경계선 만들기
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges_dilated = cv2.dilate(edges_closed, kernel_dilate, iterations=2)

            binary = edges_dilated

        elif threshold_method == 'otsu':
            _, binary = cv2.threshold(
                sl_uint8, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif threshold_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                sl_uint8, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11, C=2
            )
        elif threshold_method == 'percentile':
            # 상위 10%를 경계선으로 간주
            threshold_val = np.percentile(sl_uint8, 90)
            _, binary = cv2.threshold(
                sl_uint8, threshold_val, 255,
                cv2.THRESH_BINARY
            )
        else:
            # 기본값: Edge detection
            edges = cv2.Canny(sl_uint8, threshold1=30, threshold2=100)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            binary = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 2. 컨투어 검출
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,  # 외부 컨투어만
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 3. 가장 큰 컨투어 선택 (보통 뇌 ROI가 가장 큼)
        if len(contours) == 0:
            # 컨투어 없음 -> 전체를 마스크로
            return np.ones(preprocessed_slice.shape, dtype=bool), []

        # 면적 기준으로 필터링
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

        if len(valid_contours) == 0:
            return np.ones(preprocessed_slice.shape, dtype=bool), []

        # 가장 큰 컨투어 선택
        largest_contour = max(valid_contours, key=cv2.contourArea)

        # 4. Convex Hull로 내부를 완전히 채운 마스크 생성
        # Convex hull: 외곽 점들을 연결한 볼록 껍질
        # -> 내부에 구멍이 절대 생기지 않음
        hull = cv2.convexHull(largest_contour)

        mask = np.zeros(preprocessed_slice.shape, dtype=np.uint8)
        cv2.drawContours(mask, [hull], -1, 1, thickness=cv2.FILLED)

        return mask.astype(bool), valid_contours

    def refine_mask(self, mask, closing_radius=5, opening_radius=3):
        """
        마스크 보정: 구멍 메우기 및 매끈화

        Args:
            mask: 원본 마스크
            closing_radius: Closing 연산 반경 (구멍 메우기)
            opening_radius: Opening 연산 반경 (작은 노이즈 제거)

        Returns:
            refined_mask: 보정된 마스크
        """
        # 1. Morphological Closing (구멍 메우기)
        if closing_radius > 0:
            selem_close = morphology.disk(closing_radius)
            mask = morphology.binary_closing(mask, selem_close)

        # 2. Morphological Opening (작은 노이즈 제거)
        if opening_radius > 0:
            selem_open = morphology.disk(opening_radius)
            mask = morphology.binary_opening(mask, selem_open)

        # 3. (선택) Gaussian smoothing으로 경계 부드럽게
        mask_float = mask.astype(np.float32)
        mask_smooth = ndimage.gaussian_filter(mask_float, sigma=2.0)
        mask_refined = mask_smooth > 0.5

        return mask_refined

    def remove_z_noise(self, min_z_start=0.70, gradient_threshold=2.0):
        """
        Z-축 상단 노이즈 제거 (unified_denoise.py의 intensity_gradient_z_removal 통합)

        Mean intensity가 급격히 증가하는 지점(노이즈 경계)을 자동 감지하여
        그 이상의 슬라이스를 그라데이션으로 제거합니다.

        Args:
            min_z_start: 검색 시작 위치 (기본값: 0.70 = 70%부터)
            gradient_threshold: Gradient 임계값 (표준편차 배수)

        Returns:
            detected_boundary: 감지된 노이즈 경계 z-인덱스
        """
        print("\nZ-축 상단 노이즈 제거 중...")

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

        # 노이즈 영역 그라데이션 제거 (경계선부터 0.1, 0.09, 0.08, ..., 0.01, 0.0)
        print(f"  그라데이션 적용: Z={detected_boundary}부터 0.1씩 감소")
        for i, z in enumerate(range(detected_boundary, z_dim)):
            # 0.1에서 시작해서 0.01씩 감소
            multiplier = max(0.0, 0.1 - i * 0.01)
            self.data[:, :, z] *= multiplier
            if i < 5 or multiplier == 0.0:  # 처음 5개와 0이 되는 지점 출력
                print(f"    Z={z}: multiplier={multiplier:.2f}")

        self.metadata['steps'].append({
            'step': 'z_noise_removal',
            'boundary': int(detected_boundary),
            'removed_slices': int(z_dim - detected_boundary),
            'transition_start': int(transition_start),
            'transition_length': transition_length,
            'gradient_threshold': gradient_threshold,
            'method': 'mean_intensity_gradient_detection'
        })

        return detected_boundary

    def remove_horizontal_stripes(self, filter_strength=0.6, iterations=1, aggressive=False):
        """
        수평 스트라이프 제거 (FFT 기반)

        FFT(Fast Fourier Transform)를 사용하여 주파수 도메인에서
        z-축 방향의 수평 스트라이프를 제거합니다.

        각 (x, y) 위치에서 z-축을 따라 1D FFT를 수행하고,
        고주파 성분을 선택적으로 억제하여 스트라이프를 제거합니다.

        Args:
            filter_strength: 필터 강도 (0.0~1.0, 높을수록 강하게 제거)
            iterations: 반복 횟수
            aggressive: True이면 더 공격적으로 제거 (threshold=94, preserve_radius 감소)

        Returns:
            stripe_reduction: 스트라이프 감소율 (%)
        """
        print("\n수평 스트라이프 제거 중...")
        if aggressive:
            print("  ⚡ 공격적 모드 활성화: 더 강력한 스트라이프 제거")

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

                    # 저주파 보존 영역 설정 (더 넓게 보존 → 앨리어싱 방지)
                    center = len(fft_shift) // 2
                    if aggressive:
                        # 공격적 모드: 더 좁은 보존 영역, 더 낮은 threshold
                        preserve_radius = max(5, len(fft_shift) // 15)
                        threshold = np.percentile(magnitude, 94)  # 94th percentile
                    else:
                        # 기본 모드: 균형잡힌 설정
                        preserve_radius = max(5, len(fft_shift) // 12)  # 15→12: 25% 더 보존
                        threshold = np.percentile(magnitude, 96)  # 98→96: 더 공격적

                    # 필터 마스크 생성 (filter_strength 파라미터 사용)
                    filter_mask = np.ones_like(fft_shift, dtype=np.float64)
                    for idx in range(len(fft_shift)):
                        distance = abs(idx - center)
                        if distance > preserve_radius and magnitude[idx] > threshold:
                            # 저주파에 가까울수록 약하게 필터링 (부드러운 전환)
                            suppression = filter_strength
                            if distance < preserve_radius * 1.5:
                                # 전환 영역: 부드럽게 감쇠
                                fade = (distance - preserve_radius) / (preserve_radius * 0.5)
                                suppression = filter_strength * fade
                            filter_mask[idx] = 1.0 - suppression

                    # 필터 적용
                    fft_filtered = fft_shift * filter_mask

                    # 역변환
                    fft_back = np.fft.ifftshift(fft_filtered)
                    filtered = np.fft.ifft(fft_back).real

                    self.data[i, j, :] = filtered

            print(f"  반복 {iter_num + 1}/{iterations} 완료 (처리: {processed}, 스킵: {skipped})")

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
            'aggressive_mode': aggressive,
            'stripe_reduction_percent': float(stripe_reduction)
        })

        return stripe_reduction

    def process_volume_sagittal(self,
                                preprocess_method='gaussian',
                                smooth_sigma=2.0,
                                bg_removal='morphological',
                                threshold_method='edge',
                                min_contour_area=100,
                                refine_closing=5,
                                refine_opening=3,
                                visualize_every=20):
        """
        3D 볼륨을 Sagittal 슬라이스 단위로 처리

        Args:
            preprocess_method: 전처리 스무딩 방법
            smooth_sigma: 스무딩 강도
            bg_removal: 배경 제거 방법
            threshold_method: 이진화 방법
            min_contour_area: 최소 컨투어 면적
            refine_closing: 마스크 보정 closing 반경
            refine_opening: 마스크 보정 opening 반경
            visualize_every: N개 슬라이스마다 시각화
        """
        print("\n" + "=" * 70)
        print("경계선 기반 ROI 마스킹 시작")
        print("Boundary-based ROI Masking")
        print("=" * 70)
        print(f"\n입력 파일: {self.input_path}")
        print(f"데이터 형상: {self.shape}")
        print(f"처리 축: Sagittal (x-axis)")
        print(f"처리할 슬라이스: {self.shape[0]}개")

        print(f"\n전처리 설정:")
        print(f"  스무딩 방법: {preprocess_method} (sigma={smooth_sigma})")
        print(f"  배경 제거: {bg_removal}")
        print(f"  이진화 방법: {threshold_method}")
        print(f"  최소 컨투어 면적: {min_contour_area}")
        print(f"  마스크 보정: closing={refine_closing}, opening={refine_opening}")

        processed_count = 0
        failed_count = 0
        visualization_slices = []

        for x in range(self.shape[0]):
            if x % 10 == 0:
                print(f"  처리 중: {x}/{self.shape[0]} ({x/self.shape[0]*100:.1f}%)...")

            # Sagittal 슬라이스 추출 (y-z plane)
            sagittal_slice = self.data[x, :, :]

            # 빈 슬라이스 스킵
            if sagittal_slice.max() < 1e-6:
                self.mask_3d[x, :, :] = False
                continue

            try:
                # 1. 전처리
                preprocessed, debug_info = self.preprocess_slice(
                    sagittal_slice,
                    method=preprocess_method,
                    smooth_sigma=smooth_sigma,
                    bg_removal=bg_removal
                )

                # 2. 경계선 검출 및 마스크 생성
                mask, contours = self.detect_boundary_contour(
                    preprocessed,
                    threshold_method=threshold_method,
                    min_contour_area=min_contour_area
                )

                # 3. 마스크 보정
                mask_refined = self.refine_mask(
                    mask,
                    closing_radius=refine_closing,
                    opening_radius=refine_opening
                )

                # 4. 3D 마스크에 저장
                self.mask_3d[x, :, :] = mask_refined
                processed_count += 1

                # 5. 시각화용 슬라이스 저장
                if visualize_every > 0 and x % visualize_every == 0:
                    visualization_slices.append({
                        'x': x,
                        'original': sagittal_slice.copy(),
                        'preprocessed': preprocessed.copy(),
                        'mask': mask_refined.copy(),
                        'contours': contours
                    })

            except Exception as e:
                print(f"  경고: 슬라이스 {x} 처리 실패 - {e}")
                self.mask_3d[x, :, :] = True  # 실패하면 보존
                failed_count += 1

        print(f"\n처리 완료!")
        print(f"  성공: {processed_count}개")
        print(f"  실패: {failed_count}개")

        # 마스크 적용
        print("\n마스크 적용 중...")
        self.data[~self.mask_3d] = 0.0

        # 통계
        masked_voxels = np.sum(~self.mask_3d)
        total_voxels = self.mask_3d.size
        masked_percent = (masked_voxels / total_voxels) * 100

        print(f"  마스킹된 복셀: {masked_voxels:,} / {total_voxels:,} ({masked_percent:.2f}%)")

        self.metadata['steps'].append({
            'step': 'boundary_mask_padding',
            'processed_slices': processed_count,
            'failed_slices': failed_count,
            'masked_voxels': int(masked_voxels),
            'masked_percent': float(masked_percent),
            'parameters': {
                'preprocess_method': preprocess_method,
                'smooth_sigma': smooth_sigma,
                'bg_removal': bg_removal,
                'threshold_method': threshold_method,
                'min_contour_area': min_contour_area,
                'refine_closing': refine_closing,
                'refine_opening': refine_opening
            }
        })

        return visualization_slices

    def save_output(self):
        """처리된 데이터 및 마스크 저장"""
        print("\n결과 저장 중...")

        # 1. 처리된 NIfTI 파일 저장
        masked_nii = nib.Nifti1Image(self.data, self.nii.affine, self.nii.header)
        nib.save(masked_nii, self.output_path)
        print(f"  저장됨: {self.output_path}")

        # 2. 마스크 저장
        mask_path = self.output_path.replace('.nii.gz', '_mask.nii.gz')
        mask_nii = nib.Nifti1Image(
            self.mask_3d.astype(np.uint8),
            self.nii.affine,
            self.nii.header
        )
        nib.save(mask_nii, mask_path)
        print(f"  마스크: {mask_path}")

        # 3. 메타데이터 저장
        metadata_path = self.output_path.replace('.nii.gz', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        print(f"  메타데이터: {metadata_path}")

    def generate_visualizations(self, visualization_slices):
        """검증 시각화 생성"""
        if len(visualization_slices) == 0:
            print("\n시각화할 슬라이스가 없습니다.")
            return

        print(f"\n시각화 생성 중... ({len(visualization_slices)}개 슬라이스)")

        # 각 슬라이스마다 4개 서브플롯
        for viz_data in visualization_slices:
            x = viz_data['x']
            original = viz_data['original']
            preprocessed = viz_data['preprocessed']
            mask = viz_data['mask']

            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f'Sagittal Slice X={x} - Boundary Masking', fontsize=14)

            # 1. 원본
            axes[0, 0].imshow(original.T, cmap='gray', origin='lower')
            axes[0, 0].set_title('Original')
            axes[0, 0].axis('off')

            # 2. 전처리
            axes[0, 1].imshow(preprocessed.T, cmap='gray', origin='lower')
            axes[0, 1].set_title('Preprocessed (Enhanced)')
            axes[0, 1].axis('off')

            # 3. 마스크
            axes[1, 0].imshow(mask.T, cmap='gray', origin='lower')
            axes[1, 0].set_title('Detected Mask')
            axes[1, 0].axis('off')

            # 4. 마스크 적용 결과
            masked_result = original * mask
            axes[1, 1].imshow(masked_result.T, cmap='gray', origin='lower')
            axes[1, 1].set_title('Masked Result')
            axes[1, 1].axis('off')

            plt.tight_layout()
            viz_path = self.output_path.replace('.nii.gz', f'_viz_x{x:03d}.png')
            plt.savefig(viz_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  저장됨: {viz_path}")

        # 전체 비교 (3뷰)
        print("\n전체 비교 이미지 생성 중...")
        self._generate_3view_comparison()

    def _generate_3view_comparison(self):
        """3뷰 비교 이미지"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Original vs Masked - 3 View Comparison', fontsize=16, y=0.98)

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

        # 마스크 적용 후
        axes[1, 0].imshow(self.data[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[1, 0].set_title('Masked - Sagittal')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(self.data[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[1, 1].set_title('Masked - Coronal')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(self.data[:, :, mid_z].T, cmap='gray', origin='lower')
        axes[1, 2].set_title('Masked - Axial')
        axes[1, 2].axis('off')

        plt.tight_layout()
        comparison_path = self.output_path.replace('.nii.gz', '_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  저장됨: {comparison_path}")

    def run_full_pipeline(self,
                         preprocess_method='gaussian',
                         smooth_sigma=2.0,
                         bg_removal='morphological',
                         threshold_method='edge',
                         min_contour_area=100,
                         refine_closing=5,
                         refine_opening=3,
                         visualize_every=20,
                         remove_z_noise=True,
                         z_min_start=0.70,
                         z_gradient_threshold=2.0,
                         remove_stripes=True,
                         stripe_filter_strength=0.6,
                         stripe_iterations=1,
                         stripe_aggressive=False):
        """
        전체 파이프라인 실행

        Args:
            preprocess_method: 전처리 스무딩 ('gaussian', 'median', 'nlm')
            smooth_sigma: 스무딩 강도
            bg_removal: 배경 제거 방법 ('gaussian', 'morphological', 'tophat')
            threshold_method: 이진화 방법 ('edge', 'otsu', 'adaptive', 'percentile')
            min_contour_area: 최소 컨투어 면적
            refine_closing: 마스크 보정 closing 반경
            refine_opening: 마스크 보정 opening 반경
            visualize_every: N개 슬라이스마다 시각화 (0이면 시각화 안 함)
            remove_z_noise: Z-축 상단 노이즈 제거 여부 (기본: True)
            z_min_start: Z-축 노이즈 검색 시작 위치 (기본: 0.70)
            z_gradient_threshold: Z-축 gradient 임계값 (기본: 2.0)
            remove_stripes: 수평 스트라이프 제거 여부 (기본: True)
            stripe_filter_strength: 스트라이프 필터 강도 (기본: 0.6)
            stripe_iterations: 스트라이프 제거 반복 횟수 (기본: 1)
            stripe_aggressive: 공격적 스트라이프 제거 모드 (기본: False)
        """
        # 1. Z-축 상단 노이즈 제거 먼저 (unified_denoise.py 방식)
        # 중요: 이걸 먼저 해야 contour 검출 시 노이즈 영역이 포함되지 않음
        z_boundary = None
        if remove_z_noise:
            z_boundary = self.remove_z_noise(
                min_z_start=z_min_start,
                gradient_threshold=z_gradient_threshold
            )

        # 2. 수평 스트라이프 제거 (FFT 기반)
        # Z-노이즈 제거 후, 경계선 마스킹 전에 실행
        stripe_reduction = None
        if remove_stripes:
            stripe_reduction = self.remove_horizontal_stripes(
                filter_strength=stripe_filter_strength,
                iterations=stripe_iterations,
                aggressive=stripe_aggressive
            )

        # 3. Sagittal 슬라이스 단위 처리 (경계선 마스킹)
        # Z-노이즈와 스트라이프가 제거된 상태에서 경계선 검출
        visualization_slices = self.process_volume_sagittal(
            preprocess_method=preprocess_method,
            smooth_sigma=smooth_sigma,
            bg_removal=bg_removal,
            threshold_method=threshold_method,
            min_contour_area=min_contour_area,
            refine_closing=refine_closing,
            refine_opening=refine_opening,
            visualize_every=visualize_every
        )

        # 4. 결과 저장
        self.save_output()

        # 5. 시각화
        if visualize_every > 0:
            self.generate_visualizations(visualization_slices)

        print("\n" + "=" * 70)
        print("모든 작업이 완료되었습니다!")
        print("=" * 70)
        print(f"최종 데이터 범위: [{self.data.min():.3f}, {self.data.max():.3f}]")
        print(f"마스크 통계:")
        print(f"  보존된 복셀: {np.sum(self.mask_3d):,} ({np.sum(self.mask_3d)/self.mask_3d.size*100:.2f}%)")
        print(f"  제거된 복셀: {np.sum(~self.mask_3d):,} ({np.sum(~self.mask_3d)/self.mask_3d.size*100:.2f}%)")
        if remove_z_noise and z_boundary is not None:
            print(f"\nZ-축 노이즈:")
            print(f"  노이즈 경계: Z={z_boundary} ({z_boundary/self.shape[2]*100:.1f}%)")
            print(f"  제거된 슬라이스: {self.shape[2] - z_boundary}개")
        if remove_stripes and stripe_reduction is not None:
            print(f"\n수평 스트라이프:")
            print(f"  감소율: {stripe_reduction:.1f}%")
        print()

        return self.data, self.mask_3d, self.metadata


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
        output_file = input_file.replace('.nii.gz', '_boundary_masked.nii.gz')

    print("\n사용법:")
    print(f"  python {sys.argv[0]} <입력파일.nii.gz> [출력파일.nii.gz]")
    print(f"\n현재 설정:")
    print(f"  입력: {input_file}")
    print(f"  출력: {output_file}")

    # 프로세서 생성 및 실행
    processor = BoundaryMaskProcessor(input_file, output_file)

    # 전체 파이프라인 실행 (기본 파라미터)
    # 필요시 여기서 파라미터를 조정할 수 있습니다
    masked_data, mask_3d, metadata = processor.run_full_pipeline(
        preprocess_method='gaussian',    # 'gaussian', 'median', 'nlm'
        smooth_sigma=2.0,                # 스무딩 강도
        bg_removal='morphological',      # 'gaussian', 'morphological', 'tophat'
        threshold_method='edge',         # 'edge' (권장), 'otsu', 'adaptive', 'percentile'
        min_contour_area=100,            # 최소 컨투어 면적
        refine_closing=5,                # Closing 반경 (구멍 메우기)
        refine_opening=3,                # Opening 반경 (노이즈 제거)
        visualize_every=20,              # 20개마다 시각화 (0이면 시각화 안 함)
        remove_z_noise=True,             # Z-축 상단 노이즈 제거 여부
        z_min_start=0.70,                # Z-축 검색 시작 (70%부터)
        z_gradient_threshold=2.0,        # Gradient 임계값
        remove_stripes=True,             # 수평 스트라이프 제거 여부
        stripe_filter_strength=0.6,      # 스트라이프 필터 강도
        stripe_iterations=1,             # 스트라이프 제거 반복 횟수
        stripe_aggressive=False          # 공격적 스트라이프 제거 모드
    )

    return masked_data, mask_3d, metadata


if __name__ == '__main__':
    main()
