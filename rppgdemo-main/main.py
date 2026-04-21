#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, welch


@dataclass
class ROI:
    name: str
    rect: Tuple[float, float, float, float]  # x, y, w, h in normalized coords
    weight: float


@dataclass
class Config:
    # v2: bandpass lower bound raised from 0.6 to 0.75 Hz.
    # Reason: poly2 detrend + 0.6 Hz passband produced a spurious 0.75 Hz peak
    # in PSD that dominated the true HR peak. Raising to 0.75 Hz eliminates it.
    # See diagnostic: raw green PSD has correct peak at #2, but after poly2
    # detrend the 0.75 Hz artifact becomes #1 and pushes correct peak to #3.
    low_hz: float = 0.75
    high_hz: float = 4.75
    filter_order: int = 6
    min_bpm_valid: float = 45.0
    max_bpm_valid: float = 170.0
    clamp_min_bpm: float = 40.0
    clamp_max_bpm: float = 160.0
    welch_seg_sec: float = 4.5
    welch_overlap: float = 0.8
    nfft: int = 2048
    pos_window_sec: float = 1.8
    buffer_sec: float = 11.0
    enable_ppi_assist: bool = True
    cluster_gap_bpm: float = 10.0
    temporal_sigma_bpm: float = 24.0
    high_hr_mode_threshold: float = 95.0
    high_hr_cluster_boost: float = 0.19
    freq_conf_gate: float = 0.40
    roi_preset: str = "hybrid7"
    roi_scale_x: float = 1.1
    roi_scale_y: float = 1.1
    roi_shift_y: float = 0.0
    # Experimental: disabled by default to avoid cross-scene regressions.
    enable_adaptive_roi: bool = False
    adaptive_roi_debug: bool = False
    adaptive_roi_window_sec: int = 8
    adaptive_roi_low_hr_bpm: float = 82.0
    adaptive_roi_low_candidate_bpm: float = 85.0
    adaptive_roi_high_candidate_bpm: float = 100.0
    adaptive_roi_min_lock_secs: int = 6
    adaptive_roi_min_very_low_secs: int = 3
    adaptive_roi_very_low_hr_bpm: float = 72.0
    adaptive_roi_min_bimodal_secs: int = 4
    adaptive_roi_min_freq_conf: float = 0.62
    adaptive_roi_max_freq_conf: float = 0.82
    adaptive_roi_max_sqi_for_lock: float = 0.58
    adaptive_roi_min_ppi_gap: float = 10.0
    adaptive_roi_hold_sec: float = 12.0
    adaptive_roi_max_fallback_sec: float = 35.0
    adaptive_roi_cooldown_sec: float = 20.0
    adaptive_roi_fallback_preset: str = "classic4"
    adaptive_roi_fallback_scale_x: float = 1.0
    adaptive_roi_fallback_scale_y: float = 1.0
    adaptive_roi_fallback_shift_y: float = 0.0
    low_lock_rescue_enabled: bool = False
    low_lock_rescue_sqi_max: float = 0.58
    low_lock_rescue_hr_max: float = 92.0
    low_lock_rescue_low_candidate_bpm: float = 86.0
    low_lock_rescue_high_candidate_bpm: float = 100.0
    low_lock_rescue_min_high_count: int = 3
    low_lock_rescue_min_hit_count: int = 2
    low_lock_rescue_window: int = 6
    low_lock_rescue_lift_ratio: float = 0.65
    low_lock_rescue_max_lift_bpm: float = 24.0
    use_cbcr_candidate: bool = False
    fusion_support_weight: float = 0.55
    fusion_temporal_weight: float = 0.30
    fusion_stability_weight: float = 0.15
    quality_snr_weight: float = 0.30
    quality_stability_weight: float = 0.25
    quality_corr_weight: float = 0.25
    quality_periodicity_weight: float = 0.20
    quality_snr_norm_div: float = 15.0
    quality_signal_strength_min: float = 0.003
    ppi_gate_sqi_min: float = 0.55
    ppi_gate_std_max: float = 14.0
    ppi_gate_diff_min: float = 21.0
    ppi_gate_ratio_min: float = 1.20
    ppi_gate_ratio_max: float = 2.30
    ppi_blend_tree_self: float = 0.05
    ppi_blend_tree_ppi: float = 0.95
    ppi_blend_weak_self: float = 0.15
    ppi_blend_weak_ppi: float = 0.85
    ppi_blend_strong_self: float = 0.30
    ppi_blend_strong_ppi: float = 0.70
    peak_threshold_k: float = 0.5
    peak_window_divisor: float = 15.0
    peak_missing_threshold_scale: float = 0.7
    peak_min_distance_hz: float = 3.0
    peak_max_distance_hz: float = 0.67
    publish_quality_sqi_weight: float = 0.70
    publish_quality_freq_weight: float = 0.30
    publish_quality_low_freq_penalty: float = 0.75
    publish_conf_sqi_weight: float = 0.45
    publish_conf_freq_weight: float = 0.35
    publish_conf_bias: float = 0.20
    publish_min_freq_conf_for_output: float = 0.0
    publish_min_sqi_for_output: float = 0.0
    high_hr_bias_threshold: float = 95.0
    high_hr_bias_gain: float = 1.0
    high_hr_bias_offset: float = 5.0
    output_drop_guard_enabled: bool = True
    output_drop_guard_ref_min_bpm: float = 102.0
    output_drop_guard_out_max_bpm: float = 86.0
    output_drop_guard_ratio_max: float = 0.78
    output_drop_guard_freq_conf_min: float = 0.60
    output_drop_guard_sqi_min: float = 0.55
    output_drop_guard_floor_ratio: float = 0.88
    output_drop_guard_floor_delta: float = 15.0


class CameraSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def stop(self) -> None:
        pass


class OpenCVCamera(CameraSource):
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open webcam index={index}")

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def stop(self) -> None:
        self.cap.release()


class RealSenseCamera(CameraSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 60):
        try:
            import pyrealsense2 as rs
        except Exception as e:
            raise RuntimeError("pyrealsense2 not installed") from e

        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(cfg)

        # Lock exposure/gain like iOS camera lock strategy.
        dev = self.profile.get_device()
        sensor = dev.first_color_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, 6000)
        sensor.set_option(rs.option.gain, 16)

    def read(self) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            return None
        return np.asanyarray(color.get_data())

    def stop(self) -> None:
        self.pipeline.stop()


class Picamera2Camera(CameraSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 60):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={"FrameRate": fps},
        )
        self.cam.configure(config)
        self.cam.start()
        # Let AE/AWB converge, then lock all auto controls for rPPG stability.
        import time as _time
        _time.sleep(1.5)
        metadata = self.cam.capture_metadata()
        locked_exposure = metadata.get("ExposureTime", 20000)
        locked_gain = metadata.get("AnalogueGain", 2.0)
        locked_awb = metadata.get("ColourGains", (0.0, 0.0))
        self.cam.set_controls({
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": locked_exposure,
            "AnalogueGain": locked_gain,
            "ColourGains": locked_awb,
        })

    def read(self) -> Optional[np.ndarray]:
        frame = self.cam.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def stop(self) -> None:
        self.cam.stop()
        self.cam.close()


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _clip_rect01(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x = _clamp01(x)
    y = _clamp01(y)
    w = max(1e-4, min(w, 1.0 - x))
    h = max(1e-4, min(h, 1.0 - y))
    return x, y, w, h


def _roi_specs(preset: str) -> List[Tuple[str, float, float, float, float, float]]:
    # Relative coordinates in face bbox.
    if preset == "hybrid7":
        return [
            ("forehead_center", 0.40, 0.10, 0.20, 0.14, 1.00),
            ("forehead_left", 0.20, 0.10, 0.18, 0.14, 0.95),
            ("forehead_right", 0.62, 0.10, 0.18, 0.14, 0.95),
            ("glabella", 0.44, 0.27, 0.12, 0.12, 1.00),
            ("upper_nasal", 0.45, 0.40, 0.10, 0.13, 0.92),
            ("right_malar", 0.19, 0.47, 0.19, 0.16, 0.90),
            ("left_malar", 0.62, 0.47, 0.19, 0.16, 0.90),
        ]
    if preset == "cheek_forehead6":
        return [
            ("forehead_center", 0.38, 0.11, 0.24, 0.14, 1.00),
            ("forehead_left", 0.19, 0.12, 0.20, 0.14, 0.95),
            ("forehead_right", 0.61, 0.12, 0.20, 0.14, 0.95),
            ("upper_nasal", 0.45, 0.39, 0.10, 0.14, 0.92),
            ("right_malar", 0.18, 0.47, 0.20, 0.17, 0.90),
            ("left_malar", 0.62, 0.47, 0.20, 0.17, 0.90),
        ]
    if preset == "whole_face":
        return [
            ("whole_face", 0.12, 0.08, 0.76, 0.68, 0.85),
            ("forehead_center", 0.38, 0.11, 0.24, 0.14, 1.00),
            ("right_malar", 0.18, 0.47, 0.20, 0.17, 0.90),
            ("left_malar", 0.62, 0.47, 0.20, 0.17, 0.90),
        ]
    # classic4 (legacy iOS-mapped ROI template)
    return [
        ("forehead", 0.25, 0.08, 0.50, 0.14, 1.00),
        ("glabella", 0.46, 0.30, 0.08, 0.10, 0.95),
        ("right_malar", 0.20, 0.45, 0.16, 0.14, 0.90),
        ("left_malar", 0.64, 0.45, 0.16, 0.14, 0.90),
    ]


def build_rois_for_face(
    x_min: float,
    y_min: float,
    face_w: float,
    face_h: float,
    preset: str,
    scale_x: float,
    scale_y: float,
    shift_y: float,
) -> List[ROI]:
    out: List[ROI] = []
    sx = max(0.6, min(1.6, scale_x))
    sy = max(0.6, min(1.6, scale_y))
    dy = max(-0.12, min(0.12, shift_y))
    for name, rx, ry, rw, rh, wt in _roi_specs(preset):
        cx = rx + 0.5 * rw
        cy = ry + 0.5 * rh + dy
        rw2 = rw * sx
        rh2 = rh * sy
        rx2 = cx - 0.5 * rw2
        ry2 = cy - 0.5 * rh2
        rx2, ry2, rw2, rh2 = _clip_rect01(rx2, ry2, rw2, rh2)

        ax = x_min + face_w * rx2
        ay = y_min + face_h * ry2
        aw = face_w * rw2
        ah = face_h * rh2
        ax, ay, aw, ah = _clip_rect01(ax, ay, aw, ah)
        out.append(ROI(name, (ax, ay, aw, ah), wt))
    return out


class MPSolutionsFaceROIExtractor:
    def __init__(
        self,
        roi_preset: str = "classic4",
        roi_scale_x: float = 1.0,
        roi_scale_y: float = 1.0,
        roi_shift_y: float = 0.0,
    ):
        import mediapipe as mp

        self.mp = mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.roi_history: Deque[Dict[str, ROI]] = deque(maxlen=5)
        self.roi_preset = roi_preset
        self.roi_scale_x = roi_scale_x
        self.roi_scale_y = roi_scale_y
        self.roi_shift_y = roi_shift_y

    def set_roi_profile(self, roi_preset: str, roi_scale_x: float, roi_scale_y: float, roi_shift_y: float) -> None:
        self.roi_preset = roi_preset
        self.roi_scale_x = roi_scale_x
        self.roi_scale_y = roi_scale_y
        self.roi_shift_y = roi_shift_y
        self.roi_history.clear()

    def _skin_ratio(self, frame: np.ndarray, rect: Tuple[float, float, float, float]) -> float:
        h, w = frame.shape[:2]
        x, y, rw, rh = rect
        x0 = max(0, min(int(x * w), w - 1))
        y0 = max(0, min(int(y * h), h - 1))
        x1 = max(x0 + 1, min(int((x + rw) * w), w))
        y1 = max(y0 + 1, min(int((y + rh) * h), h))
        crop = frame[y0:y1:3, x0:x1:3]
        if crop.size == 0:
            return 0.0

        b = crop[..., 0].astype(np.float32)
        g = crop[..., 1].astype(np.float32)
        r = crop[..., 2].astype(np.float32)
        bright = 0.299 * r + 0.587 * g + 0.114 * b
        skin = (
            (r >= 95)
            & (r <= 255)
            & (g >= 40)
            & (g <= 240)
            & (b >= 20)
            & (b <= 200)
            & (r > g)
            & (r > b)
            & (g > 0.8 * b)
            & (bright >= 60)
            & (bright <= 240)
        )
        return float(np.mean(skin))

    @staticmethod
    def _ema_rect(prev: Tuple[float, float, float, float], cur: Tuple[float, float, float, float], alpha: float = 0.4):
        return tuple((1.0 - alpha) * p + alpha * c for p, c in zip(prev, cur))

    def extract(self, frame: np.ndarray) -> List[ROI]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        if not res.multi_face_landmarks:
            return []

        lm = res.multi_face_landmarks[0].landmark
        xs = np.array([p.x for p in lm])
        ys = np.array([p.y for p in lm])

        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        fw = max(1e-4, x_max - x_min)
        fh = max(1e-4, y_max - y_min)

        rois = build_rois_for_face(
            x_min,
            y_min,
            fw,
            fh,
            preset=self.roi_preset,
            scale_x=self.roi_scale_x,
            scale_y=self.roi_scale_y,
            shift_y=self.roi_shift_y,
        )

        cur = {r.name: r for r in rois}
        if self.roi_history:
            prev = self.roi_history[-1]
            smoothed = {}
            for name, roi in cur.items():
                if name in prev:
                    srect = self._ema_rect(prev[name].rect, roi.rect, alpha=0.4)
                    smoothed[name] = ROI(name, srect, roi.weight)
                else:
                    smoothed[name] = roi
            cur = smoothed

        # Skin gating: >=0.3 keep, 0.15~0.3 reduce weight, <0.15 drop.
        out: List[ROI] = []
        for roi in cur.values():
            ratio = self._skin_ratio(frame, roi.rect)
            if ratio < 0.15:
                continue
            if ratio < 0.3:
                out.append(ROI(roi.name, roi.rect, max(0.1, roi.weight * (ratio / 0.3))))
            else:
                out.append(roi)

        self.roi_history.append({r.name: r for r in out})
        return out


class MPTasksFaceROIExtractor:
    """MediaPipe tasks-only fallback for environments without mp.solutions."""

    def __init__(
        self,
        detector_model_path: str,
        roi_preset: str = "classic4",
        roi_scale_x: float = 1.0,
        roi_scale_y: float = 1.0,
        roi_shift_y: float = 0.0,
    ):
        if not detector_model_path:
            raise RuntimeError(
                "mediapipe tasks mode requires --mp-face-detector-model path "
                "(e.g. blaze face detector .tflite)"
            )
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self.mp = mp
        self._vision = vision
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path=detector_model_path,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
            min_suppression_threshold=0.3,
        )
        self.detector = vision.FaceDetector.create_from_options(options)
        self.roi_history: Deque[Dict[str, ROI]] = deque(maxlen=5)
        self.roi_preset = roi_preset
        self.roi_scale_x = roi_scale_x
        self.roi_scale_y = roi_scale_y
        self.roi_shift_y = roi_shift_y

    def set_roi_profile(self, roi_preset: str, roi_scale_x: float, roi_scale_y: float, roi_shift_y: float) -> None:
        self.roi_preset = roi_preset
        self.roi_scale_x = roi_scale_x
        self.roi_scale_y = roi_scale_y
        self.roi_shift_y = roi_shift_y
        self.roi_history.clear()

    def _skin_ratio(self, frame: np.ndarray, rect: Tuple[float, float, float, float]) -> float:
        h, w = frame.shape[:2]
        x, y, rw, rh = rect
        x0 = max(0, min(int(x * w), w - 1))
        y0 = max(0, min(int(y * h), h - 1))
        x1 = max(x0 + 1, min(int((x + rw) * w), w))
        y1 = max(y0 + 1, min(int((y + rh) * h), h))
        crop = frame[y0:y1:3, x0:x1:3]
        if crop.size == 0:
            return 0.0

        b = crop[..., 0].astype(np.float32)
        g = crop[..., 1].astype(np.float32)
        r = crop[..., 2].astype(np.float32)
        bright = 0.299 * r + 0.587 * g + 0.114 * b
        skin = (
            (r >= 95)
            & (r <= 255)
            & (g >= 40)
            & (g <= 240)
            & (b >= 20)
            & (b <= 200)
            & (r > g)
            & (r > b)
            & (g > 0.8 * b)
            & (bright >= 60)
            & (bright <= 240)
        )
        return float(np.mean(skin))

    @staticmethod
    def _ema_rect(prev: Tuple[float, float, float, float], cur: Tuple[float, float, float, float], alpha: float = 0.4):
        return tuple((1.0 - alpha) * p + alpha * c for p, c in zip(prev, cur))

    def extract(self, frame: np.ndarray) -> List[ROI]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.detections:
            return []

        # pick the largest face bbox
        best = None
        best_area = -1.0
        fh, fw = frame.shape[:2]
        for det in result.detections:
            bb = det.bounding_box
            area = float(max(0, bb.width) * max(0, bb.height))
            if area > best_area:
                best_area = area
                best = bb
        if best is None:
            return []

        x_min = max(0.0, min(1.0, best.origin_x / fw))
        y_min = max(0.0, min(1.0, best.origin_y / fh))
        face_w = max(1e-4, min(1.0 - x_min, best.width / fw))
        face_h = max(1e-4, min(1.0 - y_min, best.height / fh))

        rois = build_rois_for_face(
            x_min,
            y_min,
            face_w,
            face_h,
            preset=self.roi_preset,
            scale_x=self.roi_scale_x,
            scale_y=self.roi_scale_y,
            shift_y=self.roi_shift_y,
        )
        cur = {r.name: r for r in rois}

        if self.roi_history:
            prev = self.roi_history[-1]
            smoothed = {}
            for name, roi in cur.items():
                if name in prev:
                    srect = self._ema_rect(prev[name].rect, roi.rect, alpha=0.4)
                    smoothed[name] = ROI(name, srect, roi.weight)
                else:
                    smoothed[name] = roi
            cur = smoothed

        out: List[ROI] = []
        for roi in cur.values():
            ratio = self._skin_ratio(frame, roi.rect)
            if ratio < 0.15:
                continue
            if ratio < 0.3:
                out.append(ROI(roi.name, roi.rect, max(0.1, roi.weight * (ratio / 0.3))))
            else:
                out.append(roi)
        self.roi_history.append({r.name: r for r in out})
        return out


class OpenCVFaceROIExtractor:
    """Fallback ROI extractor that does not require mediapipe."""

    def __init__(
        self,
        roi_preset: str = "classic4",
        roi_scale_x: float = 1.0,
        roi_scale_y: float = 1.0,
        roi_shift_y: float = 0.0,
    ):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"failed to load Haar cascade: {cascade_path}")
        self.last_face: Optional[Tuple[int, int, int, int]] = None
        self.roi_history: Deque[Dict[str, ROI]] = deque(maxlen=5)
        self.frame_idx: int = 0
        self.detect_interval: int = 5
        self.roi_preset = roi_preset
        self.roi_scale_x = roi_scale_x
        self.roi_scale_y = roi_scale_y
        self.roi_shift_y = roi_shift_y

    def set_roi_profile(self, roi_preset: str, roi_scale_x: float, roi_scale_y: float, roi_shift_y: float) -> None:
        self.roi_preset = roi_preset
        self.roi_scale_x = roi_scale_x
        self.roi_scale_y = roi_scale_y
        self.roi_shift_y = roi_shift_y
        self.roi_history.clear()

    @staticmethod
    def _clamp01(v: float) -> float:
        return max(0.0, min(1.0, v))

    def _skin_ratio(self, frame: np.ndarray, rect: Tuple[float, float, float, float]) -> float:
        h, w = frame.shape[:2]
        x, y, rw, rh = rect
        x0 = max(0, min(int(x * w), w - 1))
        y0 = max(0, min(int(y * h), h - 1))
        x1 = max(x0 + 1, min(int((x + rw) * w), w))
        y1 = max(y0 + 1, min(int((y + rh) * h), h))
        crop = frame[y0:y1:3, x0:x1:3]
        if crop.size == 0:
            return 0.0

        b = crop[..., 0].astype(np.float32)
        g = crop[..., 1].astype(np.float32)
        r = crop[..., 2].astype(np.float32)
        bright = 0.299 * r + 0.587 * g + 0.114 * b
        skin = (
            (r >= 95)
            & (r <= 255)
            & (g >= 40)
            & (g <= 240)
            & (b >= 20)
            & (b <= 200)
            & (r > g)
            & (r > b)
            & (g > 0.8 * b)
            & (bright >= 60)
            & (bright <= 240)
        )
        return float(np.mean(skin))

    @staticmethod
    def _ema_rect(prev: Tuple[float, float, float, float], cur: Tuple[float, float, float, float], alpha: float = 0.4):
        return tuple((1.0 - alpha) * p + alpha * c for p, c in zip(prev, cur))

    def _detect_face(self, frame: np.ndarray, run_detect: bool) -> Optional[Tuple[int, int, int, int]]:
        if run_detect or self.last_face is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                areas = [fw * fh for (_, _, fw, fh) in faces]
                self.last_face = tuple(faces[int(np.argmax(areas))])
        return self.last_face

    def extract(self, frame: np.ndarray) -> List[ROI]:
        face = self._detect_face(frame, run_detect=(self.frame_idx % self.detect_interval == 0))
        self.frame_idx += 1
        if face is None:
            return []

        h, w = frame.shape[:2]
        fx, fy, fw, fh = face
        x_min = self._clamp01(fx / w)
        y_min = self._clamp01(fy / h)
        fwn = self._clamp01(fw / w)
        fhn = self._clamp01(fh / h)

        rois = build_rois_for_face(
            x_min,
            y_min,
            fwn,
            fhn,
            preset=self.roi_preset,
            scale_x=self.roi_scale_x,
            scale_y=self.roi_scale_y,
            shift_y=self.roi_shift_y,
        )
        cur = {r.name: r for r in rois}

        if self.roi_history:
            prev = self.roi_history[-1]
            for name in list(cur.keys()):
                if name in prev:
                    srect = self._ema_rect(prev[name].rect, cur[name].rect, alpha=0.4)
                    cur[name] = ROI(name, srect, cur[name].weight)

        out: List[ROI] = []
        for roi in cur.values():
            ratio = self._skin_ratio(frame, roi.rect)
            if ratio < 0.15:
                continue
            if ratio < 0.3:
                out.append(ROI(roi.name, roi.rect, max(0.1, roi.weight * (ratio / 0.3))))
            else:
                out.append(roi)

        self.roi_history.append({r.name: r for r in out})
        return out


def create_roi_extractor(
    mode: str,
    mp_face_detector_model: str = "",
    strict: bool = False,
    roi_preset: str = "classic4",
    roi_scale_x: float = 1.0,
    roi_scale_y: float = 1.0,
    roi_shift_y: float = 0.0,
):
    def _create_mediapipe_extractor():
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            return (
                MPSolutionsFaceROIExtractor(
                    roi_preset=roi_preset,
                    roi_scale_x=roi_scale_x,
                    roi_scale_y=roi_scale_y,
                    roi_shift_y=roi_shift_y,
                ),
                "mediapipe-solutions",
            )
        # tasks-only mediapipe (common on py3.12 builds)
        return (
            MPTasksFaceROIExtractor(
                mp_face_detector_model,
                roi_preset=roi_preset,
                roi_scale_x=roi_scale_x,
                roi_scale_y=roi_scale_y,
                roi_shift_y=roi_shift_y,
            ),
            "mediapipe-tasks",
        )

    if mode == "mediapipe":
        try:
            return _create_mediapipe_extractor()
        except Exception as e:
            if strict:
                raise
            print(f"[ROI] mediapipe unavailable ({type(e).__name__}: {e}); fallback -> opencv")
            return (
                OpenCVFaceROIExtractor(
                    roi_preset=roi_preset,
                    roi_scale_x=roi_scale_x,
                    roi_scale_y=roi_scale_y,
                    roi_shift_y=roi_shift_y,
                ),
                "opencv",
            )
    if mode == "opencv":
        return (
            OpenCVFaceROIExtractor(
                roi_preset=roi_preset,
                roi_scale_x=roi_scale_x,
                roi_scale_y=roi_scale_y,
                roi_shift_y=roi_shift_y,
            ),
            "opencv",
        )

    # auto: try mediapipe first, fallback to opencv.
    try:
        return _create_mediapipe_extractor()
    except Exception as e:
        print(f"[ROI] auto fallback: mediapipe unavailable ({type(e).__name__}: {e})")
        return (
            OpenCVFaceROIExtractor(
                roi_preset=roi_preset,
                roi_scale_x=roi_scale_x,
                roi_scale_y=roi_scale_y,
                roi_shift_y=roi_shift_y,
            ),
            "opencv",
        )


def extract_rgb_mean(frame: np.ndarray, roi: ROI) -> Optional[Tuple[float, float, float]]:
    h, w = frame.shape[:2]
    x, y, rw, rh = roi.rect
    x0 = max(0, min(int(x * w), w - 1))
    y0 = max(0, min(int(y * h), h - 1))
    x1 = max(x0 + 1, min(int((x + rw) * w), w))
    y1 = max(y0 + 1, min(int((y + rh) * h), h))

    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    b = crop[..., 0]
    g = crop[..., 1]
    r = crop[..., 2]

    valid = (
        (r > 55) & (r < 220) &
        (g > 55) & (g < 220) &
        (b > 55) & (b < 220) &
        (r < 245) & (g < 245) & (b < 245)
    )
    if float(np.mean(valid)) < 0.60:
        return None

    rv = float(np.mean(r[valid]))
    gv = float(np.mean(g[valid]))
    bv = float(np.mean(b[valid]))
    return rv, gv, bv


class AdaptiveROIController:
    """Lightweight runtime switch for low-frequency lock under OpenCV ROI mode."""

    def __init__(self, cfg: Config, extractor: object):
        self.cfg = cfg
        self.extractor = extractor
        self.history: Deque[Dict[str, float]] = deque(maxlen=max(4, int(cfg.adaptive_roi_window_sec)))
        self.enabled = (
            cfg.enable_adaptive_roi
            and isinstance(extractor, OpenCVFaceROIExtractor)
            and hasattr(extractor, "set_roi_profile")
            and extractor.roi_preset == "hybrid7"
        )
        self.primary = (
            extractor.roi_preset,
            extractor.roi_scale_x,
            extractor.roi_scale_y,
            extractor.roi_shift_y,
        )
        self.fallback = (
            cfg.adaptive_roi_fallback_preset,
            cfg.adaptive_roi_fallback_scale_x,
            cfg.adaptive_roi_fallback_scale_y,
            cfg.adaptive_roi_fallback_shift_y,
        )
        self.active_profile = "primary"
        self.fallback_since = -1e9
        self.cooldown_until = -1e9

    def _switch(self, now_sec: float, profile_name: str, profile: Tuple[str, float, float, float], reason: str) -> Optional[str]:
        preset, sx, sy, shy = profile
        self.extractor.set_roi_profile(preset, sx, sy, shy)
        self.history.clear()
        self.active_profile = profile_name
        if profile_name == "fallback":
            self.fallback_since = now_sec
            return (
                f"switch -> fallback preset={preset} sx={sx:.2f} sy={sy:.2f} "
                f"shift={shy:.2f} reason={reason}"
            )
        self.cooldown_until = now_sec + self.cfg.adaptive_roi_cooldown_sec
        return f"switch -> primary reason={reason}"

    def _lock_detected(self) -> bool:
        if len(self.history) < self.history.maxlen:
            return False
        recent = list(self.history)
        low_lock_secs = sum(1 for x in recent if x["hr_raw"] < self.cfg.adaptive_roi_low_hr_bpm)
        very_low_secs = sum(1 for x in recent if x["hr_raw"] < self.cfg.adaptive_roi_very_low_hr_bpm)
        bimodal_secs = sum(
            1
            for x in recent
            if x["low_cnt"] >= 8 and x["high_cnt"] >= 2
        )
        ppi_gaps = [x["ppi_gap"] for x in recent if math.isfinite(x["ppi_gap"])]
        if len(ppi_gaps) < max(4, len(recent) // 2):
            return False
        ppi_gap_med = float(np.median(np.array(ppi_gaps, dtype=np.float64)))
        freq_conf_med = float(np.median(np.array([x["freq_conf"] for x in recent], dtype=np.float64)))
        sqis = [x["sqi"] for x in recent if math.isfinite(x["sqi"])]
        sqi_med = float(np.median(np.array(sqis, dtype=np.float64))) if sqis else 1.0
        return (
            low_lock_secs >= self.cfg.adaptive_roi_min_lock_secs
            and very_low_secs >= self.cfg.adaptive_roi_min_very_low_secs
            and bimodal_secs >= self.cfg.adaptive_roi_min_bimodal_secs
            and ppi_gap_med >= self.cfg.adaptive_roi_min_ppi_gap
            and freq_conf_med >= self.cfg.adaptive_roi_min_freq_conf
            and freq_conf_med <= self.cfg.adaptive_roi_max_freq_conf
            and sqi_med <= self.cfg.adaptive_roi_max_sqi_for_lock
        )

    def _lock_cleared(self) -> bool:
        if not self.history:
            return False
        window = min(len(self.history), 6)
        recent = list(self.history)[-window:]
        low_lock_secs = sum(1 for x in recent if x["hr_raw"] < self.cfg.adaptive_roi_low_hr_bpm)
        bimodal_secs = sum(1 for x in recent if x["low_cnt"] >= 8 and x["high_cnt"] >= 2)
        ppi_gaps = [x["ppi_gap"] for x in recent if math.isfinite(x["ppi_gap"])]
        ppi_gap_med = float(np.median(np.array(ppi_gaps, dtype=np.float64))) if ppi_gaps else 0.0
        return low_lock_secs <= 2 or (bimodal_secs <= 1 and ppi_gap_med < 6.0)

    def update(
        self,
        now_sec: float,
        tagged: List[Tuple[float, str, str]],
        hr_raw: float,
        ppi_hr: Optional[float],
        freq_conf: float,
        sqi: Optional[float] = None,
    ) -> Optional[str]:
        if not self.enabled or not tagged:
            return None

        low_cnt = float(sum(1 for v, _, _ in tagged if v < self.cfg.adaptive_roi_low_candidate_bpm))
        high_cnt = float(sum(1 for v, _, _ in tagged if v > self.cfg.adaptive_roi_high_candidate_bpm))
        ppi_gap = float(ppi_hr - hr_raw) if ppi_hr is not None else float("nan")
        sqi_v = float(sqi) if (sqi is not None and math.isfinite(float(sqi))) else float("nan")
        self.history.append(
            {
                "hr_raw": float(hr_raw),
                "freq_conf": float(freq_conf),
                "sqi": sqi_v,
                "low_cnt": low_cnt,
                "high_cnt": high_cnt,
                "ppi_gap": ppi_gap,
            }
        )

        if self.active_profile == "primary":
            if now_sec < self.cooldown_until:
                return None
            if self._lock_detected():
                return self._switch(now_sec, "fallback", self.fallback, reason="low_lock")
            return None

        # fallback mode
        fallback_age = now_sec - self.fallback_since
        if fallback_age < self.cfg.adaptive_roi_hold_sec:
            return None
        if fallback_age >= self.cfg.adaptive_roi_max_fallback_sec:
            return self._switch(now_sec, "primary", self.primary, reason="fallback_timeout")
        if self._lock_cleared():
            return self._switch(now_sec, "primary", self.primary, reason="lock_cleared")
        return None


class RPPGAlgorithms:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def bandpass(self, x: np.ndarray, fs: float) -> np.ndarray:
        if len(x) < 16:
            return x
        n = max(1, self.cfg.filter_order // 2)
        b, a = butter(n, [self.cfg.low_hz, self.cfg.high_hz], btype="band", fs=fs)
        pad = min(3 * self.cfg.filter_order, len(x) - 1)
        if pad < 1:
            return x
        return filtfilt(b, a, x, padlen=pad)

    @staticmethod
    def detrend_poly2(x: np.ndarray) -> np.ndarray:
        t = np.arange(len(x), dtype=np.float64)
        p = np.polyfit(t, x, 2)
        return x - np.polyval(p, t)

    @staticmethod
    def robust_norm(x: np.ndarray) -> np.ndarray:
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-9
        return (x - med) / (mad * 1.4826)

    @staticmethod
    def _parabolic_peak_interpolation(left: float, peak: float, right: float, bin_freq: float, freq_resolution: float) -> float:
        # Match iOS helper in RPPGAlgorithmSupport.swift.
        denom = 2.0 * peak - left - right
        if denom <= 1e-10:
            return bin_freq
        delta = 0.5 * (right - left) / denom
        if abs(delta) > 0.5:
            return bin_freq
        return bin_freq + delta * freq_resolution

    @staticmethod
    def _calculate_physiological_score(heart_rate: float) -> float:
        if 60.0 <= heart_rate <= 120.0:
            return 1.0
        if 50.0 <= heart_rate < 60.0:
            return 0.95
        if 45.0 <= heart_rate <= 50.0:
            return 0.80
        if 40.0 <= heart_rate <= 45.0:
            return 0.50
        if 120.0 <= heart_rate <= 140.0:
            return 0.50
        if 140.0 <= heart_rate <= 160.0:
            return 0.15
        if 160.0 <= heart_rate <= 180.0:
            return 0.20
        return 0.10

    @staticmethod
    def _psd_harmonic_evidence(
        psd: np.ndarray,
        frequencies: np.ndarray,
        target_freq: float,
        ref_power: float,
        noise_threshold: float,
        freq_res: float,
    ) -> float:
        if freq_res <= 0.0 or target_freq <= 0.0:
            return 0.0
        target_idx = int(round(target_freq / freq_res))
        lo = max(0, target_idx - 2)
        hi = min(len(psd) - 1, target_idx + 2)
        if lo > hi or hi >= len(psd):
            return 0.0
        max_power = float(np.max(psd[lo:hi + 1]))
        if max_power <= noise_threshold:
            return 0.0
        return min(1.0, max_power / ref_power) if ref_power > 0 else 0.0

    def welch_hr(self, x: np.ndarray, fs: float) -> Tuple[float, float]:
        nperseg = max(8, int(self.cfg.welch_seg_sec * fs))
        noverlap = int(nperseg * self.cfg.welch_overlap)
        freqs, psd = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=self.cfg.nfft)
        if len(freqs) == 0 or len(psd) == 0:
            return 0.0, 0.0

        min_freq = self.cfg.low_hz
        max_freq = 2.5
        candidates: List[Tuple[float, float, float]] = []  # (freq, power, score)

        freq_res = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0.0
        for i in range(1, len(psd) - 1):
            freq = float(freqs[i])
            if not (min_freq <= freq <= max_freq):
                continue
            if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
                refined = self._parabolic_peak_interpolation(
                    float(psd[i - 1]),
                    float(psd[i]),
                    float(psd[i + 1]),
                    freq,
                    freq_res,
                )
                heart_rate = refined * 60.0
                physio = self._calculate_physiological_score(heart_rate)
                power = float(psd[i])
                candidates.append((refined, power, physio * power))

        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)

            # Match iOS asymmetric harmonic scoring.
            if len(candidates) >= 2:
                in_band = psd[(freqs >= min_freq) & (freqs <= max_freq)]
                median_noise = float(np.median(in_band)) if len(in_band) > 0 else 0.0
                top_count = min(len(candidates), 8)
                updated: List[Tuple[float, float, float]] = []
                for i, c in enumerate(candidates):
                    if i >= top_count:
                        updated.append(c)
                        continue
                    f, pwr, score = c
                    h_score = 0.0
                    ev2f = self._psd_harmonic_evidence(
                        psd, freqs, f * 2.0, pwr, median_noise * 2.0, freq_res
                    )
                    if 0.0 < ev2f < 0.6:
                        h_score += 0.4
                    if f * 0.5 >= min_freq:
                        ev05f = self._psd_harmonic_evidence(
                            psd, freqs, f * 0.5, pwr, median_noise * 2.0, freq_res
                        )
                        # Penalize likely sub-harmonic lock.
                        if ev05f > 0.6:
                            h_score -= 0.4
                    updated.append((f, pwr, score * (1.0 + h_score)))
                candidates = sorted(updated, key=lambda x: x[2], reverse=True)

            # v4.14 P4 diagnostic (env RPPG_WELCH_DIAG=1 to enable)
            import os as _os_diag
            if _os_diag.environ.get("RPPG_WELCH_DIAG"):
                top5 = candidates[:5]
                parts = []
                for idx, (f, pwr, sc) in enumerate(top5):
                    bpm = f * 60.0
                    physio = self._calculate_physiological_score(bpm)
                    rel_pow = pwr / max(candidates[0][1], 1e-9)
                    ev2f_d = self._psd_harmonic_evidence(psd, freqs, f * 2.0, pwr, median_noise * 2.0, freq_res) if len(candidates) >= 2 else 0.0
                    ev05f_d = self._psd_harmonic_evidence(psd, freqs, f * 0.5, pwr, median_noise * 2.0, freq_res) if (f * 0.5 >= min_freq and len(candidates) >= 2) else -1.0
                    h_d = 0.0
                    if 0 < ev2f_d < 0.6: h_d += 0.4
                    if ev05f_d > 0.6: h_d -= 0.4
                    sel = "*" if idx == 0 else ""
                    parts.append(f"{bpm:.0f}(P={rel_pow:.2f},Φ={physio:.2f},ev2f={ev2f_d:.2f},ev05f={ev05f_d:.2f},h={h_d:.1f},C={sc:.3f}){sel}")
                print(f"🔬 [Welch-Top5|PY] {' | '.join(parts)}")

            best = candidates[0]
            return float(best[0] * 60.0), float(best[1])

        # Fallback matches iOS max-in-range fallback.
        idxs = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]
        if len(idxs) == 0:
            return 0.0, 0.0
        best_idx = int(idxs[np.argmax(psd[idxs])])
        return float(freqs[best_idx] * 60.0), float(psd[best_idx])

    def green(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> Tuple[float, np.ndarray]:
        # v2: removed detrend_poly2 — poly fitting on short windows introduces
        # oscillatory residuals (0.75 Hz artifact). The bandpass filter (0.75 Hz
        # lower bound) now handles low-frequency removal instead.
        x = self.robust_norm(g.copy())
        x = self.bandpass(x, fs)
        hr, _ = self.welch_hr(x, fs)
        return hr, x

    def chrom(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> Tuple[float, np.ndarray]:
        rn = r / (np.mean(r) + 1e-9)
        gn = g / (np.mean(g) + 1e-9)
        bn = b / (np.mean(b) + 1e-9)

        x = 3.0 * rn - 2.0 * gn
        y = 1.5 * rn + gn - 1.5 * bn

        xf = self.bandpass(x, fs)
        yf = self.bandpass(y, fs)

        alpha = float(np.clip(np.std(xf) / (np.std(yf) + 1e-9), 0.1, 3.0))
        bvp = xf - alpha * yf
        hr, _ = self.welch_hr(bvp, fs)
        return hr, bvp

    def cbcr_pos(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> Tuple[float, np.ndarray]:
        """Modified POS using Cb/Cr projection (Li et al., 2021).

        Exploits the anti-phase property of Cb and Cr channels for
        noise cancellation.  Two-step normalisation: intensity-normalise
        first, then convert to YCbCr chrominance components.
        """
        total = r + g + b + 1e-9
        rn = r / total
        gn = g / total
        bn = b / total

        cb = -0.168 * rn - 0.331 * gn + 0.499 * bn
        cr =  0.499 * rn - 0.418 * gn - 0.081 * bn

        cb = self.detrend_poly2(cb)
        cr = self.detrend_poly2(cr)

        cb_f = self.bandpass(cb, fs)
        cr_f = self.bandpass(cr, fs)

        alpha = float(np.clip(np.std(cb_f) / (np.std(cr_f) + 1e-9), 0.1, 3.0))
        bvp = cb_f + alpha * cr_f
        hr, _ = self.welch_hr(bvp, fs)
        return hr, bvp

    def pos(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, fs: float) -> Tuple[float, np.ndarray]:
        n = len(r)
        w = max(8, int(self.cfg.pos_window_sec * fs))
        h = np.zeros(n, dtype=np.float64)
        if n < w + 2:
            return 0.0, h

        # Vectorized POS using stride_tricks for sliding windows.
        from numpy.lib.stride_tricks import sliding_window_view
        rw = sliding_window_view(r, w)  # shape: (n-w+1, w)
        gw = sliding_window_view(g, w)
        bw = sliding_window_view(b, w)
        # Window means: shape (n-w+1,)
        sr = np.mean(rw, axis=1)
        sg = np.mean(gw, axis=1)
        sb = np.mean(bw, axis=1)
        # Normalize each window by its mean
        rn = rw / (sr[:, None] + 1e-9)
        gn = gw / (sg[:, None] + 1e-9)
        bn = bw / (sb[:, None] + 1e-9)

        s1 = gn - bn
        s2 = gn + bn - 2.0 * rn
        std_s1 = np.std(s1, axis=1)
        std_s2 = np.std(s2, axis=1)
        alpha = std_s1 / (std_s2 + 1e-9)

        hn = s1 + alpha[:, None] * s2
        hn = hn - np.mean(hn, axis=1, keepdims=True)

        # Overlap-add: each hn[i] contributes to h[i:i+w]
        # Equivalent to the original loop but vectorized via cumulative addition.
        for i in range(hn.shape[0]):
            h[i:i + w] += hn[i]

        h = self.bandpass(h, fs)
        hr, _ = self.welch_hr(h, fs)
        return hr, h


class FusionEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_valid_hr: Optional[float] = None
        self.last_output_hr: Optional[float] = None
        self.history: Deque[float] = deque(maxlen=20)
        self.recent_for_constraints: Deque[float] = deque(maxlen=5)
        self.ppi_recent: Deque[float] = deque(maxlen=5)
        self.low_lock_hits: Deque[int] = deque(maxlen=max(3, int(cfg.low_lock_rescue_window)))
        self.last_freq_confidence: float = 0.0

    def reset_tracking_state(self) -> None:
        self.last_valid_hr = None
        self.last_output_hr = None
        self.history.clear()
        self.recent_for_constraints.clear()
        self.ppi_recent.clear()
        self.low_lock_hits.clear()
        self.last_freq_confidence = 0.0

    @staticmethod
    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        m = len(s) // 2
        return (s[m - 1] + s[m]) / 2.0 if len(s) % 2 == 0 else s[m]

    def _cluster_values(self, values: List[float]) -> List[List[float]]:
        if not values:
            return []
        s = sorted(values)
        clusters: List[List[float]] = [[s[0]]]
        for v in s[1:]:
            if abs(v - clusters[-1][-1]) <= self.cfg.cluster_gap_bpm:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return clusters

    def _cluster_score(self, cluster: List[float], total_count: int) -> float:
        support = len(cluster) / max(total_count, 1)
        spread = float(np.std(cluster)) if len(cluster) > 1 else 0.0
        stability = 1.0 / (1.0 + spread)
        center = self._median(cluster)

        if self.last_valid_hr is None:
            temporal = 0.5
        else:
            temporal = math.exp(-abs(center - self.last_valid_hr) / max(1e-6, self.cfg.temporal_sigma_bpm))

        ws = max(0.0, float(self.cfg.fusion_support_weight))
        wt = max(0.0, float(self.cfg.fusion_temporal_weight))
        wst = max(0.0, float(self.cfg.fusion_stability_weight))
        wsum = ws + wt + wst
        if wsum <= 1e-9:
            ws, wt, wst = 0.55, 0.30, 0.15
            wsum = 1.0
        ws /= wsum
        wt /= wsum
        wst /= wsum
        score = ws * support + wt * temporal + wst * stability

        high_hr_anchor = self.last_valid_hr if self.last_valid_hr is not None else center
        if high_hr_anchor >= self.cfg.high_hr_mode_threshold and center >= self.cfg.high_hr_mode_threshold:
            score += self.cfg.high_hr_cluster_boost

        return float(max(0.0, min(1.0, score)))

    def _remove_outliers(self, tagged: List[Tuple[float, str, str]], threshold: float = 20.0) -> List[Tuple[float, str, str]]:
        """Remove candidates far from the group median, but preserve any that
        are close to last_valid_hr (temporal continuity protection)."""
        if len(tagged) < 4:
            return tagged
        med = float(np.median([v[0] for v in tagged]))
        result = []
        for v in tagged:
            near_median = abs(v[0] - med) <= threshold
            near_history = (self.last_valid_hr is not None
                            and abs(v[0] - self.last_valid_hr) <= threshold)
            if near_median or near_history:
                result.append(v)
        return result if result else tagged

    def harmonic_temporal_fusion(
        self,
        tagged: List[Tuple[float, str, str]],
        sqi: Optional[float] = None,
    ) -> Tuple[float, float]:
        valid_raw = [x for x in tagged if self.cfg.min_bpm_valid <= x[0] <= self.cfg.max_bpm_valid]
        if not valid_raw:
            return 0.0, 0.0

        valid = self._remove_outliers(valid_raw)
        if not valid:
            return 0.0, 0.0

        values = [v[0] for v in valid]
        harmonic_hr = self.harmonic_aware_fusion(valid)
        clusters = self._cluster_values(values)
        if not clusters:
            return harmonic_hr, 0.0

        scored = sorted(
            ((self._cluster_score(c, len(values)), self._median(c), c) for c in clusters),
            key=lambda x: x[0],
            reverse=True,
        )
        best_score, best_center, _ = scored[0]

        # Blend harmonic selection and temporal-cluster selection.
        if harmonic_hr > 0.0:
            if abs(harmonic_hr - best_center) <= self.cfg.cluster_gap_bpm:
                hr = 0.65 * harmonic_hr + 0.35 * best_center
            else:
                # Prefer the temporally consistent cluster if harmonic choice jumps too far.
                if self.last_valid_hr is not None and abs(harmonic_hr - self.last_valid_hr) > 1.3 * abs(best_center - self.last_valid_hr):
                    hr = best_center
                else:
                    hr = harmonic_hr
        else:
            hr = best_center

        freq_conf = best_score
        if len(scored) >= 2:
            freq_conf += 0.20 * max(0.0, scored[0][0] - scored[1][0])
        if self.last_valid_hr is not None:
            continuity = math.exp(-abs(hr - self.last_valid_hr) / max(1e-6, self.cfg.temporal_sigma_bpm))
            freq_conf = 0.75 * freq_conf + 0.25 * continuity
        freq_conf = float(max(0.0, min(1.0, freq_conf)))

        if self.cfg.low_lock_rescue_enabled:
            arr = np.array([v[0] for v in valid_raw], dtype=np.float64)
            low_cnt = int(np.sum(arr < self.cfg.low_lock_rescue_low_candidate_bpm))
            high_vals = arr[arr > self.cfg.low_lock_rescue_high_candidate_bpm]
            p90 = float(np.quantile(arr, 0.90)) if len(arr) >= 3 else float(np.max(arr))
            sqi_ok = (sqi is None) or (float(sqi) <= self.cfg.low_lock_rescue_sqi_max)
            low_locked = (
                hr <= self.cfg.low_lock_rescue_hr_max
                and sqi_ok
                and low_cnt >= max(4, int(0.5 * len(arr)))
                and len(high_vals) >= self.cfg.low_lock_rescue_min_high_count
                and p90 >= self.cfg.low_lock_rescue_high_candidate_bpm
            )
            self.low_lock_hits.append(1 if low_locked else 0)
            if sum(self.low_lock_hits) >= self.cfg.low_lock_rescue_min_hit_count and len(high_vals) > 0:
                target = float(np.median(high_vals))
                if target > hr:
                    lift = (target - hr) * self.cfg.low_lock_rescue_lift_ratio
                    lift = min(self.cfg.low_lock_rescue_max_lift_bpm, lift)
                    hr = min(self.cfg.clamp_max_bpm, hr + max(0.0, lift))

        self.last_freq_confidence = freq_conf
        return hr, freq_conf

    def harmonic_aware_fusion(self, tagged: List[Tuple[float, str, str]]) -> float:
        valid = [x for x in tagged if self.cfg.min_bpm_valid <= x[0] <= self.cfg.max_bpm_valid]
        if len(valid) < 4:
            return self._median([v[0] for v in valid])

        bpms = sorted(v[0] for v in valid)
        gaps = [bpms[i + 1] - bpms[i] for i in range(len(bpms) - 1)]
        split = int(np.argmax(gaps)) + 1
        max_gap = gaps[split - 1]

        low = bpms[:split]
        high = bpms[split:]
        if max_gap < 15.0 or len(low) < 2 or len(high) < 2:
            return self._median(bpms)

        low_med = self._median(low)
        high_med = self._median(high)
        ratio = high_med / max(low_med, 1e-6)
        if not (1.6 < ratio < 2.4):
            return self._median(bpms)

        boundary = (low_med + high_med) / 2.0
        rois = sorted({v[2] for v in valid})
        high_votes = 0
        low_votes = 0
        for roi in rois:
            roi_vals = [v[0] for v in valid if v[2] == roi]
            hi = sum(1 for x in roi_vals if x >= boundary)
            lo = len(roi_vals) - hi
            if hi >= 2:
                high_votes += 1
            elif lo >= 2:
                low_votes += 1

        if high_votes > low_votes:
            winner = high
        elif low_votes > high_votes:
            winner = low
        else:
            if self.last_valid_hr is None:
                winner = low
            else:
                winner = low if abs(low_med - self.last_valid_hr) <= abs(high_med - self.last_valid_hr) else high

        return self._median(winner)

    def apply_physiological_constraints(self, hr: float) -> float:
        if hr <= 0:
            return 0.0

        constrained = max(self.cfg.clamp_min_bpm, min(self.cfg.clamp_max_bpm, hr))

        if len(self.recent_for_constraints) > 0:
            recent = np.array(self.recent_for_constraints, dtype=np.float64)
            recent_avg = float(np.mean(recent))
            recent_std = float(np.std(recent))

            if recent_std < 5.0:
                max_deviation = max(25.0, recent_std * 2.5)
            elif recent_std < 15.0:
                max_deviation = max(28.0, recent_std * 2.0)
            else:
                max_deviation = 30.0

            deviation = abs(constrained - recent_avg)
            if deviation > max_deviation:
                direction = 1.0 if constrained > recent_avg else -1.0
                return recent_avg + direction * max_deviation

        if self.last_valid_hr is not None:
            if len(self.recent_for_constraints) > 0:
                ref = np.array(list(self.recent_for_constraints)[-3:], dtype=np.float64)
                recent_std = float(np.sqrt(np.mean((ref - self.last_valid_hr) ** 2)))
                if recent_std < 3.0:
                    base_max_change = 22.0
                elif recent_std < 8.0:
                    base_max_change = 25.0
                else:
                    base_max_change = 28.0
            else:
                base_max_change = 25.0

            difference = constrained - self.last_valid_hr
            if abs(difference) > base_max_change:
                direction = 1.0 if difference > 0 else -1.0
                return self.last_valid_hr + direction * base_max_change
        else:
            if constrained < 45.0:
                return 45.0
            if constrained > self.cfg.max_bpm_valid:
                return self.cfg.max_bpm_valid

        return constrained

    def update_and_get_best(self, hr: float) -> float:
        if hr > 0:
            self.history.append(hr)
            self.last_valid_hr = hr
            self.recent_for_constraints.append(hr)

        if not self.history:
            return 0.0

        window = list(self.history)[-20:]
        if len(window) < 3:
            out = float(window[-1])
            self.last_output_hr = out
            return out

        result_pool = sorted(window)
        if len(window) >= 5:
            tail5 = window[-5:]
            spread = max(tail5) - min(tail5)
            if spread < 15.0:
                best_start = len(window) - 5
                for i in range(best_start - 1, -1, -1):
                    seg = window[i:]
                    if max(seg) - min(seg) < 15.0:
                        best_start = i
                    else:
                        break
                result_pool = sorted(window[best_start:])

        trim = max(1, int(len(result_pool) * 0.15))
        if len(result_pool) > 2 * trim + 1:
            trimmed = result_pool[trim:len(result_pool) - trim]
        else:
            trimmed = result_pool

        best = float(trimmed[len(trimmed) // 2])
        if best >= self.cfg.high_hr_bias_threshold:
            best = best * self.cfg.high_hr_bias_gain + self.cfg.high_hr_bias_offset
            best = max(self.cfg.clamp_min_bpm, min(self.cfg.clamp_max_bpm, best))
        self.last_output_hr = best
        return best

    def _apply_output_drop_guard(self, candidate_hr: float, sqi: float, freq_conf: float) -> float:
        if not self.cfg.output_drop_guard_enabled:
            return candidate_hr
        if candidate_hr <= 0.0:
            return candidate_hr

        ref_hr = self.last_output_hr if self.last_output_hr is not None else self.last_valid_hr
        if ref_hr is None or ref_hr <= 0.0:
            return candidate_hr

        ratio_to_ref = candidate_hr / max(ref_hr, 1e-6)
        if (
            ref_hr >= self.cfg.output_drop_guard_ref_min_bpm
            and candidate_hr <= self.cfg.output_drop_guard_out_max_bpm
            and ratio_to_ref <= self.cfg.output_drop_guard_ratio_max
            and freq_conf >= self.cfg.output_drop_guard_freq_conf_min
            and sqi >= self.cfg.output_drop_guard_sqi_min
        ):
            floor_by_ratio = ref_hr * self.cfg.output_drop_guard_floor_ratio
            floor_by_delta = ref_hr - self.cfg.output_drop_guard_floor_delta
            return max(candidate_hr, floor_by_ratio, floor_by_delta)
        return candidate_hr

    def apply_ppi_assist(self, hr: float, ppi_hr: Optional[float], sqi: float, freq_conf: float = 0.0) -> float:
        """Use time-domain PPI HR to recover from harmonic lock in high-HR cases."""
        if hr <= 0 or ppi_hr is None:
            return self._apply_output_drop_guard(hr, sqi, freq_conf)
        if not (45.0 <= ppi_hr <= 150.0):
            return self._apply_output_drop_guard(hr, sqi, freq_conf)

        self.ppi_recent.append(ppi_hr)
        recent = np.array(list(self.ppi_recent)[-3:], dtype=np.float64)
        ppi_med = float(np.median(recent))
        ppi_std = float(np.std(recent))
        diff = ppi_med - hr
        ratio = ppi_med / max(hr, 1e-6)

        # Adaptive dual-domain gate learned from offline optimization.
        tree_gate = False
        if hr <= 61.04252:
            if ppi_med <= 59.97365:
                tree_gate = True
        elif hr <= 73.64661:
            if ppi_med > 75.95376:
                tree_gate = True
        else:
            if ppi_med <= 100.76003:
                if hr <= 77.91994 and sqi > 0.67477:
                    tree_gate = True
            else:
                if hr > 93.59276:
                    tree_gate = True

        if tree_gate:
            target_ppi = ppi_hr if len(self.ppi_recent) < 3 else ppi_med
            corrected = self.cfg.ppi_blend_tree_self * hr + self.cfg.ppi_blend_tree_ppi * target_ppi
            if self.last_valid_hr is not None:
                max_jump = 45.0
                d = corrected - self.last_valid_hr
                if abs(d) > max_jump:
                    corrected = self.last_valid_hr + math.copysign(max_jump, d)
            return self._apply_output_drop_guard(corrected, sqi, freq_conf)

        if len(self.ppi_recent) < 3:
            return self._apply_output_drop_guard(hr, sqi, freq_conf)
        if sqi < self.cfg.ppi_gate_sqi_min:
            return self._apply_output_drop_guard(hr, sqi, freq_conf)
        if ppi_std > self.cfg.ppi_gate_std_max:
            return self._apply_output_drop_guard(hr, sqi, freq_conf)
        if diff < self.cfg.ppi_gate_diff_min:
            return self._apply_output_drop_guard(hr, sqi, freq_conf)
        if not (self.cfg.ppi_gate_ratio_min <= ratio <= self.cfg.ppi_gate_ratio_max):
            return self._apply_output_drop_guard(hr, sqi, freq_conf)

        # If frequency confidence is weak, allow stronger correction toward PPI.
        if freq_conf < 0.55 and diff >= 10.0 and ratio >= 1.15:
            corrected = self.cfg.ppi_blend_weak_self * hr + self.cfg.ppi_blend_weak_ppi * ppi_med
        else:
            corrected = self.cfg.ppi_blend_strong_self * hr + self.cfg.ppi_blend_strong_ppi * ppi_med

        if self.last_valid_hr is not None:
            max_jump = 35.0
            d = corrected - self.last_valid_hr
            if abs(d) > max_jump:
                corrected = self.last_valid_hr + math.copysign(max_jump, d)
        return self._apply_output_drop_guard(corrected, sqi, freq_conf)


def detect_peaks_and_pnn50(
    bvp: np.ndarray,
    fs: float,
    hr_min_sec: float = 3.0,
    pnn50_min_sec: float = 20.0,
    peak_threshold_k: float = 0.5,
    peak_window_divisor: float = 15.0,
    peak_missing_threshold_scale: float = 0.7,
    peak_min_distance_hz: float = 3.0,
    peak_max_distance_hz: float = 0.67,
) -> Tuple[Optional[float], Optional[float], List[int]]:
    if len(bvp) < int(hr_min_sec * fs):
        return None, None, []

    signal = bvp.astype(np.float64).tolist()
    if len(signal) <= 10:
        return None, None, []

    peaks: List[int] = []
    min_distance = max(1, int(fs / max(0.1, peak_min_distance_hz)))
    max_distance = max(1, int(fs / max(0.1, peak_max_distance_hz)))
    mean = float(np.mean(signal))
    variance = float(np.mean([(x - mean) ** 2 for x in signal]))
    threshold = mean + peak_threshold_k * math.sqrt(variance)
    window_size = max(3, int(fs / max(1.0, peak_window_divisor)))

    def find_missing_peaks(start: int, end: int) -> List[int]:
        expected_interval = int(fs)
        search_start = start + expected_interval
        search_end = end - expected_interval
        if search_start >= search_end:
            return []
        if (search_end - search_start) <= window_size * 2:
            return []
        lower_threshold = threshold * peak_missing_threshold_scale
        for i in range(search_start, search_end):
            cur = signal[i]
            if cur <= lower_threshold:
                continue
            is_local_max = True
            for j in range(i - window_size, i + window_size + 1):
                if j != i and 0 <= j < len(signal) and signal[j] >= cur:
                    is_local_max = False
                    break
            if not is_local_max:
                continue
            distance_to_start = i - start
            distance_to_end = end - i
            min_dist = expected_interval // 2
            if distance_to_start >= min_dist and distance_to_end >= min_dist:
                return [i]
        return []

    for i in range(window_size, len(signal) - window_size):
        current = signal[i]
        if current <= threshold:
            continue
        is_local_max = True
        for j in range(i - window_size, i + window_size + 1):
            if j != i and signal[j] >= current:
                is_local_max = False
                break
        if not is_local_max:
            continue

        if peaks:
            last_peak = peaks[-1]
            distance = i - last_peak
            if distance < min_distance:
                if current > signal[last_peak]:
                    peaks.pop()
                    peaks.append(i)
                continue
            if distance > max_distance:
                peaks.extend(find_missing_peaks(last_peak, i))
        peaks.append(i)

    if len(peaks) < 2:
        return None, None, []

    intervals_ms = [int((peaks[i] - peaks[i - 1]) * (1000.0 / fs)) for i in range(1, len(peaks))]
    if len(intervals_ms) <= 2:
        return None, None, peaks

    valid = [v for v in intervals_ms if 333 <= v <= 1500]
    if len(valid) >= 3:
        mean_i = float(np.mean(valid))
        std_i = float(np.std(valid))
        lo, hi = mean_i - 2.5 * std_i, mean_i + 2.5 * std_i

        def conservative_correction(original: float, target: float, max_correction: float) -> float:
            diff = target - original
            adiff = abs(diff)
            if adiff <= max_correction:
                return target
            factor = max_correction / adiff
            return original + diff * factor

        def interpolate_anomaly(index: int, vals: List[int], mean_val: float) -> int:
            count = len(vals)
            original = float(vals[index])
            max_correction = mean_val * 0.5
            if index == 0 or index == count - 1:
                return int(conservative_correction(original, mean_val, max_correction))

            valid_before: Optional[float] = None
            valid_after: Optional[float] = None
            lo_r = mean_val * 0.4
            hi_r = mean_val * 2.0

            for j in range(index - 1, max(-1, index - 4), -1):
                if j < 0:
                    break
                val = float(vals[j])
                if 333 <= val <= 1500 and lo_r <= val <= hi_r:
                    valid_before = val
                    break

            for j in range(index + 1, min(count, index + 4)):
                val = float(vals[j])
                if 333 <= val <= 1500 and lo_r <= val <= hi_r:
                    valid_after = val
                    break

            if valid_before is not None and valid_after is not None:
                target = valid_before * 0.6 + valid_after * 0.4
            elif valid_before is not None:
                target = valid_before * 0.9 + mean_val * 0.1
            elif valid_after is not None:
                target = valid_after * 0.9 + mean_val * 0.1
            else:
                target = mean_val
            return int(conservative_correction(original, target, max_correction))

        corrected = intervals_ms[:]
        for i, cur in enumerate(corrected):
            if cur < 333 or cur > 1500 or cur < lo or cur > hi:
                corrected[i] = interpolate_anomaly(i, corrected, mean_i)
        intervals_ms = corrected

    avg_ppi = float(np.mean(intervals_ms))
    avg_hr = 60000.0 / avg_ppi if avg_ppi > 0 else 0.0

    # pNN50 is an HRV-style marker and needs longer windows to be physiologically meaningful.
    pnn50_ratio: Optional[float] = None
    min_intervals_for_pnn50 = 15
    if len(bvp) >= int(pnn50_min_sec * fs) and len(intervals_ms) >= min_intervals_for_pnn50:
        nn50 = 0
        for i in range(len(intervals_ms) - 1):
            if abs(intervals_ms[i + 1] - intervals_ms[i]) > 50:
                nn50 += 1
        pnn50_ratio = float(nn50) / float(len(intervals_ms) - 1)
    return avg_hr, pnn50_ratio, peaks


def calculate_frequency_domain_snr(signal: np.ndarray, fs: float) -> float:
    n = len(signal)
    if n < 2:
        return 0.0
    spectrum = np.fft.rfft(signal)
    power = np.abs(spectrum) ** 2
    freq_resolution = fs / n
    min_hr_freq = 0.65
    max_hr_freq = min(4.0, fs / 2.0 - freq_resolution)
    min_bin = int(min_hr_freq / freq_resolution)
    max_bin = min(int(max_hr_freq / freq_resolution), len(power) - 1)
    if max_bin <= min_bin:
        return 0.0

    signal_power = float(np.sum(power[min_bin:max_bin + 1]))
    dc_cutoff = max(1, int(0.3 / freq_resolution))
    noise_power = 0.0
    if dc_cutoff < min_bin:
        noise_power += float(np.sum(power[dc_cutoff:min_bin]))
    if max_bin + 1 < len(power):
        noise_power += float(np.sum(power[max_bin + 1:]))

    signal_bw = max_bin - min_bin + 1
    noise_bw = max(1, (min_bin - dc_cutoff) + (len(power) - max_bin - 1))
    avg_signal = signal_power / signal_bw
    avg_noise = noise_power / noise_bw
    if avg_noise > 1e-10:
        snr_db = 10.0 * math.log10(max(avg_signal / avg_noise, 1e-10))
        return max(0.0, min(50.0, snr_db))
    return 30.0


def assess_signal_quality(
    signal: np.ndarray,
    fs: float,
    snr_weight: float = 0.30,
    stability_weight: float = 0.25,
    corr_weight: float = 0.25,
    periodicity_weight: float = 0.20,
    snr_norm_div: float = 15.0,
    signal_strength_min: float = 0.003,
) -> Tuple[float, float]:
    if len(signal) <= 1:
        return 0.0, 0.0
    mean = float(np.mean(signal))
    variance = float(np.mean((signal - mean) ** 2))
    signal_strength = math.sqrt(variance)
    if signal_strength < signal_strength_min:
        return 0.0, 0.0

    snr = calculate_frequency_domain_snr(signal, fs)
    normalized_snr = min(snr / max(1e-6, snr_norm_div), 1.0)
    std = math.sqrt(float(np.mean((signal - mean) ** 2)))
    stability = 1.0 / (1.0 + std)

    max_lag_corr = min(len(signal) // 2, 30)
    corr = 0.0
    for lag in range(1, max_lag_corr):
        val = float(np.mean(signal[:-lag] * signal[lag:]))
        corr = max(corr, abs(val))
    corr = min(corr, 1.0)

    max_lag_per = min(len(signal) // 2, 50)
    periodicity = 0.0
    for lag in range(1, max_lag_per):
        val = float(np.mean(signal[:-lag] * signal[lag:]))
        periodicity = max(periodicity, abs(val))
    periodicity = min(periodicity, 1.0)

    ws = max(0.0, snr_weight)
    wt = max(0.0, stability_weight)
    wc = max(0.0, corr_weight)
    wp = max(0.0, periodicity_weight)
    wsum = ws + wt + wc + wp
    if wsum <= 1e-9:
        ws, wt, wc, wp = 0.3, 0.25, 0.25, 0.2
        wsum = 1.0
    ws /= wsum
    wt /= wsum
    wc /= wsum
    wp /= wsum
    quality = ws * normalized_snr + wt * stability + wc * corr + wp * periodicity
    return max(0.0, min(1.0, quality)), snr


class SignalQualityController:
    # v2: added calibration phase tracking.  The first CALIBRATION_SECONDS
    # from t0 are marked as "CALIBRATING" — the HR value is still computed
    # and published, but downstream consumers (UI) should treat it as
    # unreliable.  This is a product-level signal, not an algorithm change.
    CALIBRATION_SECONDS: float = 20.0

    def __init__(self, publish_interval: float = 1.0, t0: float = 0.0):
        self.state = "BAD"
        self.is_calibrating = True
        self.last_good_hr = 0.0
        self.marginal_blend_factor = 0.55
        self.last_publish_time = 0.0
        self.publish_interval = publish_interval
        self.t0 = t0

    @staticmethod
    def determine_state(signal_quality: float) -> str:
        if signal_quality >= 0.5:
            return "GOOD"
        if signal_quality >= 0.3:
            return "MARGINAL"
        return "BAD"

    def apply(self, hr: float, confidence: float, signal_quality: float, now: float) -> Tuple[Optional[float], float, str]:
        self.state = self.determine_state(signal_quality)
        self.is_calibrating = (now - self.t0) < self.CALIBRATION_SECONDS
        final_hr = hr
        final_conf = confidence

        if self.state == "GOOD":
            final_hr = hr
            self.last_good_hr = hr
        elif self.state == "MARGINAL":
            if self.last_good_hr > 0:
                final_hr = self.marginal_blend_factor * hr + (1.0 - self.marginal_blend_factor) * self.last_good_hr
                final_conf = confidence * 0.7
                drift = abs(hr - self.last_good_hr)
                if drift < 10.0:
                    self.last_good_hr = 0.85 * self.last_good_hr + 0.15 * final_hr
                elif drift >= 15.0:
                    self.last_good_hr = 0.6 * self.last_good_hr + 0.4 * hr
                else:
                    self.last_good_hr = 0.75 * self.last_good_hr + 0.25 * final_hr
            else:
                self.last_good_hr = hr
        else:
            if self.last_good_hr > 0:
                final_hr = self.last_good_hr
                final_conf = 0.0
            else:
                return None, 0.0, self.state

        effective_interval = 2.0 if self.state == "BAD" else self.publish_interval
        if now - self.last_publish_time >= effective_interval:
            self.last_publish_time = now
            return final_hr, final_conf, self.state
        return None, final_conf, self.state


def choose_camera(source: str, fps: int) -> CameraSource:
    if source == "realsense":
        return RealSenseCamera(fps=fps)
    if source == "picamera":
        return Picamera2Camera(fps=fps)
    if source == "webcam":
        return OpenCVCamera(0)

    # auto: try D455 -> picamera2 -> webcam
    try:
        return RealSenseCamera(fps=fps)
    except Exception:
        pass
    try:
        return Picamera2Camera(fps=fps)
    except Exception:
        return OpenCVCamera(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="rPPG on Raspberry Pi (D455/webcam): HR + pNN50")
    parser.add_argument("--source", choices=["auto", "realsense", "picamera", "webcam"], default="auto")
    parser.add_argument("--fps", type=int, default=60, help="camera target fps (for D455)")
    parser.add_argument("--roi-mode", choices=["auto", "mediapipe", "opencv"], default="auto",
                        help="ROI extractor mode: mediapipe or opencv fallback")
    parser.add_argument("--roi-preset", choices=["classic4", "hybrid7", "cheek_forehead6", "whole_face"],
                        default="hybrid7", help="ROI geometry preset")
    parser.add_argument("--roi-scale-x", type=float, default=1.1, help="ROI width scale in face-relative coords")
    parser.add_argument("--roi-scale-y", type=float, default=1.1, help="ROI height scale in face-relative coords")
    parser.add_argument("--roi-shift-y", type=float, default=0.0, help="ROI vertical shift in face-relative coords")
    parser.add_argument("--mp-face-detector-model", default="",
                        help="path to MediaPipe tasks face detector model (.tflite/.task) for py3.12 tasks-only builds")
    parser.add_argument("--strict-roi", action="store_true",
                        help="fail fast if selected ROI backend cannot be initialized")
    parser.add_argument("--enable-ppi-assist", action="store_true",
                        help="force-enable PPI-assisted HR correction")
    parser.add_argument("--disable-ppi-assist", action="store_true",
                        help="force-disable PPI-assisted HR correction")
    parser.add_argument("--enable-adaptive-roi", action="store_true",
                        help="enable lightweight ROI adaptation (experimental, opencv/hybrid7 only)")
    parser.add_argument("--disable-adaptive-roi", action="store_true",
                        help="disable lightweight ROI adaptation")
    parser.add_argument("--debug-adaptive-roi", action="store_true",
                        help="print adaptive ROI switch events")
    parser.add_argument("--publish-min-freq-conf", type=float, default=0.0,
                        help="only publish output when freq_conf >= threshold (0 disables gate)")
    parser.add_argument("--publish-min-sqi", type=float, default=0.0,
                        help="only publish output when SQI >= threshold (0 disables gate)")
    parser.add_argument("--show", action="store_true", help="show debug window")
    parser.add_argument("--print-every", type=float, default=1.0, help="seconds")
    parser.add_argument("--debug-hr", action="store_true", help="print per-ROI/algo HR candidates")
    args = parser.parse_args()

    cfg = Config()
    if args.enable_ppi_assist and args.disable_ppi_assist:
        raise RuntimeError("cannot set both --enable-ppi-assist and --disable-ppi-assist")
    if args.enable_ppi_assist:
        cfg.enable_ppi_assist = True
    if args.disable_ppi_assist:
        cfg.enable_ppi_assist = False
    if args.enable_adaptive_roi and args.disable_adaptive_roi:
        raise RuntimeError("cannot set both --enable-adaptive-roi and --disable-adaptive-roi")
    if args.enable_adaptive_roi:
        cfg.enable_adaptive_roi = True
    if args.disable_adaptive_roi:
        cfg.enable_adaptive_roi = False
    if args.debug_adaptive_roi:
        cfg.adaptive_roi_debug = True
    cfg.publish_min_freq_conf_for_output = float(max(0.0, min(1.0, args.publish_min_freq_conf)))
    cfg.publish_min_sqi_for_output = float(max(0.0, min(1.0, args.publish_min_sqi)))
    cfg.roi_preset = args.roi_preset
    cfg.roi_scale_x = args.roi_scale_x
    cfg.roi_scale_y = args.roi_scale_y
    cfg.roi_shift_y = args.roi_shift_y
    camera = choose_camera(args.source, args.fps)
    extractor, roi_mode_used = create_roi_extractor(
        args.roi_mode,
        mp_face_detector_model=args.mp_face_detector_model,
        strict=args.strict_roi,
        roi_preset=cfg.roi_preset,
        roi_scale_x=cfg.roi_scale_x,
        roi_scale_y=cfg.roi_scale_y,
        roi_shift_y=cfg.roi_shift_y,
    )
    alg = RPPGAlgorithms(cfg)
    fusion = FusionEngine(cfg)
    adaptive_roi_ctl = AdaptiveROIController(cfg, extractor)
    t0 = time.time()
    quality_ctl = SignalQualityController(publish_interval=1.0, t0=t0)
    print(f"[ROI] mode={roi_mode_used}")

    # per-ROI signal buffers
    roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
    roi_weights: Dict[str, float] = {}
    t_buf: Deque[float] = deque(maxlen=5000)

    last_print = 0.0
    last_compute = 0.0
    compute_interval = 0.5  # HR computation every 0.5s; sampling stays at full camera rate
    fps_count = 0
    fps_val = 0.0
    fps_last_time = 0.0

    # Persistent state (updated by compute block, read by print/show)
    fs = float(args.fps)
    hr_best = 0.0
    hr_published: Optional[float] = None
    pnn50 = None
    ppi_hr = None
    sqi = 0.0
    snr_db = 0.0
    freq_conf = 0.0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            now = time.time()
            t_buf.append(now)
            fps_count += 1
            if now - fps_last_time >= 1.0:
                fps_val = fps_count / (now - fps_last_time)
                fps_count = 0
                fps_last_time = now

            rois = extractor.extract(frame)
            if not rois:
                if now - last_print >= args.print_every:
                    elapsed = now - t0
                    print(
                        f"t={elapsed:6.1f}s fs={fs:5.1f} FPS={fps_val:5.1f} HR=   0.0 BPM "
                        f"PPI_HR=NA pNN50=NA SQI=0.000 FCONF=0.000 "
                        f"SNR=0.0 state=NO_FACE"
                    )
                    last_print = now
                if args.show:
                    cv2.imshow("rPPG", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            for roi in rois:
                rgb = extract_rgb_mean(frame, roi)
                if rgb is None:
                    continue

                if roi.name not in roi_buf:
                    max_len = int(cfg.buffer_sec * max(10, args.fps))
                    roi_buf[roi.name] = {
                        "r": deque(maxlen=max_len),
                        "g": deque(maxlen=max_len),
                        "b": deque(maxlen=max_len),
                    }
                roi_buf[roi.name]["r"].append(rgb[0])
                roi_buf[roi.name]["g"].append(rgb[1])
                roi_buf[roi.name]["b"].append(rgb[2])
                roi_weights[roi.name] = roi.weight

            # Estimate effective sampling rate from timestamp buffer.
            if len(t_buf) > 40:
                dt = np.diff(np.array(t_buf, dtype=np.float64)[-120:])
                fs = float(1.0 / max(1e-6, np.median(dt)))
            else:
                fs = float(args.fps)

            # --- Heavy HR computation: only every compute_interval seconds ---
            if now - last_compute >= compute_interval:
                last_compute = now

                tagged: List[Tuple[float, str, str]] = []
                merged_rgb: Dict[str, List[np.ndarray]] = {"r": [], "g": [], "b": []}
                merged_w: List[float] = []

                min_len = int(max(cfg.welch_seg_sec * fs, cfg.pos_window_sec * fs) + 5)
                # Limit signal length for computation: Welch needs ~welch_seg_sec,
                # keep 2x for overlap+margin but cap to avoid O(n^2) POS slowdown.
                max_compute_len = int(cfg.welch_seg_sec * fs * 2.5) + 10
                for name, buf in roi_buf.items():
                    if len(buf["g"]) < min_len:
                        continue

                    r = np.array(buf["r"], dtype=np.float64)[-max_compute_len:]
                    g = np.array(buf["g"], dtype=np.float64)[-max_compute_len:]
                    b = np.array(buf["b"], dtype=np.float64)[-max_compute_len:]

                    hr_g, _ = alg.green(r, g, b, fs)
                    hr_c, _ = alg.chrom(r, g, b, fs)
                    hr_p, _ = alg.pos(r, g, b, fs)
                    hr_cb = 0.0
                    if cfg.use_cbcr_candidate:
                        hr_cb, _ = alg.cbcr_pos(r, g, b, fs)

                    if cfg.min_bpm_valid <= hr_g <= cfg.max_bpm_valid:
                        tagged.append((hr_g, "GREEN", name))
                    if cfg.min_bpm_valid <= hr_c <= cfg.max_bpm_valid:
                        tagged.append((hr_c, "CHROM", name))
                    if cfg.min_bpm_valid <= hr_p <= cfg.max_bpm_valid:
                        tagged.append((hr_p, "POS", name))
                    if cfg.use_cbcr_candidate and cfg.min_bpm_valid <= hr_cb <= cfg.max_bpm_valid:
                        tagged.append((hr_cb, "CBCR", name))

                    w = roi_weights.get(name, 1.0)
                    merged_rgb["r"].append(r)
                    merged_rgb["g"].append(g)
                    merged_rgb["b"].append(b)
                    merged_w.append(w)

                if args.debug_hr and tagged:
                    by_algo: Dict[str, List[float]] = {}
                    for val, algo, roi in tagged:
                        by_algo.setdefault(algo, []).append(val)
                    parts = []
                    for algo in ["GREEN", "CHROM", "POS", "CBCR"]:
                        if algo in by_algo:
                            vals = by_algo[algo]
                            parts.append(f"{algo}=[{','.join(f'{v:.0f}' for v in sorted(vals))}]")
                    print(f"  >> candidates({len(tagged)}): {' '.join(parts)}")

                hr_final: Optional[float] = None
                hr_published = None
                # pNN50 branch: use merged POS-BVP (weighted ROIs).
                if merged_w:
                    min_common = min(len(x) for x in merged_rgb["r"])
                    if min_common >= min_len:
                        w = np.array(merged_w, dtype=np.float64)
                        w = w / (np.sum(w) + 1e-9)
                        rr = np.vstack([x[-min_common:] for x in merged_rgb["r"]])
                        gg = np.vstack([x[-min_common:] for x in merged_rgb["g"]])
                        bb = np.vstack([x[-min_common:] for x in merged_rgb["b"]])
                        r_mix = np.sum(rr * w[:, None], axis=0)
                        g_mix = np.sum(gg * w[:, None], axis=0)
                        b_mix = np.sum(bb * w[:, None], axis=0)
                        # Match iOS quality path: preprocess then SQI on signal window.
                        quality_signal = alg.bandpass(alg.robust_norm(alg.detrend_poly2(g_mix)), fs)
                        sqi, snr_db = assess_signal_quality(
                            quality_signal,
                            fs,
                            snr_weight=cfg.quality_snr_weight,
                            stability_weight=cfg.quality_stability_weight,
                            corr_weight=cfg.quality_corr_weight,
                            periodicity_weight=cfg.quality_periodicity_weight,
                            snr_norm_div=cfg.quality_snr_norm_div,
                            signal_strength_min=cfg.quality_signal_strength_min,
                        )
                        _, bvp = alg.pos(r_mix, g_mix, b_mix, fs)
                        ppi_hr, pnn50, _ = detect_peaks_and_pnn50(
                            bvp,
                            fs,
                            peak_threshold_k=cfg.peak_threshold_k,
                            peak_window_divisor=cfg.peak_window_divisor,
                            peak_missing_threshold_scale=cfg.peak_missing_threshold_scale,
                            peak_min_distance_hz=cfg.peak_min_distance_hz,
                            peak_max_distance_hz=cfg.peak_max_distance_hz,
                        )

                if tagged:
                    hr_raw, freq_conf = fusion.harmonic_temporal_fusion(tagged, sqi=sqi)
                    adapt_msg = adaptive_roi_ctl.update(
                        now_sec=now - t0,
                        tagged=tagged,
                        hr_raw=hr_raw,
                        ppi_hr=ppi_hr,
                        freq_conf=freq_conf,
                        sqi=sqi,
                    )
                    switched_profile = bool(adapt_msg and adapt_msg.startswith("switch ->"))
                    if switched_profile:
                        # Drop stale history when ROI geometry changes.
                        roi_buf.clear()
                        roi_weights.clear()
                        fusion.reset_tracking_state()
                    if adapt_msg and cfg.adaptive_roi_debug:
                        print(f"[ROI-ADAPT] t={now - t0:6.1f}s {adapt_msg}")
                    if not switched_profile:
                        hr_final = fusion.apply_physiological_constraints(hr_raw)

                if hr_final is not None and cfg.enable_ppi_assist:
                    hr_final = fusion.apply_ppi_assist(hr_final, ppi_hr, sqi, freq_conf=freq_conf)
                if hr_final is not None:
                    hr_best = fusion.update_and_get_best(hr_final)

                if hr_best > 0:
                    combined_quality = max(
                        0.0,
                        min(
                            1.0,
                            cfg.publish_quality_sqi_weight * sqi +
                            cfg.publish_quality_freq_weight * freq_conf,
                        ),
                    )
                    if freq_conf < cfg.freq_conf_gate:
                        combined_quality *= cfg.publish_quality_low_freq_penalty
                    conf = max(
                        0.0,
                        min(
                            1.0,
                            cfg.publish_conf_sqi_weight * sqi +
                            cfg.publish_conf_freq_weight * freq_conf +
                            cfg.publish_conf_bias,
                        ),
                    )
                    published, _, _ = quality_ctl.apply(hr_best, conf, combined_quality, now)
                    if published is not None and (
                        freq_conf >= cfg.publish_min_freq_conf_for_output
                        and sqi >= cfg.publish_min_sqi_for_output
                    ):
                        hr_published = published

            if now - last_print >= args.print_every:
                elapsed = now - t0
                out_hr = hr_published if hr_published is not None else hr_best
                cal_tag = " [CALIBRATING]" if quality_ctl.is_calibrating else ""
                if pnn50 is None:
                    print(
                        f"t={elapsed:6.1f}s fs={fs:5.1f} FPS={fps_val:5.1f} HR={out_hr:6.1f} BPM "
                        f"PPI_HR=NA pNN50=NA SQI={sqi:0.3f} FCONF={freq_conf:0.3f} "
                        f"SNR={snr_db:0.1f} state={quality_ctl.state}{cal_tag}"
                    )
                else:
                    print(
                        f"t={elapsed:6.1f}s fs={fs:5.1f} FPS={fps_val:5.1f} HR={out_hr:6.1f} BPM "
                        f"PPI_HR={ppi_hr:6.1f} BPM pNN50(exp)={pnn50:0.3f} ({pnn50*100:0.1f}%) "
                        f"SQI={sqi:0.3f} FCONF={freq_conf:0.3f} SNR={snr_db:0.1f} state={quality_ctl.state}{cal_tag}"
                    )
                last_print = now

            if args.show:
                for roi in rois:
                    h, w = frame.shape[:2]
                    x, y, rw, rh = roi.rect
                    x0, y0 = int(x * w), int(y * h)
                    x1, y1 = int((x + rw) * w), int((y + rh) * h)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    cv2.putText(frame, roi.name, (x0, max(0, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                ui_hr = hr_published if hr_published is not None else hr_best
                cv2.putText(frame, f"HR={ui_hr:0.1f} BPM", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if pnn50 is not None:
                    cv2.putText(frame, f"pNN50(exp)={pnn50:0.3f}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"{quality_ctl.state} SQI={sqi:0.2f} FC={freq_conf:0.2f}", (12, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 2)
                cv2.imshow("rPPG", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
