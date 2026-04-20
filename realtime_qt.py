#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from main import (
    Config,
    FusionEngine,
    RPPGAlgorithms,
    SignalQualityController,
    assess_signal_quality,
    choose_camera,
    create_roi_extractor,
    detect_peaks_and_pnn50,
    extract_rgb_mean,
)


class RealtimePlot(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7.5, 4.2), tight_layout=True)
        self.ax_sig = self.figure.add_subplot(2, 1, 1)
        self.ax_hr = self.figure.add_subplot(2, 1, 2)
        super().__init__(self.figure)

    def update_plot(self, green_hist: List[float], hr_hist: List[float]) -> None:
        self.ax_sig.clear()
        self.ax_hr.clear()

        if green_hist:
            g = np.array(green_hist[-240:], dtype=np.float64)
            g = (g - np.mean(g)) / (np.std(g) + 1e-9)
            self.ax_sig.plot(g, color="#42a5f5", linewidth=1.2)
        self.ax_sig.set_title("Mixed Green Signal (normalized)")
        self.ax_sig.set_ylim(-3.5, 3.5)
        self.ax_sig.grid(True, alpha=0.25)

        if hr_hist:
            self.ax_hr.plot(hr_hist[-240:], color="#ff9800", linewidth=1.3)
        self.ax_hr.set_title("Realtime HR Trend (BPM)")
        self.ax_hr.set_ylim(40, 170)
        self.ax_hr.grid(True, alpha=0.25)

        self.draw_idle()


class RealtimeWindow(QMainWindow):
    def __init__(self, defaults: argparse.Namespace) -> None:
        super().__init__()
        self.setWindowTitle("Realtime rPPG Monitor (Qt)")
        self.resize(1500, 920)

        self.defaults = defaults
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.camera = None
        self.extractor = None
        self.alg: Optional[RPPGAlgorithms] = None
        self.fusion: Optional[FusionEngine] = None
        self.quality_ctl: Optional[SignalQualityController] = None
        self.cfg: Optional[Config] = None

        self.roi_buf: Dict[str, Dict[str, Deque[float]]] = {}
        self.roi_weights: Dict[str, float] = {}
        self.t_buf: Deque[float] = deque(maxlen=5000)
        self.g_hist: Deque[float] = deque(maxlen=800)
        self.hr_hist: Deque[float] = deque(maxlen=800)

        self.hr_best = 0.0
        self.hrp_pub: Optional[float] = None
        self.ppi_hr: Optional[float] = None
        self.pnn50: Optional[float] = None
        self.sqi = 0.0
        self.freq_conf = 0.0
        self.snr_db = 0.0
        self.fs = 0.0
        self.last_plot_update = 0.0

        self._build_ui()
        self._apply_defaults()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)

        # Left control + metrics
        left = QWidget()
        left_layout = QVBoxLayout(left)

        cfg_box = QGroupBox("Runtime Config")
        form = QFormLayout(cfg_box)

        self.src_combo = QComboBox()
        self.src_combo.addItems(["auto", "realsense", "webcam"])
        form.addRow("source", self.src_combo)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(15, 120)
        form.addRow("fps", self.fps_spin)

        self.roi_mode_combo = QComboBox()
        self.roi_mode_combo.addItems(["auto", "mediapipe", "opencv"])
        form.addRow("roi_mode", self.roi_mode_combo)

        self.roi_preset_combo = QComboBox()
        self.roi_preset_combo.addItems(["hybrid7", "classic4", "cheek_forehead6", "whole_face"])
        form.addRow("roi_preset", self.roi_preset_combo)

        self.scale_x_spin = QDoubleSpinBox()
        self.scale_x_spin.setRange(0.6, 1.6)
        self.scale_x_spin.setSingleStep(0.02)
        form.addRow("roi_scale_x", self.scale_x_spin)

        self.scale_y_spin = QDoubleSpinBox()
        self.scale_y_spin.setRange(0.6, 1.6)
        self.scale_y_spin.setSingleStep(0.02)
        form.addRow("roi_scale_y", self.scale_y_spin)

        self.shift_y_spin = QDoubleSpinBox()
        self.shift_y_spin.setRange(-0.12, 0.12)
        self.shift_y_spin.setSingleStep(0.01)
        form.addRow("roi_shift_y", self.shift_y_spin)

        self.strict_combo = QComboBox()
        self.strict_combo.addItems(["false", "true"])
        form.addRow("strict_roi", self.strict_combo)

        left_layout.addWidget(cfg_box)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        left_layout.addLayout(btn_row)

        metric_box = QGroupBox("Realtime Metrics")
        metric_form = QFormLayout(metric_box)
        self.lbl_hr = QLabel("-")
        self.lbl_ppi = QLabel("-")
        self.lbl_pnn50 = QLabel("-")
        self.lbl_sqi = QLabel("-")
        self.lbl_fconf = QLabel("-")
        self.lbl_snr = QLabel("-")
        self.lbl_fs = QLabel("-")
        self.lbl_state = QLabel("-")
        metric_form.addRow("HR", self.lbl_hr)
        metric_form.addRow("PPI_HR", self.lbl_ppi)
        metric_form.addRow("pNN50(exp)", self.lbl_pnn50)
        metric_form.addRow("SQI", self.lbl_sqi)
        metric_form.addRow("FCONF", self.lbl_fconf)
        metric_form.addRow("SNR(dB)", self.lbl_snr)
        metric_form.addRow("fs(est)", self.lbl_fs)
        metric_form.addRow("state", self.lbl_state)
        left_layout.addWidget(metric_box)
        left_layout.addStretch(1)

        # Right video + chart
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.video_label = QLabel("Camera preview")
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border:1px solid #333;")
        right_layout.addWidget(self.video_label, 3)

        self.plot = RealtimePlot()
        right_layout.addWidget(self.plot, 2)

        root_layout.addWidget(left, 1)
        root_layout.addWidget(right, 3)
        self.setCentralWidget(root)

    def _apply_defaults(self) -> None:
        self.src_combo.setCurrentText(self.defaults.source)
        self.fps_spin.setValue(self.defaults.fps)
        self.roi_mode_combo.setCurrentText(self.defaults.roi_mode)
        self.roi_preset_combo.setCurrentText(self.defaults.roi_preset)
        self.scale_x_spin.setValue(self.defaults.roi_scale_x)
        self.scale_y_spin.setValue(self.defaults.roi_scale_y)
        self.shift_y_spin.setValue(self.defaults.roi_shift_y)
        self.strict_combo.setCurrentText("true" if self.defaults.strict_roi else "false")

    def _reset_buffers(self) -> None:
        self.roi_buf.clear()
        self.roi_weights.clear()
        self.t_buf.clear()
        self.g_hist.clear()
        self.hr_hist.clear()
        self.hr_best = 0.0
        self.hrp_pub = None
        self.ppi_hr = None
        self.pnn50 = None
        self.sqi = 0.0
        self.freq_conf = 0.0
        self.snr_db = 0.0
        self.fs = 0.0
        self.last_plot_update = 0.0

    def _build_config(self) -> Config:
        cfg = Config()
        cfg.roi_preset = self.roi_preset_combo.currentText()
        cfg.roi_scale_x = float(self.scale_x_spin.value())
        cfg.roi_scale_y = float(self.scale_y_spin.value())
        cfg.roi_shift_y = float(self.shift_y_spin.value())
        return cfg

    def start(self) -> None:
        if self.timer.isActive():
            return
        try:
            self.cfg = self._build_config()
            source = self.src_combo.currentText()
            fps = int(self.fps_spin.value())
            self.camera = choose_camera(source, fps)
            self.extractor, roi_used = create_roi_extractor(
                self.roi_mode_combo.currentText(),
                mp_face_detector_model=self.defaults.mp_face_detector_model,
                strict=(self.strict_combo.currentText() == "true"),
                roi_preset=self.cfg.roi_preset,
                roi_scale_x=self.cfg.roi_scale_x,
                roi_scale_y=self.cfg.roi_scale_y,
                roi_shift_y=self.cfg.roi_shift_y,
            )
            self.alg = RPPGAlgorithms(self.cfg)
            self.fusion = FusionEngine(self.cfg)
            self.quality_ctl = SignalQualityController(publish_interval=1.0)
            self._reset_buffers()
            self.timer.start(1)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.statusBar().showMessage(f"Running: source={source}, roi={roi_used}, preset={self.cfg.roi_preset}")
        except Exception as e:
            self.stop()
            QMessageBox.critical(self, "Start Failed", str(e))

    def stop(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        self.camera = None
        self.extractor = None
        self.alg = None
        self.fusion = None
        self.quality_ctl = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("Stopped")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop()
        super().closeEvent(event)

    def _update_metrics_ui(self) -> None:
        out_hr = self.hrp_pub if self.hrp_pub is not None else self.hr_best
        self.lbl_hr.setText(f"{out_hr:0.1f} bpm" if out_hr > 0 else "-")
        self.lbl_ppi.setText(f"{self.ppi_hr:0.1f} bpm" if self.ppi_hr is not None else "-")
        self.lbl_pnn50.setText(f"{self.pnn50:0.3f}" if self.pnn50 is not None else "-")
        self.lbl_sqi.setText(f"{self.sqi:0.3f}")
        self.lbl_fconf.setText(f"{self.freq_conf:0.3f}")
        self.lbl_snr.setText(f"{self.snr_db:0.1f}")
        self.lbl_fs.setText(f"{self.fs:0.1f}")
        self.lbl_state.setText(self.quality_ctl.state if self.quality_ctl is not None else "-")

    def _tick(self) -> None:
        if self.camera is None or self.extractor is None or self.alg is None or self.fusion is None or self.quality_ctl is None or self.cfg is None:
            return

        frame = self.camera.read()
        if frame is None:
            return

        now = time.time()
        self.t_buf.append(now)
        rois = self.extractor.extract(frame)

        for roi in rois:
            rgb = extract_rgb_mean(frame, roi)
            if rgb is None:
                continue
            if roi.name not in self.roi_buf:
                max_len = int(self.cfg.buffer_sec * max(10, int(self.fps_spin.value())))
                self.roi_buf[roi.name] = {
                    "r": deque(maxlen=max_len),
                    "g": deque(maxlen=max_len),
                    "b": deque(maxlen=max_len),
                }
            self.roi_buf[roi.name]["r"].append(rgb[0])
            self.roi_buf[roi.name]["g"].append(rgb[1])
            self.roi_buf[roi.name]["b"].append(rgb[2])
            self.roi_weights[roi.name] = roi.weight

        if len(self.t_buf) > 40:
            dt = np.diff(np.array(self.t_buf, dtype=np.float64)[-120:])
            self.fs = float(1.0 / max(1e-6, np.median(dt)))
        else:
            self.fs = float(self.fps_spin.value())

        tagged: List[Tuple[float, str, str]] = []
        merged_rgb: Dict[str, List[np.ndarray]] = {"r": [], "g": [], "b": []}
        merged_w: List[float] = []

        min_len = int(max(self.cfg.welch_seg_sec * self.fs, self.cfg.pos_window_sec * self.fs) + 5)
        for name, buf in self.roi_buf.items():
            if len(buf["g"]) < min_len:
                continue
            r = np.array(buf["r"], dtype=np.float64)
            g = np.array(buf["g"], dtype=np.float64)
            b = np.array(buf["b"], dtype=np.float64)

            hr_g, _ = self.alg.green(r, g, b, self.fs)
            hr_c, _ = self.alg.chrom(r, g, b, self.fs)
            hr_p, _ = self.alg.pos(r, g, b, self.fs)

            if self.cfg.min_bpm_valid <= hr_g <= self.cfg.max_bpm_valid:
                tagged.append((hr_g, "GREEN", name))
            if self.cfg.min_bpm_valid <= hr_c <= self.cfg.max_bpm_valid:
                tagged.append((hr_c, "CHROM", name))
            if self.cfg.min_bpm_valid <= hr_p <= self.cfg.max_bpm_valid:
                tagged.append((hr_p, "POS", name))

            merged_rgb["r"].append(r)
            merged_rgb["g"].append(g)
            merged_rgb["b"].append(b)
            merged_w.append(self.roi_weights.get(name, 1.0))

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

                self.g_hist.append(float(g_mix[-1]))
                qsig = self.alg.bandpass(self.alg.robust_norm(self.alg.detrend_poly2(g_mix)), self.fs)
                self.sqi, self.snr_db = assess_signal_quality(qsig, self.fs)
                _, bvp = self.alg.pos(r_mix, g_mix, b_mix, self.fs)
                self.ppi_hr, self.pnn50, _ = detect_peaks_and_pnn50(bvp, self.fs)

        if tagged:
            hr_raw, self.freq_conf = self.fusion.harmonic_temporal_fusion(tagged)
            hr_final = self.fusion.apply_physiological_constraints(hr_raw)
            if self.cfg.enable_ppi_assist:
                hr_final = self.fusion.apply_ppi_assist(hr_final, self.ppi_hr, self.sqi, freq_conf=self.freq_conf)
            self.hr_best = self.fusion.update_and_get_best(hr_final)
            self.hr_hist.append(self.hr_best)

        self.hrp_pub = None
        if self.hr_best > 0:
            combined_quality = max(0.0, min(1.0, 0.7 * self.sqi + 0.3 * self.freq_conf))
            if self.freq_conf < self.cfg.freq_conf_gate:
                combined_quality *= 0.75
            conf = max(0.0, min(1.0, 0.45 * self.sqi + 0.35 * self.freq_conf + 0.20))
            published, _, _ = self.quality_ctl.apply(self.hr_best, conf, combined_quality, now)
            if published is not None:
                self.hrp_pub = published

        # Draw ROIs and labels.
        h, w = frame.shape[:2]
        for roi in rois:
            x, y, rw, rh = roi.rect
            x0, y0 = int(x * w), int(y * h)
            x1, y1 = int((x + rw) * w), int((y + rh) * h)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 230, 90), 1)
            cv2.putText(
                frame,
                f"{roi.name}:{roi.weight:.2f}",
                (x0, max(10, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (40, 230, 90),
                1,
                cv2.LINE_AA,
            )

        # Frame overlay
        out_hr = self.hrp_pub if self.hrp_pub is not None else self.hr_best
        cv2.putText(frame, f"HR={out_hr:0.1f} bpm", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)
        cv2.putText(frame, f"SQI={self.sqi:0.2f} FC={self.freq_conf:0.2f} {self.quality_ctl.state}", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 220, 220), 2)
        if self.ppi_hr is not None:
            cv2.putText(frame, f"PPI={self.ppi_hr:0.1f}", (12, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 255, 180), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self._update_metrics_ui()
        if now - self.last_plot_update >= 0.25:
            self.plot.update_plot(list(self.g_hist), list(self.hr_hist))
            self.last_plot_update = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Qt UI for camera-based rPPG")
    parser.add_argument("--source", choices=["auto", "realsense", "webcam"], default="auto")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--roi-mode", choices=["auto", "mediapipe", "opencv"], default="opencv")
    parser.add_argument("--roi-preset", choices=["hybrid7", "classic4", "cheek_forehead6", "whole_face"], default="hybrid7")
    parser.add_argument("--roi-scale-x", type=float, default=1.1)
    parser.add_argument("--roi-scale-y", type=float, default=1.1)
    parser.add_argument("--roi-shift-y", type=float, default=0.0)
    parser.add_argument("--strict-roi", action="store_true")
    parser.add_argument("--mp-face-detector-model", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication(sys.argv)
    win = RealtimeWindow(args)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
