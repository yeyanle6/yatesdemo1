#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


@dataclass
class SummaryRow:
    group: str
    stem: str
    roi_mode_requested: str
    roi_mode_used: str
    video_path: str
    ecg_csv: str
    n: int
    mae: float
    rmse: float
    mape: float
    corr: float


def _to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return math.nan


def _to_int(v: str) -> int:
    try:
        return int(float(v))
    except Exception:
        return 0


def _safe_text(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.6g}"
    return str(v)


class MatplotlibPanel(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.ax_line = self.figure.add_subplot(2, 1, 1)
        self.ax_scatter = self.figure.add_subplot(2, 1, 2)
        super().__init__(self.figure)

    def clear(self) -> None:
        self.ax_line.clear()
        self.ax_scatter.clear()
        self.draw_idle()

    def plot_series(self, secs: List[int], ecg: List[float], rppg: List[float]) -> None:
        self.ax_line.clear()
        self.ax_scatter.clear()

        self.ax_line.plot(secs, ecg, label="ECG HR", linewidth=1.5)
        self.ax_line.plot(secs, rppg, label="rPPG HR", linewidth=1.5)
        self.ax_line.set_title("ECG vs rPPG Over Time")
        self.ax_line.set_xlabel("sec")
        self.ax_line.set_ylabel("BPM")
        self.ax_line.grid(True, alpha=0.3)
        self.ax_line.legend(loc="best")

        self.ax_scatter.scatter(ecg, rppg, s=14, alpha=0.7)
        if ecg:
            lo = min(min(ecg), min(rppg))
            hi = max(max(ecg), max(rppg))
            self.ax_scatter.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
        self.ax_scatter.set_title("Scatter: ECG vs rPPG")
        self.ax_scatter.set_xlabel("ECG HR")
        self.ax_scatter.set_ylabel("rPPG HR")
        self.ax_scatter.grid(True, alpha=0.3)

        self.draw_idle()


class ResultAnalyzerWindow(QMainWindow):
    def __init__(self, initial_dir: Path) -> None:
        super().__init__()
        self.setWindowTitle("rPPG Result Analyzer (Qt)")
        self.resize(1400, 900)

        self.results_dir = initial_dir
        self.summary_files: List[Path] = []
        self.current_summary_file: Optional[Path] = None
        self.current_detail_file: Optional[Path] = None
        self.summary_rows: List[SummaryRow] = []
        self.detail_rows: List[Dict[str, str]] = []
        self.filtered_indices: List[int] = []

        self._build_ui()
        self._refresh_summary_files()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.dir_edit = QLineEdit(str(self.results_dir))
        self.dir_edit.setReadOnly(True)
        btn_browse = QPushButton("Choose Results Dir")
        btn_browse.clicked.connect(self._choose_dir)
        top.addWidget(QLabel("Results:"))
        top.addWidget(self.dir_edit, 1)
        top.addWidget(btn_browse)
        root.addLayout(top)

        bar = QHBoxLayout()
        self.summary_combo = QComboBox()
        self.summary_combo.currentIndexChanged.connect(self._load_selected_summary)
        self.filter_group = QComboBox()
        self.filter_group.addItem("ALL")
        self.filter_group.currentIndexChanged.connect(self._apply_filter)
        self.filter_roi = QComboBox()
        self.filter_roi.addItem("ALL")
        self.filter_roi.currentIndexChanged.connect(self._apply_filter)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("search stem/video...")
        self.search_edit.textChanged.connect(self._apply_filter)
        btn_export = QPushButton("Export Filtered Summary CSV")
        btn_export.clicked.connect(self._export_filtered_summary)

        bar.addWidget(QLabel("Summary File:"))
        bar.addWidget(self.summary_combo, 2)
        bar.addWidget(QLabel("Group:"))
        bar.addWidget(self.filter_group)
        bar.addWidget(QLabel("ROI Used:"))
        bar.addWidget(self.filter_roi)
        bar.addWidget(self.search_edit, 2)
        bar.addWidget(btn_export)
        root.addLayout(bar)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.summary_table = QTableWidget(0, 9)
        self.summary_table.setHorizontalHeaderLabels(
            ["group", "stem", "roi_req", "roi_used", "n", "mae", "rmse", "mape", "corr"]
        )
        self.summary_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.summary_table.setSelectionMode(QTableWidget.SingleSelection)
        self.summary_table.itemSelectionChanged.connect(self._on_summary_row_selected)
        self.summary_table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.summary_table, 1)

        metric_box = QGroupBox("Selected Sample Stats")
        metric_grid = QGridLayout(metric_box)
        self.lbl_selected = QLabel("-")
        self.lbl_points = QLabel("-")
        self.lbl_mae = QLabel("-")
        self.lbl_rmse = QLabel("-")
        self.lbl_mape = QLabel("-")
        self.lbl_corr = QLabel("-")
        self.lbl_state = QLabel("-")
        metric_grid.addWidget(QLabel("sample"), 0, 0)
        metric_grid.addWidget(self.lbl_selected, 0, 1)
        metric_grid.addWidget(QLabel("points"), 1, 0)
        metric_grid.addWidget(self.lbl_points, 1, 1)
        metric_grid.addWidget(QLabel("MAE"), 2, 0)
        metric_grid.addWidget(self.lbl_mae, 2, 1)
        metric_grid.addWidget(QLabel("RMSE"), 3, 0)
        metric_grid.addWidget(self.lbl_rmse, 3, 1)
        metric_grid.addWidget(QLabel("MAPE"), 4, 0)
        metric_grid.addWidget(self.lbl_mape, 4, 1)
        metric_grid.addWidget(QLabel("Corr"), 5, 0)
        metric_grid.addWidget(self.lbl_corr, 5, 1)
        metric_grid.addWidget(QLabel("States"), 6, 0)
        metric_grid.addWidget(self.lbl_state, 6, 1)
        left_layout.addWidget(metric_box)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.plot_panel = MatplotlibPanel()
        right_layout.addWidget(self.plot_panel, 1)

        self.detail_table = QTableWidget(0, 8)
        self.detail_table.setHorizontalHeaderLabels(
            ["sec", "ecg_hr", "rppg_hr", "error", "abs_error", "ape%", "state", "sqi"]
        )
        self.detail_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.detail_table, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(central)

    def _choose_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose result directory", str(self.results_dir))
        if not d:
            return
        self.results_dir = Path(d)
        self.dir_edit.setText(str(self.results_dir))
        self._refresh_summary_files()

    def _refresh_summary_files(self) -> None:
        self.summary_files = sorted(self.results_dir.glob("rppg_ecg_summary_*.csv"))
        self.summary_combo.blockSignals(True)
        self.summary_combo.clear()
        for p in self.summary_files:
            self.summary_combo.addItem(p.name, str(p))
        self.summary_combo.blockSignals(False)

        if not self.summary_files:
            QMessageBox.warning(self, "No Summary CSV", f"No summary files found in:\n{self.results_dir}")
            self.summary_rows = []
            self.detail_rows = []
            self._render_summary_table()
            self.plot_panel.clear()
            return
        self.summary_combo.setCurrentIndex(0)
        self._load_selected_summary()

    def _resolve_detail_file(self, summary_file: Path) -> Optional[Path]:
        name = summary_file.name.replace("summary", "comparison")
        p = summary_file.parent / name
        return p if p.exists() else None

    def _load_csv_dicts(self, path: Path) -> List[Dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))

    def _load_selected_summary(self) -> None:
        p_text = self.summary_combo.currentData()
        if not p_text:
            return
        summary_file = Path(p_text)
        detail_file = self._resolve_detail_file(summary_file)
        if detail_file is None:
            QMessageBox.warning(self, "Missing Detail CSV", f"Detail file not found for:\n{summary_file.name}")
            return

        self.current_summary_file = summary_file
        self.current_detail_file = detail_file

        rows = self._load_csv_dicts(summary_file)
        self.summary_rows = []
        for r in rows:
            self.summary_rows.append(
                SummaryRow(
                    group=r.get("group", ""),
                    stem=r.get("stem", ""),
                    roi_mode_requested=r.get("roi_mode_requested", r.get("roi_mode", "")),
                    roi_mode_used=r.get("roi_mode_used", r.get("roi_mode", "")),
                    video_path=r.get("video_path", ""),
                    ecg_csv=r.get("ecg_csv", ""),
                    n=_to_int(r.get("n", "0")),
                    mae=_to_float(r.get("mae", "nan")),
                    rmse=_to_float(r.get("rmse", "nan")),
                    mape=_to_float(r.get("mape", "nan")),
                    corr=_to_float(r.get("corr", "nan")),
                )
            )

        self.detail_rows = self._load_csv_dicts(detail_file)
        self._rebuild_filter_options()
        self._apply_filter()

    def _rebuild_filter_options(self) -> None:
        groups = sorted({r.group for r in self.summary_rows if r.group and r.group != "ALL"})
        rois = sorted({r.roi_mode_used for r in self.summary_rows if r.roi_mode_used and r.group != "ALL"})

        self.filter_group.blockSignals(True)
        self.filter_group.clear()
        self.filter_group.addItem("ALL")
        for g in groups:
            self.filter_group.addItem(g)
        self.filter_group.blockSignals(False)

        self.filter_roi.blockSignals(True)
        self.filter_roi.clear()
        self.filter_roi.addItem("ALL")
        for r in rois:
            self.filter_roi.addItem(r)
        self.filter_roi.blockSignals(False)

    def _apply_filter(self) -> None:
        group_filter = self.filter_group.currentText()
        roi_filter = self.filter_roi.currentText()
        q = self.search_edit.text().strip().lower()

        self.filtered_indices = []
        for i, r in enumerate(self.summary_rows):
            if r.group == "ALL":
                continue
            if group_filter != "ALL" and r.group != group_filter:
                continue
            if roi_filter != "ALL" and r.roi_mode_used != roi_filter:
                continue
            if q:
                s = f"{r.stem} {Path(r.video_path).name}".lower()
                if q not in s:
                    continue
            self.filtered_indices.append(i)
        self._render_summary_table()

    def _render_summary_table(self) -> None:
        self.summary_table.setRowCount(0)
        for row_idx, idx in enumerate(self.filtered_indices):
            r = self.summary_rows[idx]
            self.summary_table.insertRow(row_idx)
            vals = [
                r.group,
                r.stem,
                r.roi_mode_requested,
                r.roi_mode_used,
                r.n,
                r.mae,
                r.rmse,
                r.mape,
                r.corr,
            ]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(_safe_text(v))
                if col >= 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.summary_table.setItem(row_idx, col, item)
        self.summary_table.resizeColumnsToContents()
        if self.summary_table.rowCount() > 0:
            self.summary_table.selectRow(0)
        else:
            self.plot_panel.clear()
            self.detail_table.setRowCount(0)

    def _collect_detail_for_sample(self, group: str, stem: str, roi_mode: str) -> List[Dict[str, str]]:
        out = []
        for r in self.detail_rows:
            if r.get("group", "") != group:
                continue
            if r.get("stem", "") != stem:
                continue
            if r.get("roi_mode", r.get("roi_mode_used", "")) != roi_mode:
                continue
            out.append(r)
        out.sort(key=lambda x: _to_int(x.get("sec", "0")))
        return out

    def _on_summary_row_selected(self) -> None:
        selected = self.summary_table.selectionModel().selectedRows()
        if not selected:
            return
        table_row = selected[0].row()
        if table_row < 0 or table_row >= len(self.filtered_indices):
            return
        idx = self.filtered_indices[table_row]
        s = self.summary_rows[idx]

        rows = self._collect_detail_for_sample(s.group, s.stem, s.roi_mode_used)
        self._render_detail(rows, s)

    def _render_detail(self, rows: List[Dict[str, str]], s: SummaryRow) -> None:
        self.lbl_selected.setText(f"{s.group}/{s.stem} ({s.roi_mode_used})")
        self.lbl_points.setText(str(len(rows)))

        ecg = [_to_float(r.get("ecg_hr", "nan")) for r in rows]
        rppg = [_to_float(r.get("rppg_hr", "nan")) for r in rows]
        err = [_to_float(r.get("error", "nan")) for r in rows]
        mape = [_to_float(r.get("ape_percent", "nan")) for r in rows]
        valid_err = [x for x in err if not math.isnan(x)]
        valid_mape = [x for x in mape if not math.isnan(x)]

        mae = sum(abs(x) for x in valid_err) / len(valid_err) if valid_err else math.nan
        rmse = math.sqrt(sum(x * x for x in valid_err) / len(valid_err)) if valid_err else math.nan
        mape_v = sum(valid_mape) / len(valid_mape) if valid_mape else math.nan
        corr = math.nan
        if len(ecg) >= 2:
            pairs = [(x, y) for x, y in zip(ecg, rppg) if not math.isnan(x) and not math.isnan(y)]
            if len(pairs) >= 2:
                xs = [p[0] for p in pairs]
                ys = [p[1] for p in pairs]
                mx = sum(xs) / len(xs)
                my = sum(ys) / len(ys)
                cov = sum((x - mx) * (y - my) for x, y in pairs)
                vx = sum((x - mx) ** 2 for x in xs)
                vy = sum((y - my) ** 2 for y in ys)
                if vx > 0 and vy > 0:
                    corr = cov / math.sqrt(vx * vy)

        states = sorted({r.get("state", "") for r in rows if r.get("state", "")})
        self.lbl_mae.setText(_safe_text(mae))
        self.lbl_rmse.setText(_safe_text(rmse))
        self.lbl_mape.setText(_safe_text(mape_v))
        self.lbl_corr.setText(_safe_text(corr))
        self.lbl_state.setText(", ".join(states) if states else "-")

        secs = [_to_int(r.get("sec", "0")) for r in rows]
        ecg_series = [(_to_float(r.get("ecg_hr", "nan"))) for r in rows]
        rppg_series = [(_to_float(r.get("rppg_hr", "nan"))) for r in rows]
        valid_idx = [i for i, (a, b) in enumerate(zip(ecg_series, rppg_series)) if not math.isnan(a) and not math.isnan(b)]
        p_secs = [secs[i] for i in valid_idx]
        p_ecg = [ecg_series[i] for i in valid_idx]
        p_rppg = [rppg_series[i] for i in valid_idx]
        if p_secs:
            self.plot_panel.plot_series(p_secs, p_ecg, p_rppg)
        else:
            self.plot_panel.clear()

        cols = ["sec", "ecg_hr", "rppg_hr", "error", "abs_error", "ape_percent", "state", "sqi"]
        self.detail_table.setRowCount(0)
        for row_idx, r in enumerate(rows):
            self.detail_table.insertRow(row_idx)
            for col_idx, key in enumerate(cols):
                item = QTableWidgetItem(_safe_text(r.get(key, "")))
                if key not in ("state",):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.detail_table.setItem(row_idx, col_idx, item)
        self.detail_table.resizeColumnsToContents()

    def _export_filtered_summary(self) -> None:
        if not self.filtered_indices:
            QMessageBox.information(self, "No Data", "No filtered rows to export.")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Save filtered summary",
            str(self.results_dir / "filtered_summary.csv"),
            "CSV Files (*.csv)",
        )
        if not target:
            return
        cols = [
            "group",
            "stem",
            "roi_mode_requested",
            "roi_mode_used",
            "video_path",
            "ecg_csv",
            "n",
            "mae",
            "rmse",
            "mape",
            "corr",
        ]
        with Path(target).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for idx in self.filtered_indices:
                r = self.summary_rows[idx]
                writer.writerow(
                    {
                        "group": r.group,
                        "stem": r.stem,
                        "roi_mode_requested": r.roi_mode_requested,
                        "roi_mode_used": r.roi_mode_used,
                        "video_path": r.video_path,
                        "ecg_csv": r.ecg_csv,
                        "n": r.n,
                        "mae": r.mae,
                        "rmse": r.rmse,
                        "mape": r.mape,
                        "corr": r.corr,
                    }
                )
        QMessageBox.information(self, "Exported", f"Saved:\n{target}")


def main() -> None:
    if len(sys.argv) > 1:
        start_dir = Path(sys.argv[1]).expanduser().resolve()
    else:
        start_dir = Path(__file__).resolve().parent / "results"
    app = QApplication(sys.argv)
    w = ResultAnalyzerWindow(start_dir)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

