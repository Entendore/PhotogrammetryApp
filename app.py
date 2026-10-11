#!/usr/bin/env python3
"""
Photogrammetry Application - PySide6
Full pipeline: Feature Detection → Matching → Sparse 3D Reconstruction → Point Cloud Viewer
"""

import sys, os, numpy as np
from pathlib import Path
from itertools import combinations

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem,
    QSplitter, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QTabWidget, QMessageBox, QDockWidget, QCheckBox,
    QTextEdit, QFrame, QGridLayout, QSizePolicy
)
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QAction, QFont, QColor

import cv2

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure


# ═══════════════════════════════════════════════════════════════════════
#  UTILITY
# ═══════════════════════════════════════════════════════════════════════

def cv2_to_qpixmap(cv_img, max_size=600):
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
    elif cv_img.shape[2] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGB)
    else:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w = cv_img.shape[:2]
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)))
    h, w, ch = cv_img.shape
    return QPixmap.fromImage(QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888))


def generate_example_data(output_dir, num_images=10):
    """Generate synthetic images of a 3D scene for demo."""
    os.makedirs(output_dir, exist_ok=True)
    img_size, focal = 800, 600
    cx, cy = img_size / 2, img_size / 2

    np.random.seed(42)
    n_pts = 300
    pts3d, pt_cols = [], []
    # Points on cube faces
    for _ in range(n_pts):
        face = np.random.randint(6)
        u, v = np.random.rand(2) * 2 - 1
        s = 1.8
        if   face == 0: pts3d.append([u*s, v*s,  s])
        elif face == 1: pts3d.append([u*s, v*s, -s])
        elif face == 2: pts3d.append([u*s,  s, v*s])
        elif face == 3: pts3d.append([u*s, -s, v*s])
        elif face == 4: pts3d.append([ s, u*s, v*s])
        else:           pts3d.append([-s, u*s, v*s])
        pt_cols.append(np.random.randint(60, 255, 3).tolist())
    pts3d = np.array(pts3d, dtype=np.float64)
    pt_cols = np.array(pt_cols, dtype=np.uint8)

    for i in range(num_images):
        img = np.full((img_size, img_size, 3), 235, dtype=np.uint8)
        ang = (i / num_images) * 2 * np.pi
        cam_d = 7
        cam_pos = np.array([cam_d * np.sin(ang),
                            2.0 + np.sin(ang * 0.5),
                            cam_d * np.cos(ang)])
        fwd = -cam_pos / np.linalg.norm(cam_pos)
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        # Draw grid on ground
        for g in np.linspace(-3, 3, 13):
            for t in np.linspace(-3, 3, 50):
                for p1, p2 in [([g, 0, t], [g, 0, t+0.12]),
                               ([t, 0, g], [t+0.12, 0, g])]:
                    for p in (p1, p2):
                        d = np.array(p) - cam_pos
                        x_, y_, z_ = d@right, d@up, d@fwd
                        if z_ > 0.1:
                            px = int(focal*x_/z_ + cx)
                            py = int(-focal*y_/z_ + cy)
                            if 0 <= px < img_size and 0 <= py < img_size:
                                cv2.circle(img, (px, py), 2, (180, 180, 180), -1)

        # Draw 3D points
        for pt, col in zip(pts3d, pt_cols):
            d = pt - cam_pos
            x_, y_, z_ = d@right, d@up, d@fwd
            if z_ > 0.3:
                px, py = int(focal*x_/z_ + cx), int(-focal*y_/z_ + cy)
                if 0 <= px < img_size and 0 <= py < img_size:
                    cv2.circle(img, (px, py), 6, col.tolist(), -1)
                    cv2.circle(img, (px, py), 6, (40, 40, 40), 1)

        cv2.putText(img, f"View {i+1}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (50, 50, 50), 3)
        noise = np.random.randint(0, 8, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        cv2.imwrite(os.path.join(output_dir, f"view_{i+1:02d}.png"), img)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════
#  PHOTOGARAMMETRY PIPELINE
# ═══════════════════════════════════════════════════════════════════════

class PhotogrammetryPipeline:
    def __init__(self):
        self.images, self.image_paths, self.gray_images = [], [], []
        self.keypoints, self.descriptors = [], []
        self.matches = {}
        self.point_cloud = self.point_colors = self.K = None
        self.detector_type = "SIFT"
        self.max_features = 5000
        self.ratio_thresh = 0.75
        self.reproj_err = 4.0

    # ── Load ──────────────────────────────────────────────────────────
    def load_images(self, paths):
        self.images, self.image_paths, self.gray_images = [], [], []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            if max(h, w) > 1600:
                img = cv2.resize(img, (int(w * 1600 / max(h, w)),
                                       int(h * 1600 / max(h, w))))
            self.images.append(img)
            self.image_paths.append(p)
            self.gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        return len(self.images)

    # ── Detect ────────────────────────────────────────────────────────
    def detect_features(self, callback=None):
        self.keypoints, self.descriptors = [], []
        det = (cv2.SIFT_create(self.max_features) if self.detector_type == "SIFT"
               else cv2.ORB_create(self.max_features))
        for i, g in enumerate(self.gray_images):
            kp, des = det.detectAndCompute(g, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)
            if callback:
                callback(i + 1, len(self.gray_images))

    # ── Match ─────────────────────────────────────────────────────────
    def match_features(self, callback=None):
        self.matches = {}
        n = len(self.images)
        norm = cv2.NORM_HAMMING if self.detector_type == "ORB" else cv2.NORM_L2
        bf = cv2.BFMatcher(norm)
        pairs = list(combinations(range(n), 2))
        for idx, (i, j) in enumerate(pairs):
            if self.descriptors[i] is None or self.descriptors[j] is None:
                continue
            try:
                raw = bf.knnMatch(self.descriptors[i], self.descriptors[j], k=2)
                good = [m for m, n_ in raw
                        if m.distance < self.ratio_thresh * n_.distance]
                if len(good) >= 8:
                    self.matches[(i, j)] = good
            except Exception:
                pass
            if callback:
                callback(idx + 1, len(pairs))

    # ── Estimate Intrinsics ──────────────────────────────────────────
    def estimate_intrinsics(self):
        if not self.images:
            return
        h, w = self.images[0].shape[:2]
        f = max(w, h)
        self.K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)

    # ── Reconstruct ──────────────────────────────────────────────────
    def reconstruct(self, callback=None):
        if not self.matches or self.K is None:
            return None, None
        all_pts, all_cols = [], []
        sorted_pairs = sorted(self.matches.keys(),
                              key=lambda k: len(self.matches[k]), reverse=True)
        for pidx, (i, j) in enumerate(sorted_pairs[:25]):
            gm = self.matches[(i, j)]
            pts1 = np.float64([self.keypoints[i][m.queryIdx].pt for m in gm])
            pts2 = np.float64([self.keypoints[j][m.trainIdx].pt for m in gm])
            if len(pts1) < 10:
                continue
            E, _ = cv2.findEssentialMat(pts1, pts2, self.K,
                                        method=cv2.RANSAC,
                                        prob=0.999,
                                        threshold=self.reproj_err)
            if E is None or E.shape != (3, 3):
                continue
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
            inl = mask.ravel().astype(bool)
            p1, p2 = pts1[inl], pts2[inl]
            if len(p1) < 5:
                continue
            P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
            P2 = self.K @ np.hstack((R, t))
            pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
            pts3 = cv2.convertPointsFromHomogeneous(pts4d.T).reshape(-1, 3)
            # Cheirality
            valid = pts3[:, 2] > 0
            pts_cam2 = (R @ pts3.T + t).T
            valid &= pts_cam2[:, 2] > 0
            if valid.sum() < 5:
                continue
            dists = np.linalg.norm(pts3, axis=1)
            med = np.median(dists[valid])
            valid &= dists < med * 6
            pts3, p1v = pts3[valid], p1[valid]
            h, w = self.images[i].shape[:2]
            for pt in p1v:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= y < h and 0 <= x < w:
                    b, g, r = self.images[i][y, x]
                    all_cols.append([r/255, g/255, b/255])
                else:
                    all_cols.append([.5, .5, .5])
            all_pts.append(pts3)
            if callback:
                callback(pidx + 1, min(len(sorted_pairs), 25))
        if all_pts:
            self.point_cloud = np.vstack(all_pts)
            self.point_colors = np.array(all_cols)
            self._remove_outliers()
            return self.point_cloud, self.point_colors
        return None, None

    def _remove_outliers(self):
        if self.point_cloud is None or len(self.point_cloud) < 10:
            return
        med = np.median(self.point_cloud, axis=0)
        d = np.linalg.norm(self.point_cloud - med, axis=1)
        ok = d < np.percentile(d, 95)
        self.point_cloud, self.point_colors = self.point_cloud[ok], self.point_colors[ok]

    # ── Export PLY ────────────────────────────────────────────────────
    def export_ply(self, path):
        if self.point_cloud is None:
            return False
        with open(path, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(self.point_cloud)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(len(self.point_cloud)):
                x, y, z = self.point_cloud[i]
                r, g, b = (self.point_colors[i] * 255).astype(int) if self.point_colors is not None else (200,200,200)
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")
        return True


# ═══════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════

class PipelineWorker(QThread):
    progress = Signal(int, int)   # current, total
    log = Signal(str)
    finished = Signal(str)        # status message

    def __init__(self, pipeline, steps):
        super().__init__()
        self.pipeline = pipeline
        self.steps = steps  # list of step names

    def run(self):
        try:
            for step in self.steps:
                if step == "detect":
                    self.log.emit("🔍 Detecting features …")
                    self.pipeline.detect_features(
                        lambda c, t: self.progress.emit(c, t))
                    total_kp = sum(len(k) for k in self.pipeline.keypoints)
                    self.log.emit(f"   Found {total_kp} keypoints across "
                                  f"{len(self.pipeline.images)} images")

                elif step == "match":
                    self.log.emit("🔗 Matching features …")
                    self.pipeline.match_features(
                        lambda c, t: self.progress.emit(c, t))
                    self.log.emit(f"   {len(self.pipeline.matches)} image pairs matched")

                elif step == "intrinsics":
                    self.log.emit("📐 Estimating camera intrinsics …")
                    self.pipeline.estimate_intrinsics()
                    self.log.emit(f"   Focal length ≈ {self.pipeline.K[0,0]:.0f} px")

                elif step == "reconstruct":
                    self.log.emit("🏗️  Reconstructing 3D points …")
                    pc, cols = self.pipeline.reconstruct(
                        lambda c, t: self.progress.emit(c, t))
                    if pc is not None:
                        self.log.emit(f"   Point cloud: {len(pc)} points")
                    else:
                        self.log.emit("   ⚠ Reconstruction failed – try more images or different settings")

            self.finished.emit("done")
        except Exception as e:
            self.log.emit(f"❌ Error: {e}")
            self.finished.emit("error")


# ═══════════════════════════════════════════════════════════════════════
#  POINT CLOUD VIEWER  (Matplotlib 3D)
# ═══════════════════════════════════════════════════════════════════════

class PointCloudViewer(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6, 5), dpi=100, facecolor="#1e1e2e")
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d", facecolor="#1e1e2e")
        self._style_axes()
        self.point_size = 2

    def _style_axes(self):
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.label.set_color("white")
            axis.set_pane_color((0.12, 0.12, 0.18, 1.0))
            axis._axinfo["tick"]["color"] = (1, 1, 1, 0.5)
            axis._axinfo["grid"]["color"] = (1, 1, 1, 0.15)
        self.ax.tick_params(colors="white", labelsize=7)
        self.ax.set_xlabel("X", color="white")
        self.ax.set_ylabel("Y", color="white")
        self.ax.set_zlabel("Z", color="white")

    def show_point_cloud(self, pts, colors=None):
        self.ax.cla()
        self._style_axes()
        if pts is None or len(pts) == 0:
            self.ax.set_title("No point cloud data", color="white")
            self.draw()
            return
        c = colors if colors is not None else np.ones((len(pts), 3)) * 0.8
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=c, s=self.point_size, depthshade=True)
        # Auto-scale
        margin = 0.1
        for i, setter in enumerate([self.ax.set_xlim, self.ax.set_ylim, self.ax.set_zlim]):
            lo, hi = pts[:, i].min(), pts[:, i].max()
            span = hi - lo if hi - lo > 0 else 1
            setter(lo - margin * span, hi + margin * span)
        self.ax.set_title(f"Point Cloud  ({len(pts)} points)", color="white", fontsize=10)
        self.draw()


# ═══════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow { background: #1e1e2e; }
QDockWidget { color: #cdd6f4; font-weight: bold; }
QDockWidget::title { background: #313244; padding: 6px; }
QWidget { background: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', sans-serif; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 14px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #45475a; border: none; border-radius: 5px; padding: 8px 16px; color: #cdd6f4; font-weight: bold; }
QPushButton:hover { background: #585b70; }
QPushButton:pressed { background: #6c7086; }
QPushButton:disabled { background: #313244; color: #6c7086; }
QPushButton#greenBtn { background: #a6e3a1; color: #1e1e2e; }
QPushButton#greenBtn:hover { background: #94e2d5; }
QPushButton#redBtn { background: #f38ba8; color: #1e1e2e; }
QPushButton#redBtn:hover { background: #eba0ac; }
QPushButton#blueBtn { background: #89b4fa; color: #1e1e2e; }
QPushButton#blueBtn:hover { background: #74c7ec; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #313244; border: 1px solid #45475a; border-radius: 4px; padding: 4px; }
QTabWidget::pane { border: 1px solid #45475a; border-radius: 4px; }
QTabBar::tab { background: #313244; padding: 8px 18px; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background: #45475a; }
QListWidget { background: #181825; border: 1px solid #45475a; border-radius: 4px; }
QListWidget::item { padding: 4px; }
QListWidget::item:selected { background: #45475a; }
QTextEdit { background: #181825; border: 1px solid #45475a; border-radius: 4px; font-family: 'Consolas', monospace; font-size: 12px; }
QProgressBar { background: #313244; border-radius: 4px; text-align: center; color: #cdd6f4; font-weight: bold; }
QProgressBar::chunk { background: #a6e3a1; border-radius: 4px; }
QLabel#headerLabel { font-size: 18px; font-weight: bold; color: #89b4fa; }
QLabel#stepLabel { font-size: 13px; font-weight: bold; color: #f9e2af; }
QSplitter::handle { background: #45475a; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📷 Photogrammetry Studio")
        self.setMinimumSize(1280, 800)
        self.pipeline = PhotogrammetryPipeline()
        self.worker = None
        self.example_dir = os.path.join(os.path.expanduser("~"), ".photogrammetry_example")

        self._build_menu()
        self._build_toolbar()
        self._build_image_dock()
        self._build_control_dock()
        self._build_log_dock()
        self._build_central()
        self._connect_signals()
        self._update_status()

    # ── Menu ──────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = self.menuBar()
        file_m = mb.addMenu("&File")
        file_m.addAction("Load Images…", self._load_images, "Ctrl+O")
        file_m.addAction("Generate Example Data", self._generate_example)
        file_m.addSeparator()
        file_m.addAction("Export Point Cloud (PLY)…", self._export_ply, "Ctrl+S")
        file_m.addSeparator()
        file_m.addAction("Quit", self.close, "Ctrl+Q")

        help_m = mb.addMenu("&Help")
        help_m.addAction("About…", lambda: QMessageBox.about(
            self, "About", "Photogrammetry Studio\n\n"
            "Feature detection → Matching →\n"
            "Sparse 3D reconstruction → Point cloud viewer\n\n"
            "Built with PySide6 + OpenCV"))

    # ── Toolbar ───────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        self.btn_load = tb.addAction("📂 Load Images")
        self.btn_load.triggered.connect(self._load_images)
        self.btn_example = tb.addAction("🎲 Example Data")
        self.btn_example.triggered.connect(self._generate_example)
        tb.addSeparator()
        self.btn_detect = tb.addAction("🔍 Detect Features")
        self.btn_detect.triggered.connect(lambda: self._run_pipeline(["detect"]))
        self.btn_match = tb.addAction("🔗 Match Features")
        self.btn_match.triggered.connect(lambda: self._run_pipeline(["detect", "match"]))
        self.btn_recon = tb.addAction("🏗️ Reconstruct 3D")
        self.btn_recon.triggered.connect(lambda: self._run_pipeline(
            ["detect", "match", "intrinsics", "reconstruct"]))
        self.btn_auto = tb.addAction("⚡ Auto Run All")
        self.btn_auto.triggered.connect(lambda: self._run_pipeline(
            ["detect", "match", "intrinsics", "reconstruct"]))
        tb.addSeparator()
        self.btn_export = tb.addAction("💾 Export PLY")
        self.btn_export.triggered.connect(self._export_ply)

    # ── Image List Dock ──────────────────────────────────────────────
    def _build_image_dock(self):
        dock = QDockWidget("Images", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        lay = QVBoxLayout(w)
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(120, 90))
        self.image_list.setSpacing(4)
        self.image_list.currentRowChanged.connect(self._on_image_selected)
        lay.addWidget(self.image_list)
        self.lbl_img_count = QLabel("No images loaded")
        lay.addWidget(self.lbl_img_count)
        dock.setWidget(w)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    # ── Control Dock ─────────────────────────────────────────────────
    def _build_control_dock(self):
        dock = QDockWidget("Settings", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(6)

        # Feature detection
        g1 = QGroupBox("Feature Detection")
        g1l = QGridLayout(g1)
        g1l.addWidget(QLabel("Detector:"), 0, 0)
        self.cmb_detector = QComboBox()
        self.cmb_detector.addItems(["SIFT", "ORB"])
        g1l.addWidget(self.cmb_detector, 0, 1)
        g1l.addWidget(QLabel("Max Features:"), 1, 0)
        self.spn_features = QSpinBox()
        self.spn_features.setRange(500, 50000)
        self.spn_features.setValue(5000)
        self.spn_features.setSingleStep(500)
        g1l.addWidget(self.spn_features, 1, 1)
        lay.addWidget(g1)

        # Matching
        g2 = QGroupBox("Matching")
        g2l = QGridLayout(g2)
        g2l.addWidget(QLabel("Ratio Threshold:"), 0, 0)
        self.spn_ratio = QDoubleSpinBox()
        self.spn_ratio.setRange(0.5, 0.95)
        self.spn_ratio.setValue(0.75)
        self.spn_ratio.setSingleStep(0.05)
        g2l.addWidget(self.spn_ratio, 0, 1)
        lay.addWidget(g2)

        # Reconstruction
        g3 = QGroupBox("Reconstruction")
        g3l = QGridLayout(g3)
        g3l.addWidget(QLabel("Reproj. Error:"), 0, 0)
        self.spn_reproj = QDoubleSpinBox()
        self.spn_reproj.setRange(0.5, 10.0)
        self.spn_reproj.setValue(4.0)
        self.spn_reproj.setSingleStep(0.5)
        g3l.addWidget(self.spn_reproj, 0, 1)
        g3l.addWidget(QLabel("Point Size:"), 1, 0)
        self.spn_ptsize = QSpinBox()
        self.spn_ptsize.setRange(1, 20)
        self.spn_ptsize.setValue(2)
        g3l.addWidget(self.spn_ptsize, 1, 1)
        self.chk_color = QCheckBox("Color points")
        self.chk_color.setChecked(True)
        g3l.addWidget(self.chk_color, 2, 0, 1, 2)
        lay.addWidget(g3)

        # Action buttons
        self.btn_run = QPushButton("⚡  Run Full Pipeline")
        self.btn_run.setObjectName("greenBtn")
        lay.addWidget(self.btn_run)
        self.btn_run.clicked.connect(lambda: self._run_pipeline(
            ["detect", "match", "intrinsics", "reconstruct"]))

        self.btn_stop = QPushButton("⏹  Stop")
        self.btn_stop.setObjectName("redBtn")
        self.btn_stop.setEnabled(False)
        lay.addWidget(self.btn_stop)
        self.btn_stop.clicked.connect(self._stop_pipeline)

        lay.addStretch()
        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ── Log Dock ─────────────────────────────────────────────────────
    def _build_log_dock(self):
        dock = QDockWidget("Log", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)
        dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    # ── Central Widget ───────────────────────────────────────────────
    def _build_central(self):
        self.tabs = QTabWidget()

        # --- Images Tab ---
        img_w = QWidget()
        img_l = QVBoxLayout(img_w)
        self.lbl_image = QLabel("Load images to begin")
        self.lbl_image.setAlignment(Qt.AlignCenter)
        self.lbl_image.setStyleSheet("background: #181825; border-radius: 6px;")
        img_l.addWidget(self.lbl_image)
        self.tabs.addTab(img_w, "🖼️ Images")

        # --- Features Tab ---
        feat_w = QWidget()
        feat_l = QVBoxLayout(feat_w)
        self.lbl_features = QLabel("Run feature detection to view keypoints")
        self.lbl_features.setAlignment(Qt.AlignCenter)
        self.lbl_features.setStyleSheet("background: #181825; border-radius: 6px;")
        feat_l.addWidget(self.lbl_features)
        self.tabs.addTab(feat_w, "🔍 Features")

        # --- Matches Tab ---
        match_w = QWidget()
        match_l = QVBoxLayout(match_w)
        match_ctrl = QHBoxLayout()
        match_ctrl.addWidget(QLabel("Image pair:"))
        self.cmb_pair_i = QSpinBox(); self.cmb_pair_j = QSpinBox()
        self.cmb_pair_i.setPrefix("I1: "); self.cmb_pair_j.setPrefix("I2: ")
        match_ctrl.addWidget(self.cmb_pair_i)
        match_ctrl.addWidget(self.cmb_pair_j)
        self.btn_show_match = QPushButton("Show Matches")
        self.btn_show_match.setObjectName("blueBtn")
        match_ctrl.addWidget(self.btn_show_match)
        self.btn_show_match.clicked.connect(self._show_matches)
        match_ctrl.addStretch()
        match_l.addLayout(match_ctrl)
        self.lbl_matches = QLabel("Run matching to view correspondences")
        self.lbl_matches.setAlignment(Qt.AlignCenter)
        self.lbl_matches.setStyleSheet("background: #181825; border-radius: 6px;")
        match_l.addWidget(self.lbl_matches)
        self.tabs.addTab(match_w, "🔗 Matches")

        # --- 3D Viewer Tab ---
        view_w = QWidget()
        view_l = QVBoxLayout(view_w)
        self.pc_viewer = PointCloudViewer()
        toolbar = NavigationToolbar(self.pc_viewer, self)
        view_l.addWidget(toolbar)
        view_l.addWidget(self.pc_viewer)
        self.tabs.addTab(view_w, "🌐 3D Viewer")

        self.setCentralWidget(self.tabs)

        # Progress bar in status bar
        self.progress = QProgressBar()
        self.progress.setMaximumHeight(18)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Ready")

    # ── Signals ──────────────────────────────────────────────────────
    def _connect_signals(self):
        pass

    # ── Load Images ──────────────────────────────────────────────────
    def _load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All (*)")
        if not paths:
            return
        n = self.pipeline.load_images(paths)
        self._refresh_image_list()
        self._log(f"Loaded {n} images")
        self._update_status()
        if n > 0:
            self._show_image(0)

    def _generate_example(self):
        self._log("Generating example data …")
        generate_example_data(self.example_dir)
        paths = sorted(Path(self.example_dir).glob("*.png"))
        n = self.pipeline.load_images([str(p) for p in paths])
        self._refresh_image_list()
        self._log(f"Generated & loaded {n} example images")
        self._update_status()
        if n > 0:
            self._show_image(0)

    def _refresh_image_list(self):
        self.image_list.clear()
        for i, (img, path) in enumerate(zip(self.pipeline.images,
                                              self.pipeline.image_paths)):
            pm = cv2_to_qpixmap(img, max_size=120)
            item = QListWidgetItem(QIcon(pm),
                                    f"  {i+1}. {Path(path).name}")
            item.setSizeHint(QSize(140, 100))
            self.image_list.addItem(item)
        self.lbl_img_count.setText(f"{len(self.pipeline.images)} images")
        self.cmb_pair_i.setRange(0, max(len(self.pipeline.images)-1, 0))
        self.cmb_pair_j.setRange(1, max(len(self.pipeline.images), 1))

    # ── Show Image / Features / Matches ─────────────────────────────
    def _on_image_selected(self, row):
        if 0 <= row < len(self.pipeline.images):
            self._show_image(row)

    def _show_image(self, idx):
        pm = cv2_to_qpixmap(self.pipeline.images[idx], max_size=900)
        self.lbl_image.setPixmap(pm)
        self.tabs.setCurrentIndex(0)
        # Also show features if available
        if self.pipeline.keypoints:
            self._show_features(idx)

    def _show_features(self, idx):
        if idx >= len(self.pipeline.keypoints):
            return
        vis = cv2.drawKeypoints(self.pipeline.images[idx],
                                 self.pipeline.keypoints[idx], None,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.lbl_features.setPixmap(cv2_to_qpixmap(vis, max_size=900))

    def _show_matches(self):
        i, j = self.cmb_pair_i.value(), self.cmb_pair_j.value()
        if i == j:
            j = i + 1
            self.cmb_pair_j.setValue(j)
        key = (min(i, j), max(i, j))
        if key not in self.pipeline.matches:
            self._log(f"No matches between image {key[0]+1} and {key[1]+1}")
            return
        gm = self.pipeline.matches[key]
        kp1 = self.pipeline.keypoints[key[0]]
        kp2 = self.pipeline.keypoints[key[1]]
        vis = cv2.drawMatches(self.pipeline.images[key[0]], kp1,
                               self.pipeline.images[key[1]], kp2,
                               gm[:100], None,
                               matchColor=(0, 255, 0),
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        self.lbl_matches.setPixmap(cv2_to_qpixmap(vis, max_size=1200))
        self.tabs.setCurrentIndex(2)
        self._log(f"Showing {len(gm)} matches between image {key[0]+1} & {key[1]+1}")

    # ── Run / Stop Pipeline ─────────────────────────────────────────
    def _apply_settings(self):
        self.pipeline.detector_type = self.cmb_detector.currentText()
        self.pipeline.max_features = self.spn_features.value()
        self.pipeline.ratio_thresh = self.spn_ratio.value()
        self.pipeline.reproj_err = self.spn_reproj.value()
        self.pc_viewer.point_size = self.spn_ptsize.value()

    def _run_pipeline(self, steps):
        if not self.pipeline.images:
            QMessageBox.warning(self, "No Images",
                                "Load images first (File → Load Images or generate example data).")
            return
        self._apply_settings()
        self._set_busy(True)
        self.worker = PipelineWorker(self.pipeline, steps)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.start()

    def _stop_pipeline(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self._log("⏹ Pipeline stopped by user")
            self._set_busy(False)

    def _set_busy(self, busy):
        for a in [self.btn_detect, self.btn_match, self.btn_recon, self.btn_auto]:
            a.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        self.btn_run.setEnabled(not busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)  # indeterminate initially

    def _on_progress(self, cur, total):
        self.progress.setRange(0, total)
        self.progress.setValue(cur)

    def _on_pipeline_finished(self, status):
        self._set_busy(False)
        self.progress.setVisible(False)
        if status == "done":
            self._log("✅ Pipeline complete!")
            self.statusBar().showMessage("Pipeline complete", 3000)
            # Refresh views
            if self.pipeline.keypoints and self.pipeline.images:
                row = self.image_list.currentRow()
                if row < 0:
                    row = 0
                self._show_features(row)
                self.tabs.setCurrentIndex(1)
            if self.pipeline.point_cloud is not None:
                cols = self.pipeline.point_colors if self.chk_color.isChecked() else None
                self.pc_viewer.show_point_cloud(self.pipeline.point_cloud, cols)
                self.tabs.setCurrentIndex(3)
        else:
            self._log("⚠ Pipeline finished with errors")
        self._update_status()

    # ── Export ────────────────────────────────────────────────────────
    def _export_ply(self):
        if self.pipeline.point_cloud is None:
            QMessageBox.warning(self, "No Data",
                                "Run reconstruction first to generate a point cloud.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Point Cloud", "output.ply", "PLY (*.ply)")
        if path:
            if self.pipeline.export_ply(path):
                self._log(f"💾 Exported point cloud to {path}")
                QMessageBox.information(self, "Export Complete",
                                        f"Point cloud saved to:\n{path}\n\n"
                                        f"{len(self.pipeline.point_cloud)} points")
            else:
                self._log("❌ Export failed")

    # ── Helpers ───────────────────────────────────────────────────────
    def _log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def _update_status(self):
        n = len(self.pipeline.images)
        nk = sum(len(k) for k in self.pipeline.keypoints) if self.pipeline.keypoints else 0
        nm = len(self.pipeline.matches)
        np_ = len(self.pipeline.point_cloud) if self.pipeline.point_cloud is not None else 0
        self.statusBar().showMessage(
            f"Images: {n}  |  Keypoints: {nk}  |  "
            f"Matched pairs: {nm}  |  3D points: {np_}")


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLE)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    win = MainWindow()
    win.show()

    # Show a welcome tip
    win._log("══════════════════════════════════════════")
    win._log("  Welcome to Photogrammetry Studio! 📷")
    win._log("══════════════════════════════════════════")
    win._log("")
    win._log("Quick start:")
    win._log("  1. Click 🎲 Example Data (or 📂 Load Images)")
    win._log("  2. Click ⚡ Auto Run All")
    win._log("  3. Explore tabs: Images → Features → Matches → 3D Viewer")
    win._log("  4. Export point cloud as PLY")
    win._log("")
    win._log("Tip: For best results, use 8+ photos of an object")
    win._log("     taken from different angles with ~60% overlap.")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()