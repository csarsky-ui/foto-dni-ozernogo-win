from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .engine import LoadedPhoto, PhotoEngine, PortraitPreviewResult, RenderedSheet
from .models import (
    ColorMode,
    CornerStyle,
    CutMarkStyle,
    EditSettings,
    LogoStyle,
    PhotoFormatPreset,
)
from .print_win import (
    configure_and_save_win32_profile,
    detect_epson_l8050,
    has_saved_epson_profile,
    print_direct_with_saved_win32_profile,
    print_with_qt_dialog,
)


def rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class _LoadWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, engine: PhotoEngine, path: Path) -> None:
        super().__init__()
        self.engine = engine
        self.path = path

    def run(self) -> None:
        try:
            loaded = self.engine.load_photo(self.path)
            self.finished.emit(loaded)
        except Exception as exc:
            self.failed.emit(str(exc))


class _RenderWorker(QObject):
    finished = Signal(int, str, object)  # gen, mode, RenderedSheet | PortraitPreviewResult
    failed = Signal(int, str)

    def __init__(self, engine: PhotoEngine, gen: int, mode: str, photo: LoadedPhoto, settings: EditSettings) -> None:
        super().__init__()
        self.engine = engine
        self.gen = gen
        self.mode = mode
        self.photo = photo
        self.settings = settings

    def run(self) -> None:
        try:
            if self.mode == "sheet":
                res = self.engine.render_sheet(self.photo, self.settings)
            else:
                res = self.engine.render_closeup_portrait(self.photo, self.settings)
            self.finished.emit(self.gen, self.mode, res)
        except Exception as exc:
            self.failed.emit(self.gen, str(exc))


class PreviewCanvas(QWidget):
    file_dropped = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setMinimumSize(460, 560)
        self._pixmap: Optional[QPixmap] = None
        self._drop_hover = False
        self._loading_text: Optional[str] = None
        self._show_ruler = False
        self._ruler_info: Optional[dict] = None

    def set_loading(self, text: Optional[str]) -> None:
        self._loading_text = text
        self.update()

    def set_preview(self, pixmap: Optional[QPixmap], show_ruler: bool = False, ruler_info: Optional[dict] = None) -> None:
        self._pixmap = pixmap
        self._show_ruler = show_ruler
        self._ruler_info = ruler_info
        self._loading_text = None
        self.update()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self._drop_hover = True
            self.update()
            event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._drop_hover = False
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:
        self._drop_hover = False
        self.update()
        urls = event.mimeData().urls()
        if urls:
            local = urls[0].toLocalFile()
            if local:
                self.file_dropped.emit(Path(local))

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Calm studio gray backdrop
        p.fillRect(self.rect(), QColor(42, 45, 49))

        if self._loading_text:
            p.setPen(QColor(220, 225, 230))
            font = QFont()
            font.setPointSize(12)
            p.setFont(font)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._loading_text)
            return

        if self._pixmap is None or self._pixmap.isNull():
            box = self.rect().adjusted(40, 40, -40, -40)
            pen_col = QColor(78, 170, 92) if self._drop_hover else QColor(110, 118, 128)
            pen = QPen(pen_col, 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRoundedRect(box, 14, 14)

            p.setPen(QColor(235, 238, 242))
            f1 = QFont()
            f1.setPointSize(14)
            f1.setBold(True)
            p.setFont(f1)
            p.drawText(box.adjusted(16, -24, -16, -24), Qt.AlignmentFlag.AlignCenter, "Перетащите сюда RAW (.CR2, .CR3) или JPG")

            p.setPen(QColor(168, 175, 184))
            f2 = QFont()
            f2.setPointSize(10)
            p.setFont(f2)
            p.drawText(
                box.adjusted(16, 26, -16, 26),
                Qt.AlignmentFlag.AlignCenter,
                "Вертикальный лист 10×15 см · 300 DPI · sRGB · логотип «Дни Озёрного»",
            )
            return

        # Fit pixmap with clean padding and subtle paper drop shadow
        pad = 24
        avail_w = max(80, self.width() - pad * 2)
        avail_h = max(80, self.height() - pad * 2)
        aspect = self._pixmap.width() / max(1.0, float(self._pixmap.height()))
        fit_w = min(avail_w, avail_h * aspect)
        fit_h = fit_w / aspect
        x = (self.width() - fit_w) * 0.5
        y = (self.height() - fit_h) * 0.5
        target = QRectF(x, y, fit_w, fit_h)

        # Soft shadow behind paper
        p.fillRect(target.translated(4, 6), QColor(0, 0, 0, 90))
        p.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))
        p.setPen(QPen(QColor(0, 0, 0, 55), 1))
        p.drawRect(target)

        if self._show_ruler and self._ruler_info:
            self._paint_ruler(p, target, self._ruler_info)

        if self._drop_hover:
            p.setPen(QPen(QColor(78, 170, 92), 3))
            p.fillRect(self.rect().adjusted(8, 8, -8, -8), QColor(78, 170, 92, 24))
            p.drawRoundedRect(self.rect().adjusted(8, 8, -8, -8), 10, 10)

    def _paint_ruler(self, p: QPainter, rect: QRectF, info: dict) -> None:
        tw = max(1.0, float(info["tile_w_mm"]))
        th = max(1.0, float(info["tile_h_mm"]))
        top_mm = float(info["headroom_mm"])
        head_mm = float(info["head_mm"])
        eye_y_mm = float(info["eye_line_mm"])
        le_mm = float(info["left_eye_mm"])
        re_mm = float(info["right_eye_mm"])
        is_passport = bool(info.get("is_passport", False))

        y_top = rect.top() + rect.height() * max(0.0, min(0.5, top_mm / th))
        y_eye = rect.top() + rect.height() * max(0.05, min(0.85, eye_y_mm / th))
        y_chin = rect.top() + rect.height() * max(0.1, min(0.98, (top_mm + head_mm) / th))
        x_l = rect.left() + rect.width() * max(0.05, min(0.49, le_mm / tw))
        x_r = rect.left() + rect.width() * max(0.51, min(0.95, 1.0 - re_mm / tw))

        # Cyan horizontal guides + center vertical
        pen_c = QPen(QColor(0, 220, 240, 200), 1, Qt.PenStyle.DashLine)
        p.setPen(pen_c)
        p.drawLine(QRectF(rect.left(), y_top, rect.width(), 0).topLeft(), QRectF(rect.left(), y_top, rect.width(), 0).topRight())
        p.drawLine(QRectF(rect.left(), y_chin, rect.width(), 0).topLeft(), QRectF(rect.left(), y_chin, rect.width(), 0).topRight())
        p.drawLine(QRectF(rect.center().x(), rect.top(), 0, rect.height()).topLeft(), QRectF(rect.center().x(), rect.top(), 0, rect.height()).bottomLeft())

        # Yellow eye line & left/right pupil markers
        pen_y = QPen(QColor(255, 220, 40, 220), 1, Qt.PenStyle.DashLine)
        p.setPen(pen_y)
        p.drawLine(QRectF(rect.left(), y_eye, rect.width(), 0).topLeft(), QRectF(rect.left(), y_eye, rect.width(), 0).topRight())
        p.drawLine(int(x_l), int(max(rect.top(), y_eye - 28)), int(x_l), int(min(rect.bottom(), y_eye + 28)))
        p.drawLine(int(x_r), int(max(rect.top(), y_eye - 28)), int(x_r), int(min(rect.bottom(), y_eye + 28)))

        # HUD metrics pill
        top_ok = (not is_passport) or (4.0 <= top_mm <= 6.0)
        head_ok = (not is_passport) or (30.0 <= head_mm <= 32.0)
        lines = [
            (f"Верх до макушки: {top_mm:.1f} мм" + (" (норма 5±1)" if is_passport else ""), top_ok),
            (f"Макушка–подбородок: {head_mm:.1f} мм" + (" (норма 30–32)" if is_passport else ""), head_ok),
            (f"Глаза от краёв (Л/П): {le_mm:.1f} / {re_mm:.1f} мм", True),
        ]
        f = QFont("Consolas", 9)
        p.setFont(f)
        hud = QRectF(rect.left() + 8, rect.top() + 8, 285, 62)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 175))
        p.drawRoundedRect(hud, 6, 6)
        ty = int(hud.top() + 18)
        for text, ok in lines:
            p.setPen(QColor(255, 255, 255) if ok else QColor(255, 225, 70))
            p.drawText(int(hud.left() + 10), ty, text)
            ty += 18


class MainWindow(QMainWindow):
    def __init__(self, initial_file: Optional[Path] = None) -> None:
        super().__init__()
        self.setWindowTitle("Фото на документы — Дни Озёрного")
        self.resize(1120, 820)
        self.setMinimumSize(940, 700)

        self.engine = PhotoEngine()
        if self.engine.assets_dir:
            ico = self.engine.assets_dir / "app_icon.ico"
            if ico.is_file():
                self.setWindowIcon(QIcon(str(ico)))

        self.settings = EditSettings()
        self.loaded_photo: Optional[LoadedPhoto] = None
        self.current_sheet: Optional[RenderedSheet] = None
        self.last_saved_psd: Optional[Path] = None
        self.preview_mode = "sheet"
        self._render_gen = 0
        self._active_threads: list[QThread] = []
        self._updating_ui = False

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(65)
        self._render_timer.timeout.connect(self._start_render_job)

        self._build_ui()
        self._build_menus()
        self._refresh_win_profile_badge()

        if initial_file and initial_file.is_file():
            QTimer.singleShot(100, lambda: self.load_file(initial_file))

    def _build_menus(self) -> None:
        menu_file = self.menuBar().addMenu("Файл")

        act_open = QAction("Открыть RAW / фото…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self.open_file_dialog)
        menu_file.addAction(act_open)

        menu_file.addSeparator()

        act_psd = QAction("Сохранить PSD (многослойный)…", self)
        act_psd.setShortcut(QKeySequence.StandardKey.Save)
        act_psd.triggered.connect(self.save_psd)
        menu_file.addAction(act_psd)

        act_jpg = QAction("Сохранить JPEG (300 DPI)…", self)
        act_jpg.setShortcut(QKeySequence("Ctrl+Shift+E"))
        act_jpg.triggered.connect(self.save_jpeg)
        menu_file.addAction(act_jpg)

        menu_file.addSeparator()

        act_print = QAction("Печать 10×15 (EPSON L8050)…", self)
        act_print.setShortcut(QKeySequence.StandardKey.Print)
        act_print.triggered.connect(self.print_sheet)
        menu_file.addAction(act_print)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ===== LEFT PANEL =====
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(0)

        top_bar = QFrame()
        top_bar.setFrameShape(QFrame.Shape.StyledPanel)
        tb_lay = QHBoxLayout(top_bar)
        tb_lay.setContentsMargins(12, 8, 12, 8)

        self.btn_open = QPushButton("Открыть RAW / фото…")
        self.btn_open.clicked.connect(self.open_file_dialog)
        tb_lay.addWidget(self.btn_open)
        tb_lay.addStretch()

        self.btn_mode_sheet = QPushButton("Лист 10×15")
        self.btn_mode_sheet.setCheckable(True)
        self.btn_mode_sheet.setChecked(True)
        self.btn_mode_portrait = QPushButton("Крупно портрет")
        self.btn_mode_portrait.setCheckable(True)
        mode_grp = QButtonGroup(self)
        mode_grp.setExclusive(True)
        mode_grp.addButton(self.btn_mode_sheet)
        mode_grp.addButton(self.btn_mode_portrait)
        self.btn_mode_sheet.clicked.connect(lambda: self._set_preview_mode("sheet"))
        self.btn_mode_portrait.clicked.connect(lambda: self._set_preview_mode("portrait"))
        tb_lay.addWidget(self.btn_mode_sheet)
        tb_lay.addWidget(self.btn_mode_portrait)

        self.chk_ruler = QCheckBox("Линейка мм")
        self.chk_ruler.setChecked(True)
        self.chk_ruler.setVisible(False)
        self.chk_ruler.toggled.connect(lambda _: self._schedule_render(immediate=True))
        tb_lay.addWidget(self.chk_ruler)

        left_lay.addWidget(top_bar)

        self.canvas = PreviewCanvas()
        self.canvas.file_dropped.connect(self.load_file)
        left_lay.addWidget(self.canvas, 1)

        bot_bar = QFrame()
        bot_bar.setFrameShape(QFrame.Shape.StyledPanel)
        bb_lay = QHBoxLayout(bot_bar)
        bb_lay.setContentsMargins(12, 6, 12, 6)
        self.lbl_status = QLabel("Перетащите RAW (.CR2, .CR3) или фото в окно")
        self.lbl_status.setStyleSheet("color: #666;")
        self.lbl_metrics = QLabel("")
        self.lbl_metrics.setStyleSheet("color: #555; font-family: Consolas, monospace;")
        bb_lay.addWidget(self.lbl_status, 1)
        bb_lay.addWidget(self.lbl_metrics)
        left_lay.addWidget(bot_bar)

        splitter.addWidget(left)

        # ===== RIGHT SIDEBAR =====
        right = QWidget()
        right.setMinimumWidth(360)
        right.setMaximumWidth(430)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar = QWidget()
        sb_lay = QVBoxLayout(sidebar)
        sb_lay.setContentsMargins(12, 12, 12, 12)
        sb_lay.setSpacing(10)

        # 1. Format
        g_fmt = QGroupBox("Формат на листе 10×15 (вертикальный)")
        gl = QVBoxLayout(g_fmt)
        self.cmb_preset = QComboBox()
        for p in PhotoFormatPreset:
            self.cmb_preset.addItem(p.title, p)
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        gl.addWidget(self.cmb_preset)
        self.lbl_preset_sub = QLabel(self.settings.preset.subtitle)
        self.lbl_preset_sub.setWordWrap(True)
        self.lbl_preset_sub.setStyleSheet("color: #666; font-size: 11px;")
        gl.addWidget(self.lbl_preset_sub)
        sb_lay.addWidget(g_fmt)

        # 2. Style (Color/BW, Corner, White BG)
        g_style = QGroupBox("Вариант фото")
        gl = QVBoxLayout(g_style)
        row_c = QHBoxLayout()
        self.btn_color = QPushButton("Цветное")
        self.btn_color.setCheckable(True)
        self.btn_color.setChecked(True)
        self.btn_bw = QPushButton("Чёрно-белое")
        self.btn_bw.setCheckable(True)
        c_grp = QButtonGroup(self)
        c_grp.setExclusive(True)
        c_grp.addButton(self.btn_color)
        c_grp.addButton(self.btn_bw)
        self.btn_color.clicked.connect(lambda: self._set_color_mode(ColorMode.COLOR))
        self.btn_bw.clicked.connect(lambda: self._set_color_mode(ColorMode.BW))
        row_c.addWidget(self.btn_color)
        row_c.addWidget(self.btn_bw)
        gl.addLayout(row_c)

        row_cor = QHBoxLayout()
        row_cor.addWidget(QLabel("Уголок:"))
        self.cmb_corner = QComboBox()
        for c in CornerStyle:
            self.cmb_corner.addItem(c.title, c)
        self.cmb_corner.currentIndexChanged.connect(self._on_controls_changed)
        row_cor.addWidget(self.cmb_corner, 1)
        gl.addLayout(row_cor)

        self.chk_white_bg = QCheckBox("Чисто-белый фон (AI-маска в PSD)")
        self.chk_white_bg.setChecked(True)
        self.chk_white_bg.toggled.connect(self._on_controls_changed)
        gl.addWidget(self.chk_white_bg)

        self.sl_mask = self._add_slider(gl, "Край маски волос", 0, 100, 50, lambda v: f"{v}%")
        sb_lay.addWidget(g_style)

        # 3. Designer Retouch
        g_ret = QGroupBox("Ретушь портрета (дизайнер)")
        gl = QVBoxLayout(g_ret)
        self.chk_redeye = QCheckBox("Убрать красные глаза")
        self.chk_redeye.setChecked(True)
        self.chk_redeye.toggled.connect(self._on_controls_changed)
        gl.addWidget(self.chk_redeye)
        self.lbl_redeye_status = QLabel("")
        self.lbl_redeye_status.setStyleSheet("color: #666; font-size: 11px;")
        gl.addWidget(self.lbl_redeye_status)

        self.sl_deshine = self._add_slider(gl, "Убрать блеск (нос/лицо)", 0, 100, 55, lambda v: f"{v}%")
        self.sl_skintone = self._add_slider(gl, "Выровнять скинтон", 0, 100, 45, lambda v: f"{v}%")
        self.sl_retouch = self._add_slider(gl, "Мягко скрыть дефекты", 0, 100, 35, lambda v: f"{v}%")
        sb_lay.addWidget(g_ret)

        # 4. Light & WB
        g_light = QGroupBox("Свет и баланс белого")
        gl = QVBoxLayout(g_light)
        self.chk_autowb = QCheckBox("Авто баланс белого (по фону)")
        self.chk_autowb.setChecked(True)
        self.chk_autowb.toggled.connect(self._on_controls_changed)
        gl.addWidget(self.chk_autowb)

        self.sl_exp = self._add_slider(gl, "Яркость (EV)", -100, 100, 28, lambda v: f"{v/100:+.2f}")
        self.sl_temp = self._add_slider(gl, "Теплее / холоднее", -900, 900, 0, lambda v: f"{v:+d} K")
        self.sl_tint = self._add_slider(gl, "Оттенок (Tint)", -25, 25, 0, lambda v: f"{v:+d}")
        self.sl_contrast = self._add_slider(gl, "Контраст", 85, 125, 104, lambda v: f"{v/100:.2f}")
        self.sl_shadows = self._add_slider(gl, "Светлее тени", 0, 40, 16, lambda v: f"{v}%")
        sb_lay.addWidget(g_light)

        # 5. Framing
        g_frame = QGroupBox("Кадрирование")
        gl = QVBoxLayout(g_frame)
        self.sl_zoom = self._add_slider(gl, "Масштаб", 72, 135, 100, lambda v: f"{v}%")
        self.sl_shifty = self._add_slider(gl, "Сдвиг вверх / вниз", -22, 22, 0, lambda v: f"{v:+d}%")
        self.sl_shiftx = self._add_slider(gl, "Сдвиг влево / вправо", -20, 20, 0, lambda v: f"{v:+d}%")
        self.sl_rot = self._add_slider(gl, "Доворот", -50, 50, 0, lambda v: f"{v/10:+.1f}°")
        btn_reset = QPushButton("Сбросить настройки фото")
        btn_reset.clicked.connect(self._reset_tweaks)
        gl.addWidget(btn_reset)
        sb_lay.addWidget(g_frame)

        # 6. Sheet layout
        g_sheet = QGroupBox("Оформление листа 10×15")
        gl = QVBoxLayout(g_sheet)
        self.sl_gap = self._add_slider(gl, "Зазор между фото", 0, 45, 25, lambda v: f"{v/10:.1f} мм")

        row_cut = QHBoxLayout()
        row_cut.addWidget(QLabel("Метки реза:"))
        self.cmb_cut = QComboBox()
        for m in CutMarkStyle:
            self.cmb_cut.addItem(m.title, m)
        self.cmb_cut.currentIndexChanged.connect(self._on_controls_changed)
        row_cut.addWidget(self.cmb_cut, 1)
        gl.addLayout(row_cut)

        row_logo = QHBoxLayout()
        row_logo.addWidget(QLabel("Логотип внизу:"))
        self.cmb_logo = QComboBox()
        for l in LogoStyle:
            self.cmb_logo.addItem(l.title, l)
        self.cmb_logo.currentIndexChanged.connect(self._on_controls_changed)
        row_logo.addWidget(self.cmb_logo, 1)
        gl.addLayout(row_logo)
        sb_lay.addWidget(g_sheet)

        # 7. Print Profile (Windows Epson L8050)
        g_pr = QGroupBox("Печать · EPSON L8050 (10×15, 1:1)")
        gl = QVBoxLayout(g_pr)
        row_pr = QHBoxLayout()
        row_pr.addWidget(QLabel("Копий:"))
        self.spn_copies = QSpinBox()
        self.spn_copies.setRange(1, 20)
        self.spn_copies.setValue(1)
        self.spn_copies.valueChanged.connect(self._on_controls_changed)
        row_pr.addWidget(self.spn_copies)
        row_pr.addStretch()
        self.chk_rot180 = QCheckBox("Поворот 180°")
        self.chk_rot180.setChecked(self.settings.print_rotate_180)
        self.chk_rot180.setToolTip("Включайте, если поворот 180° НЕ задан внутри профиля драйвера Epson")
        self.chk_rot180.toggled.connect(self._on_controls_changed)
        row_pr.addWidget(self.chk_rot180)
        gl.addLayout(row_pr)

        self.btn_cfg_epson = QPushButton("Настроить и запомнить профиль Epson L8050…")
        self.btn_cfg_epson.clicked.connect(self._configure_epson_profile)
        gl.addWidget(self.btn_cfg_epson)

        self.lbl_epson_prof = QLabel("")
        self.lbl_epson_prof.setWordWrap(True)
        self.lbl_epson_prof.setStyleSheet("color: #555; font-size: 11px;")
        gl.addWidget(self.lbl_epson_prof)
        sb_lay.addWidget(g_pr)

        sb_lay.addStretch()
        scroll.setWidget(sidebar)
        right_lay.addWidget(scroll, 1)

        # Action bar
        act_bar = QFrame()
        act_bar.setFrameShape(QFrame.Shape.StyledPanel)
        ab_lay = QVBoxLayout(act_bar)
        ab_lay.setContentsMargins(12, 10, 12, 10)
        ab_lay.setSpacing(6)

        row_main = QHBoxLayout()
        self.btn_save_psd = QPushButton("Сохранить PSD")
        self.btn_save_psd.setMinimumHeight(36)
        self.btn_save_psd.setStyleSheet(
            "QPushButton { background-color: #2f7d3a; color: white; font-weight: bold; border-radius: 6px; padding: 6px 12px; }"
            "QPushButton:hover { background-color: #266930; }"
            "QPushButton:disabled { background-color: #9aa0a6; }"
        )
        self.btn_save_psd.clicked.connect(self.save_psd)
        self.btn_save_psd.setEnabled(False)

        self.btn_print = QPushButton("Печать")
        self.btn_print.setMinimumHeight(36)
        self.btn_print.clicked.connect(self.print_sheet)
        self.btn_print.setEnabled(False)

        self.btn_save_jpg = QPushButton("JPG")
        self.btn_save_jpg.setMinimumHeight(36)
        self.btn_save_jpg.clicked.connect(self.save_jpeg)
        self.btn_save_jpg.setEnabled(False)

        row_main.addWidget(self.btn_save_psd, 2)
        row_main.addWidget(self.btn_print, 1)
        row_main.addWidget(self.btn_save_jpg, 1)
        ab_lay.addLayout(row_main)

        self.post_save_row = QWidget()
        ps_lay = QHBoxLayout(self.post_save_row)
        ps_lay.setContentsMargins(0, 0, 0, 0)
        self.btn_open_ps = QPushButton("Открыть в Photoshop")
        self.btn_open_ps.clicked.connect(self._open_in_photoshop)
        self.btn_reveal = QPushButton("Показать в папке")
        self.btn_reveal.clicked.connect(self._reveal_in_explorer)
        ps_lay.addWidget(self.btn_open_ps, 2)
        ps_lay.addWidget(self.btn_reveal, 1)
        self.post_save_row.setVisible(False)
        ab_lay.addWidget(self.post_save_row)

        right_lay.addWidget(act_bar)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _add_slider(
        self,
        layout: QVBoxLayout,
        title: str,
        min_v: int,
        max_v: int,
        init_v: int,
        fmt: Callable[[int], str],
    ) -> QSlider:
        row = QHBoxLayout()
        lbl_t = QLabel(title)
        lbl_t.setStyleSheet("font-size: 11px;")
        lbl_val = QLabel(fmt(init_v))
        lbl_val.setStyleSheet("font-size: 11px; color: #666; font-family: Consolas, monospace;")
        row.addWidget(lbl_t)
        row.addStretch()
        row.addWidget(lbl_val)
        layout.addLayout(row)

        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(min_v, max_v)
        sl.setValue(init_v)
        sl.valueChanged.connect(lambda v: (lbl_val.setText(fmt(v)), self._on_controls_changed()))
        layout.addWidget(sl)
        return sl

    # MARK: - UI State Sync
    def _set_preview_mode(self, mode: str) -> None:
        self.preview_mode = mode
        self.chk_ruler.setVisible(mode == "portrait")
        self._schedule_render(immediate=True)

    def _set_color_mode(self, mode: ColorMode) -> None:
        self.settings.color_mode = mode
        self._schedule_render(immediate=True)

    def _on_preset_changed(self) -> None:
        if self._updating_ui:
            return
        preset: PhotoFormatPreset = self.cmb_preset.currentData()
        self.settings.preset = preset
        self.lbl_preset_sub.setText(preset.subtitle)
        self._updating_ui = True
        self.sl_gap.setValue(int(round(preset.default_gap_mm * 10)))
        self._updating_ui = False
        self._on_controls_changed()

    def _on_controls_changed(self) -> None:
        if self._updating_ui:
            return
        self.settings.corner_style = self.cmb_corner.currentData()
        self.settings.white_background = self.chk_white_bg.isChecked()
        self.settings.mask_clean_strength = self.sl_mask.value() / 100.0

        self.settings.red_eye_fix = self.chk_redeye.isChecked()
        self.settings.de_shine = self.sl_deshine.value() / 100.0
        self.settings.skin_tone_polish = self.sl_skintone.value() / 100.0
        self.settings.skin_retouch = self.sl_retouch.value() / 100.0

        self.settings.auto_wb = self.chk_autowb.isChecked()
        self.settings.exposure_ev = self.sl_exp.value() / 100.0
        self.settings.temp_shift = float(self.sl_temp.value())
        self.settings.tint_shift = float(self.sl_tint.value())
        self.settings.contrast = self.sl_contrast.value() / 100.0
        self.settings.shadow_lift = self.sl_shadows.value() / 100.0

        self.settings.zoom = self.sl_zoom.value() / 100.0
        self.settings.shift_y = self.sl_shifty.value() / 100.0
        self.settings.shift_x = self.sl_shiftx.value() / 100.0
        self.settings.extra_rotation_deg = self.sl_rot.value() / 10.0

        self.settings.gap_mm = self.sl_gap.value() / 10.0
        self.settings.cut_marks = self.cmb_cut.currentData()
        self.settings.logo_style = self.cmb_logo.currentData()

        self.settings.print_copies = self.spn_copies.value()
        self.settings.print_rotate_180 = self.chk_rot180.isChecked()
        self._schedule_render()

    def _reset_tweaks(self) -> None:
        self._updating_ui = True
        self.settings.reset_photo_tweaks()
        self.sl_mask.setValue(50)
        self.chk_redeye.setChecked(True)
        self.sl_deshine.setValue(55)
        self.sl_skintone.setValue(45)
        self.sl_retouch.setValue(35)
        self.chk_autowb.setChecked(True)
        self.sl_exp.setValue(28)
        self.sl_temp.setValue(0)
        self.sl_tint.setValue(0)
        self.sl_contrast.setValue(104)
        self.sl_shadows.setValue(16)
        self.sl_zoom.setValue(100)
        self.sl_shifty.setValue(0)
        self.sl_shiftx.setValue(0)
        self.sl_rot.setValue(0)
        self._updating_ui = False
        self._schedule_render(immediate=True)

    # MARK: - File Loading & Async Rendering
    def open_file_dialog(self) -> None:
        filt = "Фото и RAW (*.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.jpg *.jpeg *.png *.tif *.tiff);;Все файлы (*.*)"
        path_str, _ = QFileDialog.getOpenFileName(self, "Выберите RAW или фотографию", "", filt)
        if path_str:
            self.load_file(Path(path_str))

    def load_file(self, path: Path) -> None:
        self.last_saved_psd = None
        self.post_save_row.setVisible(False)
        self.btn_save_psd.setEnabled(False)
        self.btn_save_jpg.setEnabled(False)
        self.btn_print.setEnabled(False)
        self.canvas.set_loading(f"Декодирую {path.name} и строю AI-маску…")
        self.lbl_status.setText(f"Загрузка {path.name}…")

        thread = QThread(self)
        worker = _LoadWorker(self.engine, path)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_photo_loaded)
        worker.failed.connect(self._on_photo_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        self._active_threads.append(thread)
        thread.start()

    def _on_photo_loaded(self, loaded: LoadedPhoto) -> None:
        self.loaded_photo = loaded
        self.btn_save_psd.setEnabled(True)
        self.btn_save_jpg.setEnabled(True)
        self.btn_print.setEnabled(True)
        deg = loaded.metrics.eye_roll_rad * 180.0 / 3.14159265
        face_note = f"лицо найдено, выравнивание {deg:+.1f}°" if loaded.metrics.detected_face else "лицо не распознано"
        self.lbl_status.setText(f"{loaded.path.name} · {loaded.metrics.image_width}×{loaded.metrics.image_height} · {face_note}")
        self._schedule_render(immediate=True)

    def _on_photo_failed(self, err: str) -> None:
        self.canvas.set_loading(None)
        self.lbl_status.setText("Ошибка загрузки файла")
        QMessageBox.critical(self, "Ошибка открытия файла", err)

    def _schedule_render(self, immediate: bool = False) -> None:
        if self.loaded_photo is None:
            return
        if immediate:
            self._render_timer.stop()
            self._start_render_job()
        else:
            self._render_timer.start()

    def _start_render_job(self) -> None:
        if self.loaded_photo is None:
            return
        self._render_gen += 1
        gen = self._render_gen
        # Snapshot dataclass copy for thread safety
        snap = EditSettings(**self.settings.__dict__)
        thread = QThread(self)
        worker = _RenderWorker(self.engine, gen, self.preview_mode, self.loaded_photo, snap)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_render_done)
        worker.failed.connect(self._on_render_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._cleanup_thread(thread))
        self._active_threads.append(thread)
        thread.start()

    def _on_render_done(self, gen: int, mode: str, result: object) -> None:
        if gen != self._render_gen:
            return
        if mode == "sheet" and isinstance(result, RenderedSheet):
            self.current_sheet = result
            self.canvas.set_preview(rgb_to_qpixmap(result.composite_rgb), show_ruler=False)
            self._update_metrics_ui(
                result.primary_headroom_mm,
                result.primary_head_mm,
                result.primary_left_eye_mm,
                result.primary_right_eye_mm,
                result.red_eye_fixed_pixels,
            )
        elif isinstance(result, PortraitPreviewResult):
            tw, th = self.settings.preset.primary_tile_mm
            ruler_info = {
                "tile_w_mm": tw,
                "tile_h_mm": th,
                "headroom_mm": result.headroom_mm,
                "head_mm": result.head_mm,
                "eye_line_mm": result.eye_line_mm,
                "left_eye_mm": result.left_eye_mm,
                "right_eye_mm": result.right_eye_mm,
                "is_passport": self.settings.preset == PhotoFormatPreset.PASSPORT_RF,
            }
            self.canvas.set_preview(rgb_to_qpixmap(result.rgb), show_ruler=self.chk_ruler.isChecked(), ruler_info=ruler_info)
            self._update_metrics_ui(
                result.headroom_mm,
                result.head_mm,
                result.left_eye_mm,
                result.right_eye_mm,
                result.red_eye_fixed_pixels,
            )

    def _on_render_failed(self, gen: int, err: str) -> None:
        if gen != self._render_gen:
            return
        self.lbl_status.setText(f"Ошибка рендеринга: {err}")

    def _update_metrics_ui(self, top_mm: float, head_mm: float, le_mm: float, re_mm: float, red_px: int) -> None:
        self.lbl_metrics.setText(f"Верх: {top_mm:.1f} · Голова: {head_mm:.1f} · Глаза: {le_mm:.1f}/{re_mm:.1f} мм")
        if not self.settings.red_eye_fix:
            self.lbl_redeye_status.setText("")
        elif red_px >= 12:
            self.lbl_redeye_status.setText("✓ Красные глаза обнаружены и убраны")
            self.lbl_redeye_status.setStyleSheet("color: #2f7d3a; font-size: 11px;")
        else:
            self.lbl_redeye_status.setText("Красные глаза на этом фото не обнаружены")
            self.lbl_redeye_status.setStyleSheet("color: #777; font-size: 11px;")

    def _cleanup_thread(self, thread: QThread) -> None:
        if thread in self._active_threads:
            self._active_threads.remove(thread)
        thread.deleteLater()

    # MARK: - Export & Printing
    def _suggested_base_name(self) -> str:
        stem = self.loaded_photo.path.stem if self.loaded_photo else "photo"
        bw = "_chb" if self.settings.color_mode == ColorMode.BW else ""
        return f"{stem}_{self.settings.preset.short_file_tag}{self.settings.corner_style.short_tag}{bw}_10x15"

    def save_psd(self) -> None:
        if self.loaded_photo is None:
            return
        default_path = str(self.loaded_photo.path.parent / f"{self._suggested_base_name()}.psd")
        out_str, _ = QFileDialog.getSaveFileName(self, "Сохранить многослойный PSD (10×15 см, 300 DPI)", default_path, "Adobe Photoshop (*.psd)")
        if not out_str:
            return
        out_path = Path(out_str)
        try:
            sheet = self.engine.render_sheet(self.loaded_photo, self.settings)
            self.engine.export_psd(sheet, self.settings, out_path)
            self.current_sheet = sheet
            self.last_saved_psd = out_path
            self.post_save_row.setVisible(True)
            mb = out_path.stat().st_size / 1_048_576.0
            self.lbl_status.setText(f"PSD сохранён: {out_path.name} ({mb:.1f} МБ, 300 DPI, sRGB)")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения PSD", str(exc))

    def save_jpeg(self) -> None:
        if self.loaded_photo is None:
            return
        default_path = str(self.loaded_photo.path.parent / f"{self._suggested_base_name()}.jpg")
        out_str, _ = QFileDialog.getSaveFileName(self, "Сохранить JPEG на печать (10×15 см, 300 DPI)", default_path, "JPEG (*.jpg *.jpeg)")
        if not out_str:
            return
        out_path = Path(out_str)
        try:
            sheet = self.engine.render_sheet(self.loaded_photo, self.settings)
            self.engine.export_jpeg(sheet, out_path)
            self.current_sheet = sheet
            self.lbl_status.setText(f"JPEG сохранён: {out_path.name} (10×15 см, 300 DPI, sRGB)")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения JPEG", str(exc))

    def _refresh_win_profile_badge(self) -> None:
        printer = detect_epson_l8050() or "не найден"
        if sys.platform != "win32":
            self.lbl_epson_prof.setText(f"Принтер: {printer} · На Windows здесь сохраняется профиль драйвера (Epson Matte / Высокий / Vivid / 180°)")
            return
        if has_saved_epson_profile():
            self.lbl_epson_prof.setText(f"✓ Профиль драйвера для «{printer}» сохранён — кнопка «Печать» печатает по нему в 1 клик.")
            self.lbl_epson_prof.setStyleSheet("color: #2f7d3a; font-size: 11px;")
        else:
            self.lbl_epson_prof.setText(
                f"Принтер: «{printer}». Нажмите кнопку выше 1 раз, выберите ваш набор настроек (Epson Matte, 10×15, Высокий, Насыщенный, 180°) и нажмите ОК."
            )
            self.lbl_epson_prof.setStyleSheet("color: #666; font-size: 11px;")

    def _configure_epson_profile(self) -> None:
        ok, msg = configure_and_save_win32_profile(self)
        self._refresh_win_profile_badge()
        if ok:
            QMessageBox.information(self, "Профиль EPSON L8050", msg)
        else:
            QMessageBox.warning(self, "Профиль EPSON L8050", msg)

    def print_sheet(self) -> None:
        if self.loaded_photo is None:
            return
        sheet = self.engine.render_sheet(self.loaded_photo, self.settings)
        self.current_sheet = sheet
        job_title = f"{self._suggested_base_name()}"

        if sys.platform == "win32" and has_saved_epson_profile():
            printer = detect_epson_l8050() or "EPSON L8050 Series"
            box = QMessageBox(self)
            box.setWindowTitle(f"Печать на {printer}")
            box.setText(f"Отправить {self.settings.print_copies} экз. на «{printer}» по сохранённому профилю?")
            box.setInformativeText(
                f"Формат: {self.settings.preset.title}\n"
                f"Масштаб: строго 1:1 (100×150 мм по центру 10×15)\n"
                f"Настройки бумаги, цвета и поворот на 180° берутся из сохранённого профиля драйвера Epson."
            )
            b_now = box.addButton("Напечатать сейчас", QMessageBox.ButtonRole.AcceptRole)
            b_dlg = box.addButton("Диалог Windows…", QMessageBox.ButtonRole.ActionRole)
            box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked == b_now:
                ok, msg = print_direct_with_saved_win32_profile(
                    sheet.composite_rgb,
                    copies=self.settings.print_copies,
                    extra_rotate_180=self.settings.print_rotate_180,
                    job_title=job_title,
                    printer_name=printer,
                )
                self.lbl_status.setText(msg)
                if not ok:
                    QMessageBox.critical(self, "Ошибка печати", msg)
                return
            if clicked != b_dlg:
                return

        ok, msg = print_with_qt_dialog(
            self,
            sheet.composite_rgb,
            rotate_180=self.settings.print_rotate_180,
            copies=self.settings.print_copies,
            job_title=job_title,
        )
        self.lbl_status.setText(msg)

    def _open_in_photoshop(self) -> None:
        if not self.last_saved_psd or not self.last_saved_psd.is_file():
            return
        if sys.platform == "win32":
            os.startfile(str(self.last_saved_psd))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(self.last_saved_psd)])

    def _reveal_in_explorer(self) -> None:
        if not self.last_saved_psd or not self.last_saved_psd.is_file():
            return
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(self.last_saved_psd)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(self.last_saved_psd)])


def run_gui(initial_file: Optional[Path] = None) -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Фото на документы — Дни Озёрного")
    app.setStyle("Fusion")
    win = MainWindow(initial_file=initial_file)
    win.show()
    return app.exec()
