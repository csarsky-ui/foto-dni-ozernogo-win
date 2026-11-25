from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import rawpy
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image, ImageOps

from .models import (
    ColorMode,
    CornerStyle,
    CutMarkStyle,
    EditSettings,
    FaceMetrics,
    LogoStyle,
    PlacedTile,
    SheetConstants,
)
from .psd_writer import PSDGuide, PSDLayer, write_psd


RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def resolve_resource_dir(subdir: str) -> Optional[Path]:
    candidates: List[Path] = []
    env_override = os.environ.get(f"FOTO_{subdir.upper()}_DIR")
    if env_override:
        candidates.append(Path(env_override))
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / subdir)
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / subdir)
        candidates.append(exe_dir / "_internal" / subdir)
    here = Path(__file__).resolve().parent.parent
    candidates.append(here / subdir)
    candidates.append(Path.cwd() / subdir)
    for c in candidates:
        if c.is_dir():
            return c
    return None


@dataclass
class LoadedPhoto:
    path: Path
    leveled_rgb: np.ndarray        # HxWx3 uint8 sRGB (already eye-leveled)
    raw_mask: np.ndarray           # HxW float32 [0..1]
    metrics: FaceMetrics
    auto_temp_delta: float
    auto_tint_delta: float
    is_raw: bool


@dataclass
class RenderedTilePack:
    placed: PlacedTile
    unmasked_rgb: np.ndarray       # HxWx3 uint8 (full portrait for PSD layer)
    mask_gray: np.ndarray          # HxW uint8 (editable Photoshop Layer Mask -2)
    composited_rgb: np.ndarray     # HxWx3 uint8 (composited over pure white)
    head_height_mm: float
    headroom_mm: float
    left_eye_mm: float
    right_eye_mm: float
    eye_line_mm: float
    red_eye_fixed_pixels: int


@dataclass
class RenderedSheet:
    width: int
    height: int
    tiles: List[RenderedTilePack]
    corner_overlay_rgba: Optional[np.ndarray]   # HxWx4 uint8
    cut_marks_rgba: Optional[np.ndarray]        # HxWx4 uint8
    logo_layer: Optional[Tuple[int, int, np.ndarray]]  # (x, y, HxWx4 uint8)
    guides: List[PSDGuide]
    composite_rgb: np.ndarray                   # HxWx3 uint8
    primary_head_mm: float
    primary_headroom_mm: float
    primary_left_eye_mm: float
    primary_right_eye_mm: float
    primary_eye_line_mm: float
    red_eye_fixed_pixels: int


@dataclass
class PortraitPreviewResult:
    rgb: np.ndarray
    head_mm: float
    headroom_mm: float
    left_eye_mm: float
    right_eye_mm: float
    eye_line_mm: float
    red_eye_fixed_pixels: int


class PhotoEngine:
    def __init__(self) -> None:
        self.assets_dir = resolve_resource_dir("assets")
        self.models_dir = resolve_resource_dir("models")

        self.logo_color_rgba = self._load_rgba("logo_color.png")
        self.logo_gray_rgba = self._load_rgba("logo_gray.png")
        self.srgb_icc_path = (self.assets_dir / "sRGB.icc") if self.assets_dir else None

        self._rmbg_session: Optional[ort.InferenceSession] = None
        self._face_landmarker = None

    def _load_rgba(self, filename: str) -> Optional[np.ndarray]:
        if not self.assets_dir:
            return None
        p = self.assets_dir / filename
        if not p.is_file():
            return None
        return np.array(Image.open(p).convert("RGBA"), dtype=np.uint8)

    def _ensure_models(self) -> None:
        if self._rmbg_session is None:
            if not self.models_dir:
                raise RuntimeError("Папка models/ не найдена.")
            model_path = self.models_dir / "rmbg_1_4_q.onnx"
            if not model_path.is_file():
                raise RuntimeError(f"Не найдена модель отделения фона: {model_path}")
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(8, (cv2.getNumberOfCPUs() or 4)))
            # Load from bytes to avoid C++ ANSI path issues with Cyrillic folders on Windows
            self._rmbg_session = ort.InferenceSession(
                model_path.read_bytes(), sess_options=opts, providers=["CPUExecutionProvider"]
            )

        if self._face_landmarker is None:
            assert self.models_dir is not None
            task_path = self.models_dir / "face_landmarker.task"
            if not task_path.is_file():
                raise RuntimeError(f"Не найдена модель лица MediaPipe: {task_path}")
            base_opts = mp_python.BaseOptions(model_asset_buffer=task_path.read_bytes())
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    # MARK: - 1. Load RAW / Image + MediaPipe Face Landmarks + RMBG-1.4 Mask
    def load_photo(self, path: Path) -> LoadedPhoto:
        self._ensure_models()
        ext = path.suffix.lower()
        is_raw = ext in RAW_EXTENSIONS

        if is_raw:
            # Open via binary file handle so Cyrillic paths on Windows never break LibRaw C++ fopen
            with open(path, "rb") as fh:
                with rawpy.imread(fh) as raw:
                    rgb16 = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=False,
                        auto_bright_thr=0.006,
                        bright=1.10,
                        output_bps=16,
                        output_color=rawpy.ColorSpace.sRGB,
                        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                        fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode.Light,
                        highlight_mode=rawpy.HighlightMode.Blend,
                    )
            # Studio portrait tone curve matching Apple CIRAWFilter baseline (gentle shadow bias + highlight rolloff)
            img_f = rgb16.astype(np.float32) / 65535.0
            # Soft shadow lift (~CIRAWFilter shadowBias 0.08) + gentle highlight knee above 0.90
            lum_r = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
            sh_lift = np.power(np.clip(1.0 - lum_r, 0.0, 1.0), 2.2) * 0.06
            img_f = img_f + sh_lift[..., None] * (1.0 - img_f)
            over = np.maximum(0.0, img_f - 0.90)
            img_f = np.where(img_f > 0.90, 0.90 + over / (1.0 + over * 1.5), img_f)
            img_rgb = np.clip(img_f * 255.0, 0, 255).astype(np.uint8)
        else:
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img_rgb = np.array(pil, dtype=np.uint8)

        # Keep working portrait crisp (cap extreme 45MP+ to 3600px long edge so interactive tweaks stay super fast;
        # still >3x higher than any 300 DPI ID tile!)
        H, W = img_rgb.shape[:2]
        max_dim = max(W, H)
        if max_dim > 3600:
            scale = 3600.0 / max_dim
            img_rgb = cv2.resize(img_rgb, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA)
            H, W = img_rgb.shape[:2]

        # Pass 1: Detect face landmarks to auto-level eye roll
        eye_roll_rad = 0.0
        leveled_rgb = img_rgb
        lms1 = self._detect_landmarks(img_rgb)
        if lms1 is not None:
            le, re = self._extract_eye_centers(lms1, W, H)
            if le is not None and re is not None:
                roll = math.atan2(re[1] - le[1], re[0] - le[0])
                if 0.002 < abs(roll) < 0.21:
                    eye_roll_rad = roll
                    cx = (le[0] + re[0]) * 0.5
                    cy = (le[1] + re[1]) * 0.5
                    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(roll), 1.0)
                    leveled_rgb = cv2.warpAffine(
                        img_rgb, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
                    )

        # Pass 2: Full RMBG-1.4 alpha matte + refined MediaPipe landmarks on leveled image
        raw_mask = self._run_rmbg_mask(leveled_rgb)
        lms2 = self._detect_landmarks(leveled_rgb)

        # High-resolution Portrait-ROI second pass for wide/landscape frames so ears, temples & hair get 2x-3x resolution
        if lms2 is not None:
            xs = [p.x for p in lms2]
            ys = [p.y for p in lms2]
            fw_norm = max(xs) - min(xs)
            fh_norm = max(ys) - min(ys)
            if fw_norm < 0.46:
                fcx = (min(xs) + max(xs)) * 0.5 * W
                fcy = (min(ys) + max(ys)) * 0.5 * H
                roi_w = min(W, int(round(fw_norm * W * 3.4)))
                roi_h = min(H, int(round(fh_norm * H * 4.4)))
                rx0 = max(0, min(W - roi_w, int(round(fcx - roi_w * 0.5))))
                ry0 = max(0, min(H - roi_h, int(round(fcy - roi_h * 0.38))))
                rx1, ry1 = rx0 + roi_w, ry0 + roi_h
                if roi_w >= 256 and roi_h >= 256:
                    roi_mask = self._run_rmbg_mask(leveled_rgb[ry0:ry1, rx0:rx1])
                    # Softly blend ROI edges (24px) into full-frame mask so zooming out never shows a seam
                    fe = min(24, roi_w // 8, roi_h // 8)
                    wy = np.ones(roi_h, dtype=np.float32)
                    wx = np.ones(roi_w, dtype=np.float32)
                    if fe > 1:
                        ramp = np.linspace(0.0, 1.0, fe, dtype=np.float32)
                        if ry0 > 0:
                            wy[:fe] = ramp
                        if ry1 < H:
                            wy[-fe:] = ramp[::-1]
                        if rx0 > 0:
                            wx[:fe] = ramp
                        if rx1 < W:
                            wx[-fe:] = ramp[::-1]
                    w2d = wy[:, None] * wx[None, :]
                    raw_mask[ry0:ry1, rx0:rx1] = roi_mask * w2d + raw_mask[ry0:ry1, rx0:rx1] * (1.0 - w2d)

        metrics = self._build_face_metrics(lms2, raw_mask, W, H, eye_roll_rad)
        auto_temp, auto_tint = self._estimate_backdrop_wb(leveled_rgb, raw_mask)

        return LoadedPhoto(
            path=path,
            leveled_rgb=leveled_rgb,
            raw_mask=raw_mask,
            metrics=metrics,
            auto_temp_delta=auto_temp,
            auto_tint_delta=auto_tint,
            is_raw=is_raw,
        )

    def _detect_landmarks(self, rgb: np.ndarray):
        H, W = rgb.shape[:2]
        # MediaPipe works fast & accurately at ~1200px
        scale = min(1.0, 1200.0 / max(W, H))
        if scale < 1.0:
            small = cv2.resize(rgb, (int(round(W * scale)), int(round(H * scale))), interpolation=cv2.INTER_AREA)
        else:
            small = rgb
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        result = self._face_landmarker.detect(mp_img)
        if result and result.face_landmarks:
            return result.face_landmarks[0]
        return None

    def _extract_eye_centers(self, lms, W: int, H: int) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        # MediaPipe 478-point FaceLandmarker: 468 = left iris center (image-left = subject right eye), 473 = right iris center
        if len(lms) >= 478:
            p_a = (lms[468].x * W, lms[468].y * H)
            p_b = (lms[473].x * W, lms[473].y * H)
            return (p_a, p_b) if p_a[0] <= p_b[0] else (p_b, p_a)
        # Fallback to eye corner midpoints (33/133 and 362/263)
        if len(lms) >= 400:
            p_a = ((lms[33].x + lms[133].x) * 0.5 * W, (lms[33].y + lms[133].y) * 0.5 * H)
            p_b = ((lms[362].x + lms[263].x) * 0.5 * W, (lms[362].y + lms[263].y) * 0.5 * H)
            return (p_a, p_b) if p_a[0] <= p_b[0] else (p_b, p_a)
        return None, None

    def _run_rmbg_mask(self, rgb: np.ndarray) -> np.ndarray:
        assert self._rmbg_session is not None
        H, W = rgb.shape[:2]
        inp = cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0 - 0.5
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        out = self._rmbg_session.run(None, {"input": inp})[0][0, 0]
        lo, hi = float(out.min()), float(out.max())
        if hi - lo > 1e-6:
            out = (out - lo) / (hi - lo)
        else:
            out = np.ones_like(out)
        mask = cv2.resize(out.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return np.clip(mask, 0.0, 1.0)

    def _build_face_metrics(
        self, lms, raw_mask: np.ndarray, W: int, H: int, eye_roll_rad: float
    ) -> FaceMetrics:
        if lms is None:
            return FaceMetrics(
                image_width=W,
                image_height=H,
                face_bbox_norm=(0.35, 0.20, 0.30, 0.35),
                hair_top_y=H * 0.12,
                eye_line_y=H * 0.35,
                chin_y=H * 0.55,
                face_center_x=W * 0.50,
                eye_roll_rad=0.0,
                left_eye_center=None,
                right_eye_center=None,
                eye_radius_px=W * 0.03,
                nose_rect=None,
                mouth_rect=None,
                detected_face=False,
            )

        xs = [p.x for p in lms]
        ys = [p.y for p in lms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        bb_norm = (min_x, min_y, max_x - min_x, max_y - min_y)

        # Chin bottom landmark #152
        chin_y = lms[152].y * H if len(lms) > 152 else max_y * H
        le, re = self._extract_eye_centers(lms, W, H)
        face_cx = ((le[0] + re[0]) * 0.5) if (le and re) else ((min_x + max_x) * 0.5 * W)
        eye_line_y = ((le[1] + re[1]) * 0.5) if (le and re) else (min_y + (max_y - min_y) * 0.38) * H
        eye_dist = math.hypot(re[0] - le[0], re[1] - le[1]) if (le and re) else ((max_x - min_x) * W * 0.42)
        eye_r = max(12.0, eye_dist * 0.22)

        # Find exact top of hair from RMBG mask within central face band
        half_band = int((max_x - min_x) * W * 0.85)
        x0 = max(0, int(face_cx) - half_band)
        x1 = min(W, int(face_cx) + half_band)
        band = raw_mask[:, x0:x1] > 0.55
        rows_with_hair = np.where(band.any(axis=1))[0]
        hair_top_y = float(rows_with_hair[0]) if len(rows_with_hair) > 0 else max(0.0, min_y * H - (chin_y - min_y * H) * 0.25)

        # Nose & mouth boxes
        nose_ids = [6, 197, 195, 5, 4, 1, 2, 98, 327]
        nxs = [lms[i].x * W for i in nose_ids if i < len(lms)]
        nys = [lms[i].y * H for i in nose_ids if i < len(lms)]
        nose_rect = (min(nxs), min(nys), max(1.0, max(nxs) - min(nxs)), max(1.0, max(nys) - min(nys))) if nxs else None

        mouth_ids = [61, 291, 0, 17, 78, 308, 13, 14]
        mxs = [lms[i].x * W for i in mouth_ids if i < len(lms)]
        mys = [lms[i].y * H for i in mouth_ids if i < len(lms)]
        mouth_rect = (min(mxs), min(mys), max(1.0, max(mxs) - min(mxs)), max(1.0, max(mys) - min(mys))) if mxs else None

        return FaceMetrics(
            image_width=W,
            image_height=H,
            face_bbox_norm=bb_norm,
            hair_top_y=hair_top_y,
            eye_line_y=eye_line_y,
            chin_y=chin_y,
            face_center_x=face_cx,
            eye_roll_rad=eye_roll_rad,
            left_eye_center=le,
            right_eye_center=re,
            eye_radius_px=eye_r,
            nose_rect=nose_rect,
            mouth_rect=mouth_rect,
            detected_face=True,
        )

    def _estimate_backdrop_wb(self, rgb: np.ndarray, raw_mask: np.ndarray) -> Tuple[float, float]:
        H, W = rgb.shape[:2]
        step = max(2, min(W, H) // 250)
        y_max = int(H * 0.65)
        sub_rgb = rgb[4:y_max:step, 4 : W - 4 : step].astype(np.float32)
        sub_m = raw_mask[4:y_max:step, 4 : W - 4 : step]

        r, g, b = sub_rgb[:, :, 0], sub_rgb[:, :, 1], sub_rgb[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        sat = np.where(max_c > 1.0, (max_c - min_c) / np.maximum(1.0, max_c), 0.0)
        valid = (sub_m < 0.08) & (lum > 110.0) & (lum < 246.0) & (sat < 0.26)
        if int(valid.sum()) < 40:
            return (-180.0, 2.0)

        avg_r = float(np.median(r[valid]))
        avg_g = float(np.median(g[valid]))
        avg_b = float(np.median(b[valid]))
        rb_diff = (avg_r - avg_b) / max(1.0, avg_g)
        gm_diff = (avg_g - 0.5 * (avg_r + avg_b)) / max(1.0, avg_g)
        temp_delta = max(-450.0, min(350.0, -rb_diff * 2400.0))
        tint_delta = max(-10.0, min(12.0, gm_diff * 110.0))
        return (temp_delta, tint_delta)

    # MARK: - 2. Render Single Tile + Designer Retouching
    def render_tile(self, photo: LoadedPhoto, tile: PlacedTile, settings: EditSettings) -> RenderedTilePack:
        m = photo.metrics
        W, H = m.image_width, m.image_height
        src_rgb = photo.leveled_rgb
        src_mask = photo.raw_mask

        # Optional manual tilt around eyes center
        if abs(settings.extra_rotation_deg) > 0.05:
            M = cv2.getRotationMatrix2D((m.face_center_x, m.eye_line_y), settings.extra_rotation_deg, 1.0)
            src_rgb = cv2.warpAffine(src_rgb, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            src_mask = cv2.warpAffine(src_mask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        head_px = max(60.0, m.chin_y - m.hair_top_y)
        aspect = tile.width / float(tile.height)
        base_crop_h = head_px / tile.head_fraction
        crop_h = base_crop_h / max(0.55, min(1.8, settings.zoom))
        if crop_h > H:
            crop_h = float(H)
        crop_w = crop_h * aspect
        if crop_w > W:
            crop_w = float(W)
            crop_h = crop_w / aspect

        eye_mid_x = (
            (m.left_eye_center[0] + m.right_eye_center[0]) * 0.5
            if (m.left_eye_center and m.right_eye_center)
            else m.face_center_x
        )

        crop_top = m.hair_top_y - crop_h * tile.headroom_fraction - crop_h * settings.shift_y
        crop_top = max(0.0, min(H - crop_h, crop_top))
        crop_left = eye_mid_x - crop_w * 0.5 - crop_w * settings.shift_x
        crop_left = max(0.0, min(W - crop_w, crop_left))

        # Exact live millimeter measurements
        actual_head_mm = (head_px / crop_h) * tile.height_mm
        actual_headroom_mm = ((m.hair_top_y - crop_top) / crop_h) * tile.height_mm
        actual_eye_line_mm = ((m.eye_line_y - crop_top) / crop_h) * tile.height_mm
        actual_left_eye_mm = (
            ((m.left_eye_center[0] - crop_left) / crop_w) * tile.width_mm
            if m.left_eye_center
            else tile.width_mm * 0.40
        )
        actual_right_eye_mm = (
            (((crop_left + crop_w) - m.right_eye_center[0]) / crop_w) * tile.width_mm
            if m.right_eye_center
            else tile.width_mm * 0.40
        )

        # Extract crop at 2x target resolution for crisp studio retouching
        work_w, work_h = tile.width * 2, tile.height * 2
        x0 = max(0, int(math.floor(crop_left)))
        y0 = max(0, int(math.floor(crop_top)))
        x1 = min(W, int(math.ceil(crop_left + crop_w)))
        y1 = min(H, int(math.ceil(crop_top + crop_h)))

        crop_rgb = cv2.resize(src_rgb[y0:y1, x0:x1], (work_w, work_h), interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
        crop_mask = cv2.resize(src_mask[y0:y1, x0:x1], (work_w, work_h), interpolation=cv2.INTER_LINEAR)

        # Edge-guided 3-channel RGB Guided Filter (He et al. 2013): snaps ears, temples, collar & epaulettes
        # cleanly onto true RGB edges while preserving natural hair wisps and locking pure white/interior!
        if settings.white_background:
            clean_mask = self._refine_mask_rgb_guided(crop_rgb, crop_mask, settings.mask_clean_strength)
        else:
            clean_mask = np.ones((work_h, work_w), dtype=np.float32)

        # Global White Balance (Auto WB + user fine-tuning)
        temp_shift = (photo.auto_temp_delta if settings.auto_wb else 0.0) + settings.temp_shift
        tint_shift = (photo.auto_tint_delta if settings.auto_wb else 0.0) + settings.tint_shift
        if abs(temp_shift) > 1.0 or abs(tint_shift) > 0.2:
            # Smooth photographic WB gains (negative temp -> cooler/less yellow, positive tint -> magenta/less green)
            t_norm = temp_shift / 3200.0
            g_norm = tint_shift / 180.0
            r_gain = 1.0 + t_norm * 0.34 + g_norm * 0.14
            g_gain = 1.0 - g_norm * 0.18
            b_gain = 1.0 - t_norm * 0.38 + g_norm * 0.12
            # Preserve overall luminance
            l_norm = 0.299 * r_gain + 0.587 * g_gain + 0.114 * b_gain
            crop_rgb[:, :, 0] *= r_gain / l_norm
            crop_rgb[:, :, 1] *= g_gain / l_norm
            crop_rgb[:, :, 2] *= b_gain / l_norm

        # Exposure (print-ready lift) in linear-ish domain
        if abs(settings.exposure_ev) > 0.005:
            ev_mul = 2.0 ** (settings.exposure_ev * 0.85)
            # Soft highlight rolloff so white skin highlights never clip harshly
            crop_rgb = np.clip(crop_rgb, 0.0, 1.0)
            lin = np.power(crop_rgb, 2.2) * ev_mul
            # Gentle knee above 0.88
            lin = np.where(lin > 0.88, 0.88 + (lin - 0.88) / (1.0 + (lin - 0.88) * 1.8), lin)
            crop_rgb = np.power(np.clip(lin, 0.0, 1.0), 1.0 / 2.2)

        # Shadow lift & gentle highlight control
        lum = 0.299 * crop_rgb[:, :, 0] + 0.587 * crop_rgb[:, :, 1] + 0.114 * crop_rgb[:, :, 2]
        if settings.shadow_lift > 0.005:
            sh_w = np.power(np.clip(1.0 - lum, 0.0, 1.0), 2.0) * settings.shadow_lift * 0.42
            crop_rgb = crop_rgb + sh_w[..., None] * (1.0 - crop_rgb)
        if settings.de_shine > 0.01:
            hl_compress = np.power(np.clip((lum - 0.72) / 0.28, 0.0, 1.0), 2.0) * settings.de_shine * 0.06
            crop_rgb = np.clip(crop_rgb - hl_compress[..., None], 0.0, 1.0)

        # Contrast & subtle saturation polish
        crop_rgb = np.clip((crop_rgb - 0.5) * settings.contrast + 0.5 + 0.012, 0.0, 1.0)
        work_u8 = np.clip(crop_rgb * 255.0, 0, 255).astype(np.uint8)

        # Designer Studio Portrait Retouching (Vectorized NumPy — fast & razor-accurate!)
        red_eye_px = 0
        if m.detected_face and (
            settings.de_shine > 0.01
            or settings.red_eye_fix
            or settings.skin_tone_polish > 0.01
            or settings.skin_retouch > 0.01
        ):
            work_u8, red_eye_px = self._apply_designer_retouch(
                rgb=work_u8,
                fg_mask=clean_mask,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_w=crop_w,
                crop_h=crop_h,
                metrics=m,
                settings=settings,
            )

        # Portrait B&W mode
        if settings.color_mode == ColorMode.BW:
            f = work_u8.astype(np.float32) / 255.0
            y = 0.37 * f[:, :, 0] + 0.48 * f[:, :, 1] + 0.15 * f[:, :, 2]
            y = np.clip((y - 0.5) * 1.08 + 0.52, 0.0, 1.0)
            g8 = np.clip(y * 255.0, 0, 255).astype(np.uint8)
            work_u8 = np.stack([g8, g8, g8], axis=-1)

        # Downscale 2x -> exact 1x (300 DPI) cleanly
        unmasked_rgb = cv2.resize(work_u8, (tile.width, tile.height), interpolation=cv2.INTER_AREA)
        if settings.white_background:
            mask_u8 = np.clip(
                cv2.resize(clean_mask, (tile.width, tile.height), interpolation=cv2.INTER_AREA) * 255.0, 0, 255
            ).astype(np.uint8)
        else:
            mask_u8 = np.full((tile.height, tile.width), 255, dtype=np.uint8)

        alpha = mask_u8.astype(np.float32)[..., None] / 255.0
        comp_rgb = np.clip(unmasked_rgb.astype(np.float32) * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)

        return RenderedTilePack(
            placed=tile,
            unmasked_rgb=unmasked_rgb,
            mask_gray=mask_u8,
            composited_rgb=comp_rgb,
            head_height_mm=actual_head_mm,
            headroom_mm=actual_headroom_mm,
            left_eye_mm=actual_left_eye_mm,
            right_eye_mm=actual_right_eye_mm,
            eye_line_mm=actual_eye_line_mm,
            red_eye_fixed_pixels=red_eye_px,
        )

    @staticmethod
    def _refine_mask_rgb_guided(
        rgb01: np.ndarray, raw_mask: np.ndarray, clean_strength: float
    ) -> np.ndarray:
        """
        Boundary-banded 3-channel RGB Guided Filter (He et al. 2013).
        Refines soft neural matte boundaries along ears, temples, collar and shoulders using the
        crisp 2x portrait RGB edges while locking pure background (0.0) and pure interior (1.0).
        """
        h, w = raw_mask.shape[:2]
        r = max(6, min(18, int(round(w * 0.011))))
        k = (2 * r + 1, 2 * r + 1)
        eps = 2.2e-4

        ir = np.ascontiguousarray(rgb01[:, :, 0], dtype=np.float32)
        ig = np.ascontiguousarray(rgb01[:, :, 1], dtype=np.float32)
        ib = np.ascontiguousarray(rgb01[:, :, 2], dtype=np.float32)
        p = np.ascontiguousarray(np.clip(raw_mask, 0.0, 1.0), dtype=np.float32)

        mean_ir = cv2.boxFilter(ir, -1, k)
        mean_ig = cv2.boxFilter(ig, -1, k)
        mean_ib = cv2.boxFilter(ib, -1, k)
        mean_p = cv2.boxFilter(p, -1, k)

        cov_r = cv2.boxFilter(ir * p, -1, k) - mean_ir * mean_p
        cov_g = cv2.boxFilter(ig * p, -1, k) - mean_ig * mean_p
        cov_b = cv2.boxFilter(ib * p, -1, k) - mean_ib * mean_p

        vrr = cv2.boxFilter(ir * ir, -1, k) - mean_ir * mean_ir + eps
        vrg = cv2.boxFilter(ir * ig, -1, k) - mean_ir * mean_ig
        vrb = cv2.boxFilter(ir * ib, -1, k) - mean_ir * mean_ib
        vgg = cv2.boxFilter(ig * ig, -1, k) - mean_ig * mean_ig + eps
        vgb = cv2.boxFilter(ig * ib, -1, k) - mean_ig * mean_ib
        vbb = cv2.boxFilter(ib * ib, -1, k) - mean_ib * mean_ib + eps

        inv_rr = vgg * vbb - vgb * vgb
        inv_rg = vgb * vrb - vrg * vbb
        inv_rb = vrg * vgb - vgg * vrb
        inv_gg = vrr * vbb - vrb * vrb
        inv_gb = vrg * vrb - vrr * vgb
        inv_bb = vrr * vgg - vrg * vrg
        det = np.maximum(1e-12, vrr * inv_rr + vrg * inv_rg + vrb * inv_rb)

        ar = (inv_rr * cov_r + inv_rg * cov_g + inv_rb * cov_b) / det
        ag = (inv_rg * cov_r + inv_gg * cov_g + inv_gb * cov_b) / det
        ab = (inv_rb * cov_r + inv_gb * cov_g + inv_bb * cov_b) / det
        b = mean_p - ar * mean_ir - ag * mean_ig - ab * mean_ib

        q = (
            cv2.boxFilter(ar, -1, k) * ir
            + cv2.boxFilter(ag, -1, k) * ig
            + cv2.boxFilter(ab, -1, k) * ib
            + cv2.boxFilter(b, -1, k)
        )

        # Smooth matte stretch inside the edge band; keep pure exterior=0 and pure interior=1 locked
        s = float(max(0.0, min(1.0, clean_strength)))
        lo = 0.06 + s * 0.12
        hi = 0.94 - s * 0.08
        stretched = np.clip((q - lo) / max(0.05, hi - lo), 0.0, 1.0)

        kb = (2 * (r + 4) + 1, 2 * (r + 4) + 1)
        band_p = cv2.boxFilter(p, -1, kb)
        stretched = np.where((band_p < 0.004) & (p < 0.02), 0.0, stretched)
        stretched = np.where((band_p > 0.996) & (p > 0.98), 1.0, stretched)
        return stretched.astype(np.float32)

    def _apply_designer_retouch(
        self,
        rgb: np.ndarray,
        fg_mask: np.ndarray,
        crop_left: float,
        crop_top: float,
        crop_w: float,
        crop_h: float,
        metrics: FaceMetrics,
        settings: EditSettings,
    ) -> Tuple[np.ndarray, int]:
        out = rgb.copy()
        work_h, work_w = out.shape[:2]
        sx = work_w / crop_w
        sy = work_h / crop_h

        def map_pt(pt: Tuple[float, float]) -> Tuple[float, float]:
            return ((pt[0] - crop_left) * sx, (pt[1] - crop_top) * sy)

        le_pt = map_pt(metrics.left_eye_center) if metrics.left_eye_center else None
        re_pt = map_pt(metrics.right_eye_center) if metrics.right_eye_center else None
        eye_r = metrics.eye_radius_px * sx

        # 1. Surgical Red-Eye Removal strictly inside exact MediaPipe iris/pupil circles
        red_eye_fixed = 0
        if settings.red_eye_fix:
            pupil_r = max(5.0, eye_r * 0.50)
            for eye in [p for p in (le_pt, re_pt) if p is not None]:
                ex, ey = eye
                x0 = max(0, int(ex - pupil_r))
                x1 = min(work_w, int(ex + pupil_r) + 1)
                y0 = max(0, int(ey - pupil_r))
                y1 = min(work_h, int(ey + pupil_r) + 1)
                if x1 <= x0 or y1 <= y0:
                    continue
                yy, xx = np.mgrid[y0:y1, x0:x1]
                dist = np.hypot(xx - ex, yy - ey)
                patch = out[y0:y1, x0:x1].astype(np.float32)
                r, g, b = patch[:, :, 0], patch[:, :, 1], patch[:, :, 2]
                gb = np.maximum(g, b)
                is_red = (dist <= pupil_r) & (np.minimum(g, b) < 170.0) & (r > 65.0) & (r > gb * 1.65) & ((r - gb) > 42.0)
                cnt = int(is_red.sum())
                if cnt > 0:
                    red_eye_fixed += cnt
                    strength = np.clip((r - gb * 1.45) / 52.0, 0.0, 1.0) * (1.0 - (dist / pupil_r) * 0.15)
                    target_l = (g * 0.50 + b * 0.50) * 0.35
                    patch[:, :, 0] = np.where(is_red, r * (1.0 - strength) + target_l * strength, r)
                    patch[:, :, 1] = np.where(is_red, g * (1.0 - strength * 0.45) + target_l * strength * 0.45, g)
                    patch[:, :, 2] = np.where(is_red, b * (1.0 - strength * 0.45) + target_l * strength * 0.45, b)
                    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

        # Face geometry in working tile pixels
        face_cx = (metrics.face_center_x - crop_left) * sx
        face_top_y = (metrics.hair_top_y + (metrics.eye_line_y - metrics.hair_top_y) * 0.35 - crop_top) * sy
        face_bot_y = (metrics.chin_y - crop_top) * sy
        face_cy = (face_top_y + face_bot_y) * 0.52
        face_rx = max(20.0, metrics.face_bbox_norm[2] * metrics.image_width * sx * 0.52)
        face_ry = max(25.0, (face_bot_y - face_top_y) * 0.56)

        if metrics.mouth_rect:
            mx, my, mw, mh = metrics.mouth_rect
            mouth_cx = (mx + mw * 0.5 - crop_left) * sx
            mouth_cy = (my + mh * 0.5 - crop_top) * sy
            mouth_rx = max(12.0, mw * sx * 0.65)
            mouth_ry = max(8.0, mh * sy * 0.78)
        else:
            mouth_cx, mouth_cy = face_cx, face_cy + face_ry * 0.45
            mouth_rx, mouth_ry = face_rx * 0.36, face_ry * 0.14

        if metrics.nose_rect:
            nx, ny, nw, nh = metrics.nose_rect
            nose_cx = (nx + nw * 0.5 - crop_left) * sx
            nose_cy = (ny + nh * 0.5 - crop_top) * sy
        else:
            nose_cx, nose_cy = face_cx, face_cy + face_ry * 0.08
        nose_rx = max(12.0, face_rx * 0.24)
        nose_ry = max(18.0, face_ry * 0.34)

        roi_x0 = max(0, int(face_cx - face_rx * 1.25))
        roi_x1 = min(work_w, int(face_cx + face_rx * 1.25) + 1)
        roi_y0 = max(0, int(face_cy - face_ry * 1.25))
        roi_y1 = min(work_h, int(face_cy + face_ry * 1.35) + 1)
        if roi_x1 - roi_x0 < 16 or roi_y1 - roi_y0 < 16:
            return out, red_eye_fixed

        roi = out[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float32)
        roi_fg = fg_mask[roi_y0:roi_y1, roi_x0:roi_x1]
        yy, xx = np.mgrid[roi_y0:roi_y1, roi_x0:roi_x1].astype(np.float32)

        def smoothstep(e0: float, e1: float, val: np.ndarray) -> np.ndarray:
            t = np.clip((val - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)

        # Face oval skin mask (excluding eyes, eyebrows, lips!)
        f_dist = ((xx - face_cx) / (face_rx * 1.05)) ** 2 + ((yy - face_cy) / (face_ry * 1.12)) ** 2
        mask_w = smoothstep(1.0, 0.58, f_dist) * (roi_fg > 0.78).astype(np.float32)

        for eye in [p for p in (le_pt, re_pt) if p is not None]:
            ex, ey = eye
            e_dist = ((xx - ex) / (eye_r * 1.28)) ** 2 + ((yy - ey) / (eye_r * 0.92)) ** 2
            mask_w *= smoothstep(0.55, 1.0, e_dist)
            b_dist = ((xx - ex) / (eye_r * 1.48)) ** 2 + ((yy - (ey - eye_r * 0.92)) / (eye_r * 0.52)) ** 2
            mask_w *= smoothstep(0.45, 1.0, b_dist)

        m_dist = ((xx - mouth_cx) / mouth_rx) ** 2 + ((yy - mouth_cy) / mouth_ry) ** 2
        mask_w *= smoothstep(0.50, 1.0, m_dist)

        r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        skin_gate = (
            smoothstep(75.0, 118.0, lum)
            * smoothstep(-4.0, 14.0, r - b)
            * ((r >= g - 6.0) & (g >= b - 4.0)).astype(np.float32)
        )
        w = mask_w * skin_gate

        # Average healthy cheek skin reference
        cheek_sel = (
            (((xx - face_cx) / face_rx) ** 2 + ((yy - face_cy) / face_ry) ** 2 < 0.65)
            & (r > 95) & (g > 65) & (b > 45) & (r > g) & (g > b) & ((r - b) > 15) & ((r - b) < 110)
        )
        if int(cheek_sel.sum()) > 20:
            avg_r = float(np.median(r[cheek_sel]))
            avg_g = float(np.median(g[cheek_sel]))
            avg_b = float(np.median(b[cheek_sel]))
        else:
            avg_r, avg_g, avg_b = 210.0, 172.0, 148.0
        avg_lum = max(1.0, 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b)

        # Low-freq matte reference + small-scale blemish reference
        med_k = max(5, int(round(face_rx * 0.065))) * 2 + 1
        sm_k = max(2, int(round(face_rx * 0.022))) * 2 + 1
        med_roi = cv2.blur(roi, (med_k, med_k))
        sm_roi = cv2.blur(roi, (sm_k, sm_k))

        # (A) Gentle edge-preserving skin micro-blemish softening
        if settings.skin_retouch > 0.01:
            diff = np.mean(np.abs(roi - sm_roi), axis=-1)
            edge_preserve = np.exp(-(diff * diff) / 380.0)
            sm_lum = 0.299 * sm_roi[:, :, 0] + 0.587 * sm_roi[:, :, 1] + 0.114 * sm_roi[:, :, 2]
            spot_boost = np.where((sm_lum - lum) > 3.0, 1.35, 1.0)
            blend = np.minimum(0.52, w * float(settings.skin_retouch) * 0.44 * edge_preserve * spot_boost)[..., None]
            roi = roi * (1.0 - blend) + sm_roi * blend

        # (B) Flash glare / de-shine on nose, forehead & cheeks -> local matte skin tone
        if settings.de_shine > 0.01:
            cur_lum = 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2]
            loc_lum = 0.299 * med_roi[:, :, 0] + 0.587 * med_roi[:, :, 1] + 0.114 * med_roi[:, :, 2]
            n_dist = ((xx - nose_cx) / nose_rx) ** 2 + ((yy - nose_cy) / nose_ry) ** 2
            nose_boost = np.where(n_dist < 1.0, 1.35, 1.0)

            high_peak = smoothstep(190.0, 232.0, cur_lum)
            local_peak = smoothstep(2.5, 22.0, cur_lum - loc_lum)
            shine_mask = np.maximum(high_peak * 0.72, local_peak * smoothstep(172.0, 212.0, cur_lum))
            shine_blend = np.minimum(0.70, w * float(settings.de_shine) * 0.68 * shine_mask * nose_boost)[..., None]

            target_lum = np.minimum(cur_lum, np.maximum(loc_lum * 0.985, avg_lum * 1.08))
            scale = (target_lum / avg_lum)[..., None]
            avg_vec = np.array([avg_r, avg_g, avg_b], dtype=np.float32)[None, None, :]
            matte = np.clip(avg_vec * scale * 0.55 + med_roi * 0.45, 0.0, 255.0)
            roi = roi * (1.0 - shine_blend) + matte * shine_blend

        # (C) Subtle skin-tone harmonization (color-mode polish, zero blur!)
        if settings.skin_tone_polish > 0.01:
            cur_lum = np.maximum(1.0, 0.299 * roi[:, :, 0] + 0.587 * roi[:, :, 1] + 0.114 * roi[:, :, 2])
            ratio = (cur_lum / avg_lum)[..., None]
            avg_vec = np.array([avg_r, avg_g, avg_b], dtype=np.float32)[None, None, :]
            ideal = avg_vec * ratio
            redness_excess = np.maximum(0.0, (roi[:, :, 0] - roi[:, :, 1]) - (avg_r - avg_g) * (cur_lum / avg_lum))
            red_boost = 1.0 + np.minimum(0.7, redness_excess / 18.0)
            t_blend = np.minimum(0.42, w * float(settings.skin_tone_polish) * 0.26 * red_boost)[..., None]
            roi = roi * (1.0 - t_blend) + ideal * t_blend

        out[roi_y0:roi_y1, roi_x0:roi_x1] = np.clip(np.rint(roi), 0, 255).astype(np.uint8)
        return out, red_eye_fixed

    # MARK: - 3. Render Complete Vertical 10x15 Sheet
    def render_sheet(self, photo: LoadedPhoto, settings: EditSettings) -> RenderedSheet:
        W = SheetConstants.SHEET_WIDTH_PX
        H = SheetConstants.SHEET_HEIGHT_PX
        placed_tiles = settings.preset.compute_tiles(settings.gap_mm)

        cache: Dict[str, RenderedTilePack] = {}
        rendered_tiles: List[RenderedTilePack] = []
        for pt in placed_tiles:
            key = f"{pt.width}x{pt.height}_{int(round(pt.head_fraction * 1000))}_{int(round(pt.headroom_fraction * 1000))}"
            if key in cache:
                ex = cache[key]
                rendered_tiles.append(RenderedTilePack(
                    placed=pt,
                    unmasked_rgb=ex.unmasked_rgb,
                    mask_gray=ex.mask_gray,
                    composited_rgb=ex.composited_rgb,
                    head_height_mm=ex.head_height_mm,
                    headroom_mm=ex.headroom_mm,
                    left_eye_mm=ex.left_eye_mm,
                    right_eye_mm=ex.right_eye_mm,
                    eye_line_mm=ex.eye_line_mm,
                    red_eye_fixed_pixels=ex.red_eye_fixed_pixels,
                ))
            else:
                pack = self.render_tile(photo, pt, settings)
                cache[key] = pack
                rendered_tiles.append(pack)

        comp = np.full((H, W, 3), 255, dtype=np.uint8)
        for t in rendered_tiles:
            pt = t.placed
            comp[pt.y : pt.y + pt.height, pt.x : pt.x + pt.width] = t.composited_rgb

        # Optional Corners layer
        corner_rgba = None
        if settings.corner_style != CornerStyle.NONE:
            corner_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            for t in rendered_tiles:
                self._draw_corner(settings.corner_style, t.placed, corner_rgba)
            self._alpha_blend(corner_rgba, comp)

        # Optional Cut Marks layer
        cut_rgba = None
        if settings.cut_marks != CutMarkStyle.NONE:
            cut_rgba = np.zeros((H, W, 4), dtype=np.uint8)
            self._draw_cut_marks(settings.cut_marks, placed_tiles, cut_rgba)
            self._alpha_blend(cut_rgba, comp)

        # Bottom "Дни Озёрного" logo (strictly below photos & bottom cut ticks!)
        logo_layer = None
        use_gray = (
            (settings.color_mode == ColorMode.BW)
            if settings.logo_style == LogoStyle.AUTO
            else (settings.logo_style == LogoStyle.GRAY)
        )
        src_logo = (self.logo_gray_rgba if use_gray else self.logo_color_rgba)
        if src_logo is None:
            src_logo = self.logo_color_rgba if self.logo_color_rgba is not None else self.logo_gray_rgba

        if settings.logo_style != LogoStyle.NONE and src_logo is not None:
            max_tile_bot = max((pt.y + pt.height for pt in placed_tiles), default=H - 220)
            tick_clear = 18 if settings.cut_marks == CutMarkStyle.NONE else 44
            safe_top = max_tile_bot + tick_clear
            bottom_margin = 32
            avail_h = max(48, H - bottom_margin - safe_top)

            lh, lw = src_logo.shape[:2]
            aspect = lw / float(lh)
            target_h = min(avail_h * 0.88, 138.0)
            target_w = min(W * 0.42, target_h * aspect)
            final_w = max(80, int(round(target_w)))
            final_h = max(24, int(round(final_w / aspect)))
            logo_x = (W - final_w) // 2
            logo_y = safe_top + max(0, (H - bottom_margin - safe_top - final_h) // 2)

            # Premultiply -> resize -> unpremultiply for crisp anti-aliased logo edges!
            a_f = src_logo[:, :, 3:4].astype(np.float32) / 255.0
            pm = np.concatenate([src_logo[:, :, :3].astype(np.float32) * a_f, src_logo[:, :, 3:4].astype(np.float32)], axis=-1)
            resized_pm = cv2.resize(pm, (final_w, final_h), interpolation=cv2.INTER_AREA)
            ra = np.clip(resized_pm[:, :, 3:4], 0.0, 255.0)
            rgb_unpm = np.where(ra > 0.5, np.clip(resized_pm[:, :, :3] * 255.0 / np.maximum(1.0, ra), 0, 255), 0)
            logo_rgba = np.concatenate([rgb_unpm, ra], axis=-1).astype(np.uint8)
            logo_layer = (logo_x, logo_y, logo_rgba)

            sub_comp = comp[logo_y : logo_y + final_h, logo_x : logo_x + final_w]
            self._alpha_blend(logo_rgba, sub_comp)

        # Photoshop Guides
        xs = sorted({x for pt in placed_tiles for x in (pt.x, pt.x + pt.width)})
        ys = sorted({y for pt in placed_tiles for y in (pt.y, pt.y + pt.height)})
        guides = [PSDGuide(gx, False) for gx in xs] + [PSDGuide(gy, True) for gy in ys]

        first = rendered_tiles[0] if rendered_tiles else None
        return RenderedSheet(
            width=W,
            height=H,
            tiles=rendered_tiles,
            corner_overlay_rgba=corner_rgba,
            cut_marks_rgba=cut_rgba,
            logo_layer=logo_layer,
            guides=guides,
            composite_rgb=comp,
            primary_head_mm=first.head_height_mm if first else 0.0,
            primary_headroom_mm=first.headroom_mm if first else 0.0,
            primary_left_eye_mm=first.left_eye_mm if first else 0.0,
            primary_right_eye_mm=first.right_eye_mm if first else 0.0,
            primary_eye_line_mm=first.eye_line_mm if first else 0.0,
            red_eye_fixed_pixels=first.red_eye_fixed_pixels if first else 0,
        )

    def render_closeup_portrait(self, photo: LoadedPhoto, settings: EditSettings) -> PortraitPreviewResult:
        tiles = settings.preset.compute_tiles(settings.gap_mm)
        base = tiles[0]
        hi_tile = PlacedTile(
            index=1, x=0, y=0,
            width=base.width * 2, height=base.height * 2,
            width_mm=base.width_mm, height_mm=base.height_mm,
            head_fraction=base.head_fraction, headroom_fraction=base.headroom_fraction,
        )
        pack = self.render_tile(photo, hi_tile, settings)
        comp = pack.composited_rgb.copy()
        if settings.corner_style != CornerStyle.NONE:
            c_buf = np.zeros((hi_tile.height, hi_tile.width, 4), dtype=np.uint8)
            self._draw_corner(settings.corner_style, hi_tile, c_buf)
            self._alpha_blend(c_buf, comp)
        return PortraitPreviewResult(
            rgb=comp,
            head_mm=pack.head_height_mm,
            headroom_mm=pack.headroom_mm,
            left_eye_mm=pack.left_eye_mm,
            right_eye_mm=pack.right_eye_mm,
            eye_line_mm=pack.eye_line_mm,
            red_eye_fixed_pixels=pack.red_eye_fixed_pixels,
        )

    # MARK: - 4. Export Layered PSD & 300 DPI sRGB JPEG
    def export_psd(self, sheet: RenderedSheet, settings: EditSettings, out_path: Path) -> None:
        W, H = sheet.width, sheet.height
        layers: List[PSDLayer] = []

        white = np.full((H, W), 255, dtype=np.uint8)
        layers.append(PSDLayer(
            ascii_name="White Background",
            unicode_name="Фон: белый",
            left=0, top=0, width=W, height=H,
            channels=[(0, white), (1, white), (2, white)],
            has_mask=False,
        ))

        for t in sheet.tiles:
            pt = t.placed
            a_full = np.full((pt.height, pt.width), 255, dtype=np.uint8)
            chs = [
                (-1, a_full),
                (0, t.unmasked_rgb[:, :, 0]),
                (1, t.unmasked_rgb[:, :, 1]),
                (2, t.unmasked_rgb[:, :, 2]),
            ]
            if settings.white_background:
                chs.append((-2, t.mask_gray))
            dim_tag = f"{int(round(pt.width_mm))}x{int(round(pt.height_mm))}"
            layers.append(PSDLayer(
                ascii_name=f"Photo {pt.index} ({dim_tag})",
                unicode_name=f"Фото {pt.index} ({dim_tag} мм)",
                left=pt.x, top=pt.y, width=pt.width, height=pt.height,
                channels=chs,
                has_mask=settings.white_background,
                mask_default_color=0,
            ))

        if sheet.corner_overlay_rgba is not None:
            c = sheet.corner_overlay_rgba
            layers.append(PSDLayer(
                ascii_name="Corners",
                unicode_name="Уголки",
                left=0, top=0, width=W, height=H,
                channels=[(-1, c[:, :, 3]), (0, c[:, :, 0]), (1, c[:, :, 1]), (2, c[:, :, 2])],
            ))

        if sheet.cut_marks_rgba is not None:
            m = sheet.cut_marks_rgba
            layers.append(PSDLayer(
                ascii_name="Cut Marks",
                unicode_name="Метки реза",
                left=0, top=0, width=W, height=H,
                channels=[(-1, m[:, :, 3]), (0, m[:, :, 0]), (1, m[:, :, 1]), (2, m[:, :, 2])],
            ))

        if sheet.logo_layer is not None:
            lx, ly, lrgba = sheet.logo_layer
            lh, lw = lrgba.shape[:2]
            layers.append(PSDLayer(
                ascii_name="Dni Ozernogo Logo",
                unicode_name="Дни Озёрного",
                left=lx, top=ly, width=lw, height=lh,
                channels=[(-1, lrgba[:, :, 3]), (0, lrgba[:, :, 0]), (1, lrgba[:, :, 1]), (2, lrgba[:, :, 2])],
            ))

        write_psd(
            canvas_width=W,
            canvas_height=H,
            layers=layers,
            guides=sheet.guides,
            composite_rgb=sheet.composite_rgb,
            out_path=out_path,
            dpi=300,
            icc_profile_path=self.srgb_icc_path,
        )

    def export_jpeg(self, sheet: RenderedSheet, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        icc = self.srgb_icc_path.read_bytes() if (self.srgb_icc_path and self.srgb_icc_path.is_file()) else None
        pil = Image.fromarray(sheet.composite_rgb, mode="RGB")
        save_kw = {"quality": 95, "dpi": (300, 300), "subsampling": 0}
        if icc:
            save_kw["icc_profile"] = icc
        pil.save(out_path, format="JPEG", **save_kw)

    # MARK: - Drawing Helpers
    def _draw_corner(self, style: CornerStyle, tile: PlacedTile, canvas_rgba: np.ndarray) -> None:
        if style == CornerStyle.NONE:
            return
        H, W = canvas_rgba.shape[:2]
        is_right = style in (CornerStyle.RIGHT_OVAL, CornerStyle.RIGHT_STRAIGHT)
        is_oval = style in (CornerStyle.RIGHT_OVAL, CornerStyle.LEFT_OVAL)

        rx = tile.width * (0.36 if is_oval else 0.31)
        ry = tile.height * (0.27 if is_oval else 0.25)
        cx = float(tile.x + tile.width - 1) if is_right else float(tile.x)
        cy = float(tile.y + tile.height - 1)

        min_x = max(0, tile.x, int(math.floor(cx - rx - 4 if is_right else cx)))
        max_x = min(W - 1, tile.x + tile.width - 1, int(math.ceil(cx if is_right else cx + rx + 4)))
        min_y = max(0, tile.y, int(math.floor(cy - ry - 4)))
        max_y = min(H - 1, tile.y + tile.height - 1)
        if max_x < min_x or max_y < min_y:
            return

        yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1].astype(np.float32)
        dx = np.abs(xx - cx)
        dy = np.abs(yy - cy)

        if is_oval:
            dist = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
            t = np.clip((1.0 - dist) / 0.38, 0.0, 1.0)
            alpha_f = np.where(dist <= 0.62, 1.0, np.where(dist >= 1.0, 0.0, t * t * (3.0 - 2.0 * t)))
        else:
            val = (dx / rx) + (dy / ry)
            alpha_f = np.clip((1.02 - val) / 0.06, 0.0, 1.0)

        a_u8 = np.clip(np.rint(alpha_f * 255.0), 0, 255).astype(np.uint8)
        patch = canvas_rgba[min_y : max_y + 1, min_x : max_x + 1]
        mask = a_u8 > 0
        patch[mask, 0] = 255
        patch[mask, 1] = 255
        patch[mask, 2] = 255
        patch[:, :, 3] = np.maximum(patch[:, :, 3], a_u8)

    def _draw_cut_marks(self, style: CutMarkStyle, tiles: List[PlacedTile], canvas_rgba: np.ndarray) -> None:
        if style == CutMarkStyle.NONE or not tiles:
            return
        H, W = canvas_rgba.shape[:2]

        def draw_line_h(x0: int, x1: int, y: int, gray: int, alpha: int) -> None:
            if 0 <= y < H:
                xa, xb = max(0, x0), min(W - 1, x1)
                if xb >= xa:
                    canvas_rgba[y, xa : xb + 1, :3] = gray
                    canvas_rgba[y, xa : xb + 1, 3] = np.maximum(canvas_rgba[y, xa : xb + 1, 3], alpha)

        def draw_line_v(x: int, y0: int, y1: int, gray: int, alpha: int) -> None:
            if 0 <= x < W:
                ya, yb = max(0, y0), min(H - 1, y1)
                if yb >= ya:
                    canvas_rgba[ya : yb + 1, x, :3] = gray
                    canvas_rgba[ya : yb + 1, x, 3] = np.maximum(canvas_rgba[ya : yb + 1, x, 3], alpha)

        if style == CutMarkStyle.FRAME_AND_TICKS:
            for pt in tiles:
                x0, x1 = pt.x - 1, pt.x + pt.width
                y0, y1 = pt.y - 1, pt.y + pt.height
                draw_line_h(x0, x1, y0, 185, 220)
                draw_line_h(x0, x1, y1, 185, 220)
                draw_line_v(x0, y0, y1, 185, 220)
                draw_line_v(x1, y0, y1, 185, 220)

        min_x = min(pt.x for pt in tiles)
        max_x = max(pt.x + pt.width for pt in tiles)
        min_y = min(pt.y for pt in tiles)
        max_y = max(pt.y + pt.height for pt in tiles)
        tick_gap, tick_len = 8, 24

        x_lines = sorted({v for pt in tiles for v in (pt.x - 1, pt.x + pt.width)})
        y_lines = sorted({v for pt in tiles for v in (pt.y - 1, pt.y + pt.height)})

        for x in x_lines:
            draw_line_v(x, min_y - 1 - tick_gap - tick_len + 1, min_y - 1 - tick_gap, 120, 255)
            draw_line_v(x, max_y + tick_gap, max_y + tick_gap + tick_len - 1, 120, 255)
        for y in y_lines:
            draw_line_h(min_x - 1 - tick_gap - tick_len + 1, min_x - 1 - tick_gap, y, 120, 255)
            draw_line_h(max_x + tick_gap, max_x + tick_gap + tick_len - 1, y, 120, 255)

    def _alpha_blend(self, overlay_rgba: np.ndarray, target_rgb: np.ndarray) -> None:
        a = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        blended = overlay_rgba[:, :, :3].astype(np.float32) * a + target_rgb.astype(np.float32) * (1.0 - a)
        target_rgb[:] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
