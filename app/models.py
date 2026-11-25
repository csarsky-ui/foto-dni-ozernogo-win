from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class SheetConstants:
    DPI: float = 300.0
    SHEET_WIDTH_MM: float = 100.0
    SHEET_HEIGHT_MM: float = 150.0
    # Vertical 10x15 cm @ 300 DPI = 1181 x 1772 px
    SHEET_WIDTH_PX: int = 1181
    SHEET_HEIGHT_PX: int = 1772

    @staticmethod
    def mm_to_px(mm: float) -> int:
        return int(round(mm / 25.4 * SheetConstants.DPI))


@dataclass(frozen=True)
class PlacedTile:
    index: int
    x: int
    y: int
    width: int
    height: int
    width_mm: float
    height_mm: float
    head_fraction: float
    headroom_fraction: float


class PhotoFormatPreset(str, Enum):
    F3X4_6 = "3x4_6"
    PASSPORT_RF = "passport_rf"
    VISA_SCHENGEN = "visa_schengen"
    F25X35_6 = "25x35_6"
    F25X35_9 = "25x35_9"
    F4X5_4 = "4x5_4"
    F4X6_4 = "4x6_4"
    F9X12_1 = "9x12_1"
    COMBO = "combo"

    @property
    def title(self) -> str:
        return {
            self.F3X4_6: "3×4 см — 6 шт (2×3)",
            self.PASSPORT_RF: "3,5×4,5 см — Паспорт РФ (4 шт)",
            self.VISA_SCHENGEN: "3,5×4,5 см — Виза / Загран (4 шт)",
            self.F25X35_6: "2,5×3,5 см — Удостоверение (6 шт)",
            self.F25X35_9: "2,5×3,5 см — Удостоверение (9 шт)",
            self.F4X5_4: "4×5 см — Личное дело (4 шт)",
            self.F4X6_4: "4×6 см — Военкомат / МВД (4 шт)",
            self.F9X12_1: "9×12 см — Портрет (1 шт)",
            self.COMBO: "Комби: 3×4 (4 шт) + 3,5×4,5 (2 шт)",
        }[self]

    @property
    def short_file_tag(self) -> str:
        return {
            self.F3X4_6: "3x4_6sht",
            self.PASSPORT_RF: "pasport_35x45_4sht",
            self.VISA_SCHENGEN: "viza_35x45_4sht",
            self.F25X35_6: "udost_25x35_6sht",
            self.F25X35_9: "udost_25x35_9sht",
            self.F4X5_4: "4x5_4sht",
            self.F4X6_4: "4x6_4sht",
            self.F9X12_1: "9x12_1sht",
            self.COMBO: "kombi_3x4_35x45",
        }[self]

    @property
    def subtitle(self) -> str:
        return {
            self.F3X4_6: "30×40 мм · анкета, пропуск, военный билет · голова ~24.5 мм, верх 3.5 мм",
            self.PASSPORT_RF: "35×45 мм · Паспорт РФ · верх 5 мм, голова 30–32 мм, глаза ~14 мм от краёв",
            self.VISA_SCHENGEN: "35×45 мм · ICAO / Шенген · верх 3–4 мм, голова 32–36 мм (70–80%)",
            self.F25X35_6: "25×35 мм · удостоверение ветерана · 6 шт с широкими полями",
            self.F25X35_9: "25×35 мм · удостоверение ветерана · 9 шт экономично (3×3)",
            self.F4X5_4: "40×50 мм · личное дело / справка · голова ~30 мм",
            self.F4X6_4: "40×60 мм · личное дело / МВД / военкомат · по грудь",
            self.F9X12_1: "90×120 мм · личное дело / крупный портрет · 1 шт",
            self.COMBO: "2 шт 35×45 мм (Паспорт РФ) сверху + 4 шт 30×40 мм снизу",
        }[self]

    @property
    def default_gap_mm(self) -> float:
        return {
            self.F3X4_6: 2.5,
            self.PASSPORT_RF: 4.0,
            self.VISA_SCHENGEN: 4.0,
            self.F25X35_6: 3.5,
            self.F25X35_9: 3.0,
            self.F4X5_4: 3.5,
            self.F4X6_4: 2.8,
            self.F9X12_1: 0.0,
            self.COMBO: 2.5,
        }[self]

    @property
    def primary_tile_mm(self) -> Tuple[float, float]:
        return {
            self.F3X4_6: (30.0, 40.0),
            self.PASSPORT_RF: (35.0, 45.0),
            self.VISA_SCHENGEN: (35.0, 45.0),
            self.F25X35_6: (25.0, 35.0),
            self.F25X35_9: (25.0, 35.0),
            self.F4X5_4: (40.0, 50.0),
            self.F4X6_4: (40.0, 60.0),
            self.F9X12_1: (90.0, 120.0),
            self.COMBO: (30.0, 40.0),
        }[self]

    @property
    def primary_head_spec(self) -> Tuple[float, float]:
        return {
            self.F3X4_6: (24.5 / 40.0, 3.5 / 40.0),
            self.PASSPORT_RF: (31.0 / 45.0, 5.0 / 45.0),
            self.VISA_SCHENGEN: (33.8 / 45.0, 3.8 / 45.0),
            self.F25X35_6: (21.5 / 35.0, 3.0 / 35.0),
            self.F25X35_9: (21.5 / 35.0, 3.0 / 35.0),
            self.F4X5_4: (30.0 / 50.0, 4.5 / 50.0),
            self.F4X6_4: (33.5 / 60.0, 5.0 / 60.0),
            self.F9X12_1: (62.0 / 120.0, 9.0 / 120.0),
            self.COMBO: (24.5 / 40.0, 3.5 / 40.0),
        }[self]

    def compute_tiles(self, gap_mm: float) -> List[PlacedTile]:
        W = SheetConstants.SHEET_WIDTH_PX
        H = SheetConstants.SHEET_HEIGHT_PX
        clamped_gap = max(0.0, min(5.0, gap_mm))

        if self == self.COMBO:
            gap_px = SheetConstants.mm_to_px(min(clamped_gap, 3.0))
            top_w = SheetConstants.mm_to_px(35.0)
            top_h = SheetConstants.mm_to_px(45.0)
            bot_w = SheetConstants.mm_to_px(30.0)
            bot_h = SheetConstants.mm_to_px(40.0)
            total_h = top_h + gap_px + bot_h + gap_px + bot_h
            reserved_bottom = SheetConstants.mm_to_px(15.5)
            avail_h = H - reserved_bottom
            start_y = max(SheetConstants.mm_to_px(4.0), (avail_h - total_h) // 2)

            top_block_w = 2 * top_w + gap_px
            top_start_x = (W - top_block_w) // 2
            bot_block_w = 2 * bot_w + gap_px
            bot_start_x = (W - bot_block_w) // 2

            tiles: List[PlacedTile] = []
            for c in range(2):
                tiles.append(PlacedTile(
                    index=len(tiles) + 1,
                    x=top_start_x + c * (top_w + gap_px),
                    y=start_y,
                    width=top_w,
                    height=top_h,
                    width_mm=35.0,
                    height_mm=45.0,
                    head_fraction=31.0 / 45.0,
                    headroom_fraction=5.0 / 45.0,
                ))
            y_row2 = start_y + top_h + gap_px
            for r in range(2):
                for c in range(2):
                    tiles.append(PlacedTile(
                        index=len(tiles) + 1,
                        x=bot_start_x + c * (bot_w + gap_px),
                        y=y_row2 + r * (bot_h + gap_px),
                        width=bot_w,
                        height=bot_h,
                        width_mm=30.0,
                        height_mm=40.0,
                        head_fraction=24.5 / 40.0,
                        headroom_fraction=3.5 / 40.0,
                    ))
            return tiles

        cols, rows = {
            self.F3X4_6: (2, 3),
            self.PASSPORT_RF: (2, 2),
            self.VISA_SCHENGEN: (2, 2),
            self.F25X35_6: (2, 3),
            self.F25X35_9: (3, 3),
            self.F4X5_4: (2, 2),
            self.F4X6_4: (2, 2),
            self.F9X12_1: (1, 1),
        }[self]

        w_mm, h_mm = self.primary_tile_mm
        tile_w = SheetConstants.mm_to_px(w_mm)
        tile_h = SheetConstants.mm_to_px(h_mm)
        gap_px = 0 if (cols == 1 and rows == 1) else SheetConstants.mm_to_px(clamped_gap)

        block_w = cols * tile_w + max(0, cols - 1) * gap_px
        block_h = rows * tile_h + max(0, rows - 1) * gap_px

        reserved_bottom = SheetConstants.mm_to_px(18.5)
        avail_h = H - reserved_bottom
        start_x = (W - block_w) // 2
        start_y = max(SheetConstants.mm_to_px(4.5), (avail_h - block_h) // 2)

        head_frac, headroom_frac = self.primary_head_spec
        tiles = []
        for r in range(rows):
            for c in range(cols):
                tiles.append(PlacedTile(
                    index=len(tiles) + 1,
                    x=start_x + c * (tile_w + gap_px),
                    y=start_y + r * (tile_h + gap_px),
                    width=tile_w,
                    height=tile_h,
                    width_mm=w_mm,
                    height_mm=h_mm,
                    head_fraction=head_frac,
                    headroom_fraction=headroom_frac,
                ))
        return tiles


class CornerStyle(str, Enum):
    NONE = "none"
    RIGHT_OVAL = "right_oval"
    LEFT_OVAL = "left_oval"
    RIGHT_STRAIGHT = "right_straight"
    LEFT_STRAIGHT = "left_straight"

    @property
    def title(self) -> str:
        return {
            self.NONE: "Без уголка",
            self.RIGHT_OVAL: "Справа внизу (овал)",
            self.LEFT_OVAL: "Слева внизу (овал)",
            self.RIGHT_STRAIGHT: "Справа внизу (прямой)",
            self.LEFT_STRAIGHT: "Слева внизу (прямой)",
        }[self]

    @property
    def short_tag(self) -> str:
        return {
            self.NONE: "",
            self.RIGHT_OVAL: "_ugolokR",
            self.LEFT_OVAL: "_ugolokL",
            self.RIGHT_STRAIGHT: "_ugolokR_pryam",
            self.LEFT_STRAIGHT: "_ugolokL_pryam",
        }[self]


class ColorMode(str, Enum):
    COLOR = "color"
    BW = "bw"

    @property
    def title(self) -> str:
        return {
            self.COLOR: "Цветное",
            self.BW: "Чёрно-белое",
        }[self]


class CutMarkStyle(str, Enum):
    FRAME_AND_TICKS = "frame_ticks"
    TICKS_ONLY = "ticks_only"
    NONE = "none"

    @property
    def title(self) -> str:
        return {
            self.FRAME_AND_TICKS: "Рамка + засечки",
            self.TICKS_ONLY: "Только засечки",
            self.NONE: "Без меток",
        }[self]


class LogoStyle(str, Enum):
    AUTO = "auto"
    COLOR = "color"
    GRAY = "gray"
    NONE = "none"

    @property
    def title(self) -> str:
        return {
            self.AUTO: "Авто (зелёный / серый для ЧБ)",
            self.COLOR: "Цветной «Дни Озёрного»",
            self.GRAY: "Серый «Дни Озёрного»",
            self.NONE: "Без логотипа",
        }[self]


@dataclass
class EditSettings:
    preset: PhotoFormatPreset = PhotoFormatPreset.F3X4_6
    color_mode: ColorMode = ColorMode.COLOR
    corner_style: CornerStyle = CornerStyle.NONE
    white_background: bool = True
    cut_marks: CutMarkStyle = CutMarkStyle.FRAME_AND_TICKS
    logo_style: LogoStyle = LogoStyle.AUTO
    gap_mm: float = 2.5

    # Print options (Windows Epson L8050: 180 deg rotation is stored inside the saved Epson driver profile)
    print_rotate_180: bool = False
    print_copies: int = 1

    # Framing fine-tuning
    zoom: float = 1.0
    shift_y: float = 0.0
    shift_x: float = 0.0
    extra_rotation_deg: float = 0.0

    # Light & WB
    auto_wb: bool = True
    exposure_ev: float = 0.28       # slightly bright print baseline
    temp_shift: float = 0.0         # -900..+900 K
    tint_shift: float = 0.0         # -25..+25
    contrast: float = 1.04
    shadow_lift: float = 0.16

    # Designer Portrait Retouch
    de_shine: float = 0.55
    red_eye_fix: bool = True
    skin_tone_polish: float = 0.45
    skin_retouch: float = 0.35

    # Mask edge
    mask_clean_strength: float = 0.50

    def reset_photo_tweaks(self) -> None:
        self.zoom = 1.0
        self.shift_y = 0.0
        self.shift_x = 0.0
        self.extra_rotation_deg = 0.0
        self.auto_wb = True
        self.exposure_ev = 0.28
        self.temp_shift = 0.0
        self.tint_shift = 0.0
        self.contrast = 1.04
        self.shadow_lift = 0.16
        self.de_shine = 0.55
        self.red_eye_fix = True
        self.skin_tone_polish = 0.45
        self.skin_retouch = 0.35
        self.mask_clean_strength = 0.50


@dataclass
class FaceMetrics:
    image_width: int
    image_height: int
    face_bbox_norm: Tuple[float, float, float, float]  # x, y, w, h top-down
    hair_top_y: float
    eye_line_y: float
    chin_y: float
    face_center_x: float
    eye_roll_rad: float
    left_eye_center: Optional[Tuple[float, float]]     # top-down px
    right_eye_center: Optional[Tuple[float, float]]    # top-down px
    eye_radius_px: float
    nose_rect: Optional[Tuple[float, float, float, float]]
    mouth_rect: Optional[Tuple[float, float, float, float]]
    detected_face: bool
