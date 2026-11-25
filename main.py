from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Guard stdout/stderr for windowed (console=False) PyInstaller builds on Windows
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from PIL import Image

from app.engine import PhotoEngine
from app.models import ColorMode, CornerStyle, EditSettings, LogoStyle, PhotoFormatPreset


def run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Foto na dokumenty — Dni Ozernogo CLI")
    parser.add_argument("input", type=Path, help="Input RAW or image file")
    parser.add_argument("--preset", default="3x4_6", choices=[p.value for p in PhotoFormatPreset])
    parser.add_argument("--out", type=Path, default=None, help="Output layered PSD path")
    parser.add_argument("--jpg", type=Path, default=None, help="Output 300 DPI JPEG path")
    parser.add_argument("--portrait", type=Path, default=None, help="Output 2x portrait closeup JPEG path")
    parser.add_argument("--bw", action="store_true")
    parser.add_argument("--corner", default="none", choices=[c.value for c in CornerStyle])
    parser.add_argument("--logo", default="auto", choices=[l.value for l in LogoStyle])
    args = parser.parse_args(argv)

    settings = EditSettings()
    settings.preset = PhotoFormatPreset(args.preset)
    settings.gap_mm = settings.preset.default_gap_mm
    settings.color_mode = ColorMode.BW if args.bw else ColorMode.COLOR
    settings.corner_style = CornerStyle(args.corner)
    settings.logo_style = LogoStyle(args.logo)

    engine = PhotoEngine()
    t0 = time.time()
    loaded = engine.load_photo(args.input)
    t_load = time.time() - t0
    sheet = engine.render_sheet(loaded, settings)
    t_render = time.time() - t0 - t_load

    if args.out:
        engine.export_psd(sheet, settings, args.out)
        print(
            f"PSD OK: {args.out} (load={t_load:.2f}s, render={t_render:.2f}s, "
            f"top={sheet.primary_headroom_mm:.1f}mm, head={sheet.primary_head_mm:.1f}mm, "
            f"eyes={sheet.primary_left_eye_mm:.1f}/{sheet.primary_right_eye_mm:.1f}mm, "
            f"redEyePx={sheet.red_eye_fixed_pixels}, autoWB={loaded.auto_temp_delta:.0f}K/{loaded.auto_tint_delta:.1f})"
        )
    if args.jpg:
        engine.export_jpeg(sheet, args.jpg)
        print(
            f"JPG OK: {args.jpg} (top={sheet.primary_headroom_mm:.1f}mm, head={sheet.primary_head_mm:.1f}mm, "
            f"eyes={sheet.primary_left_eye_mm:.1f}/{sheet.primary_right_eye_mm:.1f}mm, redEyePx={sheet.red_eye_fixed_pixels})"
        )
    if args.portrait:
        p = engine.render_closeup_portrait(loaded, settings)
        args.portrait.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(p.rgb, mode="RGB").save(args.portrait, format="JPEG", quality=95)
        print(f"PORTRAIT OK: {args.portrait}")
    return 0


def main() -> int:
    if "--cli" in sys.argv:
        idx = sys.argv.index("--cli")
        return run_cli(sys.argv[idx + 1 :])

    from app.ui import run_gui

    initial = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_file() else None
    return run_gui(initial_file=initial)


if __name__ == "__main__":
    raise SystemExit(main())
