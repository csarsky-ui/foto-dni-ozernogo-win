from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Guard stdout/stderr for windowed (console=False) PyInstaller builds on Windows
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")


def _extract_log_path(argv: list[str]) -> Path | None:
    if "--log" in argv:
        idx = argv.index("--log")
        if idx + 1 < len(argv):
            return Path(argv[idx + 1])
    return None


def _write_crash_log(text: str, explicit_log: Path | None = None) -> None:
    targets: list[Path] = []
    if explicit_log is not None:
        targets.append(explicit_log)
    try:
        base = Path(os.environ.get("APPDATA", str(Path.home()))) if sys.platform == "win32" else (Path.home() / ".config")
        targets.append(base / "FotoDniOzernogo" / "crash.log")
    except Exception:
        pass
    for p in targets:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        except Exception:
            pass


def run_cli(argv: list[str]) -> int:
    from PIL import Image

    from app.engine import PhotoEngine
    from app.models import ColorMode, CornerStyle, EditSettings, LogoStyle, PhotoFormatPreset

    parser = argparse.ArgumentParser(description="Foto na dokumenty — Dni Ozernogo CLI")
    parser.add_argument("input", type=Path, help="Input RAW or image file")
    parser.add_argument("--preset", default="3x4_6", choices=[p.value for p in PhotoFormatPreset])
    parser.add_argument("--out", type=Path, default=None, help="Output layered PSD path")
    parser.add_argument("--jpg", type=Path, default=None, help="Output 300 DPI JPEG path")
    parser.add_argument("--portrait", type=Path, default=None, help="Output 2x portrait closeup JPEG path")
    parser.add_argument("--log", type=Path, default=None, help="Optional log/crash output path")
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

    lines: list[str] = []
    if args.out:
        engine.export_psd(sheet, settings, args.out)
        lines.append(
            f"PSD OK: {args.out} (load={t_load:.2f}s, render={t_render:.2f}s, "
            f"top={sheet.primary_headroom_mm:.1f}mm, head={sheet.primary_head_mm:.1f}mm, "
            f"eyes={sheet.primary_left_eye_mm:.1f}/{sheet.primary_right_eye_mm:.1f}mm, "
            f"redEyePx={sheet.red_eye_fixed_pixels}, autoWB={loaded.auto_temp_delta:.0f}K/{loaded.auto_tint_delta:.1f})"
        )
    if args.jpg:
        jpg_settings = EditSettings(**settings.__dict__)
        jpg_settings.logo_style = LogoStyle.NONE
        jpg_sheet = engine.render_sheet(loaded, jpg_settings)
        engine.export_jpeg(jpg_sheet, args.jpg)
        lines.append(
            f"JPG OK (no logo): {args.jpg} (top={jpg_sheet.primary_headroom_mm:.1f}mm, head={jpg_sheet.primary_head_mm:.1f}mm, "
            f"eyes={jpg_sheet.primary_left_eye_mm:.1f}/{jpg_sheet.primary_right_eye_mm:.1f}mm, redEyePx={jpg_sheet.red_eye_fixed_pixels}, pimples={jpg_sheet.pimples_fixed_count})"
        )
    if args.portrait:
        p = engine.render_closeup_portrait(loaded, settings)
        args.portrait.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(p.rgb, mode="RGB").save(args.portrait, format="JPEG", quality=95)
        lines.append(f"PORTRAIT OK: {args.portrait}")

    summary = "\n".join(lines) + "\n"
    print(summary, end="")
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(summary, encoding="utf-8")
    return 0


def main() -> int:
    explicit_log = _extract_log_path(sys.argv[1:])
    try:
        if "--cli" in sys.argv:
            idx = sys.argv.index("--cli")
            return run_cli(sys.argv[idx + 1 :])

        from app.ui import run_gui

        initial = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_file() else None
        return run_gui(initial_file=initial)
    except Exception:
        tb = traceback.format_exc()
        _write_crash_log(tb, explicit_log=explicit_log)
        try:
            sys.stderr.write(tb)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
