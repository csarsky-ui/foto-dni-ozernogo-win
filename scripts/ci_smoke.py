from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_synthetic_portrait(path: Path) -> None:
    img = np.full((1200, 900, 3), 238, dtype=np.uint8)
    yy, xx = np.mgrid[0:1200, 0:900]
    head = ((xx - 450) / 145.0) ** 2 + ((yy - 430) / 195.0) ** 2 <= 1.0
    body = ((xx - 450) / 310.0) ** 2 + ((yy - 980) / 360.0) ** 2 <= 1.0
    img[body] = [68, 76, 62]
    img[head] = [212, 174, 152]
    for ex in (395, 505):
        eye = (xx - ex) ** 2 + (yy - 405) ** 2 <= 14 ** 2
        img[eye] = [40, 48, 55]
    Image.fromarray(img, mode="RGB").save(path, format="JPEG", quality=92)


def verify_outputs(psd_path: Path, jpg_path: Path) -> None:
    assert psd_path.is_file(), f"Missing PSD: {psd_path}"
    assert jpg_path.is_file(), f"Missing JPG: {jpg_path}"

    psd_bytes = psd_path.read_bytes()
    assert psd_bytes[:4] == b"8BPS", f"Invalid PSD header: {psd_bytes[:4]!r}"
    assert len(psd_bytes) > 200_000, f"PSD too small: {len(psd_bytes)} bytes"

    im = Image.open(jpg_path)
    assert im.size == (1181, 1772), f"Unexpected JPG dimensions: {im.size}"
    assert jpg_path.stat().st_size > 20_000, f"JPG too small: {jpg_path.stat().st_size} bytes"
    print(f"VERIFIED OK: PSD={len(psd_bytes):,} B (8BPS), JPG={im.size[0]}x{im.size[1]} ({jpg_path.stat().st_size:,} B)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pre", "exe"], required=True)
    parser.add_argument("--exe", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        src = tmp / "smoke_in.jpg"
        out_psd = tmp / "smoke_out.psd"
        out_jpg = tmp / "smoke_out.jpg"
        log_file = tmp / "exe_run.log"
        make_synthetic_portrait(src)

        if args.phase == "pre":
            import os
            from app.engine import PhotoEngine
            from app.models import EditSettings, PhotoFormatPreset

            eng = PhotoEngine()
            loaded = eng.load_photo(src)
            st = EditSettings(preset=PhotoFormatPreset.PASSPORT_RF)
            sheet = eng.render_sheet(loaded, st)
            eng.export_psd(sheet, st, out_psd)
            eng.export_jpeg(sheet, out_jpg)
            verify_outputs(out_psd, out_jpg)

            # Also verify async PySide6 MainWindow load -> preset switch -> corner/cut/logo switch ->
            # portrait closeup mode -> second photo reload end-to-end!
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            from PySide6.QtCore import QEventLoop, QTimer
            from PySide6.QtWidgets import QApplication
            from app.models import CornerStyle, CutMarkStyle, LogoStyle
            from app.ui import MainWindow

            qt_app = QApplication.instance() or QApplication([])
            win = MainWindow()

            def wait_for_render(action_fn, timeout_ms: int = 30000) -> str:
                res = {"done": False, "mode": None, "err": None}
                loop = QEventLoop()

                def on_ok(gen, mode, _obj):
                    if gen == win._render_gen:
                        res["done"] = True
                        res["mode"] = mode
                        loop.quit()

                def on_rfail(gen, err):
                    if gen == win._render_gen:
                        res["err"] = f"render: {err}"
                        loop.quit()

                def on_lfail(gen, err):
                    if gen == win._load_gen:
                        res["err"] = f"load: {err}"
                        loop.quit()

                win._render_thread.rendered_ok.connect(on_ok)
                win._render_thread.failed.connect(on_rfail)
                win._loader_thread.failed.connect(on_lfail)
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(loop.quit)
                timer.start(timeout_ms)
                try:
                    action_fn()
                    loop.exec()
                finally:
                    timer.stop()
                    win._render_thread.rendered_ok.disconnect(on_ok)
                    win._render_thread.failed.disconnect(on_rfail)
                    win._loader_thread.failed.disconnect(on_lfail)

                assert res["err"] is None, f"GUI step failed: {res['err']}"
                assert res["done"], "GUI step timed out before rendering!"
                assert "Ошибка" not in win.lbl_status.text(), f"Unexpected error status: {win.lbl_status.text()}"
                assert win.canvas._loading_text is None, f"Loading overlay stuck: {win.canvas._loading_text}"
                return str(res["mode"])

            # 1) Initial photo load -> sheet render
            m1 = wait_for_render(lambda: win.load_file(src))
            assert m1 == "sheet", f"Expected sheet mode, got {m1}"

            # 2) Switch preset via QComboBox (reproduces PySide6 str-Enum bug!)
            m2 = wait_for_render(lambda: win.cmb_preset.setCurrentIndex(1))
            assert m2 == "sheet"
            assert isinstance(win.settings.preset, PhotoFormatPreset), f"preset type broken: {type(win.settings.preset)}"
            assert win.settings.preset == PhotoFormatPreset.PASSPORT_RF

            # 3) Switch corner, cut marks, logo via QComboBox + check suggested filename
            wait_for_render(lambda: (
                win.cmb_corner.setCurrentIndex(1),
                win.cmb_cut.setCurrentIndex(1),
                win.cmb_logo.setCurrentIndex(1),
            ))
            assert isinstance(win.settings.corner_style, CornerStyle)
            assert isinstance(win.settings.cut_marks, CutMarkStyle)
            assert isinstance(win.settings.logo_style, LogoStyle)
            assert "_ugolokR" in win._suggested_base_name()

            # 4) Switch to full closeup portrait preview mode ("Крупно портрет")
            m4 = wait_for_render(lambda: win._set_preview_mode("portrait"))
            assert m4 == "portrait", f"Expected portrait mode, got {m4}"

            # 5) Switch preset to COMBO while in portrait mode, then back to sheet
            m5 = wait_for_render(lambda: win.cmb_preset.setCurrentIndex(len(list(PhotoFormatPreset)) - 1))
            assert m5 == "portrait"
            assert win.settings.preset == PhotoFormatPreset.COMBO
            m6 = wait_for_render(lambda: win._set_preview_mode("sheet"))
            assert m6 == "sheet"

            # 6) Reload a second photo sequentially on the same persistent AI loader thread
            src2 = tmp / "smoke_in_2.jpg"
            make_synthetic_portrait(src2)
            m7 = wait_for_render(lambda: win.load_file(src2))
            assert m7 == "sheet"

            win.close()
            print("GUI ASYNC LOAD + PRESET + PORTRAIT + RELOAD OK")
        else:
            if not args.exe or not args.exe.is_file():
                raise FileNotFoundError(f"Built EXE not found: {args.exe}")
            cmd = [
                str(args.exe),
                "--cli",
                str(src),
                "--preset",
                "passport_rf",
                "--out",
                str(out_psd),
                "--jpg",
                str(out_jpg),
                "--log",
                str(log_file),
            ]
            res = subprocess.run(cmd, check=False, timeout=120)
            if log_file.is_file():
                print("=== EXE LOG ===")
                print(log_file.read_text(encoding="utf-8", errors="replace"))
                print("=== END EXE LOG ===")
            if res.returncode != 0:
                raise RuntimeError(f"EXE exited with code {res.returncode}")
            verify_outputs(out_psd, out_jpg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
