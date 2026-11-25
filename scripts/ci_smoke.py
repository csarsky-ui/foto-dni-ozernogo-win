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
    # Torso & head oval silhouette so RMBG-1.4 has a clear foreground subject
    yy, xx = np.mgrid[0:1200, 0:900]
    head = ((xx - 450) / 145.0) ** 2 + ((yy - 430) / 195.0) ** 2 <= 1.0
    body = ((xx - 450) / 310.0) ** 2 + ((yy - 980) / 360.0) ** 2 <= 1.0
    img[body] = [68, 76, 62]
    img[head] = [212, 174, 152]
    # Simple eye circles
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
        make_synthetic_portrait(src)

        if args.phase == "pre":
            from app.engine import PhotoEngine
            from app.models import EditSettings, PhotoFormatPreset

            eng = PhotoEngine()
            loaded = eng.load_photo(src)
            st = EditSettings(preset=PhotoFormatPreset.PASSPORT_RF)
            sheet = eng.render_sheet(loaded, st)
            eng.export_psd(sheet, st, out_psd)
            eng.export_jpeg(sheet, out_jpg)
            verify_outputs(out_psd, out_jpg)
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
            ]
            res = subprocess.run(cmd, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"EXE exited with code {res.returncode}")
            verify_outputs(out_psd, out_jpg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
