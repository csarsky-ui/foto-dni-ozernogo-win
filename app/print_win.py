from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QMarginsF, QRectF, QSizeF
from PySide6.QtGui import QImage, QPainter, QPageLayout, QPageSize
from PySide6.QtPrintSupport import QPrintDialog, QPrinter, QPrinterInfo
from PySide6.QtWidgets import QDialog, QWidget


def get_profile_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home())))
    else:
        base = Path.home() / ".config"
    folder = base / "FotoDniOzernogo"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "epson_l8050_profile.devmode"


def has_saved_epson_profile() -> bool:
    p = get_profile_path()
    return p.is_file() and p.stat().st_size >= 220


def detect_epson_l8050() -> Optional[str]:
    names = QPrinterInfo.availablePrinterNames()
    for n in names:
        low = n.lower()
        if "epson" in low and "8050" in low:
            return n
    default_info = QPrinterInfo.defaultPrinter()
    return default_info.printerName() if not default_info.isNull() else (names[0] if names else None)


def configure_and_save_win32_profile(
    parent_widget: Optional[QWidget] = None,
    printer_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Opens the native Windows EPSON L8050 Driver Properties dialog via winspool.drv (pure ctypes)
    and saves the complete binary DEVMODEW (public header + private Epson DriverExtra settings:
    Epson Matte, 10x15, High Quality, EPSON Vivid, Gamma 2.2, Bi-D Off, Reverse Order, Rotate 180°).
    """
    if sys.platform != "win32":
        return False, "Сохранение бинарного профиля драйвера DEVMODE доступно только в Windows."

    target = printer_name or detect_epson_l8050()
    if not target:
        return False, "В системе не найдено ни одного принтера."

    import ctypes
    from ctypes import wintypes

    winspool = ctypes.WinDLL("winspool.drv", use_last_error=True)
    winspool.OpenPrinterW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.HANDLE), ctypes.c_void_p]
    winspool.OpenPrinterW.restype = wintypes.BOOL
    winspool.DocumentPropertiesW.argtypes = [
        wintypes.HWND,
        wintypes.HANDLE,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    winspool.DocumentPropertiesW.restype = ctypes.c_long
    winspool.ClosePrinter.argtypes = [wintypes.HANDLE]
    winspool.ClosePrinter.restype = wintypes.BOOL

    DM_OUT_BUFFER = 0x00000002
    DM_IN_PROMPT = 0x00000004
    DM_IN_BUFFER = 0x00000008
    IDOK = 1

    hwnd = wintypes.HWND(int(parent_widget.winId())) if parent_widget is not None else wintypes.HWND(0)
    hprinter = wintypes.HANDLE()
    if not winspool.OpenPrinterW(target, ctypes.byref(hprinter), None):
        err = ctypes.get_last_error()
        return False, f"Не удалось открыть принтер «{target}» (код Windows {err})."

    try:
        # 1. Query exact DEVMODEW + private Epson DriverExtra buffer size
        size = winspool.DocumentPropertiesW(hwnd, hprinter, target, None, None, 0)
        if size < 220:
            return False, f"Драйвер «{target}» вернул некорректный размер DEVMODE ({size})."

        dm_buf = (ctypes.c_ubyte * size)()
        prof_file = get_profile_path()

        loaded_saved = False
        if prof_file.is_file():
            saved = prof_file.read_bytes()
            if len(saved) == size:
                ctypes.memmove(dm_buf, saved, size)
                loaded_saved = True

        if not loaded_saved:
            res_init = winspool.DocumentPropertiesW(hwnd, hprinter, target, dm_buf, None, DM_OUT_BUFFER)
            if res_init < 0:
                return False, f"Не удалось получить текущие настройки драйвера «{target}»."

        # 2. Show genuine Windows Epson Driver Properties dialog
        res = winspool.DocumentPropertiesW(
            hwnd,
            hprinter,
            target,
            dm_buf,
            dm_buf,
            DM_IN_BUFFER | DM_IN_PROMPT | DM_OUT_BUFFER,
        )
        if res != IDOK:
            return False, "Настройка профиля отменена."

        raw_bytes = bytes(dm_buf)
        prof_file.write_bytes(raw_bytes)
        return (
            True,
            f"Профиль драйвера «{target}» сохранён ({len(raw_bytes)} байт). "
            f"Теперь кнопка «Печать» печатает по этим настройкам в 1 клик.",
        )
    except Exception as exc:
        return False, f"Ошибка сохранения профиля драйвера: {exc}"
    finally:
        winspool.ClosePrinter(hprinter)


def print_direct_with_saved_win32_profile(
    composite_rgb: np.ndarray,
    copies: int = 1,
    extra_rotate_180: bool = False,
    job_title: str = "Foto 10x15",
    printer_name: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Sends the exact 100.0 x 150.0 mm (1:1 physical scale) 300 DPI sheet directly to EPSON L8050
    using the saved Windows DEVMODEW profile via pure Win32 GDI (CreateDCW + HALFTONE StretchDIBits).
    """
    if sys.platform != "win32":
        return False, "Прямая Win32-печать доступна только на Windows."

    prof_file = get_profile_path()
    if not prof_file.is_file():
        return False, "Профиль Epson ещё не сохранён. Нажмите «Настроить профиль L8050» один раз."

    target = printer_name or detect_epson_l8050()
    if not target:
        return False, "Принтер EPSON L8050 не найден."

    try:
        import ctypes
        from ctypes import wintypes

        raw_dm = bytearray(prof_file.read_bytes())
        if len(raw_dm) < 220:
            return False, "Сохранённый файл профиля повреждён. Настройте профиль заново."

        # Patch dmCopies inside DEVMODEW (offset 86) & set DM_COPIES bit (0x100) in dmFields (offset 72)
        DM_COPIES = 0x00000100
        dm_fields = struct.unpack_from("<I", raw_dm, 72)[0]
        struct.pack_into("<I", raw_dm, 72, dm_fields | DM_COPIES)
        struct.pack_into("<h", raw_dm, 86, max(1, min(20, int(copies))))

        class DOCINFOW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_int),
                ("lpszDocName", wintypes.LPCWSTR),
                ("lpszOutput", wintypes.LPCWSTR),
                ("lpszDatatype", wintypes.LPCWSTR),
                ("fwType", wintypes.DWORD),
            ]

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", wintypes.DWORD),
                ("biWidth", wintypes.LONG),
                ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD),
            ]

        gdi32 = ctypes.WinDLL("gdi32.dll", use_last_error=True)
        gdi32.CreateDCW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_void_p]
        gdi32.CreateDCW.restype = wintypes.HDC
        gdi32.DeleteDC.argtypes = [wintypes.HDC]
        gdi32.DeleteDC.restype = wintypes.BOOL
        gdi32.GetDeviceCaps.argtypes = [wintypes.HDC, ctypes.c_int]
        gdi32.GetDeviceCaps.restype = ctypes.c_int
        gdi32.SetStretchBltMode.argtypes = [wintypes.HDC, ctypes.c_int]
        gdi32.SetStretchBltMode.restype = ctypes.c_int
        gdi32.SetBrushOrgEx.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        gdi32.SetBrushOrgEx.restype = wintypes.BOOL
        gdi32.StartDocW.argtypes = [wintypes.HDC, ctypes.POINTER(DOCINFOW)]
        gdi32.StartDocW.restype = ctypes.c_int
        gdi32.StartPage.argtypes = [wintypes.HDC]
        gdi32.StartPage.restype = ctypes.c_int
        gdi32.EndPage.argtypes = [wintypes.HDC]
        gdi32.EndPage.restype = ctypes.c_int
        gdi32.EndDoc.argtypes = [wintypes.HDC]
        gdi32.EndDoc.restype = ctypes.c_int
        gdi32.AbortDoc.argtypes = [wintypes.HDC]
        gdi32.AbortDoc.restype = ctypes.c_int
        gdi32.StretchDIBits.argtypes = [
            wintypes.HDC,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.UINT,
            wintypes.DWORD,
        ]
        gdi32.StretchDIBits.restype = ctypes.c_int

        LOGPIXELSX = 88
        LOGPIXELSY = 90
        PHYSICALWIDTH = 110
        PHYSICALHEIGHT = 111
        PHYSICALOFFSETX = 112
        PHYSICALOFFSETY = 113
        HALFTONE = 4
        DIB_RGB_COLORS = 0
        SRCCOPY = 0x00CC0020

        dm_buf = (ctypes.c_ubyte * len(raw_dm)).from_buffer(raw_dm)
        hdc = gdi32.CreateDCW("WINSPOOL", target, None, ctypes.byref(dm_buf))
        if not hdc:
            err = ctypes.get_last_error()
            return False, f"Не удалось создать контекст печати для «{target}» (код {err})."

        try:
            dpi_x = gdi32.GetDeviceCaps(hdc, LOGPIXELSX)
            dpi_y = gdi32.GetDeviceCaps(hdc, LOGPIXELSY)
            phys_w = gdi32.GetDeviceCaps(hdc, PHYSICALWIDTH)
            phys_h = gdi32.GetDeviceCaps(hdc, PHYSICALHEIGHT)
            off_x = gdi32.GetDeviceCaps(hdc, PHYSICALOFFSETX)
            off_y = gdi32.GetDeviceCaps(hdc, PHYSICALOFFSETY)

            # Exact 100.0 x 150.0 mm in printer hardware dots, centered on physical paper
            target_w = int(round(100.0 / 25.4 * dpi_x))
            target_h = int(round(150.0 / 25.4 * dpi_y))
            x0 = int(round((phys_w - target_w) / 2.0)) - off_x
            y0 = int(round((phys_h - target_h) / 2.0)) - off_y

            img = np.ascontiguousarray(composite_rgb, dtype=np.uint8)
            if extra_rotate_180:
                img = np.ascontiguousarray(img[::-1, ::-1, :])

            # Classic bottom-up 24-bit BGR DIB with 4-byte scanline alignment (100% compatible with all Epson drivers)
            h, w, _ = img.shape
            bgr_bottom_up = img[::-1, :, ::-1]
            row_bytes = w * 3
            stride = (row_bytes + 3) & ~3
            if stride == row_bytes:
                dib_arr = np.ascontiguousarray(bgr_bottom_up, dtype=np.uint8)
            else:
                dib_arr = np.zeros((h, stride), dtype=np.uint8)
                dib_arr[:, :row_bytes] = bgr_bottom_up.reshape(h, row_bytes)

            bmi = BITMAPINFOHEADER()
            bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
            bmi.biWidth = w
            bmi.biHeight = h
            bmi.biPlanes = 1
            bmi.biBitCount = 24
            bmi.biCompression = 0  # BI_RGB
            bmi.biSizeImage = stride * h
            bmi.biXPelsPerMeter = 11811  # 300 DPI
            bmi.biYPelsPerMeter = 11811

            di = DOCINFOW()
            di.cbSize = ctypes.sizeof(DOCINFOW)
            di.lpszDocName = job_title
            di.lpszOutput = None
            di.lpszDatatype = None
            di.fwType = 0

            job_id = gdi32.StartDocW(hdc, ctypes.byref(di))
            if job_id <= 0:
                return False, "Принтер отклонил запуск задания печати (StartDocW)."
            try:
                gdi32.StartPage(hdc)
                gdi32.SetStretchBltMode(hdc, HALFTONE)
                gdi32.SetBrushOrgEx(hdc, 0, 0, None)
                drawn = gdi32.StretchDIBits(
                    hdc,
                    x0,
                    y0,
                    target_w,
                    target_h,
                    0,
                    0,
                    w,
                    h,
                    dib_arr.ctypes.data,
                    ctypes.byref(bmi),
                    DIB_RGB_COLORS,
                    SRCCOPY,
                )
                if drawn <= 0:
                    raise RuntimeError("StretchDIBits вернул 0 строк.")
                gdi32.EndPage(hdc)
                gdi32.EndDoc(hdc)
            except Exception:
                gdi32.AbortDoc(hdc)
                raise

            return True, f"Отправлено на «{target}» ({copies} экз., профиль драйвера, масштаб 1:1)."
        finally:
            gdi32.DeleteDC(hdc)
    except Exception as exc:
        return False, f"Ошибка прямой печати: {exc}"


def print_with_qt_dialog(
    parent: Optional[QWidget],
    composite_rgb: np.ndarray,
    rotate_180: bool = True,
    copies: int = 1,
    job_title: str = "Foto 10x15",
) -> Tuple[bool, str]:
    """
    Opens the native OS Print Dialog (Windows / macOS) preconfigured for 10x15 cm (4x6 in)
    Portrait at exact 1:1 physical millimeter scale (100.0 x 150.0 mm centered on page).
    """
    printer = QPrinter(QPrinter.PrinterMode.HighResolution)
    printer.setResolution(300)
    printer.setDocName(job_title)
    printer.setCopyCount(max(1, min(20, int(copies))))
    printer.setFullPage(True)

    target = detect_epson_l8050()
    if target:
        printer.setPrinterName(target)

    page_size = QPageSize(QSizeF(101.6, 152.4), QPageSize.Unit.Millimeter, "10x15 cm (4x6 in)")
    layout = QPageLayout(
        page_size,
        QPageLayout.Orientation.Portrait,
        QMarginsF(0, 0, 0, 0),
        QPageLayout.Unit.Millimeter,
    )
    printer.setPageLayout(layout)

    dlg = QPrintDialog(printer, parent)
    dlg.setWindowTitle("Печать 10×15 см (масштаб 1:1)")
    if dlg.exec() != QDialog.DialogCode.Accepted:
        return False, "Печать отменена."

    arr = np.ascontiguousarray(composite_rgb, dtype=np.uint8)
    if rotate_180:
        arr = np.ascontiguousarray(arr[::-1, ::-1, :])

    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()

    painter = QPainter()
    if not painter.begin(printer):
        return False, "Не удалось начать печать на выбранном принтере."
    try:
        dpi_x = float(printer.logicalDpiX())
        dpi_y = float(printer.logicalDpiY())
        page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
        draw_w = 100.0 / 25.4 * dpi_x
        draw_h = 150.0 / 25.4 * dpi_y
        dx = page_rect.x() + (page_rect.width() - draw_w) * 0.5
        dy = page_rect.y() + (page_rect.height() - draw_h) * 0.5
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawImage(QRectF(dx, dy, draw_w, draw_h), qimg)
    finally:
        painter.end()

    return True, f"Отправлено на печать ({printer.printerName()}, 10×15 см, масштаб 1:1)."
