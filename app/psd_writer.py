from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PSDLayer:
    ascii_name: str
    unicode_name: str
    left: int
    top: int
    width: int
    height: int
    channels: List[Tuple[int, np.ndarray]]  # (channel_id, 2D uint8 array HxW)
    has_mask: bool = False
    mask_default_color: int = 0


@dataclass
class PSDGuide:
    position_px: int
    is_horizontal: bool


def _packbits_row(row: bytes) -> bytes:
    n = len(row)
    out = bytearray()
    i = 0
    while i < n:
        # Look for run of >= 3 identical bytes
        b0 = row[i]
        run_len = 1
        while i + run_len < n and run_len < 128 and row[i + run_len] == b0:
            run_len += 1
        if run_len >= 3:
            out.append((257 - run_len) & 0xFF)  # signed int8(1 - run_len)
            out.append(b0)
            i += run_len
        else:
            start = i
            lit_len = 0
            while i < n and lit_len < 128:
                b = row[i]
                if i + 2 < n and row[i + 1] == b and row[i + 2] == b:
                    break
                i += 1
                lit_len += 1
            if lit_len > 0:
                out.append(lit_len - 1)
                out.extend(row[start : start + lit_len])
    return bytes(out)


def _encode_plane_rle(plane: np.ndarray) -> Tuple[List[int], bytes]:
    arr = np.ascontiguousarray(plane, dtype=np.uint8)
    h = arr.shape[0]
    counts: List[int] = []
    chunks = bytearray()
    for y in range(h):
        packed = _packbits_row(arr[y].tobytes())
        counts.append(len(packed))
        chunks.extend(packed)
    return counts, bytes(chunks)


def _encode_channel_rle(plane: np.ndarray) -> bytes:
    counts, body = _encode_plane_rle(plane)
    header = bytearray(struct.pack(">H", 1))  # Compression = 1 (PackBits RLE)
    for c in counts:
        header.extend(struct.pack(">H", c))
    header.extend(body)
    return bytes(header)


def write_psd(
    canvas_width: int,
    canvas_height: int,
    layers: List[PSDLayer],
    guides: List[PSDGuide],
    composite_rgb: np.ndarray,  # HxWx3 uint8 sRGB
    out_path: Path,
    dpi: int = 300,
    icc_profile_path: Optional[Path] = None,
) -> None:
    psd = bytearray()

    # 1. File Header (26 bytes)
    psd.extend(b"8BPS")
    psd.extend(struct.pack(">H", 1))          # Version 1
    psd.extend(b"\x00" * 6)                   # Reserved
    psd.extend(struct.pack(">H", 3))          # 3 composite RGB channels
    psd.extend(struct.pack(">I", canvas_height))
    psd.extend(struct.pack(">I", canvas_width))
    psd.extend(struct.pack(">H", 8))          # 8-bit
    psd.extend(struct.pack(">H", 3))          # RGB Color Mode

    # 2. Color Mode Data (0 bytes)
    psd.extend(struct.pack(">I", 0))

    # 3. Image Resources Section
    resources = bytearray()

    def append_res(res_id: int, payload: bytes) -> None:
        resources.extend(b"8BIM")
        resources.extend(struct.pack(">HH", res_id, 0))  # empty padded Pascal string
        resources.extend(struct.pack(">I", len(payload)))
        resources.extend(payload)
        if len(payload) % 2 != 0:
            resources.append(0)

    # 3a. ResolutionInfo (0x03ED): 300 DPI, display units = cm
    res_info = struct.pack(">IHH IHH", dpi << 16, 1, 2, dpi << 16, 1, 2)
    append_res(0x03ED, res_info)

    # 3b. Guides (0x0408)
    if guides:
        g_data = bytearray(struct.pack(">IIII", 1, 576, 576, len(guides)))
        for g in guides:
            g_data.extend(struct.pack(">iB", int(g.position_px * 32), 1 if g.is_horizontal else 0))
        append_res(0x0408, bytes(g_data))

    # 3c. sRGB ICC Profile (0x040F)
    if icc_profile_path and icc_profile_path.is_file():
        append_res(0x040F, icc_profile_path.read_bytes())

    psd.extend(struct.pack(">I", len(resources)))
    psd.extend(resources)

    # 4. Layer and Mask Information Section
    encoded_layers: List[List[Tuple[int, bytes]]] = []
    for layer in layers:
        ch_blobs = [(ch_id, _encode_channel_rle(ch_plane)) for ch_id, ch_plane in layer.channels]
        encoded_layers.append(ch_blobs)

    layer_info = bytearray()
    layer_info.extend(struct.pack(">h", len(layers)))

    for idx, layer in enumerate(layers):
        ch_blobs = encoded_layers[idx]
        layer_info.extend(struct.pack(
            ">iiiiH",
            layer.top,
            layer.left,
            layer.top + layer.height,
            layer.left + layer.width,
            len(ch_blobs),
        ))
        for ch_id, blob in ch_blobs:
            layer_info.extend(struct.pack(">hI", ch_id, len(blob)))
        layer_info.extend(b"8BIMnorm")
        layer_info.extend(struct.pack(">BBBB", 255, 0, 0, 0))  # opacity, clipping, flags=visible, filler

        extra = bytearray()
        # Layer Mask Data
        if layer.has_mask:
            extra.extend(struct.pack(
                ">IiiiiBBH",
                20,
                layer.top,
                layer.left,
                layer.top + layer.height,
                layer.left + layer.width,
                layer.mask_default_color,
                0,
                0,
            ))
        else:
            extra.extend(struct.pack(">I", 0))

        # Layer Blending Ranges
        extra.extend(struct.pack(">I", 0))

        # Pascal ASCII Name padded to 4 bytes
        ascii_bytes = layer.ascii_name.encode("ascii", errors="replace")[:255]
        extra.append(len(ascii_bytes))
        extra.extend(ascii_bytes)
        pad4 = (4 - ((1 + len(ascii_bytes)) % 4)) % 4
        if pad4:
            extra.extend(b"\x00" * pad4)

        # Additional Layer Info: 'luni' (Unicode Layer Name, even UTF-16 code units -> 4-byte aligned)
        u_name = layer.unicode_name
        u16 = u_name.encode("utf-16-be")
        if (len(u16) // 2) % 2 != 0:
            u_name = u_name + " "
            u16 = u_name.encode("utf-16-be")
        luni_payload = struct.pack(">I", len(u16) // 2) + u16
        extra.extend(b"8BIMluni")
        extra.extend(struct.pack(">I", len(luni_payload)))
        extra.extend(luni_payload)

        layer_info.extend(struct.pack(">I", len(extra)))
        layer_info.extend(extra)

    for ch_blobs in encoded_layers:
        for _, blob in ch_blobs:
            layer_info.extend(blob)

    if len(layer_info) % 2 != 0:
        layer_info.append(0)

    layer_and_mask = bytearray()
    layer_and_mask.extend(struct.pack(">I", len(layer_info)))
    layer_and_mask.extend(layer_info)
    layer_and_mask.extend(struct.pack(">I", 0))  # Global layer mask info = 0

    psd.extend(struct.pack(">I", len(layer_and_mask)))
    psd.extend(layer_and_mask)

    # 5. Merged Composite Image (3 RGB channels, PackBits RLE)
    psd.extend(struct.pack(">H", 1))
    all_counts: List[int] = []
    all_bodies = bytearray()
    for ch_idx in range(3):
        counts, body = _encode_plane_rle(composite_rgb[:, :, ch_idx])
        all_counts.extend(counts)
        all_bodies.extend(body)
    for c in all_counts:
        psd.extend(struct.pack(">H", c))
    psd.extend(all_bodies)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(psd))
