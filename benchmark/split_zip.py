"""Minimal verified extractor for PKWARE split ZIP archives.

The implementation follows the PKWARE APPNOTE record layout.  It is purposely
read-only and supports only extraction of non-encrypted entries.  Every output
is checked against the central-directory uncompressed size and CRC-32.
"""

from __future__ import annotations

import bz2
import lzma
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
EOCD = struct.Struct("<4s4H2IH")
CENTRAL = struct.Struct("<4s6H3I5H2I")
LOCAL = struct.Struct("<4s5H3I2H")


@dataclass(frozen=True)
class SplitZipEntry:
    name: str
    flags: int
    method: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    start_disk: int
    local_offset: int


def _zip64_values(
    extra: bytes,
    uncompressed: int,
    compressed: int,
    offset: int,
    disk: int,
) -> tuple[int, int, int, int]:
    position = 0
    payload = None
    while position + 4 <= len(extra):
        field_id, size = struct.unpack_from("<HH", extra, position)
        position += 4
        value = extra[position : position + size]
        position += size
        if field_id == 0x0001:
            payload = value
            break
    if payload is None:
        if 0xFFFFFFFF in (uncompressed, compressed, offset) or disk == 0xFFFF:
            raise RuntimeError("ZIP64 sentinel without ZIP64 extra field")
        return uncompressed, compressed, offset, disk
    position = 0

    def take(fmt: str) -> int:
        nonlocal position
        value = struct.unpack_from(fmt, payload, position)[0]
        position += struct.calcsize(fmt)
        return int(value)

    if uncompressed == 0xFFFFFFFF:
        uncompressed = take("<Q")
    if compressed == 0xFFFFFFFF:
        compressed = take("<Q")
    if offset == 0xFFFFFFFF:
        offset = take("<Q")
    if disk == 0xFFFF:
        disk = take("<I")
    return uncompressed, compressed, offset, disk


def read_central_directory(segments: list[Path]) -> list[SplitZipEntry]:
    if not segments:
        raise ValueError("At least one ZIP segment is required")
    last = segments[-1]
    with last.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        tail_size = min(size, 65_557)
        stream.seek(size - tail_size)
        tail = stream.read(tail_size)
    eocd_at = tail.rfind(EOCD_SIGNATURE)
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record was not found")
    fields = EOCD.unpack_from(tail, eocd_at)
    _, disk_number, central_disk, entries_on_disk, total_entries, central_size, central_offset, comment_size = fields
    if disk_number != len(segments) - 1:
        raise RuntimeError(f"Archive expects final disk {disk_number}, received {len(segments) - 1}")
    if central_disk >= len(segments):
        raise RuntimeError(f"Central directory starts on missing disk {central_disk}")
    if entries_on_disk != total_entries:
        raise RuntimeError("Central directory spanning multiple disks is not supported")
    if eocd_at + EOCD.size + comment_size > len(tail):
        raise RuntimeError("Truncated ZIP comment")

    entries = []
    with segments[central_disk].open("rb") as stream:
        stream.seek(central_offset)
        consumed = 0
        while len(entries) < total_entries:
            fixed = stream.read(CENTRAL.size)
            if len(fixed) != CENTRAL.size:
                raise RuntimeError("Truncated central-directory entry")
            values = CENTRAL.unpack(fixed)
            if values[0] != CENTRAL_SIGNATURE:
                raise RuntimeError("Invalid central-directory signature")
            (
                _,
                _made_by,
                _needed,
                flags,
                method,
                _time,
                _date,
                crc32,
                compressed,
                uncompressed,
                name_size,
                extra_size,
                comment_size,
                start_disk,
                _internal,
                _external,
                local_offset,
            ) = values
            name_bytes = stream.read(name_size)
            extra = stream.read(extra_size)
            comment = stream.read(comment_size)
            if len(name_bytes) != name_size or len(extra) != extra_size or len(comment) != comment_size:
                raise RuntimeError("Truncated central-directory variable data")
            uncompressed, compressed, local_offset, start_disk = _zip64_values(
                extra, uncompressed, compressed, local_offset, start_disk
            )
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding)
            entries.append(
                SplitZipEntry(
                    name=name,
                    flags=flags,
                    method=method,
                    crc32=crc32,
                    compressed_size=compressed,
                    uncompressed_size=uncompressed,
                    start_disk=start_disk,
                    local_offset=local_offset,
                )
            )
            consumed += CENTRAL.size + name_size + extra_size + comment_size
    if consumed != central_size:
        raise RuntimeError(f"Central-directory size mismatch: parsed {consumed}, expected {central_size}")
    return entries


def _read_across_segments(
    segments: list[Path], start_disk: int, offset: int, size: int
) -> bytes:
    remaining = size
    output = bytearray()
    disk = start_disk
    position = offset
    while remaining:
        if disk >= len(segments):
            raise RuntimeError("Compressed data extends beyond the final ZIP segment")
        with segments[disk].open("rb") as stream:
            stream.seek(position)
            chunk = stream.read(remaining)
        output.extend(chunk)
        remaining -= len(chunk)
        if remaining:
            if not chunk and position == 0:
                raise RuntimeError("Unable to read compressed entry data")
            disk += 1
            position = 0
    return bytes(output)


def _entry_data(segments: list[Path], entry: SplitZipEntry) -> bytes:
    fixed = _read_across_segments(segments, entry.start_disk, entry.local_offset, LOCAL.size)
    values = LOCAL.unpack(fixed)
    if values[0] != LOCAL_SIGNATURE:
        raise RuntimeError(f"Invalid local header for {entry.name}")
    flags, method, name_size, extra_size = values[2], values[3], values[-2], values[-1]
    if flags & 0x1:
        raise RuntimeError(f"Encrypted ZIP entry is not supported: {entry.name}")
    if method != entry.method:
        raise RuntimeError(f"Compression-method mismatch for {entry.name}")
    data_offset = entry.local_offset + LOCAL.size + name_size + extra_size
    return _read_across_segments(
        segments, entry.start_disk, data_offset, entry.compressed_size
    )


def _decompress(method: int, payload: bytes) -> bytes:
    if method == 0:
        return payload
    if method == 8:
        return zlib.decompress(payload, -15)
    if method == 12:
        return bz2.decompress(payload)
    if method == 14:
        return lzma.decompress(payload)
    raise RuntimeError(f"Unsupported ZIP compression method: {method}")


def extract_split_zip(segments: list[Path], destination: Path) -> int:
    segments = [path.resolve() for path in segments]
    entries = read_central_directory(segments)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    extracted = 0
    for position, entry in enumerate(entries, 1):
        target = (destination / entry.name).resolve()
        if target != resolved_destination and resolved_destination not in target.parents:
            raise RuntimeError(f"Unsafe ZIP member path: {entry.name}")
        if entry.name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        payload = _decompress(entry.method, _entry_data(segments, entry))
        if len(payload) != entry.uncompressed_size:
            raise RuntimeError(f"Uncompressed size mismatch for {entry.name}")
        if zlib.crc32(payload) & 0xFFFFFFFF != entry.crc32:
            raise RuntimeError(f"CRC-32 mismatch for {entry.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
        extracted += 1
        if extracted % 500 == 0:
            print(f"Verified split-ZIP extraction {extracted}/{total_file_entries(entries)}", flush=True)
    return extracted


def total_file_entries(entries: list[SplitZipEntry]) -> int:
    return sum(not entry.name.endswith("/") for entry in entries)

