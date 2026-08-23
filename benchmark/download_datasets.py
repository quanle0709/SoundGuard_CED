"""Reproducibly acquire bounded public data for the expanded benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmark_data"
EXTERNAL = DATA / "external"
ESC_META_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC_AUDIO_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"
VIVOS_URL = "https://zenodo.org/records/7068130/files/vivos.tar.gz?download=1"
VIVOS_MD5 = "72972a8b14050f3f11ea7c3debabd7af"
VIVOS_SIZE = 1474408300
DEMAND_BASE = "https://zenodo.org/api/records/1227121/files/{archive}/content"
DEMAND_ARCHIVES = {
    "traffic": "STRAFFIC_16k.zip",
    "cafeteria": "PCAFETER_16k.zip",
    "office": "OOFFICE_16k.zip",
    "home": "DLIVING_16k.zip",
}
DEMAND_SIZES = {"STRAFFIC_16k.zip": 118572691, "PCAFETER_16k.zip": 107431494,
                "OOFFICE_16k.zip": 88995191, "DLIVING_16k.zip": 80170627}
SELECTED_ESC = {
    "siren": ("siren", "exact"),
    "car_horn": ("vehicle_horn", "compatible"),
    "dog": ("dog_barking", "compatible"),
    "crying_baby": ("baby_crying", "exact"),
    "glass_breaking": ("glass_breaking", "exact"),
    "crackling_fire": ("fire", "compatible"),
    "door_wood_knock": ("door_activity", "compatible"),
}


def download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "SoundGuard-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream, 1024 * 1024)
    temporary.replace(path)
    return path


def digest(path: Path, algorithm: str = "sha256") -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def complete_with_ranges(url: str, path: Path, total_size: int, workers: int = 16) -> Path:
    """Resume a large byte-identical archive using bounded parallel HTTP ranges."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.stat().st_size if path.exists() else 0
    if existing >= total_size:
        return path
    completed_chunks = []
    for candidate in path.parent.glob(f"{path.name}.range-*-*"):
        try:
            start_text, end_text = candidate.name.rsplit("range-", 1)[1].split("-")
            completed_chunks.append((int(start_text), int(end_text), candidate))
        except ValueError:
            continue
    completed_chunks.sort()
    reusable = bool(completed_chunks and completed_chunks[-1][1] == total_size - 1 and
                    all(current[1] + 1 == following[0]
                        for current, following in zip(completed_chunks, completed_chunks[1:])) and
                    all(path.stat().st_size == end - start + 1 for start, end, path in completed_chunks))
    if reusable:
        ranges = [(start, end) for start, end, _ in completed_chunks]
        base_length = ranges[0][0]
    else:
        remaining = total_size - existing
        width = (remaining + workers - 1) // workers
        ranges = [(start, min(total_size - 1, start + width - 1))
                  for start in range(existing, total_size, width)]
        base_length = existing

    def fetch(item: tuple[int, int]) -> Path:
        start, end = item
        chunk = path.with_name(f"{path.name}.range-{start}-{end}")
        expected = end - start + 1
        if chunk.exists() and chunk.stat().st_size == expected:
            return chunk
        temporary = chunk.with_suffix(chunk.suffix + ".part")
        for attempt in range(1, 9):
            acquired = temporary.stat().st_size if temporary.exists() else 0
            if acquired == expected:
                break
            if acquired > expected:
                temporary.unlink()
                acquired = 0
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "SoundGuard-benchmark/1.0",
                    "Range": f"bytes={start + acquired}-{end}",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response, temporary.open("ab") as stream:
                    if getattr(response, "status", None) != 206:
                        raise RuntimeError(
                            f"Range request returned HTTP {getattr(response, 'status', None)}"
                        )
                    shutil.copyfileobj(response, stream, 1024 * 1024)
            except (OSError, TimeoutError):
                if attempt == 8:
                    raise
                time.sleep(min(30, attempt * 3))
        if temporary.stat().st_size != expected:
            raise RuntimeError(f"Range size mismatch for {start}-{end}")
        temporary.replace(chunk)
        return chunk

    if reusable:
        chunks = [candidate for _, _, candidate in completed_chunks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(fetch, ranges))
    merged = path.with_suffix(path.suffix + ".merged")
    with merged.open("wb") as output:
        if base_length:
            with path.open("rb") as source:
                remaining = base_length
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        raise RuntimeError("Partial archive ended before the first completed range")
                    output.write(block); remaining -= len(block)
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, output, 1024 * 1024)
    if merged.stat().st_size != total_size:
        raise RuntimeError("Merged range download has incorrect size")
    merged.replace(path)
    for chunk in chunks:
        chunk.unlink()
    return path


def prepare_esc50() -> int:
    root = EXTERNAL / "esc50"
    metadata = download(ESC_META_URL, root / "esc50.csv")
    with metadata.open(encoding="utf-8", newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["category"] in SELECTED_ESC]

    def acquire(row: dict[str, str]) -> None:
        download(ESC_AUDIO_URL.format(filename=row["filename"]), root / "audio" / row["filename"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(acquire, selected))
    manifest_rows = []
    for row in selected:
        target, mapping_type = SELECTED_ESC[row["category"]]
        path = root / "audio" / row["filename"]
        manifest_rows.append({
            "sample_id": path.stem, "path": path.relative_to(ROOT).as_posix(),
            "source_label": row["category"], "true_label": target,
            "mapping_type": mapping_type, "fold": row["fold"], "esc10": row["esc10"],
            "sha256": digest(path), "leakage_status": "POSSIBLE SOURCE OVERLAP",
        })
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader(); writer.writerows(manifest_rows)
    if len(manifest_rows) != 280:
        raise RuntimeError(f"Expected 280 ESC-50 clips, found {len(manifest_rows)}")
    return len(manifest_rows)


def _safe_member(name: str) -> bool:
    parts = Path(name).parts
    return bool(parts) and not Path(name).is_absolute() and ".." not in parts


def prepare_vivos(limit: int = 100) -> int:
    root = EXTERNAL / "vivos"
    archive = root / "vivos.tar.gz"
    if not archive.exists():
        partial = root / "vivos.tar.gz.part"
        complete_with_ranges(VIVOS_URL, partial, VIVOS_SIZE)
        partial.replace(archive)
    if digest(archive, "md5") != VIVOS_MD5:
        raise RuntimeError("VIVOS MD5 mismatch")
    extracted = root / "selected"
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        names = bundle.getnames()
        prompt_name = next(name for name in names if name.lower().endswith("test/prompts.txt"))
        prompt_file = bundle.extractfile(prompt_name)
        if prompt_file is None:
            raise RuntimeError("VIVOS test prompts missing")
        prompts = {}
        for raw in io.TextIOWrapper(prompt_file, encoding="utf-8"):
            utterance, transcript = raw.rstrip("\n").split(" ", 1)
            prompts[utterance] = transcript
        wav_members = sorted((member for member in bundle.getmembers()
                              if member.name.lower().endswith(".wav") and "/test/" in f"/{member.name.lower()}"),
                             key=lambda member: member.name)
        by_speaker = {}
        for member in wav_members:
            by_speaker.setdefault(Path(member.name).parent.name, []).append(member)
        selected = []
        depth = 0
        while len(selected) < limit and any(depth < len(items) for items in by_speaker.values()):
            for speaker in sorted(by_speaker):
                if depth < len(by_speaker[speaker]) and len(selected) < limit:
                    selected.append(by_speaker[speaker][depth])
            depth += 1
        rows = []
        for member in selected:
            if not _safe_member(member.name):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            utterance = Path(member.name).stem
            source = bundle.extractfile(member)
            if source is None or utterance not in prompts:
                continue
            speaker = Path(member.name).parent.name
            output = extracted / speaker / f"{utterance}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                with output.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
            rows.append({"sample_id": utterance, "path": output.relative_to(ROOT).as_posix(),
                         "corpus": "VIVOS", "speaker": speaker,
                         "reference": prompts[utterance], "split": "test", "sha256": digest(output)})
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    if len(rows) < min(limit, 50):
        raise RuntimeError(f"Too few VIVOS utterances: {len(rows)}")
    return len(rows)


def prepare_demand() -> int:
    root = EXTERNAL / "demand"
    selected = []
    for environment, archive_name in DEMAND_ARCHIVES.items():
        archive = root / "archives" / archive_name
        if not archive.exists():
            partial = archive.with_suffix(archive.suffix + ".part")
            complete_with_ranges(DEMAND_BASE.format(archive=archive_name), partial,
                                 DEMAND_SIZES[archive_name], workers=8)
            partial.replace(archive)
        with zipfile.ZipFile(archive) as bundle:
            candidates = sorted(name for name in bundle.namelist()
                                if name.lower().endswith("ch01.wav") or name.lower().endswith("ch1.wav"))
            if not candidates:
                candidates = sorted(name for name in bundle.namelist() if name.lower().endswith(".wav"))
            if not candidates or not _safe_member(candidates[0]):
                raise RuntimeError(f"No safe WAV in {archive_name}")
            output = root / "selected" / f"{environment}.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                with bundle.open(candidates[0]) as source, output.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
            selected.append({"environment": environment, "path": output.relative_to(ROOT).as_posix(),
                             "archive": archive_name, "member": candidates[0], "sha256": digest(output)})
    manifest = root / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=selected[0].keys())
        writer.writeheader(); writer.writerows(selected)
    return len(selected)


def write_dataset_notes(counts: dict[str, int]) -> None:
    text = f"""# Expanded benchmark datasets

All acquisition is reproducible with `python benchmark/download_datasets.py`. Downloaded binaries live under the gitignored `benchmark_data/external/` directory.

| Dataset | Source | License | Downloaded portion | Selected classes / environments | N | Evaluation role | Leakage status |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| ESC-50 | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 | All official clips from seven mapped classes | siren, car_horn, dog, crying_baby, glass_breaking, crackling_fire, door_wood_knock | {counts.get('esc50', 0)} | CED, emergency, robustness, DTLN/CED | POSSIBLE SOURCE OVERLAP with AudioSet; no local fine-tuning found |
| VIVOS | https://zenodo.org/records/7068130 | CC BY-NC-SA 4.0 | 100 official test utterances selected deterministically round-robin by speaker | Vietnamese read speech, 19 test speakers where available | {counts.get('vivos', 0)} | clean/noisy Google Vietnamese STT | Provider-training membership undisclosed |
| DEMAND | https://zenodo.org/records/1227120 | CC BY-SA 3.0 | One 16 kHz channel from four environments | traffic, cafeteria, office, home/living | {counts.get('demand', 0)} | real environmental noise mixing | CED: no known overlap; DTLN upstream DNS membership cannot be reconstructed |

ESC-50 selection is class-complete (40 clips per mapped class), fixed before prediction, and uses official metadata. VIVOS selection is deterministic and transcript-grounded. No generated transcript or model-derived ground truth is used.
"""
    (DATA / "DATASETS.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("esc50", "vivos", "demand"))
    parser.add_argument("--vivos-limit", type=int, default=100)
    args = parser.parse_args()
    counts = {}
    if args.only in (None, "esc50"):
        counts["esc50"] = prepare_esc50(); print(f"ESC-50 ready: {counts['esc50']}")
    if args.only in (None, "vivos"):
        counts["vivos"] = prepare_vivos(args.vivos_limit); print(f"VIVOS ready: {counts['vivos']}")
    if args.only in (None, "demand"):
        counts["demand"] = prepare_demand(); print(f"DEMAND ready: {counts['demand']}")
    for key in ("esc50", "vivos", "demand"):
        manifest = EXTERNAL / key / "manifest.csv"
        if key not in counts and manifest.exists():
            with manifest.open(encoding="utf-8") as stream:
                counts[key] = sum(1 for _ in csv.DictReader(stream))
    write_dataset_notes(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
