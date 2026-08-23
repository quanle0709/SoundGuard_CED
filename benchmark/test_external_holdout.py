from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

from benchmark.external_holdout_common import (
    build_reference_index,
    canonical_pcm_hash,
    esc50_freesound_ids,
    landmark_fingerprint,
    match_landmarks,
    verify_aligned_waveform_ncc,
)
from benchmark.split_zip import CENTRAL, EOCD, extract_split_zip


class ExternalHoldoutMethodologyTests(unittest.TestCase):
    def test_esc50_has_2000_rows_and_1524_unique_source_ids(self):
        self.assertEqual(len(esc50_freesound_ids()), 1524)

    def test_canonical_pcm_hash_ignores_container_subtype(self):
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(32_000) * 0.05).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcm16 = root / "pcm16.wav"
            floating = root / "float.wav"
            quantized = np.rint(audio * 32768.0) / 32768.0
            sf.write(pcm16, quantized, 16_000, subtype="PCM_16")
            sf.write(floating, quantized, 16_000, subtype="FLOAT")
            self.assertEqual(canonical_pcm_hash(pcm16), canonical_pcm_hash(floating))

    def test_landmarks_find_trimmed_same_recording(self):
        sample_rate = 8_000
        rng = np.random.default_rng(42)
        source = rng.standard_normal(sample_rate * 8)
        filtered = sosfilt(
            butter(6, [200, 3400], btype="bandpass", fs=sample_rate, output="sos"),
            source,
        )
        envelope = np.repeat(rng.uniform(0.1, 1.0, 64), 1000)
        audio = (filtered * envelope).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            trimmed = root / "trimmed.wav"
            sf.write(reference, audio, sample_rate, subtype="PCM_16")
            sf.write(trimmed, audio[sample_rate : sample_rate * 7], sample_rate, subtype="PCM_16")
            index, fingerprints = build_reference_index([("reference", reference)])
            evidence = match_landmarks(landmark_fingerprint(trimmed), index, fingerprints)
            self.assertEqual(evidence["decision"], "EXCLUDE_HIGH_CONFIDENCE_NEAR_DUPLICATE")
            self.assertEqual(evidence["matched_reference"], "reference")
            verification = verify_aligned_waveform_ncc(
                trimmed, reference, int(evidence["dominant_offset_frames"])
            )
            self.assertGreaterEqual(verification["aligned_overlap_seconds"], 5.9)
            self.assertGreaterEqual(abs(verification["aligned_waveform_ncc"]), 0.99)

    def test_landmarks_do_not_equate_unrelated_noise(self):
        rng = np.random.default_rng(42)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            candidate = root / "candidate.wav"
            sf.write(reference, rng.standard_normal(48_000).astype(np.float32), 8_000)
            sf.write(candidate, rng.standard_normal(48_000).astype(np.float32), 8_000)
            index, fingerprints = build_reference_index([("reference", reference)])
            evidence = match_landmarks(landmark_fingerprint(candidate), index, fingerprints)
            self.assertEqual(evidence["decision"], "NO_MATCH")

    def test_split_zip_extractor_handles_entry_crossing_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = root / "ordinary.zip"
            first_payload = np.random.default_rng(1).bytes(120_000)
            second_payload = np.random.default_rng(2).bytes(80_000)
            with zipfile.ZipFile(ordinary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("dataset/first.bin", first_payload)
                bundle.writestr("dataset/second.bin", second_payload)
            raw = bytearray(ordinary.read_bytes())
            eocd_at = raw.rfind(b"PK\x05\x06")
            eocd = list(EOCD.unpack_from(raw, eocd_at))
            central_offset = eocd[6]
            central_positions = []
            position = central_offset
            while position < central_offset + eocd[5]:
                values = list(CENTRAL.unpack_from(raw, position))
                central_positions.append((position, values))
                position += CENTRAL.size + values[10] + values[11] + values[12]
            second_local_offset = central_positions[1][1][16]
            boundary = second_local_offset - 1000
            self.assertGreater(boundary, central_positions[0][1][16])

            for central_at, values in central_positions:
                local_offset = values[16]
                if local_offset >= boundary:
                    values[13] = 1
                    values[16] = local_offset - boundary
                    CENTRAL.pack_into(raw, central_at, *values)
            eocd[1] = 1
            eocd[2] = 1
            eocd[6] = central_offset - boundary
            EOCD.pack_into(raw, eocd_at, *eocd)

            first_segment = root / "archive.z01"
            last_segment = root / "archive.zip"
            first_segment.write_bytes(raw[:boundary])
            last_segment.write_bytes(raw[boundary:])
            destination = root / "output"
            self.assertEqual(extract_split_zip([first_segment, last_segment], destination), 2)
            self.assertEqual((destination / "dataset" / "first.bin").read_bytes(), first_payload)
            self.assertEqual((destination / "dataset" / "second.bin").read_bytes(), second_payload)


if __name__ == "__main__":
    unittest.main()
