# SoundGuard expanded benchmark: dataset leakage audit

Audit date: 2026-08-19  
Exact production CED checkpoint: `mispeech/ced-tiny`  
Repository baseline: `24346d879634f65eeb8a647bdba0ac84e249c63d`

This audit was completed before expanded-dataset download. Repository-wide searches covered source, configuration, documentation, benchmark manifests, model references, training/fine-tuning terms, and the bundled DTLN documentation. No SoundGuard CED training notebook, local fine-tuning code, model weights, or train/test membership file exists in this repository. The exact CED checkpoint's model card states that CED was trained on AudioSet. Consequently, AudioSet is not an independent evaluation source for this checkpoint.

| Dataset | Used in training? | Known train files/splits? | Safe for evaluation? | Notes |
| --- | --- | --- | --- | --- |
| AudioSet | **YES** for the stock CED checkpoint | No exact upstream clip membership is stored locally | **NO** as independent CED validation | The CED model card identifies AudioSet as training data. AudioSet is excluded from expanded evaluation. |
| ESC-50 | No evidence that the exact checkpoint was trained or fine-tuned on ESC-50 | Official metadata supplies five folds; no project-specific train fold exists because no local fine-tuning was found | **YES with `POSSIBLE SOURCE OVERLAP` caveat** | ESC-50 is a distinct labeled benchmark, but exact AudioSet source membership cannot be reconstructed locally and public-web source overlap cannot be ruled out. Results are external dataset-level evaluation, not guaranteed source-clip-independent validation. All 40 clips from each selected official class are included; no result-driven selection occurs. |
| UrbanSound8K | No evidence in the checkpoint documentation or repository | Official ten-fold metadata exists; no project training split exists | **YES with `POSSIBLE SOURCE OVERLAP` caveat**, if used | Public source overlap with AudioSet cannot be ruled out. A mapping is documented, but the full archive is not required for tonight's primary 280-clip evaluation and is not selected by default. |
| FSD50K | No evidence in the checkpoint documentation or repository | Official development/evaluation splits exist; none are referenced locally | `POSSIBLE TRAINING OVERLAP`; not selected | Both FSD50K and AudioSet originate from broad public sound collections, so source overlap cannot be excluded without file-level upstream membership. Its large download is unnecessary for the bounded experiment. |
| VIVOS | Not applicable to CED; Google STT training membership is undisclosed | Official corpus train/test structure and human transcripts are available | **Usable for external STT evaluation with provider-training uncertainty** | No project training occurs. It is impossible to prove that a proprietary cloud STT model never encountered a public utterance; this limitation is reported rather than treated as known leakage. |
| Mozilla Common Voice Vietnamese | Not applicable to CED; Google STT training membership is undisclosed | Official validated splits exist | **Usable with provider-training uncertainty**, if obtained legitimately | Not selected by default if license acceptance/authentication gates automated acquisition; no gate will be bypassed. |
| DEMAND | No evidence in CED training; not named in bundled DTLN training instructions | Official recordings are environments, not class-labeled CED train samples | **YES for noise robustness; DTLN overlap unknown but unlikely** | Bundled DTLN documentation says its weights were trained on 500 h of DNS Challenge noisy speech, not DEMAND. DEMAND is used only as an additive noise source. Exact upstream DNS noise membership cannot be cryptographically reconstructed, so DTLN results retain an overlap-uncertainty note. |

## Evidence and decision

- CED model card: <https://huggingface.co/mispeech/ced-tiny> (AudioSet training statement and Apache-2.0 model license).
- ESC-50 official repository: <https://github.com/karolpiczak/ESC-50> (2,000 five-second clips, 50 classes, five official folds, CC BY-NC dataset license).
- UrbanSound8K official page: <https://urbansounddataset.weebly.com/urbansound8k.html> (official metadata/folds and license).
- VIVOS official replacement archive: <https://zenodo.org/records/7068130> (the record identifies itself as the official replacement for the former AILAB distribution).
- DEMAND official research archive: <https://zenodo.org/records/1227120> (CC BY-SA 3.0).
- Bundled DTLN provenance: `external/DTLN-master/README.md` states that the distributed model was trained on DNS Challenge noisy speech.

Primary CED evaluation will therefore use all clips in seven semantically valid ESC-50 classes (expected N=280), with `POSSIBLE SOURCE OVERLAP` attached to every reported CED result. It must not be described as proven leakage-free or as independent AudioSet validation. AudioSet itself is excluded.
