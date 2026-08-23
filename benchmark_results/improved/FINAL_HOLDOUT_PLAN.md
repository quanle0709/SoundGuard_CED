# Final untouched holdout plan

The 280 ESC-50 clips are development data and cannot be reused as an untouched final claim. The candidate is now frozen before holdout acquisition.

Preferred holdout: a preregistered FSD50K evaluation subset with independently mapped and human-audited instances for siren, vehicle horn, fire/crackle, glass/shatter, infant crying, dog bark, and door knock. FSD50K/Freesound is source-distinct from AudioSet, although uploader/audio duplication must still be hash- and fingerprint-audited. Use at least 40 non-duplicated clips per supported class plus acoustically plausible negatives. UrbanSound8K may provide a partial secondary check for dog bark, car horn, and siren, but cannot validate all safety categories.

Protocol: freeze taxonomy/code hashes first; create mappings from ontology documentation without viewing predictions; deduplicate by file hash and acoustic fingerprint against development/training sources where possible; run exactly once; publish every sample and FP/FN; do not tune afterward. A fresh corpus was not downloaded in this phase to avoid a large unplanned acquisition and accidental iterative holdout use.
