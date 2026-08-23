# Glass Breaking root cause

All 40 clean ESC-50 samples were inspected. Strict correct: 0. Conservative ontology-aware top-1 correct: 5. Dominant wrong labels: [('Breaking', 23), ('Chink, clink', 7), ('Glass', 5), ('Smash, crash', 2), ('Crackle', 1), ('Slap, smack', 1), ('Hammer', 1)]. Mean wrong top-1 confidence: 0.6221.

The CED ontology contains: ['Glass', 'Shatter']. SoundGuard's target is coarser than the model ontology. The current canonicalizer recognizes some but not all of these semantically related labels, so exact/canonical scoring drops valid parent or sibling concepts. Remaining failures are genuine model/domain errors under this diagnostic; no evidence proves waveform peak normalization, input shape, or the model's internal mean pooling is the dominant cause. Every clip's top-5, duration, sample rate, amplitude, RMS, and input shape is in the companion CSV. No production mapping or preprocessing was changed.
