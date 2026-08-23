# Dog Barking root cause

All 40 clean ESC-50 samples were inspected. Strict correct: 2. Conservative ontology-aware top-1 correct: 11. Dominant wrong labels: [('Animal', 29), ('Dog', 8), ('Bow-wow', 1)]. Mean wrong top-1 confidence: 0.6835.

The CED ontology contains: ['Bark', 'Bow-wow', 'Dog']. SoundGuard's target is coarser than the model ontology. The current canonicalizer recognizes some but not all of these semantically related labels, so exact/canonical scoring drops valid parent or sibling concepts. Remaining failures are genuine model/domain errors under this diagnostic; no evidence proves waveform peak normalization, input shape, or the model's internal mean pooling is the dominant cause. Every clip's top-5, duration, sample rate, amplitude, RMS, and input shape is in the companion CSV. No production mapping or preprocessing was changed.
