# Manual benchmark checklist

- [ ] Confirm every manifest WAV plays correctly and matches its written ground truth.
- [ ] Confirm no source WAV timestamp, size, or hash changes after a run.
- [ ] Review CED top predictions for plausible near-miss labels.
- [ ] Review FINAL transcripts only; never score PARTIAL text.
- [ ] For noisy speech, confirm both DTLN conditions use the same source hash and run ID.
- [ ] Listen to representative temporary DTLN output during a controlled manual run if needed.
- [ ] Exercise multi-cycle emergency sequences in the documented cycle order.
- [ ] Confirm false alerts and missed emergencies against human judgment.
- [ ] Record microphone/device tests separately; they are outside this fixed-file benchmark.
- [ ] Treat results from a small hand-selected dataset as preliminary, not real-world proof.
