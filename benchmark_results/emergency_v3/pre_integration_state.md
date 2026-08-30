# Emergency V3 pre-integration state

- Git commit: `24346d879634f65eeb8a647bdba0ac84e249c63d`
- Source branch: `improve-ced-safety-v2`
- Worktree: dirty; no user work discarded
- V2 source snapshot: present, SHA-256 verified, ZIP integrity passed
- Golden baseline manifest: present
- Improved V2 report: present
- Golden legacy regression: **PASS** (included in complete suite)
- Complete V2 suite: **136 passed, 0 failed, 5 subtests passed**
- Warnings: 4 dependency/deprecation warnings

## Dirty status at freeze

```text
 M .gitignore
 M README.md
 M app.py
 M audio_pipeline.py
 M emergency_system.py
 M requirements.txt
 M sound_classifier.py
 M test_audio_pipeline.py
?? benchmark/
?? benchmark_data/
?? benchmark_results/
?? display_transport.py
?? firmware/
?? personalization/
?? requirements-dev.txt
?? sound_taxonomy.py
?? test_display_transport.py
?? test_personalization.py
```

## Production hashes

```json
{
  "app.py": "7b6fd7f74b98cc44189b6c2899db5bcdeb8eee981f64a47115160090913f16ff",
  "audio_pipeline.py": "357f69e2aa1c3c2197142d853b9aa92aef733d79cfdc20b4e0ffabdf4788d91a",
  "display_transport.py": "b46827107b6041628b98705ac862cc2d2ab4bc05770e0b5e502eb396ed0a8eaa",
  "emergency_system.py": "d2813fca63304f4d53f23eee14928836ef974fe70570541090702000de363f79",
  "fusion_engine.py": "767fd6af263869b283e50f5914d42eea753ec3f1dcbefbf1705cecf42cf31701",
  "sound_classifier.py": "30f5beb37201d5108f45ad605917010a00d9670441861734bdb8e219fbcf87ad",
  "sound_taxonomy.py": "d16ba3c8d0050f14d04852524ea87e051705bf33bf1ebdaf187059ea2a137958",
  "speech_enhancer.py": "f1ec7fba4ee6284a8ea924456d18a0fc8741416e47feaa2091b73b8810f58fb0",
  "speech_recognizer.py": "486a89a950169413e9b3443ba28f85e02703396dbdb5a074ebae154594f79768",
  "personalization/profile_generator.py": "589c7014de4d655ac7a57e9d1295c9684c8409b21d84a6023bb5d689373118ee",
  "personalization/profile_validator.py": "b2594009b8e7aedf19c8ed0ed96d39f11b3c10ba4f569844dcaaea121dc5e3ad",
  "personalization/profile_manager.py": "3e78789b2c92360186f6005012369af4266ada3589f0d53aa7f92b711609124e",
  "personalization/sound_labels.py": "614afc2c3ff7ea19ca94d3b2755af2f30dfe7cea89e9b68c5b053a112760f4cf",
  "personalization/role_knowledge.py": "6c5df39a0f3188c74631552dd0280424af5808bd1b76e14be3c9d6250987bbb9",
  "personalization/schemas.py": "c7712f4808a1f73514fd99b2b32ef7245db9fd53d13dab0a92b4d1c54bc1c396",
  "personalization/web_server.py": "249e1f4a4a82cb3350a18d899663f7477cdb3292bc865513163858d4446f5d78"
}
```

Recovery remains the verified `benchmark_results/improvement/baseline_source_snapshot.zip` plus the V2 branch. Emergency V3 work starts from this exact dirty state and must be default-off if integrated.
