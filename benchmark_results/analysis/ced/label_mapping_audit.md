# CED label mapping audit

Model: `mispeech/ced-tiny`; cached ontology has 527 labels. Diagnostic equivalences do not change baseline scoring.

| Source class | Expected SoundGuard label | Exact model ontology label(s) | Mapping valid? | Mapping confidence | Notes |
|---|---|---|---|---|---|
| crying_baby | baby_crying | Baby cry; Crying | SEMANTICALLY VALID | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |
| dog | dog_barking | Bark; Bow-wow; Dog | SEMANTICALLY VALID | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |
| door_wood_knock | door_activity | Knock | SEMANTICALLY VALID | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |
| crackling_fire | fire | Crackle; Fire | SEMANTICALLY VALID | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |
| glass_breaking | glass_breaking | Glass; Shatter | SEMANTICALLY VALID | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |
| siren | siren | Ambulance (siren); Civil defense siren; Police car (siren); Siren | EXACT | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |
| car_horn | vehicle_horn | Air horn; Honk; Toot; Vehicle horn | EXACT | high | SoundGuard uses a coarser canonical category than the 527-label AudioSet ontology. |

Ontology source SHA-256: `3a060c57c28b8138cf66bbab2c0dcab79f07c871b600f52a06f2652bf3cad2bd`. `Glass breaking` and `dog barking` are not literal model labels; they map to AudioSet children/parents such as Glass/Shatter and Dog/Bark.
