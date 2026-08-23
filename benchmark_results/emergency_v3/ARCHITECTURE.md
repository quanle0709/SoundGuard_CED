# Emergency V3 architecture

                         RAW AUDIO
                            |
               +------------+------------+
               |                         |
               v                         v
          CED-Tiny V2              EfficientSED fmn10
       normal CED classification    approved safety labels only
               |                         |
               | displayed unchanged     | thresholded evidence
               +------------+------------+
                            |
                   V2 OR specialist policy
                            |
                 existing emergency layer
                            |
                  existing HUD/alert output

Emergency V3 is disabled by default. When enabled, one isolated worker is started lazily on the first CED audio window and reused. The normal CED label/confidence remains the displayed and fusion input. If V2 already has dangerous evidence it wins; otherwise a validated EfficientSED detection contributes one categorical evidence item to the existing emergency state machine. This preserves one emergency update per audio window and therefore preserves existing temporal confirmation/cooldown behavior.

Worker failure, missing dependencies/checkpoint, malformed output, or inference failure returns no specialist evidence and leaves the V2 path active. STT, DTLN, personalization, fusion logic, HUD transport, firmware, direction, and vibration are not inputs to or dependencies of the specialist.
