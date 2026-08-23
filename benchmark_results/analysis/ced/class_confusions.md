# CED class confusions

Full 280-sample clean ESC-50 analysis. Top-k was re-extracted from the frozen model without changing preprocessing.
| Class | N | Recall | F1 | Dominant wrong top-1 labels |
|---|---:|---:|---:|---|
| baby_crying | 40 | 0.650 | 0.788 | [["Crying, sobbing", 11], ["Laughter", 1], ["Child singing", 1], ["Babbling", 1]] |
| dog_barking | 40 | 0.050 | 0.095 | [["Animal", 29], ["Dog", 8], ["Bow-wow", 1]] |
| door_activity | 40 | 0.900 | 0.947 | [["Chopping (food)", 1], ["Machine gun", 1], ["Drum", 1], ["Door", 1]] |
| fire | 40 | 0.500 | 0.667 | [["Vehicle", 4], ["Rain on surface", 3], ["Patter", 2], ["Crackle", 2], ["Animal", 2]] |
| glass_breaking | 40 | 0.000 | 0.000 | [["Breaking", 23], ["Chink, clink", 7], ["Glass", 5], ["Smash, crash", 2], ["Crackle", 1]] |
| siren | 40 | 0.600 | 0.750 | [["Emergency vehicle", 10], ["Police car (siren)", 3], ["Civil defense siren", 1], ["Ambulance (siren)", 1], ["Train", 1]] |
| vehicle_horn | 40 | 0.450 | 0.621 | [["Vehicle", 17], ["Music", 2], ["Toot", 2], ["Speech", 1]] |
