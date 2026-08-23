"""Auditable baseline domain knowledge; roles are merged, never selected as presets."""

ROLE_KNOWLEDGE: dict[str, dict] = {
    "driver": {"contexts": ["road"], "priorities": {"siren": 5, "vehicle_horn": 5, "explosion": 5, "screaming": 4, "speech": 3, "dog_barking": 1, "door_activity": 1}},
    "motorcyclist_cyclist": {"contexts": ["road"], "priorities": {"vehicle_horn": 5, "siren": 5, "explosion": 5, "screaming": 4, "speech": 3, "door_activity": 1}},
    "pedestrian_commuter": {"contexts": ["road", "public_transport"], "priorities": {"vehicle_horn": 5, "siren": 5, "screaming": 4, "speech": 3, "door_activity": 1}},
    "construction_worker": {"contexts": ["construction_site", "workplace"], "priorities": {"siren": 5, "explosion": 5, "fire": 5, "screaming": 4, "vehicle_horn": 4, "speech": 3}},
    "factory_warehouse_worker": {"contexts": ["factory_warehouse", "workplace"], "priorities": {"siren": 5, "explosion": 5, "fire": 5, "screaming": 4, "vehicle_horn": 4, "speech": 3}},
    "parent_infant_caregiver": {"contexts": ["home"], "priorities": {"baby_crying": 5, "smoke_alarm": 5, "fire": 5, "glass_breaking": 4, "door_activity": 4, "screaming": 4, "speech": 3, "vehicle_horn": 1}},
    "pregnant_expectant_mother": {"contexts": ["home", "outdoors"], "priorities": {"smoke_alarm": 5, "fire": 5, "explosion": 5, "vehicle_horn": 5, "screaming": 4, "glass_breaking": 4, "siren": 4, "speech": 3, "door_activity": 3}},
    "older_adult_independent": {"contexts": ["home"], "priorities": {"smoke_alarm": 5, "fire": 5, "screaming": 5, "glass_breaking": 5, "door_activity": 4, "siren": 4, "speech": 3}},
    "student": {"contexts": ["school", "home"], "priorities": {"fire": 5, "siren": 5, "vehicle_horn": 5, "screaming": 5, "speech": 3}},
    "healthcare_care_worker": {"contexts": ["healthcare", "workplace"], "priorities": {"siren": 5, "screaming": 5, "glass_breaking": 4, "speech": 3, "door_activity": 3}},
}

DEFAULT_PRIORITIES = {
    "gunshot": 5, "explosion": 5, "fire": 5, "smoke_alarm": 5,
    "glass_breaking": 4, "screaming": 4, "siren": 5,
    "vehicle_horn": 3, "baby_crying": 3, "door_activity": 2,
    "dog_barking": 2, "speech": 2,
}
