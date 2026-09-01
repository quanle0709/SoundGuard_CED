import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from emergency_system import EmergencySystem
from personalization.priority_adapter import PriorityAdapter
from personalization.profile_generator import (
    OpenAIResponseError,
    _generate_openai,
    _json_schema,
    _normalize_openai_profile,
    generate_profile,
    generate_rule_based,
    resolve_provider_config,
)
from personalization.profile_manager import ProfileManager
from personalization.profile_validator import UNIVERSAL_MIN_PRIORITY, validate_profile
from personalization.schemas import ProfileValidationError


PERSONAS = {
    "driver": "I drive for work almost every day, spend many hours in city traffic, and need to notice emergency vehicles and surrounding traffic quickly.",
    "student_motorcyclist": "I am a high school student and ride a motorbike to school every day. I spend most of my time at school, on the road, and at home.",
    "parent": "I stay home with my infant for much of the day and I need to notice if the baby cries or if something dangerous happens in the house.",
    "mixed": "I am a student, ride a motorbike to school, live with my grandmother, and often take care of my baby brother.",
}

NEGATION_REGRESSION_PERSONA = (
    "I am a high-school student. I ride a motorbike to school daily. "
    "I spend time at school, home, and on the road. "
    "I sometimes care for a younger sibling. "
    "I am not regularly around construction machinery or medical equipment."
)


class FakeHTTPResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body


class PersonalizationTests(unittest.TestCase):
    def _structured_profile(self, *, smoke_priority=1):
        return {
            "profile_version": "1.0",
            "roles": ["student"],
            "contexts": ["school", "home", "road"],
            "responsibilities": ["daily commute"],
            "priority_profile": [
                {"label": "vehicle_horn", "priority": 5},
                {"label": "smoke_alarm", "priority": smoke_priority},
            ],
            "reasoning_summary": [
                {"label": "vehicle_horn", "reason": "Daily road exposure."},
                {"label": "smoke_alarm", "reason": "Safety event."},
            ],
        }

    def test_openai_schema_uses_strict_supported_array_shape(self):
        schema = _json_schema()
        def schema_keys(value):
            if isinstance(value, dict):
                keys = set(value)
                for child in value.values():
                    keys.update(schema_keys(child))
                return keys
            if isinstance(value, list):
                keys = set()
                for child in value:
                    keys.update(schema_keys(child))
                return keys
            return set()

        keys = schema_keys(schema)
        self.assertEqual(schema["properties"]["profile_version"]["enum"], ["1.0"])
        self.assertEqual(schema["properties"]["priority_profile"]["type"], "array")
        self.assertEqual(schema["properties"]["reasoning_summary"]["type"], "array")
        self.assertNotIn("propertyNames", keys)
        self.assertNotIn("minProperties", keys)
        self.assertNotIn("uniqueItems", keys)
        self.assertNotIn("const", keys)
        self.assertEqual(
            schema["properties"]["priority_profile"]["items"]["additionalProperties"],
            False,
        )
        self.assertEqual(
            schema["properties"]["reasoning_summary"]["items"]["additionalProperties"],
            False,
        )

    def test_provider_config_defaults_to_openai_and_resolves_at_runtime(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "first-key"}, clear=True):
            first = resolve_provider_config()
            os.environ["OPENAI_API_KEY"] = "second-key"
            second = resolve_provider_config()
        self.assertEqual(first["provider"], "openai")
        self.assertEqual(first["model"], "gpt-5-mini")
        self.assertEqual(first["api_key"], "first-key")
        self.assertEqual(second["api_key"], "second-key")

    def test_model_override_applies_to_either_provider(self):
        for provider, key_name in (
            ("openai", "OPENAI_API_KEY"),
            ("openrouter", "OPENROUTER_API_KEY"),
        ):
            with self.subTest(provider=provider), patch.dict(
                os.environ,
                {
                    "SOUNDGUARD_AI_PROVIDER": provider,
                    "SOUNDGUARD_AI_MODEL": "custom/model",
                    key_name: "test-key",
                },
                clear=True,
            ):
                self.assertEqual(resolve_provider_config()["model"], "custom/model")

    def test_unsupported_provider_falls_back_without_api_request(self):
        with patch.dict(
            os.environ,
            {"SOUNDGUARD_AI_PROVIDER": "unsupported"},
            clear=True,
        ), patch(
            "personalization.profile_generator.urllib.request.urlopen"
        ) as open_url:
            profile, provider = generate_profile([PERSONAS["driver"]])
        self.assertEqual(provider, "deterministic_fallback:configuration")
        self.assertIn("driver", profile["roles"])
        open_url.assert_not_called()

    def test_openrouter_default_model_and_mocked_structured_request(self):
        raw_profile = self._structured_profile()
        response = {"choices": [{"message": {"content": json.dumps(raw_profile)}}]}
        with patch.dict(
            os.environ,
            {
                "SOUNDGUARD_AI_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
            },
            clear=True,
        ), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            return_value=FakeHTTPResponse(response),
        ) as open_url:
            normalized, provider = generate_profile(["I am a student."])
        request = open_url.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(provider, "openrouter")
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(sent["model"], "openai/gpt-oss-120b:free")
        self.assertEqual(sent["response_format"]["type"], "json_schema")
        self.assertEqual(sent["provider"], {"require_parameters": True})
        self.assertEqual(
            sent["response_format"]["json_schema"]["schema"],
            _json_schema(),
        )
        self.assertEqual(normalized["priority_profile"]["vehicle_horn"], 5)

    def test_openrouter_bad_response_uses_classified_fallback(self):
        with patch.dict(
            os.environ,
            {
                "SOUNDGUARD_AI_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
            },
            clear=True,
        ), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            return_value=FakeHTTPResponse({"choices": []}),
        ):
            profile, provider = generate_profile([PERSONAS["student_motorcyclist"]])
        self.assertEqual(provider, "deterministic_fallback:response")
        self.assertIn("student", profile["roles"])

    def test_openrouter_response_diagnostics_report_shape_without_content(self):
        private_content = "PRIVATE_INTERVIEW_TEXT"
        response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": private_content},
            }],
            "model": "openai/gpt-oss-120b",
        }
        with patch.dict(
            os.environ,
            {
                "SOUNDGUARD_AI_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "test-key",
            },
            clear=True,
        ), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            return_value=FakeHTTPResponse(response),
        ), self.assertLogs("personalization.profile_generator", level="WARNING") as logs:
            _, provider = generate_profile([PERSONAS["driver"]])
        diagnostic = logs.output[0]
        self.assertEqual(provider, "deterministic_fallback:response")
        self.assertIn("structured JSON parsing failed", diagnostic)
        self.assertIn("finish_reason='stop'", diagnostic)
        self.assertIn("content_type=str", diagnostic)
        self.assertIn(f"content_length={len(private_content)}", diagnostic)
        self.assertNotIn(private_content, diagnostic)

    def test_openrouter_auth_failure_uses_classified_fallback(self):
        error = urllib.error.HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"invalid key"}}'),
        )
        with patch.dict(
            os.environ,
            {
                "SOUNDGUARD_AI_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "bad-key",
            },
            clear=True,
        ), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            side_effect=error,
        ), self.assertLogs("personalization.profile_generator", level="WARNING") as logs:
            _, provider = generate_profile([PERSONAS["driver"]])
        self.assertEqual(provider, "deterministic_fallback:authentication")
        self.assertIn("OpenRouter HTTP 401", logs.output[0])

    def test_openrouter_distinguishes_operational_api_failures(self):
        failures = (
            (
                {"error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded"}}},
                "rate_limit",
            ),
            (
                {"error": {"code": 404, "message": "No endpoints found for this model"}},
                "model_unavailable",
            ),
            (
                {"error": {"code": 503, "message": "Upstream unavailable", "metadata": {"error_type": "provider_unavailable"}}},
                "provider_unavailable",
            ),
        )
        for response, reason in failures:
            with self.subTest(reason=reason), patch.dict(
                os.environ,
                {
                    "SOUNDGUARD_AI_PROVIDER": "openrouter",
                    "OPENROUTER_API_KEY": "test-key",
                },
                clear=True,
            ), patch(
                "personalization.profile_generator.urllib.request.urlopen",
                return_value=FakeHTTPResponse(response),
            ):
                _, provider = generate_profile([PERSONAS["driver"]])
            self.assertEqual(provider, f"deterministic_fallback:{reason}")

    def test_valid_structured_outputs_response_is_normalized(self):
        raw_profile = self._structured_profile()
        response = {
            "output": [{
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps(raw_profile)}],
            }],
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            return_value=FakeHTTPResponse(response),
        ) as open_url:
            normalized = _generate_openai(["I am a student."])
        request = open_url.call_args.args[0]
        sent = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(request.method, "POST")
        self.assertEqual(sent["text"]["format"]["type"], "json_schema")
        self.assertEqual(normalized["priority_profile"]["vehicle_horn"], 5)
        self.assertIsInstance(normalized["priority_profile"], dict)
        self.assertIsInstance(normalized["reasoning_summary"], dict)

    def test_schema_normalization_preserves_universal_critical_enforcement(self):
        normalized = _normalize_openai_profile(self._structured_profile(smoke_priority=1))
        self.assertEqual(normalized["priority_profile"]["smoke_alarm"], 5)
        self.assertIn("smoke_alarm", normalized["reasoning_summary"])

    def test_normalization_repairs_reasoning_label_missing_from_priorities(self):
        raw_profile = self._structured_profile()
        raw_profile["reasoning_summary"].append({
            "label": "explosion",
            "reason": "Collisions are critical for a professional driver.",
        })

        normalized = _normalize_openai_profile(raw_profile)

        self.assertEqual(normalized["priority_profile"]["explosion"], 5)
        self.assertEqual(
            normalized["reasoning_summary"]["explosion"],
            "Collisions are critical for a professional driver.",
        )

        inconsistent_internal = dict(normalized)
        inconsistent_internal["priority_profile"] = dict(normalized["priority_profile"])
        del inconsistent_internal["priority_profile"]["explosion"]
        with self.assertRaisesRegex(
            ProfileValidationError,
            "reasoning label is not prioritized: explosion",
        ):
            validate_profile(inconsistent_internal)

    def test_malformed_openai_response_is_rejected(self):
        malformed = self._structured_profile()
        malformed["priority_profile"] = [{"label": "vehicle_horn"}]
        with self.assertRaises(OpenAIResponseError):
            _normalize_openai_profile(malformed)

        malformed = self._structured_profile()
        malformed["reasoning_summary"].append({
            "label": "arbitrary_noise",
            "reason": "Unsupported output must not bypass validation.",
        })
        with self.assertRaises(OpenAIResponseError):
            _normalize_openai_profile(malformed)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            return_value=FakeHTTPResponse({"output": []}),
        ):
            with self.assertRaises(OpenAIResponseError):
                _generate_openai(["I am a student."])

    def test_api_schema_failure_logs_reason_and_uses_deterministic_fallback(self):
        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/responses",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"code":"invalid_json_schema","message":"schema invalid"}}'),
        )
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}), patch(
            "personalization.profile_generator.urllib.request.urlopen",
            side_effect=error,
        ), self.assertLogs("personalization.profile_generator", level="WARNING") as logs:
            profile, provider = generate_profile([PERSONAS["student_motorcyclist"]])
        self.assertTrue(provider.startswith("deterministic_fallback:schema"))
        self.assertIn("schema", logs.output[0])
        self.assertIn("student", profile["roles"])

    def test_driver_persona(self):
        profile = generate_rule_based([PERSONAS["driver"]])
        self.assertIn("driver", profile["roles"])
        self.assertIn("road", profile["contexts"])
        self.assertEqual(profile["priority_profile"]["siren"], 5)
        self.assertEqual(profile["priority_profile"]["vehicle_horn"], 5)
        self.assertEqual(profile["priority_profile"]["explosion"], 5)
        self.assertLessEqual(profile["priority_profile"]["door_activity"], 2)

    def test_student_motorcyclist_persona(self):
        profile = generate_rule_based([PERSONAS["student_motorcyclist"]])
        self.assertTrue({"student", "motorcyclist_cyclist"}.issubset(profile["roles"]))
        self.assertTrue({"road", "school", "home"}.issubset(profile["contexts"]))
        self.assertEqual(profile["priority_profile"]["vehicle_horn"], 5)
        self.assertEqual(profile["priority_profile"]["siren"], 5)

    def test_parent_persona(self):
        profile = generate_rule_based([PERSONAS["parent"]])
        self.assertIn("parent_infant_caregiver", profile["roles"])
        self.assertIn("home", profile["contexts"])
        self.assertEqual(profile["priority_profile"]["baby_crying"], 5)
        self.assertEqual(profile["priority_profile"]["smoke_alarm"], 5)
        self.assertEqual(profile["priority_profile"]["glass_breaking"], 4)
        self.assertLessEqual(profile["priority_profile"]["vehicle_horn"], 2)

    def test_mixed_role_persona_combines_priorities(self):
        profile = generate_rule_based([PERSONAS["mixed"]])
        self.assertTrue({"student", "motorcyclist_cyclist", "parent_infant_caregiver"}.issubset(profile["roles"]))
        self.assertTrue({"road", "school", "home"}.issubset(profile["contexts"]))
        self.assertEqual(profile["priority_profile"]["vehicle_horn"], 5)
        self.assertEqual(profile["priority_profile"]["baby_crying"], 5)

    def test_exact_negation_regression_persona(self):
        profile = generate_rule_based([NEGATION_REGRESSION_PERSONA])
        self.assertEqual(
            set(profile["roles"]),
            {"student", "motorcyclist_cyclist", "parent_infant_caregiver"},
        )
        self.assertEqual(set(profile["contexts"]), {"road", "school", "home"})
        self.assertNotIn("construction_worker", profile["roles"])
        self.assertNotIn("pedestrian_commuter", profile["roles"])
        self.assertNotIn("construction_site", profile["contexts"])
        self.assertNotIn("workplace", profile["contexts"])
        self.assertNotIn("public_transport", profile["contexts"])

    def test_commute_alone_does_not_infer_travel_mode(self):
        profile = generate_rule_based([
            "I commute daily between home and school, but I have not said that I walk or use public transport."
        ])
        self.assertNotIn("pedestrian_commuter", profile["roles"])
        self.assertNotIn("public_transport", profile["contexts"])

    def test_negated_role_and_context_terms_are_not_positive_evidence(self):
        profile = generate_rule_based([
            "I do not drive, do not work in construction, and never use a bus or train."
        ])
        self.assertNotIn("driver", profile["roles"])
        self.assertNotIn("construction_worker", profile["roles"])
        self.assertNotIn("construction_site", profile["contexts"])
        self.assertNotIn("public_transport", profile["contexts"])

    def test_universal_minimum_overrides_ai_output(self):
        profile = generate_rule_based(["I work at home."])
        for label in UNIVERSAL_MIN_PRIORITY:
            profile["priority_profile"][label] = 1
        validated = validate_profile(profile)
        for label, minimum in UNIVERSAL_MIN_PRIORITY.items():
            self.assertGreaterEqual(validated.priority_profile[label], minimum)

    def test_strict_validation(self):
        profile = generate_rule_based(["I am a student."])
        invalid = dict(profile)
        invalid["arbitrary_command"] = "VIBRATE_LEFT"
        with self.assertRaises(ProfileValidationError):
            validate_profile(invalid)
        invalid = json.loads(json.dumps(profile))
        invalid["priority_profile"]["speech"] = 6
        with self.assertRaises(ProfileValidationError):
            validate_profile(invalid)
        invalid = json.loads(json.dumps(profile))
        invalid["priority_profile"]["arbitrary_noise"] = 3
        with self.assertRaises(ProfileValidationError):
            validate_profile(invalid)
        invalid = json.loads(json.dumps(profile))
        invalid["roles"].append("wizard")
        with self.assertRaises(ProfileValidationError):
            validate_profile(invalid)
        invalid = json.loads(json.dumps(profile))
        del invalid["contexts"]
        with self.assertRaises(ProfileValidationError):
            validate_profile(invalid)
        invalid = json.loads(json.dumps(profile))
        invalid["profile_version"] = "2.0"
        with self.assertRaises(ProfileValidationError):
            validate_profile(invalid)

    def test_conflicting_aliases_are_rejected(self):
        profile = generate_rule_based(["I drive."])
        profile["priority_profile"]["car horn"] = 1
        with self.assertRaises(ProfileValidationError):
            validate_profile(profile)
        profile = generate_rule_based(["I drive."])
        profile["priority_profile"]["car horn"] = profile["priority_profile"]["vehicle_horn"]
        with self.assertRaises(ProfileValidationError):
            validate_profile(profile)

    def test_missing_or_malformed_profile_uses_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "default.json"
            default.write_text(
                Path("personalization/default_profile.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            manager = ProfileManager(root / "missing.json", default)
            _, source = manager.load()
            self.assertEqual(source, "default")
            manager.profile_path.write_text("not json", encoding="utf-8")
            _, source = manager.load()
            self.assertEqual(source, "default")
            default.write_text("broken too", encoding="utf-8")
            _, source = manager.load()
            self.assertEqual(source, "built_in_default")

    def test_core_is_unchanged_without_provider_and_personalized_with_one(self):
        baseline = EmergencySystem(decision_mode="single_shot").process_sound_event("Car horn", 0.99)
        self.assertEqual(baseline["alert_level"], "MEDIUM")
        personalized = EmergencySystem(
            decision_mode="single_shot",
            priority_provider=lambda label, context: 5 if label == "vehicle_horn" else None,
        ).process_sound_event("Car horn", 0.99)
        self.assertEqual(personalized["alert_level"], "CRITICAL")

    def test_adapter_hot_reloads_applied_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ProfileManager(root / "user.json", Path("personalization/default_profile.json"))
            adapter = PriorityAdapter(manager)
            self.assertEqual(adapter.get_priority("car horn"), 3)
            manager.save(generate_rule_based([PERSONAS["driver"]]))
            self.assertEqual(adapter.get_priority("Car horn"), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
