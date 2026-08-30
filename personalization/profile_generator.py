"""LLM-backed structured generation with a deterministic offline fallback."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

from .role_knowledge import DEFAULT_PRIORITIES, ROLE_KNOWLEDGE
from .schemas import ALLOWED_FIELDS, PROFILE_VERSION, VALID_CONTEXTS, VALID_ROLES
from .sound_labels import SUPPORTED_SOUND_LABELS
from .profile_validator import validate_profile


LOGGER = logging.getLogger(__name__)
SAFE_COMPOUND_ENV = "SOUNDGUARD_SAFE_COMPOUND_EVIDENCE"

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "openrouter": "openai/gpt-oss-120b:free",
}


class OpenAIIntegrationError(RuntimeError):
    """Classified failure from configuration, transport, API, or response parsing."""

    def __init__(self, message: str, *, kind: str, status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


class OpenAIConfigurationError(OpenAIIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, kind="configuration")


class OpenAINetworkError(OpenAIIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, kind="network")


class OpenAIAPIError(OpenAIIntegrationError):
    def __init__(self, message: str, *, kind: str = "api", status: int | None = None):
        super().__init__(message, kind=kind, status=status)


class OpenAIResponseError(OpenAIIntegrationError):
    def __init__(self, message: str):
        super().__init__(message, kind="response")


ROLE_KEYWORDS = {
    "driver": ("drive", "driver", "driving", "taxi"),
    "motorcyclist_cyclist": ("motorbike", "motorcycle", "bike", "bicycle", "cyclist"),
    # Generic commuting/travel is not evidence of a travel mode. A pedestrian
    # role requires an explicit walking/on-foot statement.
    "pedestrian_commuter": ("walk to", "walking to", "travel on foot", "on foot", "pedestrian"),
    "construction_worker": ("construction", "building site"),
    "factory_warehouse_worker": ("factory", "warehouse", "forklift"),
    "parent_infant_caregiver": ("baby", "infant", "younger sibling", "baby brother", "baby sister", "caregiver", "parent"),
    "pregnant_expectant_mother": ("pregnant", "expectant"),
    "older_adult_independent": ("older adult", "senior", "live independently"),
    "student": ("student", "school", "university", "college"),
    "healthcare_care_worker": ("nurse", "doctor", "healthcare", "hospital", "care worker"),
}
CONTEXT_KEYWORDS = {
    "road": ("road", "traffic", "drive", "motorbike", "motorcycle", "bike"),
    "public_transport": ("bus", "train", "metro", "public transport"),
    "school": ("school", "university", "college", "student"),
    "home": ("home", "house", "live with"),
    "construction_site": ("construction", "building site"),
    "factory_warehouse": ("factory", "warehouse", "forklift"),
    "healthcare": ("hospital", "clinic", "healthcare"),
    "outdoors": ("outdoors", "outside"),
    "workplace": ("workplace", "office", "at work"),
}


NEGATION_WORDS = frozenset({"no", "not", "never", "without", "neither"})
NEGATED_CONTRACTION_TAILS = frozenset({"don", "doesn", "didn", "isn", "aren", "wasn", "weren"})
NEGATION_SCOPE_BREAKERS = frozenset({"but", "however", "although", "yet"})


def _normalize_evidence_text(text: str) -> str:
    value = str(text).lower().replace("’", "'")
    if os.environ.get(SAFE_COMPOUND_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        # Preserve intra-token compound information before punctuation cleanup.
        # This blocks factory-method/baby-blue without an exception list.
        value = re.sub(r"(?<=[a-z0-9])[-‐‑–—](?=[a-z0-9])", "compoundjoiner", value)
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _is_negated(text: str, match_start: int) -> bool:
    """Detect a nearby negation that scopes over a matched role/context phrase."""
    prefix_tokens = re.findall(r"[a-z0-9']+", text[max(0, match_start - 90):match_start])
    window = prefix_tokens[-10:]
    breaker_positions = [
        index for index, token in enumerate(window)
        if token in NEGATION_SCOPE_BREAKERS
    ]
    if breaker_positions:
        window = window[breaker_positions[-1] + 1:]
    if not window:
        return False
    for index, token in enumerate(window):
        if token == "not" and index + 1 < len(window) and window[index + 1] == "only":
            continue
        if token in NEGATION_WORDS:
            return True
        if token.endswith("n't") or token in NEGATED_CONTRACTION_TAILS:
            return True
    return False


def _has_positive_evidence(text: str, phrases: tuple[str, ...]) -> bool:
    """Require a whole phrase occurrence that is not inside local negation scope."""
    for phrase in phrases:
        normalized_phrase = _normalize_evidence_text(phrase)
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            if not _is_negated(text, match.start()):
                return True
    return False


def generate_rule_based(answers: list[str]) -> dict:
    text = _normalize_evidence_text(" ".join(answers))
    roles = [
        role for role, keywords in ROLE_KEYWORDS.items()
        if _has_positive_evidence(text, keywords)
    ]
    contexts = [
        name for name, keywords in CONTEXT_KEYWORDS.items()
        if _has_positive_evidence(text, keywords)
    ]
    priorities = dict(DEFAULT_PRIORITIES)
    reasons = {label: "SoundGuard safe default." for label in priorities}
    role_priorities: dict[str, list[tuple[int, str]]] = {}
    for role in roles:
        knowledge = ROLE_KNOWLEDGE[role]
        for label, value in knowledge["priorities"].items():
            role_priorities.setdefault(label, []).append((value, role))
    for label, candidates in role_priorities.items():
        value, role = max(candidates)
        priorities[label] = value
        reasons[label] = f"Set by the combined {role.replace('_', ' ')} role and context."
    responsibilities = []
    if _has_positive_evidence(text, ("baby", "infant", "younger sibling", "baby brother", "baby sister")):
        responsibilities.append("care for an infant or younger child")
    if _has_positive_evidence(text, ("grandmother", "grandfather", "older adult", "senior")):
        responsibilities.append("share a household with an older adult")
    if _has_positive_evidence(text, ("commute", "school", "drive for work", "motorbike")):
        responsibilities.append("regular travel or commute")
    payload = {
        "profile_version": PROFILE_VERSION,
        "roles": roles,
        "contexts": contexts,
        "responsibilities": responsibilities,
        "priority_profile": priorities,
        "reasoning_summary": reasons,
    }
    return validate_profile(payload).to_dict()


def _json_schema() -> dict:
    priority_item = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": sorted(SUPPORTED_SOUND_LABELS)},
            "priority": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        "required": ["label", "priority"],
        "additionalProperties": False,
    }
    reasoning_item = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": sorted(SUPPORTED_SOUND_LABELS)},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    }
    properties = {
        "profile_version": {"type": "string", "enum": [PROFILE_VERSION]},
        "roles": {"type": "array", "items": {"type": "string", "enum": sorted(VALID_ROLES)}},
        "contexts": {"type": "array", "items": {"type": "string", "enum": sorted(VALID_CONTEXTS)}},
        "responsibilities": {"type": "array", "items": {"type": "string"}},
        "priority_profile": {"type": "array", "items": priority_item, "minItems": 1},
        "reasoning_summary": {"type": "array", "items": reasoning_item},
    }
    return {"type": "object", "properties": properties, "required": sorted(ALLOWED_FIELDS), "additionalProperties": False}


def _normalize_openai_profile(raw_profile: object) -> dict:
    """Convert the strict API arrays back to SoundGuard's internal dictionaries."""
    if not isinstance(raw_profile, dict):
        raise OpenAIResponseError("Structured Outputs profile must be an object")

    priorities = raw_profile.get("priority_profile")
    if not isinstance(priorities, list):
        raise OpenAIResponseError("priority_profile must be an array")
    normalized_priorities: dict[str, int] = {}
    for item in priorities:
        if not isinstance(item, dict) or set(item) != {"label", "priority"}:
            raise OpenAIResponseError("each priority_profile item must contain only label and priority")
        label = item["label"]
        priority = item["priority"]
        if not isinstance(label, str) or isinstance(priority, bool) or not isinstance(priority, int):
            raise OpenAIResponseError("priority_profile item has invalid label or priority types")
        if label in normalized_priorities:
            raise OpenAIResponseError(f"duplicate priority label from OpenAI: {label}")
        normalized_priorities[label] = priority

    reasoning = raw_profile.get("reasoning_summary")
    if not isinstance(reasoning, list):
        raise OpenAIResponseError("reasoning_summary must be an array")
    normalized_reasoning: dict[str, str] = {}
    for item in reasoning:
        if not isinstance(item, dict) or set(item) != {"label", "reason"}:
            raise OpenAIResponseError("each reasoning_summary item must contain only label and reason")
        label = item["label"]
        reason = item["reason"]
        if not isinstance(label, str) or not isinstance(reason, str):
            raise OpenAIResponseError("reasoning_summary item has invalid label or reason types")
        if label in normalized_reasoning:
            raise OpenAIResponseError(f"duplicate reasoning label from OpenAI: {label}")
        normalized_reasoning[label] = reason

    # JSON Schema cannot express a cross-array subset constraint. Preserve the
    # model's explanation while assigning SoundGuard's deterministic baseline
    # to any explained label whose priority item was omitted. The strict
    # validator still rejects inconsistent profiles presented through any
    # other path.
    for label in sorted(normalized_reasoning.keys() - normalized_priorities.keys()):
        if label not in DEFAULT_PRIORITIES:
            raise OpenAIResponseError(f"unsupported reasoning label from OpenAI: {label}")
        normalized_priorities[label] = DEFAULT_PRIORITIES[label]

    internal_profile = dict(raw_profile)
    internal_profile["priority_profile"] = normalized_priorities
    internal_profile["reasoning_summary"] = normalized_reasoning
    try:
        # This performs the existing role/context/label checks and Universal
        # Critical minimum enforcement after API normalization.
        return validate_profile(internal_profile).to_dict()
    except ValueError as exc:
        raise OpenAIResponseError(f"normalized OpenAI profile failed validation: {exc}") from exc


def _extract_response_text(body: object) -> str:
    if not isinstance(body, dict):
        raise OpenAIResponseError("OpenAI response body must be an object")
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    output = body.get("output")
    if not isinstance(output, list):
        raise OpenAIResponseError("OpenAI response did not contain output text")
    for item in output:
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []) if isinstance(item.get("content"), list) else []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
            if content.get("type") == "refusal":
                raise OpenAIResponseError("OpenAI refused to generate a profile")
    raise OpenAIResponseError("OpenAI response did not contain output text")


def _classify_api_error(
    *, status: int | None, error_type: str, code: str, detail: str
) -> str:
    evidence = f"{error_type} {code} {detail}".lower()
    if status in {401, 403} or any(
        marker in evidence for marker in ("authentication", "invalid_api_key", "invalid api key")
    ):
        return "authentication"
    if status == 429 or "rate_limit" in evidence or "rate limit" in evidence:
        return "rate_limit"
    if any(
        marker in evidence
        for marker in (
            "model_not_found",
            "model unavailable",
            "model is unavailable",
            "no endpoints found",
        )
    ):
        return "model_unavailable"
    if any(
        marker in evidence
        for marker in ("provider_unavailable", "provider_overloaded", "provider unavailable")
    ):
        return "provider_unavailable"
    if "schema" in evidence or "json_schema" in evidence:
        return "schema"
    return "api"


def _parsed_api_error(body: object) -> tuple[str, str, str] | None:
    if not isinstance(body, dict) or not body.get("error"):
        return None
    error = body["error"]
    if isinstance(error, dict):
        metadata = error.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        detail = str(error.get("message") or error.get("code") or "API request failed")[:500]
        code = str(error.get("code") or "")
        error_type = str(
            body.get("error_type")
            or metadata.get("error_type")
            or error.get("type")
            or ""
        )
        return detail, code, error_type
    return str(error)[:500], "", ""


def _http_error(exc: urllib.error.HTTPError, provider_name: str = "OpenAI") -> OpenAIAPIError:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except OSError:
        raw = ""
    detail = raw[:500]
    code = ""
    error_type = ""
    try:
        parsed = json.loads(raw)
        parsed_error = _parsed_api_error(parsed)
        if parsed_error:
            detail, code, error_type = parsed_error
    except json.JSONDecodeError:
        pass
    kind = _classify_api_error(
        status=exc.code,
        error_type=error_type,
        code=code,
        detail=detail,
    )
    return OpenAIAPIError(
        f"{provider_name} HTTP {exc.code} ({kind}): {detail or exc.reason}",
        kind=kind,
        status=exc.code,
    )


def resolve_provider_config() -> dict[str, str]:
    """Resolve provider settings from the current process environment."""
    provider = os.environ.get("SOUNDGUARD_AI_PROVIDER", "openai").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise OpenAIConfigurationError(
            f"unsupported SOUNDGUARD_AI_PROVIDER: {provider or '<empty>'}"
        )

    key_name = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        raise OpenAIConfigurationError(f"{key_name} is not configured")
    return {
        "provider": provider,
        "api_key": api_key,
        "model": os.environ.get("SOUNDGUARD_AI_MODEL", "").strip() or DEFAULT_MODELS[provider],
        "url": OPENAI_RESPONSES_URL if provider == "openai" else OPENROUTER_CHAT_URL,
    }


def _instructions() -> str:
    return (
        "Create one SoundGuard alert profile by combining all applicable roles, contexts, and responsibilities. "
        "Treat negated exposure as negative evidence. Require explicit positive evidence for every role and context. "
        "Commuting or travel alone must never imply pedestrian_commuter or public_transport; require an explicit "
        "walking/on-foot or bus/train/metro statement respectively. Never issue hardware commands and only use "
        "schema labels. Every reasoning_summary label must have a matching priority_profile item. "
        "Here is the V1 baseline role knowledge; "
        "merge every applicable role by taking the most important relevant priority, then explain concise results: "
        + json.dumps(ROLE_KNOWLEDGE, separators=(",", ":"))
    )


def _read_api_response(request: urllib.request.Request) -> object:
    provider_name = "OpenRouter" if "openrouter.ai" in request.full_url else "OpenAI"
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            raw_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise _http_error(exc, provider_name) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OpenAINetworkError(f"AI provider network failure: {exc}") from exc
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise OpenAIResponseError(
            f"HTTP response decoding failed: invalid JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    parsed_error = _parsed_api_error(body)
    if parsed_error:
        detail, code, error_type = parsed_error
        status = None
        if isinstance(body, dict) and isinstance(body.get("status"), int):
            status = body["status"]
        kind = _classify_api_error(
            status=status,
            error_type=error_type,
            code=code,
            detail=detail,
        )
        raise OpenAIAPIError(
            f"{provider_name} API error ({kind}): {detail}",
            kind=kind,
            status=status,
        )
    return body


def _openrouter_response_shape(body: object) -> str:
    """Describe response structure without logging generated or user content."""
    if not isinstance(body, dict):
        return f"body_type={type(body).__name__}"
    parts = [f"top_level_keys={sorted(str(key) for key in body)}"]
    choices = body.get("choices")
    if not isinstance(choices, list):
        parts.append(f"choices_type={type(choices).__name__}")
        return ", ".join(parts)
    parts.append(f"choices_count={len(choices)}")
    if not choices or not isinstance(choices[0], dict):
        return ", ".join(parts)
    choice = choices[0]
    parts.append(f"choice_keys={sorted(str(key) for key in choice)}")
    parts.append(f"finish_reason={choice.get('finish_reason')!r}")
    message = choice.get("message")
    if not isinstance(message, dict):
        parts.append(f"message_type={type(message).__name__}")
        return ", ".join(parts)
    parts.append(f"message_keys={sorted(str(key) for key in message)}")
    content = message.get("content")
    parts.append(f"content_type={type(content).__name__}")
    if isinstance(content, str):
        stripped = content.lstrip()
        parts.append(f"content_length={len(content)}")
        parts.append(f"markdown_fence={stripped.startswith('```')}")
    elif isinstance(content, list):
        parts.append(f"content_parts={len(content)}")
        parts.append(
            "content_part_types="
            + repr([
                item.get("type") if isinstance(item, dict) else type(item).__name__
                for item in content
            ])
        )
    if "parsed" in message:
        parts.append(f"parsed_type={type(message['parsed']).__name__}")
    return ", ".join(parts)


def _generate_openai(answers: list[str], config: dict[str, str] | None = None) -> dict:
    config = config or resolve_provider_config()
    payload = {
        "model": config["model"],
        "instructions": _instructions(),
        "input": "Personalization interview answers:\n" + "\n".join(f"- {answer}" for answer in answers),
        "text": {"format": {"type": "json_schema", "name": "soundguard_profile", "strict": True, "schema": _json_schema()}},
    }
    request = urllib.request.Request(
        config["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {config['api_key']}", "Content-Type": "application/json"},
        method="POST",
    )
    body = _read_api_response(request)
    output_text = _extract_response_text(body)
    try:
        raw_profile = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise OpenAIResponseError("OpenAI output text was not valid JSON") from exc
    return _normalize_openai_profile(raw_profile)


def _generate_openrouter(answers: list[str], config: dict[str, str]) -> dict:
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": _instructions()},
            {
                "role": "user",
                "content": "Personalization interview answers:\n"
                + "\n".join(f"- {answer}" for answer in answers),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "soundguard_profile",
                "strict": True,
                "schema": _json_schema(),
            },
        },
        "provider": {"require_parameters": True},
    }
    request = urllib.request.Request(
        config["url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://127.0.0.1:8765",
            "X-Title": "SoundGuard Personalization",
        },
        method="POST",
    )
    body = _read_api_response(request)
    response_shape = _openrouter_response_shape(body)
    try:
        output_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAIResponseError(
            f"OpenRouter choices extraction failed; {response_shape}"
        ) from exc
    if not isinstance(output_text, str) or not output_text.strip():
        raise OpenAIResponseError(
            f"OpenRouter message.content extraction failed; {response_shape}"
        )
    try:
        raw_profile = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise OpenAIResponseError(
            "OpenRouter structured JSON parsing failed "
            f"at line {exc.lineno}, column {exc.colno}; {response_shape}"
        ) from exc
    return _normalize_openai_profile(raw_profile)


def generate_profile(answers: list[str]) -> tuple[dict, str]:
    """Return a validated profile and the provider used; never fail closed."""
    try:
        config = resolve_provider_config()
        provider = config["provider"]
        if provider == "openrouter":
            return _generate_openrouter(answers, config), provider
        return _generate_openai(answers, config), provider
    except OpenAIIntegrationError as exc:
        LOGGER.warning(
            "AI personalization failed (%s): %s; using deterministic fallback",
            exc.kind,
            exc,
        )
        return generate_rule_based(answers), f"deterministic_fallback:{exc.kind}"
