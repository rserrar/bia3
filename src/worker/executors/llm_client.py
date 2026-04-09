import json
import glob
import re
from pathlib import Path
from typing import Any
from urllib import request as urlrequest
from urllib.error import HTTPError

from src.shared.settings import load_settings


class LlmRequestError(RuntimeError):
    def __init__(self, message: str, llm_trace: dict[str, Any]):
        super().__init__(message)
        self.llm_trace = llm_trace


def _truncate(text: Any, max_len: int = 20000) -> str | None:
    if text is None:
        return None
    value = str(text)
    if len(value) <= max_len:
        return value
    return value[:max_len]


def _base_llm_trace(*, model: str, endpoint: str, prompt: str) -> dict[str, Any]:
    return {
        "provider": "openai_chat",
        "model": model,
        "endpoint": endpoint,
        "prompt": _truncate(prompt),
        "response_raw": None,
        "response_parsed": None,
        "parse_ok": False,
        "error_type": None,
        "error_message": None,
    }


def _extract_balanced_payload(text: str, start: int, opening: str, closing: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
            continue
        if char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise RuntimeError("LLM JSON payload not closed")


def _extract_first_json_payload(text: str) -> str:
    object_start = text.find("{")
    array_start = text.find("[")
    starts = [index for index in [object_start, array_start] if index >= 0]
    if not starts:
        raise RuntimeError("LLM response does not contain JSON payload")
    start = min(starts)
    opening = text[start]
    if opening not in "{[":
        raise RuntimeError("LLM response does not contain JSON payload")
    closing = "}" if opening == "{" else "]"
    return _extract_balanced_payload(text, start, opening, closing)


def _extract_json(text: str) -> Any:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    extracted = _extract_first_json_payload(raw)
    parsed = json.loads(extracted)
    if isinstance(parsed, (dict, list)):
        return parsed
    return {}


def normalize_llm_candidate_payload(payload: dict | list) -> dict:
    if isinstance(payload, dict):
        parsed_payload: Any = payload.get("_llm_parsed_payload")
        if isinstance(parsed_payload, (dict, list)):
            payload = parsed_payload
    if isinstance(payload, list):
        first = payload[0] if payload else {}
        payload = first if isinstance(first, dict) else {}
    if not isinstance(payload, dict):
        return {}

    full = payload.get("model_definition_full") if isinstance(payload.get("model_definition_full"), dict) else None
    summary = payload.get("model_definition_summary") if isinstance(payload.get("model_definition_summary"), dict) else None
    if full and summary:
        return {
            "model_definition_full": full,
            "model_definition_summary": summary,
        }

    proposal_raw = payload.get("proposal")
    proposal = proposal_raw if isinstance(proposal_raw, dict) else {}
    model_definition_raw = proposal.get("model_definition")
    model_definition = model_definition_raw if isinstance(model_definition_raw, dict) else None
    if model_definition is None and isinstance(payload.get("architecture_definition"), dict):
        model_definition = payload
    if model_definition is not None:
        return {
            "model_definition_full": model_definition,
            "model_definition_summary": {
                "kind": "legacy_model_definition",
                "source": "normalized_from_v2_style",
            },
        }
    return {}


def _resolve_repo_path(file_path: str) -> Path:
    raw = Path(file_path)
    if raw.is_absolute():
        return raw
    return (Path(__file__).resolve().parents[3] / raw).resolve()


def _read_text_if_exists(file_path: str) -> str:
    try:
        path = _resolve_repo_path(file_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception:
        pass
    return ""


def _read_json_if_exists(file_path: str) -> dict[str, Any]:
    raw = _read_text_if_exists(file_path)
    if raw.strip() == "":
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _inputs_description(experiment: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in experiment.get("input_features_config", []) if isinstance(experiment.get("input_features_config"), list) else []:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature_name", "")).strip() or "unknown"
        cols = int(item.get("total_columns", 0) or 0)
        default_layer = str(item.get("default_input_layer_name", "")).strip()
        source_csv = str(item.get("source_csv_key", "")).strip()
        desc = str(item.get("description", "")).strip()
        extras: list[str] = []
        if default_layer:
            extras.append(f"default_input_layer_name={default_layer}")
        if source_csv:
            extras.append(f"csv={source_csv}")
        rows.append(f"- feature_name={feature} · cols={cols} · {' · '.join(extras)} · {desc}")
    if rows:
        return "\n".join(rows)
    return "No input features config available in experiment_config.json"


def _outputs_description(experiment: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in experiment.get("output_targets_config", []) if isinstance(experiment.get("output_targets_config"), list) else []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_name", "")).strip() or "unknown"
        cols = int(item.get("total_columns", 0) or 0)
        layer = str(item.get("default_output_layer_name", "")).strip()
        source_csv = str(item.get("source_csv_key", "")).strip()
        rows.append(
            f"- target_name={target} · cols={cols} · default_output_layer_name={layer} · csv={source_csv}"
        )
    if rows:
        return "\n".join(rows)
    return "No output targets config available in experiment_config.json"


def _pick_working_model_example_json() -> str:
    repo_root = Path(__file__).resolve().parents[3]
    patterns = [
        repo_root / "models" / "test" / "*.json",
        repo_root / "models" / "base" / "*.json",
    ]
    for pattern in patterns:
        for file_path in sorted(glob.glob(str(pattern))):
            try:
                text = Path(file_path).read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text.startswith("{"):
                return text
    return "{}"


def _split_validation_error(validation_error: str) -> tuple[str, str]:
    text = str(validation_error or "").strip()
    marker = "Traceback:"
    idx = text.find(marker)
    if idx < 0:
        return text if text else "No validation_error provided", "No traceback provided"
    summary = text[:idx].strip()
    traceback_text = text[idx + len(marker) :].strip()
    if summary == "":
        summary = "No validation_error summary provided"
    if traceback_text == "":
        traceback_text = "No traceback provided"
    return summary, traceback_text


def _replace_prompt_placeholders(prompt_template: str, replacements: dict[str, str]) -> str:
    prompt = prompt_template
    for key, value in replacements.items():
        prompt = prompt.replace("{{" + key + "}}", value)

    unresolved = sorted(set(re.findall(r"\{\{\s*([^}]+?)\s*\}\}", prompt)))
    for name in unresolved:
        cleaned = str(name).strip()
        prompt = prompt.replace("{{" + cleaned + "}}", f"[UNAVAILABLE:{cleaned}]")
        prompt = prompt.replace("{{ " + cleaned + " }}", f"[UNAVAILABLE:{cleaned}]")
    return prompt


def generate_candidate_via_openai(api_key: str, model: str, prompt: str, endpoint: str) -> dict[str, Any]:
    llm_trace = _base_llm_trace(model=model, endpoint=endpoint, prompt=prompt)
    print(f"[INFO] LLM request -> endpoint={endpoint} model={model}", flush=True)
    print(f"[LLM] prompt size={len(str(prompt))}", flush=True)
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return ONLY valid JSON with keys model_definition_full and model_definition_summary.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        url=endpoint,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlrequest.urlopen(req, timeout=60) as resp:
            content = resp.read().decode("utf-8")
            print("[INFO] LLM response received", flush=True)
            llm_trace["response_raw"] = _truncate(content)
            print(f"[LLM] response size={len(content)}", flush=True)
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="ignore")
        llm_trace["response_raw"] = _truncate(body)
        llm_trace["error_type"] = "llm_api_error"
        llm_trace["error_message"] = f"OpenAI HTTP {error.code}: {body}"
        print("[LLM] parse_ok=False", flush=True)
        raise LlmRequestError(str(llm_trace["error_message"]), llm_trace) from error

    try:
        api_payload = json.loads(content)
    except Exception as error:
        llm_trace["error_type"] = "llm_api_error"
        llm_trace["error_message"] = f"Invalid API JSON response: {error}"
        print("[LLM] parse_ok=False", flush=True)
        raise LlmRequestError(str(llm_trace["error_message"]), llm_trace) from error

    message = (
        api_payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    llm_trace["response_raw"] = _truncate(message)
    print(f"[LLM] response size={len(str(message))}", flush=True)

    try:
        parsed = _extract_json(str(message))
    except Exception as error:
        llm_trace["parse_ok"] = False
        llm_trace["error_type"] = "llm_parse_error"
        llm_trace["error_message"] = str(error)
        print("[LLM] parse_ok=False", flush=True)
        raise LlmRequestError(f"LLM parse error: {error}", llm_trace) from error

    llm_trace["parse_ok"] = True
    llm_trace["response_parsed"] = parsed if isinstance(parsed, (dict, list)) else None
    print("[LLM] parse_ok=True", flush=True)

    if isinstance(parsed, dict):
        parsed["_llm_raw_response"] = api_payload
        parsed["_llm_response_text"] = _truncate(message)
        parsed["_llm_prompt_text"] = _truncate(prompt)
        parsed["_llm_model"] = model
        parsed["_llm_endpoint"] = endpoint
        parsed["_llm_trace"] = llm_trace
        return parsed
    return {
        "_llm_parsed_payload": parsed,
        "_llm_raw_response": api_payload,
        "_llm_response_text": _truncate(message),
        "_llm_prompt_text": _truncate(prompt),
        "_llm_model": model,
        "_llm_endpoint": endpoint,
        "_llm_trace": llm_trace,
    }


def repair_model_definition_via_openai(
    *,
    api_key: str,
    model: str,
    endpoint: str,
    model_definition_full: dict,
    validation_error: str,
    fix_prompt_file: str,
) -> dict:
    prompt_template = ""
    try:
        path = _resolve_repo_path(fix_prompt_file)
        if path.exists():
            prompt_template = path.read_text(encoding="utf-8")
    except Exception:
        prompt_template = ""

    if prompt_template.strip() == "":
        prompt_template = (
            "Repair this model definition JSON so it compiles in Keras and can run one smoke-fit batch. "
            "Return only JSON with key model_definition_full."
        )

    settings = load_settings()
    experiment = _read_json_if_exists(settings.experiment_config_file)
    architecture_guide = _read_text_if_exists(settings.architecture_guide_file)
    summary_error, traceback_error = _split_validation_error(validation_error)

    replacements = {
        "validation_error": summary_error,
        "error_traceback": traceback_error,
        "buggy_model_json": json.dumps(model_definition_full, ensure_ascii=False, indent=2),
        "working_model_example_json": _pick_working_model_example_json(),
        "available_inputs_description": _inputs_description(experiment),
        "available_outputs_description": _outputs_description(experiment),
        "architecture_guide_content": architecture_guide if architecture_guide.strip() else "No architecture guide content available",
    }
    prompt = _replace_prompt_placeholders(prompt_template, replacements)

    return generate_candidate_via_openai(api_key=api_key, model=model, prompt=prompt, endpoint=endpoint)
