import json
from pathlib import Path
from urllib import request as urlrequest
from urllib.error import HTTPError


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


def _extract_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    extracted = _extract_first_json_payload(raw)
    parsed = json.loads(extracted)
    return parsed if isinstance(parsed, dict) else {}


def normalize_llm_candidate_payload(payload: dict) -> dict:
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


def generate_candidate_via_openai(api_key: str, model: str, prompt: str, endpoint: str) -> dict:
    print(f"[INFO] LLM request -> endpoint={endpoint} model={model}", flush=True)
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
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI HTTP {error.code}: {body}") from error
    payload = json.loads(content)
    message = (
        payload.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return _extract_json(str(message))


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
        path = Path(fix_prompt_file)
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[3] / path).resolve()
        if path.exists():
            prompt_template = path.read_text(encoding="utf-8")
    except Exception:
        prompt_template = ""

    if prompt_template.strip() == "":
        prompt_template = (
            "Repair this model definition JSON so it compiles in Keras and can run one smoke-fit batch. "
            "Return only JSON with key model_definition_full."
        )
    prompt = (
        prompt_template
        .replace("{{validation_error}}", validation_error)
        .replace("{{buggy_model_json}}", json.dumps(model_definition_full, ensure_ascii=False, indent=2))
    )
    return generate_candidate_via_openai(api_key=api_key, model=model, prompt=prompt, endpoint=endpoint)
