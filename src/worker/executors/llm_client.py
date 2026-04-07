import json
from urllib import request as urlrequest
from urllib.error import HTTPError


def _extract_json(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    parsed = json.loads(raw)
    return parsed if isinstance(parsed, dict) else {}


def generate_candidate_via_openai(api_key: str, model: str, prompt: str, endpoint: str) -> dict:
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
