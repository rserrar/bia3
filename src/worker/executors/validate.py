from typing import Any, cast

from src.shared.settings import load_settings
from .llm_client import normalize_llm_candidate_payload, repair_model_definition_via_openai
from .model_runtime import run_smoke_fit


def execute_validate_candidate(payload: dict) -> dict:
    raw_model_definition = payload.get("model_definition_full")
    model_definition_full: dict[str, Any]
    if isinstance(raw_model_definition, dict):
        model_definition_full = dict(raw_model_definition)
    else:
        model_definition_full = {}
    force_fail = bool(payload.get("force_fail", False))
    compile_ok = False
    smoke_ok = False
    smoke_result = {}
    error_message = None

    if not force_fail:
        try:
            smoke_result = run_smoke_fit(
                model_definition_full=model_definition_full,
                smoke_batches=int(payload.get("smoke_batches", 3) or 3),
                feature_dim=int(payload.get("feature_dim", 16) or 16),
                batch_size=int(payload.get("batch_size", 8) or 8),
            )
            compile_ok = True
            smoke_ok = True
        except Exception as error:
            error_message = str(error)

    repaired_model_definition_full = None
    if (not compile_ok or not smoke_ok) and model_definition_full:
        settings = load_settings()
        if settings.llm_mode == "openai_chat" and settings.llm_api_key.strip() != "":
            try:
                print("[INFO] Validation failed, requesting LLM repair", flush=True)
                repaired_payload = repair_model_definition_via_openai(
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    endpoint=settings.llm_endpoint,
                    model_definition_full=model_definition_full,
                    validation_error=error_message or "compile/smoke validation failed",
                    fix_prompt_file=settings.fix_error_prompt_file,
                )
                normalized = normalize_llm_candidate_payload(repaired_payload)
                repaired_raw = normalized.get("model_definition_full")
                repaired_full = cast(dict[str, Any], repaired_raw) if isinstance(repaired_raw, dict) else None
                if repaired_full:
                    smoke_result = run_smoke_fit(
                        model_definition_full=repaired_full,
                        smoke_batches=int(payload.get("smoke_batches", 3) or 3),
                        feature_dim=int(payload.get("feature_dim", 16) or 16),
                        batch_size=int(payload.get("batch_size", 8) or 8),
                    )
                    compile_ok = True
                    smoke_ok = True
                    error_message = None
                    repaired_model_definition_full = repaired_full
                    print("[INFO] LLM repair succeeded", flush=True)
            except Exception as repair_error:
                print(f"[WARN] LLM repair failed: {repair_error}", flush=True)

    report = {
        "schema_ok": True,
        "compile_ok": compile_ok,
        "smoke_fit_ok": smoke_ok,
        "smoke_batches": int(payload.get("smoke_batches", 3) or 3),
        "smoke_result": smoke_result,
        "error_message": error_message,
        "repaired": repaired_model_definition_full is not None,
    }
    result = {
        "status": "completed",
        "validation_report": report,
    }
    if repaired_model_definition_full is not None:
        result["repaired_model_definition_full"] = repaired_model_definition_full
    return result
