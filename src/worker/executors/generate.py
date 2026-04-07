from uuid import uuid4

from src.shared.settings import load_settings
from .llm_client import generate_candidate_via_openai


def _fallback_definition(candidate_id: str) -> tuple[dict, dict]:
    model_full = {
        "model_id": candidate_id,
        "architecture_definition": {
            "used_inputs": [{"input_layer_name": "input_main", "source_feature_name": "entrada_valors"}],
            "branches": [
                {
                    "branch_id": "b1",
                    "layers": [
                        {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
                        {"type": "Dense", "params": {"units": 32, "activation": "relu"}},
                    ],
                }
            ],
            "output_heads": [
                {"output_layer_name": "output_sortida_valors", "maps_to_target_config_name": "sortida_valors"}
            ],
        },
        "training_config": {"fit": {"epochs": 3, "batch_size": 32}},
    }
    model_summary = {
        "kind": "dense_baseline",
        "layers": 2,
        "params_hint": "small",
        "expected_behavior": "stable_baseline",
    }
    return model_full, model_summary


def _llm_prompt() -> str:
    return (
        "Generate one Keras model definition as JSON for tabular regression. "
        "Return an object with keys model_definition_full and model_definition_summary. "
        "model_definition_full must contain architecture_definition.used_inputs, branches[].layers[], output_heads[]."
    )


def _candidate_from_llm(candidate_id: str) -> tuple[dict, dict] | None:
    settings = load_settings()
    if settings.llm_mode != "openai_chat" or settings.llm_api_key.strip() == "":
        return None
    try:
        payload = generate_candidate_via_openai(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            prompt=_llm_prompt(),
            endpoint=settings.llm_endpoint,
        )
    except Exception as error:
        print(f"[WARN] LLM generation failed, fallback to baseline: {error}")
        return None
    full = payload.get("model_definition_full") if isinstance(payload.get("model_definition_full"), dict) else None
    summary = payload.get("model_definition_summary") if isinstance(payload.get("model_definition_summary"), dict) else None
    if not full or not summary:
        return None
    full["model_id"] = candidate_id
    return full, summary


def execute_generate_candidate(payload: dict) -> dict:
    target = int(payload.get("target_candidates", 1) or 1)
    candidates = []
    for _ in range(max(1, target)):
        candidate_id = f"cand_{uuid4().hex[:12]}"
        llm_out = _candidate_from_llm(candidate_id)
        if llm_out is None:
            model_full, model_summary = _fallback_definition(candidate_id)
        else:
            model_full, model_summary = llm_out
        candidates.append(
            {
                "candidate_id": candidate_id,
                "fingerprint": uuid4().hex,
                "model_definition_full": model_full,
                "model_definition_summary": model_summary,
            }
        )
    return {"status": "completed", "candidates": candidates}
