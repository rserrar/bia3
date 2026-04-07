from uuid import uuid4
from pathlib import Path
from typing import Any

from src.shared.settings import load_settings
from .llm_client import generate_candidate_via_openai, normalize_llm_candidate_payload
from .v2_prompt_builder import V2PromptBuilder


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
    return ""


def _build_prompt_from_v2_builder(payload_context: dict[str, Any]) -> str:
    settings = load_settings()
    repo_root = Path(__file__).resolve().parents[3]
    builder = V2PromptBuilder(
        repo_root=repo_root,
        prompt_template_file=settings.prompt_template_file,
        architecture_guide_file=settings.architecture_guide_file,
        experiment_config_file=settings.experiment_config_file,
        num_new_models=settings.llm_num_new_models,
        num_reference_models=settings.llm_num_reference_models,
    )
    context: dict[str, Any] = {
        "run_id": payload_context.get("run_id", "v3_run"),
        "generation": int(payload_context.get("generation", 0) or 0),
        "code_version": payload_context.get("code_version", "v3-colab-worker"),
        "latest_metrics": payload_context.get("latest_metrics", {}),
        "reference_models": payload_context.get("reference_models", []),
        "recent_generated_models": payload_context.get("recent_generated_models", []),
    }
    prompt = builder.build_prompt(context)
    if prompt.strip() == "":
        return (
            "Generate one Keras model definition as JSON for tabular regression. "
            "Return an object with keys model_definition_full and model_definition_summary. "
            "model_definition_full must contain architecture_definition.used_inputs, branches[].layers[], output_heads[]."
        )
    return prompt


def _candidate_from_llm(candidate_id: str, payload_context: dict[str, Any]) -> tuple[dict, dict] | None:
    settings = load_settings()
    if settings.llm_mode != "openai_chat" or settings.llm_api_key.strip() == "":
        return None
    try:
        payload = generate_candidate_via_openai(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            prompt=_build_prompt_from_v2_builder(payload_context),
            endpoint=settings.llm_endpoint,
        )
    except Exception as error:
        print(f"[WARN] LLM generation failed, fallback to baseline: {error}")
        return None
    normalized = normalize_llm_candidate_payload(payload)
    full = normalized.get("model_definition_full") if isinstance(normalized.get("model_definition_full"), dict) else None
    summary = normalized.get("model_definition_summary") if isinstance(normalized.get("model_definition_summary"), dict) else None
    if not full or not summary:
        return None
    full["model_id"] = candidate_id
    return full, summary


def execute_generate_candidate(payload: dict) -> dict:
    target = int(payload.get("target_candidates", 1) or 1)
    candidates = []
    for _ in range(max(1, target)):
        candidate_id = f"cand_{uuid4().hex[:12]}"
        llm_out = _candidate_from_llm(candidate_id, payload)
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
