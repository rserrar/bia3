from uuid import uuid4
from pathlib import Path
from typing import Any

from src.shared.settings import load_settings
from .llm_client import generate_candidate_via_openai, normalize_llm_candidate_payload
from .v2_prompt_builder import V2PromptBuilder


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalize_shape(total_columns: Any, default: int) -> list[int]:
    return [max(1, _as_int(total_columns, default))]


def _extract_available_inputs(payload_context: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for item in _as_list_of_dicts(payload_context.get("available_inputs")):
        input_name = str(item.get("input_layer_name", item.get("feature_name", ""))).strip()
        if input_name == "":
            continue
        source_name = str(item.get("source_feature_name", item.get("feature_name", input_name))).strip() or input_name
        shape = item.get("shape")
        if not isinstance(shape, list) or len(shape) == 0:
            shape = _normalize_shape(item.get("total_columns"), 128)
        candidates.append({"input_layer_name": input_name, "source_feature_name": source_name, "shape": shape})

    for item in _as_list_of_dicts(payload_context.get("input_features_config")):
        feature_name = str(item.get("feature_name", "")).strip()
        if feature_name == "":
            continue
        input_name = str(item.get("default_input_layer_name", f"input_{feature_name}")).strip() or f"input_{feature_name}"
        candidates.append(
            {
                "input_layer_name": input_name,
                "source_feature_name": feature_name,
                "shape": _normalize_shape(item.get("total_columns"), 128),
            }
        )

    for ref in _as_list_of_dicts(payload_context.get("reference_models")):
        model_full = _as_dict(ref.get("model_definition_full"))
        arch = _as_dict(model_full.get("architecture_definition"))
        for item in _as_list_of_dicts(arch.get("used_inputs")):
            input_name = str(item.get("input_layer_name", "")).strip()
            source_name = str(item.get("source_feature_name", input_name)).strip()
            shape = item.get("shape")
            if input_name == "":
                continue
            if not isinstance(shape, list) or len(shape) == 0:
                shape = _normalize_shape(item.get("total_columns"), 128)
            candidates.append(
                {
                    "input_layer_name": input_name,
                    "source_feature_name": source_name or input_name,
                    "shape": shape,
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        name = str(item.get("input_layer_name", "")).strip()
        if name:
            deduped[name] = item
    return list(deduped.values())


def _extract_available_targets(payload_context: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for item in _as_list_of_dicts(payload_context.get("output_targets_config_runtime")):
        target_name = str(item.get("target_name", "")).strip()
        output_name = str(item.get("default_output_layer_name", target_name or "output_main")).strip()
        if output_name == "":
            continue
        candidates.append(
            {
                "output_layer_name": output_name,
                "maps_to_target_config_name": target_name or output_name,
                "units": max(1, _as_int(item.get("total_columns"), 1)),
                "activation": str(item.get("activation_output_layer", "linear") or "linear"),
            }
        )

    for item in _as_list_of_dicts(payload_context.get("output_targets_config")):
        target_name = str(item.get("target_name", "")).strip()
        output_name = str(item.get("default_output_layer_name", target_name or "output_main")).strip()
        if output_name == "":
            continue
        candidates.append(
            {
                "output_layer_name": output_name,
                "maps_to_target_config_name": target_name or output_name,
                "units": max(1, _as_int(item.get("total_columns"), 1)),
                "activation": str(item.get("activation_output_layer", "linear") or "linear"),
            }
        )

    for ref in _as_list_of_dicts(payload_context.get("reference_models")):
        model_full = _as_dict(ref.get("model_definition_full"))
        arch = _as_dict(model_full.get("architecture_definition"))
        runtime_targets = _as_list_of_dicts(model_full.get("output_targets_config_runtime"))
        runtime_by_target = {
            str(item.get("target_name", "")).strip(): item
            for item in runtime_targets
            if str(item.get("target_name", "")).strip() != ""
        }
        runtime_by_layer = {
            str(item.get("default_output_layer_name", "")).strip(): item
            for item in runtime_targets
            if str(item.get("default_output_layer_name", "")).strip() != ""
        }
        for head in _as_list_of_dicts(arch.get("output_heads")):
            output_name = str(head.get("output_layer_name", "")).strip()
            maps_to = str(head.get("maps_to_target_config_name", output_name)).strip() or output_name
            if output_name == "":
                continue
            runtime = runtime_by_target.get(maps_to) or runtime_by_layer.get(output_name) or {}
            candidates.append(
                {
                    "output_layer_name": output_name,
                    "maps_to_target_config_name": maps_to,
                    "units": max(1, _as_int(head.get("units", runtime.get("total_columns", 1)), 1)),
                    "activation": str(head.get("activation", runtime.get("activation_output_layer", "linear")) or "linear"),
                }
            )

    deduped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        output_name = str(item.get("output_layer_name", "")).strip()
        if output_name:
            deduped[output_name] = item
    return list(deduped.values())


def _summarize_model_definition(model_full: dict[str, Any]) -> dict[str, Any]:
    arch = _as_dict(model_full.get("architecture_definition"))
    used_inputs = _as_list_of_dicts(arch.get("used_inputs"))
    branches = _as_list_of_dicts(arch.get("branches"))
    merges = _as_list_of_dicts(arch.get("merges"))
    output_heads = _as_list_of_dicts(arch.get("output_heads"))

    layer_types: list[str] = []
    for branch in branches:
        for layer in _as_list_of_dicts(branch.get("layers")):
            layer_type = str(layer.get("type", layer.get("layer_type", ""))).strip()
            if layer_type:
                layer_types.append(layer_type.lower())
    for merge in merges:
        for layer in _as_list_of_dicts(merge.get("layers_after_merge")):
            layer_type = str(layer.get("type", layer.get("layer_type", ""))).strip()
            if layer_type:
                layer_types.append(layer_type.lower())

    if any("conv1d" in layer_type for layer_type in layer_types):
        kind = "conv_temporal"
    elif len(branches) > 1:
        kind = "multi_branch"
    else:
        kind = "dense_baseline"

    complexity_score = len(layer_types) + 2 * len(merges) + len(output_heads)
    if complexity_score <= 6:
        complexity = "low"
    elif complexity_score <= 14:
        complexity = "medium"
    else:
        complexity = "high"

    return {
        "kind": kind,
        "focus": "robust_multi_output" if len(output_heads) > 1 else "stable_single_output",
        "inputs": len(used_inputs),
        "outputs": len(output_heads),
        "complexity": complexity,
        "expected_behavior": "safe baseline for compile/smoke validation and initial training",
    }


def _build_structured_fallback_model(candidate_id: str, payload_context: dict[str, Any]) -> tuple[dict, dict]:
    inputs = _extract_available_inputs(payload_context)
    if not inputs:
        inputs = [
            {"input_layer_name": "input_main", "source_feature_name": "input_main", "shape": [128]},
            {"input_layer_name": "input_aux", "source_feature_name": "input_aux", "shape": [16]},
        ]
    used_inputs = inputs[:2]

    main_input_name = str(used_inputs[0].get("input_layer_name", "input_main")).strip() or "input_main"
    branches: list[dict[str, Any]] = [
        {
            "name": "main_dense_path",
            "input_source_layer": main_input_name,
            "layers": [
                {"type": "Dense", "units": 96, "activation": "swish", "name": "main_dense_1"},
                {"type": "BatchNormalization", "name": "main_bn_1"},
                {"type": "Dropout", "rate": 0.2, "name": "main_dropout_1"},
                {"type": "Dense", "units": 48, "activation": "swish", "name": "main_dense_2"},
            ],
            "output_feature_map_name": "main_features",
        }
    ]

    backbone_feature_map = "main_features"
    merges: list[dict[str, Any]] = []
    if len(used_inputs) > 1:
        aux_input_name = str(used_inputs[1].get("input_layer_name", "input_aux")).strip() or "input_aux"
        branches.append(
            {
                "name": "aux_dense_path",
                "input_source_layer": aux_input_name,
                "layers": [
                    {"type": "Dense", "units": 32, "activation": "relu", "name": "aux_dense_1"},
                    {"type": "Dense", "units": 16, "activation": "relu", "name": "aux_dense_2"},
                ],
                "output_feature_map_name": "aux_features",
            }
        )
        backbone_feature_map = "shared_representation"
        merges.append(
            {
                "name": "baseline_fusion",
                "type": "concatenate",
                "source_feature_maps": ["main_features", "aux_features"],
                "layers_after_merge": [
                    {"type": "LayerNormalization", "name": "fusion_ln"},
                    {"type": "Dense", "units": 64, "activation": "swish", "name": "fusion_dense"},
                ],
                "output_feature_map_name": backbone_feature_map,
            }
        )

    targets = _extract_available_targets(payload_context)
    if not targets:
        targets = [
            {
                "output_layer_name": "output_main",
                "maps_to_target_config_name": "output_main",
                "units": 1,
                "activation": "linear",
            }
        ]

    output_heads: list[dict[str, Any]] = []
    for target in targets[:8]:
        output_name = str(target.get("output_layer_name", "output_main")).strip() or "output_main"
        output_heads.append(
            {
                "output_layer_name": output_name,
                "maps_to_target_config_name": str(target.get("maps_to_target_config_name", output_name)).strip() or output_name,
                "source_feature_map": backbone_feature_map,
                "units": max(1, _as_int(target.get("units"), 1)),
                "activation": str(target.get("activation", "linear") or "linear"),
                "use_bias": True,
            }
        )

    model_full: dict[str, Any] = {
        "model_id": candidate_id,
        "version": "1.0.0",
        "description": "Structured fallback baseline with dense branches and safe multi-output heads.",
        "change_log": ["Fallback generated when LLM is unavailable or returns incomplete payload."],
        "llm_reasoning": "Fallback prioritizes robustness, compatibility with runtime dialect, and predictable smoke-fit behavior.",
        "potential_next_variations": [
            "Increase hidden units in main branch",
            "Add light Conv1D path for temporal inputs",
            "Tune dropout and normalization for stability",
        ],
        "seed": 42,
        "architecture_definition": {
            "used_inputs": used_inputs,
            "branches": branches,
            "merges": merges,
            "output_heads": output_heads,
        },
        "training_config": {
            "compile": {
                "optimizer": {"type": "Nadam"},
                "dynamic_loss_config_source": "output_targets_config",
            },
            "fit": {"epochs": 3, "batch_size": 32},
        },
    }
    return model_full, _summarize_model_definition(model_full)


def _fallback_definition(candidate_id: str, payload_context: dict[str, Any]) -> tuple[dict, dict]:
    return _build_structured_fallback_model(candidate_id, payload_context)


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


def _candidate_from_llm(candidate_id: str, payload_context: dict[str, Any]) -> tuple[dict, dict, dict[str, Any]] | None:
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

    payload_meta = payload if isinstance(payload, dict) else {}
    normalized = normalize_llm_candidate_payload(payload)
    full = normalized.get("model_definition_full") if isinstance(normalized.get("model_definition_full"), dict) else None
    summary = normalized.get("model_definition_summary") if isinstance(normalized.get("model_definition_summary"), dict) else None

    if full is None:
        parsed_payload = payload_meta.get("_llm_parsed_payload")
        if isinstance(payload_meta.get("model_definition_full"), dict):
            full = payload_meta.get("model_definition_full")
        elif isinstance(parsed_payload, dict) and isinstance(parsed_payload.get("model_definition_full"), dict):
            full = parsed_payload.get("model_definition_full")
        elif isinstance(parsed_payload, list) and parsed_payload and isinstance(parsed_payload[0], dict):
            first = parsed_payload[0]
            if isinstance(first.get("model_definition_full"), dict):
                full = first.get("model_definition_full")

    if full is None:
        return None

    arch = _as_dict(full.get("architecture_definition"))
    if not arch:
        return None
    if not _as_list_of_dicts(arch.get("used_inputs")):
        return None
    if not _as_list_of_dicts(arch.get("output_heads")):
        return None

    if not isinstance(summary, dict) or len(summary) == 0:
        summary = _summarize_model_definition(full)

    full["model_id"] = candidate_id
    llm_metadata = {
        "provider": "openai_chat",
        "model": payload_meta.get("_llm_model", ""),
        "endpoint": payload_meta.get("_llm_endpoint", ""),
        "prompt_text": payload_meta.get("_llm_prompt_text", ""),
        "response_text": payload_meta.get("_llm_response_text", ""),
        "raw_response": payload_meta.get("_llm_raw_response", {}),
    }
    return full, summary, llm_metadata


def execute_generate_candidate(payload: dict) -> dict:
    target = int(payload.get("target_candidates", 1) or 1)
    candidates = []
    for _ in range(max(1, target)):
        candidate_id = f"cand_{uuid4().hex[:12]}"
        llm_out = _candidate_from_llm(candidate_id, payload)
        if llm_out is None:
            model_full, model_summary = _fallback_definition(candidate_id, payload)
            llm_metadata = {
                "provider": "fallback",
                "reason": "llm_unavailable_or_invalid",
            }
        else:
            model_full, model_summary, llm_metadata = llm_out

        if not isinstance(model_summary, dict) or len(model_summary) == 0:
            model_summary = _summarize_model_definition(model_full if isinstance(model_full, dict) else {})

        candidates.append(
            {
                "candidate_id": candidate_id,
                "fingerprint": uuid4().hex,
                "model_definition_full": model_full,
                "model_definition_summary": model_summary,
                "llm_metadata": llm_metadata,
            }
        )
    return {"status": "completed", "candidates": candidates}
