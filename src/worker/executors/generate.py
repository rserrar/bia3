from uuid import uuid4
import json
import re
from pathlib import Path
from typing import Any

from src.shared.settings import load_settings
from .llm_client import LlmRequestError, generate_candidate_via_openai, normalize_llm_candidate_payload
from ..progress import report_progress


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_non_empty_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        item = value.strip()
        return [item] if item else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _extract_parent_model_id(payload: dict[str, Any]) -> str | None:
    direct = str(payload.get("parent_model_id", "")).strip()
    if direct:
        return direct

    prompt_context = _as_dict(payload.get("prompt_context"))
    parent_model = _as_dict(prompt_context.get("parent_model"))
    parent_id = str(parent_model.get("model_id", "")).strip()
    if parent_id:
        return parent_id

    family_summary = _as_dict(prompt_context.get("family_metrics_summary"))
    parent_id = str(family_summary.get("parent_model_id", "")).strip()
    if parent_id:
        return parent_id

    return None


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


def _to_json_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return "{}"


def _inputs_description_from_config(config: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in _as_list_of_dicts(config.get("input_features_config")):
        feature = str(item.get("feature_name", "unknown"))
        cols = _as_int(item.get("total_columns"), 0)
        mandatory = bool(item.get("is_mandatory_input", False))
        default_layer = str(item.get("default_input_layer_name", ""))
        source_csv = str(item.get("source_csv_key", ""))
        desc = str(item.get("description", ""))
        rows.append(
            f"- feature_name={feature} · cols={cols} · mandatory={mandatory} · "
            f"default_input_layer_name={default_layer} · csv={source_csv} · {desc}"
        )
    return "\n".join(rows) if rows else "No input features config available"


def _outputs_description_from_config(config: dict[str, Any]) -> str:
    rows: list[str] = []
    for item in _as_list_of_dicts(config.get("output_targets_config")):
        target = str(item.get("target_name", "unknown"))
        cols = _as_int(item.get("total_columns"), 0)
        mandatory = bool(item.get("is_mandatory_output", False))
        layer = str(item.get("default_output_layer_name", ""))
        loss = str(item.get("loss_function", ""))
        activation = str(item.get("activation_output_layer", ""))
        rows.append(
            f"- target_name={target} · cols={cols} · mandatory={mandatory} · "
            f"default_output_layer_name={layer} · loss={loss} · activation={activation}"
        )
    return "\n".join(rows) if rows else "No output targets config available"


def _render_prompt_template(template: str, context: dict[str, Any]) -> tuple[str, list[str]]:
    out = template
    for key, value in context.items():
        out = out.replace("{{" + key + "}}", str(value))
        out = out.replace("{{ " + key + " }}", str(value))

    unresolved = sorted(set(re.findall(r"\{\{\s*([^}]+?)\s*\}\}", out)))
    for key in unresolved:
        cleaned = str(key).strip()
        marker = f"[UNAVAILABLE:{cleaned}]"
        out = out.replace("{{" + cleaned + "}}", marker)
        out = out.replace("{{ " + cleaned + " }}", marker)
    return out, unresolved


def _default_template_for_mode(mode: str) -> str:
    if mode == "repair":
        return (
            "Repair this model definition JSON so it compiles in Keras and can run smoke fit.\n"
            "Buggy model JSON:\n{{buggy_model_json}}\n"
            "Validation error:\n{{validation_error}}\n"
            "Traceback:\n{{error_traceback}}\n"
            "Working example:\n{{working_model_example_json}}\n"
            "Return ONLY JSON with key model_definition_full."
        )
    return (
        "Generate {{num_new_models}} Keras model candidates as JSON.\n"
        "Use architecture guide:\n{{architecture_guide_content}}\n"
        "Available inputs:\n{{available_inputs_description}}\n"
        "Available outputs:\n{{available_outputs_description}}\n"
        "Best models context:\n{{best_performing_models_json}}\n"
        "Genealogy:\n{{genealogy_case_studies}}\n"
        "Return only JSON with model_definition_full and model_definition_summary."
    )


def _template_file_for_mode(mode: str) -> str:
    if mode == "repair":
        return "prompts/fix_model_error.txt"
    if mode == "evolution":
        return "prompts/generate_evolution_models.txt"
    return "prompts/generate_exploration_models.txt"


def _normalize_architecture_aliases(model_full: dict[str, Any]) -> None:
    arch = _as_dict(model_full.get("architecture_definition"))
    if not arch:
        return

    for merge in _as_list_of_dicts(arch.get("merges")):
        merge_type = str(merge.get("type", "")).strip()
        if merge_type == "":
            alias = str(merge.get("merge_type", "")).strip()
            if alias:
                merge["type"] = alias

        sources = _as_non_empty_str_list(merge.get("source_feature_maps"))
        if not sources:
            sources = _as_non_empty_str_list(merge.get("input_source_feature_maps"))
        if sources:
            merge["source_feature_maps"] = sources

        if not str(merge.get("output_feature_map_name", "")).strip():
            for key in ("output_feature_map", "output_name", "output_layer_name"):
                alias = str(merge.get(key, "")).strip()
                if alias:
                    merge["output_feature_map_name"] = alias
                    break

        if "layers_after_merge" not in merge and isinstance(merge.get("layers"), list):
            merge["layers_after_merge"] = merge.get("layers")


def _build_prompt_context_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    mode = str(payload.get("generation_mode", "exploration")).strip().lower() or "exploration"
    prompt_context = _as_dict(payload.get("prompt_context"))
    experiment = _read_json_if_exists(settings.experiment_config_file)

    available_inputs_description = str(prompt_context.get("available_inputs_description", "")).strip()
    if available_inputs_description == "":
        source_cfg = _as_dict(prompt_context.get("experiment_config")) or experiment
        available_inputs_description = _inputs_description_from_config(source_cfg)

    available_outputs_description = str(prompt_context.get("available_outputs_description", "")).strip()
    if available_outputs_description == "":
        source_cfg = _as_dict(prompt_context.get("experiment_config")) or experiment
        available_outputs_description = _outputs_description_from_config(source_cfg)

    architecture_guide_content = str(prompt_context.get("architecture_guide_content", "")).strip()
    if architecture_guide_content == "":
        architecture_guide_content = _read_text_if_exists(settings.architecture_guide_file)

    target_candidates = 1

    context: dict[str, Any] = {
        "num_new_models": str(target_candidates),
        "available_inputs_description": available_inputs_description,
        "available_outputs_description": available_outputs_description,
        "architecture_guide_content": architecture_guide_content,
        "genealogy_case_studies": str(prompt_context.get("genealogy_case_studies", "")),
        "best_performing_models_json": "[]",
        "num_best_models_considered": "0",
        "improvement_focus": str(prompt_context.get("improvement_focus", "")),
        "exploration_hints": str(prompt_context.get("exploration_hints", "")),
        "validation_error": str(prompt_context.get("validation_error", "")),
        "error_traceback": str(prompt_context.get("error_traceback", "")),
        "buggy_model_json": _to_json_text(prompt_context.get("buggy_model")),
        "working_model_example_json": _to_json_text(prompt_context.get("working_model_example")),
        "best_models_global_json": _to_json_text(prompt_context.get("best_models_global", [])),
        "family_models_json": _to_json_text(prompt_context.get("family_models", [])),
        "parent_model_json": _to_json_text(prompt_context.get("parent_model")),
        "family_metrics_summary": _to_json_text(prompt_context.get("family_metrics_summary", {})),
        "metrics_summary": _to_json_text(prompt_context.get("metrics_summary", {})),
    }

    if mode == "evolution":
        parent_model = _as_dict(prompt_context.get("parent_model"))
        family_models = _as_list_of_dicts(prompt_context.get("family_models"))
        best = [item for item in ([parent_model] + family_models) if isinstance(item, dict) and item]
        context["best_performing_models_json"] = _to_json_text(best[:10])
        context["num_best_models_considered"] = str(len(best[:10]))
    elif mode == "exploration":
        best_global = _as_list_of_dicts(prompt_context.get("best_models_global"))
        context["best_performing_models_json"] = _to_json_text(best_global[:10])
        context["num_best_models_considered"] = str(len(best_global[:10]))

    return context


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

    # Fallback to experiment config targets when server payload does not include explicit target configs.
    if not candidates:
        settings = load_settings()
        experiment = _read_json_if_exists(settings.experiment_config_file)
        for item in _as_list_of_dicts(experiment.get("output_targets_config")):
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


def _default_llm_trace(*, generation_mode: str, template_file: str, error_type: str, error_message: str) -> dict[str, Any]:
    return {
        "provider": "openai_chat",
        "generation_mode": generation_mode,
        "template_file": template_file,
        "model": "",
        "endpoint": "",
        "prompt": None,
        "response_raw": None,
        "response_parsed": None,
        "parse_ok": False,
        "error_type": error_type,
        "error_message": error_message,
    }


def _candidate_from_llm(
    candidate_id: str,
    payload_context: dict[str, Any],
    prompt_text: str,
    generation_mode: str,
    template_file: str,
) -> tuple[tuple[dict, dict, dict[str, Any]] | None, dict[str, Any] | None, str | None]:
    settings = load_settings()
    if settings.llm_mode != "openai_chat" or settings.llm_api_key.strip() == "":
        return None, None, "llm_disabled_or_unconfigured"
    try:
        payload = generate_candidate_via_openai(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            prompt=prompt_text,
            endpoint=settings.llm_endpoint,
        )
    except LlmRequestError as error:
        llm_trace = error.llm_trace if isinstance(error.llm_trace, dict) else None
        reason = str((llm_trace or {}).get("error_type") or "llm_request_error")
        print(f"[LLM] fallback reason={reason}", flush=True)
        return None, llm_trace, reason
    except Exception as error:
        print(f"[WARN] LLM generation failed, fallback to baseline: {error}")
        llm_trace = _default_llm_trace(
            generation_mode=generation_mode,
            template_file=template_file,
            error_type="llm_unhandled_error",
            error_message=str(error),
        )
        print("[LLM] fallback reason=llm_unhandled_error", flush=True)
        return None, llm_trace, "llm_unhandled_error"

    payload_meta = payload if isinstance(payload, dict) else {}
    llm_trace = payload_meta.get("_llm_trace") if isinstance(payload_meta.get("_llm_trace"), dict) else None
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

    if full is None and isinstance(payload_meta.get("proposal"), dict):
        proposal = _as_dict(payload_meta.get("proposal"))
        maybe_full = proposal.get("model_definition")
        if isinstance(maybe_full, dict):
            full = maybe_full

    if full is None:
        reason = str(payload_meta.get("_llm_error_type", "llm_invalid_payload"))
        if llm_trace is None:
            llm_trace = _default_llm_trace(
                generation_mode=generation_mode,
                template_file=template_file,
                error_type=reason,
                error_message="model_definition_full missing after normalization",
            )
        print(f"[LLM] fallback reason={reason}", flush=True)
        return None, llm_trace, reason

    _normalize_architecture_aliases(full)

    arch = _as_dict(full.get("architecture_definition"))
    if not arch:
        reason = "llm_invalid_architecture_definition"
        if llm_trace is None:
            llm_trace = _default_llm_trace(
                generation_mode=generation_mode,
                template_file=template_file,
                error_type=reason,
                error_message="architecture_definition missing or invalid",
            )
        print(f"[LLM] fallback reason={reason}", flush=True)
        return None, llm_trace, reason
    if not _as_list_of_dicts(arch.get("used_inputs")):
        reason = "llm_invalid_used_inputs"
        if llm_trace is None:
            llm_trace = _default_llm_trace(
                generation_mode=generation_mode,
                template_file=template_file,
                error_type=reason,
                error_message="used_inputs missing or invalid",
            )
        print(f"[LLM] fallback reason={reason}", flush=True)
        return None, llm_trace, reason
    if not _as_list_of_dicts(arch.get("output_heads")):
        reason = "llm_invalid_output_heads"
        if llm_trace is None:
            llm_trace = _default_llm_trace(
                generation_mode=generation_mode,
                template_file=template_file,
                error_type=reason,
                error_message="output_heads missing or invalid",
            )
        print(f"[LLM] fallback reason={reason}", flush=True)
        return None, llm_trace, reason

    if not isinstance(summary, dict) or len(summary) == 0:
        summary = _summarize_model_definition(full)

    full["model_id"] = candidate_id
    llm_metadata = {
        "provider": "openai_chat",
        "generation_mode": generation_mode,
        "template_file": template_file,
        "model": payload_meta.get("_llm_model", ""),
        "endpoint": payload_meta.get("_llm_endpoint", ""),
        "prompt_text": payload_meta.get("_llm_prompt_text", ""),
        "response_text": payload_meta.get("_llm_response_text", ""),
        "raw_response": payload_meta.get("_llm_raw_response", {}),
        "parsed_model_definition": full,
        "llm_trace": llm_trace,
    }
    return (full, summary, llm_metadata), llm_trace, None


def execute_generate_candidate(payload: dict) -> dict:
    mode = str(payload.get("generation_mode", "exploration")).strip().lower() or "exploration"
    if mode not in {"repair", "evolution", "exploration"}:
        mode = "exploration"

    template_file = _template_file_for_mode(mode)
    template_text = _read_text_if_exists(template_file)
    if template_text.strip() == "":
        template_text = _default_template_for_mode(mode)

    prompt_context = _build_prompt_context_from_payload(payload)
    prompt_text, unresolved = _render_prompt_template(template_text, prompt_context)

    print(f"[GEN] mode={mode}", flush=True)
    print(f"[GEN] using template={template_file}", flush=True)
    print(f"[GEN] context keys={','.join(sorted(prompt_context.keys()))}", flush=True)
    print(f"[GEN] prompt size={len(prompt_text)}", flush=True)
    if unresolved:
        print(f"[GEN][WARN] unresolved placeholders={','.join(unresolved)}", flush=True)

    effective_context = dict(payload)
    nested_context = _as_dict(payload.get("prompt_context"))
    effective_context.update(nested_context)

    target = int(payload.get("target_candidates", 1) or 1)
    parent_model_id = _extract_parent_model_id(payload)
    report_progress({"phase": "generate_started", "target_candidates": max(1, target), "mode": mode})
    candidates = []
    for idx in range(max(1, target)):
        report_progress({"phase": "generate_candidate_request", "index": idx + 1, "total": max(1, target), "mode": mode})
        candidate_id = f"cand_{uuid4().hex[:12]}"
        llm_out, llm_trace, fallback_reason = _candidate_from_llm(
            candidate_id,
            effective_context,
            prompt_text,
            generation_mode=mode,
            template_file=template_file,
        )
        if llm_out is None:
            model_full, model_summary = _fallback_definition(candidate_id, effective_context)
            trace_prompt = llm_trace.get("prompt") if isinstance(llm_trace, dict) else None
            trace_response = llm_trace.get("response_raw") if isinstance(llm_trace, dict) else None
            trace_model = llm_trace.get("model") if isinstance(llm_trace, dict) else ""
            trace_endpoint = llm_trace.get("endpoint") if isinstance(llm_trace, dict) else ""
            llm_metadata = {
                "provider": "fallback",
                "generation_mode": mode,
                "template_file": template_file,
                "fallback_reason": fallback_reason or "llm_unavailable_or_invalid",
                "reason": fallback_reason or "llm_unavailable_or_invalid",
                "model": trace_model,
                "endpoint": trace_endpoint,
                "prompt_text": trace_prompt or "",
                "response_text": trace_response or "",
                "raw_response": {},
                "llm_trace": llm_trace,
            }
            print(f"[LLM] fallback reason={llm_metadata['fallback_reason']}", flush=True)
        else:
            model_full, model_summary, llm_metadata = llm_out
            if "llm_trace" not in llm_metadata:
                llm_metadata["llm_trace"] = llm_trace

        if not isinstance(model_summary, dict) or len(model_summary) == 0:
            model_summary = _summarize_model_definition(model_full if isinstance(model_full, dict) else {})

        candidates.append(
            {
                "candidate_id": candidate_id,
                "fingerprint": uuid4().hex,
                "model_definition_full": model_full,
                "model_definition_summary": model_summary,
                "llm_metadata": llm_metadata,
                "parent_model_id": parent_model_id,
            }
        )
        report_progress({"phase": "generate_candidate_done", "index": idx + 1, "total": max(1, target), "candidate_id": candidate_id})
    report_progress({"phase": "generate_completed", "count": len(candidates), "mode": mode})
    return {"status": "completed", "candidates": candidates}
