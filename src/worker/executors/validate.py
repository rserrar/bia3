import traceback
from typing import Any, cast

from src.shared.settings import load_settings
from .llm_client import normalize_llm_candidate_payload, repair_model_definition_via_openai
from .model_runtime import run_smoke_fit_real_data
from ..progress import report_progress


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _layer_kind(layer: dict[str, Any]) -> str:
    kind = layer.get("type")
    if kind is None:
        kind = layer.get("layer_type")
    return str(kind or "").strip()


def _collect_potential_feature_maps(architecture: dict[str, Any]) -> set[str]:
    names: set[str] = set()

    for inp in _as_list_of_dicts(architecture.get("used_inputs")):
        input_name = str(inp.get("input_layer_name", "")).strip()
        source_name = str(inp.get("source_feature_name", "")).strip()
        if input_name:
            names.add(input_name)
        if source_name:
            names.add(source_name)

    for branch in _as_list_of_dicts(architecture.get("branches")):
        branch_name = str(branch.get("name", branch.get("branch_id", ""))).strip()
        if branch_name:
            names.add(branch_name)
        out_name = str(branch.get("output_feature_map_name", "")).strip()
        if out_name:
            names.add(out_name)
        for layer in _as_list_of_dicts(branch.get("layers")):
            layer_name = str(layer.get("name", "")).strip()
            if layer_name:
                names.add(layer_name)

    for merge in _as_list_of_dicts(architecture.get("merges")):
        merge_name = str(merge.get("name", "")).strip()
        if merge_name:
            names.add(merge_name)
        out_name = str(merge.get("output_feature_map_name", "")).strip()
        if out_name:
            names.add(out_name)
        for layer in _as_list_of_dicts(merge.get("layers_after_merge")):
            layer_name = str(layer.get("name", "")).strip()
            if layer_name:
                names.add(layer_name)

    return names


def validate_model_definition_schema(model_definition_full: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if not isinstance(model_definition_full, dict):
        return ["model_definition_full must be a dict"]

    architecture = model_definition_full.get("architecture_definition")
    if not isinstance(architecture, dict):
        return ["architecture_definition is required and must be a dict"]

    used_inputs_value = architecture.get("used_inputs")
    if not isinstance(used_inputs_value, list) or len(used_inputs_value) == 0:
        errors.append("architecture_definition.used_inputs must be a non-empty list")
    else:
        for idx, raw_input in enumerate(used_inputs_value):
            if not isinstance(raw_input, dict):
                errors.append(f"used_inputs[{idx}] must be a dict")
                continue
            input_layer_name = raw_input.get("input_layer_name")
            source_feature_name = raw_input.get("source_feature_name")
            if not _is_non_empty_string(input_layer_name):
                errors.append(f"used_inputs[{idx}].input_layer_name must be a non-empty string")
            if not _is_non_empty_string(source_feature_name):
                errors.append(f"used_inputs[{idx}].source_feature_name must be a non-empty string")

            if "shape" in raw_input:
                shape = raw_input.get("shape")
                if not isinstance(shape, list) or len(shape) == 0:
                    errors.append(f"used_inputs[{idx}].shape must be a non-empty list of ints")
                else:
                    for shape_idx, dim in enumerate(shape):
                        if not isinstance(dim, int):
                            errors.append(f"used_inputs[{idx}].shape[{shape_idx}] must be int")

    branches_value = architecture.get("branches")
    if branches_value is not None:
        if not isinstance(branches_value, list):
            errors.append("architecture_definition.branches must be a list")
        else:
            for branch_idx, raw_branch in enumerate(branches_value):
                if not isinstance(raw_branch, dict):
                    errors.append(f"branches[{branch_idx}] must be a dict")
                    continue
                branch_name = str(raw_branch.get("name", raw_branch.get("branch_id", ""))).strip()
                if branch_name == "":
                    errors.append(f"branches[{branch_idx}] must include name or branch_id")
                branch_input = str(raw_branch.get("input_source_layer", raw_branch.get("input_layer_name", ""))).strip()
                if branch_input == "":
                    errors.append(f"branches[{branch_idx}] must include input_source_layer or input_layer_name")
                layers = raw_branch.get("layers")
                if not isinstance(layers, list):
                    errors.append(f"branches[{branch_idx}].layers must be a list")
                else:
                    for layer_idx, raw_layer in enumerate(layers):
                        if not isinstance(raw_layer, dict):
                            errors.append(f"branches[{branch_idx}].layers[{layer_idx}] must be a dict")
                            continue
                        if _layer_kind(raw_layer) == "":
                            errors.append(f"branches[{branch_idx}].layers[{layer_idx}] must include type or layer_type")
                if not _is_non_empty_string(raw_branch.get("output_feature_map_name")):
                    errors.append(f"branches[{branch_idx}].output_feature_map_name must be a non-empty string")

    merges_value = architecture.get("merges")
    if merges_value is not None:
        if not isinstance(merges_value, list):
            errors.append("architecture_definition.merges must be a list")
        else:
            for merge_idx, raw_merge in enumerate(merges_value):
                if not isinstance(raw_merge, dict):
                    errors.append(f"merges[{merge_idx}] must be a dict")
                    continue
                if not _is_non_empty_string(raw_merge.get("name")):
                    errors.append(f"merges[{merge_idx}].name must be a non-empty string")
                if not _is_non_empty_string(raw_merge.get("type")):
                    errors.append(f"merges[{merge_idx}].type must be a non-empty string")
                source_maps = raw_merge.get("source_feature_maps")
                if not isinstance(source_maps, list) or len(source_maps) == 0:
                    errors.append(f"merges[{merge_idx}].source_feature_maps must be a non-empty list")
                if not _is_non_empty_string(raw_merge.get("output_feature_map_name")):
                    errors.append(f"merges[{merge_idx}].output_feature_map_name must be a non-empty string")
                if "layers_after_merge" in raw_merge and not isinstance(raw_merge.get("layers_after_merge"), list):
                    errors.append(f"merges[{merge_idx}].layers_after_merge must be a list when present")

    output_heads_value = architecture.get("output_heads")
    if not isinstance(output_heads_value, list) or len(output_heads_value) == 0:
        errors.append("architecture_definition.output_heads must be a non-empty list")
    else:
        for head_idx, raw_head in enumerate(output_heads_value):
            if not isinstance(raw_head, dict):
                errors.append(f"output_heads[{head_idx}] must be a dict")
                continue
            if not _is_non_empty_string(raw_head.get("output_layer_name")):
                errors.append(f"output_heads[{head_idx}].output_layer_name must be a non-empty string")

            if "source_feature_map" in raw_head and not _is_non_empty_string(raw_head.get("source_feature_map")):
                errors.append(f"output_heads[{head_idx}].source_feature_map must be a non-empty string when present")
            if "maps_to_target_config_name" in raw_head and not _is_non_empty_string(raw_head.get("maps_to_target_config_name")):
                errors.append(f"output_heads[{head_idx}].maps_to_target_config_name must be a non-empty string when present")

    potential_feature_maps = _collect_potential_feature_maps(architecture)
    known_feature_maps: set[str] = set()
    for inp in _as_list_of_dicts(architecture.get("used_inputs")):
        input_name = str(inp.get("input_layer_name", "")).strip()
        source_name = str(inp.get("source_feature_name", "")).strip()
        if input_name:
            known_feature_maps.add(input_name)
        if source_name:
            known_feature_maps.add(source_name)

    multi_input_required = {
        "Add",
        "Multiply",
        "Concatenate",
        "AttentionKeras",
        "MultiHeadAttentionKeras",
    }

    for branch_idx, branch in enumerate(_as_list_of_dicts(architecture.get("branches"))):
        branch_input = str(branch.get("input_source_layer", branch.get("input_layer_name", ""))).strip()
        if branch_input and branch_input not in known_feature_maps and branch_input not in potential_feature_maps:
            errors.append(f"branches[{branch_idx}] input source '{branch_input}' is not a known input/feature map")

        for layer_idx, layer in enumerate(_as_list_of_dicts(branch.get("layers"))):
            kind = _layer_kind(layer)
            explicit_source = str(layer.get("explicit_input_source_feature_map", "")).strip()
            if explicit_source and explicit_source not in known_feature_maps and explicit_source not in potential_feature_maps:
                errors.append(
                    f"branches[{branch_idx}].layers[{layer_idx}] explicit_input_source_feature_map '{explicit_source}' is unknown"
                )

            if kind in multi_input_required:
                sources = layer.get("input_source_feature_maps")
                if not isinstance(sources, list) or len(sources) == 0:
                    errors.append(
                        f"branches[{branch_idx}].layers[{layer_idx}] ({kind}) requires non-empty input_source_feature_maps"
                    )
                else:
                    for src_idx, src_name in enumerate(sources):
                        src = str(src_name).strip()
                        if src == "":
                            errors.append(
                                f"branches[{branch_idx}].layers[{layer_idx}].input_source_feature_maps[{src_idx}] must be non-empty"
                            )
                        elif src not in known_feature_maps and src not in potential_feature_maps:
                            errors.append(
                                f"branches[{branch_idx}].layers[{layer_idx}] references unknown feature map '{src}'"
                            )

            layer_name = str(layer.get("name", "")).strip()
            if layer_name:
                known_feature_maps.add(layer_name)

        branch_out = str(branch.get("output_feature_map_name", "")).strip()
        if branch_out:
            known_feature_maps.add(branch_out)

    for merge_idx, merge in enumerate(_as_list_of_dicts(architecture.get("merges"))):
        source_maps = _as_list(merge.get("source_feature_maps"))
        for src_idx, src_name in enumerate(source_maps):
            src = str(src_name).strip()
            if src == "":
                errors.append(f"merges[{merge_idx}].source_feature_maps[{src_idx}] must be non-empty")
            elif src not in known_feature_maps and src not in potential_feature_maps:
                errors.append(f"merges[{merge_idx}] references unknown source_feature_map '{src}'")

        for layer_idx, layer in enumerate(_as_list_of_dicts(merge.get("layers_after_merge"))):
            kind = _layer_kind(layer)
            explicit_source = str(layer.get("explicit_input_source_feature_map", "")).strip()
            if explicit_source and explicit_source not in known_feature_maps and explicit_source not in potential_feature_maps:
                errors.append(
                    f"merges[{merge_idx}].layers_after_merge[{layer_idx}] explicit_input_source_feature_map '{explicit_source}' is unknown"
                )
            if kind in multi_input_required:
                sources = layer.get("input_source_feature_maps")
                if not isinstance(sources, list) or len(sources) == 0:
                    errors.append(
                        f"merges[{merge_idx}].layers_after_merge[{layer_idx}] ({kind}) requires non-empty input_source_feature_maps"
                    )

            layer_name = str(layer.get("name", "")).strip()
            if layer_name:
                known_feature_maps.add(layer_name)

        merge_out = str(merge.get("output_feature_map_name", "")).strip()
        if merge_out:
            known_feature_maps.add(merge_out)

    for head_idx, head in enumerate(_as_list_of_dicts(architecture.get("output_heads"))):
        source_feature_map = str(head.get("source_feature_map", "")).strip()
        if source_feature_map and source_feature_map not in known_feature_maps and source_feature_map not in potential_feature_maps:
            errors.append(f"output_heads[{head_idx}] references unknown source_feature_map '{source_feature_map}'")

    return errors


def _run_runtime_validation(payload: dict[str, Any], model_definition_full: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    use_real_data = bool(payload.get("use_real_data", settings.real_data_mode))
    if not use_real_data:
        raise RuntimeError("validate_candidate requires real data mode (set V3_REAL_DATA_MODE=true)")
    requested_rows = int(payload.get("max_real_rows", settings.max_real_rows) or settings.max_real_rows)
    smoke_rows = min(max(128, requested_rows), int(settings.validate_smoke_max_rows))
    smoke_result = run_smoke_fit_real_data(
        model_definition_full=model_definition_full,
        experiment_config_file=settings.experiment_config_file,
        base_data_dir=settings.data_dir,
        max_rows=smoke_rows,
        batch_size=int(payload.get("batch_size", 8) or 8),
        cache_dtype=settings.data_cache_dtype,
        use_memmap_cache=settings.use_memmap_cache,
        fail_on_non_finite=settings.fail_on_non_finite,
        non_finite_sample_cols=settings.non_finite_sample_cols,
        non_finite_sample_rows=settings.non_finite_sample_rows,
    )
    smoke_result["max_rows_requested"] = requested_rows
    smoke_result["max_rows_used"] = smoke_rows
    return smoke_result


def _infer_runtime_flags(error_message: str | None) -> tuple[bool, bool]:
    message = (error_message or "").lower()

    build_hints = [
        "graph disconnected",
        "input 0 of layer",
        "expected ndim",
        "requires inputs with matching shapes",
        "unknown layer",
        "could not locate",
    ]
    compile_hints = [
        "compile",
        "optimizer",
        "loss",
        "metric",
    ]
    fit_hints = [
        "data cardinality",
        "dimensions must be equal",
        "failed to convert",
        "incompatible shapes",
        "during training",
    ]

    if any(hint in message for hint in fit_hints):
        return True, True
    if any(hint in message for hint in compile_hints):
        return True, False
    if any(hint in message for hint in build_hints):
        return False, False
    return False, False


def execute_validate_candidate(payload: dict) -> dict:
    raw_model_definition = payload.get("model_definition_full")
    model_definition_full: dict[str, Any]
    if isinstance(raw_model_definition, dict):
        model_definition_full = dict(raw_model_definition)
    else:
        model_definition_full = {}

    force_fail = bool(payload.get("force_fail", False))
    report_progress({"phase": "validate_started", "force_fail": force_fail})
    schema_errors = validate_model_definition_schema(model_definition_full)
    schema_ok = len(schema_errors) == 0

    build_ok = False
    compile_ok = False
    smoke_ok = False
    smoke_result = {}
    error_type: str | None = None
    error_message: str | None = None
    error_traceback: str | None = None

    if force_fail:
        error_type = "ForcedValidationFailure"
        error_message = "forced validation failure"
        error_traceback = None

    if schema_ok and not force_fail:
        try:
            report_progress({"phase": "validate_runtime_start"})
            smoke_result = _run_runtime_validation(payload, model_definition_full)
            build_ok = True
            compile_ok = True
            smoke_ok = True
            report_progress({"phase": "validate_runtime_done"})
        except Exception as error:
            error_type = type(error).__name__
            error_message = str(error)
            error_traceback = traceback.format_exc()
            build_ok, compile_ok = _infer_runtime_flags(error_message)

    if not schema_ok and error_message is None:
        error_type = "SchemaValidationError"
        error_message = "; ".join(schema_errors[:5])
        error_traceback = None

    repaired_model_definition_full = None
    should_attempt_repair = schema_ok and bool(model_definition_full) and (not build_ok or not compile_ok or not smoke_ok)
    if should_attempt_repair:
        settings = load_settings()
        if settings.llm_mode == "openai_chat" and settings.llm_api_key.strip() != "":
            try:
                print("[INFO] Validation failed, requesting LLM repair", flush=True)
                validation_error_parts = [error_message or "build/compile/smoke validation failed"]
                if error_traceback:
                    validation_error_parts.append("Traceback:\n" + error_traceback)
                repaired_payload = repair_model_definition_via_openai(
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    endpoint=settings.llm_endpoint,
                    model_definition_full=model_definition_full,
                    validation_error="\n\n".join(validation_error_parts),
                    fix_prompt_file=settings.fix_error_prompt_file,
                )
                normalized = normalize_llm_candidate_payload(repaired_payload)
                repaired_raw = normalized.get("model_definition_full")
                repaired_full = cast(dict[str, Any], repaired_raw) if isinstance(repaired_raw, dict) else None
                if repaired_full:
                    repaired_schema_errors = validate_model_definition_schema(repaired_full)
                    if len(repaired_schema_errors) == 0 and not force_fail:
                        smoke_result = _run_runtime_validation(payload, repaired_full)
                        build_ok = True
                        compile_ok = True
                        smoke_ok = True
                        error_type = None
                        error_message = None
                        error_traceback = None
                        repaired_model_definition_full = repaired_full
                        print("[INFO] LLM repair succeeded", flush=True)
                    elif len(repaired_schema_errors) > 0:
                        error_type = "RepairedSchemaValidationError"
                        error_message = "; ".join(repaired_schema_errors[:5])
                        error_traceback = None
                else:
                    error_type = error_type or "LLMRepairEmptyResult"
                    error_message = error_message or "LLM repair did not return model_definition_full"
            except Exception as repair_error:
                print(f"[WARN] LLM repair failed: {repair_error}", flush=True)
                error_type = type(repair_error).__name__
                error_message = str(repair_error)
                error_traceback = traceback.format_exc()

    report = {
        "schema_ok": schema_ok,
        "schema_errors": schema_errors,
        "build_ok": build_ok,
        "compile_ok": compile_ok,
        "smoke_fit_ok": smoke_ok,
        "smoke_batches": int(payload.get("smoke_batches", 3) or 3),
        "smoke_result": smoke_result,
        "error_type": error_type,
        "error_message": error_message,
        "error_traceback": error_traceback,
        "repaired": repaired_model_definition_full is not None,
    }
    result = {
        "status": "completed",
        "validation_report": report,
    }
    if repaired_model_definition_full is not None:
        result["repaired_model_definition_full"] = repaired_model_definition_full
    report_progress({
        "phase": "validate_completed",
        "schema_ok": schema_ok,
        "build_ok": build_ok,
        "compile_ok": compile_ok,
        "smoke_fit_ok": smoke_ok,
    })
    return result
