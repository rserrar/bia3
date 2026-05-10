from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _input_specs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _as_list_of_dicts(experiment.get("input_features_config")):
        source = str(item.get("feature_name", "")).strip()
        layer = str(item.get("default_input_layer_name", "")).strip()
        cols = int(item.get("total_columns", 0) or 0)
        if source and layer and cols > 0:
            out.append({"source_feature_name": source, "input_layer_name": layer, "total_columns": cols})
    return out


def _output_specs(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in _as_list_of_dicts(experiment.get("output_targets_config")):
        target = str(item.get("target_name", "")).strip()
        layer = str(item.get("default_output_layer_name", "")).strip()
        cols = int(item.get("total_columns", 0) or 0)
        if target and layer and cols > 0:
            out.append({"target_name": target, "output_layer_name": layer, "total_columns": cols})
    return out


def _guess_input_source(name: str, allowed_sources: set[str]) -> str | None:
    n = name.strip().lower()
    direct = {
        "input_dades_1": "extra_data_details",
        "input_extra": "extra_data_details",
        "input_extra_data": "extra_data_details",
        "input_last_price": "last_closing_price_feature",
        "input_prices_full_800": "prices_hist_full_800",
        "input_prices_main_700": "prices_hist_main_700",
        "input_prices_last_100": "prices_hist_last_100",
        "input_min_hist_800": "min_prices_hist_800",
        "input_max_hist_800": "max_prices_hist_800",
        "input_volume_full_800": "volume_hist_full_800",
        "prices_hist_800": "prices_hist_full_800",
        "volum_hist_800": "volume_hist_full_800",
    }
    if n in direct and direct[n] in allowed_sources:
        return direct[n]
    if "extra" in n and "extra_data_details" in allowed_sources:
        return "extra_data_details"
    if "volume" in n and "volume_hist_full_800" in allowed_sources:
        return "volume_hist_full_800"
    if "last" in n and "100" in n and "prices_hist_last_100" in allowed_sources:
        return "prices_hist_last_100"
    if "main" in n and "700" in n and "prices_hist_main_700" in allowed_sources:
        return "prices_hist_main_700"
    if ("full" in n or "800" in n or "prices" in n) and "prices_hist_full_800" in allowed_sources:
        return "prices_hist_full_800"
    if "min" in n and "min_prices_hist_800" in allowed_sources:
        return "min_prices_hist_800"
    if "max" in n and "max_prices_hist_800" in allowed_sources:
        return "max_prices_hist_800"
    return None


def normalize_model_definition_to_experiment(model_definition_full: dict[str, Any], experiment: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(model_definition_full, dict):
        return model_definition_full
    architecture = _as_dict(model_definition_full.get("architecture_definition"))
    if not architecture:
        return model_definition_full

    inputs = _input_specs(experiment)
    outputs = _output_specs(experiment)
    if not inputs or not outputs:
        return model_definition_full

    source_to_layer = {x["source_feature_name"]: x["input_layer_name"] for x in inputs}
    source_to_cols = {x["source_feature_name"]: int(x["total_columns"]) for x in inputs}
    layer_to_source = {x["input_layer_name"]: x["source_feature_name"] for x in inputs}
    allowed_sources = set(source_to_layer.keys())

    target_to_layer = {x["target_name"]: x["output_layer_name"] for x in outputs}
    target_to_cols = {x["target_name"]: int(x["total_columns"]) for x in outputs}
    layer_to_target = {x["output_layer_name"]: x["target_name"] for x in outputs}
    allowed_targets = set(target_to_layer.keys())

    alias_map: dict[str, str] = {}
    normalized_inputs: list[dict[str, Any]] = []
    seen_sources: set[str] = set()

    for raw in _as_list_of_dicts(architecture.get("used_inputs")):
        current_layer = str(raw.get("input_layer_name", "")).strip()
        current_source = str(raw.get("source_feature_name", "")).strip()
        old_layer = current_layer
        old_source = current_source

        if current_source not in allowed_sources:
            if current_layer in layer_to_source:
                current_source = layer_to_source[current_layer]
            else:
                guessed = _guess_input_source(current_source or current_layer, allowed_sources)
                if guessed:
                    current_source = guessed

        if current_source not in allowed_sources:
            continue
        canonical_layer = source_to_layer[current_source]
        current_layer = canonical_layer

        raw["source_feature_name"] = current_source
        raw["input_layer_name"] = current_layer
        raw["shape"] = [source_to_cols[current_source]]

        for key in {old_layer, old_source, current_layer, current_source}:
            k = str(key or "").strip()
            if k:
                alias_map[k] = current_layer

        if current_source in seen_sources:
            continue
        seen_sources.add(current_source)
        normalized_inputs.append(raw)

    for spec in inputs:
        src = str(spec["source_feature_name"])
        if src in seen_sources:
            continue
        normalized_inputs.append(
            {
                "input_layer_name": str(spec["input_layer_name"]),
                "source_feature_name": src,
                "shape": [int(spec["total_columns"])],
            }
        )
        seen_sources.add(src)
        alias_map[src] = str(spec["input_layer_name"])
        alias_map[str(spec["input_layer_name"])] = str(spec["input_layer_name"])

    if normalized_inputs:
        architecture["used_inputs"] = normalized_inputs

    for branch in _as_list_of_dicts(architecture.get("branches")):
        key = str(branch.get("input_source_layer", branch.get("input_layer_name", ""))).strip()
        if key in alias_map:
            branch["input_source_layer"] = alias_map[key]

    for head in _as_list_of_dicts(architecture.get("output_heads")):
        target_name = str(head.get("maps_to_target_config_name", "")).strip()
        output_layer = str(head.get("output_layer_name", "")).strip()

        if target_name not in allowed_targets:
            if output_layer in layer_to_target:
                target_name = layer_to_target[output_layer]
            else:
                low = (target_name or output_layer).lower()
                if "stop" in low and "stop_loss_prediction" in allowed_targets:
                    target_name = "stop_loss_prediction"
                elif "take" in low or "profit" in low:
                    if "take_profit_prediction" in allowed_targets:
                        target_name = "take_profit_prediction"

        if target_name not in allowed_targets:
            continue
        head["maps_to_target_config_name"] = target_name
        head["output_layer_name"] = target_to_layer[target_name]
        head["units"] = int(target_to_cols[target_name])

    model_definition_full["architecture_definition"] = architecture
    return model_definition_full
