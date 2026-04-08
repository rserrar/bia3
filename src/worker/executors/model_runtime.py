from __future__ import annotations

import base64
import tempfile
import gc
from pathlib import Path
from typing import Any

import numpy as np

from .data_pipeline_v2 import (
    derive_additional_features_and_targets,
    load_all_raw_data_sources,
    load_experiment_config,
)


def _tf():
    import tensorflow as tf

    return tf


def build_keras_model(
    model_definition_full: dict[str, Any],
    feature_dim: int = 16,
    input_dims_map: dict[str, int] | None = None,
    output_units_map: dict[str, int] | None = None,
):
    tf = _tf()

    def _layer_kind(layer: dict[str, Any], default: str = "Dense") -> str:
        kind = layer.get("type")
        if kind is None:
            kind = layer.get("layer_type")
        return str(kind or default)

    def _layer_params(layer: dict[str, Any]) -> dict[str, Any]:
        params = layer.get("params")
        if isinstance(params, dict):
            merged = dict(layer)
            merged.pop("params", None)
            merged.update(params)
            return merged
        return layer

    def _apply_supported_layer(x: Any, layer: dict[str, Any], default_activation: str = "relu") -> Any:
        layer_type = _layer_kind(layer)
        p = _layer_params(layer)
        if layer_type == "Dense":
            units = int(p.get("units", 32) or 32)
            activation = str(p.get("activation", default_activation))
            return tf.keras.layers.Dense(units, activation=activation)(x)
        if layer_type == "Dropout":
            rate = float(p.get("rate", 0.2) or 0.2)
            return tf.keras.layers.Dropout(rate)(x)
        return x

    arch_raw = model_definition_full.get("architecture_definition")
    arch: dict[str, Any] = arch_raw if isinstance(arch_raw, dict) else {}
    used_inputs_value = arch.get("used_inputs", [])
    branches_value = arch.get("branches", [])
    shared_layers_value = arch.get("shared_layers", [])
    output_heads_value = arch.get("output_heads", [])
    used_inputs_raw = used_inputs_value if isinstance(used_inputs_value, list) else []
    branches_raw = branches_value if isinstance(branches_value, list) else []
    shared_layers_raw = shared_layers_value if isinstance(shared_layers_value, list) else []
    output_heads_raw = output_heads_value if isinstance(output_heads_value, list) else []

    used_inputs = [item for item in used_inputs_raw if isinstance(item, dict)]
    branches = [item for item in branches_raw if isinstance(item, dict)]
    shared_layers = [item for item in shared_layers_raw if isinstance(item, dict)]
    output_heads = [item for item in output_heads_raw if isinstance(item, dict)]

    if not used_inputs:
        used_inputs = [{"input_layer_name": "input_main"}]
    if not branches:
        branches = [{"layers": [{"type": "Dense", "params": {"units": 32, "activation": "relu"}}]}]
    if not output_heads:
        output_heads = [{"output_layer_name": "output_main"}]

    input_layers = {}
    for inp in used_inputs:
        name = str(inp.get("input_layer_name", "input_main"))
        dim = feature_dim
        if isinstance(input_dims_map, dict) and name in input_dims_map:
            dim = max(1, int(input_dims_map[name]))
        input_layers[name] = tf.keras.Input(shape=(dim,), name=name)

    layer_values = list(input_layers.values())
    if len(layer_values) == 1:
        base = layer_values[0]
    else:
        base = tf.keras.layers.Concatenate(name="concat_all_inputs")(layer_values)

    branch_outputs = []
    feature_maps: dict[str, Any] = {}
    for branch in branches:
        branch_input_name = str(branch.get("input_layer_name", "")).strip()
        x = input_layers.get(branch_input_name, base)
        layers_value = branch.get("layers", [])
        layers = layers_value if isinstance(layers_value, list) else []
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            x = _apply_supported_layer(x, layer)
            fmap_name = str(layer.get("output_feature_map_name", "")).strip()
            if fmap_name:
                feature_maps[fmap_name] = x
        branch_outputs.append(x)
        branch_name = str(branch.get("branch_name", branch.get("branch_id", ""))).strip()
        if branch_name:
            feature_maps[branch_name] = x

    if len(branch_outputs) == 1:
        merged = branch_outputs[0]
    else:
        merged = tf.keras.layers.Concatenate()(branch_outputs)

    for layer in shared_layers:
        layer_type = _layer_kind(layer)
        if layer_type == "Concatenate":
            map_names = layer.get("input_source_feature_maps")
            if isinstance(map_names, list):
                tensors = [feature_maps.get(str(name), None) for name in map_names]
                tensors = [tensor for tensor in tensors if tensor is not None]
                if len(tensors) >= 2:
                    merged = tf.keras.layers.Concatenate(name=str(layer.get("name", "concat_shared")))(tensors)
                    continue
            continue
        merged = _apply_supported_layer(merged, layer)

    outputs = []
    for head in output_heads:
        out_name = str(head.get("output_layer_name", "output_main"))
        units = 1
        if isinstance(output_units_map, dict) and out_name in output_units_map:
            units = max(1, int(output_units_map[out_name]))
        head_x = merged
        head_layers_value = head.get("layers", [])
        head_layers = head_layers_value if isinstance(head_layers_value, list) else []
        for layer in head_layers:
            if not isinstance(layer, dict):
                continue
            head_x = _apply_supported_layer(head_x, layer, default_activation="linear")
        outputs.append(tf.keras.layers.Dense(units, activation="linear", name=out_name)(head_x))

    model = tf.keras.Model(inputs=list(input_layers.values()), outputs=outputs if len(outputs) > 1 else outputs[0])
    if len(model.output_names) <= 1:
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    else:
        losses = {name: "mse" for name in model.output_names}
        metrics = {name: ["mae"] for name in model.output_names}
        model.compile(optimizer="adam", loss=losses, metrics=metrics)
    return model, list(input_layers.keys()), list(model.output_names)


def _sanitize_real_array(arr: np.ndarray, label: str) -> np.ndarray:
    out = arr.astype("float32", copy=False)
    if not np.isfinite(out).all():
        print(f"[WARN] Non-finite values detected in {label}; sanitizing", flush=True)
        out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    return out


def run_smoke_fit(model_definition_full: dict[str, Any], smoke_batches: int = 3, feature_dim: int = 16, batch_size: int = 8) -> dict[str, Any]:
    tf = _tf()
    model, input_names, output_names = build_keras_model(model_definition_full, feature_dim=feature_dim)

    n = max(1, int(smoke_batches)) * max(2, int(batch_size))
    x_data = {name: np.random.randn(n, feature_dim).astype("float32") for name in input_names}

    if len(model.outputs) == 1:
        y_data = np.random.randn(n, 1).astype("float32")
    else:
        y_data = {}
        for tensor, name in zip(model.outputs, output_names):
            units = int(tensor.shape[-1] or 1)
            y_data[name] = np.random.randn(n, units).astype("float32")

    history = model.fit(x_data, y_data, epochs=1, batch_size=batch_size, verbose=0)
    tf.keras.backend.clear_session()
    loss = float(history.history.get("loss", [0.0])[-1]) if history.history else 0.0
    mae = 0.0
    if history.history:
        if "mae" in history.history:
            mae = float(history.history.get("mae", [0.0])[-1])
        else:
            mae_keys = sorted([key for key in history.history.keys() if key.endswith("_mae")])
            if mae_keys:
                mae = float(np.mean([float(history.history[key][-1]) for key in mae_keys]))
    return {"loss": loss, "mae": mae}


def render_model_plot_png_base64(model_definition_full: dict[str, Any], feature_dim: int = 16) -> str | None:
    tf = _tf()
    try:
        model, _, _ = build_keras_model(model_definition_full, feature_dim=feature_dim)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            tf.keras.utils.plot_model(model, to_file=tmp.name, show_shapes=True, show_dtype=False)
            tmp.seek(0)
            raw = tmp.read()
        tf.keras.backend.clear_session()
        if not raw:
            return None
        return base64.b64encode(raw).decode("ascii")
    except Exception:
        try:
            tf.keras.backend.clear_session()
        except Exception:
            pass
        return None


def run_smoke_fit_real_data(
    model_definition_full: dict[str, Any],
    *,
    experiment_config_file: str,
    base_data_dir: str,
    max_rows: int = 4096,
    batch_size: int = 16,
    cache_dtype: str = "float32",
    use_memmap_cache: bool = True,
) -> dict[str, Any]:
    tf = _tf()
    exp = load_experiment_config(experiment_config_file)
    input_cfg_value = exp.get("input_features_config", [])
    output_cfg_value = exp.get("output_targets_config", [])
    data_paths_value = exp.get("data_paths", {})
    input_cfg_raw = input_cfg_value if isinstance(input_cfg_value, list) else []
    output_cfg_raw = output_cfg_value if isinstance(output_cfg_value, list) else []
    data_paths_raw = data_paths_value if isinstance(data_paths_value, dict) else {}
    input_cfg = [item for item in input_cfg_raw if isinstance(item, dict)]
    output_cfg = [item for item in output_cfg_raw if isinstance(item, dict)]
    data_paths = {str(k): v for k, v in data_paths_raw.items()}

    raw = load_all_raw_data_sources(
        data_paths,
        input_cfg,
        output_cfg,
        base_data_dir=base_data_dir,
        cache_dtype=cache_dtype,
        use_memmap_cache=use_memmap_cache,
    )
    all_data = derive_additional_features_and_targets(raw, input_cfg, output_cfg)

    input_dims_map: dict[str, int] = {}
    output_units_map: dict[str, int] = {}

    arch_raw = model_definition_full.get("architecture_definition")
    arch: dict[str, Any] = arch_raw if isinstance(arch_raw, dict) else {}
    used_inputs_value = arch.get("used_inputs", [])
    output_heads_value = arch.get("output_heads", [])
    used_inputs_raw = used_inputs_value if isinstance(used_inputs_value, list) else []
    output_heads_raw = output_heads_value if isinstance(output_heads_value, list) else []
    used_inputs = [item for item in used_inputs_raw if isinstance(item, dict)]
    output_heads = [item for item in output_heads_raw if isinstance(item, dict)]

    output_cfg_map: dict[str, int] = {}
    output_alias_to_target: dict[str, str] = {}
    data_path_alias_to_source: dict[str, str] = {}
    for source_key, file_name in data_paths.items():
        if not isinstance(source_key, str) or not isinstance(file_name, str):
            continue
        base = Path(file_name).name
        stem = Path(base).stem
        if stem:
            data_path_alias_to_source[stem] = source_key
        if base:
            data_path_alias_to_source[base] = source_key
    for item in output_cfg:
        target_name = str(item.get("target_name", "")).strip()
        layer_name = str(item.get("default_output_layer_name", "")).strip()
        source_csv_key = str(item.get("source_csv_key", "")).strip()
        cols = max(1, int(item.get("total_columns", 1) or 1))
        if target_name:
            output_cfg_map[target_name] = cols
            output_alias_to_target[target_name] = target_name
        if layer_name:
            output_cfg_map[layer_name] = cols
            if target_name:
                output_alias_to_target[layer_name] = target_name
        if source_csv_key:
            output_cfg_map[source_csv_key] = cols
            if target_name:
                output_alias_to_target[source_csv_key] = target_name

    for inp in used_inputs:
        input_name = str(inp.get("input_layer_name", "")).strip()
        source_feature_name = str(inp.get("source_feature_name", input_name)).strip()
        arr = all_data.get(source_feature_name)
        arr_np = arr if isinstance(arr, np.ndarray) else None
        if input_name and arr_np is not None and arr_np.ndim >= 2 and arr_np.shape[1] > 0:
            input_dims_map[input_name] = int(arr_np.shape[1])

    for head in output_heads:
        maps_to = str(head.get("maps_to_target_config_name", "")).strip()
        out_name = str(head.get("output_layer_name", "")).strip()
        if out_name == "":
            continue
        units = output_cfg_map.get(maps_to) or output_cfg_map.get(out_name)
        if units is None:
            target_name = maps_to or out_name
            arr = all_data.get(target_name)
            arr_np = arr if isinstance(arr, np.ndarray) else None
            if arr_np is not None and arr_np.ndim >= 2 and arr_np.shape[1] > 0:
                units = int(arr_np.shape[1])
        output_units_map[out_name] = max(1, int(units or 1))

    model, input_names, _ = build_keras_model(
        model_definition_full,
        feature_dim=16,
        input_dims_map=input_dims_map,
        output_units_map=output_units_map,
    )

    x_data: dict[str, np.ndarray] = {}
    y_map: dict[str, np.ndarray] = {}

    for idx, input_name in enumerate(input_names):
        source_feature_name = input_name
        if idx < len(used_inputs):
            source_feature_name = str(used_inputs[idx].get("source_feature_name", input_name))
        arr = all_data.get(source_feature_name)
        arr_np = arr if isinstance(arr, np.ndarray) else None
        if arr_np is None or arr_np.size == 0:
            raise RuntimeError(f"missing real input data for feature: {source_feature_name}")
        x_data[input_name] = _sanitize_real_array(arr_np, f"input:{source_feature_name}")

    for head in output_heads:
        maps_to = str(head.get("maps_to_target_config_name", "")).strip()
        out_name = str(head.get("output_layer_name", "")).strip()
        target_name = maps_to or out_name
        canonical_target = output_alias_to_target.get(target_name, target_name)
        if canonical_target != target_name:
            target_name = canonical_target
        arr = all_data.get(target_name)
        if (not isinstance(arr, np.ndarray) or arr.size == 0) and target_name in data_path_alias_to_source:
            source_key = data_path_alias_to_source[target_name]
            arr = all_data.get(source_key)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise RuntimeError(f"missing real target data for target: {target_name}")
        if out_name == "":
            out_name = target_name
        y_map[out_name] = _sanitize_real_array(arr, f"target:{target_name}")

    if not y_map:
        raise RuntimeError("no real targets found for output_heads")

    lengths = [arr.shape[0] for arr in list(x_data.values()) + list(y_map.values()) if isinstance(arr, np.ndarray) and arr.ndim >= 2]
    if not lengths:
        raise RuntimeError("real data arrays are empty")
    n = min(min(lengths), max(8, int(max_rows)))

    x_fit = {k: v[:n] for k, v in x_data.items()}
    if len(model.outputs) == 1:
        only_output = model.output_names[0]
        if only_output in y_map:
            y_fit: Any = y_map[only_output][:n]
        else:
            y_fit = list(y_map.values())[0][:n]
    else:
        y_fit = {name: y_map[name][:n] for name in model.output_names if name in y_map}
        if len(y_fit) != len(model.output_names):
            missing = [name for name in model.output_names if name not in y_fit]
            raise RuntimeError(f"missing real target data for output heads: {', '.join(missing)}")

    history = model.fit(x_fit, y_fit, epochs=1, batch_size=batch_size, verbose=0)
    tf.keras.backend.clear_session()
    del raw
    del all_data
    gc.collect()
    loss = float(history.history.get("loss", [0.0])[-1]) if history.history else 0.0
    mae = 0.0
    if history.history:
        if "mae" in history.history:
            mae = float(history.history.get("mae", [0.0])[-1])
        else:
            mae_keys = sorted([key for key in history.history.keys() if key.endswith("_mae")])
            if mae_keys:
                mae = float(np.mean([float(history.history[key][-1]) for key in mae_keys]))
    return {"loss": loss, "mae": mae, "samples": n}
