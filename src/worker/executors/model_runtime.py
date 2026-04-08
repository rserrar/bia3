from __future__ import annotations

import base64
import tempfile
import gc
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
    arch = model_definition_full.get("architecture_definition") if isinstance(model_definition_full.get("architecture_definition"), dict) else {}
    used_inputs = arch.get("used_inputs") if isinstance(arch.get("used_inputs"), list) and arch.get("used_inputs") else []
    branches = arch.get("branches") if isinstance(arch.get("branches"), list) and arch.get("branches") else []
    output_heads = arch.get("output_heads") if isinstance(arch.get("output_heads"), list) and arch.get("output_heads") else []

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
    for branch in branches:
        x = base
        layers = branch.get("layers") if isinstance(branch.get("layers"), list) else []
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            layer_type = str(layer.get("type", "Dense"))
            params = layer.get("params") if isinstance(layer.get("params"), dict) else {}
            if layer_type == "Dense":
                units = int(params.get("units", 32) or 32)
                activation = str(params.get("activation", "relu"))
                x = tf.keras.layers.Dense(units, activation=activation)(x)
        branch_outputs.append(x)

    if len(branch_outputs) == 1:
        merged = branch_outputs[0]
    else:
        merged = tf.keras.layers.Concatenate()(branch_outputs)

    outputs = []
    for head in output_heads:
        out_name = str(head.get("output_layer_name", "output_main"))
        units = 1
        if isinstance(output_units_map, dict) and out_name in output_units_map:
            units = max(1, int(output_units_map[out_name]))
        outputs.append(tf.keras.layers.Dense(units, activation="linear", name=out_name)(merged))

    model = tf.keras.Model(inputs=list(input_layers.values()), outputs=outputs if len(outputs) > 1 else outputs[0])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model, list(input_layers.keys()), [getattr(o, "name", "output") for o in (outputs if outputs else [model.output])]


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
        y_data = {name.split(":")[0]: np.random.randn(n, 1).astype("float32") for name in output_names}

    history = model.fit(x_data, y_data, epochs=1, batch_size=batch_size, verbose=0)
    tf.keras.backend.clear_session()
    loss = float(history.history.get("loss", [0.0])[-1]) if history.history else 0.0
    mae = float(history.history.get("mae", [0.0])[-1]) if history.history else 0.0
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
    input_cfg = exp.get("input_features_config") if isinstance(exp.get("input_features_config"), list) else []
    output_cfg = exp.get("output_targets_config") if isinstance(exp.get("output_targets_config"), list) else []
    data_paths = exp.get("data_paths") if isinstance(exp.get("data_paths"), dict) else {}

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

    arch = model_definition_full.get("architecture_definition") if isinstance(model_definition_full.get("architecture_definition"), dict) else {}
    used_inputs = arch.get("used_inputs") if isinstance(arch.get("used_inputs"), list) else []
    output_heads = arch.get("output_heads") if isinstance(arch.get("output_heads"), list) else []

    output_cfg_map: dict[str, int] = {}
    for item in output_cfg:
        if not isinstance(item, dict):
            continue
        target_name = str(item.get("target_name", "")).strip()
        layer_name = str(item.get("default_output_layer_name", "")).strip()
        cols = max(1, int(item.get("total_columns", 1) or 1))
        if target_name:
            output_cfg_map[target_name] = cols
        if layer_name:
            output_cfg_map[layer_name] = cols

    for inp in used_inputs:
        if not isinstance(inp, dict):
            continue
        input_name = str(inp.get("input_layer_name", "")).strip()
        source_feature_name = str(inp.get("source_feature_name", input_name)).strip()
        arr = all_data.get(source_feature_name)
        if input_name and isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[1] > 0:
            input_dims_map[input_name] = int(arr.shape[1])

    for head in output_heads:
        if not isinstance(head, dict):
            continue
        maps_to = str(head.get("maps_to_target_config_name", "")).strip()
        out_name = str(head.get("output_layer_name", "")).strip()
        if out_name == "":
            continue
        units = output_cfg_map.get(maps_to) or output_cfg_map.get(out_name)
        if units is None:
            target_name = maps_to or out_name
            arr = all_data.get(target_name)
            if isinstance(arr, np.ndarray) and arr.ndim >= 2 and arr.shape[1] > 0:
                units = int(arr.shape[1])
        output_units_map[out_name] = max(1, int(units or 1))

    model, input_names, _ = build_keras_model(
        model_definition_full,
        feature_dim=16,
        input_dims_map=input_dims_map,
        output_units_map=output_units_map,
    )

    x_data: dict[str, np.ndarray] = {}
    y_list: list[np.ndarray] = []

    for idx, input_name in enumerate(input_names):
        source_feature_name = input_name
        if idx < len(used_inputs) and isinstance(used_inputs[idx], dict):
            source_feature_name = str(used_inputs[idx].get("source_feature_name", input_name))
        arr = all_data.get(source_feature_name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise RuntimeError(f"missing real input data for feature: {source_feature_name}")
        x_data[input_name] = _sanitize_real_array(arr, f"input:{source_feature_name}")

    for head in output_heads:
        if not isinstance(head, dict):
            continue
        maps_to = str(head.get("maps_to_target_config_name", "")).strip()
        out_name = str(head.get("output_layer_name", "")).strip()
        target_name = maps_to or out_name
        arr = all_data.get(target_name)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            raise RuntimeError(f"missing real target data for target: {target_name}")
        y_list.append(_sanitize_real_array(arr, f"target:{target_name}"))

    if not y_list:
        raise RuntimeError("no real targets found for output_heads")

    lengths = [arr.shape[0] for arr in list(x_data.values()) + y_list if isinstance(arr, np.ndarray) and arr.ndim >= 2]
    if not lengths:
        raise RuntimeError("real data arrays are empty")
    n = min(min(lengths), max(8, int(max_rows)))

    x_fit = {k: v[:n] for k, v in x_data.items()}
    if len(y_list) == 1 and len(model.outputs) == 1:
        y_fit: Any = y_list[0][:n]
    else:
        y_fit = [arr[:n] for arr in y_list]

    history = model.fit(x_fit, y_fit, epochs=1, batch_size=batch_size, verbose=0)
    tf.keras.backend.clear_session()
    del raw
    del all_data
    gc.collect()
    loss = float(history.history.get("loss", [0.0])[-1]) if history.history else 0.0
    mae = float(history.history.get("mae", [0.0])[-1]) if history.history else 0.0
    return {"loss": loss, "mae": mae, "samples": n}
