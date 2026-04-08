from __future__ import annotations

import base64
import gc
import tempfile
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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


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
        for key, value in params.items():
            merged[key] = value
        return merged
    return dict(layer)


def _as_shape_tuple(shape_value: Any) -> tuple[int, ...] | None:
    if isinstance(shape_value, int):
        return (max(1, int(shape_value)),)
    if not isinstance(shape_value, list):
        return None
    dims: list[int] = []
    for value in shape_value:
        try:
            parsed = int(value)
        except Exception:
            return None
        if parsed <= 0:
            return None
        dims.append(parsed)
    return tuple(dims) if dims else None


def _resolve_initializer(tf: Any, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return tf.keras.initializers.get(value)
        except Exception:
            return None
    if isinstance(value, dict):
        raw_type = value.get("type") or value.get("class_name")
        if raw_type is None:
            return None
        init_type = str(raw_type)
        params = {k: v for k, v in value.items() if k not in {"type", "class_name"}}
        cls = getattr(tf.keras.initializers, init_type, None)
        if cls is not None:
            try:
                return cls(**params)
            except Exception:
                return None
        try:
            return tf.keras.initializers.get(init_type)
        except Exception:
            return None
    return None


def _resolve_regularizer(tf: Any, value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    reg_type = str(value.get("type", "")).strip().lower()
    l1 = float(value.get("l1", 0.0) or 0.0)
    l2 = float(value.get("l2", 0.0) or 0.0)
    if reg_type == "l1_l2":
        return tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
    if reg_type == "l1":
        return tf.keras.regularizers.l1(l1)
    if reg_type == "l2":
        return tf.keras.regularizers.l2(l2)
    return None


def _safe_tensor_name(tensor: Any) -> str:
    name = str(getattr(tensor, "name", ""))
    return name.split(":", 1)[0] if ":" in name else name


def _build_lambda_slice(tf: Any, axis: int, start: Any, end: Any, step: Any):
    def _slice_fn(t: Any) -> Any:
        rank = len(t.shape)
        safe_axis = int(axis)
        if safe_axis < 0:
            safe_axis += rank
        safe_axis = max(0, min(rank - 1, safe_axis))
        slices = [slice(None)] * rank
        slices[safe_axis] = slice(start, end, step)
        return t[tuple(slices)]

    return tf.keras.layers.Lambda(_slice_fn)


def _get_multi_input_tensors(feature_maps: dict[str, Any], names_value: Any) -> list[Any]:
    if not isinstance(names_value, list):
        return []
    tensors: list[Any] = []
    for item in names_value:
        key = str(item).strip()
        if key == "":
            continue
        tensor = feature_maps.get(key)
        if tensor is not None:
            tensors.append(tensor)
    return tensors


def _resolve_optimizer(tf: Any, optimizer_value: Any) -> Any:
    if isinstance(optimizer_value, str) and optimizer_value.strip() != "":
        try:
            return tf.keras.optimizers.get(optimizer_value)
        except Exception:
            return tf.keras.optimizers.Adam()
    if isinstance(optimizer_value, dict):
        kind = str(optimizer_value.get("type", "Adam")).strip() or "Adam"
        params = {k: v for k, v in optimizer_value.items() if k != "type"}
        cls = getattr(tf.keras.optimizers, kind, None)
        if cls is not None:
            try:
                return cls(**params)
            except Exception:
                pass
        try:
            return tf.keras.optimizers.get(kind)
        except Exception:
            return tf.keras.optimizers.Adam()
    return tf.keras.optimizers.Adam()


def _apply_layer(
    tf: Any,
    x: Any,
    layer: dict[str, Any],
    feature_maps: dict[str, Any],
    default_activation: str = "relu",
) -> Any:
    p = _layer_params(layer)
    layer_type = _layer_kind(p)
    layer_name = str(p.get("name", "")).strip() or None

    explicit_source = str(p.get("explicit_input_source_feature_map", "")).strip()
    if explicit_source != "" and feature_maps.get(explicit_source) is not None:
        x = feature_maps[explicit_source]

    multi_tensors = _get_multi_input_tensors(feature_maps, p.get("input_source_feature_maps"))

    def _dense_kwargs() -> dict[str, Any]:
        return {
            "name": layer_name,
            "use_bias": bool(p.get("use_bias", True)),
            "kernel_initializer": _resolve_initializer(tf, p.get("kernel_initializer")),
            "bias_initializer": _resolve_initializer(tf, p.get("bias_initializer")),
            "kernel_regularizer": _resolve_regularizer(tf, p.get("kernel_regularizer")),
            "bias_regularizer": _resolve_regularizer(tf, p.get("bias_regularizer")),
            "activity_regularizer": _resolve_regularizer(tf, p.get("activity_regularizer")),
        }

    if layer_type == "Dense":
        units = int(p.get("units", 32) or 32)
        activation = str(p.get("activation", default_activation))
        kwargs = _dense_kwargs()
        return tf.keras.layers.Dense(units, activation=activation, **kwargs)(x)

    if layer_type == "Dropout":
        rate = float(p.get("rate", 0.2) or 0.2)
        return tf.keras.layers.Dropout(rate=rate, name=layer_name)(x)

    if layer_type == "SpatialDropout1D":
        rate = float(p.get("rate", 0.2) or 0.2)
        return tf.keras.layers.SpatialDropout1D(rate=rate, name=layer_name)(x)

    if layer_type == "BatchNormalization":
        kwargs = {k: v for k, v in p.items() if k not in {"type", "layer_type", "name", "params", "explicit_input_source_feature_map", "input_source_feature_maps"}}
        if layer_name is not None:
            kwargs["name"] = layer_name
        return tf.keras.layers.BatchNormalization(**kwargs)(x)

    if layer_type == "LayerNormalization":
        kwargs = {k: v for k, v in p.items() if k not in {"type", "layer_type", "name", "params", "explicit_input_source_feature_map", "input_source_feature_maps"}}
        if layer_name is not None:
            kwargs["name"] = layer_name
        return tf.keras.layers.LayerNormalization(**kwargs)(x)

    if layer_type == "Reshape":
        target_shape = _as_shape_tuple(p.get("target_shape"))
        if target_shape is None:
            target_shape = (int(p.get("units", 1) or 1),)
        return tf.keras.layers.Reshape(target_shape=target_shape, name=layer_name)(x)

    if layer_type == "Conv1D":
        kwargs = {
            "filters": int(p.get("filters", 16) or 16),
            "kernel_size": int(p.get("kernel_size", 3) or 3),
            "activation": p.get("activation", default_activation),
            "padding": str(p.get("padding", "valid")),
            "strides": int(p.get("strides", 1) or 1),
            "dilation_rate": int(p.get("dilation_rate", 1) or 1),
            "use_bias": bool(p.get("use_bias", True)),
            "name": layer_name,
            "kernel_initializer": _resolve_initializer(tf, p.get("kernel_initializer")),
            "bias_initializer": _resolve_initializer(tf, p.get("bias_initializer")),
            "kernel_regularizer": _resolve_regularizer(tf, p.get("kernel_regularizer")),
            "bias_regularizer": _resolve_regularizer(tf, p.get("bias_regularizer")),
            "activity_regularizer": _resolve_regularizer(tf, p.get("activity_regularizer")),
        }
        return tf.keras.layers.Conv1D(**kwargs)(x)

    if layer_type == "SeparableConv1D":
        kwargs = {
            "filters": int(p.get("filters", 16) or 16),
            "kernel_size": int(p.get("kernel_size", 3) or 3),
            "activation": p.get("activation", default_activation),
            "padding": str(p.get("padding", "valid")),
            "strides": int(p.get("strides", 1) or 1),
            "dilation_rate": int(p.get("dilation_rate", 1) or 1),
            "depth_multiplier": int(p.get("depth_multiplier", 1) or 1),
            "use_bias": bool(p.get("use_bias", True)),
            "name": layer_name,
            "depthwise_initializer": _resolve_initializer(tf, p.get("depthwise_initializer")),
            "pointwise_initializer": _resolve_initializer(tf, p.get("pointwise_initializer")),
            "bias_initializer": _resolve_initializer(tf, p.get("bias_initializer")),
            "depthwise_regularizer": _resolve_regularizer(tf, p.get("kernel_regularizer")),
            "pointwise_regularizer": _resolve_regularizer(tf, p.get("kernel_regularizer")),
            "bias_regularizer": _resolve_regularizer(tf, p.get("bias_regularizer")),
            "activity_regularizer": _resolve_regularizer(tf, p.get("activity_regularizer")),
        }
        return tf.keras.layers.SeparableConv1D(**kwargs)(x)

    if layer_type == "LSTM":
        kwargs = {
            "units": int(p.get("units", 32) or 32),
            "activation": str(p.get("activation", "tanh")),
            "recurrent_activation": str(p.get("recurrent_activation", "sigmoid")),
            "return_sequences": bool(p.get("return_sequences", False)),
            "use_bias": bool(p.get("use_bias", True)),
            "name": layer_name,
            "kernel_initializer": _resolve_initializer(tf, p.get("kernel_initializer")),
            "recurrent_initializer": _resolve_initializer(tf, p.get("recurrent_initializer")),
            "bias_initializer": _resolve_initializer(tf, p.get("bias_initializer")),
            "kernel_regularizer": _resolve_regularizer(tf, p.get("kernel_regularizer")),
            "bias_regularizer": _resolve_regularizer(tf, p.get("bias_regularizer")),
            "activity_regularizer": _resolve_regularizer(tf, p.get("activity_regularizer")),
        }
        return tf.keras.layers.LSTM(**kwargs)(x)

    if layer_type == "MaxPooling1D":
        kwargs = {
            "pool_size": int(p.get("pool_size", 2) or 2),
            "strides": int(p.get("strides", 2) or 2),
            "padding": str(p.get("padding", "valid")),
            "name": layer_name,
        }
        return tf.keras.layers.MaxPooling1D(**kwargs)(x)

    if layer_type == "GlobalMaxPooling1D":
        keepdims = bool(p.get("keepdims", False))
        return tf.keras.layers.GlobalMaxPooling1D(keepdims=keepdims, name=layer_name)(x)

    if layer_type == "GlobalAveragePooling1D":
        keepdims = bool(p.get("keepdims", False))
        return tf.keras.layers.GlobalAveragePooling1D(keepdims=keepdims, name=layer_name)(x)

    if layer_type == "Flatten":
        return tf.keras.layers.Flatten(name=layer_name)(x)

    if layer_type == "Activation":
        activation = str(p.get("activation_function", p.get("activation", default_activation)))
        return tf.keras.layers.Activation(activation, name=layer_name)(x)

    if layer_type == "LambdaSlice":
        slice_params = _as_dict(p.get("slice_params"))
        axis = int(slice_params.get("axis", 1) or 1)
        start = slice_params.get("start", None)
        end = slice_params.get("end", None)
        step = slice_params.get("step", None)
        layer_obj = _build_lambda_slice(tf, axis=axis, start=start, end=end, step=step)
        if layer_name:
            layer_obj._name = layer_name
        return layer_obj(x)

    if layer_type == "Add":
        if len(multi_tensors) < 2:
            return x
        return tf.keras.layers.Add(name=layer_name)(multi_tensors)

    if layer_type == "Multiply":
        if len(multi_tensors) < 2:
            return x
        return tf.keras.layers.Multiply(name=layer_name)(multi_tensors)

    if layer_type == "Concatenate":
        if len(multi_tensors) < 2:
            return x
        axis = int(p.get("axis", -1) or -1)
        return tf.keras.layers.Concatenate(axis=axis, name=layer_name)(multi_tensors)

    if layer_type == "AttentionKeras":
        tensors = multi_tensors if len(multi_tensors) >= 2 else [x, x]
        params = _as_dict(p.get("params"))
        kwargs = {}
        if "dropout" in params:
            kwargs["dropout"] = float(params.get("dropout", 0.0) or 0.0)
        if layer_name is not None:
            kwargs["name"] = layer_name
        att = tf.keras.layers.Attention(**kwargs)
        call_kwargs = {}
        if "use_causal_mask" in params:
            call_kwargs["use_causal_mask"] = bool(params.get("use_causal_mask"))
        if len(tensors) >= 3:
            return att([tensors[0], tensors[1], tensors[2]], **call_kwargs)
        return att([tensors[0], tensors[1]], **call_kwargs)

    if layer_type == "MultiHeadAttentionKeras":
        tensors = multi_tensors if len(multi_tensors) >= 1 else [x]
        constructor_params = _as_dict(p.get("constructor_params"))
        num_heads = int(constructor_params.get("num_heads", p.get("num_heads", 2)) or 2)
        key_dim = int(constructor_params.get("key_dim", p.get("key_dim", 16)) or 16)
        kwargs = {
            "num_heads": num_heads,
            "key_dim": key_dim,
            "name": layer_name,
        }
        if "value_dim" in constructor_params:
            kwargs["value_dim"] = int(constructor_params.get("value_dim") or 0)
        if "dropout" in constructor_params:
            kwargs["dropout"] = float(constructor_params.get("dropout", 0.0) or 0.0)
        if "use_bias" in constructor_params:
            kwargs["use_bias"] = bool(constructor_params.get("use_bias"))
        kernel_init = _resolve_initializer(tf, constructor_params.get("kernel_initializer"))
        if kernel_init is not None:
            kwargs["kernel_initializer"] = kernel_init
        bias_init = _resolve_initializer(tf, constructor_params.get("bias_initializer"))
        if bias_init is not None:
            kwargs["bias_initializer"] = bias_init

        layer_obj = tf.keras.layers.MultiHeadAttention(**kwargs)
        call_params = _as_dict(p.get("call_params"))
        call_kwargs = {}
        if "use_causal_mask" in call_params:
            call_kwargs["use_causal_mask"] = bool(call_params.get("use_causal_mask"))
        if "attention_mask" in call_params:
            mask_name = str(call_params.get("attention_mask", "")).strip()
            if mask_name and feature_maps.get(mask_name) is not None:
                call_kwargs["attention_mask"] = feature_maps[mask_name]
        return_scores = bool(call_params.get("return_attention_scores", False))
        if return_scores:
            call_kwargs["return_attention_scores"] = True

        if len(tensors) == 1:
            q = tensors[0]
            v = tensors[0]
            out = layer_obj(query=q, value=v, key=v, **call_kwargs)
        elif len(tensors) == 2:
            q = tensors[0]
            v = tensors[1]
            out = layer_obj(query=q, value=v, key=v, **call_kwargs)
        else:
            q = tensors[0]
            v = tensors[1]
            k = tensors[2]
            out = layer_obj(query=q, value=v, key=k, **call_kwargs)

        if return_scores and isinstance(out, tuple) and len(out) == 2:
            if layer_name:
                feature_maps[f"{layer_name}_scores"] = out[1]
            return out[0]
        return out

    return x


def build_keras_model(
    model_definition_full: dict[str, Any],
    feature_dim: int = 16,
    input_dims_map: dict[str, int] | None = None,
    output_units_map: dict[str, int] | None = None,
):
    tf = _tf()

    arch = _as_dict(model_definition_full.get("architecture_definition"))
    used_inputs = _as_list_of_dicts(arch.get("used_inputs"))
    branches = _as_list_of_dicts(arch.get("branches"))
    merges = _as_list_of_dicts(arch.get("merges"))
    shared_layers = _as_list_of_dicts(arch.get("shared_layers"))
    output_heads = _as_list_of_dicts(arch.get("output_heads"))

    if not used_inputs:
        used_inputs = [{"input_layer_name": "input_main"}]
    if not branches:
        branches = [{"layers": [{"type": "Dense", "params": {"units": 32, "activation": "relu"}}]}]
    if not output_heads:
        output_heads = [{"output_layer_name": "output_main"}]

    input_layers: dict[str, Any] = {}
    feature_maps: dict[str, Any] = {}

    for inp in used_inputs:
        p = _as_dict(inp)
        name = str(p.get("input_layer_name", "input_main")).strip() or "input_main"
        source_feature_name = str(p.get("source_feature_name", name)).strip() or name
        shape = _as_shape_tuple(p.get("shape"))
        if shape is None:
            inferred = None
            if isinstance(input_dims_map, dict):
                if name in input_dims_map:
                    inferred = max(1, int(input_dims_map[name]))
                elif source_feature_name in input_dims_map:
                    inferred = max(1, int(input_dims_map[source_feature_name]))
            if inferred is None:
                inferred = max(1, int(feature_dim))
            shape = (inferred,)
        input_tensor = tf.keras.Input(shape=shape, name=name)
        input_layers[name] = input_tensor
        feature_maps[name] = input_tensor
        if source_feature_name not in feature_maps:
            feature_maps[source_feature_name] = input_tensor

    layer_values = list(input_layers.values())
    if len(layer_values) == 1:
        merged_base = layer_values[0]
    else:
        merged_base = tf.keras.layers.Concatenate(name="concat_all_inputs")(layer_values)
    feature_maps["merged_inputs"] = merged_base

    branch_outputs = []
    for branch in branches:
        branch_p = _as_dict(branch)
        branch_name = str(branch_p.get("name", branch_p.get("branch_id", ""))).strip()
        branch_input_name = str(branch_p.get("input_source_layer", branch_p.get("input_layer_name", ""))).strip()
        x = feature_maps.get(branch_input_name)
        if x is None:
            x = input_layers.get(branch_input_name)
        if x is None:
            x = merged_base

        layers = _as_list_of_dicts(branch_p.get("layers"))
        for layer in layers:
            x = _apply_layer(tf, x, layer, feature_maps, default_activation="relu")
            layer_name = str(_layer_params(layer).get("name", "")).strip()
            if layer_name:
                feature_maps[layer_name] = x

        branch_outputs.append(x)
        output_feature_map_name = str(branch_p.get("output_feature_map_name", "")).strip()
        if output_feature_map_name:
            feature_maps[output_feature_map_name] = x
        if branch_name:
            feature_maps[branch_name] = x

    if len(branch_outputs) == 1:
        merged = branch_outputs[0]
    else:
        try:
            merged = tf.keras.layers.Concatenate(name="concat_all_branches")(branch_outputs)
        except Exception:
            merged = branch_outputs[0]
    feature_maps["merged_branches"] = merged

    for merge in merges:
        merge_p = _layer_params(merge)
        merge_name = str(merge_p.get("name", "")).strip()
        merge_type = str(merge_p.get("type", "concatenate")).strip().lower()
        source_feature_maps = merge_p.get("source_feature_maps")
        tensors = _get_multi_input_tensors(feature_maps, source_feature_maps)
        if len(tensors) == 0:
            tensors = [merged]

        if merge_type == "add":
            if len(tensors) >= 2:
                x = tf.keras.layers.Add(name=merge_name or None)(tensors)
            else:
                x = tensors[0]
        elif merge_type == "multiply":
            if len(tensors) >= 2:
                x = tf.keras.layers.Multiply(name=merge_name or None)(tensors)
            else:
                x = tensors[0]
        else:
            if len(tensors) == 1:
                x = tensors[0]
            else:
                x = tf.keras.layers.Concatenate(name=merge_name or None)(tensors)

        for layer in _as_list_of_dicts(merge_p.get("layers_after_merge")):
            x = _apply_layer(tf, x, layer, feature_maps, default_activation="relu")
            layer_name = str(_layer_params(layer).get("name", "")).strip()
            if layer_name:
                feature_maps[layer_name] = x

        output_feature_map_name = str(merge_p.get("output_feature_map_name", "")).strip()
        if merge_name:
            feature_maps[merge_name] = x
        if output_feature_map_name:
            feature_maps[output_feature_map_name] = x
        merged = x

    for layer in shared_layers:
        layer_type = _layer_kind(layer)
        if layer_type == "Concatenate":
            tensors = _get_multi_input_tensors(feature_maps, _layer_params(layer).get("input_source_feature_maps"))
            if isinstance(tensors, list):
                if len(tensors) >= 2:
                    shared_name = str(_layer_params(layer).get("name", "concat_shared"))
                    merged = tf.keras.layers.Concatenate(name=shared_name)(tensors)
                    continue
            continue
        merged = _apply_layer(tf, merged, layer, feature_maps, default_activation="relu")
        layer_name = str(_layer_params(layer).get("name", "")).strip()
        if layer_name:
            feature_maps[layer_name] = merged

    feature_maps["merged"] = merged

    outputs = []
    output_meta: list[dict[str, str]] = []
    for head in output_heads:
        head_p = _layer_params(head)
        out_name = str(head_p.get("output_layer_name", "output_main")).strip() or "output_main"
        maps_to = str(head_p.get("maps_to_target_config_name", "")).strip()
        source_feature_map = str(head_p.get("source_feature_map", "")).strip()
        units_raw = head_p.get("units")
        units = int(units_raw) if units_raw is not None else 0
        if units <= 0 and isinstance(output_units_map, dict):
            if out_name in output_units_map:
                units = max(1, int(output_units_map[out_name]))
            elif maps_to in output_units_map:
                units = max(1, int(output_units_map[maps_to]))
        if units <= 0:
            units = 1

        head_x = feature_maps.get(source_feature_map, merged) if source_feature_map else merged

        head_layers = _as_list_of_dicts(head_p.get("layers"))
        for layer in head_layers:
            head_x = _apply_layer(tf, head_x, layer, feature_maps, default_activation="linear")
            layer_name = str(_layer_params(layer).get("name", "")).strip()
            if layer_name:
                feature_maps[layer_name] = head_x

        activation = str(head_p.get("activation", "linear"))
        final_dense = tf.keras.layers.Dense(
            units,
            activation=activation,
            name=out_name,
            use_bias=bool(head_p.get("use_bias", True)),
            kernel_initializer=_resolve_initializer(tf, head_p.get("kernel_initializer")),
            bias_initializer=_resolve_initializer(tf, head_p.get("bias_initializer")),
            kernel_regularizer=_resolve_regularizer(tf, head_p.get("kernel_regularizer")),
            bias_regularizer=_resolve_regularizer(tf, head_p.get("bias_regularizer")),
            activity_regularizer=_resolve_regularizer(tf, head_p.get("activity_regularizer")),
        )(head_x)
        outputs.append(final_dense)
        output_meta.append({"output_layer_name": out_name, "maps_to_target_config_name": maps_to})

    model = tf.keras.Model(inputs=list(input_layers.values()), outputs=outputs if len(outputs) > 1 else outputs[0])

    training_config = _as_dict(model_definition_full.get("training_config"))
    compile_config = _as_dict(training_config.get("compile"))
    optimizer = _resolve_optimizer(tf, compile_config.get("optimizer", "Adam"))

    runtime_output_cfg = _as_list_of_dicts(model_definition_full.get("output_targets_config_runtime"))
    runtime_by_target: dict[str, dict[str, Any]] = {}
    runtime_by_layer: dict[str, dict[str, Any]] = {}
    for item in runtime_output_cfg:
        target_name = str(item.get("target_name", "")).strip()
        default_layer_name = str(item.get("default_output_layer_name", "")).strip()
        if target_name:
            runtime_by_target[target_name] = item
        if default_layer_name:
            runtime_by_layer[default_layer_name] = item

    if len(model.output_names) == 1:
        output_name = model.output_names[0]
        mapped = next((m for m in output_meta if m.get("output_layer_name") == output_name), {})
        maps_to = str(mapped.get("maps_to_target_config_name", "")).strip()
        cfg = runtime_by_target.get(maps_to) or runtime_by_layer.get(output_name) or {}
        loss = str(cfg.get("loss_function", "mse"))
        metrics_cfg = cfg.get("metrics", ["mae"])
        if isinstance(metrics_cfg, str):
            metrics = [metrics_cfg]
        elif isinstance(metrics_cfg, list) and metrics_cfg:
            metrics = [str(item) for item in metrics_cfg]
        else:
            metrics = ["mae"]
        loss_weight = float(cfg.get("loss_weight", 1.0) or 1.0)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weight)
    else:
        losses: dict[str, Any] = {}
        metrics_map: dict[str, list[str]] = {}
        loss_weights: dict[str, float] = {}
        for output_name in model.output_names:
            mapped = next((m for m in output_meta if m.get("output_layer_name") == output_name), {})
            maps_to = str(mapped.get("maps_to_target_config_name", "")).strip()
            cfg = runtime_by_target.get(maps_to) or runtime_by_layer.get(output_name) or {}
            losses[output_name] = str(cfg.get("loss_function", "mse"))
            metrics_cfg = cfg.get("metrics", ["mae"])
            if isinstance(metrics_cfg, str):
                metrics_map[output_name] = [metrics_cfg]
            elif isinstance(metrics_cfg, list) and metrics_cfg:
                metrics_map[output_name] = [str(item) for item in metrics_cfg]
            else:
                metrics_map[output_name] = ["mae"]
            loss_weights[output_name] = float(cfg.get("loss_weight", 1.0) or 1.0)
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics_map, loss_weights=loss_weights)

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
    x_data: dict[str, np.ndarray] = {}
    for name, tensor in zip(input_names, model.inputs):
        dims: list[int] = []
        for dim in tensor.shape[1:]:
            if dim is None:
                dims.append(max(1, int(feature_dim)))
            else:
                dims.append(max(1, int(dim)))
        if not dims:
            dims = [max(1, int(feature_dim))]
        x_data[name] = np.random.randn(n, *dims).astype("float32")

    if len(model.outputs) == 1:
        out_name = model.output_names[0]
        out_layer = model.get_layer(out_name)
        units = int(getattr(out_layer, "units", 1) or 1)
        y_data = np.random.randn(n, units).astype("float32")
    else:
        y_data_list: list[np.ndarray] = []
        for name in output_names:
            out_layer = model.get_layer(name)
            units = int(getattr(out_layer, "units", 1) or 1)
            y_data_list.append(np.random.randn(n, units).astype("float32"))
        y_data = y_data_list

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
