from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.worker.executors.model_runtime import build_keras_model, run_smoke_fit


def _base_model(
    model_id: str,
    used_inputs: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    output_heads: list[dict[str, Any]],
    merges: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "architecture_definition": {
            "used_inputs": used_inputs,
            "branches": branches,
            "merges": merges or [],
            "output_heads": output_heads,
        },
        "training_config": {
            "compile": {
                "optimizer": {"type": "Adam"},
            }
        },
    }


def _case_dense_and_activations() -> dict[str, Any]:
    return _base_model(
        "case_dense_activations",
        [{"input_layer_name": "in_main", "source_feature_name": "in_main", "shape": [16]}],
        [
            {
                "name": "b_main",
                "input_source_layer": "in_main",
                "layers": [
                    {"type": "GaussianNoise", "stddev": 0.01, "name": "noise1"},
                    {"type": "Dense", "units": 32, "activation": "relu", "name": "d1"},
                    {"type": "BatchNormalization", "name": "bn1"},
                    {"type": "LayerNormalization", "name": "ln1"},
                    {"type": "Dropout", "rate": 0.1, "name": "drop1"},
                    {"type": "LeakyReLU", "alpha": 0.2, "name": "lrelu1"},
                    {"type": "PReLU", "name": "prelu1"},
                    {"type": "ELU", "alpha": 1.0, "name": "elu1"},
                    {"type": "ReLU", "name": "relu1"},
                    {"type": "Softmax", "name": "soft1"},
                ],
                "output_feature_map_name": "feat_main",
            }
        ],
        [{"output_layer_name": "out_main", "source_feature_map": "feat_main", "units": 3}],
    )


def _case_conv_pooling() -> dict[str, Any]:
    return _base_model(
        "case_conv_pooling",
        [{"input_layer_name": "in_seq", "source_feature_name": "in_seq", "shape": [32]}],
        [
            {
                "name": "b_conv",
                "input_source_layer": "in_seq",
                "layers": [
                    {"type": "Reshape", "target_shape": [32, 1], "name": "r1"},
                    {"type": "Conv1D", "filters": 8, "kernel_size": 3, "padding": "same", "name": "c1"},
                    {"type": "SeparableConv1D", "filters": 8, "kernel_size": 3, "padding": "same", "name": "sc1"},
                    {"type": "SpatialDropout1D", "rate": 0.1, "name": "sd1"},
                    {"type": "MaxPooling1D", "pool_size": 2, "strides": 2, "name": "mp1"},
                    {"type": "AveragePooling1D", "pool_size": 2, "strides": 2, "name": "ap1"},
                    {"type": "GlobalMaxPooling1D", "name": "gmp1"},
                ],
                "output_feature_map_name": "feat_conv",
            }
        ],
        [{"output_layer_name": "out_conv", "source_feature_map": "feat_conv", "units": 1}],
    )


def _case_recurrent_wrappers() -> dict[str, Any]:
    return _base_model(
        "case_recurrent_wrappers",
        [{"input_layer_name": "in_seq", "source_feature_name": "in_seq", "shape": [20, 8]}],
        [
            {
                "name": "b_rec",
                "input_source_layer": "in_seq",
                "layers": [
                    {"type": "Masking", "mask_value": 0.0, "name": "mask1"},
                    {
                        "type": "Bidirectional",
                        "name": "bi1",
                        "wrapped_layer": {"type": "LSTM", "units": 16, "return_sequences": True},
                    },
                    {
                        "type": "TimeDistributed",
                        "name": "td1",
                        "wrapped_layer": {"type": "Dense", "units": 12, "activation": "relu"},
                    },
                    {"type": "GRU", "units": 10, "return_sequences": True, "name": "gru1"},
                    {"type": "GlobalAveragePooling1D", "name": "gap1"},
                    {"type": "RepeatVector", "n": 3, "name": "rep1"},
                    {"type": "Flatten", "name": "flat1"},
                ],
                "output_feature_map_name": "feat_rec",
            }
        ],
        [{"output_layer_name": "out_rec", "source_feature_map": "feat_rec", "units": 1}],
    )


def _case_lambda_slice() -> dict[str, Any]:
    return _base_model(
        "case_lambda_slice",
        [{"input_layer_name": "in_slice", "source_feature_name": "in_slice", "shape": [40, 4]}],
        [
            {
                "name": "b_slice",
                "input_source_layer": "in_slice",
                "layers": [
                    {
                        "type": "LambdaSlice",
                        "name": "slice1",
                        "slice_params": {"axis": 1, "start": 0, "end": 10, "step": None},
                    },
                    {"type": "GlobalAveragePooling1D", "name": "gap1"},
                ],
                "output_feature_map_name": "feat_slice",
            }
        ],
        [{"output_layer_name": "out_slice", "source_feature_map": "feat_slice", "units": 1}],
    )


def _case_multi_input_layers() -> dict[str, Any]:
    return _base_model(
        "case_multi_input_layers",
        [
            {"input_layer_name": "in_a", "source_feature_name": "in_a", "shape": [8]},
            {"input_layer_name": "in_b", "source_feature_name": "in_b", "shape": [8]},
        ],
        [
            {
                "name": "b_mix",
                "input_source_layer": "in_a",
                "layers": [
                    {"type": "Add", "name": "add1", "input_source_feature_maps": ["in_a", "in_b"]},
                    {"type": "Multiply", "name": "mul1", "input_source_feature_maps": ["in_a", "in_b"]},
                    {
                        "type": "Concatenate",
                        "name": "cat1",
                        "input_source_feature_maps": ["add1", "mul1"],
                    },
                    {"type": "Dense", "units": 8, "activation": "relu", "name": "d1"},
                ],
                "output_feature_map_name": "feat_mix",
            }
        ],
        [{"output_layer_name": "out_mix", "source_feature_map": "feat_mix", "units": 1}],
    )


def _case_attention_keras() -> dict[str, Any]:
    return _base_model(
        "case_attention_keras",
        [
            {"input_layer_name": "q", "source_feature_name": "q", "shape": [12, 8]},
            {"input_layer_name": "v", "source_feature_name": "v", "shape": [12, 8]},
        ],
        [
            {
                "name": "b_att",
                "input_source_layer": "q",
                "layers": [
                    {
                        "type": "AttentionKeras",
                        "name": "att1",
                        "input_source_feature_maps": ["q", "v"],
                        "params": {"dropout": 0.0, "use_causal_mask": False},
                    },
                    {"type": "GlobalAveragePooling1D", "name": "gap1"},
                ],
                "output_feature_map_name": "feat_att",
            }
        ],
        [{"output_layer_name": "out_att", "source_feature_map": "feat_att", "units": 1}],
    )


def _case_mha_keras() -> dict[str, Any]:
    return _base_model(
        "case_mha_keras",
        [
            {"input_layer_name": "q", "source_feature_name": "q", "shape": [12, 8]},
            {"input_layer_name": "v", "source_feature_name": "v", "shape": [12, 8]},
        ],
        [
            {
                "name": "b_mha",
                "input_source_layer": "q",
                "layers": [
                    {
                        "type": "MultiHeadAttentionKeras",
                        "name": "mha1",
                        "input_source_feature_maps": ["q", "v"],
                        "constructor_params": {"num_heads": 2, "key_dim": 4, "dropout": 0.0},
                        "call_params": {"use_causal_mask": False},
                    },
                    {"type": "GlobalAveragePooling1D", "name": "gap1"},
                ],
                "output_feature_map_name": "feat_mha",
            }
        ],
        [{"output_layer_name": "out_mha", "source_feature_map": "feat_mha", "units": 1}],
    )


def _case_merge_concat() -> dict[str, Any]:
    return _base_model(
        "case_merge_concat",
        [
            {"input_layer_name": "in_a", "source_feature_name": "in_a", "shape": [8]},
            {"input_layer_name": "in_b", "source_feature_name": "in_b", "shape": [8]},
        ],
        [
            {
                "name": "b1",
                "input_source_layer": "in_a",
                "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "b1d"}],
                "output_feature_map_name": "f1",
            },
            {
                "name": "b2",
                "input_source_layer": "in_b",
                "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "b2d"}],
                "output_feature_map_name": "f2",
            },
        ],
        [{"output_layer_name": "out_merge_c", "source_feature_map": "merged_c", "units": 1}],
        merges=[
            {
                "name": "merge_c",
                "type": "concatenate",
                "source_feature_maps": ["f1", "f2"],
                "layers_after_merge": [{"type": "Dense", "units": 10, "activation": "relu", "name": "md1"}],
                "output_feature_map_name": "merged_c",
            }
        ],
    )


def _case_merge_add() -> dict[str, Any]:
    return _base_model(
        "case_merge_add",
        [
            {"input_layer_name": "in_a", "source_feature_name": "in_a", "shape": [8]},
            {"input_layer_name": "in_b", "source_feature_name": "in_b", "shape": [8]},
        ],
        [
            {
                "name": "b1",
                "input_source_layer": "in_a",
                "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "b1d"}],
                "output_feature_map_name": "f1",
            },
            {
                "name": "b2",
                "input_source_layer": "in_b",
                "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "b2d"}],
                "output_feature_map_name": "f2",
            },
        ],
        [{"output_layer_name": "out_merge_a", "source_feature_map": "merged_a", "units": 1}],
        merges=[
            {
                "name": "merge_a",
                "type": "add",
                "source_feature_maps": ["f1", "f2"],
                "layers_after_merge": [{"type": "Dense", "units": 10, "activation": "relu", "name": "md1"}],
                "output_feature_map_name": "merged_a",
            }
        ],
    )


def _case_merge_multiply() -> dict[str, Any]:
    return _base_model(
        "case_merge_multiply",
        [
            {"input_layer_name": "in_a", "source_feature_name": "in_a", "shape": [8]},
            {"input_layer_name": "in_b", "source_feature_name": "in_b", "shape": [8]},
        ],
        [
            {
                "name": "b1",
                "input_source_layer": "in_a",
                "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "b1d"}],
                "output_feature_map_name": "f1",
            },
            {
                "name": "b2",
                "input_source_layer": "in_b",
                "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "b2d"}],
                "output_feature_map_name": "f2",
            },
        ],
        [{"output_layer_name": "out_merge_m", "source_feature_map": "merged_m", "units": 1}],
        merges=[
            {
                "name": "merge_m",
                "type": "multiply",
                "source_feature_maps": ["f1", "f2"],
                "layers_after_merge": [{"type": "Dense", "units": 10, "activation": "relu", "name": "md1"}],
                "output_feature_map_name": "merged_m",
            }
        ],
    )


def _case_unsupported_layer() -> dict[str, Any]:
    return _base_model(
        "case_unsupported",
        [{"input_layer_name": "in1", "source_feature_name": "in1", "shape": [8]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in1",
                "layers": [{"type": "NotSupportedLayer", "name": "bad1"}],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_gru_missing_units() -> dict[str, Any]:
    return _base_model(
        "case_gru_missing_units",
        [{"input_layer_name": "in_seq", "source_feature_name": "in_seq", "shape": [12, 4]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in_seq",
                "layers": [{"type": "GRU", "name": "gru_missing_units"}],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_conv1d_missing_kernel_size() -> dict[str, Any]:
    return _base_model(
        "case_conv1d_missing_kernel_size",
        [{"input_layer_name": "in_seq", "source_feature_name": "in_seq", "shape": [16, 2]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in_seq",
                "layers": [{"type": "Conv1D", "filters": 8, "name": "conv_missing_ks"}],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_bidirectional_missing_wrapped() -> dict[str, Any]:
    return _base_model(
        "case_bidirectional_missing_wrapped",
        [{"input_layer_name": "in_seq", "source_feature_name": "in_seq", "shape": [12, 4]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in_seq",
                "layers": [{"type": "Bidirectional", "name": "bi_missing_wrapped"}],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_time_distributed_invalid_inner() -> dict[str, Any]:
    return _base_model(
        "case_time_distributed_invalid_inner",
        [{"input_layer_name": "in_seq", "source_feature_name": "in_seq", "shape": [12, 4]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in_seq",
                "layers": [
                    {
                        "type": "TimeDistributed",
                        "name": "td_invalid_inner",
                        "wrapped_layer": {"type": "Conv1D", "filters": 8, "kernel_size": 3},
                    }
                ],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_add_single_tensor() -> dict[str, Any]:
    return _base_model(
        "case_add_single_tensor",
        [{"input_layer_name": "in1", "source_feature_name": "in1", "shape": [8]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in1",
                "layers": [{"type": "Add", "name": "add_bad", "input_source_feature_maps": ["in1"]}],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_concat_missing_sources() -> dict[str, Any]:
    return _base_model(
        "case_concat_missing_sources",
        [{"input_layer_name": "in1", "source_feature_name": "in1", "shape": [8]}],
        [
            {
                "name": "b1",
                "input_source_layer": "in1",
                "layers": [
                    {
                        "type": "Concatenate",
                        "name": "cat_bad",
                        "input_source_feature_maps": ["does_not_exist_a", "does_not_exist_b"],
                    }
                ],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_attention_insufficient_inputs() -> dict[str, Any]:
    return _base_model(
        "case_attention_insufficient_inputs",
        [{"input_layer_name": "q", "source_feature_name": "q", "shape": [12, 8]}],
        [
            {
                "name": "b1",
                "input_source_layer": "q",
                "layers": [{"type": "AttentionKeras", "name": "att_bad", "input_source_feature_maps": ["q"]}],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_attention_bad_rank() -> dict[str, Any]:
    return _base_model(
        "case_attention_bad_rank",
        [
            {"input_layer_name": "q", "source_feature_name": "q", "shape": [8]},
            {"input_layer_name": "v", "source_feature_name": "v", "shape": [8]},
        ],
        [
            {
                "name": "b1",
                "input_source_layer": "q",
                "layers": [
                    {
                        "type": "AttentionKeras",
                        "name": "att_rank_bad",
                        "input_source_feature_maps": ["q", "v"],
                    }
                ],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_mha_insufficient_inputs() -> dict[str, Any]:
    return _base_model(
        "case_mha_insufficient_inputs",
        [{"input_layer_name": "q", "source_feature_name": "q", "shape": [12, 8]}],
        [
            {
                "name": "b1",
                "input_source_layer": "q",
                "layers": [
                    {
                        "type": "MultiHeadAttentionKeras",
                        "name": "mha_bad",
                        "input_source_feature_maps": [],
                        "constructor_params": {"num_heads": 2, "key_dim": 4},
                    }
                ],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def _case_mha_bad_rank() -> dict[str, Any]:
    return _base_model(
        "case_mha_bad_rank",
        [
            {"input_layer_name": "q", "source_feature_name": "q", "shape": [8]},
            {"input_layer_name": "v", "source_feature_name": "v", "shape": [8]},
        ],
        [
            {
                "name": "b1",
                "input_source_layer": "q",
                "layers": [
                    {
                        "type": "MultiHeadAttentionKeras",
                        "name": "mha_rank_bad",
                        "input_source_feature_maps": ["q", "v"],
                        "constructor_params": {"num_heads": 2, "key_dim": 4},
                    }
                ],
                "output_feature_map_name": "f1",
            }
        ],
        [{"output_layer_name": "out1", "source_feature_map": "f1", "units": 1}],
    )


def main() -> None:
    cases: list[tuple[str, dict[str, Any], bool]] = [
        ("dense_and_activations", _case_dense_and_activations(), False),
        ("conv_pooling", _case_conv_pooling(), False),
        ("recurrent_wrappers", _case_recurrent_wrappers(), False),
        ("lambda_slice", _case_lambda_slice(), False),
        ("multi_input_layers", _case_multi_input_layers(), False),
        ("attention_keras", _case_attention_keras(), False),
        ("mha_keras", _case_mha_keras(), False),
        ("merge_concat", _case_merge_concat(), False),
        ("merge_add", _case_merge_add(), False),
        ("merge_multiply", _case_merge_multiply(), False),
        ("unsupported_layer", _case_unsupported_layer(), True),
        ("gru_missing_units", _case_gru_missing_units(), True),
        ("conv1d_missing_kernel_size", _case_conv1d_missing_kernel_size(), True),
        ("bidirectional_missing_wrapped", _case_bidirectional_missing_wrapped(), True),
        ("time_distributed_invalid_inner", _case_time_distributed_invalid_inner(), True),
        ("add_single_tensor", _case_add_single_tensor(), True),
        ("concat_missing_sources", _case_concat_missing_sources(), True),
        ("attention_insufficient_inputs", _case_attention_insufficient_inputs(), True),
        ("attention_bad_rank", _case_attention_bad_rank(), True),
        ("mha_insufficient_inputs", _case_mha_insufficient_inputs(), True),
        ("mha_bad_rank", _case_mha_bad_rank(), True),
    ]

    results: list[dict[str, Any]] = []
    ok = True
    for name, model_def, expect_error in cases:
        row: dict[str, Any] = {
            "case": name,
            "expect_error": expect_error,
        }
        try:
            model, input_names, output_names = build_keras_model(model_def)
            if expect_error:
                row["ok"] = False
                row["error"] = "expected error but build succeeded"
                ok = False
            else:
                smoke = run_smoke_fit(model_def, smoke_batches=1, batch_size=4)
                row["ok"] = True
                row["inputs"] = len(input_names)
                row["outputs"] = len(output_names)
                row["params"] = int(model.count_params())
                row["smoke"] = smoke
        except Exception as error:
            if expect_error:
                row["ok"] = True
                row["error_type"] = type(error).__name__
                row["error"] = str(error)
            else:
                row["ok"] = False
                row["error_type"] = type(error).__name__
                row["error"] = str(error)
                ok = False
        results.append(row)

    summary = {
        "ok": ok,
        "total": len(results),
        "passed": sum(1 for item in results if bool(item.get("ok"))),
        "failed": sum(1 for item in results if not bool(item.get("ok"))),
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
