from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.worker.executors.model_runtime import run_smoke_fit_real_data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-data smoke-fit for model with newly supported layers")
    parser.add_argument("--experiment-config", default="config/experiment_config.json")
    parser.add_argument("--base-data-dir", default=os.getenv("V3_DATA_DIR", "data"))
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cache-dtype", default="float32")
    parser.add_argument("--use-memmap-cache", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model = {
        "model_id": "real_data_new_layers_probe",
        "architecture_definition": {
            "used_inputs": [
                {
                    "input_layer_name": "input_prices_full_800",
                    "source_feature_name": "prices_hist_full_800",
                    "shape": [800],
                }
            ],
            "branches": [
                {
                    "name": "branch_main",
                    "input_source_layer": "input_prices_full_800",
                    "layers": [
                        {"type": "Reshape", "target_shape": [800, 1], "name": "reshape_1"},
                        {"type": "GaussianNoise", "stddev": 0.01, "name": "noise_1"},
                        {"type": "Masking", "mask_value": 0.0, "name": "mask_1"},
                        {
                            "type": "Bidirectional",
                            "name": "bi_lstm_1",
                            "wrapped_layer": {
                                "type": "LSTM",
                                "units": 12,
                                "return_sequences": False,
                            },
                        },
                        {"type": "ReLU", "name": "relu_1"},
                        {"type": "Dense", "units": 16, "activation": "relu", "name": "dense_1"},
                    ],
                    "output_feature_map_name": "main_features",
                }
            ],
            "output_heads": [
                {
                    "output_layer_name": "output_stop_loss",
                    "maps_to_target_config_name": "stop_loss_prediction",
                    "source_feature_map": "main_features",
                    "units": 1,
                }
            ],
        },
        "training_config": {
            "compile": {
                "optimizer": {"type": "Nadam"},
                "dynamic_loss_config_source": "output_targets_config",
            }
        },
    }

    experiment_config = Path(args.experiment_config)
    if not experiment_config.is_absolute():
        experiment_config = (REPO_ROOT / experiment_config).resolve()

    base_data_dir = Path(args.base_data_dir)
    if not base_data_dir.is_absolute():
        base_data_dir = (REPO_ROOT / base_data_dir).resolve()

    result = run_smoke_fit_real_data(
        model,
        experiment_config_file=str(experiment_config),
        base_data_dir=str(base_data_dir),
        max_rows=max(8, int(args.max_rows)),
        batch_size=max(2, int(args.batch_size)),
        cache_dtype=str(args.cache_dtype),
        use_memmap_cache=bool(args.use_memmap_cache),
    )

    print(
        json.dumps(
            {
                "ok": True,
                "experiment_config": str(experiment_config),
                "base_data_dir": str(base_data_dir),
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
