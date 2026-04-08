from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _mb(arr: np.ndarray) -> float:
    return round(float(arr.nbytes) / (1024.0 * 1024.0), 2)


def _log(message: str) -> None:
    print(f"[data-v2] {message}", flush=True)


def load_experiment_config(experiment_config_file: str) -> dict[str, Any]:
    path = Path(experiment_config_file)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parents[3] / path).resolve()
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def load_all_raw_data_sources(
    data_paths_config: dict[str, Any],
    input_features_cfg: list[dict[str, Any]],
    output_targets_cfg: list[dict[str, Any]],
    base_data_dir: str,
    cache_dtype: str = "float32",
    use_memmap_cache: bool = True,
) -> dict[str, np.ndarray]:
    loaded_data: dict[str, np.ndarray] = {}
    keys_to_load: set[str] = set()
    total_loaded_mb = 0.0

    for feat_conf in input_features_cfg:
        source_key = str(feat_conf.get("source_csv_key", "")).strip()
        if source_key:
            keys_to_load.add(source_key)

    for target_conf in output_targets_cfg:
        source_key = str(target_conf.get("source_csv_key", "")).strip()
        if source_key:
            keys_to_load.add(source_key)

    for csv_key in keys_to_load:
        file_name = str(data_paths_config.get(csv_key, "")).strip()
        if file_name == "":
            loaded_data[csv_key] = np.array([], dtype=np.float32)
            continue
        file_path = os.path.join(base_data_dir, file_name)
        dtype_norm = "float16" if cache_dtype == "float16" else "float32"
        npy_path = file_path + (".fp16.npy" if dtype_norm == "float16" else ".npy")

        try:
            if os.path.exists(npy_path) and os.path.exists(file_path) and os.path.getmtime(npy_path) >= os.path.getmtime(file_path):
                mmap_mode = "r" if use_memmap_cache else None
                arr = np.load(npy_path, mmap_mode=mmap_mode)
                if dtype_norm == "float16" and arr.dtype != np.float16:
                    arr = np.load(npy_path).astype(np.float16)
                if dtype_norm == "float32" and arr.dtype != np.float32:
                    arr = np.load(npy_path).astype(np.float32)
                loaded_data[csv_key] = arr
                current_mb = _mb(arr)
                total_loaded_mb += current_mb
                _log(f"cache mmap carregada: {file_name}.npy dtype={arr.dtype} ~{current_mb}MB")
                continue

            arr = pd.read_csv(file_path, header=None, dtype=np.float32).values
            if dtype_norm == "float16":
                arr = arr.astype(np.float16)
            loaded_data[csv_key] = arr
            current_mb = _mb(arr)
            total_loaded_mb += current_mb
            _log(f"csv carregat: {file_name} dtype={arr.dtype} ~{current_mb}MB")
            try:
                np.save(npy_path, arr)
            except Exception:
                pass
        except Exception:
            try:
                arr = pd.read_csv(file_path, header=None).values.astype(np.float32)
                if dtype_norm == "float16":
                    arr = arr.astype(np.float16)
                loaded_data[csv_key] = arr
                total_loaded_mb += _mb(arr)
                try:
                    np.save(npy_path, arr)
                except Exception:
                    pass
            except Exception:
                loaded_data[csv_key] = np.array([], dtype=np.float32)

    _log(f"fonts carregades: {len(loaded_data)} arrays ~{round(total_loaded_mb, 2)}MB")
    return loaded_data


def derive_additional_features_and_targets(
    data_dict: dict[str, np.ndarray],
    input_features_cfg: list[dict[str, Any]],
    output_targets_cfg: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    for feat_conf in input_features_cfg:
        feature_name = str(feat_conf.get("feature_name", "")).strip()
        source_key = str(feat_conf.get("source_csv_key", "")).strip()
        if feature_name == "" or source_key == "":
            continue
        source = data_dict.get(source_key, np.array([], dtype=np.float32))
        if source.size == 0:
            data_dict[feature_name] = np.array([], dtype=np.float32)
            continue

        derive_col = feat_conf.get("derive_last_value_from_col")
        slice_params = feat_conf.get("slice_params")
        if isinstance(derive_col, int):
            if source.ndim == 2 and source.shape[1] > derive_col:
                data_dict[feature_name] = source[:, derive_col : derive_col + 1]
            else:
                data_dict[feature_name] = np.array([], dtype=np.float32)
        elif isinstance(slice_params, list) and len(slice_params) == 2:
            data_dict[feature_name] = source[:, slice(slice_params[0], slice_params[1])]
        else:
            data_dict[feature_name] = source

    for target_conf in output_targets_cfg:
        target_name = str(target_conf.get("target_name", "")).strip()
        source_key = str(target_conf.get("source_csv_key", "")).strip()
        if target_name == "" or source_key == "":
            continue
        source = data_dict.get(source_key, np.array([], dtype=np.float32))
        if source.size == 0:
            data_dict[target_name] = np.array([], dtype=np.float32)
            continue
        slice_params = target_conf.get("derive_target_slice_params")
        if isinstance(slice_params, list) and len(slice_params) == 2:
            data_dict[target_name] = source[:, slice(slice_params[0], slice_params[1])]
        else:
            data_dict[target_name] = source
    return data_dict
