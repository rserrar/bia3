from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config(repo_root: Path) -> Path:
    env_cfg = os.getenv("V3_LLM_EXPERIMENT_CONFIG_FILE", "").strip()
    candidates = []
    if env_cfg:
        candidates.append(Path(env_cfg))
    candidates.append(Path("config/experiment_config.drive_runtime.json"))
    candidates.append(Path("config/experiment_config.json"))
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else (repo_root / candidate).resolve()
        if path.exists():
            return path
    raise FileNotFoundError("No experiment config found.")


def _resolve_data_dir(repo_root: Path, cfg: dict) -> Path:
    env_data_dir = os.getenv("V3_DATA_DIR", "").strip()
    data_dir_str = env_data_dir if env_data_dir else str(cfg.get("data_dir", "data"))
    path = Path(data_dir_str)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _scan_csv_minmax(csv_path: Path, chunksize: int) -> dict:
    total_rows = 0
    total_cols = None
    global_min = np.inf
    global_max = -np.inf
    finite_count = 0
    nan_count = 0
    posinf_count = 0
    neginf_count = 0

    for chunk in pd.read_csv(csv_path, header=None, chunksize=chunksize):
        values = chunk.to_numpy(dtype=np.float64, copy=False)
        if values.ndim != 2:
            continue

        rows, cols = values.shape
        total_rows += int(rows)
        if total_cols is None:
            total_cols = int(cols)

        finite_mask = np.isfinite(values)
        finite_vals = values[finite_mask]
        if finite_vals.size > 0:
            chunk_min = float(np.min(finite_vals))
            chunk_max = float(np.max(finite_vals))
            if chunk_min < global_min:
                global_min = chunk_min
            if chunk_max > global_max:
                global_max = chunk_max
            finite_count += int(finite_vals.size)

        nan_count += int(np.isnan(values).sum())
        posinf_count += int(np.isposinf(values).sum())
        neginf_count += int(np.isneginf(values).sum())

    return {
        "file": str(csv_path),
        "rows": total_rows,
        "cols": int(total_cols or 0),
        "finite_min": None if not np.isfinite(global_min) else global_min,
        "finite_max": None if not np.isfinite(global_max) else global_max,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check min/max values in runtime CSV files.")
    parser.add_argument("--key", type=str, default="", help="Optional data_paths key (example: entrada_valors)")
    parser.add_argument("--file", type=str, default="", help="Optional direct CSV path")
    parser.add_argument("--chunksize", type=int, default=20000, help="Rows per chunk")
    args = parser.parse_args()

    repo_root = _repo_root()
    cfg_path = _resolve_config(repo_root)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Invalid experiment config.")

    data_dir = _resolve_data_dir(repo_root, cfg)
    data_paths = cfg.get("data_paths") if isinstance(cfg.get("data_paths"), dict) else {}

    targets: list[tuple[str, Path]] = []
    if args.file.strip():
        raw = Path(args.file.strip())
        csv_path = raw if raw.is_absolute() else (repo_root / raw).resolve()
        targets.append(("direct", csv_path))
    elif args.key.strip():
        key = args.key.strip()
        rel = str(data_paths.get(key, "")).strip()
        if rel == "":
            raise KeyError(f"data_paths key not found: {key}")
        targets.append((key, (data_dir / rel).resolve()))
    else:
        for key, rel in data_paths.items():
            if isinstance(rel, str) and rel.lower().endswith(".csv"):
                targets.append((str(key), (data_dir / rel).resolve()))

    results = []
    missing = []
    for key, csv_path in targets:
        if not csv_path.exists():
            missing.append({"key": key, "file": str(csv_path)})
            continue
        stats = _scan_csv_minmax(csv_path, chunksize=max(1000, int(args.chunksize)))
        stats["key"] = key
        results.append(stats)

    print(
        json.dumps(
            {
                "ok": len(missing) == 0,
                "config_file": str(cfg_path),
                "data_dir": str(data_dir),
                "checked": len(results),
                "missing": missing,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if missing:
        raise FileNotFoundError("Some CSV files were not found.")


if __name__ == "__main__":
    main()
