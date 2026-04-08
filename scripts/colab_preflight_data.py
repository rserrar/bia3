from __future__ import annotations

import json
import os
from pathlib import Path

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
    raise FileNotFoundError("No s'ha trobat cap fitxer de configuracio d'experiment.")


def _resolve_data_dir(repo_root: Path, cfg: dict) -> Path:
    env_data_dir = os.getenv("V3_DATA_DIR", "").strip()
    data_dir_str = env_data_dir if env_data_dir else str(cfg.get("data_dir", "data"))
    path = Path(data_dir_str)
    return path if path.is_absolute() else (repo_root / path).resolve()


def main() -> None:
    repo_root = _repo_root()
    cfg_path = _resolve_config(repo_root)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise RuntimeError("Experiment config invalid (not object)")

    data_dir = _resolve_data_dir(repo_root, cfg)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data_dir no trobat o invalid: {data_dir}")

    data_paths = cfg.get("data_paths") if isinstance(cfg.get("data_paths"), dict) else {}
    if not data_paths:
        raise RuntimeError("data_paths buit o invalid al config")

    csv_checks = []
    missing = []
    for key, value in data_paths.items():
        if not isinstance(value, str):
            continue
        if not value.lower().endswith(".csv"):
            continue
        csv_path = (data_dir / value).resolve()
        if not csv_path.exists():
            missing.append({"key": key, "file": str(csv_path)})
            continue
        sample = pd.read_csv(csv_path, header=None, nrows=4)
        csv_checks.append(
            {
                "key": key,
                "file": str(csv_path),
                "rows_sampled": int(sample.shape[0]),
                "cols": int(sample.shape[1]),
            }
        )

    input_cfg_raw = cfg.get("input_features_config")
    output_cfg_raw = cfg.get("output_targets_config")
    input_cfg = input_cfg_raw if isinstance(input_cfg_raw, list) else []
    output_cfg = output_cfg_raw if isinstance(output_cfg_raw, list) else []

    summary = {
        "ok": len(missing) == 0,
        "repo_root": str(repo_root),
        "config_file": str(cfg_path),
        "data_dir": str(data_dir),
        "csv_checked": len(csv_checks),
        "missing_csv": missing,
        "input_features": len([x for x in input_cfg if isinstance(x, dict)]),
        "output_targets": len([x for x in output_cfg if isinstance(x, dict)]),
        "csv_preview": csv_checks[:12],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if missing:
        raise FileNotFoundError("Falten CSVs requerits al data_dir.")


if __name__ == "__main__":
    main()
