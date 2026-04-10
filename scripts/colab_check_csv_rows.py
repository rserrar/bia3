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


def _count_rows(csv_path: Path) -> int:
    rows = 0
    with csv_path.open("rb") as handle:
        for _ in handle:
            rows += 1
    return rows


def _count_cols(csv_path: Path) -> int:
    sample = pd.read_csv(csv_path, header=None, nrows=1)
    return int(sample.shape[1])


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

    csv_rows = []
    missing = []
    for key, value in data_paths.items():
        if not isinstance(value, str) or not value.lower().endswith(".csv"):
            continue
        csv_path = (data_dir / value).resolve()
        if not csv_path.exists():
            missing.append({"key": key, "file": str(csv_path)})
            continue
        rows = _count_rows(csv_path)
        cols = _count_cols(csv_path)
        csv_rows.append({"key": key, "file": str(csv_path), "rows": rows, "cols": cols})

    row_values = [item["rows"] for item in csv_rows]
    rows_consistent = len(set(row_values)) <= 1 if row_values else False

    summary = {
        "ok": len(missing) == 0 and rows_consistent,
        "config_file": str(cfg_path),
        "data_dir": str(data_dir),
        "csv_checked": len(csv_rows),
        "missing_csv": missing,
        "rows_consistent": rows_consistent,
        "common_rows": row_values[0] if rows_consistent and row_values else None,
        "min_rows": min(row_values) if row_values else None,
        "max_rows": max(row_values) if row_values else None,
        "csv_rows": csv_rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if missing:
        raise FileNotFoundError("Falten CSVs requerits al data_dir.")
    if not rows_consistent:
        raise RuntimeError("Els CSV no tenen el mateix nombre de files.")


if __name__ == "__main__":
    main()
