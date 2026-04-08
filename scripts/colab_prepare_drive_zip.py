from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "true" if default else "false").strip().lower()
    return value in {"1", "true", "yes"}


def _load_required_csvs() -> set[str]:
    raw = os.getenv("V3_REQUIRED_CSVS", "").strip()
    if raw == "":
        return {
            "entrada_valors.csv",
            "entrada_extra.csv",
            "min.csv",
            "max.csv",
            "sortida_min.csv",
            "sortida_max.csv",
            "sortida_tb.csv",
            "sortida_sl.csv",
            "sortida_sn.csv",
            "sortida_valors.csv",
        }
    return {item.strip() for item in raw.split(",") if item.strip() != ""}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    zip_path = Path(os.getenv("V3_DATA_ZIP_PATH", "/content/drive/MyDrive/b-ia/dades/borsa_min.zip")).expanduser()
    runtime_name = os.getenv("V3_DATASET_RUNTIME_NAME", "borsa_drive_runtime").strip() or "borsa_drive_runtime"
    clean_extract = _env_bool("V3_DATA_CLEAN_EXTRACT", False)

    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP no trobat: {zip_path}")

    extract_root = repo_root / "data" / "runtime_drive"
    target_dir = extract_root / runtime_name
    marker = target_dir / ".extract_complete"

    if clean_extract and target_dir.exists():
        shutil.rmtree(target_dir)

    if not marker.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        marker.write_text("ok", encoding="utf-8")

    compat_pairs = [
        ("sortida_min_7d.csv", "sortida_min.csv"),
        ("sortida_max_7d.csv", "sortida_max.csv"),
        ("sortida_valors_7d.csv", "sortida_valors.csv"),
    ]
    required_csvs = _load_required_csvs()

    candidate_dirs = [target_dir] + [p for p in target_dir.rglob("*") if p.is_dir()]
    dataset_dir = None
    for candidate in candidate_dirs:
        for src_name, dst_name in compat_pairs:
            src = candidate / src_name
            dst = candidate / dst_name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
        existing = {p.name for p in candidate.glob("*.csv")}
        if required_csvs.issubset(existing):
            dataset_dir = candidate
            break

    if dataset_dir is None:
        raise FileNotFoundError("No s'ha trobat cap carpeta amb tots els CSV requerits dins del ZIP.")

    base_cfg_path = repo_root / "config" / "experiment_config.json"
    runtime_cfg_path = repo_root / "config" / "experiment_config.drive_runtime.json"
    cfg = json.loads(base_cfg_path.read_text(encoding="utf-8"))
    cfg["data_dir"] = dataset_dir.relative_to(repo_root).as_posix()
    runtime_cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "zip_path": str(zip_path),
        "extract_root": str(target_dir),
        "dataset_dir": str(dataset_dir),
        "runtime_config": str(runtime_cfg_path),
        "data_dir_in_config": cfg["data_dir"],
        "env_to_set": {
            "V3_LLM_EXPERIMENT_CONFIG_FILE": "config/experiment_config.drive_runtime.json",
            "V3_DATA_DIR": cfg["data_dir"],
            "V3_REAL_DATA_MODE": "true",
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
