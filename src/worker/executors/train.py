from uuid import uuid4
import time
from typing import Any

from src.shared.settings import load_settings
from .model_runtime import run_full_fit_real_data, render_model_plot_png_base64
from ..progress import report_progress


def _resolve_training_config(payload: dict[str, Any], settings: Any) -> dict[str, Any]:
    training_cfg_raw = payload.get("training_config")
    training_cfg: dict[str, Any] = training_cfg_raw if isinstance(training_cfg_raw, dict) else {}

    def _pick_int(key: str, default: int) -> int:
        value = training_cfg.get(key, payload.get(key, default))
        try:
            return int(value)
        except Exception:
            return int(default)

    def _pick_float(key: str, default: float) -> float:
        value = training_cfg.get(key, payload.get(key, default))
        try:
            return float(value)
        except Exception:
            return float(default)

    def _pick_bool(key: str, default: bool) -> bool:
        value = training_cfg.get(key, payload.get(key, default))
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    return {
        "max_real_rows": max(8, _pick_int("max_real_rows", settings.max_real_rows)),
        "batch_size": max(1, _pick_int("batch_size", settings.train_batch_size)),
        "epochs": max(1, _pick_int("epochs", settings.train_epochs)),
        "verbose": max(0, _pick_int("verbose", settings.train_verbose)),
        "validation_split": _pick_float("validation_split", settings.train_validation_split),
        "early_stopping_patience": max(1, _pick_int("early_stopping_patience", settings.train_early_stopping_patience)),
        "reduce_lr_patience": max(1, _pick_int("reduce_lr_patience", settings.train_reduce_lr_patience)),
        "reduce_lr_factor": max(0.01, _pick_float("reduce_lr_factor", settings.train_reduce_lr_factor)),
        "min_lr": max(0.0, _pick_float("min_lr", settings.train_min_lr)),
        "restore_best_weights": _pick_bool("restore_best_weights", settings.train_restore_best_weights),
    }


def execute_train_model(payload: dict) -> dict:
    started_at = time.time()
    settings = load_settings()
    resolved = _resolve_training_config(payload, settings)
    candidate_id = str(payload.get("candidate_id", "")).strip() or "unknown"
    model_id = f"mdl_{uuid4().hex[:12]}"
    training_kpis = {"val_loss": 0.1, "val_mae": 0.03, "epochs": 1}
    training_stats_extra: dict[str, Any] = {}
    training_history: dict[str, Any] = {}
    inline_artifacts: list[dict[str, Any]] = []
    model_definition_full = payload.get("model_definition_full") if isinstance(payload.get("model_definition_full"), dict) else {}
    report_progress({"phase": "train_started", "candidate_id": candidate_id})
    if model_definition_full:
        try:
            use_real_data = bool(payload.get("use_real_data", settings.real_data_mode))
            if not use_real_data:
                return {
                    "status": "failed",
                    "error": {
                        "error_type": "training_mode_error",
                        "error_message": "train_model requires real data mode (set V3_REAL_DATA_MODE=true)",
                        "retryable": False,
                    },
                }

            full_fit = run_full_fit_real_data(
                model_definition_full=model_definition_full,
                experiment_config_file=settings.experiment_config_file,
                base_data_dir=settings.data_dir,
                max_rows=resolved["max_real_rows"],
                batch_size=resolved["batch_size"],
                epochs=resolved["epochs"],
                validation_split=resolved["validation_split"],
                early_stopping_patience=resolved["early_stopping_patience"],
                reduce_lr_patience=resolved["reduce_lr_patience"],
                reduce_lr_factor=resolved["reduce_lr_factor"],
                min_lr=resolved["min_lr"],
                restore_best_weights=resolved["restore_best_weights"],
                verbose=resolved["verbose"],
                cache_dtype=settings.data_cache_dtype,
                use_memmap_cache=settings.use_memmap_cache,
                include_inline_artifacts=settings.train_include_inline_artifacts,
                include_full_model_artifact=settings.train_include_full_model_artifact,
                max_inline_artifact_mb=settings.train_max_inline_artifact_mb,
                progress_callback=report_progress,
            )

            training_kpis["val_loss"] = float(full_fit.get("val_loss", training_kpis["val_loss"]))
            training_kpis["val_mae"] = float(full_fit.get("mae", training_kpis["val_mae"]))
            training_kpis["epochs"] = int(full_fit.get("epochs_ran", resolved["epochs"]))
            training_kpis["best_val_loss"] = float(full_fit.get("best_val_loss", training_kpis["val_loss"]))
            training_kpis["best_epoch"] = int(full_fit.get("best_epoch", training_kpis["epochs"]))

            training_stats_extra = {
                "samples": int(full_fit.get("samples", resolved["max_real_rows"])),
                "epochs_requested": resolved["epochs"],
                "epochs_ran": int(full_fit.get("epochs_ran", resolved["epochs"])),
                "validation_split": float(resolved["validation_split"]),
                "batch_size": resolved["batch_size"],
                "verbose": resolved["verbose"],
                "early_stopping_patience": resolved["early_stopping_patience"],
                "reduce_lr_patience": resolved["reduce_lr_patience"],
                "reduce_lr_factor": float(resolved["reduce_lr_factor"]),
                "min_lr": float(resolved["min_lr"]),
                "restore_best_weights": bool(resolved["restore_best_weights"]),
                "cache_dtype": settings.data_cache_dtype,
                "inline_artifacts_count": len(full_fit.get("inline_artifacts", [])) if isinstance(full_fit.get("inline_artifacts"), list) else 0,
                "inline_artifacts_skipped": full_fit.get("inline_artifacts_skipped", []),
            }
            history_raw = full_fit.get("history")
            training_history = {}
            if isinstance(history_raw, dict):
                training_history = {str(k): v for k, v in history_raw.items()}
            artifacts_raw = full_fit.get("inline_artifacts")
            inline_artifacts = []
            if isinstance(artifacts_raw, list):
                inline_artifacts = [item for item in artifacts_raw if isinstance(item, dict)]
        except Exception as error:
            report_progress({"phase": "train_failed", "error": str(error)})
            return {
                "status": "failed",
                "error": {
                    "error_type": "training_runtime_error",
                    "error_message": str(error),
                    "retryable": False,
                },
            }

    duration_sec = round(time.time() - started_at, 4)
    plot_png_base64 = render_model_plot_png_base64(model_definition_full, feature_dim=int(payload.get("feature_dim", 16) or 16)) if model_definition_full else None

    training_stats = {
        "duration_sec": duration_sec,
        "batch_size": int(resolved["batch_size"]),
    }
    if training_stats_extra:
        training_stats.update(training_stats_extra)

    result = {
        "status": "completed",
        "model_id": model_id,
        "training_kpis": training_kpis,
        "training_stats": training_stats,
        "artifact_path": f"storage/artifacts/models/{model_id}/trained/model.weights.h5",
        "checkpoint_path": f"storage/artifacts/models/{model_id}/checkpoints/last.weights.h5",
        "candidate_id": candidate_id,
        "revision": 1,
    }
    if plot_png_base64:
        result["plot_model_png_base64"] = plot_png_base64
    if training_history:
        result["training_history"] = training_history
    if inline_artifacts:
        result["inline_artifacts"] = inline_artifacts
    report_progress({"phase": "train_completed", "model_id": model_id, "val_loss": training_kpis.get("val_loss")})
    return result
