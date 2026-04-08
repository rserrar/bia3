from uuid import uuid4
import time

from src.shared.settings import load_settings
from .model_runtime import run_smoke_fit, run_smoke_fit_real_data, render_model_plot_png_base64


def execute_train_model(payload: dict) -> dict:
    started_at = time.time()
    candidate_id = str(payload.get("candidate_id", "")).strip() or "unknown"
    model_id = f"mdl_{uuid4().hex[:12]}"
    training_kpis = {
        "val_loss": 0.1,
        "val_mae": 0.03,
        "epochs": 3,
    }
    model_definition_full = payload.get("model_definition_full") if isinstance(payload.get("model_definition_full"), dict) else {}
    if model_definition_full:
        try:
            settings = load_settings()
            use_real_data = bool(payload.get("use_real_data", settings.real_data_mode))
            if use_real_data:
                smoke = run_smoke_fit_real_data(
                    model_definition_full=model_definition_full,
                    experiment_config_file=settings.experiment_config_file,
                    base_data_dir=settings.data_dir,
                    max_rows=int(payload.get("max_real_rows", settings.max_real_rows) or settings.max_real_rows),
                    batch_size=int(payload.get("batch_size", 16) or 16),
                    cache_dtype=settings.data_cache_dtype,
                    use_memmap_cache=settings.use_memmap_cache,
                )
            else:
                smoke = run_smoke_fit(
                    model_definition_full=model_definition_full,
                    smoke_batches=int(payload.get("train_smoke_batches", 5) or 5),
                    feature_dim=int(payload.get("feature_dim", 16) or 16),
                    batch_size=int(payload.get("batch_size", 16) or 16),
                )
            training_kpis["val_loss"] = float(smoke.get("loss", training_kpis["val_loss"]))
            training_kpis["val_mae"] = float(smoke.get("mae", training_kpis["val_mae"]))
        except Exception as error:
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

    result = {
        "status": "completed",
        "model_id": model_id,
        "training_kpis": training_kpis,
        "training_stats": {
            "duration_sec": duration_sec,
            "smoke_batches": int(payload.get("train_smoke_batches", 5) or 5),
            "feature_dim": int(payload.get("feature_dim", 16) or 16),
            "batch_size": int(payload.get("batch_size", 16) or 16),
        },
        "artifact_path": f"storage/artifacts/models/{model_id}/trained/model.weights.h5",
        "checkpoint_path": f"storage/artifacts/models/{model_id}/checkpoints/last.weights.h5",
        "candidate_id": candidate_id,
        "revision": 1,
    }
    if plot_png_base64:
        result["plot_model_png_base64"] = plot_png_base64
    return result
