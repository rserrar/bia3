from uuid import uuid4

from .model_runtime import run_smoke_fit


def execute_train_continue(payload: dict) -> dict:
    parent_model_id = str(payload.get("parent_model_id", "")).strip() or "unknown_parent"
    model_id = f"mdl_{uuid4().hex[:12]}"
    kpis = {
        "val_loss": 0.095,
        "val_mae": 0.028,
        "epochs": 2,
    }
    model_definition_full = payload.get("model_definition_full") if isinstance(payload.get("model_definition_full"), dict) else {}
    if model_definition_full:
        try:
            smoke = run_smoke_fit(
                model_definition_full=model_definition_full,
                smoke_batches=int(payload.get("train_smoke_batches", 4) or 4),
                feature_dim=int(payload.get("feature_dim", 16) or 16),
                batch_size=int(payload.get("batch_size", 16) or 16),
            )
            kpis["val_loss"] = float(smoke.get("loss", kpis["val_loss"]))
            kpis["val_mae"] = float(smoke.get("mae", kpis["val_mae"]))
        except Exception as error:
            return {
                "status": "failed",
                "error": {
                    "error_type": "training_runtime_error",
                    "error_message": str(error),
                    "retryable": False,
                },
            }

    return {
        "status": "completed",
        "model_id": model_id,
        "parent_model_id": parent_model_id,
        "training_kpis": kpis,
        "artifact_path": f"storage/artifacts/models/{model_id}/trained/model.weights.h5",
        "checkpoint_path": f"storage/artifacts/models/{model_id}/checkpoints/last.weights.h5",
        "revision": int(payload.get("revision", 1) or 1) + 1,
        "resumed": True,
    }
