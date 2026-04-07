from uuid import uuid4


def execute_train_continue(payload: dict) -> dict:
    parent_model_id = str(payload.get("parent_model_id", "")).strip() or "unknown_parent"
    model_id = f"mdl_{uuid4().hex[:12]}"
    return {
        "status": "completed",
        "model_id": model_id,
        "parent_model_id": parent_model_id,
        "training_kpis": {
            "val_loss": 0.095,
            "val_mae": 0.028,
            "epochs": 2,
        },
        "artifact_path": f"storage/artifacts/models/{model_id}/trained/model.weights.h5",
        "checkpoint_path": f"storage/artifacts/models/{model_id}/checkpoints/last.weights.h5",
        "revision": int(payload.get("revision", 1) or 1) + 1,
        "resumed": True,
    }
