from uuid import uuid4


def execute_train_model(payload: dict) -> dict:
    candidate_id = str(payload.get("candidate_id", "")).strip() or "unknown"
    model_id = f"mdl_{uuid4().hex[:12]}"
    return {
        "status": "completed",
        "model_id": model_id,
        "training_kpis": {
            "val_loss": 0.1,
            "val_mae": 0.03,
            "epochs": 3,
        },
        "artifact_path": f"storage/artifacts/models/{model_id}/trained/model.weights.h5",
        "checkpoint_path": f"storage/artifacts/models/{model_id}/checkpoints/last.weights.h5",
        "candidate_id": candidate_id,
        "revision": 1,
    }
