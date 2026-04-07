import time
from uuid import uuid4
from typing import Any, cast

from .executors.generate import execute_generate_candidate
from .executors.validate import execute_validate_candidate
from .executors.train import execute_train_model
from .executors.train_continue import execute_train_continue
from .client import WorkerApiClient
from src.shared.settings import load_settings


def execute_task(task: dict[str, Any]) -> dict[str, Any]:
    task_type = str(task.get("task_type", ""))
    payload_raw = task.get("payload")
    payload: dict[str, Any] = cast(dict[str, Any], payload_raw) if isinstance(payload_raw, dict) else {}

    if task_type == "generate_candidate":
        return execute_generate_candidate(payload)
    if task_type == "validate_candidate":
        return execute_validate_candidate(payload)
    if task_type == "train_model":
        return execute_train_model(payload)
    if task_type == "train_continue":
        return execute_train_continue(payload)

    return {
        "status": "failed",
        "error": {
            "error_type": "unknown_task_type",
            "error_message": f"Unsupported task_type: {task_type}",
            "retryable": False,
        },
    }


def run_worker_loop() -> None:
    settings = load_settings()
    client = WorkerApiClient(settings.api_base_url)
    worker_payload = {
        "worker_id": settings.worker_id,
        "worker_version": settings.worker_version,
        "dataset_profile_id": settings.dataset_profile_id,
        "capabilities": {
            "tasks": [
                "generate_candidate",
                "validate_candidate",
                "train_model",
                "train_continue",
            ]
        },
    }

    try:
        client.register(worker_payload)
    except Exception as error:
        print(f"[WARN] Worker register failed: {error}")

    while True:
        try:
            claim = client.claim(
                {
                    "worker_id": settings.worker_id,
                    "dataset_profile_id": settings.dataset_profile_id,
                }
            )
            action = str(claim.get("action", "wait"))
            if action != "task":
                retry_after = int(claim.get("retry_after_seconds", settings.worker_poll_seconds) or settings.worker_poll_seconds)
                time.sleep(max(1, retry_after))
                continue

            task = claim.get("task") if isinstance(claim.get("task"), dict) else None
            if task is None:
                time.sleep(settings.worker_poll_seconds)
                continue

            task_id = str(task.get("task_id", "")).strip()
            attempt = int(task.get("attempt", 1) or 1)
            if task_id == "":
                time.sleep(settings.worker_poll_seconds)
                continue

            client.start(task_id, {"worker_id": settings.worker_id})
            client.heartbeat(task_id, {"worker_id": settings.worker_id, "progress": {"phase": "started"}})

            result = execute_task(task)
            idem_key = f"{task_id}:{attempt}:{uuid4().hex[:8]}"
            if str(result.get("status", "completed")) == "failed":
                error_payload = result.get("error") if isinstance(result.get("error"), dict) else {
                    "error_type": "execution_error",
                    "error_message": "task failed",
                    "retryable": False,
                }
                client.fail(
                    task_id,
                    {
                        "worker_id": settings.worker_id,
                        "attempt": attempt,
                        "idempotency_key": idem_key,
                        "error": error_payload,
                    },
                )
            else:
                client.complete(
                    task_id,
                    {
                        "worker_id": settings.worker_id,
                        "attempt": attempt,
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                )
        except Exception as error:
            print(f"[WARN] Worker loop error: {error}")
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    run_worker_loop()
