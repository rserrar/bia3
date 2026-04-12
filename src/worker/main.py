import time
import threading
from uuid import uuid4
from typing import Any, cast

from .executors.generate import execute_generate_candidate
from .executors.validate import execute_validate_candidate
from .executors.train import execute_train_model
from .executors.train_continue import execute_train_continue
from .client import WorkerApiClient
from .progress import set_reporter, clear_reporter
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
        print(f"[INFO] Worker registered: {settings.worker_id}", flush=True)
    except Exception as error:
        print(f"[WARN] Worker register failed: {error}", flush=True)

    idle_counter = 0
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
                idle_counter += 1
                if idle_counter % 6 == 0:
                    print(f"[INFO] No task, waiting {retry_after}s", flush=True)
                time.sleep(max(1, retry_after))
                continue

            idle_counter = 0

            task = claim.get("task") if isinstance(claim.get("task"), dict) else None
            if task is None:
                time.sleep(settings.worker_poll_seconds)
                continue

            task_id = str(task.get("task_id", "")).strip()
            attempt = int(task.get("attempt", 1) or 1)
            if task_id == "":
                time.sleep(settings.worker_poll_seconds)
                continue

            task_type = str(task.get("task_type", ""))
            print(f"[INFO] Claimed task {task_id} ({task_type}) attempt={attempt}", flush=True)

            print(f"[INFO] Sending start -> server task_id={task_id}", flush=True)
            client.start(task_id, {"worker_id": settings.worker_id})
            print(f"[INFO] Sending heartbeat -> server task_id={task_id}", flush=True)
            client.heartbeat(task_id, {"worker_id": settings.worker_id, "progress": {"phase": "started"}})

            stop_heartbeat = threading.Event()

            def _heartbeat_loop() -> None:
                while not stop_heartbeat.wait(settings.worker_heartbeat_seconds):
                    try:
                        client.heartbeat(task_id, {"worker_id": settings.worker_id})
                    except Exception as hb_error:
                        print(f"[WARN] Heartbeat failed task_id={task_id}: {hb_error}", flush=True)

            hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            hb_thread.start()

            def _report(progress: dict[str, Any]) -> None:
                client.progress(task_id, {"worker_id": settings.worker_id, "progress": progress})

            set_reporter(_report)
            try:
                result = execute_task(task)
            finally:
                clear_reporter()
                stop_heartbeat.set()
                hb_thread.join(timeout=2.0)

            idem_key = f"{task_id}:{attempt}:{uuid4().hex[:8]}"
            if str(result.get("status", "completed")) == "failed":
                error_payload = result.get("error") if isinstance(result.get("error"), dict) else {
                    "error_type": "execution_error",
                    "error_message": "task failed",
                    "retryable": False,
                }
                print(f"[INFO] Sending fail -> server task_id={task_id}", flush=True)
                client.fail(
                    task_id,
                    {
                        "worker_id": settings.worker_id,
                        "attempt": attempt,
                        "idempotency_key": idem_key,
                        "error": error_payload,
                    },
                )
                print(f"[WARN] Task failed {task_id} ({task_type})", flush=True)
            else:
                print(f"[INFO] Sending complete -> server task_id={task_id}", flush=True)
                client.complete(
                    task_id,
                    {
                        "worker_id": settings.worker_id,
                        "attempt": attempt,
                        "idempotency_key": idem_key,
                        "result": result,
                    },
                )
                print(f"[INFO] Task completed {task_id} ({task_type})", flush=True)
        except Exception as error:
            print(f"[WARN] Worker loop error: {error}", flush=True)
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    run_worker_loop()
