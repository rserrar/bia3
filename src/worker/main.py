import time
import threading
import multiprocessing as mp
import tempfile
import os
import json
from uuid import uuid4
from typing import Any, cast

from .executors.generate import execute_generate_candidate
from .executors.validate import execute_validate_candidate
from .executors.train import execute_train_model
from .executors.train_continue import execute_train_continue
from .executors.recommend_train_continue import execute_recommend_train_continue
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
    if task_type == "recommend_train_continue":
        return execute_recommend_train_continue(payload)

    return {
        "status": "failed",
        "error": {
            "error_type": "unknown_task_type",
            "error_message": f"Unsupported task_type: {task_type}",
            "retryable": False,
        },
    }


def _execute_task_in_subprocess(task: dict[str, Any], result_file_path: str) -> None:
    payload: dict[str, Any]
    try:
        payload = {"ok": True, "result": execute_task(task)}
    except Exception as error:
        payload = {
            "ok": False,
            "error": {
                "error_type": "worker_subprocess_exception",
                "error_message": str(error),
                "retryable": True,
            },
        }

    try:
        with open(result_file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except Exception:
        pass


def _is_missing_real_data_failure(result: dict[str, Any]) -> bool:
    if str(result.get("status", "")).lower() != "failed":
        return False
    error = result.get("error")
    if not isinstance(error, dict):
        return False
    message = str(error.get("error_message", "")).lower()
    return "missing real input data for feature:" in message


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
                "recommend_train_continue",
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
            max_lease_lost_heartbeat_errors = 3
            lease_lost_event = threading.Event()

            def _heartbeat_loop() -> None:
                lease_lost_errors = 0
                while not stop_heartbeat.wait(settings.worker_heartbeat_seconds):
                    try:
                        client.heartbeat(task_id, {"worker_id": settings.worker_id, "progress": {"phase": "heartbeat"}})
                        lease_lost_errors = 0
                    except Exception as hb_error:
                        message = str(hb_error)
                        if "leased by another worker" in message:
                            lease_lost_errors += 1
                            if lease_lost_errors >= max_lease_lost_heartbeat_errors:
                                print(
                                    f"[WARN] Heartbeat stopped task_id={task_id}: lease lost ({lease_lost_errors} consecutive errors)",
                                    flush=True,
                                )
                                lease_lost_event.set()
                                break
                        print(f"[WARN] Heartbeat failed task_id={task_id}: {hb_error}", flush=True)

            hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            hb_thread.start()

            result: dict[str, Any] = {
                "status": "failed",
                "error": {
                    "error_type": "worker_result_missing",
                    "error_message": "worker subprocess returned no result",
                    "retryable": True,
                },
            }
            lease_lost_abort = False
            result_file = tempfile.NamedTemporaryFile(prefix="v3-worker-result-", suffix=".json", delete=False)
            result_file_path = result_file.name
            result_file.close()
            task_process = mp.Process(target=_execute_task_in_subprocess, args=(task, result_file_path), daemon=True)
            task_process.start()

            try:
                while task_process.is_alive():
                    if lease_lost_event.is_set():
                        lease_lost_abort = True
                        break
                    time.sleep(0.5)

                if lease_lost_abort:
                    if task_process.is_alive():
                        task_process.terminate()
                        task_process.join(timeout=5.0)
                    print(f"[WARN] Aborting task_id={task_id} locally after lease loss", flush=True)
                    continue

                task_process.join(timeout=2.0)
                subprocess_result: dict[str, Any] = {}
                try:
                    if os.path.isfile(result_file_path):
                        with open(result_file_path, "r", encoding="utf-8") as handle:
                            loaded = json.load(handle)
                            if isinstance(loaded, dict):
                                subprocess_result = loaded
                except Exception:
                    subprocess_result = {}

                if bool(subprocess_result.get("ok", False)):
                    raw_result = subprocess_result.get("result")
                    if isinstance(raw_result, dict):
                        result = raw_result
                else:
                    raw_error = subprocess_result.get("error")
                    error_payload = raw_error if isinstance(raw_error, dict) else {
                        "error_type": "worker_subprocess_exception",
                        "error_message": "worker subprocess failed",
                        "retryable": True,
                    }
                    result = {
                        "status": "failed",
                        "error": error_payload,
                    }
            finally:
                stop_heartbeat.set()
                hb_thread.join(timeout=2.0)
                try:
                    if os.path.isfile(result_file_path):
                        os.remove(result_file_path)
                except Exception:
                    pass

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
                if _is_missing_real_data_failure(result):
                    print(
                        f"[ERROR] Missing real training data detected on task_id={task_id}; stopping worker loop to avoid repeated bad claims",
                        flush=True,
                    )
                    break
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
