from .model_runtime import run_smoke_fit


def execute_validate_candidate(payload: dict) -> dict:
    model_definition_full = payload.get("model_definition_full") if isinstance(payload.get("model_definition_full"), dict) else {}
    force_fail = bool(payload.get("force_fail", False))
    compile_ok = False
    smoke_ok = False
    smoke_result = {}
    error_message = None

    if not force_fail:
        try:
            smoke_result = run_smoke_fit(
                model_definition_full=model_definition_full,
                smoke_batches=int(payload.get("smoke_batches", 3) or 3),
                feature_dim=int(payload.get("feature_dim", 16) or 16),
                batch_size=int(payload.get("batch_size", 8) or 8),
            )
            compile_ok = True
            smoke_ok = True
        except Exception as error:
            error_message = str(error)

    report = {
        "schema_ok": True,
        "compile_ok": compile_ok,
        "smoke_fit_ok": smoke_ok,
        "smoke_batches": int(payload.get("smoke_batches", 3) or 3),
        "smoke_result": smoke_result,
        "error_message": error_message,
    }
    return {
        "status": "completed",
        "validation_report": report,
    }
