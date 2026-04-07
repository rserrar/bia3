def execute_validate_candidate(payload: dict) -> dict:
    force_fail = bool(payload.get("force_fail", False))
    report = {
        "schema_ok": True,
        "compile_ok": not force_fail,
        "smoke_fit_ok": not force_fail,
        "smoke_batches": int(payload.get("smoke_batches", 3) or 3),
    }
    return {
        "status": "completed",
        "validation_report": report,
    }
