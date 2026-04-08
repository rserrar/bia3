from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.worker.executors.generate import execute_generate_candidate
from src.worker.executors.validate import execute_validate_candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM generation test using current prompts and a base model")
    parser.add_argument("--base-model-path", default="models/test/example_b_08_28_translated.json")
    parser.add_argument("--experiment-config", default="config/experiment_config.json")
    parser.add_argument("--target-candidates", type=int, default=1)
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--require-llm-success", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return raw


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (repo_root / path).resolve()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT

    base_model_path = _resolve_path(repo_root, args.base_model_path)
    if not base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    base_model = _read_json(base_model_path)

    exp_path = _resolve_path(repo_root, args.experiment_config)
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    exp = _read_json(exp_path)
    input_features_config = exp.get("input_features_config") if isinstance(exp.get("input_features_config"), list) else []
    output_targets_config = exp.get("output_targets_config") if isinstance(exp.get("output_targets_config"), list) else []

    payload = {
        "target_candidates": max(1, int(args.target_candidates)),
        "run_id": "colab-llm-test",
        "generation": 1,
        "code_version": "colab-test",
        "latest_metrics": {"note": "manual_llm_test"},
        "reference_models": [
            {
                "model_id": "base_example",
                "model_definition_full": base_model,
                "last_evaluation_metrics": {"val_loss": 0.1, "val_mae": 0.03},
            }
        ],
        "recent_generated_models": [],
        "input_features_config": input_features_config,
        "output_targets_config": output_targets_config,
    }

    generated = execute_generate_candidate(payload)
    if str(generated.get("status", "")) != "completed":
        raise RuntimeError(f"Generation did not complete: {generated}")

    candidates = generated.get("candidates") if isinstance(generated.get("candidates"), list) else []
    if not candidates:
        raise RuntimeError("Generation returned no candidates")

    provider_counts: dict[str, int] = {}
    candidate_reports: list[dict[str, Any]] = []

    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        llm_metadata = _as_dict(candidate.get("llm_metadata"))
        provider = str(llm_metadata.get("provider", "unknown"))
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

        model_definition_full = candidate.get("model_definition_full") if isinstance(candidate.get("model_definition_full"), dict) else {}
        validation = execute_validate_candidate(
            {
                "model_definition_full": model_definition_full,
                "smoke_batches": max(1, int(args.smoke_batches)),
                "batch_size": max(2, int(args.batch_size)),
                "feature_dim": max(1, int(args.feature_dim)),
                "use_real_data": False,
            }
        )
        report = _as_dict(validation.get("validation_report"))
        candidate_reports.append(
            {
                "candidate_index": idx,
                "candidate_id": str(candidate.get("candidate_id", "")),
                "provider": provider,
                "schema_ok": bool(report.get("schema_ok", False)),
                "build_ok": bool(report.get("build_ok", False)),
                "compile_ok": bool(report.get("compile_ok", False)),
                "smoke_fit_ok": bool(report.get("smoke_fit_ok", False)),
                "repaired": bool(report.get("repaired", False)),
                "error_type": report.get("error_type"),
                "error_message": report.get("error_message"),
            }
        )

    llm_success = provider_counts.get("openai_chat", 0) > 0
    if args.require_llm_success and not llm_success:
        raise RuntimeError("No candidate generated by provider=openai_chat (all fallback or invalid)")

    summary = {
        "ok": True,
        "base_model_path": str(base_model_path),
        "experiment_config": str(exp_path),
        "total_candidates": len(candidate_reports),
        "provider_counts": provider_counts,
        "llm_success": llm_success,
        "candidates": candidate_reports,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
