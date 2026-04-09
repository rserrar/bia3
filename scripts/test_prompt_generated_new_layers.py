from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.settings import load_settings
from src.worker.executors.generate import execute_generate_candidate
from src.worker.executors.validate import execute_validate_candidate


NEW_LAYER_TYPES = {
    "GRU",
    "Bidirectional",
    "AveragePooling1D",
    "RepeatVector",
    "TimeDistributed",
    "GaussianNoise",
    "LeakyReLU",
    "PReLU",
    "ELU",
    "ReLU",
    "Softmax",
    "Masking",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checks prompt-generated models include new layers and pass validate")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--target-candidates", type=int, default=2)
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=16)
    parser.add_argument("--use-real-data", action="store_true", default=False)
    return parser.parse_args()


def _extract_layer_types(model_definition_full: dict[str, Any]) -> set[str]:
    arch = model_definition_full.get("architecture_definition") if isinstance(model_definition_full.get("architecture_definition"), dict) else {}
    out: set[str] = set()
    for branch in arch.get("branches", []) if isinstance(arch.get("branches"), list) else []:
        if not isinstance(branch, dict):
            continue
        for layer in branch.get("layers", []) if isinstance(branch.get("layers"), list) else []:
            if not isinstance(layer, dict):
                continue
            kind = str(layer.get("type", layer.get("layer_type", ""))).strip()
            if kind:
                out.add(kind)
    for merge in arch.get("merges", []) if isinstance(arch.get("merges"), list) else []:
        if not isinstance(merge, dict):
            continue
        for layer in merge.get("layers_after_merge", []) if isinstance(merge.get("layers_after_merge"), list) else []:
            if not isinstance(layer, dict):
                continue
            kind = str(layer.get("type", layer.get("layer_type", ""))).strip()
            if kind:
                out.add(kind)
    return out


def main() -> None:
    settings = load_settings()
    if settings.llm_mode != "openai_chat" or settings.llm_api_key.strip() == "":
        raise RuntimeError("LLM is not enabled. Set V3_LLM_MODE=openai_chat and V3_OPENAI_API_KEY.")

    args = _parse_args()
    attempts = max(1, int(args.attempts))

    attempt_rows: list[dict[str, Any]] = []
    for idx in range(attempts):
        generated = execute_generate_candidate(
            {
                "target_candidates": max(1, int(args.target_candidates)),
                "run_id": "prompt-new-layer-test",
                "generation": idx + 1,
                "code_version": "prompt-new-layer-test",
                "latest_metrics": {},
                "reference_models": [],
                "recent_generated_models": [],
            }
        )
        candidates = generated.get("candidates") if isinstance(generated.get("candidates"), list) else []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            model_full = cand.get("model_definition_full") if isinstance(cand.get("model_definition_full"), dict) else {}
            layer_types = _extract_layer_types(model_full)
            hits = sorted(layer_types & NEW_LAYER_TYPES)

            validation = execute_validate_candidate(
                {
                    "model_definition_full": model_full,
                    "smoke_batches": max(1, int(args.smoke_batches)),
                    "batch_size": max(2, int(args.batch_size)),
                    "feature_dim": max(1, int(args.feature_dim)),
                    "use_real_data": bool(args.use_real_data),
                }
            )
            report = validation.get("validation_report") if isinstance(validation.get("validation_report"), dict) else {}
            row = {
                "attempt": idx + 1,
                "candidate_id": str(cand.get("candidate_id", "")),
                "provider": str((cand.get("llm_metadata") or {}).get("provider", "")) if isinstance(cand.get("llm_metadata"), dict) else "",
                "new_layer_hits": hits,
                "schema_ok": bool(report.get("schema_ok", False)),
                "build_ok": bool(report.get("build_ok", False)),
                "compile_ok": bool(report.get("compile_ok", False)),
                "smoke_fit_ok": bool(report.get("smoke_fit_ok", False)),
                "error_type": report.get("error_type"),
                "error_message": report.get("error_message"),
            }
            attempt_rows.append(row)

            if hits and row["schema_ok"] and row["build_ok"] and row["compile_ok"] and row["smoke_fit_ok"]:
                print(json.dumps({"ok": True, "found": row, "attempts_checked": len(attempt_rows)}, ensure_ascii=False, indent=2))
                return

    print(json.dumps({"ok": False, "attempts_checked": len(attempt_rows), "results": attempt_rows}, ensure_ascii=False, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
