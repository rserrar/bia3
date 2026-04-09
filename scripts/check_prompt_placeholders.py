from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.settings import load_settings
from src.worker.executors import llm_client
from src.worker.executors.generate import (
    _build_prompt_context_from_payload,
    _read_text_if_exists,
    _render_prompt_template,
    _template_file_for_mode,
)


def _find_unresolved(text: str) -> list[str]:
    return sorted(set(re.findall(r"\{\{\s*([^}]+?)\s*\}\}", text)))


def _sample_payload(mode: str) -> dict[str, Any]:
    common_prompt_context = {
        "available_inputs_description": "- feature_name=prices_hist_full_800 · cols=800",
        "available_outputs_description": "- target_name=stop_loss_prediction · cols=1",
        "architecture_guide_content": "Guide content",
        "genealogy_case_studies": "run_id=run_test\nobjective=improve",
        "metrics_summary": {"trained_models_count": 3, "best_val_loss": 0.45, "best_val_mae": 0.21},
        "best_models_global": [
            {
                "model_id": "M-000001",
                "candidate_id": "cand_x",
                "training_kpis": {"val_loss": 0.45, "val_mae": 0.21},
                "model_definition_summary": {"kind": "dense_baseline"},
            }
        ],
        "parent_model": {
            "model_id": "M-000001",
            "training_kpis": {"val_loss": 0.45, "val_mae": 0.21},
            "model_definition_summary": {"kind": "dense_baseline"},
        },
        "family_models": [
            {
                "model_id": "M-000002",
                "training_kpis": {"val_loss": 0.48, "val_mae": 0.23},
                "model_definition_summary": {"kind": "multi_branch"},
            }
        ],
        "family_metrics_summary": {"family_size": 1, "family_best_val_loss": 0.48},
        "improvement_focus": "Reduce val_loss while preserving stability",
        "exploration_hints": "Try one novel but robust branch",
        "buggy_model": {
            "architecture_definition": {
                "used_inputs": [{"input_layer_name": "input_a", "source_feature_name": "feat_a", "shape": [8]}],
                "branches": [
                    {
                        "name": "b1",
                        "input_source_layer": "input_a",
                        "layers": [{"type": "Dense", "units": 16, "activation": "relu", "name": "d1"}],
                        "output_feature_map_name": "b1_out",
                    }
                ],
                "output_heads": [{"output_layer_name": "out1", "source_feature_map": "b1_out", "units": 1}],
            }
        },
        "working_model_example": {
            "architecture_definition": {
                "used_inputs": [{"input_layer_name": "input_a", "source_feature_name": "feat_a", "shape": [8]}],
                "branches": [
                    {
                        "name": "ok",
                        "input_source_layer": "input_a",
                        "layers": [{"type": "Dense", "units": 8, "activation": "relu", "name": "d_ok"}],
                        "output_feature_map_name": "ok_out",
                    }
                ],
                "output_heads": [{"output_layer_name": "out1", "source_feature_map": "ok_out", "units": 1}],
            }
        },
        "validation_error": "ValueError: shape mismatch",
        "error_traceback": "Traceback:\nline 1",
    }

    return {
        "generation_mode": mode,
        "target_candidates": 2,
        "prompt_context": common_prompt_context,
    }


def _check_generate_prompt_modes() -> list[str]:
    errors: list[str] = []
    for mode in ("repair", "evolution", "exploration"):
        payload = _sample_payload(mode)
        context = _build_prompt_context_from_payload(payload)
        template_file = _template_file_for_mode(mode)
        template = _read_text_if_exists(template_file)
        if template.strip() == "":
            errors.append(f"[{mode}] template file not found or empty: {template_file}")
            continue
        rendered, unresolved = _render_prompt_template(template, context)
        if unresolved:
            errors.append(f"[{mode}] unresolved placeholders: {', '.join(unresolved)}")
        leftovers = _find_unresolved(rendered)
        if leftovers:
            errors.append(f"[{mode}] unresolved placeholders in rendered prompt: {', '.join(leftovers)}")
    return errors


def _check_repair_prompt_llm_client_path() -> list[str]:
    errors: list[str] = []
    settings = load_settings()
    template = llm_client._read_text_if_exists(settings.fix_error_prompt_file)
    if template.strip() == "":
        return [f"[repair-llm-client] fix prompt template not found: {settings.fix_error_prompt_file}"]

    experiment = llm_client._read_json_if_exists(settings.experiment_config_file)
    validation_error = "ValueError: example\n\nTraceback:\nline1"
    summary_error, traceback_error = llm_client._split_validation_error(validation_error)
    replacements = {
        "validation_error": summary_error,
        "error_traceback": traceback_error,
        "buggy_model_json": json.dumps({"a": 1}, ensure_ascii=False, indent=2),
        "working_model_example_json": llm_client._pick_working_model_example_json(),
        "available_inputs_description": llm_client._inputs_description(experiment),
        "available_outputs_description": llm_client._outputs_description(experiment),
        "architecture_guide_content": llm_client._read_text_if_exists(settings.architecture_guide_file)
        or "No architecture guide content available",
    }
    rendered = llm_client._replace_prompt_placeholders(template, replacements)
    leftovers = _find_unresolved(rendered)
    if leftovers:
        errors.append(f"[repair-llm-client] unresolved placeholders: {', '.join(leftovers)}")
    return errors


def main() -> None:
    errors: list[str] = []
    errors.extend(_check_generate_prompt_modes())
    errors.extend(_check_repair_prompt_llm_client_path())

    if errors:
        print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    print(json.dumps({"ok": True, "checks": ["generate_modes", "repair_llm_client"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
