from uuid import uuid4


def execute_generate_candidate(payload: dict) -> dict:
    target = int(payload.get("target_candidates", 1) or 1)
    candidates = []
    for _ in range(max(1, target)):
        candidate_id = f"cand_{uuid4().hex[:12]}"
        model_full = {
            "model_id": candidate_id,
            "architecture_definition": {
                "used_inputs": [{"input_layer_name": "input_main", "source_feature_name": "entrada_valors"}],
                "branches": [
                    {
                        "branch_id": "b1",
                        "layers": [
                            {"type": "Dense", "params": {"units": 64, "activation": "relu"}},
                            {"type": "Dense", "params": {"units": 32, "activation": "relu"}},
                        ],
                    }
                ],
                "output_heads": [
                    {"output_layer_name": "output_sortida_valors", "maps_to_target_config_name": "sortida_valors"}
                ],
            },
            "training_config": {"fit": {"epochs": 3, "batch_size": 32}},
        }
        model_summary = {
            "kind": "dense_baseline",
            "layers": 2,
            "params_hint": "small",
            "expected_behavior": "stable_baseline",
        }
        candidates.append(
            {
                "candidate_id": candidate_id,
                "fingerprint": uuid4().hex,
                "model_definition_full": model_full,
                "model_definition_summary": model_summary,
            }
        )
    return {"status": "completed", "candidates": candidates}
