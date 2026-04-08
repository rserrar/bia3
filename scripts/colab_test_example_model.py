from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.worker.executors.model_runtime import build_keras_model, render_model_plot_png_base64, run_smoke_fit
from src.worker.executors.train import execute_train_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile/train smoke test for example model JSON")
    parser.add_argument("--model-path", default="models/test/example_b_08_28_translated.json")
    parser.add_argument("--smoke-batches", type=int, default=2)
    parser.add_argument("--train-smoke-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, default=16)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Model JSON must be an object: {path}")
    return raw


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT
    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (repo_root / model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_definition_full = _load_json(model_path)
    model, input_names, output_names = build_keras_model(model_definition_full, feature_dim=args.feature_dim)

    smoke = run_smoke_fit(
        model_definition_full=model_definition_full,
        smoke_batches=max(1, int(args.smoke_batches)),
        feature_dim=max(1, int(args.feature_dim)),
        batch_size=max(2, int(args.batch_size)),
    )

    train_out = execute_train_model(
        {
            "candidate_id": "colab_example_model",
            "model_definition_full": model_definition_full,
            "train_smoke_batches": max(1, int(args.train_smoke_batches)),
            "feature_dim": max(1, int(args.feature_dim)),
            "batch_size": max(2, int(args.batch_size)),
            "use_real_data": False,
        }
    )
    plot_png = render_model_plot_png_base64(model_definition_full, feature_dim=max(1, int(args.feature_dim)))

    report = {
        "ok": True,
        "model_path": str(model_path),
        "model_inputs": len(input_names),
        "model_outputs": len(output_names),
        "input_names": input_names,
        "output_names": output_names,
        "smoke_fit": smoke,
        "train_executor_status": str(train_out.get("status", "unknown")),
        "train_executor": train_out,
        "plot_png_base64_present": bool(plot_png),
        "model_params": int(model.count_params()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
