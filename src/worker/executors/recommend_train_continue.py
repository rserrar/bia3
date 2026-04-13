from typing import Any

from src.shared.settings import load_settings
from .llm_client import recommend_train_continue_via_openai
from ..progress import report_progress


def execute_recommend_train_continue(payload: dict[str, Any]) -> dict[str, Any]:
    settings = load_settings()
    if settings.llm_mode == "off" or settings.llm_api_key.strip() == "":
        return {
            "status": "failed",
            "error": {
                "error_type": "llm_unavailable",
                "error_message": "LLM recommendation requires V3_LLM_MODE and API key",
                "retryable": False,
            },
        }

    report_progress({"phase": "recommend_train_continue_started", "model_id": str(payload.get("model_id", ""))})
    recommendation = recommend_train_continue_via_openai(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        endpoint=settings.llm_endpoint,
        prompt_file=settings.train_continue_recommendation_prompt_file,
        model_overview=payload.get("model_overview") if isinstance(payload.get("model_overview"), dict) else {},
        parent_overview=payload.get("parent_overview") if isinstance(payload.get("parent_overview"), dict) else {},
        champion_overview=payload.get("champion_overview") if isinstance(payload.get("champion_overview"), dict) else {},
        model_comparison_summary=payload.get("model_comparison_summary") if isinstance(payload.get("model_comparison_summary"), dict) else {},
        training_history_summary=payload.get("training_history_summary") if isinstance(payload.get("training_history_summary"), dict) else {},
        family_history_summary=payload.get("family_history_summary") if isinstance(payload.get("family_history_summary"), dict) else {},
        current_training_config=payload.get("current_training_config") if isinstance(payload.get("current_training_config"), dict) else {},
        available_training_fields=payload.get("available_training_fields") if isinstance(payload.get("available_training_fields"), dict) else {},
    )
    report_progress({
        "phase": "recommend_train_continue_completed",
        "decision": recommendation.get("decision"),
        "expected_benefit": recommendation.get("expected_benefit"),
    })
    return {
        "status": "completed",
        "recommendation": recommendation,
        "model_id": str(payload.get("model_id", "")).strip() or None,
    }
