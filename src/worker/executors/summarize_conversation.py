from typing import Any
from uuid import uuid4
from datetime import datetime

from src.shared.settings import load_settings
from .llm_client import generate_candidate_via_openai
from ..progress import report_progress


_SUMMARY_SYSTEM_PROMPT = """You are an AI research assistant. Your task is to summarize a technical conversation about model evolution decisions.

The conversation contains a history of recommend_train_continue decisions for a family of machine learning models.

Provide a concise summary (max 5 bullet points) covering:
1. What models were evaluated
2. What decisions were taken (continue/stop/retry_variant) 
3. Which decisions led to improvements (child models with better scores)
4. Which approaches were unsuccessful
5. Key patterns or lessons learned

Focus on facts and metrics. Be specific about model IDs and scores."""


def execute_summarize_conversation(payload: dict) -> dict:
    report_progress({"phase": "summarize_started"})

    conversation_id = str(payload.get("conversation_id", ""))
    entity_type = str(payload.get("entity_type", "family"))
    entity_id = str(payload.get("entity_id", ""))
    messages_raw = payload.get("messages", [])

    if not isinstance(messages_raw, list) or len(messages_raw) == 0:
        return {
            "status": "completed",
            "summary": "No messages to summarize.",
            "conversation_id": conversation_id,
            "entity_type": entity_type,
            "entity_id": entity_id,
        }

    settings = load_settings()
    if settings.llm_mode != "openai_chat" or settings.llm_api_key.strip() == "":
        return {
            "status": "failed",
            "error": {
                "error_type": "llm_unavailable",
                "error_message": "LLM not configured for summarization",
                "retryable": False,
            },
        }

    conversation_text_lines = []
    for msg in messages_raw:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        created = str(msg.get("created_at", ""))[:19]
        conversation_text_lines.append(f"[{created}] {role}: {content}")

    conversation_text = "\n".join(conversation_text_lines)

    prompt_text = (
        f"### Conversation to summarize (ID: {conversation_id})\n"
        f"Entity: {entity_type} / {entity_id}\n"
        f"Total messages: {len(messages_raw)}\n\n"
        f"### Conversation history:\n{conversation_text}\n\n"
        f"### Instructions:\n"
        f"Summarize the above conversation in max 5 bullet points. "
        f"Include: models evaluated, decisions taken, outcomes, and key lessons. "
        f"Be specific and concise."
    )

    report_progress({"phase": "summarize_llm_request"})

    try:
        response = generate_candidate_via_openai(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            prompt=prompt_text,
            endpoint=settings.llm_endpoint,
        )
    except Exception as error:
        return {
            "status": "failed",
            "error": {
                "error_type": "llm_request_error",
                "error_message": f"LLM request failed: {error}",
                "retryable": True,
            },
        }

    payload_meta = response if isinstance(response, dict) else {}
    parsed = payload_meta.get("_llm_parsed_payload")
    if parsed is None and isinstance(payload_meta, dict):
        parsed = {k: v for k, v in payload_meta.items() if not k.startswith("_llm_")}
    if isinstance(parsed, dict):
        summary_text = str(parsed.get("summary", parsed.get("content", "")))
    elif isinstance(parsed, str):
        summary_text = parsed
    else:
        raw_text = payload_meta.get("_llm_response_text", "")
        if raw_text:
            summary_text = str(raw_text)
        else:
            summary_text = "LLM response could not be parsed."

    report_progress({"phase": "summarize_completed"})

    return {
        "status": "completed",
        "summary": summary_text.strip(),
        "conversation_id": conversation_id,
        "entity_type": entity_type,
        "entity_id": entity_id,
    }
