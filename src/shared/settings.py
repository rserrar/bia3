from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    worker_id: str
    worker_version: str
    dataset_profile_id: str
    worker_poll_seconds: int
    llm_mode: str
    llm_api_key: str
    llm_model: str
    llm_endpoint: str


def load_settings() -> Settings:
    return Settings(
        api_base_url=os.getenv("V3_API_BASE_URL", "http://127.0.0.1:8090"),
        worker_id=os.getenv("V3_WORKER_ID", "worker-local-1"),
        worker_version=os.getenv("V3_WORKER_VERSION", "0.1.0"),
        dataset_profile_id=os.getenv("V3_DATASET_PROFILE_ID", "default"),
        worker_poll_seconds=int(os.getenv("V3_WORKER_POLL_SECONDS", "5")),
        llm_mode=os.getenv("V3_LLM_MODE", "off"),
        llm_api_key=os.getenv("V3_OPENAI_API_KEY", ""),
        llm_model=os.getenv("V3_OPENAI_MODEL", "gpt-4o-mini"),
        llm_endpoint=os.getenv("V3_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
    )
