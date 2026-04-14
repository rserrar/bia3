from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    api_base_url: str
    worker_id: str
    worker_version: str
    dataset_profile_id: str
    worker_poll_seconds: int
    worker_heartbeat_seconds: int
    llm_mode: str
    llm_api_key: str
    llm_model: str
    llm_endpoint: str
    prompt_template_file: str
    architecture_guide_file: str
    experiment_config_file: str
    llm_num_new_models: int
    llm_num_reference_models: int
    fix_error_prompt_file: str
    train_continue_recommendation_prompt_file: str
    real_data_mode: bool
    data_dir: str
    max_real_rows: int
    data_cache_dtype: str
    use_memmap_cache: bool
    train_epochs: int
    train_batch_size: int
    train_verbose: int
    train_validation_split: float
    train_early_stopping_patience: int
    train_reduce_lr_patience: int
    train_reduce_lr_factor: float
    train_min_lr: float
    train_restore_best_weights: bool
    train_initial_learning_rate: float
    train_optimizer: str
    train_seed: int
    train_target_metric: str
    train_target_metric_mode: str
    train_max_training_minutes: int
    train_business_metric_sl_weight: float
    train_business_metric_tb_weight: float
    train_business_min_relative_improvement: float
    train_business_improvement_window: int
    train_soft_max_epochs: int
    train_include_inline_artifacts: bool
    train_include_full_model_artifact: bool
    train_max_inline_artifact_mb: int


def load_settings() -> Settings:
    return Settings(
        api_base_url=os.getenv("V3_API_BASE_URL", "http://127.0.0.1:8090"),
        worker_id=os.getenv("V3_WORKER_ID", "worker-local-1"),
        worker_version=os.getenv("V3_WORKER_VERSION", "0.1.0"),
        dataset_profile_id=os.getenv("V3_DATASET_PROFILE_ID", "default"),
        worker_poll_seconds=int(os.getenv("V3_WORKER_POLL_SECONDS", "5")),
        worker_heartbeat_seconds=max(15, int(os.getenv("V3_WORKER_HEARTBEAT_SECONDS", "60") or 60)),
        llm_mode=os.getenv("V3_LLM_MODE", "off"),
        llm_api_key=os.getenv("V3_OPENAI_API_KEY", ""),
        llm_model=os.getenv("V3_OPENAI_MODEL", "gpt-4o-mini"),
        llm_endpoint=os.getenv("V3_OPENAI_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
        prompt_template_file=os.getenv("V3_LLM_PROMPT_TEMPLATE_FILE", "prompts/generate_exploration_models.txt"),
        architecture_guide_file=os.getenv("V3_LLM_ARCHITECTURE_GUIDE_FILE", "prompts/instruccions.md"),
        experiment_config_file=os.getenv("V3_LLM_EXPERIMENT_CONFIG_FILE", "config/experiment_config.json"),
        llm_num_new_models=int(os.getenv("V3_LLM_NUM_NEW_MODELS", "1")),
        llm_num_reference_models=int(os.getenv("V3_LLM_NUM_REFERENCE_MODELS", "3")),
        fix_error_prompt_file=os.getenv("V3_LLM_FIX_ERROR_PROMPT_FILE", "prompts/fix_model_error.txt"),
        train_continue_recommendation_prompt_file=os.getenv("V3_LLM_TRAIN_CONTINUE_PROMPT_FILE", "prompts/recommend_train_continue.txt"),
        real_data_mode=os.getenv("V3_REAL_DATA_MODE", "false").lower() in {"1", "true", "yes"},
        data_dir=os.getenv("V3_DATA_DIR", "data"),
        max_real_rows=int(os.getenv("V3_MAX_REAL_ROWS", "4096")),
        data_cache_dtype=os.getenv("V3_DATA_CACHE_DTYPE", "float32").strip().lower(),
        use_memmap_cache=os.getenv("V3_USE_MEMMAP_CACHE", "true").lower() in {"1", "true", "yes"},
        train_epochs=max(1, int(os.getenv("V3_TRAIN_EPOCHS", "500") or 500)),
        train_batch_size=max(1, int(os.getenv("V3_TRAIN_BATCH_SIZE", "1024") or 1024)),
        train_verbose=max(0, int(os.getenv("V3_TRAIN_VERBOSE", "1") or 1)),
        train_validation_split=float(os.getenv("V3_TRAIN_VALIDATION_SPLIT", "0.15") or 0.15),
        train_early_stopping_patience=max(1, int(os.getenv("V3_TRAIN_EARLY_STOPPING_PATIENCE", "10") or 10)),
        train_reduce_lr_patience=max(1, int(os.getenv("V3_TRAIN_REDUCE_LR_PATIENCE", "5") or 5)),
        train_reduce_lr_factor=float(os.getenv("V3_TRAIN_REDUCE_LR_FACTOR", "0.5") or 0.5),
        train_min_lr=float(os.getenv("V3_TRAIN_MIN_LR", "0.000001") or 0.000001),
        train_restore_best_weights=os.getenv("V3_TRAIN_RESTORE_BEST_WEIGHTS", "true").lower() in {"1", "true", "yes"},
        train_initial_learning_rate=float(os.getenv("V3_TRAIN_INITIAL_LEARNING_RATE", "0.001") or 0.001),
        train_optimizer=os.getenv("V3_TRAIN_OPTIMIZER", "adam").strip() or "adam",
        train_seed=int(os.getenv("V3_TRAIN_SEED", "42") or 42),
        train_target_metric=os.getenv("V3_TRAIN_TARGET_METRIC", "early_stop_score").strip() or "early_stop_score",
        train_target_metric_mode=os.getenv("V3_TRAIN_TARGET_METRIC_MODE", "min").strip().lower() or "min",
        train_max_training_minutes=max(0, int(os.getenv("V3_TRAIN_MAX_TRAINING_MINUTES", "0") or 0)),
        train_business_metric_sl_weight=float(os.getenv("V3_TRAIN_BUSINESS_SL_WEIGHT", "0.5") or 0.5),
        train_business_metric_tb_weight=float(os.getenv("V3_TRAIN_BUSINESS_TB_WEIGHT", "0.5") or 0.5),
        train_business_min_relative_improvement=float(os.getenv("V3_TRAIN_BUSINESS_MIN_RELATIVE_IMPROVEMENT", "0.002") or 0.002),
        train_business_improvement_window=max(2, int(os.getenv("V3_TRAIN_BUSINESS_IMPROVEMENT_WINDOW", "10") or 10)),
        train_soft_max_epochs=max(0, int(os.getenv("V3_TRAIN_SOFT_MAX_EPOCHS", "120") or 120)),
        train_include_inline_artifacts=os.getenv("V3_TRAIN_INCLUDE_INLINE_ARTIFACTS", "true").lower() in {"1", "true", "yes"},
        train_include_full_model_artifact=os.getenv("V3_TRAIN_INCLUDE_FULL_MODEL_ARTIFACT", "true").lower() in {"1", "true", "yes"},
        train_max_inline_artifact_mb=max(16, int(os.getenv("V3_TRAIN_MAX_INLINE_ARTIFACT_MB", "256") or 256)),
    )
