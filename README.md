# V3 Colab Worker

Repositori minim per executar workers Colab contra l'API V3.

## Setup rapid

1. Copia `.env.example` a `.env` (o defineix variables al notebook).
2. Defineix com a minim:
   - `V3_API_BASE_URL=https://control.einavirtual.com/v3/public/index.php`
   - `V3_WORKER_ID` unic per notebook
   - opcional LLM real:
     - `V3_LLM_MODE=openai_chat`
     - `V3_OPENAI_API_KEY=...`
     - `V3_OPENAI_MODEL=gpt-4o-mini`
     - `V3_OPENAI_ENDPOINT=https://api.openai.com/v1/chat/completions`
     - `V3_LLM_PROMPT_TEMPLATE_FILE=prompts/generate_new_models.txt`
     - `V3_LLM_ARCHITECTURE_GUIDE_FILE=prompts/instruccions.md`
     - `V3_LLM_EXPERIMENT_CONFIG_FILE=config/experiment_config.json`
     - `V3_LLM_FIX_ERROR_PROMPT_FILE=prompts/fix_model_error.txt`
3. Executa:

```bash
python scripts/run_worker.py
```

## Notes

- Aquest repo no inclou servidor ni controller.
- El worker es "tonto": `claim -> execute -> report`.
- `validate_candidate` fa compilacio/mini-fit real amb TensorFlow quan rep `model_definition_full`.
- El prompt LLM reutilitza l'estructura de `V2PromptBuilder`.
