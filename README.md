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
3. Executa:

```bash
python scripts/run_worker.py
```

## Notes

- Aquest repo no inclou servidor ni controller.
- El worker es "tonto": `claim -> execute -> report`.
- `validate_candidate` fa compilacio/mini-fit real amb TensorFlow quan rep `model_definition_full`.
