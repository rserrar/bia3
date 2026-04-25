# Runtime Change Log

Small operational log to correlate behavior changes with deploy moments.

## 2026-04-25

- **Training data update**:
  - New dataset loaded for training with `440000` records.
  - Scope: runtime/data change (no model-policy code change by itself).
  - Purpose: keep a clear temporal marker to compare before/after training behavior and model quality.

- **Server-side changes documented separately**:
  - See `V3/server-php/CHANGELOG_RUNTIME.md` for API/policy/runtime server updates.

## 2026-04-22 17:07:06Z

- **Pre-update snapshot** (`run_fresh_1`):
  - latest task: `T-002941` (`generate_candidate`, `pending`, `2026-04-22T17:05:17Z`)
  - running tasks: 5
    - `T-002938` (`train_model`, worker `worker-colab-327b9e`, `2026-04-22T17:06:15Z`)
    - `T-002937` (`train_model`, worker `worker-colab-6e3b43`, `2026-04-22T17:07:00Z`)
    - `T-002933` (`validate_candidate`, worker `worker-colab-3f6f13`, `2026-04-22T17:06:57Z`)
    - `T-002934` (`validate_candidate`, worker `worker-colab-b8980a`, `2026-04-22T17:06:19Z`)
    - `T-002932` (`train_model`, worker `worker-colab-88b26f`, `2026-04-22T17:06:44Z`)

- **Prompt generation update prepared**:
  - `prompts/generate_exploration_models.txt`
    - explicit competitiveness objective (reduce gap vs champion/top run)
    - mandatory novelty budget (1 structural change + 1 refinement)
    - stronger anti-repetition on saturated patterns
    - schema-vs-competitiveness decision rule clarified
  - `prompts/generate_evolution_models.txt`
    - clearer incremental exploitation semantics (no strong redesign)
    - explicit competitiveness objective (gap vs champion/top family)
    - stronger anti-repetition on saturated family patterns
    - schema-vs-competitiveness decision rule clarified
