# Guia curta d'arquitectura per LLM

Objectiu: generar models tabulars per regressio multisalida que compilin i puguin fer un mini-entrenament sense errors.

## Estructura minima requerida

- `architecture_definition.used_inputs`: llista de diccionaris amb `input_layer_name`.
- `architecture_definition.branches`: llista de branques amb `layers`.
- `architecture_definition.output_heads`: minim 1 sortida amb `output_layer_name`.

## Bones practiques

- Prefereix capes `Dense` simples en fases inicials.
- Evita arquitectures massa profundes en propostes de prova.
- Mantingues activacions estables (`relu`, `linear`).
- Evita canvis trivials sobre models recents.

## Restriccions de resposta

- Retorna nomes JSON valid.
- No afegeixis text extra, comentaris ni markdown.
