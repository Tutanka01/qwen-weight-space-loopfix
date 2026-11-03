# Reproduire l'expérience

Toutes les commandes se lancent depuis la racine du repo. Les chemins sont centralisés dans
[`loopfix/paths.py`](../loopfix/paths.py).

## Prérequis

- macOS Apple Silicon (l'analyse utilise le GPU via MPS ; testé sur M5 Pro, 24 Go)
- [uv](https://docs.astral.sh/uv/)
- [LM Studio](https://lmstudio.ai/) avec le modèle `unsloth/Qwen3.5-0.8B-GGUF` (fichier `Qwen3.5-0.8B-Q8_0.gguf`),
  le serveur local actif (port 1234) et la CLI `lms`

```bash
uv sync                                   # crée .venv : torch, transformers 5.12.1, gguf, numpy
mkdir -p models && cp ~/.lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf models/
```

> **Contexte LM Studio** : il est partagé entre les requêtes parallèles. Pour 4 requêtes de 8192 tokens, chargez le
> modèle avec au moins 36k de contexte.

## Le plus simple : télécharger le modèle

Le modèle est publié sur [Hugging Face](https://huggingface.co/Tutanka01/Qwen3.5-0.8B-loopfix-GGUF) :

```bash
lms get Tutanka01/Qwen3.5-0.8B-loopfix-GGUF
# ou bien
uvx hf download Tutanka01/Qwen3.5-0.8B-loopfix-GGUF Qwen3.5-0.8B-loopfix128-Q8_0.gguf --local-dir models/loopfix128
shasum -a 256 models/loopfix128/Qwen3.5-0.8B-loopfix128-Q8_0.gguf
# attendu : 93dd99ff5b350060b45256973504c4363d18e5da7718bb2feec66f2f81ab9d8b
```

## Raccourci : reconstruire loopfix128 sans refaire l'analyse

Les éditions (`edits/`) et les directions (`results/acts/dirs.npz`) sont versionnées :

```bash
mkdir -p models/loopfix128
uv run scripts/build/write_gguf.py edits/loopfix_q03_top128.json models/loopfix128/Qwen3.5-0.8B-loopfix128-Q8_0.gguf
lms import -l -y --user-repo local/qwen35-loopfix128 "$PWD/models/loopfix128/Qwen3.5-0.8B-loopfix128-Q8_0.gguf"
```

Le fichier obtenu est identique bit pour bit à celui qui a été évalué et publié (même SHA-256 que ci-dessus).

## Pipeline complet

### 1. Mesure de référence (environ 1 h)

```bash
lms load qwen3.5-0.8b -c 36000 --parallel 4 -y
uv run scripts/eval/measure.py --config default   --out results/baseline/default.jsonl
uv run scripts/eval/measure.py --config qwen_reco --out results/baseline/qwen_reco.jsonl
uv run scripts/eval/summarize.py results/baseline/default.jsonl results/baseline/qwen_reco.jsonl
```

### 2 à 6. Analyse sur les prompts impairs (GPU, environ 30 min en tout)

```bash
uv run scripts/analysis/collect_acts.py         # activations -> results/acts/*.npz (~160 Mo)
uv run scripts/analysis/analyze_dirs.py         # directions par couche -> results/acts/dirs.npz
uv run scripts/analysis/search_contrast.py      # optionnel : l'essai raté -> results/search/contrast.json
uv run scripts/analysis/neuron_stats.py         # stats par neurone -> results/acts/neuron_stats.pt
uv run scripts/analysis/search_loop_neurons.py  # proj vs zero -> results/search/loop_neurons.json
uv run scripts/analysis/search_scale.py         # 128/256/512 neurones -> results/search/scale.json + edits/*.json
```

### 7. Écrire les GGUF modifiés et les brancher dans LM Studio

```bash
for n in 128 256; do
  mkdir -p models/loopfix$n
  uv run scripts/build/write_gguf.py edits/loopfix_q03_top$n.json models/loopfix$n/Qwen3.5-0.8B-loopfix$n-Q8_0.gguf
  lms import -l -y --user-repo local/qwen35-loopfix$n "$PWD/models/loopfix$n/Qwen3.5-0.8B-loopfix$n-Q8_0.gguf"
done
```

`-l` crée seulement un lien symbolique. Pour retirer les modèles : `rm -r ~/.lmstudio/models/local/qwen35-loopfix*`.

### 8. Évaluation sur les 30 prompts pairs (environ 20 min)

```bash
lms load qwen3.5-0.8b      --identifier base       -c 36000 --parallel 4 -y
lms load qwen35-loopfix128 --identifier loopfix128 -c 36000 --parallel 4 -y
lms load qwen35-loopfix256 --identifier loopfix256 -c 36000 --parallel 4 -y
for m in base loopfix128 loopfix256; do
  uv run scripts/eval/measure.py --model $m --prompts data/prompts_fr_test.jsonl --out results/heldout/$m.jsonl &
done; wait
uv run scripts/eval/compare.py --seed 0 results/heldout/base.jsonl results/heldout/loopfix128.jsonl results/heldout/loopfix256.jsonl
lms unload --all
```

Les générations utilisent un échantillonnage aléatoire : les chiffres exacts varient d'un run à l'autre. La baisse des
boucles, elle, est largement au-dessus de ce bruit (intervalle de confiance à 95 % de [−60, −20] points).
