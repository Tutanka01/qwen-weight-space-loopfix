# loopfix : guérir Qwen3.5-0.8B de ses boucles en modifiant ses poids à la main

En mode thinking, Qwen3.5-0.8B tourne souvent en rond (« *Wait, I need to make sure…* ») jusqu'à la limite de tokens,
sans jamais répondre. Ce projet corrige ce défaut **en modifiant directement 0,016 % de ses poids**, sans
entraînement, sans gradient et sans données. C'est une reproduction, sur un petit modèle, de l'idée de
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), faite avec Claude Code comme agent.

**Modèle sur Hugging Face : [Tutanka01/Qwen3.5-0.8B-loopfix-GGUF](https://huggingface.co/Tutanka01/Qwen3.5-0.8B-loopfix-GGUF)**
(GGUF Q8_0, 812 Mo, utilisable directement dans LM Studio ou llama.cpp)

## Avant / après

> **Traduis 'good morning' en français.**

| | Réflexion | Réponse |
|---|---|---|
| Qwen3.5-0.8B d'origine | trouve « Bonjour », puis répète **32 fois** « *Wait, I think there might be some confusion here…* » jusqu'à 8192 tokens | *(aucune)* |
| **loopfix128** | réflexion courte, 3284 tokens au total | **Bonjour** |

## Résultats

30 prompts de test en français, jamais utilisés pendant l'analyse, avec les réglages par défaut d'LM Studio, une
limite de 8192 tokens et la même seed pour les trois modèles.

| Modèle | Poids modifiés | Bloqué en boucle | A répondu | Tokens médians |
|---|---|---|---|---|
| Qwen3.5-0.8B d'origine | 0 | 56,7 % | 43,3 % | 8192 |
| **loopfix128** | **131 072 (0,016 %)** | **16,7 %** | **86,7 %** | 3412 |
| loopfix256 | 262 144 (0,033 %) | 13,3 % | 86,7 % | 3632 |

Pour loopfix128, la baisse des boucles est de **−40 points** (intervalle de confiance à 95 % de [−60, −20]).

| Détail (loopfix128) | Modèle d'origine | loopfix128 |
|---|---|---|
| Boucles avec un persona (« Parle comme un marseillais »…) | 60,0 % | 13,3 % |
| Boucles sans persona | 53,3 % | 20,0 % |
| Réponses en français (score de langue) | 0,96 | 0,94 |

**Comparaison avec un simple réglage d'échantillonnage** (modèle d'origine, 60 prompts) :

| Méthode | Bloqué en boucle |
|---|---|
| Réglages par défaut | 58,3 % |
| Anti-répétition (temp 1.0, top_p 0.95, top_k 20, presence 1.5) | 31,7 % |
| **Édition des poids, réglages par défaut** (loopfix128, 30 prompts de test) | **16,7 %** |

**Les essais qui n'ont pas marché**, mesurés avec un proxy sur les traces de test (le tableau complet est dans
[docs/METHOD.md](docs/METHOD.md)) :

| Édition | Poussée vers la conclusion | Conclusions normales | Dégâts |
|---|---|---|---|
| Ablation « façon article », couches 15 et 19 | un peu ↑ | **abîmées** ✗ | faibles |
| Neurones spécifiques aux boucles, mis à zéro | ↑ | intactes | **×100** ✗ |
| **Neurones spécifiques aux boucles, retrait de direction** | **↑↑** | **intactes** ✓ | **quasi nuls** ✓ |

## Ce que ça ne fait pas

Comme le concluait l'article, l'édition **retire un obstacle mais n'ajoute aucune capacité**. Le modèle répond
enfin, avec ses limites de modèle de 0,8B : erreurs factuelles et grammaticales, langue qui dérive parfois. Seuls
des prompts de conversation en français ont été testés (30 prompts, 1 seed).

## Comment

1. On mesure les boucles via LM Studio.
2. On charge le GGUF dans PyTorch : la sortie est validée identique à LM Studio.
3. On repère, dans les activations, la direction « je relance ma réflexion ». Les MLP qui l'écrivent sont ceux des
   couches 15, 18 et 19 ; l'article trouvait 15 et 19.
4. On retire cette direction des colonnes `ffn_down` de **128 neurones qui ne s'activent que pendant les boucles** :

   ```
   W_down[:, j] ← W_down[:, j] − (d · W_down[:, j]) d
   ```
5. On réécrit le GGUF et on évalue dans LM Studio.

- La démarche complète, avec les échecs : **[docs/METHOD.md](docs/METHOD.md)**
- Les commandes pour tout reproduire : **[docs/REPRODUCE.md](docs/REPRODUCE.md)**

## Utiliser le modèle

```bash
lms get Tutanka01/Qwen3.5-0.8B-loopfix-GGUF                  # LM Studio
llama-cli -hf Tutanka01/Qwen3.5-0.8B-loopfix-GGUF --jinja      # llama.cpp
```

On peut aussi le reconstruire **bit pour bit** à partir du GGUF d'Unsloth, en une minute
(voir [docs/REPRODUCE.md](docs/REPRODUCE.md)).
SHA-256 : `93dd99ff5b350060b45256973504c4363d18e5da7718bb2feec66f2f81ab9d8b`.

## Organisation

```
loopfix/            bibliothèque : chargement GGUF→PyTorch, traces, labo d'édition, sélection de neurones, chemins
scripts/eval/       measure.py (génération via LM Studio), summarize.py, compare.py
scripts/analysis/   collect_acts.py, analyze_dirs.py, neuron_stats.py, search_*.py
scripts/build/      write_gguf.py (écrit le GGUF modifié)
data/               60 prompts français (impairs = analyse, pairs = test)
edits/              éditions retenues : neurones modifiés par couche
results/            baseline/, heldout/ (évaluation finale), search/ (scores des candidats), acts/dirs.npz
docs/               METHOD.md, REPRODUCE.md
```

## Crédits

Modèle de base : [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) (Apache 2.0), quantification GGUF Q8_0
par [Unsloth](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF). Idée et méthode : billet
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
