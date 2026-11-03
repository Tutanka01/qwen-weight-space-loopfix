# Weight-space programming sur Qwen3.5-0.8B : supprimer ses boucles de réflexion

Reproduction, à petite échelle et en français, de l'idée de
[Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) :
un LLM agent (ici Claude Code) **lit et modifie directement les poids** d'un modèle pour corriger un
comportement, **sans descente de gradient, sans entraînement, sans données d'entraînement**.

Le défaut visé est que Qwen3.5-0.8B, en mode thinking, tourne en boucle (« *Wait, I need to make sure…* ») et
n'arrive jamais à répondre, même à « Salut ça va ? ».

## Résultat

Évaluation sur **30 prompts de test jamais utilisés pendant l'analyse**, avec les réglages par défaut d'LM Studio,
une limite de 8192 tokens et la même seed pour les trois modèles :

| Modèle | Poids modifiés | Bloqué en boucle (tronqué à 8192) | A donné une réponse | Tokens médians |
|---|---|---|---|---|
| Qwen3.5-0.8B d'origine | 0 | **56,7 %** | 43,3 % | 8192 |
| **loopfix128** | 131 072 (0,016 % du modèle) | **16,7 %** | **86,7 %** | 3412 |
| loopfix256 | 262 144 (0,033 %) | **13,3 %** | 86,7 % | 3632 |

- loopfix128 réduit les boucles de **−40 points**, avec un intervalle de confiance à 95 % de [−60, −20] (bootstrap apparié par prompt).
- L'effet est le même avec un persona en prompt système (60 % → 13 %) et sans (53 % → 20 %).
- La modification fait **mieux que le meilleur réglage d'échantillonnage** (temp 1.0, top_p 0.95, top_k 20,
  presence_penalty 1.5), qui laisse encore 32 % de boucles.
- Les réponses restent en français : score de langue 0,94 à 0,96, contre 0,96 pour le modèle d'origine.

Comme dans l'article, la modification **retire un obstacle mais n'ajoute pas de capacité**. Le modèle répond enfin,
avec la qualité d'un modèle de 0,8B : erreurs factuelles et grammaticales (« Je vais aller au marché hier »),
réponses parfois dans une autre langue. Un chatbot marseillais reste hors de portée : le modèle ne connaît pas ce
vocabulaire et place même la Seine à Marseille.

Tableaux bruts : [`results/final_comparison.txt`](results/final_comparison.txt) et
[`results/baseline_summary.txt`](results/baseline_summary.txt).

## La démarche, étape par étape

### 0. Point de départ

Dans LM Studio, avec le prompt système « Parle comme un marseillais » et la question « Salut ça va ? », le modèle
réfléchit jusqu'à la limite de tokens sans jamais répondre. Il alterne sans fin entre deux brouillons de réponse.
C'est le même type d'obstacle que dans l'expérience 8 de l'article (les boucles qui faisaient échouer le modèle en maths).

Contraintes du projet : **aucun modèle téléchargé**. On utilise le GGUF déjà présent dans LM Studio
(`unsloth/Qwen3.5-0.8B-GGUF`, Q8_0), copié dans `models/`. Les dépendances Python sont installées par `uv`,
uniquement dans `.venv`.

### 1. Mesurer d'abord (`measure.py`, `summarize.py`)

On utilise 60 prompts de conversation en français ([`data/prompts_fr.jsonl`](data/prompts_fr.jsonl)) : 30 avec
un persona en prompt système (marseillais, pirate, grand-mère…) et 30 sans. Ils sont envoyés au serveur LM Studio
(API compatible OpenAI). On mesure le taux de troncature (la réponse atteint 8192 tokens), le taux de réponses
données et la répétition de lignes dans la réflexion.

| Réglage | Tronqué | Réponse donnée |
|---|---|---|
| Défaut LM Studio | 58,3 % | 41,7 % |
| Anti-répétition (temp 1.0, presence 1.5) | 31,7 % | 70,0 % |

L'échantillonnage aide, mais ne règle pas le problème : l'obstacle est bien dans le modèle. Avec la pénalité de
présence, la répétition mesurée tombe à 0 %, mais c'est un artefact : le modèle tourne en rond en reformulant au lieu de
recopier ses lignes mot pour mot.

> Piège rencontré : LM Studio partage son contexte entre les requêtes parallèles. Avec 8192 de contexte et 4 slots,
> les requêtes de 4096 tokens échouent avec « Context size has been exceeded ». Il faut
> `contexte ≥ workers × (max_tokens + prompt)`.

### 2. Charger le GGUF dans Python (`qwen_gguf.py`)

`transformers` ne sait pas charger un GGUF `qwen35`. On déquantifie donc chaque tenseur et on inverse les
transformations du convertisseur llama.cpp :

- les poids des RMSNorm sont stockés sous la forme `1 + w`, il faut retirer 1 (sauf pour la norme « gated » du DeltaNet) ;
- `ssm_a = -exp(A_log)` ;
- le `conv1d` est stocké aplati.

L'architecture est hybride : 24 couches, dont une sur quatre en attention classique et les autres en Gated DeltaNet,
avec des MLP de 3584 neurones.

**Validation** : même nombre de tokens de prompt que LM Studio, et sortie greedy **identique** à LM Studio sur
60 tokens. Sans cette vérification, toute l'analyse qui suit aurait pu être fausse.

### 3. Collecter les activations (`collect_acts.py`)

On sépare les prompts en deux jeux : les **numéros impairs servent à l'analyse**, les **numéros pairs sont réservés
au test**. Les traces des prompts impairs (enregistrées par LM Studio) sont rejouées dans le modèle, et on enregistre
le flux résiduel et la sortie des MLP de chaque couche à quatre types de positions :

- `restart_loop` : juste avant une relance (« Wait », « Okay », « Actually »…) dans une trace qui boucle ;
- `restart_ok` : juste avant une relance dans une trace qui finit par répondre ;
- `stop` : juste avant `</think>` ;
- `neutral` : des positions au hasard dans une réflexion saine.

### 4. Trouver la direction « je relance » (`analyze_dirs.py`)

Pour chaque couche, on calcule la direction `d_l = moyenne(relance en boucle) − moyenne(avant </think>)`. Les MLP
qui écrivent le plus cette direction sont ceux des **couches 15, 18 et 19**. L'article trouvait les couches 15 et 19
sur le même modèle, et on retombe dessus indépendamment.

### 5. Premier essai, raté (`search_contrast.py`)

On retire `d_l` des 64 colonnes `ffn_down` les plus « pro-relance » des couches 15 et 19, comme dans l'article. Pour
noter chaque modification, on utilise un proxy calculé sur les traces de test, où chaque token suivant est imposé
(teacher forcing) :

| Modification | Écart relance ↑ | log P(`</think>`) là où il concluait (base −1,43) | KL positions neutres |
|---|---|---|---|
| L15+L19, 64 colonnes | −18,30 → −16,97 | **−2,29** ✗ | 2e-4 |
| L15 à L22, 64 colonnes | → −14,88 | **−3,43** ✗ | 8e-4 |

La modification pousse un peu vers la conclusion aux endroits où le modèle relançait, mais elle **abîme les
conclusions qui marchaient déjà**. La raison : retirer une direction de contraste ramène les deux situations vers le
milieu. Une ablation n'est sans danger que si la direction vaut à peu près zéro dans les états sains.

### 6. Ce qui marche : les neurones spécifiques aux boucles (`neuron_stats.py`, `neuron_selection.py`, `search_loop_neurons.py`, `search_scale.py`)

La plupart des neurones « relance » s'activent autant dans les traces saines que dans les boucles : ils servent à la
réflexion normale. Quelques-uns ne s'activent que pendant les boucles. Exemple, le neurone 11 de la couche 19 :
activation −1,06 sur les relances en boucle, +0,03 sur les relances saines, environ −0,1 ailleurs.

Critère retenu : pousser vers « relance » **plus** en boucle que dans une relance saine, et rester calme
(|activation| < 0,3) sur les relances saines, avant `</think>` et aux positions neutres. Sur les neurones retenus,
on applique la même opération que dans l'article :

```
W_down[:, j] ← W_down[:, j] − (d_l · W_down[:, j]) d_l
```

| Modification | Poids modifiés | Écart relance ↑ | log P(`</think>`) (base −1,43) | KL |
|---|---|---|---|---|
| 128 neurones, `proj` | 131 072 | **−16,22** | −1,40 ✓ | 9e-5 |
| 128 neurones, mis à zéro | 131 072 | −16,61 | −1,32 | **1,2e-2** ✗ |
| 256 neurones, `proj` | 262 144 | **−15,78** | −1,46 ✓ | 1,7e-4 |
| 512 neurones, `proj` | 524 288 | −15,29 | −1,60 ✗ | 4e-4 |

Retirer seulement la direction (`proj`) fait jusqu'à 100 fois moins de dégâts que mettre les neurones à zéro. Le gain
plafonne vers 256 neurones ; à 512, la conclusion normale commence à se dégrader. Les 128 neurones retenus sont
répartis sur les couches 14 à 23.

### 7. Écrire le GGUF modifié (`write_gguf.py`)

Le script copie le GGUF d'origine, requantifie en Q8_0 uniquement les tenseurs `ffn_down` modifiés, et les réécrit
au même endroit. Vérification faite : un seul tenseur change, et tout le reste du fichier est identique bit pour bit.
En Q8_0, 32 neurones voisins partagent une même échelle, donc les voisins d'un neurone modifié peuvent bouger de moins
d'un pas de quantification.

### 8. Le vrai test dans LM Studio

Les GGUF modifiés sont branchés dans LM Studio par lien symbolique (`lms import -l`). On charge ensuite des instances
séparées avec le même contexte (36k, 4 slots), et on mesure les 30 prompts de test avec les réglages par défaut. Le
résultat est le tableau en haut de ce README.

## Limites

- 30 prompts de test et 1 seed : l'effet est statistiquement net, mais les pourcentages exacts sont imprécis.
- Seuls des prompts de conversation en français ont été testés : rien en maths ou en code, aucun benchmark.
- Les prompts de test sont de la même famille que ceux de l'analyse ; un jeu vraiment différent serait une preuve
  plus forte.
- Le proxy « écart relance » est une moyenne sur des positions où le modèle relance avec beaucoup de confiance. Il
  sert à classer les candidats ; seule la génération réelle (étape 8) fait foi.

## Reproduire

Prérequis : macOS Apple Silicon, [uv](https://docs.astral.sh/uv/), LM Studio avec `unsloth/Qwen3.5-0.8B-GGUF`
(Q8_0) et son serveur local actif sur le port 1234, avec un contexte suffisant (≥ 36k pour 4 × 8192 tokens).

```bash
uv sync
mkdir -p models && cp ~/.lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf models/

# 1. référence (modèle LM Studio "qwen3.5-0.8b")
uv run measure.py --config default   --out results/base_default.jsonl
uv run measure.py --config qwen_reco --out results/base_qwen_reco.jsonl
uv run summarize.py results/base_*.jsonl

# 3-6. analyse sur les prompts impairs (GPU Apple, quelques minutes par étape)
uv run collect_acts.py results/base_default.jsonl results/base_qwen_reco.jsonl
uv run analyze_dirs.py
uv run search_contrast.py        # essai raté, optionnel
uv run neuron_stats.py
uv run search_loop_neurons.py
uv run search_scale.py           # écrit edits/loopfix_q03_top{128,256,512}.json

# 7. GGUF modifié + import par lien symbolique
mkdir -p models/loopfix128
uv run write_gguf.py edits/loopfix_q03_top128.json models/loopfix128/Qwen3.5-0.8B-loopfix128-Q8_0.gguf
lms import -l -y --user-repo local/qwen35-loopfix128 "$PWD/models/loopfix128/Qwen3.5-0.8B-loopfix128-Q8_0.gguf"

# 8. évaluation sur les prompts pairs
lms load qwen3.5-0.8b      --identifier qwen-base-eval -c 36000 --parallel 4 -y
lms load qwen35-loopfix128 --identifier loopfix128     -c 36000 --parallel 4 -y
uv run measure.py --model qwen-base-eval --prompts data/prompts_fr_test.jsonl --out results/test_qwen-base-eval.jsonl
uv run measure.py --model loopfix128     --prompts data/prompts_fr_test.jsonl --out results/test_loopfix128.jsonl
uv run compare.py --seed 0 results/test_qwen-base-eval.jsonl results/test_loopfix128.jsonl
```

Les éditions retenues sont versionnées : [`edits/`](edits/) contient les neurones par couche, et
`results/acts/dirs.npz` les directions par couche (100 Ko). Pour reconstruire un modèle modifié sans refaire
l'analyse, il suffit de lancer `uv sync`, de copier le GGUF d'origine, puis d'appeler `write_gguf.py`.

Pour retirer les modèles modifiés d'LM Studio : `rm -r ~/.lmstudio/models/local/qwen35-loopfix*`. Ce ne sont que
des liens symboliques.

## Fichiers

| Fichier | Rôle |
|---|---|
| `measure.py` | génère via l'API LM Studio et mesure boucles, réponses, répétition et langue |
| `summarize.py` / `compare.py` | tableaux de synthèse / comparaison appariée avec intervalle de confiance |
| `qwen_gguf.py` | charge le GGUF Qwen3.5 (Q8_0) dans `transformers` (déquantification et inversion des transformations llama.cpp) |
| `collect_acts.py` | rejoue les traces et enregistre les activations aux positions clés |
| `analyze_dirs.py` | direction « relance contre conclusion » par couche, couches MLP qui l'écrivent |
| `edit_lab.py` | applique des éditions en mémoire et calcule le proxy sur les traces de test |
| `search_contrast.py` | recherche n°1 (ratée) : ablation de la direction de contraste |
| `neuron_stats.py` / `neuron_selection.py` | statistiques par neurone / sélection des neurones spécifiques aux boucles |
| `search_loop_neurons.py` / `search_scale.py` | recherches n°2 et n°3 (celles qui marchent) |
| `write_gguf.py` | écrit un GGUF modifié (seuls les `ffn_down` concernés sont requantifiés) |
| `data/` | prompts (60 au total, dont les 30 pairs de test dans `prompts_fr_test.jsonl`) |
| `edits/` | éditions retenues (neurones par couche) |
| `results/` | traces brutes (jsonl), scores des recherches, tableaux finaux |
