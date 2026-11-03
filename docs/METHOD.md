# Démarche détaillée

Ce document raconte toute l'expérience, y compris les échecs. Les commandes sont dans [REPRODUCE.md](REPRODUCE.md).

## 0. Point de départ

Dans LM Studio, avec le prompt système « Parle comme un marseillais » et la question « Salut ça va ? »,
Qwen3.5-0.8B réfléchit jusqu'à la limite de tokens sans répondre. Il alterne sans fin entre deux brouillons :

```
"Salut ! Ça va ? Je suis bien, merci. Le soleil est beau dans la ville." -> Wait, I need to make sure...
Let's try: "Salut ! Ça va ? Je suis bien, merci. On mange à la Seine ?" -> No, let's just be friendly.
Okay, final decision: ...   (× des dizaines)
```

C'est le même type d'obstacle que dans l'expérience 8 de l'article, où les boucles faisaient échouer le modèle en
maths. Au passage, on voit aussi que le modèle ne connaît pas le parler marseillais : il place la Seine à Marseille.
C'est une capacité qui manque, et une modification des poids ne peut pas l'ajouter. On a donc pris la boucle pour cible.

Contrainte du projet : aucun modèle téléchargé. On utilise le GGUF présent dans LM Studio
(`unsloth/Qwen3.5-0.8B-GGUF`, Q8_0), et les dépendances Python sont installées uniquement dans `.venv` via `uv`.

## 1. Mesurer d'abord — `scripts/eval/measure.py`

On utilise 60 prompts de conversation en français ([`data/prompts_fr.jsonl`](../data/prompts_fr.jsonl)) : 30 avec
un persona en prompt système (marseillais, pirate, grand-mère, prof sévère…) et 30 sans. Ils sont envoyés à l'API
d'LM Studio avec une limite de 8192 tokens.

| Réglage | Tronqué | Réponse | Répétition de lignes |
|---|---|---|---|
| Défaut LM Studio | 58,3 % | 41,7 % | 63,3 % |
| Anti-répétition (temp 1.0, top_p 0.95, top_k 20, presence 1.5) | 31,7 % | 70,0 % | 0 %* |

\* La pénalité de présence empêche seulement de recopier les lignes mot pour mot : le modèle continue de tourner en
rond en reformulant. L'échantillonnage aide, mais ne règle pas le problème : l'obstacle est dans le modèle.

Les boucles sont des auto-vérifications sans fin. Exemples de lignes répétées :
« *Wait, the user might be expecting a response that's in French* » (115 fois sur « Salut ça va ? »),
« *Wait, maybe they meant…* » (91 fois sur une correction de phrase).

## 2. Charger le GGUF dans PyTorch — `loopfix/gguf_model.py`

`transformers` 5.12 ne charge pas les GGUF `qwen35`. On déquantifie donc chaque tenseur et on inverse les
transformations du convertisseur llama.cpp :

| Dans le GGUF | Dans `transformers` |
|---|---|
| RMSNorm stockées sous la forme `1 + w` | `w` (on retire 1, sauf pour la norme « gated » du DeltaNet) |
| `ssm_a = -exp(A_log)` | `A_log = log(-ssm_a)` |
| `ssm_conv1d` `[C, K]` | `conv1d.weight` `[C, 1, K]` |
| `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` | `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b` |

L'architecture est hybride : 24 couches, dont une sur quatre en attention classique (avec porte) et les autres en
Gated DeltaNet, des MLP de 3584 neurones, et des embeddings partagés avec la sortie.

**Validation** : même nombre de tokens de prompt que LM Studio (17), et sortie greedy **identique** à LM Studio sur
60 tokens. Sans cette vérification, rien de ce qui suit ne serait fiable.

## 3. Collecter les activations — `scripts/analysis/collect_acts.py`

Les prompts sont séparés en deux jeux : les numéros **impairs servent à l'analyse**, les numéros **pairs sont
réservés au test** et ne servent jamais à choisir l'édition. Les traces des prompts impairs (60 traces : 30 prompts
× 2 réglages) sont rejouées dans le modèle, chaque token suivant étant imposé (teacher
forcing). On enregistre le flux résiduel et la sortie des MLP de chaque couche à quatre types de positions :

| Type | Position | Nombre |
|---|---|---|
| `restart_loop` | juste avant une relance (« Wait », « Okay », « Actually »…) dans une trace qui boucle | 700 |
| `restart_ok` | juste avant une relance dans une trace qui finit par répondre | 613 |
| `stop` | juste avant `</think>` | 32 |
| `neutral` | position au hasard dans une réflexion saine | 320 |

## 4. La direction « je relance » — `scripts/analysis/analyze_dirs.py`

Pour chaque couche l, on calcule `d_l = moyenne(restart_loop) − moyenne(stop)`, puis on mesure à quel point chaque
MLP pousse le flux résiduel le long de cette direction :

| Couche MLP | Contribution à la direction de la couche 19 |
|---|---|
| **19** | +0,61 |
| **18** | +0,38 |
| **15** | +0,36 |
| 16 | +0,29 |
| 17 | +0,25 |

On retrouve indépendamment les **couches 15 et 19** identifiées par l'article sur le même modèle.

Attention, la séparation est déjà quasi parfaite dès la couche 0 (AUC 0,98) : une partie de la différence vient
simplement du contexte local. C'est la concentration de la poussée dans les couches 15 à 19 qui est informative.

## 5. Premier essai, raté — `scripts/analysis/search_contrast.py`

On retire `d_l` des colonnes `ffn_down` les plus « pro-relance », comme dans l'article. Les candidats sont notés sur
les **traces de test** avec trois mesures, chaque token suivant étant imposé :

- **écart relance** : log P(`</think>`) − log P(token de relance réel) aux positions de relance. On veut qu'il monte.
- **log P(`</think>`)** là où le modèle concluait vraiment. On veut qu'il reste stable (base −1,43).
- **KL** entre le modèle d'origine et le modèle modifié, à des positions neutres. On veut qu'elle reste proche de 0.

| Édition | Poids modifiés | Écart relance | log P(`</think>`) | KL |
|---|---|---|---|---|
| aucune | 0 | −18,30 | −1,43 | 0 |
| L15+L19, 64 colonnes (façon article) | 131 072 | −16,97 | **−2,29** ✗ | 2e-4 |
| L15+L18+L19, 128 colonnes | 393 216 | −16,04 | −2,80 ✗ | 6e-4 |
| L15 à L22, 64 colonnes | 458 752 | −14,88 | −3,43 ✗ | 8e-4 |
| L15+L18+L19, toutes les colonnes | 11 010 048 | −15,77 | −2,68 ✗ | 9e-4 |

**Pourquoi c'est raté** : on retire une direction de *contraste* (relance − conclusion). L'ablation ramène les
deux situations vers le milieu, donc les conclusions saines se dégradent aussi. Une ablation n'est sans danger que
si la direction vaut à peu près zéro dans les états sains.

## 6. Ce qui marche : les neurones spécifiques aux boucles — `neuron_stats.py`, `loopfix/selection.py`, `search_loop_neurons.py`, `search_scale.py`

Les neurones « relance » les plus forts s'activent autant dans les traces saines que dans les boucles : ils servent
à la réflexion normale. Quelques-uns sont propres aux boucles :

| Neurone | Relance en boucle | Relance saine | Avant `</think>` | Neutre |
|---|---|---|---|---|
| L19 n11 | **−1,06** | +0,03 | −0,14 | −0,09 |
| L21 n85 | **+0,72** | +0,05 | +0,08 | +0,20 |
| L16 n438 | **+0,26** | +0,05 | +0,06 | +0,06 |

Critère retenu : un neurone est candidat s'il pousse vers « relance » **plus en boucle que dans une relance saine**,
et s'il reste calme (|activation| < seuil) sur les relances saines, avant `</think>` et aux positions neutres. On
compare deux façons d'agir sur ces neurones : `proj` (retirer `d_l` de leur colonne) et `zero` (mettre leur colonne à zéro).

| Édition (seuil 0,3) | Poids modifiés | Écart relance | log P(`</think>`) | KL |
|---|---|---|---|---|
| 8 neurones `proj` | 8 192 | −17,77 | −1,42 | 8e-6 |
| 32 neurones `proj` | 32 768 | −17,22 | −1,41 | 3e-5 |
| **128 neurones `proj`** | **131 072** | **−16,22** | **−1,40** ✓ | **9e-5** |
| 128 neurones `zero` | 131 072 | −16,61 | −1,32 | **1,2e-2** ✗ |
| **256 neurones `proj`** | 262 144 | **−15,78** | −1,46 ✓ | 1,7e-4 |
| 512 neurones `proj` | 524 288 | −15,29 | −1,60 ✗ | 4e-4 |

Avec un seuil plus strict (0,15), les résultats sont proches mais un peu plus faibles (128 neurones `proj` :
−16,82 / −1,44 / 6e-5). `proj` fait toujours 50 à 100 fois moins de dégâts que `zero`. Le gain plafonne vers 256
neurones ; à 512, les conclusions saines commencent à se dégrader. Les 128 neurones de loopfix128 sont répartis
sur les couches 14 à 23 (23 dans L15, 27 dans L23, 10 dans L19…) ; la liste complète est dans
[`edits/loopfix_q03_top128.json`](../edits/loopfix_q03_top128.json).

## 7. Écrire le GGUF — `scripts/build/write_gguf.py`

Le script copie le GGUF d'origine et réécrit au même endroit uniquement les tenseurs `ffn_down` modifiés, requantifiés
en Q8_0. Vérifié : un seul tenseur change par couche éditée, et le reste du fichier est identique bit pour bit. En
Q8_0, 32 neurones voisins partagent une même échelle, donc les voisins d'un neurone modifié peuvent bouger de moins
d'un pas de quantification.

## 8. Le test réel dans LM Studio

Les GGUF sont branchés dans LM Studio par lien symbolique (`lms import -l`). On charge ensuite trois instances
séparées (modèle d'origine, loopfix128, loopfix256) avec le même contexte (36k, 4 slots), et on évalue les **30 prompts
de test** avec les réglages par défaut. Résultats dans [`results/heldout/comparison.txt`](../results/heldout/comparison.txt) :

| Modèle | Bloqué en boucle | A répondu | IC 95 % du Δ boucles |
|---|---|---|---|
| Modèle d'origine | 56,7 % | 43,3 % | — |
| loopfix128 | 16,7 % | 86,7 % | [−60, −20] points |
| loopfix256 | 13,3 % | 86,7 % | [−63, −23] points |

Contrôle qualitatif : les réponses factuelles restent justes quand le modèle de base les connaissait (Canberra,
Léonard de Vinci, « Bonjour »). Les erreurs typiques d'un modèle de 0,8B persistent : « Je vais aller au marché
hier », un trou noir « de taille infinie », quelques réponses dans une autre langue (déjà présentes dans le modèle
d'origine).

## Limites

- 30 prompts de test et 1 seed : l'effet est statistiquement net, mais les pourcentages exacts sont imprécis.
- Uniquement de la conversation en français : rien en maths ou en code, aucun benchmark standard.
- Les prompts de test sont de la même famille que ceux de l'analyse ; un jeu vraiment différent serait une preuve
  plus forte.
- Le proxy « écart relance » est une moyenne sur des positions où le modèle relance avec beaucoup de confiance. Il
  sert à classer les candidats ; seule la génération réelle (étape 8) fait foi.

## Pièges rencontrés

- **Contexte d'LM Studio partagé entre les slots parallèles** : avec 8192 de contexte et 4 slots, les requêtes de
  4096 tokens échouent avec « Context size has been exceeded ». Il faut `contexte ≥ workers × (max_tokens + prompt)`.
- **Template de chat** : sans `enable_thinking=True`, le template insère un bloc `<think></think>` vide et le
  modèle ne réfléchit pas.
- **Logits sur 8k tokens** : 8192 × 248k mots de vocabulaire, soit 8 Go. Pour lire les activations, on n'appelle que
  le corps du modèle (`model.model`).
