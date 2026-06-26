# AIRLLM AUTOPSY — Audit scientifique de la base de départ

> Phase 0 du projet *Post-Transformer Neural Runtime*.
> Objectif de cette phase : comprendre **exactement** ce que fait AirLLM, sans
> illusion ni marketing, avant d'écrire la moindre ligne de code de recherche.
>
> Règle appliquée ici : **aucune affirmation sans preuve dans le code**. Chaque
> mécanisme décrit est référencé par fichier:ligne. Les chiffres sont soit tirés
> du dépôt, soit dérivés par calcul explicite (et signalés comme tels), jamais
> inventés.

---

## 0. Résumé exécutif (TL;DR honnête)

AirLLM **n'est pas** une technique de compression de modèle, ni une réduction
du nombre de paramètres, ni une réduction du calcul. C'est un **moteur de
streaming disque → mémoire couche par couche**.

L'idée tient en une phrase :

> Au lieu de charger les 70B / 405B paramètres en mémoire d'un coup, on garde le
> modèle **sur le disque**, découpé en une couche = un fichier, et on charge /
> calcule / libère **une seule couche à la fois**.

Conséquence directe et fondamentale :

| Ce qu'AirLLM réduit | Ce qu'AirLLM **ne** réduit **pas** |
|---|---|
| La mémoire **de pointe** (RAM/VRAM) | Le nombre de paramètres |
| → ≈ taille d'**une seule couche** | Le stockage disque (sauf quantization optionnelle) |
| | Le nombre de FLOPs (calcul matriciel identique) |
| | Le coût **total** d'I/O (au contraire : il l'explose) |

Le prix payé est un déplacement du goulot d'étranglement : on passe d'un système
**borné par la mémoire** à un système **borné par la bande passante disque**. La
vitesse devient ≈ `bande_passante_disque / taille_du_modèle` tokens/s. C'est
pourquoi AirLLM permet l'*impossible* (faire tourner du 405B sur 8 Go) mais
au prix d'une lenteur structurelle (de l'ordre de secondes à dizaines de
secondes **par token** sur du gros modèle).

C'est exactement le bon point de départ pour ce projet, **mais pour une raison
contre-intuitive** : AirLLM est la preuve par l'exemple que *le paradigme
poids-denses-explicites est un mur*. Il ne contourne pas le mur, il le longe.
Notre projet cherche à le percer.

---

## 1. Que fait *réellement* AirLLM ?

### 1.1 Vue d'ensemble du pipeline

```
repo HF / chemin local
        │
        ▼
find_or_create_local_splitted_path()      utils.py:341
        │   download (snapshot_download)
        ▼
split_and_save_layers()                   utils.py:188
        │   1 couche  ->  1 fichier .safetensors sur disque
        │   (embed, layer.0 … layer.N-1, norm, lm_head)
        │   + compression 4bit/8bit OPTIONNELLE ici
        ▼
   <model>/splitted_model[.4bit|.8bit]/
        │
        ▼
AirLLMBaseModel.forward()                 airllm_base.py:396
        │   pour CHAQUE forward :
        │     del self.model ; init_model()      (ligne 421-423)
        │     pour chaque couche i :
        │        load_layer_to_cpu(i)            (disque -> RAM)
        │        move_layer_to_device(i)         (RAM -> GPU)
        │        out = layer(x)                  (calcul)
        │        layer.to("meta") ; clean_memory (libère)
        ▼
     logits
```

### 1.2 Le découpage (offline, une fois)

`split_and_save_layers()` (`utils.py:188`) lit l'index du checkpoint
(`model.safetensors.index.json` ou `pytorch_model.bin.index.json`), détermine la
liste ordonnée des « couches » :

```
[embed_tokens] + [layers.0 … layers.N-1] + [norm] + [lm_head]
```

(voir `utils.py:222-229`) et **ré-écrit chaque couche dans son propre fichier
safetensors** (`persist_model`, `utils.py:323`). C'est une transformation de
*layout disque* : les mêmes poids, réorganisés pour permettre un chargement
sélectif couche par couche. Rien n'est appris, rien n'est jeté (hors
quantization).

### 1.3 L'inférence (le cœur)

Dans `forward()` (`airllm_base.py:396`), pour chaque appel :

1. **Reboot complet du modèle** : `del self.model; clean_memory(); self.init_model()`
   (`airllm_base.py:421-423`). Le modèle est re-instancié en *meta tensors*
   (poids vides, zéro mémoire) à chaque forward.

2. **Boucle couche par couche** (`airllm_base.py:450`) :
   - `load_layer_to_cpu()` lit le fichier safetensors de la couche depuis le
     disque vers la RAM (`airllm_base.py:269`).
   - `move_layer_to_device()` matérialise les poids sur le GPU via
     `set_module_tensor_to_device` (`airllm_base.py:302-323`).
   - la couche est exécutée sur la séquence (`airllm_base.py:500-586`).
   - **la couche est immédiatement détruite** : `layer.to("meta")` +
     `clean_memory()` (`airllm_base.py:597-600`).

3. **Prefetching** (`airllm_base.py:441-487`) : un `ThreadPoolExecutor` charge
   les poids de la couche `i+1` depuis le disque **pendant** que le GPU calcule
   la couche `i`. C'est le seul vrai gain de débit interne (≈ 10 % annoncé par
   le projet, README « v2.5 »), car il recouvre l'I/O par le calcul.

### 1.4 La compression (optionnelle, orthogonale)

`compress_layer_state_dict()` (`utils.py:157`) applique, **au moment du
découpage**, une quantization bitsandbytes :
- `4bit` → `quantize_nf4` (NF4, blocksize 64) — `utils.py:159-165`
- `8bit` → `quantize_blockwise` (blocksize 2048) — `utils.py:166-174`

Au chargement, `uncompress_layer_state_dict()` (`utils.py:85`) déquantifie la
couche vers fp16 avant calcul. Le gain est **double et indirect** :
- moins d'octets à lire sur le disque → I/O ÷ ~4 (4bit) ou ÷ 2 (8bit) → vitesse ;
- moins d'espace disque.

C'est de la quantization classique post-hoc. Elle ne change pas le paradigme :
les poids restent des matrices denses, juste encodées sur moins de bits.

### 1.5 Détail crucial souvent négligé : pas de cache KV sur transformers récents

```python
if cache_utils_installed:
    # we don't support kv cache for new version yet
    use_cache = False
```
(`airllm_base.py:410-412`)

Sur les versions modernes de `transformers`, **le cache KV est désactivé**.
Combiné au *reboot complet du modèle à chaque forward* (§1.3.1), cela signifie
qu'une génération auto-régressive de `T` tokens relit **tout le modèle depuis le
disque `T` fois**, et recalcule l'attention sur tout le préfixe à chaque pas.
C'est la limite de performance dominante (voir §2).

---

## 2. Quelle limite vient de l'*architecture* (de l'implémentation AirLLM) ?

Ces limites sont propres à AirLLM et pourraient en principe être améliorées sans
changer de paradigme.

### 2.1 Le débit est borné par le disque, pas par le calcul

À chaque token (cache KV désactivé), AirLLM lit l'intégralité des poids depuis
le disque. Donc, en première approximation :

```
temps_par_token  ≈  taille_modèle_sur_disque / bande_passante_disque
tokens_par_s     ≈  bande_passante_disque / taille_modèle_sur_disque
```

Ordres de grandeur (calcul dérivé, à valider par benchmark en Phase 5) :

| Modèle | Taille fp16 | NVMe ~2 Go/s | 4bit, NVMe ~2 Go/s |
|---|---|---|---|
| 7B | ~14 Go | ~7 s/token | ~1.7 s/token |
| 70B | ~140 Go | ~70 s/token | ~17 s/token |
| 405B | ~810 Go | ~400 s/token | ~100 s/token |

> ⚠️ Chiffres **dérivés analytiquement** depuis le mécanisme du code, pas
> mesurés. Ils servent de cadre, pas de résultat. Le prefetching et le cache OS
> peuvent améliorer ; l'absence de cache KV peut empirer. À mesurer en Phase 5.

Sur Raspberry Pi (cible du projet), la bande passante d'une carte SD / USB est
de l'ordre de 40–400 Mo/s : on est alors à **plusieurs minutes par token** pour
un gros modèle. AirLLM seul ne rend pas la cible « trillion sur Raspberry Pi »
réaliste — c'est précisément ce qui motive un changement de paradigme.

### 2.2 Reboot du modèle à chaque forward

`del self.model; init_model()` à chaque `forward()` (`airllm_base.py:421`)
ré-instancie tout le graphe (même en meta) et re-déplace les buffers. Surcoût
fixe par forward, négligeable face à l'I/O mais réel.

### 2.3 Cache KV désactivé sur transformers récents

(`airllm_base.py:410`) Transforme une génération en `O(T)` passes complètes du
modèle au lieu d'une seule + `T` pas incrémentaux. C'est probablement le levier
d'optimisation interne le plus important encore disponible *sans changer de
paradigme* (réintroduire un cache KV persistant entre forwards).

### 2.4 La granularité = une couche transformer entière

La mémoire de pointe ≈ taille de **la plus grosse couche** (+ activations + KV +
buffers). Pour un 70B (hidden ≈ 8192), une couche ≈ 1.5–1.7 Go en fp16. On ne
peut pas descendre sous « une couche » avec ce design. Une granularité plus fine
(par tête, par bloc, par neurone) nécessiterait un autre moteur — c'est une
porte ouverte vers nos axes Sparse / MoE.

### 2.5 Dépendance forte à `transformers` / `bitsandbytes` / CUDA

Le moteur s'appuie sur `AutoModelForCausalLM`, `BetterTransformer`,
`set_module_tensor_to_device`, `bitsandbytes` (CUDA pour quantize/dequantize,
`utils.py:92` `.cuda()`). Le support CPU/Mac existe (`airllm_llama_mlx.py`,
README « CPU inference ») mais la quantization 4/8bit est liée à CUDA. Pour une
cible embarquée/offline, c'est une dette d'architecture.

### 2.6 Le stockage n'est pas réduit (hors quantization)

Le modèle complet vit sur le disque : 140 Go pour un 70B fp16, 810 Go pour un
405B fp16. AirLLM déplace le problème de la RAM vers le disque ; il ne le fait
pas disparaître. La quantization 4bit le ramène à ~1/4, mais on reste dans
l'ordre « centaines de Go » pour les très gros modèles.

---

## 3. Quelle limite vient du *paradigme* actuel des LLM ?

Ces limites ne sont **pas** des défauts d'AirLLM : ce sont les propriétés du
paradigme « poids denses explicites + multiplication matricielle » qu'AirLLM
hérite et ne peut pas dépasser de l'intérieur.

### 3.1 Les paramètres sont matérialisés explicitement

Un modèle de `P` paramètres = `P` nombres réellement stockés, un par un. La
seule question qu'AirLLM se pose est *« où les stocker et quand les charger »*.
Il ne se demande jamais *« ai-je besoin de les stocker tous ? »*. Le projet
Post-Transformer pose précisément cette seconde question.

### 3.2 Le calcul est dominé par `X · W` (matmul dense)

Chaque couche fait des produits matriciels `activation × poids` de coût
`O(d²)` (voire `O(d_model · d_ff)`) par token. AirLLM exécute **exactement les
mêmes FLOPs** qu'une inférence normale (`airllm_base.py:535,569`). Il n'optimise
pas le calcul, il le *séquence*. Tant qu'on reste sur la matmul dense, le coût
arithmétique est incompressible.

### 3.3 Densité et redondance

Le paradigme suppose des matrices **denses** : chaque poids compte, tout le
temps, pour toute entrée. La littérature (MoE, lottery tickets, low-rank des
mises à jour, quantization extrême qui marche) suggère fortement que cette
densité est en grande partie une **redondance** : une fraction des paramètres
est active/utile pour une entrée donnée. Le paradigme dense ne sait pas
exploiter cela ; AirLLM non plus (il charge la couche entière, toujours).

### 3.4 Représentation statique

Les poids sont figés et **énumérés**. Rien dans le format ne capture une
éventuelle **structure génératrice** (répétitions, symétries, basse dimension
intrinsèque, règle de génération). On stocke le *résultat* du modèle, jamais sa
*description la plus courte*. C'est le lien direct avec la complexité de
Kolmogorov / MDL (Axe 1) et les hypernetworks (Axe 3).

### 3.5 Le conflit fondamental que ce projet attaque

> Mémoire et calcul croissent **linéairement avec le nombre de paramètres
> énumérés**, alors que la « capacité utile » d'un modèle pourrait croître avec
> une quantité bien plus petite : sa **complexité descriptionnelle** ou sa
> **dimension intrinsèque**.

Si (et c'est l'hypothèse à prouver) la capacité réelle dépend d'une grandeur
`K ≪ P`, alors énumérer `P` poids est une représentation *inefficace*, pas une
*nécessité*. AirLLM optimise la logistique de `P` ; nous voulons attaquer `P`
lui-même.

---

## 4. Quelle rupture serait nécessaire ?

AirLLM a poussé à son maximum la stratégie *« garder le paradigme, optimiser la
logistique »*. Le rendement décroissant est atteint : on est borné par le disque
et par la taille brute du modèle. Toute amélioration interne (cache KV,
granularité plus fine, meilleur prefetch) est incrémentale.

La rupture nécessaire est de **ne plus stocker `P` valeurs**. Quatre familles de
ruptures, du plus mûr au plus spéculatif — chacune correspond à un axe du
projet, avec ici une **première estimation de plausibilité** à confirmer par
expérience (aucune n'est validée à ce stade) :

| Rupture | Idée | Axe projet | Plausibilité a priori\* |
|---|---|---|---|
| **Sparsité / routage** | n'activer qu'une petite fraction des poids par token (MoE, sparse) | Axe 6 | **Élevée** — déjà industrialisé (Mixtral est dans le repo : `airllm_mixtral.py`) |
| **Quantization extrême / VQ** | remplacer les valeurs par index + dictionnaire partagé | Axe 9 / 5 | **Élevée** — 4bit marche déjà ici ; 2bit/ternaire/VQ = continuité |
| **Factorisation / bas rang** | remplacer une grosse matrice par un produit de petites | Axe 5 | **Moyenne** — efficace sur *deltas*/fine-tuning, plus risqué sur poids de base |
| **Génération des poids** | un petit générateur produit les poids à la demande (hypernetwork, neural field, MDL/Kolmogorov) | Axes 1,3,4,8 | **Faible mais à fort impact** — c'est le vrai pari scientifique |
| **Calcul non-matriciel** | ODE neuronales, systèmes dynamiques, mémoire associative, binaire/ternaire | Phase 2, Axes 2,7 | **Spéculative** — à traiter comme recherche exploratoire |

\* *Plausibilité a priori = jugement de départ pour prioriser, PAS un résultat.
Chaque case sera tranchée VALIDÉ / PARTIEL / RÉFUTÉ par benchmark (Phase 4).*

### 4.1 Reformulation de la question scientifique finale

La question du projet —

> *Les trillions de paramètres sont-ils une nécessité fondamentale, ou seulement
> une représentation inefficace de l'intelligence ?*

— se décompose, à la lumière de cette autopsie, en trois sous-questions
**falsifiables** :

1. **(Compression sans perte de capacité)** Existe-t-il une représentation de
   taille `K ≪ P` qui reproduit le comportement du modèle à une tolérance ε
   donnée, sur un benchmark fixé ? → Axes 1,2,5,8,9.
2. **(Activation partielle)** Pour une entrée donnée, quelle fraction `f` des
   paramètres suffit à reproduire la sortie à ε près ? Si `f ≪ 1` de façon
   stable et *prévisible* (routable), la densité est une redondance. → Axes 6,7.
3. **(Génération à la demande)** Peut-on produire les poids nécessaires par un
   calcul plus court que leur énumération, plus vite que de les lire sur disque ?
   → Axes 1,3,4.

Un « oui » mesuré et reproductible à **l'une** de ces questions est déjà une
contribution. Un « non » rigoureux l'est tout autant.

### 4.2 Ce que cette autopsie impose à la suite du projet (garde-fous)

1. **Comparer à AirLLM, pas à un fantasme.** La baseline est AirLLM réel,
   mesuré sur la même machine, même modèle, même prompt (Phase 5). Pas les
   chiffres marketing du README.
2. **Mesurer les 6 axes systématiquement** : RAM, VRAM, stockage, tokens/s,
   énergie, **qualité** (perplexité ou exactitude sur un set fixe). Une méthode
   qui gagne en mémoire mais détruit la qualité n'a rien prouvé.
3. **Séparer les gains de logistique des gains de paradigme.** Réintroduire un
   cache KV ou changer de disque accélère AirLLM sans rien prouver sur la
   question scientifique. Ces gains doivent être attribués à la *baseline*, pas
   à la nouveauté.
4. **Falsifiabilité d'abord.** Chaque axe = un `REPORT.md` avec hypothèse,
   protocole, résultat chiffré, verdict (VALIDÉ / PARTIEL / RÉFUTÉ). Pas de
   verdict sans benchmark reproductible. Un axe non testé reste « NON TESTÉ »,
   jamais « validé par construction ».
5. **Honnêteté sur l'échelle.** Aucune expérience sérieuse ne « met un 1T sur un
   Pi » au départ. On valide les *mécanismes* sur petits modèles (vérité-terrain
   accessible : on peut comparer aux vrais poids), puis on extrapole prudemment.

---

## 5. Inventaire factuel du dépôt (pour référence)

Modules du moteur (`air_llm/airllm/`) :

| Fichier | Rôle |
|---|---|
| `airllm_base.py` | moteur de streaming couche par couche (cœur, §1-2) |
| `utils.py` | découpage disque, (dé)compression, gestion espace/mémoire |
| `auto_model.py` | détection auto du type de modèle |
| `airllm.py`, `airllm_llama_mlx.py` | wrappers Llama (CUDA / MLX Mac) |
| `airllm_mistral.py`, `airllm_mixtral.py` | Mistral / **Mixtral (MoE déjà présent !)** |
| `airllm_qwen.py`, `airllm_qwen2.py`, `airllm_chatglm.py`, `airllm_baichuan.py`, `airllm_internlm.py` | familles de modèles |
| `persist/` | sérialisation safetensors / MLX des couches |
| `profiler.py` | mesure des temps par étape (`LayeredProfiler`) |

Observation stratégique : **`airllm_mixtral.py` existe déjà**. Mixtral est un
Mixture-of-Experts. C'est le point d'entrée naturel et le moins risqué pour
l'Axe 6 (Sparse Intelligence) : le routage existe, il « suffit » de ne streamer
que les experts actifs par token au lieu de toute la couche.

---

## 6. Conclusion de la Phase 0

AirLLM répond brillamment à une question logistique — *« comment faire tenir un
modèle énorme dans une petite mémoire »* — par le streaming disque couche par
couche, au prix d'être borné par la bande passante disque et de ne réduire ni
le nombre de paramètres, ni le calcul, ni (hors quantization) le stockage.

Il constitue donc la **borne supérieure du paradigme dense** : tout ce qu'on
peut faire en gardant les poids explicites et la matmul dense, AirLLM le fait
déjà ou presque. Pour aller plus loin, il faut s'attaquer non pas à *où* sont
les `P` paramètres, mais à *l'existence même* de ces `P` paramètres énumérés.

C'est l'objet des phases suivantes. La prochaine étape concrète, la moins
risquée et la plus mesurable, est de **fixer la baseline AirLLM expérimentale**
(Phase 5, un harnais de benchmark reproductible : RAM/VRAM/disque/tok-s/énergie/
qualité) puis d'attaquer l'**Axe 6 (sparsité, via le Mixtral déjà présent)**,
qui a la plus forte plausibilité a priori et un point d'ancrage déjà dans le
code.

> Aucune victoire ne sera revendiquée sans un benchmark reproductible comparant
> le nouveau moteur à cette baseline, sur la même machine et le même modèle.
