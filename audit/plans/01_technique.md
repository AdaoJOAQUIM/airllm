# Plan 01 — Technique : cartographie réelle du fork AirLLM

> Source : lecture du code réel (commit `75436d1`, branche `claude/airllm-1t-raspberry-pi-jolk12`).
> Toute affirmation est étayée par `fichier:ligne` ou marquée `[HYPOTHÈSE]`.

## 1. Chemin d'inférence couche-par-couche (le cœur)

`AirLLMBaseModel.forward` (`air_llm/airllm/airllm_base.py:396-643`) implémente le *layer
streaming*. Séquence réelle observée :

1. **Reboot du modèle à CHAQUE forward** : `del self.model; clean_memory(); self.init_model()`
   (`airllm_base.py:421-423`). Le modèle « méta » (poids vides) est reconstruit à chaque
   appel `forward`, donc à chaque token généré.
2. **Construction du masque d'attention plein** `max_seq_len × max_seq_len`
   (`airllm_base.py:429-432`) — mémoire O(seq²), indépendant de la longueur réelle.
3. **Boucle de streaming** (`airllm_base.py:450-600`) sur
   `zip(self.layer_names, self.layers)` :
   - prefetch de la couche `i+1` via `ThreadPoolExecutor.submit(self.load_layer_to_cpu, ...)`
     (`airllm_base.py:447,481`),
   - `future.result()` pour récupérer la couche `i` (`airllm_base.py:458`),
   - `move_layer_to_device` (`airllm_base.py:469`, def `:302-323`),
   - exécution du sous-module (`airllm_base.py:500-586`),
   - **éviction** : `layer.to("meta")` puis `clean_memory()` (`airllm_base.py:597-600`).
4. Concat final des logits (`airllm_base.py:602`).

**Conséquence dure :** la liste des couches streamées est
`[embed] + layers[0..N-1] + [norm] + [lm_head]` (`airllm_base.py:141-143`). Donc **tous les
poids du modèle traversent le disque→RAM→device une fois par token** (sauf KV-cache, cf. §4).

## 2. Où vit chaque octet (carte mémoire)

| Étage | Mécanisme | Référence | Persistance |
|---|---|---|---|
| Stockage froid | 1 fichier `.safetensors` par couche + marqueur `.done` | `persist/safetensor_model_persister.py:27-37` | disque (carte SD / USB) |
| Lecture | `safetensors.torch.load_file(...)` **chargement complet** du tensor en RAM | `safetensor_model_persister.py:36-37`, `utils.py:115-117` | RAM CPU, transitoire |
| Pin (CUDA only) | `state_dict[k].pin_memory()` si `prefetching and cuda` | `airllm_base.py:287-294` | RAM page-locked |
| Device | `set_module_tensor_to_device(..., dtype=self.running_dtype)` | `airllm_base.py:317-319` | RAM/VRAM, **1 couche à la fois** |
| Éviction | `layer.to("meta")` + `clean_memory()` (`malloc_trim`+`empty_cache`) | `airllm_base.py:597-600`, `utils.py:75-82` | libère après chaque couche |

**Diagramme du flux mémoire (par couche, par token) :**

```
[SD/USB .safetensors] --load_file()--> [RAM CPU state_dict]
        (lecture COMPLÈTE, pas de mmap lazy)        |
                                                     | set_module_tensor_to_device(dtype=running_dtype)
                                                     v
                                          [device: 1 couche matérialisée] --forward--> activations
                                                     |
                                                     | layer.to("meta") + clean_memory()
                                                     v
                                              [poids libérés]
```

Pic mémoire ≈ taille de **la plus grosse couche** (+ activations + masque seq²), pas du
modèle entier. C'est l'invariant qui permet « 70B sur 4GB » (README:9).

## 3. Format des poids & étape de découpage

- Découpage offline : `split_and_save_layers` (`utils.py:188-339`) lit l'index
  (`model.safetensors.index.json` ou `pytorch_model.bin.index.json`, `utils.py:208-215`),
  regroupe par préfixe de couche (`utils.py:315`), compresse optionnellement, et écrit un
  shard par couche (`utils.py:321-323`).
- Téléchargement HF avec `ignore_patterns=['*.safetensors','*.bin']` d'abord (métadonnées),
  puis shard-par-shard à la demande (`utils.py:377-379`, `:289-293`).
- `check_space` (`utils.py:134-155`) vérifie l'espace disque avant split.

## 4. KV-cache & quantization actuelle

- **KV-cache désactivé** sur transformers récent : `if cache_utils_installed: use_cache = False`
  (`airllm_base.py:410-412`). **Conséquence majeure** : à chaque nouveau token, la séquence
  entière est recalculée → coût de lecture disque = `N_tokens × poids_modèle` (quadratique en
  tokens lus). C'est le pire cas pour un débit disque limité.
- **Compression = LOSSY et CUDA-only** : `quantize_nf4` / `quantize_blockwise`
  (`utils.py:162,169`) et leurs inverses `dequantize_nf4` / `dequantize_blockwise`
  (`utils.py:94,105`) appellent `.cuda()` en dur. Donc : (a) avec perte → **exclue** de notre
  voie principale ; (b) **inutilisable sur Pi** (pas de CUDA). De plus la compression désactive
  le prefetch (`airllm_base.py:152-154`).

## 5. Graphe de dépendances (modules)

```
auto_model.AutoModel.from_pretrained ──> airllm_<arch>.AirLLM<Arch> ──> AirLLMBaseModel
                                                                              │
   utils.find_or_create_local_splitted_path ── split_and_save_layers ────────┤
   utils.load_layer ── ModelPersister.get_model_persister().load_model ──────┤
   persist.{Safetensor,Mlx}ModelPersister ───────────────────────────────────┘
   profiler.LayeredProfiler (instrumentation)
```

`AutoModel` route par `config.architectures[0]` (`auto_model.py:27-45`). Les classes d'arch
(`airllm_mixtral.py`, `airllm_qwen2.py`, …) ne font qu'overrider des hooks ; **`AirLLMMixtral`
n'exploite aucune sparsité MoE** (`airllm_mixtral.py:8-14` hérite tel quel → tous les experts
sont lus, cf. Plan 09).

---

## Section transversale A — Justification

- **Pourquoi cartographier d'abord ?** Toute la mission repose sur l'invariant « sans perte ».
  On ne peut prouver qu'une optimisation ne touche pas le résultat que si l'on sait
  exactement *où* le résultat est produit. Le point névralgique est `forward`
  (`airllm_base.py:396-643`) : c'est là, et seulement là, que les valeurs du modèle sont
  calculées. Tout le reste (split, persist, load) est du transport d'octets.
- **Pourquoi le format safetensors par couche ?** Choix existant justifié : safetensors permet
  un chargement tensor-par-tensor et un futur `safe_open(..., framework, device)` avec mmap
  (cf. Plan 09). L'alternative `pytorch_model.bin` (`torch.load`) est un pickle non-mmappable
  → écartée pour le streaming.

## Section transversale B — Preuve sans perte (de cette phase)

Cette phase **ne modifie aucun octet ni calcul** : c'est une lecture/cartographie. Preuve
triviale par construction (aucun `Edit`/`Write` sur le code produit ici). L'artefact est un
document. Invariant respecté par vacuité.

## Section transversale C — Tests

- **Test d'inventaire** : script qui, pour un modèle de référence, énumère
  `self.layer_names` (`airllm_base.py:141-143`) et vérifie qu'il existe exactement un shard
  `.safetensors` + `.done` par nom (`safetensor_model_persister.py:20-25`). Oracle : la somme
  des tailles de shards == taille des poids d'origine (±entête safetensors).
- **Test de flux** : exécuter `forward` en `profiling_mode=True` (`airllm_base.py:414-419,
  625-634`) sur un petit modèle (ex. un Llama jouet) et vérifier que chaque couche est
  chargée exactement une fois par token.

## Section transversale D — Stress tests

- **RAM saturée** : forcer `max_seq_len` élevé → confirmer que le pic mémoire suit la plus
  grosse couche + masque seq² (`airllm_base.py:429`), pas le modèle entier. Mesurer le point
  de rupture OOM.
- **Carte SD lente** : mesurer le temps de `load_layer_to_cpu` (`airllm_base.py:269-300`) sous
  `profiling_mode` et vérifier que le profiler capture `load_safe_tensor`
  (`airllm_base.py:280`).

## Section transversale E — Mitigation

- **Risque** : la cartographie omet un chemin d'arch spécifique (chatglm a `rotary_pos_emb`,
  `airllm_base.py:236-238, 265-267`). **Parade** : re-vérifier `set_layer_names_dict` de
  chaque classe d'arch avant d'optimiser cette arch. **Plan B** : restreindre le périmètre
  initial à Llama/Mixtral (les seuls visés par l'objectif 1T) et marquer les autres
  `[HORS-PÉRIMÈTRE]`.
