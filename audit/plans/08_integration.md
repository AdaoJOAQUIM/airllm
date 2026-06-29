# Plan 08 — Intégration : brancher chaque brique sans casser le reste ni la parité

## Principe d'intégration : tout passe par des points d'extension déjà existants

Le code AirLLM est conçu par héritage : `AirLLMBaseModel` + hooks surchargeables par arch
(`auto_model.py:27-45`). On **n'écrit pas un fork divergent** ; on insère les optimisations aux
coutures existantes, derrière des flags par défaut **off** (parité garantie tant qu'inactif).

### Points de couture (seams) identifiés

| Brique | Seam (où brancher) | Référence | Default |
|---|---|---|---|
| A1 MoE | nouvelle classe `AirLLMMixtralSparse(AirLLMMixtral)` + override du chargement/forward de couche | `airllm_mixtral.py:8-14`, `airllm_base.py:469,500-586` | off |
| A1 split | option `expert_sharding=True` dans `split_and_save_layers` | `utils.py:188-339` | off |
| A2 mmap | nouveau `MmapSafetensorModelPersister(ModelPersister)` sélectionné par flag | `persist/model_persister.py:10-30`, `safetensor_model_persister.py:36-37` | persister actuel |
| A3 KV | flag `lossless_kv_cache` levant `use_cache=False` | `airllm_base.py:410-412` | off |
| A4 no-reboot | flag `persist_skeleton` autour de `del/init_model` | `airllm_base.py:421-423` | off (reboot actuel) |
| A5 backbone RAM | cache dict `{layer_name: state_dict}` pour couches denses | `airllm_base.py:597-600` | off |
| A6 compress | persister compressé sans perte (zstd) sélectionnable | `persist/*` | off |
| Profiler | déjà branché | `airllm_base.py:276-298`, `profiler.py` | on en mode profiling |

### Règle d'or d'intégration

> Chaque brique est derrière un flag dont la valeur par défaut **reproduit exactement le
> comportement actuel**. Donc : `tests existants verts` ⇒ intégration neutre. La parité ne peut
> être affectée que si un flag est explicitement activé, et alors l'oracle (Plan 05) garde.

## Le ModelPersister : pivot d'intégration propre

`ModelPersister.get_model_persister()` (`persist/model_persister.py:10-30`) est un singleton qui
choisit déjà l'implémentation selon la plateforme (mlx sur macOS, safetensors sinon). C'est le
**point d'injection naturel** pour A2 (mmap) et A6 (compression sans perte) : on ajoute des
implémentations sœurs sans toucher au `forward`. Le `forward` ne connaît que l'interface
`load_model(layer_name, path)` (`model_persister.py:38`) → découplage déjà en place. ✔

## Ordre d'intégration (cohérent avec Plan 07)

1. A2 mmap (persister) — le plus isolé, aucun impact sur le forward.
2. A1 MoE (split + classe arch) — le gros morceau, isolé dans une sous-classe.
3. A3 KV — touche le `forward` (flag local).
4. A4/A5 — autour du `forward`/cycle de vie.
5. A6 — persister.

On intègre du plus isolé au plus intrusif pour limiter la surface de régression à chaque étape.

## Compatibilité multi-architectures

L'objectif 1T vise Mixtral/MoE et Llama. Les autres arch (chatglm, qwen, baichuan, internlm)
**doivent rester fonctionnelles** : comme tout est derrière des flags off, leur chemin est
inchangé. Test de non-régression : `test_automodel.py` (existant, `air_llm/tests/`) doit rester
vert sans modification.

---

## Section transversale A — Justification

On branche via héritage + flags off-par-défaut plutôt que via un rewrite parce que (a) ça rend
la non-régression **prouvable triviale** (flag off ⇒ code mort ⇒ comportement identique), et
(b) ça permet d'activer/désactiver chaque brique indépendamment pour l'attribution causale des
gains/régressions. L'alternative (réécrire `forward`) maximiserait le risque de casser la
parité partout à la fois — rejetée.

## Section transversale B — Preuve sans perte

Preuve d'intégration en deux temps : (1) **flag off** ⇒ chemin d'exécution byte-identique à
l'actuel (preuve par construction : la branche optimisée n'est pas atteinte) ; (2) **flag on**
⇒ parité déléguée à l'oracle de la brique (Plan 05) avec son `ε` (Plan 04). L'intégration
n'introduit donc aucune perte propre.

## Section transversale C — Tests

- `test_flags_off_identity` : pour chaque flag, off ⇒ `torch.equal(logits, baseline)`.
- `test_persister_interface` : un persister mmap/compressé rend le **même** `state_dict` (mêmes
  clés, mêmes valeurs byte-identiques) que le persister actuel.
- `test_automodel_nonregression` : arch non-MoE inchangées.

## Section transversale D — Stress tests

Activer **toutes** les briques ensemble (A1+A2+A3+A5+A6) et vérifier qu'elles n'interagissent
pas pour casser la parité (ex. mmap + éviction `to("meta")` `airllm_base.py:597` ne doit pas
libérer une page mmappée encore référencée). Run combiné sous RAM bridée + SD lente.

## Section transversale E — Mitigation

- **Risque d'interaction A2×éviction** : un poids mmappé évincé puis re-routé doit être
  rechargé proprement. **Parade** : que l'éviction `to("meta")` relâche la référence Python
  mais laisse l'OS gérer la page ; test dédié. **Plan B** : désactiver le cache mmap sur les
  couches MoE si conflit.
- **Risque : un flag on par erreur en prod casse la parité silencieusement**. **Parade** :
  gardien CI qui exécute l'oracle avec la config livrée. **Plan B** : journaliser la config de
  flags active à chaque run.
