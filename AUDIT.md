# AirLLM — Audit technique complet / Full Technical Audit

> Autopsie, audit, validation/réfutation, plan de mitigation et corrections.
> Date : 2026-06-28 — Périmètre : l'ensemble du dépôt, focus sur le paquet `air_llm/airllm`.

---

## 1. Résumé exécutif

**AirLLM** permet d'exécuter l'inférence de très grands LLM (70B → 405B) sur des
GPU à faible mémoire en chargeant le modèle **couche par couche** depuis le disque,
en exécutant chaque couche, puis en libérant la VRAM. L'idée centrale est solide et
la valeur réelle. Le code, en revanche, porte les marques d'une croissance rapide :
chemins morts, fonctionnalités cassées non testées, dépendances non reproductibles,
et absence d'intégration continue.

Verdict global : **concept valide, ingénierie fragile.** Aucun défaut de
conception bloquant, mais plusieurs bugs concrets (dont un qui rend une option
publique inutilisable, et une optimisation de performance silencieusement inopérante).

| Sévérité | Nombre | Exemples |
|----------|--------|----------|
| 🔴 Haute | 4 | `delete_original` plante ; `pin_memory()` no-op ; chemins `output_attentions/hidden_states` cassés ; retour `attentions` mal conditionné |
| 🟠 Moyenne | 5 | Reproductibilité des dépendances ; post-install réseau ; `trust_remote_code` ; absence de CI ; tests non hermétiques |
| 🟡 Basse | 6 | Messages de log trompeurs, code mort, magie de chaînes, duplication DRY |

Les correctifs sûrs et autonomes ont été **appliqués dans ce commit** (voir §6).
Les éléments nécessitant un banc d'essai GPU sont **documentés mais non corrigés à l'aveugle**.

---

## 2. Architecture (telle que validée par lecture du code)

```
AutoModel.from_pretrained()                  air_llm/airllm/auto_model.py
   └─ détecte l'architecture via config.architectures[0]
      └─ instancie une sous-classe AirLLM*  (Llama2/Mistral/Mixtral/QWen/QWen2/
                                              Baichuan/InternLM/ChatGLM/…/MLX)
            └─ AirLLMBaseModel               air_llm/airllm/airllm_base.py
                 ├─ find_or_create_local_splitted_path()   utils.py
                 │     └─ snapshot_download + split_and_save_layers()
                 │           └─ ModelPersister (Safetensor | MLX)   persist/
                 ├─ init_model()  → modèle "meta" (poids vides) via accelerate
                 └─ forward()     → boucle couche-par-couche :
                       charge (thread) → pin → vers GPU → exécute → .to("meta")
```

Points de conception **validés** (revue manuelle) :

- **Préchargement asynchrone** d'une couche pendant que la précédente s'exécute
  (`ThreadPoolExecutor`, `future.result()`) — schéma correct.
- **Modèle « meta »** via `accelerate.init_empty_weights()` pour éviter d'allouer
  la VRAM des poids — correct.
- **Marqueurs `.done`** à côté de chaque shard pour détecter les écritures
  partielles/corrompues — bonne intuition de robustesse.
- **Abstraction `ModelPersister`** (Safetensors sur Linux/Win, MLX `.npz` sur
  macOS) — séparation propre.

---

## 3. Constats détaillés — validés / réfutés

### 🔴 H1 — `delete_original=True` plante par variable non liée  *(CORRIGÉ)*
`utils.py :: remove_real_and_linked_file()`. `targetpath` n'était assigné que
si `realpath(to_delete) != to_delete`. Pour un fichier régulier (non-symlink),
`os.remove(to_delete)` puis `if (targetpath)` lève **`NameError`**. Toute la
voie `delete_original=True` (option publique du constructeur) est inutilisable
sur des fichiers non symboliques.
**Validé** par lecture + analyse statique. **Corrigé** : résolution du target
avant suppression, garde `targetpath != to_delete and os.path.exists(...)`.

### 🔴 H2 — Le préchargement `pin_memory()` est un no-op silencieux  *(CORRIGÉ)*
`airllm_base.py :: load_layer_to_cpu()`. `state_dict[k].pin_memory()` renvoie un
**nouveau** tenseur épinglé sans réassignation ; l'original reste non épinglé.
Le transfert hôte→GPU non bloquant censé être accéléré par la mémoire épinglée
ne l'était donc jamais. Coût caché de performance sur le chemin chaud.
**Validé** (sémantique PyTorch). **Corrigé** : `state_dict[k] = state_dict[k].pin_memory()`.

### 🔴 H3 — `output_attentions` / `output_hidden_states` sont cassés  *(CORRIGÉ via garde explicite)*
`airllm_base.py :: forward()`. Plusieurs défauts cumulés :
- `all_hidden_states = [] * len(self.layers)` ⇒ toujours `[]` (la multiplication
  de liste vide ne pré-alloue rien) ; les `all_hidden_states[i].append(...)`
  ultérieurs lèveraient `IndexError`.
- Ligne ~518 : `all_hidden_states[i].append(new_seq)` utilise **`new_seq` avant
  affectation** (confirmé par pyflakes : *undefined name 'new_seq'*).
- Usage incohérent : tantôt indexation `[i].append`, tantôt `+=`.
- La condition de retour utilisait `all_hidden_states` pour décider de
  `attentions` (corrigé en H4).

Ces options étaient **non fonctionnelles et non testées** (plantage garanti).
Un correctif *fonctionnel* exige un banc d'essai (petit modèle réel + comparaison
aux sorties HF), donc il n'est **pas** appliqué à l'aveugle. **Appliqué ici** :
suppression des deux `append` provablement mortes (dont celle référençant
`new_seq` avant affectation) et **garde explicite `NotImplementedError`** en tête
de `forward()`. On transforme ainsi un plantage obscur en contrat clair, sans
prétendre supporter une fonctionnalité non validée.

### 🔴 H4 — Mauvaise variable conditionne le retour `attentions`  *(CORRIGÉ)*
`airllm_base.py :: forward()`, construction de `CausalLMOutputWithPast`. La
sortie `attentions` était calculée `tuple(all_self_attns) if all_hidden_states
is not None`. Si `output_attentions` est demandé **sans** `output_hidden_states`,
`attentions` retournait `None` à tort ; dans le cas inverse, `tuple(None)`
planterait. **Corrigé** : condition basée sur `all_self_attns`.

### 🟠 M1 — Reproductibilité des dépendances
- `requirements.txt` (racine) est orienté **entraînement** (qlora) :
  `bitsandbytes==0.39.0`, `transformers @ git+…HEAD` (non épinglé → build
  non déterministe), `peft@v0.3.0`, `accelerate@v0.20.3`. Il **entre en conflit**
  avec les besoins du paquet d'inférence `air_llm` (transformers récent).
- `air_llm/setup.py` n'impose **aucune borne de version** (`transformers`,
  `accelerate`, `optimum`…). Une mise à jour amont peut casser le runtime.
**Mitigation** : séparer `requirements-train.txt` / `requirements-infer.txt` ;
ajouter des bornes minimales testées dans `install_requires`.

### 🟠 M2 — `setup.py` exécute un `pip install --upgrade transformers` au post-install
`PostInstallCommand`. Effets réseau pendant l'installation, non déterministe,
peut échouer/pendre en CI ou environnement restreint, et écrase la version de
l'utilisateur sans consentement. **Mitigation** : remplacer par une **borne de
version** dans `install_requires` (`transformers>=<min testée>`), supprimer le
hook post-install.

### 🟠 M3 — `trust_remote_code=True` partout
`auto_model.py`, `airllm_base.py` (config, tokenizer, modèle). Exécute du code
arbitraire provenant du repo HF distant. Légitime pour certains modèles, mais
c'est une **surface RCE** qui devrait être **opt-in** et documentée.
**Mitigation** : exposer un paramètre `trust_remote_code: bool = False`,
le propager, avertir dans le README.

### 🟠 M4 — Aucune intégration continue  *(CORRIGÉ — ajout d'une CI légère)*
`.github/` ne contenait que `FUNDING.yml`. Aucun lint, build, ni test
automatisé. H1 et H3 auraient été détectés par une simple passe pyflakes.
**Corrigé** : `.github/workflows/ci.yml` — byte-compile + pyflakes (gate sur
*undefined name* / *before assignment*), matrice Python 3.9 / 3.11, sans réseau.

### 🟠 M5 — Tests non hermétiques et de couverture minime
`tests/test_automodel.py` télécharge des configs depuis HF (réseau requis,
peut « flaker ») ; `test_compression.py` exige CUDA + bitsandbytes. Aucun test
unitaire pur sur la logique critique (`split_and_save_layers`, naming des
shards, `check_space`, `uncompress_layer_state_dict`).
**Mitigation** : tests unitaires hermétiques avec petits tenseurs factices et
arborescences temporaires ; marquer les tests réseau/GPU `@pytest.mark`.

### 🟡 Constats mineurs (basse sévérité)
- **B1** *(CORRIGÉ)* `str(dtype).strip('torch.')` strippe un **ensemble de
  caractères**, pas le préfixe `torch.` ; fragile pour des dtypes finissant par
  t/o/r/c/h. → `split('.')[-1]`.
- **B2** *(CORRIGÉ)* Log trompeur : « either … is available » alors que c'est le
  cas **ni l'un ni l'autre** (« neither … nor »).
- **B3** Code mort important : imports inutilisés massifs (pyflakes), gros bloc
  commenté dans `find_or_create_local_splitted_path`, variables assignées jamais
  lues (`n_seq`, `torch_dtype`, `TEST_NO_LAYERED`).
- **B4** `forward()` : `del self.model; init_model()` **ré-instancie le modèle à
  chaque appel** (nécessaire pour recharger les buffers, mais coûteux et non
  documenté). À profiler.
- **B5** Duplication DRY : la détection macOS (`platform == "darwin"`) est
  recopiée dans 5+ fichiers. Centraliser.
- **B6** `check_space()` : facteur magique `0.2813` (ratio 4-bit) non expliqué ;
  documenter ou nommer la constante.
- **B7** Conventions de nommage de shards reposant sur la présence/absence d'un
  point final (`layer_name + 'safetensors'` vs `+ '.safetensors'`) — fonctionne
  mais très fragile ; un seul refactor casserait le cache existant.

---

## 4. Sécurité

| Item | Risque | Recommandation |
|------|--------|----------------|
| `trust_remote_code=True` (défaut implicite) | RCE depuis repo HF | Rendre opt-in (M3) |
| `pip install --upgrade` au post-install | Exécution réseau non maîtrisée à l'install | Supprimer (M2) |
| `torch.load(...)` sur shards `.bin` | Désérialisation pickle non fiable | Préférer safetensors ; avertir sur `.bin` |
| Jetons HF passés en clair en paramètres | Fuite potentielle via logs | Ne jamais logger `hf_token` (déjà respecté) |

Aucun secret commité détecté. `LICENSE` (MIT) présent et cohérent avec `setup.py`.

---

## 5. Plan de mitigation priorisé

**Sprint 1 (sûr, autonome) — fait dans ce commit**
1. ✅ H1 `remove_real_and_linked_file` (NameError).
2. ✅ H2 `pin_memory()` réassigné.
3. ✅ H3 garde `NotImplementedError` + suppression du code mort.
4. ✅ H4 condition de retour `attentions`.
5. ✅ B1 dtype `split('.')[-1]`, B2 message de log.
6. ✅ M4 workflow CI léger.

**Sprint 2 (nécessite revue produit / banc d'essai)**
6. ⏳ H3 (suite) : *implémenter réellement* `output_attentions/hidden_states`
   sous test (la garde `NotImplementedError` est en place en attendant).
7. ⏳ M1/M2 : séparer les requirements, borner les versions, retirer le
   post-install réseau.
8. ⏳ M3 : `trust_remote_code` opt-in.
9. ⏳ M5 : tests unitaires hermétiques (split/naming/check_space/uncompress).

**Sprint 3 (hygiène)**
10. ⏳ B3–B7 : purge du code mort, DRY macOS, constantes nommées, robustesse du
    nommage des shards.

---

## 6. Correctifs appliqués dans ce commit

| Fichier | Changement |
|---------|-----------|
| `air_llm/airllm/utils.py` | H1 : résolution du symlink avant suppression + garde ; B1 : `split('.')[-1]` (×2) |
| `air_llm/airllm/airllm_base.py` | H2 : réassignation `pin_memory()` ; H3 : garde `NotImplementedError` + suppression des `append` mortes ; H4 : `attentions` conditionné sur `all_self_attns` ; B2 : message de log |
| `.github/workflows/ci.yml` | M4 : CI byte-compile + pyflakes (gate undefined/unbound) |
| `AUDIT.md` | Le présent rapport |

Tous les fichiers modifiés passent `python -m py_compile`. Les correctifs sont
volontairement **conservateurs** : aucune modification du comportement nominal
d'inférence, uniquement la suppression de défauts confirmés. Les fonctionnalités
non testables sans GPU/réseau sont documentées plutôt que patchées à l'aveugle.
