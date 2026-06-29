# Plan 03 — Amélioration : backlog priorisé (impact × coût × risque)

Notation : Impact (octets/token ou latence), Coût (effort), Risque (technique/sans-perte).
Score = Impact / (Coût × Risque). Tout item est **sans perte** (justifié colonne « Preuve »).
Les items avec perte sont en **annexe écartée**.

## Backlog principal (ordonné par score)

| # | Item | Impact | Coût | Risque | Preuve sans perte | Réf. code |
|---|---|---|---|---|---|---|
| A1 | **Streaming MoE par expert** : ne charger que les experts top-k routés | ★★★★★ (÷10–80 octets/token) | ★★★★ | moyen | Experts hors top-k ont gating `g_e=0` → somme inchangée | `airllm_mixtral.py:8-14`, `airllm_base.py:450-600` |
| A2 | **mmap au lieu de `load_file`** : `safe_open` + page cache OS | ★★★★ (réutilisation inter-tokens, 0 copie) | ★★ | faible | Change OÙ/QUAND on lit, pas la valeur | `safetensor_model_persister.py:36-37` |
| A3 | **Restaurer KV-cache sans perte** (supprimer `use_cache=False`) | ★★★★ (÷N_tokens sur le recompute) | ★★★ | moyen | KV-cache = mémoïsation exacte d'attention causale | `airllm_base.py:410-412` |
| A4 | **Supprimer le reboot par forward** : garder le squelette méta entre tokens | ★★ (overhead fixe) | ★★ | faible | `init_model` ne crée que des poids vides ; idempotent | `airllm_base.py:421-423` |
| A5 | **Cache RAM du backbone dense** (embed, attn, norm, lm_head) lus chaque token | ★★★ | ★★ | faible | Garder en RAM ≠ recalcul ; bits identiques | `airllm_base.py:141-143,597-600` |
| A6 | **Compression SANS PERTE des shards froids** (zstd/LZ4 sur octets safetensors) | ★★ (÷1,1–1,3 disque) | ★★★ | faible | Décompression exacte byte-à-byte | `persist/safetensor_model_persister.py:27-37` |
| A7 | **Masque d'attention paresseux** (éviter l'alloc seq²) | ★ (RAM activations) | ★ | faible | Masque ré-exprimé, valeurs identiques | `airllm_base.py:429-432` |
| A8 | **Pré-tri des shards par ordre de routage probable** (lecture séquentielle) | ★★ (débit séq > aléatoire SD) | ★★★ | moyen | Réordonnance lecture, résultat inchangé | `utils.py:188-339` |

**Pari principal = A1** (cf. Plan 09). A2/A3/A5 sont les multiplicateurs sans-perte qui le
rendent vivable. A4/A7 sont de la dette/propreté. A6/A8 sont des gains de second ordre.

## Détails des 3 premiers

### A1 — Streaming MoE par expert
- État actuel : `move_layer_to_device` (`airllm_base.py:302-323`) matérialise **tout** le
  `state_dict` de la couche, donc tous les experts. Le forward Mixtral sélectionne ensuite
  top-k *en mémoire* → les experts non utilisés ont été lus pour rien.
- Cible : (1) exécuter le routeur/gating d'abord, (2) ne charger depuis le disque que les
  shards des experts sélectionnés, (3) exécuter, (4) évincer. Nécessite de **sous-découper**
  chaque couche MoE en shards par expert dans `split_and_save_layers` (`utils.py:315`).

### A2 — mmap
- `load_file` (`safetensor_model_persister.py:37`) lit le fichier entier en RAM anonyme.
  Remplacer par `safetensors.safe_open(path, framework="pt", device="cpu")` + slicing paresseux
  laisse l'OS gérer le page cache : un expert « chaud » re-routé reste en cache → lecture disque
  évitée **sans changer la valeur**.

### A3 — KV-cache
- `airllm_base.py:410-412` force `use_cache=False` « we don't support kv cache for new version
  yet ». Le code possède déjà la plomberie KV (`airllm_base.py:434-437, 517-546, 603-608`),
  désactivée. Restaurer = mémoïser les `(k,v)` causaux ⇒ on ne recalcule plus la préfixe ⇒
  octets lus passent de `O(N²·W)` à `O(N·W)` (W = poids actifs). Mémoïsation exacte = sans perte.

## Annexe — pistes écartées (AVEC PERTE, hors voie principale)

- ❌ **Compression 4bit/8bit existante** (`utils.py:157-176`) : nf4/blockwise = quantization
  approximative → viole l'invariant. De plus CUDA-only (`.cuda()` en dur, `utils.py:162,169`)
  donc inutilisable sur Pi. README revendique « tiny accuracy loss » — perte ≠ 0 → écartée.
- ❌ Pruning d'experts « rarement utilisés » : supprime des poids → change la sortie sur les
  tokens qui les router aient routés. Écartée.
- ❌ Distillation / speculative decoding avec draft model : approxime. Écartée.

---

## Section transversale A — Justification

Priorisation par score Impact/(Coût×Risque) plutôt que par enthousiasme : A1 domine car seul
levier « ordre de grandeur ». A2/A3 sont mis avant A4 car ils réduisent des *octets lus*
(le mur), tandis que A4 ne réduit qu'un overhead CPU fixe (pas le mur). On refuse l'argument
d'autorité « le reboot est sale donc prioritaire » : mesuré, le reboot pèse `[HYPOTHÈSE]`
≪ que la lecture disque (à chiffrer Sprint 1).

## Section transversale B — Preuve sans perte

Chaque item porte sa preuve dans le tableau (colonne « Preuve sans perte »). Règle d'admission
au backlog : un item n'entre que si sa preuve est « par construction » OU planifiée par oracle
(Plan 05). A1 et A3 exigent un oracle bit-à-bit (ils touchent l'ordre des calculs) ; A2/A4/A5/
A6 sont sans perte par construction triviale.

## Section transversale C — Tests

Pour CHAQUE item du backlog, critère d'entrée en « done » = test de parité de sortie passé
(oracle Plan 05) + test unitaire de non-régression mémoire (pic RAM mesuré). A1 : test
spécifique « top-k routés == experts à poids non nul » sur ≥1000 tokens.

## Section transversale D — Stress tests

- A1 : prompt adverse maximisant la diversité d'experts (worst-case bande passante).
- A2 : carte SD quasi pleine + fragmentée → mesurer dégradation mmap.
- A3 : contexte long (proche `max_seq_len`) → vérifier que le KV-cache ne déborde pas la RAM
  (le KV-cache d'un 1T peut être volumineux ; à chiffrer).

## Section transversale E — Mitigation

- **Risque A1** : un routeur non-déterministe (ex. bruit d'arrondi sur le top-k aux égalités)
  pourrait router différemment selon l'ordre de calcul. **Parade** : figer la règle de
  départage du top-k (indice le plus bas gagne) et la prouver identique au modèle de référence.
  **Plan B** : si le routage dépend d'un état non reproductible, charger un sur-ensemble
  (top-k+marge) — toujours sans perte, gain réduit.
- **Risque A3** : taille KV-cache. **Parade** : KV-cache sur disque mmappé (sans perte).
  **Plan B** : revenir au recompute sur les couches où le KV ne tient pas.
