# Plan 09 — Rupture (paradigm shift) : les idées hors-cadre, sans perte

Cadre : on change la règle du jeu, pas l'incrément. Toutes les ruptures retenues sont **sans
perte** au sens de l'invariant (par construction ou oracle). Familles avec perte (quantization,
pruning, distillation) **exclues**.

---

## RUPTURE 1 (★ PARI PRINCIPAL) — Streaming par EXPERT routé, pas par couche

### L'idée
Le streaming AirLLM lit l'unité « couche ». Pour un MoE, une couche = routeur + N experts. Le
forward dense charge les N experts puis le gating en sélectionne k (`airllm_mixtral.py:8-14`
hérite du streaming dense → **tous** les experts traversent le disque). **Rupture : changer
l'unité de streaming de la *couche* à l'*expert routé*.** On exécute le routeur d'abord, puis on
ne lit du disque que les k experts dont le poids de gating est non nul.

### Gain chiffré
Pour un MoE à N experts, k actifs : facteur de réduction des octets d'experts ≈ `N/k`.
- Mixtral 8×7B : N=8, k=2 → ÷4 sur la part experts.
- MoE 1T type Kimi K2 (~384 experts routés, ~8 actifs `[HYPOTHÈSE chiffres publics]`) → ÷~48 sur
  la part experts. Combiné au backbone caché (A5), octets/token ≈ experts actifs seuls ≈ ~32 B
  params bf16 ≈ 64 Go au lieu de ~2 To dense. **C'est le facteur qui fait passer de
  « heures/token » à « minutes/token » (Plan 06-R1).**

### PREUVE sans perte (par construction → ε = 0)
Soit la sortie d'une couche MoE : `y = Σ_{e=1}^{N} g_e · E_e(x)`, où `g_e` est le poids de
gating (top-k : `g_e = 0` pour `e ∉ top-k`).
1. Les termes `e ∉ top-k` valent `0 · E_e(x) = 0` (vecteur nul, activations finies).
2. En IEEE-754, `s + 0.0 = s` pour tout `s` fini (le seul cas pathologique `−0.0` est neutralisé
   par convention d'arrondi ; activations finies ⇒ pas de NaN/Inf).
3. Donc `Σ_{e∈top-k} g_e·E_e(x) = Σ_{e=1}^{N} g_e·E_e(x)` **au bit près**, *à condition de
   sommer dans le même ordre que la référence*.
⇒ Ne pas lire les experts hors top-k laisse `y` **bit-à-bit identique**. `ε = 0`.

**Condition de validité (à prouver par oracle)** : reproduire exactement (a) la fonction de
gating et son top-k (incl. départage des égalités — Plan 06-R5), (b) l'ordre de sommation des
experts. Oracle bit-à-bit : `y_sparse == y_dense` sur le jeu adverse-MoE (Plan 05).

### Mini-démonstration
Sur Mixtral 8×7B (tient en streaming AirLLM aujourd'hui) : exécuter le forward dense (tous
experts) et le forward sparse (top-2), comparer `torch.equal(logits)`. Attendu : True. C'est le
verrou de preuve du Sprint 2 (Plan 07).

### Risque & mitigation
- **R5** (égalités de gating) → départage figé identique à la réf ; sinon sur-ensemble (sans
  perte, gain moindre). 
- Granularité des shards par expert : beaucoup de petits fichiers → I/O aléatoire sur SD lente.
  Mitigation : regrouper les experts d'une couche en un fichier indexé, lire par offset (mmap
  A2) ; lecture séquentielle des seuls experts routés via `safe_open`.

---

## RUPTURE 2 — Le disque EST la mémoire : mmap + page cache comme RAM virtuelle infinie

Plutôt que `load_file` (copie complète en RAM anonyme, `safetensor_model_persister.py:36-37`),
mapper les shards en mémoire (`safe_open`/mmap). Les poids deviennent des pages gérées par
l'OS : un expert chaud re-routé est servi depuis le page cache **sans relecture disque**, sans
copie, sans changer un bit. **Rupture conceptuelle** : on cesse de distinguer « RAM » et
« disque » ; on expose 2 To de poids comme un espace adressable et on laisse l'OS arbitrer.
- **Sans perte** : mmap ne change que *où/quand* l'octet est lu. ε = 0 par construction.
- **Gain** : transforme A1 (par-token) en un système où la *localité temporelle* du routage est
  exploitée gratuitement (R4). Synergie directe avec Rupture 1.

---

## RUPTURE 3 — Persistance de l'état entre tokens (anti-amnésie)

Aujourd'hui chaque token : reboot du modèle (`airllm_base.py:421-423`), KV-cache jeté
(`airllm_base.py:410-412`), backbone relu. **Rupture** : traiter l'inférence comme un *processus
à état persistant* — squelette méta conservé, KV-cache mémoïsé (exact), backbone dense résident
en RAM (il est petit devant les experts). On ne « repart pas de zéro » à chaque token.
- **Sans perte** : conserver un résultat ≠ le recalculer ⇒ ε = 0 ; le KV-cache est une
  mémoïsation exacte de l'attention causale.
- **Gain** : supprime le recompute quadratique (O(N²·W)→O(N·W)) et l'overhead de reboot.

---

## RUPTURE 4 — Compression SANS PERTE adaptée aux poids (octets, pas valeurs)

Les shards safetensors sont stockés bruts. **Rupture** : appliquer une compression *réversible*
(zstd niveau adapté, ou un codec exploitant la structure des exposants bf16) sur les octets, et
décompresser à la volée sur ARM. ε = 0 (décompression exacte byte-à-byte).
- **Gain honnête** : les mantisses bf16/fp16 sont quasi-aléatoires → ratio attendu modeste
  (~1,1–1,3×). Mais sur la part *backbone* relue souvent, ou pour réduire l'empreinte disque
  d'un 1T (faisabilité de stockage sur une seule SD), c'est un gain réel et gratuit en parité.
- **Trade-off** : surcoût CPU de décompression vs gain d'octets lus. Rentable seulement si
  `temps_decompression < octets_gagnés/débit`. À mesurer sur SoC (Sprint 6).

---

## RUPTURE 5 — Reconstruction déterministe : ne pas stocker ce qu'on peut recalculer

Certains tenseurs sont *dérivables* exactement d'autres (poids liés `tie_weights`,
`airllm_base.py:227 ; rotary embeddings calculés `airllm_base.py:236-238`). **Rupture** : ne
streamer que les tenseurs « sources » et **reconstruire** les dérivés à la volée par une
fonction déterministe — c'est sans perte par définition (même fonction que la réf). Gain
marginal mais nul risque. Étend l'idée « état persistant » : on remplace de l'I/O par du compute
déterministe quand le compute est moins cher que la lecture (vrai sur SD lente).

---

## Classement & désignation

| Rang | Rupture | Gain | Sans perte | Risque |
|---|---|---|---|---|
| **1 (PARI PRINCIPAL)** | Streaming par expert routé | ÷ N/k (ordre de grandeur) | ε=0 par construction | moyen (R5) |
| 2 | mmap = RAM virtuelle | localité gratuite | ε=0 | faible |
| 3 | État persistant (KV+backbone+squelette) | O(N²)→O(N) + overhead | ε=0 | faible/moyen |
| 4 | Compression sans perte octets | ÷1,1–1,3 | ε=0 | faible |
| 5 | Reconstruction déterministe | marginal | ε=0 | très faible |

**Pari principal = Rupture 1.** C'est la **seule** qui casse le mur de bande passante (Plan
06-R1) d'un ordre de grandeur en restant sans perte. Les ruptures 2–5 sont ses multiplicateurs
et amortisseurs ; aucune ne suffit seule. Démonstration que le pari ne touche jamais le résultat
du modèle : §« PREUVE sans perte » de la Rupture 1 (somme inchangée par ajout de termes nuls,
ordre de sommation préservé, top-k reproduit à l'identique → oracle bit-à-bit).

---

## Section transversale A — Justification

On retient ces 5 ruptures parce qu'elles couvrent les 3 seuls leviers sans-perte d'octets/token :
**ne pas lire l'inutile** (R1, exactitude de la sparsité), **réutiliser le déjà-lu** (R2, R3),
**réduire/recréer les octets** (R4, R5). C'est exhaustif par construction : tout octet non lu
sans perte est soit nul (R1), soit déjà en mémoire (R2/R3), soit reconstructible/compressible
(R4/R5). Aucune autre famille sans-perte n'existe → le classement est complet, pas arbitraire.

## Section transversale B — Preuve sans perte

Fournie par rupture (chaque §« PREUVE/sans perte »). R1 vise ε=0 par neutralité du zéro IEEE-754
+ ordre préservé (oracle). R2/R5 ε=0 par construction. R3 ε=0 (mémoïsation). R4 ε=0
(décompression exacte). Toute rupture dont la preuve n'est pas atteinte reste interdite (Plan 04).

## Section transversale C — Tests

- R1 : `test_moe_bitexact` (dense vs sparse) sur jeu adverse.
- R2 : `test_mmap_identity` (mmap vs load_file byte-identique).
- R3 : `test_kv_parity` + `test_no_reboot_identity`.
- R4 : `test_lossless_roundtrip` (compress→decompress == original).
- R5 : `test_reconstruction_exact` (dérivé reconstruit == dérivé stocké).

## Section transversale D — Stress tests

- R1 sous prompt adverse-MoE (diversité max d'experts) + SD lente : mesurer si le gain tient.
- R2 sous RAM saturée : le page cache évince-t-il sans corrompre ? logits stables.
- R3 sur contexte long (KV volumineux) : parité + non-OOM.
- 72 h combiné : toutes ruptures actives, vérifier parité début vs fin (run lent vs rapide →
  mêmes logits, prouve que vitesse n'altère pas le résultat).

## Section transversale E — Mitigation

- **R1 échoue (R5 non levable)** → sur-ensemble d'experts (sans perte, gain réduit), puis repli
  sur R2+R3+R4 (gains de second ordre) ; si insuffisant, verdict d'impossibilité chiffré
  (Plan 06).
- **R2 conflit avec éviction** (Plan 08-E) → cache mmap désactivable par couche.
- **R4 non rentable sur ARM** (décompression trop lente) → désactiver, garder shards bruts.
- **Plan B global** : si même R1+R2+R3 ne franchissent pas le seuil offline du Sprint 1, la
  mission bascule sur le livrable honnête « mur physique chiffré + débit offline maximal », ce
  que l'énoncé autorise explicitement.
