# Toward a theory of learnable capability description length

*Working draft — definitions, first lemmas, and falsifiable conjectures.
This document does what founding a piece of mathematics actually
requires: a precise definition of an object the field keeps gesturing at
without naming, the elementary results that make it well-behaved, and
open problems stated so that they can be lost.*

## 0. Intellectual ancestry (what this is NOT the first of)

The static object below is a transport of **Kolmogorov's structure
function** and **sophistication** (Kolmogorov 1974; Vereshchagin &
Vitányi 2004) from strings to task distributions, in the spirit of MDL
(Rissanen 1978). What we believe is genuinely under-developed is the
**resource- and algorithm-bounded** version (Def. 3) and its gap (Def. 4)
— the object that every empirical law of modern deep learning (capacity
≈ 2 bits/param, quantization cliffs, the Densing law) is secretly a
statement about.

## 1. Definitions

Fix a universal programming language U, a task distribution D over
inputs, a loss ℓ, and a tolerance ε ≥ 0.

**Definition 1 (Capability).** The ε-capability of D is the set
F(D, ε) = { f : L_D(f) ≤ ε } of functions performing at level ε on D.

**Definition 2 (Capability Description Length).**
CDL_ε(D) = min { |p| : p a U-program, ⟦p⟧ ∈ F(D, ε) }.
The shortest program — any program, weights or code — that *has* the
capability. Uncomputable in general (it inherits Kolmogorov's
uncomputability), but bounded above by every artifact we can exhibit.

**Definition 3 (Learnable Capability Description Length).** For a family
of learning algorithms A (e.g., SGD variants over parametric
architectures) and a compute budget C:
LCDL_ε^{A,C}(D) = min { |encode(θ)| : θ reachable by some a ∈ A within
compute C, with L_D(θ) ≤ ε },
where |encode(θ)| is the shortest encoding of the learned parameters
(quantization + entropy coding included). Unlike CDL this is an
*empirical* quantity: every trained-and-compressed model is an upper
bound, and the field's entire compression literature is a battle to
lower it.

**Definition 4 (Learning efficiency gap).**
Γ_ε^{A,C}(D) = LCDL_ε^{A,C}(D) / CDL_ε(D) ≥ 1 (Lemma 3).
Γ measures how far *learning* is from *description*: how much longer the
programs found by optimization are than the shortest programs that exist.

## 2. First lemmas

**Lemma 1 (Monotonicity, subadditivity).** CDL_ε(D) is non-increasing in
ε, and CDL_ε(D₁ ⊕ D₂) ≤ CDL_ε(D₁) + CDL_ε(D₂) + O(log).
*Proof.* Weakening the requirement can only shrink nothing; for
subadditivity, concatenate the two programs with an O(log)-bit
dispatcher on the task tag. ∎

**Lemma 2 (Fact floor — the 12 GB theorem).** If D requires recalling N
independent uniform values of b bits each with error rate ≤ δ, then
CDL_ε(D) ≥ N·b·(1 − H₂(δ)/b − δ·log₂(2^b − 1)/b) ≈ N·b for small δ.
*Proof.* A program with the capability is a lossy code for the value
table; apply the rate–distortion / Fano bound to the (uniform) source. ∎
*This is the precise form of "1T params of facts cannot fit in 12 GB",
and the yardstick behind `factstore.density_report()`.*

**Lemma 3 (Learning cannot beat description).**
LCDL_ε^{A,C}(D) ≥ CDL_ε(D) for every A, C.
*Proof.* A learned-and-encoded θ achieving ε IS a program achieving ε. ∎

**Lemma 4 (Universal search closes the gap, at a price).** If A contains
Levin universal search with unbounded C, then LCDL → CDL + O(1).
*Proof.* Levin search enumerates programs in description-length order
(THEORY.md, Thm 7a); given enough compute it finds a shortest one. ∎
*So Γ > 1 is entirely a story about bounded compute and restricted
algorithm families — i.e., about the real world.*

**Proposition 5 (The gap can be exponential — assembled from known
theorems).** There are task families where CDL is O(n) bits yet every
statistical-query learner (which includes noisy-gradient SGD) needs
2^Ω(n) queries to reach ε: parity functions on n bits.
*Proof sketch.* A parity is described by its n-bit mask, so
CDL = n + O(1) (Def. 2). Kearns (1998) proved parities require 2^Ω(n)
statistical queries; Abbe & Sandon (2020) and successors place
noisy-SGD-trained networks inside the SQ frame. Hence within any
realistic compute budget, LCDL is undefined-or-huge while CDL stays
linear: Γ is effectively unbounded. ∎
*Moral: the shortest program for a capability can be
information-theoretically tiny and simultaneously invisible to gradient
descent. Compression limits and learning limits are DIFFERENT walls —
conflating them is the standard error this framework is built to
prevent.*

## 3. Falsifiable conjectures

**Conjecture A (Universal parametric capacity constant).** For fact
tasks (Lemma 2 setting) and gradient training to convergence with
sufficient exposures, the extractable information per parameter of the
*trained weights* approaches an architecture-independent constant
κ ≈ 2 bits/param (int-8-robust, destroyed below ~4 bits/weight).
*Status:* transformers: κ ≈ 2 (Allen-Zhu & Li, arXiv:2404.05405). Our
bench, plain MLP at 23k params: κ ≥ 1.58 (lower-bound estimator;
docs/EXPERIMENTS.md, E1). Refute it with one architecture measurably off
the constant after controlling exposures and the estimator — or explain
the constant from first principles (a rate–distortion analysis of SGD
noise is the obvious attack).

**Conjecture B (Densing = Γ decay, with an asymptote).** The empirical
"Densing law" (capability density doubling ~3.5 months,
arXiv:2412.04315) is the community lowering LCDL at fixed ε by
algorithmic progress; it must flatten at Γ → Γ_min(A) > 1 for the
SGD-family, strictly above the CDL floor by Proposition 5's obstruction.
*Prediction:* density growth for a fixed capability saturates within the
decade; the saturation level identifies Γ_min empirically.

**Conjecture C (Program-augmented systems change the exponent, not just
the constant).** For capabilities with algorithmic structure (Lemma 2
does NOT apply), a reasoner+interpreter+library system (THEORY.md,
concept architecture) achieves LCDL polylog in the fact-equivalent size —
because procedures compress as programs (Kolmogorov) while weights pay
Conjecture A's linear rate. *This is the formal version of "knowledge as
programs beats knowledge as weights", and the bench can test it: measure
LCDL for algorithmic task families with and without an interpreter in the
loop.*

## 4. Why this could be "mathematics worthy of 2050"

Not because the definitions are hard — because they make previously
incommensurable things comparable on one axis (bits): quantization
research is upper-bounding LCDL; capacity scaling laws are measuring
LCDL/params; hardness-of-learning results are lower-bounding Γ;
the Hutter prize and the Densing law are time series of LCDL. A single
object underneath five literatures is exactly the situation before
Shannon unified telegraphy, cryptography and thermodynamic entropy.
The 2050-grade work is proving Conjecture A or B — or refuting them with
the instrument in this repository.
