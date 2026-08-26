# Ideas for scaling coating-design RL to many layers

CoatOpt finds Pareto-optimal optical coatings (reflectivity, absorption, thermal
noise, thickness) with constrained PPO. It works well for few layers but becomes
slow and unstable as the layer count grows. This document reviews relevant ideas
from RL and adjacent fields and proposes concrete changes, ordered by expected
impact.

## Why large-layer is hard in this setup

The current pipeline: sequential layer-by-layer construction; hybrid
discrete (material) + continuous (thickness) actions; an LSTM over a padded
`max_layers` stack; a **terminal** log-distance reward from a numpy
transfer-matrix merit function; on-policy PPO with GAE; an objective-cycling
curriculum with fixed soft-hinge constraint penalties.

Four things degrade with depth:

1. **Sparse terminal credit over a long horizon.** Reward only lands when the
   full stack is complete, and PPO/GAE must propagate it back through ~50 hybrid
   decisions. Advantage variance grows with depth.
2. **Combinatorial exploration with only entropy annealing.** Per layer the agent
   picks a material *and* a continuous thickness; over many layers the search
   space explodes, and the only exploration knob is entropy decay. Mode collapse
   onto a repeated motif is the likely failure.
3. **Sharp, multimodal reward landscape.** Thin-film reflectivity is resonant:
   near quarter-wave, small thickness changes swing the reward hard.
   Score-function (REINFORCE/PPO) gradients on the continuous head are
   high-variance on exactly this kind of surface.
4. **On-policy sample hunger + non-stationary curriculum.** PPO discards data
   each update, and objective cycling + per-objective value heads make the target
   non-stationary, so each phase partly re-learns.

The single biggest missed lever: **the merit function is pure numpy** — the
transfer-matrix physics is analytically differentiable, but it is treated as a
black box. That discards exact reward gradients that would address (1)-(3) at
once.

## Tier 1 — highest leverage

### 1. Exploit differentiable physics (port merit to PyTorch/JAX) — *implemented in this branch*

The transfer-matrix reflectivity, Brownian thermal noise, and analytic absorption
are smooth functions of the layer optical thicknesses. Reimplementing the merit
function in an autodiff framework gives `d(reward)/d(thickness)` for free. Two
payoffs:

- **Analytic policy gradients for the continuous head.** Backprop reward straight
  into `thickness_mean` instead of using high-variance score-function estimates
  (cf. stochastic value gradients / differentiable simulation). Turns the hardest,
  sharpest sub-problem into a smooth, low-variance update. *Implemented* in the
  sequential HPPO trainer behind `analytic_thickness_grad = true` /
  `analytic_grad_weight` (default off). Per episode, `d(reward)/d(thickness)` is
  computed on the final stack via `compute_thickness_reward_grad`, aligned to the
  placement steps, and applied as an additive deterministic-mean policy-gradient
  loss on the thickness head — the discrete material head and PPO value/entropy
  terms are untouched. Keeps material and thickness jointly, sequentially decided
  (no bilevel decomposition); only the thickness gradient estimator changes.
- **Cheap local refinement (memetic RL).** RL chooses the discrete structure
  (materials, layer count); every candidate is finished with a few steps of
  gradient descent / L-BFGS on thicknesses before scoring. This is how thin-film
  design is done classically (needle optimization). RL explores structure;
  gradients nail the continuous part.

Status: a differentiable `merit_function_torch` plus a `refine_thicknesses`
memetic optimizer live in
`src/coatopt/environments/utils/differentiable_physics.py`, validated against the
numpy physics. See that module and `tests/test_differentiable_physics.py`.

### 2. Dense per-layer reward via partial-stack merit (potential-based shaping)

The simulator is fast, so compute the merit of the *partial* stack after each
layer and give a potential-based shaping reward `F = γΦ(s') − Φ(s)` with
`Φ = partial-stack merit`. Potential-based shaping is policy-invariant (Ng et
al. 1999), so it cannot distort the Pareto set, but it converts sparse terminal
credit into a dense signal — directly attacking failure (1). Composes with
everything below.

## Tier 2 — strong structural changes

### 3. GFlowNets for the constructive generator

Building an object by a sequence of discrete steps and sampling proportional to
reward is the canonical GFlowNet setting. Multi-Objective GFlowNets (MOGFN, Jain
et al. 2023) return Pareto-diverse candidate sets and are markedly more stable
than mode-seeking policy gradient — they do not collapse. This maps onto the
material-choice sequence directly (thicknesses via continuous-GFlowNet or the
gradient refinement in idea 1). Since the deliverable is a Pareto *set*, this may
be a better-matched paradigm than PPO. Higher rewrite cost, high upside.

### 4. Quality-Diversity (MAP-Elites / PGA-ME) as a robust complement

Maintain an archive of best designs binned over a behavior descriptor space
(e.g. `n_layers × total_thickness`, or dominant-material fraction). QD is very
robust on rugged multimodal landscapes and naturally yields a diverse elite set
approximating a Pareto front. Policy-Gradient-Assisted MAP-Elites (PGA-ME) fuses
this with an RL/gradient inner loop — pairs well with idea 1's local optimizer.
Lower-risk route to "diverse, stable, high-performing set" than the GFlowNet
rewrite.

### 5. Transformer over LSTM for the stack encoder

An `attn` pre-network already exists in the `hppo/core` path. For 50+ layers,
self-attention propagates credit and captures long-range interference (a layer's
effect depends on all others) better than a 2-layer LSTM squeezing everything
into a 32-d hidden state. Switch the sequential trainer's encoder to the
transformer; cheap relative to the gains.

## Tier 3 — stability fixes to the existing agent

### 6. Lagrangian constraints instead of a fixed penalty

`constraint_penalty = 5.0` is brittle: too low ignores constraints, too high
crushes the objective. Use a PID-Lagrangian / CRPO-style auto-tuned multiplier
per constraint to hit the target threshold. Drops into the existing soft-hinge
machinery.

### 7. Depth curriculum (progressive layer growing)

The curriculum is over *objectives* but not *depth*, yet depth is the stated
failure axis. Start at 6-10 layers where it already works, train to competence,
then grow `max_layers` (warm-starting the policy). Analogous to progressive
growing in GANs.

### 8. Warm-start the Pareto / BC buffer with analytic designs

BC-from-Pareto already exists. Seed the buffer with known quarter-wave (QWOT)
Bragg-stack solutions and their perturbations, so the agent starts near the good
manifold instead of discovering interference from scratch at depth.

### 9. Off-policy sample efficiency

PPO's on-policy waste hurts at scale. With differentiable physics (idea 1) the
gradient path is far more sample-efficient than any model-free method here.
Otherwise a replay-based hybrid (off-policy critic) recycles the expensive
rollouts.

## Suggested sequencing

1. Port physics to autodiff (idea 1) — unlocks 1, 2, 4, 9. **(done)**
2. Add partial-stack shaping + depth curriculum + analytic warm-start (2, 7, 8) —
   cheap, immediate stability.
3. Swap LSTM→transformer and penalty→Lagrangian (5, 6) — modest effort.
4. Evaluate the bigger paradigm bet: MOGFN (3) or PGA-ME (4) for generating the
   diverse Pareto set.
