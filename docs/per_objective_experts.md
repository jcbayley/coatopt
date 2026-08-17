# Per-objective experts (MoE) for constrained multi-objective HPPO

## Problem

The sequential trainer uses one policy, conditioned on the target objective and
constraint thresholds through the observation. That policy has to express a
different behaviour for every (target, threshold) combination. The low-risk
behaviour that scores acceptably under *every* draw is to camp in the region
that satisfies all constraints (e.g. low absorption with decent transmission),
so the policy averages instead of specialising, and the front's extremes are
under-explored. Reliability across seeds suffers for the same reason: whether
the conditioning gets learned is a coin flip.

## Design

One expert per objective. Expert i always optimises objective i as its target,
subject to the usual constraint ladder on the other objectives. No expert ever
has to represent conflicting targets, so each one can live in its natural
region of design space (the transmission expert can spend the absorption
budget; the CTN expert can sacrifice transmission).

- **Experts**: n_objectives independent policy+value networks (optionally a
  shared trunk with per-objective heads later; start fully separate).
- **Rollouts**: round-robin — each expert collects its own episodes with its
  own target and constraint draw; separate PPO update per expert.
- **Shared Pareto archive**: all experts stage candidates into the same
  environment archive (existing `_stage_pareto_candidate` /
  `flush_pareto_candidates` machinery, unchanged).
- **Knowledge sharing via BC**: each expert's update keeps the existing BC term
  toward the shared archive. This is the only coupling between experts — the
  transmission expert imitates good designs the absorption expert found, and
  vice versa, without sharing gradients. Reward scales between objectives stop
  mattering because no network ever mixes two targets.
- **Warmup**: trivial — each expert warms up on its own objective from episode
  zero (no objective cycling needed). `warmup_best` per objective comes from
  its own expert.

## Relation to existing code

- `train_hppo_multiagent.py` and `train_sac_multiagent.py` already run
  multiple agents with per-agent constraint schedules and a shared archive —
  closest starting points. Check how stale they are vs `train_hppo_sequential`
  (which has the newest buffer/LSTM/BC code) before choosing to adapt vs
  rebuild.
- Probably simplest: new `train_hppo_moe.py` cloning the sequential trainer's
  episode loop, holding a list of (policy, agent, buffer) triples, with the
  reset() target/constraint logic reduced to "expert i targets objective i".

## Evaluation

Same total episode budget as a sequential run, split across experts, 10 seeds:
- reliability = fraction of seeds whose front reaches the quarter-wave
  reference region (or hypervolume distribution across seeds), vs the
  sequential baseline;
- check specifically whether the transmission expert occupies the
  high-absorption / low-T corner that the sequential policy avoids;
- also run a same-budget-per-expert variant to separate "MoE helps" from
  "MoE needs more episodes".

## Pairs well with (separate changes, don't bundle into the first test)

- Constraint thresholds defined in physical decades instead of
  `frac x warmup_best` — removes reward-scale sensitivity from the constraint
  coupling.
- Diagnostic to keep: distance of each expert's episodes from its constraint
  floor (an expert sitting far above the floor is not spending its budget —
  the camping signature).
