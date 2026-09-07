# Target Proximity Metrics — Research Review

A comprehensive survey of methods to quantify how well multi-objective optimisation results fall near user-defined targets. Organised for review and discussion.

---

## Your Problem at a Glance

| Aspect | Details |
|--------|---------|
| **Objectives** | 3: reflectivity (maximise), absorption (minimise), thermal_noise (minimise) |
| **Targets** | `reflectivity → 1.0`, `absorption → 0.0`, `thermal_noise → 1e-23` |
| **Bounds** | `reflectivity: [0.0, 0.999999]`, `absorption: [1000, 0.1]`, `thermal_noise: [1e-24, 1e-19]` |
| **Output** | Pareto front of ~50–300 solutions per run, across 10+ runs per config |
| **Existing metric** | Hypervolume (already implemented via pymoo) |
| **Goal** | Paper-worthy metric(s) to answer: *"How many solutions actually meet the targets?"* |

> [!NOTE]
> Your objectives span wildly different scales (reflectivity ~0.99, absorption ~0.01–1000 ppm, thermal_noise ~1e-24 to 1e-19). Any distance-based metric **must** handle normalisation or log-scaling carefully, or one objective will dominate.

---

## Tier 1: Intuitive / Domain-Specific Metrics

These are easy to explain to non-MOO audiences (physicists, engineers, paper reviewers).

---

### 1. Target Region Yield (your "3D cube" idea)

**What it measures:** Fraction of Pareto solutions that simultaneously satisfy all objectives within user-defined acceptance thresholds.

**How it works:**
- Define per-objective acceptance thresholds (absolute or percentage-of-range)
- A solution "passes" if it meets ALL thresholds simultaneously
- Report: count, fraction, and optionally a **yield curve** (sweep thresholds)

**Example:**
```
Reflectivity ≥ 0.99999  → 87/150 pass (58%)
Absorption   ≤ 1.0 ppm  → 42/150 pass (28%)
Thermal_noise ≤ 5e-23   → 63/150 pass (42%)
ALL simultaneously       → 23/150 pass (15.3%)  ← headline number
```

**Pros:**
- ✅ Most intuitive — directly answers "how many usable solutions"
- ✅ Easy to explain in a paper
- ✅ Naturally extends to N objectives
- ✅ The yield curve (threshold vs. yield) is very informative

**Cons:**
- ⚠️ Threshold choice is somewhat arbitrary
- ⚠️ Binary (inside/outside) — doesn't distinguish "just missed" from "far away"

**Implementation:** Easy — pure numpy, no dependencies.

---

### 2. Per-Objective Target Achievement Rate

**What it measures:** For each objective independently, what fraction of solutions meet a given threshold.

**How it works:**
- Report a table with per-objective pass rates
- Include the conjunction ("all simultaneously") as the bottom row
- Can use multiple threshold levels (strict / relaxed)

**Pros:**
- ✅ Shows **which objective is the bottleneck** — very useful for domain insight
- ✅ Easy to present as a table in a paper
- ✅ Combines naturally with Metric 1

**Cons:**
- ⚠️ Same threshold-choice issue as Metric 1
- ⚠️ Per-objective rates don't capture trade-off structure

**Implementation:** Trivial — subset of Metric 1.

---

### 3. Normalised Target Distance (per-solution scalar)

**What it measures:** A continuous distance from each solution to the target point in normalised objective space.

**How it works:**
- Normalise each objective to [0, 1] using `objective_bounds`
- Flip directions so target = origin for all objectives
- Compute Euclidean (L2), Chebyshev (L∞), or weighted distance
- Report: **min**, **mean**, **median**, **std** across the Pareto front

**Formula:**
```
d_i = sqrt( Σ_j  w_j × ( (value_j - target_j) / range_j )² )
```

**Distance metric variants:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Euclidean (L2)** | `√(Σ Δ²)` | Overall closeness |
| **Chebyshev (L∞)** | `max(Δ)` | Worst-objective closeness (no hiding behind good objectives) |
| **Manhattan (L1)** | `Σ |Δ|` | Sum of deviations |
| **Weighted Euclidean** | `√(Σ w_j Δ²)` | Allows prioritising objectives |

> [!TIP]
> **The Chebyshev (L∞) distance is particularly interesting** for your problem. It tells you "what's the worst-case deviation across all objectives" — which is exactly what a coating designer cares about: a coating that's great on reflectivity but terrible on absorption is still unusable.

**Log-space consideration:**
Since absorption spans 0.01–1000 ppm and thermal_noise spans 1e-24 to 1e-19, normalising in **log-space** for these objectives would be much more meaningful:
```python
# For absorption (minimize, target=0):
normalised_absorption = log10(absorption) / log10(upper_bound)
# For thermal_noise (minimize, target=1e-23):
normalised_tn = (log10(thermal_noise) - log10(target)) / (log10(upper) - log10(target))
```

**Pros:**
- ✅ Continuous — distinguishes "nearly there" from "far away"
- ✅ Single scalar per solution, easy to aggregate
- ✅ The **CDF of distances** is very informative (see Metric 4)
- ✅ L∞ variant is physically meaningful for this domain

**Cons:**
- ⚠️ Choice of norm matters
- ⚠️ Normalisation choices affect results

**Implementation:** Easy — numpy only.

---

### 4. Cumulative Distance Distribution (CDF Plot)

**What it measures:** The cumulative distribution of target distances, showing how solutions cluster around the target.

**How it works:**
- Compute normalised distance for all solutions (Metric 3)
- Plot CDF: x = distance threshold, y = fraction of solutions within that distance
- Overlay multiple runs/algorithms on same plot for comparison

**Example interpretation:**
- A CDF that rises steeply near d=0 → many solutions very close to target
- A CDF that rises slowly → solutions spread out, few near target
- Compare two algorithms: the one whose CDF is "above" the other (at every d) is strictly better

**Pros:**
- ✅ Much richer than a single number — shows distribution shape
- ✅ Directly comparable across runs/algorithms
- ✅ Subsumes Metric 1 (yield at a threshold = CDF value at that threshold)

**Cons:**
- ⚠️ Not a single number — harder to put in a table (use AUC of CDF as scalar?)

**Implementation:** Easy — numpy + matplotlib.

---

## Tier 2: Standard MOO Literature Metrics

These are well-established in the multi-objective optimisation community. Paper reviewers will recognise them.

---

### 5. Inverted Generational Distance (IGD / IGD+)

**What it measures:** Average distance from a reference set to the obtained Pareto front.

**How it works:**
- Normally uses a densely-sampled reference Pareto front
- **For your case:** Use the **target point as a single-point reference**
  - IGD then reduces to: distance from target point to the nearest Pareto solution
  - This gives you "how close did the best solution get to the target?"
- Or: construct a **reference front** from the best solutions across all runs (pooled best-known front)

**Available in pymoo:**
```python
from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus

ind = IGD(reference_front)   # or IGDPlus for Pareto-compliant version
score = ind(obtained_front)
```

**Pros:**
- ✅ Well-established — strong signal for paper credibility
- ✅ Already available in pymoo (your existing dependency)
- ✅ IGD+ is weakly Pareto-compliant (better theoretical properties)
- ✅ Captures both convergence AND diversity

**Cons:**
- ⚠️ With a single target point, degenerates to "nearest point distance" — loses diversity information
- ⚠️ Requires a reference front for full power (construct from best-known solutions across runs)
- ⚠️ Not Pareto-compliant (plain IGD) — IGD+ fixes this

**Implementation:** Already available — just call pymoo.

---

### 6. Generational Distance (GD / GD+)

**What it measures:** Average distance from each obtained solution to the reference front.

**How it works:**
- Reverse of IGD: measures how far each *obtained* solution is from the reference
- GD = average distance from each solution to nearest reference point

**Available in pymoo:**
```python
from pymoo.indicators.gd import GD
from pymoo.indicators.gd_plus import GDPlus
```

**Pros:**
- ✅ Directly measures convergence
- ✅ Combined with IGD, gives a complete picture

**Cons:**
- ⚠️ Doesn't capture diversity (can have GD=0 with a single point on the front)
- ⚠️ Same reference front issue as IGD

**Implementation:** Already available — pymoo.

---

### 7. Epsilon (ε) Indicator

**What it measures:** The minimum translation/scaling needed for the obtained front to dominate the reference.

**How it works:**
- **Additive ε:** smallest ε such that for every reference point z, there exists a solution a where a_i ≤ z_i + ε for all objectives
- **Multiplicative ε:** same but with multiplication — **scale-invariant** (important for your problem!)

**Formula (additive):**
```
ε+(A, Z) = max_{z ∈ Z}  min_{a ∈ A}  max_i  (a_i - z_i)
```

**Pros:**
- ✅ Pareto-compliant
- ✅ Multiplicative version is scale-invariant (great for mixed-scale objectives)
- ✅ Single scalar that captures "how far from dominating the target"
- ✅ Easy to interpret: "our solutions need at most ε improvement to meet the reference"

**Cons:**
- ⚠️ Only measures convergence, not diversity
- ⚠️ Sensitive to outliers (uses max)
- ⚠️ Not in pymoo by default (need to implement, but it's simple)

**Implementation:** ~20 lines of numpy. Straightforward.

---

### 8. Spacing Metric (Δ)

**What it measures:** Uniformity of solution distribution along the Pareto front.

**How it works:**
- Computes distance between each solution and its nearest neighbour
- Reports standard deviation of these distances
- Lower = more uniform

**Pros:**
- ✅ Captures an important aspect: are solutions well-distributed (good for exploring trade-offs)?
- ✅ Relevant to your use case — you want diverse, explorable solutions

**Cons:**
- ⚠️ Doesn't measure closeness to target at all
- ⚠️ Supplementary metric only

**Implementation:** Easy — numpy.

---

## Tier 3: Preference-Based / Region-of-Interest Metrics

These are specifically designed for problems like yours where the user has a target/preference point.

---

### 9. R-Metric (R-HV, R-IGD) — Region-of-Interest Indicators

**What it measures:** Hypervolume or IGD restricted to a "Region of Interest" (ROI) around the target.

**How it works:**
1. Define an ROI around the target point (using Achievement Scalarizing Function or a box)
2. Filter solutions: keep only those in the ROI
3. Compute standard HV or IGD on the filtered set

**This is essentially a formalised version of your "3D cube" idea from the MOO literature!**

**Key paper:** Li, Deb, et al. (2018) "R-Metric: Evaluating the Performance of Preference-Based Evolutionary Multi-Objective Optimization Using Reference Points"

**Pros:**
- ✅ Academically rigorous version of your cube idea
- ✅ Captures both convergence (to ROI) and diversity (within ROI)
- ✅ Published methodology with citations — strong for a paper
- ✅ Can use your existing HV/IGD code, just add the filtering step

**Cons:**
- ⚠️ Not built-in to pymoo (need to implement ROI filtering, ~50-100 lines)
- ⚠️ ROI definition requires care (typically uses ASF — Achievement Scalarizing Function)
- ⚠️ More complex to explain than simple yield

**Implementation:** Moderate — ROI filtering + existing HV/IGD. Can start with simple box-based ROI.

---

### 10. Achievement Scalarizing Function (ASF) Distance

**What it measures:** The Chebyshev (L∞) distance from each solution to the target, in normalised space — but with a twist: it can identify the *closest Pareto-optimal projection* to the target.

**How it works:**
- ASF(x | z) = max_j [ w_j × (f_j(x) - z_j) ] — the weighted Chebyshev distance
- The solution minimising ASF is the "closest to target" in the Chebyshev sense
- Augmented ASF adds a small L1 penalty to ensure strict Pareto optimality

**Why it's interesting for your problem:**
- This is the standard tool in interactive multi-objective optimisation for "given a target, find the closest feasible solution"
- Could be used as a per-solution quality score, then aggregated

**Pros:**
- ✅ Theoretically grounded — projects target onto Pareto front
- ✅ Can reach any point on non-convex Pareto fronts (unlike weighted sum)
- ✅ Natural fit: your reward function already uses a log-distance to target

**Cons:**
- ⚠️ More abstract — harder to explain to non-MOO audiences
- ⚠️ Weight selection affects results

**Implementation:** Easy — ~15 lines.

---

## Tier 4: Additional Solution Quality Metrics

---

### 11. Knee Point Analysis

**What it measures:** Identifies "knee" solutions on the Pareto front where the marginal trade-off changes sharply.

**How it works:**
- Geometric: maximum perpendicular distance from front to hyperplane through extremes
- Utility-based: solutions where small gains require disproportionate losses

**Pros:**
- ✅ Identifies the "sweet spot" solutions that domain experts typically prefer
- ✅ Complementary to target-based metrics

**Cons:**
- ⚠️ Doesn't directly measure target proximity
- ⚠️ More of a solution selection tool than a performance metric

**Implementation:** Moderate.

---

## Summary Comparison

| # | Metric | What it tells you | Single scalar? | Uses targets? | In pymoo? | Impl. effort | Paper credibility |
|---|--------|-------------------|:-:|:-:|:-:|---|---|
| 1 | **Target Region Yield** | How many usable solutions | ✅ | ✅ | ❌ | Easy | ★★★ |
| 2 | **Per-Objective Achievement** | Which objective is hardest | Table | ✅ | ❌ | Easy | ★★★ |
| 3 | **Normalised Target Distance** | How close is each solution | ✅ (mean/min) | ✅ | ❌ | Easy | ★★★ |
| 4 | **CDF of Distances** | Distribution shape | Plot | ✅ | ❌ | Easy | ★★★★ |
| 5 | **IGD/IGD+** | Front quality vs reference | ✅ | Partial | ✅ | None | ★★★★★ |
| 6 | **GD/GD+** | Solution convergence | ✅ | Partial | ✅ | None | ★★★★ |
| 7 | **Epsilon Indicator** | Worst-case gap to reference | ✅ | ✅ | ❌ | Easy | ★★★★★ |
| 8 | **Spacing** | Solution uniformity | ✅ | ❌ | ❌ | Easy | ★★★ |
| 9 | **R-Metric (R-HV/R-IGD)** | Quality within target region | ✅ | ✅ | ❌ | Moderate | ★★★★★ |
| 10 | **ASF Distance** | Chebyshev projection to target | ✅ | ✅ | ❌ | Easy | ★★★★ |
| 11 | **Knee Point** | Best trade-off solutions | ❌ | ❌ | ❌ | Moderate | ★★★ |

---

## My Recommendation for the Paper

For a strong paper, I'd suggest this combination of **3–4 metrics**:

### Primary (must-have)

1. **Target Region Yield + Yield Curve** (Metric 1) — your headline number, intuitive for any reader
2. **Per-Objective Achievement Table** (Metric 2) — domain insight, shows which objective is the bottleneck

### Secondary (pick 1-2 for rigour)

3. **Normalised Target Distance with CDF** (Metrics 3+4) — continuous distribution, visually compelling, subsumes the yield as a special case
4. **R-HV or R-IGD** (Metric 9) — the academically "proper" version of your cube idea, adds credibility with MOO reviewers

### Already have

5. **Hypervolume** (existing) — keep this for overall Pareto front quality comparison between runs/algorithms

### Rationale

This combination tells the complete story:
- **"How many solutions meet the targets?"** → Yield (Metric 1)
- **"Which objectives are hardest?"** → Per-objective table (Metric 2)
- **"How does quality distribute?"** → CDF plot (Metric 4)
- **"How does overall front quality compare?"** → Hypervolume (existing)

The R-HV/R-IGD (Metric 9) adds academic rigour if targeted at an MOO-savvy venue.

---

## Implementation Notes

All proposed metrics would go in:
- [metrics.py](file:///Users/simon/Developer/Python/coatopt/src/coatopt/utils/metrics.py) — computation functions
- [compare_outputs.py](file:///Users/simon/Developer/Python/coatopt/src/coatopt/compare_outputs.py) — reporting, tables, and plots

The target and bounds info is already in the config files and accessible through `CoatingEnvironment.optimise_targets` and `CoatingEnvironment.objective_bounds`.

Estimated implementation time for all 4 recommended metrics: **2–3 hours**, including tests and integration with the comparison pipeline.
