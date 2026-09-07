# CoatOpt Conceptual Study Guide

## What this guide is for

This guide is designed to help you explain *why* CoatOpt works, not merely identify which Python file performs each operation. By the end, you should be able to answer four questions clearly:

1. What optimisation problem are we solving?
2. Why is it a multi-objective problem rather than a single-objective problem?
3. How do reinforcement-learning and evolutionary algorithms search the design space differently?
4. What constitutes convincing evidence that one method performs better than another?

The repository is best understood as an experimental framework for comparing two approaches to designing multilayer optical coatings:

- reinforcement learning, in which an agent learns a policy for constructing a coating one layer at a time;
- evolutionary optimisation, in which a population of complete coatings is improved through selection, crossover and mutation.

Both approaches use the same underlying coating physics to score candidate designs.

---

## 1. Begin with optimisation, not machine learning

An optimisation problem contains four ingredients:

- **Decision variables:** the choices the optimiser can make.
- **Objectives:** the quantities it should improve.
- **Constraints:** conditions that candidate designs must satisfy.
- **Evaluation function:** the calculation that maps a design to its objective values.

For CoatOpt, the decision variables describe a multilayer coating. Each layer has:

- a material choice, which is discrete;
- a thickness, which is continuous;
- a position in the stack, which matters because optical fields depend on layer order.

A candidate can therefore be represented schematically as

```text
[(material 1, thickness 1), ..., (material N, thickness N)].
```

This is a difficult search space because it is both **combinatorial** and **continuous**. Changing a material changes the meaning of its thickness, while changing an early layer can alter the optical field throughout the remainder of the stack.

The main physical objectives are optical transmission or reflectivity, optical absorption, and coating Brownian thermal noise. Under the newer optical formulation, transmission and absorption are evaluated and reflectivity follows from their energy balance. Some older configurations express the equivalent optical goal by maximising reflectivity.

### First mental model

Think of the physics model as an expensive judge:

```text
coating design -> optical/thermal calculation -> objective vector
```

The optimisation algorithms do not replace the physics. They decide which coatings should be presented to the judge next.

### Check your understanding

Explain why the following is not enough: “Choose the material with the lowest thermal noise for every layer.” Your answer should mention optical performance, layer order and conflicting objectives.

---

## 2. Multi-objective optimisation (MOO)

### 2.1 Why one best coating may not exist

In single-objective optimisation, candidates can be ranked using one number. In multi-objective optimisation, improving one quantity can make another worse. For example, a design with lower thermal noise may have greater absorption, while a more reflective design may require more layers or more mechanically lossy material.

Consequently, the result is normally a **set of trade-off solutions**, rather than one universally best solution.

### 2.2 Pareto dominance

Suppose all objectives have been written so that smaller is better. Design A dominates design B when:

1. A is no worse than B in every objective; and
2. A is strictly better in at least one objective.

If A has lower absorption but B has lower thermal noise, neither necessarily dominates the other. They represent different trade-offs.

The **Pareto front** is the set of non-dominated designs. Moving along it exchanges performance in one objective for performance in another.

### 2.3 A small example

Consider three designs evaluated using transmission and thermal noise:

| Design | Transmission | Thermal noise | Interpretation |
|---|---:|---:|---|
| A | 4 | 8 | strong optical performance |
| B | 6 | 5 | strong thermal performance |
| C | 8 | 9 | worse than A in both quantities |

A and B are non-dominated. A dominates C, so C should not appear on the Pareto front.

### 2.4 Preferences and scalarisation

An algorithm sometimes needs a scalar reward even though the scientific result is a vector. One common approach is a weighted sum,

```text
reward = w1 * optical_reward + w2 * absorption_reward + w3 * thermal_reward.
```

The weights express a preference, not a law of physics. Different weights can reveal different regions of the Pareto front. This is why a preference-conditioned policy can be useful: the same trained policy can respond to different requested trade-offs.

Weighted sums have limitations. In particular, they can miss some solutions on non-convex Pareto fronts. Constraints provide another formulation: optimise one objective while requiring the others to remain within specified bounds.

### 2.5 Normalisation

Transmission, absorption and thermal noise may differ by many orders of magnitude. Combining their raw numbers would cause the numerically largest scale to dominate. Normalisation maps them into comparable reward ranges.

Normalisation does not change the underlying physics, but poor bounds can distort learning. You should always be able to explain:

- which quantities are normalised;
- which bounds are used;
- whether values are clipped;
- how values are converted back into physical units for reporting.

### 2.6 Hypervolume

Hypervolume measures how much objective space is dominated by an approximation to the Pareto front, relative to a specified reference point. It rewards both:

- **convergence:** getting close to genuinely good trade-offs;
- **coverage:** finding a diverse spread of trade-offs.

Hypervolume is useful for comparing algorithms, but only when the objective directions, scaling and reference point are identical.

### Questions you should be ready to answer

- Why can two different coatings both be Pareto optimal?
- Why is a weighted sum not the same as the Pareto front?
- Why can changing the hypervolume reference point change the reported comparison?
- What is the difference between an objective and a constraint?

---

## 3. Reinforcement learning (RL)

### 3.1 The core idea

Reinforcement learning trains an agent to make a sequence of decisions. After acting in an environment, the agent receives information about how good the outcome was and adjusts its policy.

The standard components map onto CoatOpt as follows:

| RL concept | Meaning in CoatOpt |
|---|---|
| Environment | the coating-construction problem plus its physics evaluation |
| State/observation | the partially constructed stack and relevant preferences or constraints |
| Action | choose the next material and its thickness |
| Episode | construct one complete coating |
| Reward | a transformed measure of the final physical objectives |
| Policy | a learned rule for choosing the next layer |
| Return | the cumulative reward attributed to a sequence of choices |

### 3.2 Why formulate coating design sequentially?

A genetic algorithm proposes a complete coating and then scores it. RL instead asks: “Given the layers already selected, what layer should be added next?”

This gives the model the opportunity to learn reusable construction patterns. For example, it may learn that a particular material-thickness choice is useful only after a particular preceding sequence.

The hoped-for advantage is **generalisation**: a trained policy might generate useful designs for several preferences or constraint levels without restarting the search from scratch. This potential advantage must be demonstrated experimentally; it is not guaranteed merely because RL is used.

### 3.3 State and the Markov property

An RL state should contain enough information to choose the next action. If essential information about the existing stack or current objective preference is missing, the same observation may require different actions. Learning then becomes harder.

In CoatOpt, the observation can include the layer sequence, the current position, preference weights and constraint bounds. A sequence model such as an LSTM or transformer can learn relationships between earlier and later layers.

### 3.4 Rewards and credit assignment

The physical quality of a coating is most meaningful when the stack is complete. This creates a **credit-assignment problem**: which earlier layer choices deserve credit or blame for the final result?

Reward shaping can make learning easier, but it can also change the problem the agent solves. When describing the method, distinguish clearly between:

- the physical objective values;
- their transformed or normalised rewards;
- penalties for violating constraints;
- bonuses associated with Pareto improvement or exploration.

### 3.5 Exploration versus exploitation

The agent must balance:

- **exploration:** trying uncertain actions that might reveal better designs;
- **exploitation:** using actions already believed to be effective.

Too little exploration causes premature convergence. Too much exploration prevents the policy from refining good behaviour. Entropy bonuses, stochastic policies and randomised preferences or constraints all encourage exploration in different ways.

### 3.6 Actor and critic

Most modern RL methods in this repository use an actor-critic structure:

- the **actor** is the policy that chooses actions;
- the **critic** estimates how valuable a state or state-action pair is.

The critic provides a learning signal that helps the actor improve. An **advantage estimate** asks whether an action performed better or worse than the critic expected in that state.

---

## 4. PPO and the HPPO variants

### 4.1 PPO intuition

Proximal Policy Optimisation (PPO) is an **on-policy** algorithm. It generates trajectories with the current policy, estimates which actions were advantageous, and updates the policy using those recent trajectories.

The central problem is update size. A very large update based on imperfect data can destroy useful behaviour. PPO clips the policy-ratio term so that an apparently good action cannot move the new policy too far from the policy that collected the data.

You do not need to memorise the full loss equation for a presentation. The essential sentence is:

> PPO improves the probability of advantageous actions while restricting how abruptly the policy is allowed to change.

### 4.2 Important PPO concepts

- **On-policy:** experience becomes stale once the policy changes substantially.
- **Policy ratio:** compares the probability of an action under the new and old policies.
- **Clipping:** limits the incentive for excessively large policy updates.
- **Value loss:** trains the critic to predict returns.
- **Entropy bonus:** discourages the policy from becoming deterministic too early.
- **GAE:** balances bias and variance when estimating advantages.

### 4.3 What “hybrid” means here

Each coating-layer action contains both:

- a discrete material choice;
- a continuous thickness choice.

A hybrid policy therefore needs an appropriate probability distribution for each part. Conceptually, it can first assign probabilities to materials and then predict a bounded thickness distribution conditional on the selected material.

This is more faithful than forcing every thickness into a coarse set of discrete bins, but it also makes the policy and likelihood calculation more involved.

### 4.4 Preferences and constraints as policy inputs

If preference weights and constraint bounds are included in the observation, the policy is learning a family of related tasks:

```text
(partial coating, requested trade-off, constraints) -> next layer
```

This is central to the intended generalisation argument. The aim is not merely to produce one good coating, but to learn how the appropriate construction changes when the requested trade-off changes.

### 4.5 Behaviour cloning from the Pareto archive

Behaviour cloning trains the policy to imitate actions from previously successful trajectories. In this setting, the Pareto archive supplies demonstrations.

This can stabilise learning, but it creates two questions:

1. Which archived solutions are relevant to the current preference and constraints?
2. How strongly should imitation influence the policy relative to new RL experience?

Imitating the entire archive indiscriminately may mix contradictory behaviours. Selecting demonstrations relevant to the current target makes the learning signal more coherent.

---

## 5. Soft Actor-Critic (SAC)

SAC is an **off-policy** actor-critic algorithm. Unlike PPO, it stores transitions in a replay buffer and can reuse them for multiple updates. This can improve sample efficiency when physics evaluations are expensive.

SAC learns action-value functions, usually using two critics to reduce optimistic value estimates. Its objective also rewards policy entropy, so the agent seeks actions that are both high-value and sufficiently diverse.

The simplest useful comparison is:

| PPO | SAC |
|---|---|
| On-policy | Off-policy |
| Primarily uses newly collected trajectories | Reuses a replay buffer |
| Stabilises updates with clipping | Stabilises value learning with twin critics and target networks |
| Exploration encouraged by policy entropy | Entropy is part of the main optimisation objective |

For a hybrid coating action, SAC must evaluate a joint discrete-continuous choice. The policy can be factorised conceptually as

```text
P(material, thickness | state)
= P(material | state) * P(thickness | state, material).
```

The replay buffer is valuable only if old experience remains meaningful. Rapidly changing reward definitions, preferences or constraint distributions can make stored transitions less relevant.

### Questions you should be ready to answer

- What is the difference between on-policy and off-policy learning?
- Why might SAC require fewer fresh physics evaluations than PPO?
- Why does SAC use entropy?
- Why is a replay buffer potentially awkward in a changing multi-objective task?

---

## 6. Genetic and evolutionary algorithms

### 6.1 The core idea

An evolutionary algorithm maintains a population of candidate coatings. Each candidate is a complete design, sometimes called an individual or chromosome.

A typical generation contains:

1. evaluate every candidate using the physics model;
2. select promising parents;
3. create offspring through crossover;
4. modify offspring through mutation;
5. retain a strong and diverse population;
6. repeat.

Unlike RL, this process does not learn a policy for constructing coatings. It directly searches for good complete coatings.

### 6.2 Representation

In CoatOpt, an individual contains the material and thickness variables for all possible layers. The representation must also encode termination or unused layers, and repair operators enforce structural rules such as:

- a minimum number of layers before termination;
- no active layers after the first air/termination marker;
- no invalid material indices;
- restrictions on consecutive identical materials.

Representation and repair matter enormously. An optimiser that spends most of its evaluations on invalid coatings will appear inefficient even if its selection mechanism is sound.

### 6.3 Crossover and mutation

- **Crossover** combines information from two parents. It exploits existing patterns.
- **Mutation** makes random changes. It introduces novelty and prevents the population from becoming too homogeneous.

For continuous thicknesses, simulated binary crossover and polynomial mutation create bounded numerical variations. Material variables are decoded into discrete choices, so care is needed when applying continuous genetic operators to them.

### 6.4 NSGA-II

NSGA-II is a multi-objective evolutionary algorithm based on two main ideas:

- **non-dominated sorting**, which assigns candidates to successive Pareto fronts;
- **crowding distance**, which favours isolated candidates and therefore preserves diversity along a front.

Selection first prefers lower non-domination rank. Between candidates on the same front, it prefers greater crowding distance.

### 6.5 NSGA-III

NSGA-III extends the diversity mechanism using reference directions. It is particularly useful when there are several objectives and crowding distance becomes less informative. Candidate survival is encouraged across different directions in objective space.

### 6.6 MOEA/D

MOEA/D decomposes one multi-objective problem into many related scalar subproblems, each associated with a different trade-off direction. Neighbouring subproblems exchange information.

The conceptual contrast is:

- NSGA methods rank a population using dominance and diversity;
- MOEA/D divides the Pareto-front search into cooperating scalar searches.

---

## 7. RL versus evolutionary search

The most important distinction is what is learned.

| Question | Reinforcement learning | Evolutionary algorithm |
|---|---|---|
| Search unit | a sequence of layer decisions | a population of complete coatings |
| Persistent knowledge | policy and value networks | current population/archive |
| Reuse across preferences | possible through conditioning | usually requires continued search |
| Use of past evaluations | PPO: limited; SAC: replay buffer | candidates influence later generations |
| Natural output | a design-generating policy | a set of candidate designs |
| Main strength | potential generalisation and amortised search | direct, robust black-box optimisation |
| Main weakness | training instability and reward sensitivity | many evaluations and limited transfer |

Neither family is automatically superior. NSGA-II can be extremely competitive on a fixed problem. RL becomes scientifically interesting if the cost of training buys reusable capability across layer counts, materials, preferences or constraints.

### Fair comparison

A fair comparison should control or report:

- number of calls to the physics evaluator;
- wall-clock time and hardware;
- number of independent random seeds;
- identical design variables and constraints;
- identical objective calculations and units;
- equal or clearly stated stopping criteria;
- Pareto quality using the same metrics and reference points.

For this project, **physics-evaluation count** is often more informative than training iterations or generations. One PPO update and one NSGA-II generation can consume very different numbers of candidate evaluations.

### The likely presentation claim

A careful claim would be:

> Evolutionary algorithms provide a strong direct-search baseline for each fixed coating problem. The reinforcement-learning approach attempts to amortise this search by learning a conditional policy that can generate designs across multiple objective preferences and constraint levels.

That is stronger and more defensible than claiming that RL simply “finds better coatings.”

---

## 8. Conceptual flow of CoatOpt

```text
Experiment configuration
        |
        v
Materials, layer limits, objectives and constraints
        |
        +-----------------------------+
        |                             |
        v                             v
RL constructs one layer         Evolutionary method proposes
at a time using a policy         a population of full stacks
        |                             |
        +--------------+--------------+
                       |
                       v
             Shared physics evaluation
                       |
                       v
       transmission/reflectivity, absorption,
              thermal noise and thickness
                       |
        +--------------+--------------+
        |                             |
        v                             v
RL reward, critic and           non-dominated sorting,
policy update                   crossover and mutation
        |                             |
        +--------------+--------------+
                       |
                       v
              Pareto archive and metrics
```

This diagram is the backbone of a talk: two search strategies, one scientific evaluation, and a common Pareto-based comparison.

---

## 9. Four-week study plan

Assume approximately one focused hour per day, five days per week.

### Week 1: Optimisation and Pareto reasoning

**Goal:** explain the problem without mentioning neural networks.

- Day 1: identify decision variables, objectives and constraints.
- Day 2: practise dominance comparisons using points on paper.
- Day 3: draw two-objective Pareto fronts and identify dominated points.
- Day 4: learn normalisation, weighted sums and constraints.
- Day 5: understand hypervolume and its reference point.

**Deliverable:** a two-minute explanation of why no single best coating necessarily exists.

### Week 2: RL foundations and PPO

**Goal:** map the coating problem to an RL environment.

- Day 1: learn states, actions, policies, rewards and episodes.
- Day 2: study returns, value functions and advantages.
- Day 3: study actor-critic learning and exploration.
- Day 4: learn PPO clipping and on-policy data collection.
- Day 5: explain the hybrid material/thickness action and conditional preferences.

**Deliverable:** draw one complete episode from empty coating to final reward.

### Week 3: SAC and evolutionary algorithms

**Goal:** compare search mechanisms rather than list acronyms.

- Day 1: learn off-policy learning and replay buffers.
- Day 2: learn SAC entropy, critics and target networks.
- Day 3: learn population, selection, crossover and mutation.
- Day 4: learn NSGA-II sorting and crowding distance.
- Day 5: compare NSGA-II, NSGA-III and MOEA/D at a high level.

**Deliverable:** give a five-minute whiteboard comparison of PPO, SAC and NSGA-II.

### Week 4: Connect concepts to the experiment and practise the talk

**Goal:** defend the methodology and results.

- Day 1: trace a coating through the shared physics evaluation.
- Day 2: inspect one experiment configuration and translate every important setting into plain language.
- Day 3: inspect a Pareto-front output and explain three selected designs.
- Day 4: rehearse likely questions about fairness, generalisation and physical validity.
- Day 5: give the full talk without notes, then repair the weakest explanation.

**Deliverable:** a ten-slide rehearsal deck and a one-page question-and-answer sheet.

---

## 10. Active-learning exercises

### Exercise A: Pareto sorting by hand

Create ten fictional coatings with two objective values. Mark the first Pareto front, remove it, and identify the second front. Then add a third objective and observe why visual reasoning becomes harder.

### Exercise B: Design an RL formulation

Write down your proposed observation, action and reward. For every element in the observation, ask: “Could the correct next action change if this information were omitted?”

### Exercise C: Reward failure modes

Imagine absorption values are in ppm while thermal noise is around $10^{-21}$. Predict what happens if they are added without normalisation. Then consider bounds that are too narrow and clipping that is too aggressive.

### Exercise D: Be the NSGA-II algorithm

Take eight designs, sort them by dominance, estimate crowding distance, select four parents, and invent crossover and mutation operations for their material/thickness representation.

### Exercise E: Fair algorithm comparison

Suppose RL evaluates 50,000 coatings during training and then generates 1,000 designs cheaply, while NSGA-II evaluates 20,000 coatings for each requested preference. Decide which is more efficient after one, five and fifty preference queries.

### Exercise F: Explain the project at three levels

Prepare three versions:

- 30 seconds for a general scientific audience;
- 2 minutes for an optimisation researcher;
- 5 minutes for someone who wants methodological details.

---

## 11. Presentation questions to rehearse

1. Why use RL when NSGA-II already works well?
2. What exactly is the RL state and action?
3. Is the reward physically meaningful or merely a training device?
4. How are conflicting objectives handled?
5. What makes a coating Pareto optimal?
6. How do you ensure the comparison uses the same evaluation budget?
7. What does the policy generalise across?
8. Why use PPO rather than SAC, or vice versa?
9. What prevents invalid layer sequences?
10. How sensitive are the conclusions to objective scaling and constraint bounds?
11. Why is hypervolume an appropriate metric?
12. Are optical and thermal calculations identical for every optimiser?

If you can answer these clearly, you understand the conceptual core of the repository.

---

## 12. Compact glossary

- **Action:** a decision made by an RL agent.
- **Actor:** the network that represents the policy.
- **Advantage:** how much better an action was than expected.
- **Constraint:** a condition that restricts acceptable solutions.
- **Critic:** a network estimating state or action value.
- **Crowding distance:** an NSGA-II estimate of how isolated a solution is in objective space.
- **Dominance:** being no worse in every objective and better in at least one.
- **Entropy:** a measure of policy randomness.
- **Episode:** one complete sequential interaction, here the construction of a coating.
- **Hypervolume:** the dominated region between a Pareto set and a reference point.
- **Objective:** a physical quantity to maximise or minimise.
- **Off-policy:** learning from data generated by older or different policies.
- **On-policy:** learning primarily from data generated by the current policy.
- **Pareto front:** the set of non-dominated trade-off solutions.
- **Policy:** a rule or distribution for selecting actions from states.
- **Replay buffer:** stored transitions reused by an off-policy algorithm.
- **Reward:** the numerical learning signal supplied to an RL agent.
- **Scalarisation:** converting several objectives into one scalar quantity.
- **State:** the information available when choosing an action.

---

## 13. Recommended learning resources

Use these selectively; the goal is to understand the relevant chapters, not complete every resource.

### Reinforcement learning foundations

- [Sutton and Barto, *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html): focus first on Chapters 1, 3, 6 and 13.
- [OpenAI Spinning Up: key concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html): a shorter bridge from terminology to modern deep RL.
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html): focus on the intuition, clipping and pseudocode.
- [OpenAI Spinning Up: SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html): focus on off-policy learning, entropy and twin Q-functions.

### Multi-objective and evolutionary optimisation

- [pymoo: multi-objective optimisation introduction](https://pymoo.org/getting_started/part_2.html): practical introduction to obtaining a solution set.
- [pymoo: NSGA-II](https://pymoo.org/algorithms/moo/nsga2.html): non-dominated sorting and crowding distance, matching the library used by CoatOpt.
- [pymoo documentation](https://pymoo.org/): use the algorithm pages for NSGA-III and MOEA/D after NSGA-II is clear.
- Deb et al., “A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II”: read the abstract, algorithm overview and figures before attempting every equation.

### Best reading order

1. Pareto dominance and NSGA-II.
2. RL state/action/reward terminology.
3. Actor-critic and advantage estimation.
4. PPO.
5. SAC.
6. Preference-conditioned and constrained multi-objective RL.

Do not begin with PPO’s loss equation. First understand the problem representation and the reason a policy is useful.
