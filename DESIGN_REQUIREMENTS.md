# Design Requirements

## 1. Project Overview

Train a reinforcement learning agent to play the role of the fugitive ("Jack") in an asymmetric board game based on *Letters from Whitechapel*. Cops are controlled by heuristic agents. A "Director" mechanism dynamically adjusts cop knowledge to balance difficulty during training (curriculum learning). The system is evaluated on boards of varying sizes extracted as subsets of the original Whitechapel map.

A web interface allows human participants to play as Jack against heuristic cops, enabling direct performance comparison between the trained RL agent and human players of varying skill levels.

---

## 2. Game Rules & Mechanics

### 2.1 Players and Roles

| Role | Count | Information | Controlled by |
|------|-------|-------------|---------------|
| Jack (fugitive) | 1 | Full — sees entire board state | RL agent |
| Cops (pursuers) | Defined per map configuration | Partial — see only discovered information | Heuristic agents |

### 2.2 Graph Structure

The Jack and Cop graphs are **meshed together**, not merely adjacent. The board alternates between Jack nodes (circles) and Cop nodes (squares):

```
Jack_A ──[via Cop_X … Cop_Y]── Jack_B
```

Every edge in the Jack graph has an **ordered sequence of one or more associated Cop nodes** that Jack passes through to traverse it. Cops sit on these connecting squares. This means:

- Jack is always *at* a Jack node (circle) between moves.
- To move from Jack node A to Jack node B, Jack traverses a specific sequence of Cop nodes stored in the edge's `via` field.
- A cop occupying any Cop node in an edge's `via` sequence blocks that edge (when blocking is enabled).

This structure also explains the search and arrest mechanics: cops on Cop nodes are positioned on the connectors between Jack positions, giving them natural visibility into adjacent Jack nodes.

**Jack node types:**

- `jack` — a standard Jack position.
- `jack_start` — a crime scene; the only valid starting positions for Jack. One is chosen randomly at the start of each game from the map's `jack_starts` list.

### 2.3 Turn Structure

The game consists of a single night with a fixed number of rounds (defined in the map configuration):

- At the **start of the night**, Jack's starting position is revealed to all cops.
- Each **round**: Jack moves first, then each cop takes their turn.
- Cop knowledge of Jack's visited nodes is scoped to the current game (single-night trace history).
- The **turn limit** is an attribute of the map configuration.

The night ends when any of the following conditions are met:

| Condition | Winner |
|-----------|--------|
| Jack moves onto the hideout node | Jack |
| Turn limit expires | Cops |
| Jack is successfully arrested | Cops |
| Jack has no legal moves (all exit paths blocked by cops) | Cops — **togglable** |

> **Blocking rule (togglable):** Jack cannot traverse an edge if the Cop node on that edge is currently occupied by a cop. If all edges from Jack's current node are blocked, Jack is surrounded and the cops win. This is disabled by default due to increased search complexity. Must be implementable as a config flag with no changes to the core game loop.

### 2.4 Movement Rules

- **Jack:** Moves from his current Jack node to an adjacent Jack node by traversing the connecting Cop node. One move per round. If blocking is enabled, edges through occupied Cop nodes are illegal.
- **Cops:** Move 0, 1, or 2 steps along the Cops graph per round, choosing any reachable destination within that range.
- Both graphs loaded from a map config bundle (`maps/whitechapel.json` for the full board).
- **No special moves (carriages, alleys) initially.** The action space must be structured to accommodate these later without rearchitecting the core environment.

### 2.5 Cop Actions

After moving, each cop **must choose exactly one** of:

- **Search:** Queries all Jack nodes adjacent to the cop's current Cop node. Returns to the shared knowledge state which of those Jack nodes Jack has visited during the current game.
- **Arrest:** Targets a single Jack node adjacent to the cop's current Cop node.
  - If Jack is currently on that node → cops win immediately.
  - If not → all cops learn only that Jack is not on that specific node at this moment.

All search results and arrest outcomes are added to the **global shared knowledge state** (hivemind) — visible to all cops immediately.

### 2.6 Information Model

**Cops know (shared `CopKnowledge`):**
- Jack's starting position (revealed at the start of the night).
- All cop positions.
- `visited` — Jack nodes confirmed to be in his trace this game (from searches). Director-manipulable.
- `search_misses` — `(node, turn)` pairs: Jack's trace did not include this node at or before this turn. Does **not** permanently exclude the node — Jack can visit it in a later turn.
- `arrest_misses` — `(node, turn)` pairs: Jack was confirmed absent from this node at this exact moment.

**Jack knows:**
- His current position.
- All cop positions.
- The full board topology.
- His assigned hideout node.
- Which nodes cops searched and the **true results** of those searches (whether Jack genuinely visited them).

Jack does **not** see Director manipulations. If the Director suppressed a real find or injected a false one into the cops' knowledge, Jack cannot detect this — he only observes true game events. The Director's effect is visible to Jack only indirectly, through changes in cop behaviour that result from manipulated knowledge. From Jack's perspective this is indistinguishable from the cops acting on instinct or intuition.

### 2.7 Hideout Selection

At the start of each game, Jack is assigned a single hideout node. The assignment is semi-random, subject to constraints:

- Jack's **starting position is chosen randomly** from the map's `jack_starts` list at the beginning of each game.
- The hideout must be at least a minimum graph distance from the start (threshold defined per map).
- Candidates should be positioned such that natural cop patrol routes lie between the start and the hideout — i.e., reaching it requires evasion rather than a straight run.

Hideout selection uses a **distance threshold**: random selection from all Jack nodes at least N hops from the start position in the Jack graph. The threshold N is defined per map configuration (likely scaled to map diameter). No coverage scoring — distance alone is sufficient.

### 2.8 Win / Loss Conditions

- **Jack wins:** Moves onto the hideout node.
- **Cops win:** Successful arrest, turn limit expires, or Jack surrounded (if blocking enabled).

---

## 3. Director (Difficulty Balancer)

Inspired by the director systems in *Left 4 Dead* and *Alien: Isolation*. The Director modifies the cops' effective knowledge to keep training difficulty productive for the Jack agent's current learning stage.

### 3.1 Mechanism

The Director manipulates the **cops' shared knowledge state** — not the actual game state:

- **Easier for Jack:** Suppress legitimate search results — remove discovered nodes from the cops' known set even when correctly found.
- **Harder for Jack:** Inject additional reveals — add Jack nodes to the cops' known set without a cop having searched them.

The Director's interventions are **not observable** by either agent. Jack does not see the cop knowledge state at all. Cops act on the (potentially modified) knowledge state without knowing it has been altered. From both perspectives, the Director's effect is indistinguishable from natural variation in cop behaviour — which is the intent: the Director emulates human intuition, the way a human cop might "just sense" where a fugitive went, or overlook something they should have noticed.

### 3.2 Curriculum Learning Integration

The Director adjusts its intervention level based on Jack's rolling performance:
- Jack winning too consistently → increase cop information (harder).
- Jack losing too consistently → decrease cop information (easier).

**TBD:**
- Driving metric: rolling win rate over N games, average survival time, or a combined signal.
- Adjustment thresholds, step sizes, and hard caps on intervention magnitude.

---

## 4. Board Representation

### 4.1 Full Board

- **Jack graph:** 195 nodes. Each `JackEdge` carries an ordered `via` sequence of one or more Cop nodes traversed — this is what enables the blocking rule and defines search/arrest reach.
- **Cops graph:** 234 nodes.
- Both graphs plus the traversal structure are stored in `maps/whitechapel.json` (or another map config bundle).

### 4.2 Map Configuration Object

Each playable map is a configuration bundle (`maps/*.json`) containing:
- Jack nodes (id, coordinates, edges with `via` sequences)
- Cop nodes (id, coordinates, movement edges, jack neighbours)
- `jack_starts` — valid Jack starting node IDs
- `cop_starts` — pool of cop spawn node IDs
- Number of cops
- Turn limit
- Minimum hideout distance (`hideout_min_distance`)

### 4.3 Board Size Variants

Smaller boards are **connected subgraphs of the full Whitechapel board**, preserving the traversal structure between Jack and Cop nodes within the subgraph.

**TBD:**
- Subgraph selection method (hand-crafted contiguous regions preferred).
- Number of size tiers.

---

## 5. State and Observation Spaces

### 5.1 Jack's Observation

- Current position (Jack node ID)
- All cop positions (Cop node IDs)
- Assigned hideout node
- Which nodes cops searched each round and the true results (was Jack there or not)
- If blocking enabled: set of currently blocked Jack edges (derivable from cop positions and the traversal structure, so not hidden information)

### 5.2 Cops' Observation (Shared `CopKnowledge`)

- All cop positions
- Jack's starting position
- `visited` — Jack nodes confirmed in his trace (Director-manipulable)
- `search_misses` — `(node, turn)` pairs: Jack's trace did not include this node at or before that turn
- `arrest_misses` — `(node, turn)` pairs: Jack was confirmed absent from this node at that exact moment

### 5.3 Action Spaces

**Jack:** Move to an adjacent Jack node by choosing a traversable edge. If blocking is enabled, edges through occupied Cop nodes are excluded from the legal action set.

**Cops (per cop, per round):**
1. **Move:** Choose any destination reachable within 0–2 steps along the Cops graph.
2. **Action (choose one):** Search all adjacent Jack nodes XOR Arrest one specific adjacent Jack node.

---

## 6. Agent Design

### 6.1 Jack — RL Agent

- **RL library: CleanRL** — single-file PPO implementation that is fully owned and modifiable. Chosen over SB3 because the Director requires custom training loop intervention, curriculum learning requires mid-training environment changes, and the graph-based state space will need a custom policy architecture. Transparency matters for dissertation writeup.
- **Algorithm: PPO** (Proximal Policy Optimization) — standard for discrete-action game environments; on-policy nature handles changing Director difficulty without stale-experience issues.
- Reward signal:
  - **Baseline (default):** Sparse — large positive on win, large negative on loss. Plus a small per-step penalty (e.g. −0.01/step) to discourage stalling and make each round matter without heavily distorting the value function.
  - **Shaped variant (for ablation):** Add a distance-to-hideout progress bonus. Useful for the cold-start problem on large boards where a random policy rarely finds the hideout. This is kept as an experimental variable — the Director already handles difficulty, so heavy shaping may not be necessary.

### 6.2 Cops — Heuristic Agents

Cops maintain a shared **probability mass function (PMF)** over Jack's current position using a time-expanded path-count model.

**PMF computation (layer-by-layer):**

```
count[0][start] = 1
count[0][v]     = 0  for all v ≠ start

count[t+1][v]   = sum of count[t][u] for all u adjacent to v

count[t][v]     = 0  if (v, T) in search_misses  and  t <= T
                  0  if (v, T) in arrest_misses   and  t == T

P(Jack at v | turn t) ∝ count[t][v]
```

A search miss `(v, T)` means Jack's path had not visited `v` by turn T, so `count[t][v] = 0` for all `t ≤ T`. For turns `t > T` the node is reachable again — Jack may visit it later. An arrest miss `(v, T)` only zeroes out turn T specifically. When new constraints are added, counts are recomputed from scratch: O(T × E), negligible for these graph sizes. Positive finds (`visited`) are treated as confirmation only — no backward inference in the initial implementation.

**Cop coordination — PMF-guided ACO:**

Each turn, before cops move, an Ant Colony Optimisation procedure assigns movement targets:

1. The PMF serves as the pheromone map — nodes with high probability mass are high-attraction targets.
2. Run N iterations (fixed cap, tunable — suggested 20–50) simulating candidate full assignments of all cops to target nodes:
   - Cops are attracted to high-PMF nodes.
   - Within each iteration, once a cop claims coverage of a region its probability mass is discounted for other cops — this repulsion mechanic naturally spreads cops across the distribution and prevents clustering.
3. Execute the highest-scoring assignment found across all iterations.

**Action decision (search vs. arrest):**

- **Search:** target the cop node whose adjacent Jack nodes have the highest total remaining probability mass — maximises expected information gain.
- **Arrest:** attempt when `P(Jack at v)` exceeds a threshold θ (tunable — good dissertation variable). Low θ = aggressive but wasteful arrests; high θ = conservative cops that rarely commit.

---

## 7. Evaluation

### 7.1 Metrics

- Win rate (Jack wins / total games)
- Average survival time (rounds per game)
- Average rounds to capture (cop-win games only)

### 7.2 Baselines

- Trained Jack vs. heuristic cops, Director inactive
- Trained Jack vs. random cops
- Human players as Jack vs. heuristic cops

### 7.3 Ablation Studies

- Director active vs. Director inactive during training
- Sparse rewards vs. distance-shaped rewards
- Multi-size board curriculum vs. single board size training

---

## 8. System Architecture

The UI requirement means the game engine cannot be a standalone script. All game logic (environment, heuristic cops, and eventually the trained RL agent) must run server-side, exposing an API the browser communicates with.

### 8.1 Backend

- **Language/framework:** Python + FastAPI (async support, automatic OpenAPI docs, natural fit with the existing Python codebase)
- Hosts the game engine, heuristic cop logic, and the trained RL agent (for AI games)
- Manages game sessions and persists game records to a database
- Runs cop turns automatically after Jack submits a move, returning the full sequence of cop actions to the frontend for animation

### 8.2 Frontend

- Browser-based, shareable via URL — no install required for participants
- Renders the board using the existing SVG map (`Mapa_v5.svg`) as a base layer, with node positions and connections overlaid
- After Jack submits a move, plays back cop actions (movement, search, arrest) as a short animation before enabling Jack's next turn
- **Framework: plain JavaScript + SVG manipulation** — sufficient for the board rendering and interaction complexity, no build toolchain overhead

### 8.3 Database

- **SQLite** — sufficient for the expected load (a dissertation study with tens of participants over weeks will never produce simultaneous writes that cause contention)
- Stores full game records: participant skill group, move sequence (Jack and cops, every round), outcome, survival time
- Must support full state reconstruction at any step (for replay mode)
- Single file — trivial to back up; no separate database server required

### 8.4 Deployment

- **Platform:** Cloud (Railway, Render, or Fly.io — all support containerised Python apps)
- Deployed as a **Docker container** with a persistent filesystem volume (for the SQLite file and trained RL model weights)
- Requires a persistent server process — serverless/function-based hosting (Lambda, Vercel) is incompatible because game sessions maintain in-memory state between requests
- The site must be publicly accessible via a shareable URL; no authentication required for participants

---

## 9. User Interface

### 9.1 Human Play Mode

**Entry flow:**
1. Participant opens the shared URL.
2. Fills out a short form declaring their experience level: **Beginner**, **Experienced**, or **Advanced**. This groups results for analysis — no account or login required.
3. Optional tutorial explaining the rules and controls.
4. Game starts.

**In-game view (Jack's perspective — full information):**
- Board rendered from the SVG map with all nodes visible.
- Jack's current position highlighted.
- All cop positions visible.
- Assigned hideout node marked.
- Cops' shared knowledge state visible: confirmed-visited Jack nodes and confirmed-empty Jack nodes (consistent with what Jack knows in the game rules).
- Legal moves for Jack highlighted/clickable.
- Turn counter showing rounds remaining.
- After Jack moves: cop turns resolve automatically, with visual feedback showing cop movement and search/arrest outcomes before the next Jack turn begins.

**End of game:**
- Result displayed (win / caught / timeout / surrounded).
- Basic stats: rounds survived, outcome.

**Data recorded per game:**
- Participant skill group
- Full move sequence (both Jack and cops, every round)
- Game outcome
- Survival time (rounds)

### 9.2 Replay Mode (researcher only)

Available before the human study — intended for monitoring and analysing AI agent performance during and after training.

**Features:**
- List of recorded games, filterable by: player type (human / AI), skill group (for human games), outcome, map.
- Step through any game move-by-move: forward and backward.
- Full board state visible at each step, including cop knowledge state.
- Optionally: display the cops' internal belief/reachable set at each step (for debugging heuristic cop behaviour).

**Access:** Protected — not linked from the participant-facing page. Simple access control sufficient (e.g., a secret URL path or basic auth).

### 9.3 Tutorial (in-game, optional)

A brief interactive or illustrated walkthrough covering:
- The goal (reach the hideout before time runs out).
- How Jack moves (click an adjacent node).
- What the cops do each turn and what the highlighted discovered nodes mean.
- What arrest attempts look like.

---

## 10. Extensibility Requirements

Must be addable without redesigning the core environment:

- **Special Jack moves:** Carriages and alleys as additional edge types in the Jack graph, gated by a feature flag or via subclassing the action space.
- **Blocking rule:** Togglable via a map/environment config flag — no changes to the core game loop.
- **Trained cop agents:** Heuristic cop interface must be drop-in replaceable by a learned policy.
- **Arbitrary board topologies:** Environment accepts any valid (Jack graph, Cops graph, traversal structure, map config) bundle — not hardcoded to Whitechapel.
