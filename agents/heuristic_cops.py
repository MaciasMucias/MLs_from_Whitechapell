from __future__ import annotations
import math
import random

from engine.env import CopTurn
from engine.graph import Map
from engine.graph_utils import jack_bfs_distances, jack_reachable_within, reachable_cop_nodes
from engine.state import GameState
from agents.base import CopAgent


class HeuristicCops(CopAgent):
    """
    Heuristic cop agent that maintains two belief distributions and uses an
    ACO-style multi-iteration assignment to decide where each cop moves.

    Position PMF:
        counts[t][v][mask] = number of valid paths of length t ending at
        Jack node v that have collectively visited the waypoints encoded in
        mask (where waypoints = cop_knowledge.visited).

        Terminal distribution = counts[current_depth][*][full_mask], giving
        probability mass over Jack's position after the current move, given
        all confirmed visits. Search/arrest miss constraints zero out
        impossible positions.

    Hideout PMF:
        Distribution over possible hideout locations, updated each turn.
        Candidates are nodes at least hideout_min_distance hops from jack_start.
        Each candidate is weighted by the total position PMF mass that can
        still reach it within the remaining turn budget. This uses the position
        PMF (which already embeds miss constraints) so search/arrest misses
        propagate into hideout inference automatically.

    Movement (ACO):
        n_iterations random orderings and role assignments are tried. In each
        iteration every cop is independently labelled pursuer (prob
        pursuit_fraction) or searcher. Pursuers score cop nodes by coverage +
        pursuit_weight * direction_toward_hideout_centroid; searchers by
        coverage only. The assignment with the highest total score is used.

    Search vs arrest: 10% random arrest attempt on the highest-probability
    adjacent Jack node; 90% search.
    # TODO: replace random arrest with threshold-based logic once win-rate
    #       baselines exist (requires a competent Jack agent to be meaningful).
    """

    def __init__(
        self,
        arrest_prob: float = 0.1,
        pursuit_fraction: float = 0.4,
        pursuit_weight: float = 0.5,
        n_iterations: int = 30,
        rng: random.Random | None = None,
    ) -> None:
        self._arrest_prob = arrest_prob
        self._pursuit_fraction = pursuit_fraction
        self._pursuit_weight = pursuit_weight
        self._n_iterations = n_iterations
        self._rng = rng or random.Random()
        self._hideout_candidates: set[int] = set()

    # ------------------------------------------------------------------
    # CopAgent interface
    # ------------------------------------------------------------------

    def on_episode_start(self, state: GameState, game_map: Map) -> None:
        # Cache hideout candidates for this episode (fixed for the whole game).
        distances = jack_bfs_distances(state.cop_knowledge.jack_start, game_map)
        self._hideout_candidates = {
            v for v, d in distances.items()
            if d >= game_map.hideout_min_distance
        }

        # MULTI-NIGHT EXTENSION POINT
        #
        # In a full 4-night game, the hideout is fixed across all nights. Each
        # night starts from a new, revealed position — so the within-night PMF
        # resets. However, cross-night evidence can be exploited in two ways:
        #
        # 1. Hideout inference: at the end of each night, Jack's hideout is
        #    confirmed to lie somewhere reachable from that night's start within
        #    its turn limit that was NOT reached/eliminated this night. Accumulate
        #    a separate hideout_pmf: dict[int, float] across nights by multiplying
        #    in the surviving probability mass after each night ends (i.e. the
        #    posterior over hideout candidates). Persist this between episodes
        #    instead of resetting it here. Cops can then bias their movement
        #    toward high-hideout-probability regions even when the within-night
        #    position PMF is still broad.
        #
        # 2. Path tendency inference: Jack's path choices across nights may reveal
        #    a preferred routing style (e.g. always moves away from cops, tends to
        #    use a particular corridor). This would require tracking per-edge or
        #    per-region visit frequencies across nights and using them to weight
        #    the within-night DP transition probabilities — effectively a prior
        #    over Jack's movement model that sharpens with each night.
        #
        # To implement: add on_night_end(final_state, game_map) to the CopAgent
        # ABC (called by the multi-night game loop), persist hideout_pmf and any
        # learned priors on self, and pass hideout_pmf as a bias into
        # _assign_destinations so cops patrol hideout-likely regions even when
        # the position PMF is flat.

    def act(self, state: GameState, game_map: Map) -> list[CopTurn]:
        position_pmf = self._compute_pmf(state, game_map)
        hideout_pmf  = self._compute_hideout_pmf(position_pmf, state, game_map)
        destinations = self._assign_destinations(
            position_pmf, hideout_pmf, state.cop_positions, game_map
        )
        return [
            self._decide_action(cop_idx, game_map.cop_nodes[dest - 1], position_pmf)
            for cop_idx, dest in enumerate(destinations)
        ]

    # ------------------------------------------------------------------
    # Position PMF
    # ------------------------------------------------------------------

    def _compute_pmf(self, state: GameState, game_map: Map) -> dict[int, float]:
        """
        Bitmask forward DP over (turn, jack_node, visited_waypoints_mask).

        Returns a normalised probability dict {jack_node_id: probability}.
        """
        ck = state.cop_knowledge
        current_depth = state.turn + 1  # Jack just moved; state.turn not yet incremented

        # ------ waypoint index ------
        waypoints = sorted(ck.visited)  # deterministic ordering
        wp_idx: dict[int, int] = {v: i for i, v in enumerate(waypoints)}
        num_masks = 1 << len(waypoints)
        full_mask = num_masks - 1

        # ------ constraint sets ------
        # search_miss (v, T): counts[t][v][*] = 0 for t <= T
        search_exclude: dict[int, int] = {}  # node_id -> max excluded turn (inclusive)
        for (v, T) in ck.search_misses:
            if v not in search_exclude or search_exclude[v] < T:
                search_exclude[v] = T

        # arrest_miss (v, T): counts[T][v][*] = 0
        arrest_exclude: set[tuple[int, int]] = set(ck.arrest_misses)

        # ------ DP tables ------
        jack_start = ck.jack_start
        n = len(game_map.jack_nodes)

        prev: list[list[float]] = [[0.0] * num_masks for _ in range(n + 1)]
        start_mask = (1 << wp_idx[jack_start]) if jack_start in wp_idx else 0
        prev[jack_start][start_mask] = 1.0

        for t in range(1, current_depth + 1):
            curr: list[list[float]] = [[0.0] * num_masks for _ in range(n + 1)]
            for u_id in range(1, n + 1):
                for mask in range(num_masks):
                    mass = prev[u_id][mask]
                    if mass == 0.0:
                        continue
                    for edge in game_map.jack_nodes[u_id - 1].edges:
                        v_id = edge.destination.id
                        if search_exclude.get(v_id, -1) >= t:
                            continue
                        if (v_id, t) in arrest_exclude:
                            continue
                        new_mask = mask | (1 << wp_idx[v_id]) if v_id in wp_idx else mask
                        curr[v_id][new_mask] += mass
            prev = curr

        # ------ extract terminal distribution ------
        raw: dict[int, float] = {}
        for v_id in range(1, n + 1):
            mass = prev[v_id][full_mask]
            if mass > 0.0:
                raw[v_id] = mass

        if not raw:
            for v_id in range(1, n + 1):
                total = sum(prev[v_id])
                if total > 0.0:
                    raw[v_id] = total

        if not raw:
            return {node.id: 1.0 / len(game_map.jack_nodes) for node in game_map.jack_nodes}

        total = sum(raw.values())
        return {v: mass / total for v, mass in raw.items()}

    # ------------------------------------------------------------------
    # Hideout PMF
    # ------------------------------------------------------------------

    def _compute_hideout_pmf(
        self,
        position_pmf: dict[int, float],
        state: GameState,
        game_map: Map,
    ) -> dict[int, float]:
        """
        Distribution over possible hideout locations.

        Each candidate hideout is weighted by the total position PMF mass that
        can still reach it within the remaining turn budget. Candidates too far
        away to be reached score zero and are dropped. Because the position PMF
        already incorporates search/arrest miss constraints, those constraints
        propagate into hideout inference automatically.

        remaining_hops = turn_limit - 1 - state.turn
            Jack just moved this turn; he has this many more moves left.
        """
        remaining_hops = game_map.turn_limit - 1 - state.turn

        candidates = self._hideout_candidates
        if not candidates:
            # Fallback: recompute if on_episode_start was not called
            distances = jack_bfs_distances(state.cop_knowledge.jack_start, game_map)
            candidates = {v for v, d in distances.items() if d >= game_map.hideout_min_distance}

        # For each position PMF node, compute which candidates are reachable
        # within remaining_hops.  Build candidate -> accumulated weight.
        scores: dict[int, float] = {}
        for v, prob in position_pmf.items():
            reachable = jack_reachable_within(v, remaining_hops, game_map)
            for h in candidates:
                if h in reachable:
                    scores[h] = scores.get(h, 0.0) + prob

        if not scores:
            # All candidates unreachable — constraints have become inconsistent.
            # Fall back to uniform over original candidates.
            return {h: 1.0 / len(candidates) for h in candidates}

        total = sum(scores.values())
        return {h: s / total for h, s in scores.items()}

    # ------------------------------------------------------------------
    # Heading estimation
    # ------------------------------------------------------------------

    def _estimate_heading(
        self,
        position_pmf: dict[int, float],
        hideout_pmf: dict[int, float],
        game_map: Map,
    ) -> tuple[float, float, float, float]:
        """
        Returns (cx, cy, dx, dy):
          (cx, cy) — probability-weighted centroid of the position PMF.
          (dx, dy) — unit vector from position centroid toward hideout centroid.
        Returns (cx, cy, 0, 0) if both centroids coincide.
        """
        cx = sum(p * game_map.jack_nodes[v - 1].x for v, p in position_pmf.items())
        cy = sum(p * game_map.jack_nodes[v - 1].y for v, p in position_pmf.items())
        hx = sum(p * game_map.jack_nodes[h - 1].x for h, p in hideout_pmf.items())
        hy = sum(p * game_map.jack_nodes[h - 1].y for h, p in hideout_pmf.items())

        dx, dy = hx - cx, hy - cy
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return cx, cy, 0.0, 0.0
        return cx, cy, dx / mag, dy / mag

    # ------------------------------------------------------------------
    # ACO-based destination assignment
    # ------------------------------------------------------------------

    def _assign_destinations(
        self,
        position_pmf: dict[int, float],
        hideout_pmf: dict[int, float],
        cop_positions: tuple[int, ...],
        game_map: Map,
    ) -> list[int]:
        """
        Runs n_iterations random orderings and per-cop role draws (pursuer vs
        searcher). Returns the assignment with the highest total score.

        Pursuer score for a cop node:
            coverage + pursuit_weight * direction_dot_normalised
        Searcher score:
            coverage only

        direction_dot_normalised maps the raw dot product into [0, 1] across
        all reachable nodes of all cops so pursuit_weight is comparable to
        the probability-valued coverage scores.
        """
        cx, cy, dx, dy = self._estimate_heading(position_pmf, hideout_pmf, game_map)
        has_direction = (dx != 0.0 or dy != 0.0)

        # Pre-compute each cop's reachable set (same every iteration).
        reachable_sets = [
            reachable_cop_nodes(pos, game_map, max_steps=2)
            for pos in cop_positions
        ]

        # Pre-compute direction dots for all reachable cop nodes across all cops.
        # Normalise globally so the scale matches probability values.
        all_dots: dict[int, float] = {}
        if has_direction:
            for rs in reachable_sets:
                for cid in rs:
                    if cid not in all_dots:
                        cn = game_map.cop_nodes[cid - 1]
                        all_dots[cid] = (cn.x - cx) * dx + (cn.y - cy) * dy

            dot_min = min(all_dots.values())
            dot_range = max(all_dots.values()) - dot_min
            if dot_range < 1e-9:
                has_direction = False
            else:
                norm_dot: dict[int, float] = {
                    cid: (d - dot_min) / dot_range
                    for cid, d in all_dots.items()
                }

        best_score: float = -1.0
        best_assignment: list[int] = []

        for _ in range(self._n_iterations):
            # Random cop ordering and role draw for this iteration
            order = list(range(len(cop_positions)))
            self._rng.shuffle(order)
            is_pursuer = [self._rng.random() < self._pursuit_fraction for _ in cop_positions]

            remaining: dict[int, float] = dict(position_pmf)
            assignment: list[int] = [0] * len(cop_positions)
            iteration_score: float = 0.0
            occupied: set[int] = set()  # destinations claimed this iteration

            for cop_idx in order:
                # Exclude already-occupied nodes so two cops never share a node
                reachable = reachable_sets[cop_idx] - occupied
                if not reachable:
                    reachable = reachable_sets[cop_idx]  # fallback if all taken

                def node_score(cid: int, pursuer: bool = is_pursuer[cop_idx]) -> float:
                    coverage = sum(
                        remaining.get(jn.id, 0.0)
                        for jn in game_map.cop_nodes[cid - 1].jack_neighbours
                    )
                    if pursuer and has_direction:
                        return coverage + self._pursuit_weight * norm_dot.get(cid, 0.0)
                    return coverage

                best_node = max(reachable, key=node_score)
                assignment[cop_idx] = best_node
                occupied.add(best_node)

                # Use plain coverage (no pursuit bonus) to compare iterations.
                # The pursuit bias guides within-iteration selection; if it were
                # included in the comparison score, iterations that happen to draw
                # more pursuers would appear better regardless of actual coverage.
                plain_coverage = sum(
                    remaining.get(jn.id, 0.0)
                    for jn in game_map.cop_nodes[best_node - 1].jack_neighbours
                )
                iteration_score += plain_coverage

                for jn in game_map.cop_nodes[best_node - 1].jack_neighbours:
                    if jn.id in remaining:
                        remaining[jn.id] *= 0.5

            if iteration_score > best_score:
                best_score = iteration_score
                best_assignment = assignment[:]

        return best_assignment

    # ------------------------------------------------------------------
    # Action decision
    # ------------------------------------------------------------------

    def _decide_action(
        self,
        cop_idx: int,
        cop_node,  # CopNode
        pmf: dict[int, float],
    ) -> CopTurn:
        adj = cop_node.jack_neighbours
        if not adj:
            return CopTurn(cop_idx=cop_idx, destination=cop_node.id, search=True)

        # 10% random arrest on highest-probability adjacent node
        # TODO: replace with threshold-based logic once win-rate baselines exist
        if self._rng.random() < self._arrest_prob:
            best = max(adj, key=lambda jn: pmf.get(jn.id, 0.0))
            if pmf.get(best.id, 0.0) > 0.0:
                return CopTurn(
                    cop_idx=cop_idx,
                    destination=cop_node.id,
                    search=False,
                    arrest_target=best.id,
                )

        return CopTurn(cop_idx=cop_idx, destination=cop_node.id, search=True)
