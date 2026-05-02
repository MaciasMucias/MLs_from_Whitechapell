from __future__ import annotations
import math
import random

from engine.env import CopTurn
from engine.graph import Map
from engine.graph_utils import jack_bfs_distances, jack_reachable_within, reachable_cop_nodes
from engine.state import GameState
from agents.base import CopAgent, CopDecisionInfo, RoundCopDecisions


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

    Search vs arrest:
        A cop arrests all adjacent Jack nodes (arrest_all) under two conditions:
        (1) zone_mass >= effective_threshold, where zone_mass is the sum of PMF
            across all adjacent Jack nodes, and
            effective_threshold = arrest_threshold * max(min_arrest_fraction,
            remaining_turns / turn_limit). Threshold decays from arrest_threshold
            down to arrest_threshold * min_arrest_fraction as turns run out.
        (2) Every adjacent node with nonzero PMF is at the BFS frontier
            (bfs_dist >= current_depth) — at the frontier Jack can only just have
            arrived, so a positive search gives no extra information over arrest
            while arrest carries a win condition. Arrest strictly dominates.
    """

    def __init__(
        self,
        arrest_threshold: float = 0.25,
        min_arrest_fraction: float = 0.4,
        pursuit_fraction: float = 0.4,
        pursuit_weight: float = 0.5,
        hideout_blend: float = 0.5,
        n_iterations: int = 30,
        rng: random.Random | None = None,
    ) -> None:
        self._arrest_threshold = arrest_threshold
        self._min_arrest_fraction = min_arrest_fraction
        self._pursuit_fraction = pursuit_fraction
        self._pursuit_weight = pursuit_weight
        self._hideout_blend = hideout_blend
        self._n_iterations = n_iterations
        self._rng = rng or random.Random()
        self._hideout_candidates: set[int] = set()
        self._jack_start_distances: dict[int, int] = {}

    # ------------------------------------------------------------------
    # CopAgent interface
    # ------------------------------------------------------------------

    def on_episode_start(self, state: GameState, game_map: Map) -> None:
        # Cache hideout candidates for this episode (fixed for the whole game).
        distances = jack_bfs_distances(state.cop_knowledge.jack_start, game_map)
        self._jack_start_distances = distances
        base_candidates = {v for v, d in distances.items() if d >= game_map.hideout_min_distance}
        zone_candidates = base_candidates & state.hideout_zone
        self._hideout_candidates = zone_candidates if zone_candidates else base_candidates

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

    def act(self, state: GameState, game_map: Map) -> tuple[list[CopTurn], RoundCopDecisions]:
        position_pmf = self.compute_pmf(state, game_map)
        hideout_pmf  = self._compute_hideout_pmf(position_pmf, state, game_map)
        current_depth = state.turn + 1
        remaining_turns = game_map.turn_limit - 1 - state.turn
        t = max(self._min_arrest_fraction, remaining_turns / game_map.turn_limit)
        effective_threshold = self._arrest_threshold * t
        assignment   = self._assign_destinations(
            position_pmf, hideout_pmf, state.cop_positions, game_map,
            frozenset(n for n, _ in state.cop_knowledge.visited_at),
            effective_threshold=effective_threshold,
            current_depth=current_depth,
        )
        turns = [
            self._decide_action(
                cop_idx, game_map.cop_nodes[dest - 1], position_pmf,
                effective_threshold, current_depth,
            )
            for cop_idx, (dest, _role, _cov, _dir) in enumerate(assignment)
        ]
        decisions = RoundCopDecisions(
            position_pmf=position_pmf,
            hideout_pmf=hideout_pmf,
            cops=[
                CopDecisionInfo(
                    cop_idx=cop_idx,
                    role=role,
                    destination=dest,
                    coverage_score=cov,
                    direction_score=dir_score,
                )
                for cop_idx, (dest, role, cov, dir_score) in enumerate(assignment)
            ],
        )
        return turns, decisions

    # ------------------------------------------------------------------
    # Position PMF
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pmf(state: GameState, game_map: Map) -> dict[int, float]:
        """
        Bitmask forward DP over (turn, jack_node, visited_waypoints_mask).

        Returns a normalised probability dict {jack_node_id: probability}.
        """
        ck = state.cop_knowledge
        current_depth = state.turn + 1  # Jack just moved; state.turn not yet incremented

        # ------ waypoint index ------
        waypoints = sorted({n for n, _ in ck.visited_at})  # deterministic ordering
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

        # Temporal waypoint constraint: if waypoint v was first confirmed at
        # depth D, then by turn D the path must have already visited v.
        # required_masks[t] = bitmask of waypoints that must be in the path's
        # mask by turn t. Paths that lag behind this schedule are pruned.
        first_hit_depth: dict[int, int] = dict(ck.visited_at)
        required_masks: list[int] = [0] * (current_depth + 1)
        for v, d in first_hit_depth.items():
            if v in wp_idx:
                bit = 1 << wp_idx[v]
                for t in range(min(d, current_depth), current_depth + 1):
                    required_masks[t] |= bit

        # ------ DP tables ------
        jack_start = ck.jack_start
        n = len(game_map.jack_nodes)

        prev: list[list[float]] = [[0.0] * num_masks for _ in range(n + 1)]
        start_mask = (1 << wp_idx[jack_start]) if jack_start in wp_idx else 0
        prev[jack_start][start_mask] = 1.0

        for t in range(1, current_depth + 1):
            req = required_masks[t]
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
                        if (new_mask & req) != req:
                            continue
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

    # ------------------------------------------------------------------
    # ACO-based destination assignment
    # ------------------------------------------------------------------

    def _assign_destinations(
        self,
        position_pmf: dict[int, float],
        hideout_pmf: dict[int, float],
        cop_positions: tuple[int, ...],
        game_map: Map,
        confirmed_visited: frozenset[int] = frozenset(),
        effective_threshold: float = 0.0,
        current_depth: int = 0,
    ) -> list[tuple[int, str, float, float | None]]:
        """
        Runs n_iterations random orderings and per-cop role draws (pursuer vs
        searcher). Returns the assignment with the highest total score as a list
        of (destination, role, coverage_score, direction_score) per cop.

        Pursuer score for a cop node:
            coverage + pursuit_weight * proximity_normalised
        Searcher score:
            coverage only

        proximity_normalised is 1 for the cop node closest to the position PMF
        centroid and 0 for the furthest. Using the position PMF centroid (not the
        hideout centroid) means the signal is always meaningful and well-grounded
        — it rescues cops that drifted into zero-coverage corners by pulling them
        back toward where Jack is believed to be right now.
        """
        # Position PMF centroid — probability-weighted mean Jack location.
        cx = sum(p * game_map.jack_nodes[v - 1].x for v, p in position_pmf.items())
        cy = sum(p * game_map.jack_nodes[v - 1].y for v, p in position_pmf.items())

        # Hideout PMF centroid — blend toward likely destination so pursuers
        # intercept Jack's path rather than purely chasing his current position.
        hx = sum(p * game_map.jack_nodes[h - 1].x for h, p in hideout_pmf.items())
        hy = sum(p * game_map.jack_nodes[h - 1].y for h, p in hideout_pmf.items())
        blend = self._hideout_blend
        tx = (1.0 - blend) * cx + blend * hx
        ty = (1.0 - blend) * cy + blend * hy

        # Pre-compute each cop's reachable set (same every iteration).
        reachable_sets = [
            reachable_cop_nodes(pos, game_map, max_steps=2)
            for pos in cop_positions
        ]

        # Pre-compute proximity scores for all reachable cop nodes across all cops.
        # proximity = negative distance to PMF centroid; normalise globally to [0, 1].
        all_prox: dict[int, float] = {}
        for rs in reachable_sets:
            for cid in rs:
                if cid not in all_prox:
                    cn = game_map.cop_nodes[cid - 1]
                    all_prox[cid] = -math.hypot(cn.x - tx, cn.y - ty)

        prox_min = min(all_prox.values())
        prox_range = max(all_prox.values()) - prox_min
        if prox_range < 1e-9:
            norm_prox: dict[int, float] = {cid: 0.0 for cid in all_prox}
        else:
            norm_prox = {
                cid: (d - prox_min) / prox_range
                for cid, d in all_prox.items()
            }

        best_score: float = -1.0
        # Each entry: (destination, role, coverage_score, direction_score)
        best_assignment: list[tuple[int, str, float, float | None]] = []

        for _ in range(self._n_iterations):
            # Random cop ordering and role draw for this iteration
            order = list(range(len(cop_positions)))
            self._rng.shuffle(order)
            is_pursuer = [self._rng.random() < self._pursuit_fraction for _ in cop_positions]

            remaining: dict[int, float] = {
                k: v for k, v in position_pmf.items() if k not in confirmed_visited
            }
            assignment: list[tuple[int, str, float, float | None]] = [
                (0, "searcher", 0.0, None)
            ] * len(cop_positions)
            iteration_score: float = 0.0
            occupied: set[int] = set()  # destinations claimed this iteration

            for cop_idx in order:
                pursuer = is_pursuer[cop_idx]
                # Exclude already-occupied nodes so two cops never share a node
                reachable = reachable_sets[cop_idx] - occupied
                if not reachable:
                    reachable = reachable_sets[cop_idx]  # fallback if all taken

                def node_score(_cid: int, _pursuer: bool = pursuer) -> float:
                    coverage = sum(
                        remaining.get(jn.id, 0.0)
                        for jn in game_map.cop_nodes[_cid - 1].jack_neighbours
                    )
                    # Pursuers: full proximity bonus to chase toward the PMF centroid.
                    # Searchers: small proximity bonus (1/4) as a tiebreaker so they
                    # drift toward the action instead of wandering randomly when all
                    # coverage values are 0 (common in early turns when cops are far
                    # from Jack's starting zone).
                    prox_weight = self._pursuit_weight if _pursuer else self._pursuit_weight * 0.25
                    return coverage + prox_weight * norm_prox.get(_cid, 0.0)

                best_node = max(reachable, key=node_score)
                occupied.add(best_node)

                plain_coverage = sum(
                    remaining.get(jn.id, 0.0)
                    for jn in game_map.cop_nodes[best_node - 1].jack_neighbours
                )
                dir_score = norm_prox.get(best_node)
                role = "pursuer" if pursuer else "searcher"
                # Use plain coverage (no pursuit bonus) to compare iterations.
                # The pursuit bias guides within-iteration selection; if it were
                # included in the comparison score, iterations that happen to draw
                # more pursuers would appear better regardless of actual coverage.
                iteration_score += plain_coverage
                assignment[cop_idx] = (best_node, role, plain_coverage, dir_score)

                # If this cop would arrest, its adjacent Jack nodes are fully
                # resolved — zero them out so no other cop scores credit for
                # re-arresting the same node.  Arrest decision mirrors _decide_action.
                cop_adj = game_map.cop_nodes[best_node - 1].jack_neighbours
                zone_mass = sum(position_pmf.get(jn.id, 0.0) for jn in cop_adj)
                would_arrest = zone_mass > 0.0 and (
                    zone_mass >= effective_threshold
                    or all(
                        self._jack_start_distances.get(jn.id, 0) >= current_depth
                        for jn in cop_adj if position_pmf.get(jn.id, 0.0) > 0.0
                    )
                )
                for jn in cop_adj:
                    if jn.id in remaining:
                        if would_arrest:
                            remaining[jn.id] = 0.0
                        else:
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
        effective_threshold: float,
        current_depth: int,
    ) -> CopTurn:
        adj = cop_node.jack_neighbours
        if not adj:
            return CopTurn(cop_idx=cop_idx, destination=cop_node.id, search=True)

        zone_mass = sum(pmf.get(jn.id, 0.0) for jn in adj)

        if zone_mass > 0.0:
            all_frontier = all(
                self._jack_start_distances.get(jn.id, 0) >= current_depth
                for jn in adj if pmf.get(jn.id, 0.0) > 0.0
            )
            if zone_mass >= effective_threshold or all_frontier:
                return CopTurn(
                    cop_idx=cop_idx,
                    destination=cop_node.id,
                    search=False,
                    arrest_all=True,
                )

        return CopTurn(cop_idx=cop_idx, destination=cop_node.id, search=True)
