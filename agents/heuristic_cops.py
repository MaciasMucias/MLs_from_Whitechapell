from __future__ import annotations
import math
from collections import defaultdict

import numpy as np

from engine.env import CopTurn
from engine.graph import Map
from engine.graph_utils import (
    jack_bfs_distances,
    jack_reachable_within,
    reachable_cop_nodes,
)
from engine.state import GameState
from agents.base import CopAgent, CopDecisionInfo, RoundCopDecisions

# Minimum proximity range below which all nodes are treated as equidistant.
_PROX_RANGE_EPS = 1e-9
# Fallback search discount used when current_depth is zero (should never occur in practice).
_SEARCH_DISC_ZERO_DEPTH_FALLBACK = 0.5


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

    Movement (coordinate ascent):
        Runs up to max_passes convergence passes. Each pass processes cops in
        index order; for each cop, holds all others fixed at their current
        destinations and picks the best response. Stops when no cop changes.
        Roles are deterministic: the closest round(n * pursuit_fraction) cops
        to the direction target are pursuers; the rest are searchers. Pursuers
        score cop nodes by coverage + pursuit_weight * direction_toward_target;
        searchers by coverage + pursuit_weight * searcher_prox_fraction * direction.

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
        min_arrest_fraction: float = 0.8,
        pursuit_fraction: float = 0.4,
        pursuit_weight: float = 0.5,
        searcher_prox_fraction: float = 0.5,
        direction_certainty_threshold: float = 0.15,
        arrest_discount: float = 0.0,
        miss_discount_decay: float = 0.7,
        hideout_blend: float = 0.5,
        hideout_blend_floor: float = 0.330,
        max_passes: int = 5,
        cop_max_steps: int = 2,
    ) -> None:
        self._arrest_threshold = arrest_threshold
        self._min_arrest_fraction = min_arrest_fraction
        self._pursuit_fraction = pursuit_fraction
        self._pursuit_weight = pursuit_weight
        self._searcher_prox_fraction = searcher_prox_fraction
        self._direction_certainty_threshold = direction_certainty_threshold
        self._arrest_discount = arrest_discount
        self._miss_discount_decay = miss_discount_decay
        self._hideout_blend = hideout_blend
        self._hideout_blend_floor = hideout_blend_floor
        self._max_passes = max_passes
        self._cop_max_steps = cop_max_steps
        self._hideout_candidates: set[int] = set()
        self._hideout_candidate_list: list[int] = []
        self._hideout_dist_arr: np.ndarray = np.empty((0, 0), dtype=np.int32)
        self._jack_start_distances: dict[int, int] = {}
        self._last_position_pmf: dict[int, float] = {}

    # ------------------------------------------------------------------
    # CopAgent interface
    # ------------------------------------------------------------------

    def on_episode_start(self, state: GameState, game_map: Map) -> None:
        # Cache hideout candidates for this episode (fixed for the whole game).
        distances = jack_bfs_distances(state.cop_knowledge.jack_start, game_map)
        self._jack_start_distances = distances
        self._hideout_candidates = state.hideout_zone

        # Precompute distance matrix for _compute_hideout_pmf vectorization.
        # _hideout_dist_arr[i, v] = BFS distance from candidate i to Jack node v.
        # Unreachable pairs are set to n_jack (guaranteed > any remaining_hops).
        candidates_list = sorted(state.hideout_zone)
        self._hideout_candidate_list = candidates_list
        n_jack = len(game_map.jack_nodes)
        dist_arr = np.full((len(candidates_list), n_jack), n_jack, dtype=np.int32)
        for i, h in enumerate(candidates_list):
            for v, d in jack_bfs_distances(h, game_map).items():
                dist_arr[i, v] = d
        self._hideout_dist_arr = dist_arr

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

    @property
    def last_position_pmf(self) -> dict[int, float]:
        """PMF computed during the most recent act() call. For reward shaping."""
        return self._last_position_pmf

    def act(
        self, state: GameState, game_map: Map
    ) -> tuple[list[CopTurn], RoundCopDecisions]:
        position_pmf = self.compute_pmf(state, game_map)
        self._last_position_pmf = position_pmf
        hideout_pmf = self._compute_hideout_pmf(position_pmf, state, game_map)
        current_depth = state.turn + 1
        remaining_turns = game_map.turn_limit - 1 - state.turn
        t = max(self._min_arrest_fraction, remaining_turns / game_map.turn_limit)
        effective_threshold = self._arrest_threshold * t
        assignment = self._assign_destinations(
            position_pmf,
            hideout_pmf,
            state.cop_positions,
            game_map,
            frozenset(n for n, _ in state.cop_knowledge.visited_at),
            tuple(state.cop_knowledge.search_misses),
            effective_threshold=effective_threshold,
            current_depth=current_depth,
        )
        turns = [
            self._decide_action(
                cop_idx,
                game_map.cop_nodes[dest],
                position_pmf,
                effective_threshold,
                current_depth,
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
        Mass values accumulate as paths fan out; normalisation happens once at
        the end — there is no per-step normalisation.

        Caching across turns is not practical: current_depth increases every
        turn, the waypoint set (and thus bitmask dimension) can expand, and new
        search/arrest misses can invalidate any previously computed entries.
        search_exclude/arrest_exclude are O(|misses|) to rebuild, which is
        negligible compared to the DP itself.
        """
        ck = state.cop_knowledge
        current_depth = (
            state.turn + 1
        )  # Jack just moved; state.turn not yet incremented

        # ------ waypoint index ------
        waypoints = sorted({n for n, _ in ck.visited_at})  # deterministic ordering
        wp_idx: dict[int, int] = {v: i for i, v in enumerate(waypoints)}
        num_masks = 1 << len(waypoints)
        full_mask = num_masks - 1

        # ------ constraint sets ------
        # search_miss (v, T): counts[t][v][*] = 0 for t <= T
        search_exclude: dict[int, int] = {}  # node_id -> max excluded turn (inclusive)
        for v, T in ck.search_misses:
            if v not in search_exclude or search_exclude[v] < T:
                search_exclude[v] = T

        # arrest_miss (v, T): counts[T][v][*] = 0
        arrest_exclude: set[tuple[int, int]] = set(ck.arrest_misses)

        # Temporal waypoint constraint: if waypoint v was first confirmed at
        # depth D, then by turn D the path must have already visited v.
        # required_masks[t] = bitmask of waypoints that must be in the path's
        # mask by turn t. Paths that lag behind this schedule are pruned.
        # Sized current_depth+1 so index t is valid for t in 0..current_depth;
        # index 0 is unused (loop starts at t=1).
        first_hit_depth: dict[int, int] = dict(ck.visited_at)
        required_masks: list[int] = [0] * (current_depth + 1)
        for v, d in first_hit_depth.items():
            if v in wp_idx:
                bit = 1 << wp_idx[v]
                for t in range(min(d, current_depth), current_depth + 1):
                    required_masks[t] |= bit

        # ------ DP tables ------
        # Sized n because node IDs are 0-based; node.id is a valid direct index.
        jack_start = ck.jack_start
        n = len(game_map.jack_nodes)

        # Group edges by (v_id, wp_bit) so each group can be handled with one
        # numpy scatter-add instead of a Python loop over individual masks.
        # Done once before the time-step loop since the graph is constant.
        edge_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for u_node in game_map.jack_nodes:
            for edge in u_node.edges:
                v_id = edge.destination.id
                wp_b = (1 << wp_idx[v_id]) if v_id in wp_idx else 0
                edge_groups[(v_id, wp_b)].append(u_node.id)

        all_m = np.arange(num_masks, dtype=np.int32)

        # Precompute per-group arrays that don't depend on t or req.
        # target_arr[m] = m | wp_bit (None for non-waypoint groups → identity).
        group_list = [
            (v_id, u_arr, (all_m | wp_bit) if wp_bit else None)
            for (v_id, wp_bit), u_ids in edge_groups.items()
            for u_arr in [np.array(u_ids, dtype=np.int32)]
        ]

        start_mask = (1 << wp_idx[jack_start]) if jack_start in wp_idx else 0
        prev = np.zeros((n, num_masks), dtype=np.float64)
        prev[jack_start, start_mask] = 1.0

        for t in range(1, current_depth + 1):
            req = required_masks[t]
            curr = np.zeros((n, num_masks), dtype=np.float64)

            for v_id, u_arr, target_arr in group_list:
                if search_exclude.get(v_id, -1) >= t:
                    continue
                if (v_id, t) in arrest_exclude:
                    continue

                # Sum mass from all sources reaching this destination.
                combined = prev[u_arr].sum(axis=0)  # shape (num_masks,)

                if target_arr is None:
                    # Non-waypoint: target mask = source mask.
                    # Valid iff (mask & req) == req.
                    curr[v_id] += combined * ((all_m & req) == req)
                else:
                    # Waypoint: target mask = source mask | wp_bit.
                    # Valid iff (target & req) == req.
                    req_valid = (target_arr & req) == req  # (num_masks,) bool
                    curr[v_id] += np.bincount(
                        target_arr, weights=combined * req_valid, minlength=num_masks
                    )

            prev = curr

        # ------ extract terminal distribution ------
        terminal = prev[:, full_mask]  # (n,)
        raw = {int(v): float(m) for v, m in enumerate(terminal) if m > 0.0}

        if not raw:
            row_sums = prev.sum(axis=1)  # (n,)
            raw = {int(v): float(s) for v, s in enumerate(row_sums) if s > 0.0}

        if not raw:
            return {node.id: 1.0 / n for node in game_map.jack_nodes}

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
            candidates = {
                v for v, d in distances.items() if d >= game_map.hideout_min_distance
            }

        if not position_pmf:
            return {h: 1.0 / len(candidates) for h in candidates}

        # Vectorised reachability: _hideout_dist_arr[i, v] = dist from candidate i
        # to Jack node v.  A candidate reaches v if dist <= remaining_hops.
        pmf_nodes = np.array(list(position_pmf.keys()), dtype=np.int32)
        pmf_probs = np.array(list(position_pmf.values()), dtype=np.float64)
        # reachable_mask shape: (n_candidates, |pmf|)
        reachable_mask = self._hideout_dist_arr[:, pmf_nodes] <= remaining_hops
        scores_arr = reachable_mask @ pmf_probs  # (n_candidates,)

        total = float(scores_arr.sum())
        if total == 0.0:
            # All candidates unreachable — constraints have become inconsistent.
            return {h: 1.0 / len(candidates) for h in candidates}

        return {
            h: float(s) / total
            for h, s in zip(self._hideout_candidate_list, scores_arr)
            if s > 0.0
        }

    # ------------------------------------------------------------------
    # Heading estimation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Coordinate-ascent destination assignment
    # ------------------------------------------------------------------

    def _assign_destinations(
        self,
        position_pmf: dict[int, float],
        hideout_pmf: dict[int, float],
        cop_positions: tuple[int, ...],
        game_map: Map,
        confirmed_visited: frozenset[int] = frozenset(),
        search_misses: tuple[tuple[int, int], ...] = (),
        effective_threshold: float = 0.0,
        current_depth: int = 0,
    ) -> list[tuple[int, str, float, float | None]]:
        """
        Coordinate-ascent cop assignment. Runs up to max_passes convergence
        passes. Each pass processes cops in index order; for each cop, holds
        all others fixed at their current destinations and finds the best
        response (coverage + proximity bonus). Stops early when no cop changes.

        Pursuer roles are assigned deterministically: the closest
        round(n * pursuit_fraction) cops to the direction target are pursuers;
        the rest are searchers.

        The direction target is a time-blended mix of the frontier centroid
        (PMF nodes at BFS depth >= current_depth - 1) and the hideout centroid,
        identical to the previous ACO approach.
        """
        # Full PMF centroid — weighted mean over all Jack nodes with nonzero
        # probability. The previous frontier filter (BFS depth >= current_depth-1)
        # caused the centroid to be dominated by sparse outlier nodes in late turns,
        # pointing pursuers at the wrong area of the board.
        fcx = sum(p * game_map.jack_nodes[v].x for v, p in position_pmf.items())
        fcy = sum(p * game_map.jack_nodes[v].y for v, p in position_pmf.items())

        # Hideout PMF centroid — blend toward likely destination so pursuers
        # intercept Jack's path rather than purely chasing his current position.
        # Scale the blend with turn progress so that early-game pursuers track
        # Jack's frontier (blend ≈ 0) and late-game pursuers intercept his path
        # to the hideout (blend → hideout_blend). Fixed blend causes cops to aim
        # at the center of the board when Jack is on one side and the hideout is
        # on the other.
        hx = sum(p * game_map.jack_nodes[h].x for h, p in hideout_pmf.items())
        hy = sum(p * game_map.jack_nodes[h].y for h, p in hideout_pmf.items())
        turn = current_depth - 1
        blend = self._hideout_blend_floor + (
            self._hideout_blend - self._hideout_blend_floor
        ) * turn / max(1, game_map.turn_limit - 1)
        tx = (1.0 - blend) * fcx + blend * hx
        ty = (1.0 - blend) * fcy + blend * hy

        # Pre-compute each cop's reachable set (same every pass).
        reachable_sets = [
            reachable_cop_nodes(pos, game_map, max_steps=self._cop_max_steps)
            for pos in cop_positions
        ]

        # Pre-compute proximity scores for all reachable cop nodes across all cops.
        # Negated distance so that max() picks the closest node (closer → larger value).
        # Normalised globally to [0, 1] after all values are collected.
        all_prox: dict[int, float] = {}
        for rs in reachable_sets:
            for cid in rs:
                if cid not in all_prox:
                    cn = game_map.cop_nodes[cid]
                    all_prox[cid] = -math.hypot(cn.x - tx, cn.y - ty)

        prox_min = min(all_prox.values())
        prox_range = max(all_prox.values()) - prox_min
        if prox_range < _PROX_RANGE_EPS:
            norm_prox: dict[int, float] = {cid: 0.0 for cid in all_prox}
        else:
            norm_prox = {
                cid: (d - prox_min) / prox_range for cid, d in all_prox.items()
            }

        # Direction certainty: scale the proximity bonus by how concentrated the PMF
        # is.  When cops are right on Jack's tail, one node dominates (high max_p) and
        # direction guidance is reliable.  When the PMF is flat, max_p is low and the
        # centroid is unreliable, so the bonus fades smoothly to zero.
        direction_certainty = min(
            1.0, max(position_pmf.values()) / self._direction_certainty_threshold
        )

        # Most-recent miss turn per Jack node — used to compute the history discount.
        # Built once since search_misses is fixed for the entire assignment call.
        last_searched: dict[int, int] = {}
        for v, t in search_misses:
            if v not in last_searched or last_searched[v] < t:
                last_searched[v] = t

        n_cops = len(cop_positions)

        # Deterministic pursuer assignment: closest cops to target = pursuers.
        order_by_prox = sorted(
            range(n_cops),
            key=lambda i: math.hypot(
                game_map.cop_nodes[cop_positions[i]].x - tx,
                game_map.cop_nodes[cop_positions[i]].y - ty,
            ),
        )
        n_pursuers = max(1, round(n_cops * self._pursuit_fraction))
        pursuer_set = set(order_by_prox[:n_pursuers])
        is_pursuer = [i in pursuer_set for i in range(n_cops)]

        # Initialise: every cop stays at its current position.
        dests = list(cop_positions)

        def _build_remaining() -> tuple[dict[int, float], dict[int, float]]:
            ra: dict[int, float] = {
                k: v for k, v in position_pmf.items() if k not in confirmed_visited
            }
            rs: dict[int, float] = {}
            for k, v in position_pmf.items():
                if k in confirmed_visited:
                    continue
                if k in last_searched:
                    turns_since = current_depth - last_searched[k]
                    rs[k] = v * (1.0 - self._miss_discount_decay**turns_since)
                else:
                    rs[k] = v
            return ra, rs

        def _apply_deduction(
            dest_j: int,
            ra: dict[int, float],
            rs: dict[int, float],
        ) -> None:
            cn_j = game_map.cop_nodes[dest_j]
            if self._would_arrest(
                cn_j, position_pmf, effective_threshold, current_depth
            ):
                for jn in cn_j.jack_neighbours:
                    if jn.id in ra:
                        ra[jn.id] *= self._arrest_discount
            else:
                disc = (
                    1.0 / current_depth
                    if current_depth > 0
                    else _SEARCH_DISC_ZERO_DEPTH_FALLBACK
                )
                for jn in cn_j.jack_neighbours:
                    if jn.id in rs:
                        rs[jn.id] *= disc

        for _pass in range(self._max_passes):
            changed = False

            for cop_idx in range(n_cops):
                pursuer = is_pursuer[cop_idx]

                # Build remaining dicts and apply all other cops' deductions.
                ra, rs = _build_remaining()
                for j, dest_j in enumerate(dests):
                    if j != cop_idx:
                        _apply_deduction(dest_j, ra, rs)

                others_dests = {dests[j] for j in range(n_cops) if j != cop_idx}
                reachable = reachable_sets[cop_idx] - others_dests
                if not reachable:
                    continue

                # Spread scores: reward nodes that are far from other cops'
                # current destinations.  Normalised within this cop's reachable
                # set so the signal is always on the same scale as norm_prox.
                # Applies equally to pursuers and searchers — encirclement needs
                # every cop approaching from a different angle.
                if others_dests and direction_certainty < 1.0:
                    raw_spread = {
                        cid: min(
                            math.hypot(
                                game_map.cop_nodes[cid].x - game_map.cop_nodes[oc].x,
                                game_map.cop_nodes[cid].y - game_map.cop_nodes[oc].y,
                            )
                            for oc in others_dests
                        )
                        for cid in reachable
                    }
                    sp_max = max(raw_spread.values())
                    norm_spread_local: dict[int, float] = (
                        {cid: v / sp_max for cid, v in raw_spread.items()}
                        if sp_max > _PROX_RANGE_EPS
                        else {cid: 0.0 for cid in reachable}
                    )
                else:
                    norm_spread_local = {}

                def node_score(
                    _cid: int,
                    _pursuer: bool = pursuer,
                    _ra: dict[int, float] = ra,
                    _rs: dict[int, float] = rs,
                    _norm_spread: dict[int, float] = norm_spread_local,
                ) -> float:
                    cn = game_map.cop_nodes[_cid]
                    if self._would_arrest(
                        cn, position_pmf, effective_threshold, current_depth
                    ):
                        coverage = sum(_ra.get(jn.id, 0.0) for jn in cn.jack_neighbours)
                    else:
                        coverage = sum(_rs.get(jn.id, 0.0) for jn in cn.jack_neighbours)
                    prox_weight = (
                        self._pursuit_weight * direction_certainty
                        if _pursuer
                        else self._pursuit_weight
                        * self._searcher_prox_fraction
                        * direction_certainty
                    )
                    spread_weight = self._pursuit_weight * (1.0 - direction_certainty)
                    return (
                        coverage
                        + prox_weight * norm_prox.get(_cid, 0.0)
                        + spread_weight
                        * _norm_spread.get(_cid, 0.0)
                        * norm_prox.get(_cid, 0.0)
                    )

                best_node = max(reachable, key=node_score)
                if best_node != dests[cop_idx]:
                    changed = True
                    dests[cop_idx] = best_node

            if not changed:
                break

        # Final scoring pass: compute plain coverage and direction scores for
        # RoundCopDecisions using a single sequential forward pass over converged dests.
        ra_final, rs_final = _build_remaining()
        assignment: list[tuple[int, str, float, float | None]] = []
        for cop_idx in range(n_cops):
            dest = dests[cop_idx]
            cn = game_map.cop_nodes[dest]
            if self._would_arrest(cn, position_pmf, effective_threshold, current_depth):
                plain_coverage = sum(
                    ra_final.get(jn.id, 0.0) for jn in cn.jack_neighbours
                )
                for jn in cn.jack_neighbours:
                    if jn.id in ra_final:
                        ra_final[jn.id] *= self._arrest_discount
            else:
                plain_coverage = sum(
                    rs_final.get(jn.id, 0.0) for jn in cn.jack_neighbours
                )
                disc = (
                    1.0 / current_depth
                    if current_depth > 0
                    else _SEARCH_DISC_ZERO_DEPTH_FALLBACK
                )
                for jn in cn.jack_neighbours:
                    if jn.id in rs_final:
                        rs_final[jn.id] *= disc
            role = "pursuer" if is_pursuer[cop_idx] else "searcher"
            assignment.append((dest, role, plain_coverage, norm_prox.get(dest)))

        return assignment

    # ------------------------------------------------------------------
    # Shared arrest decision
    # ------------------------------------------------------------------

    def _would_arrest(
        self,
        cop_node,
        pmf: dict[int, float],
        effective_threshold: float,
        current_depth: int,
    ) -> bool:
        adj = cop_node.jack_neighbours
        if not adj:
            return False
        zone_mass = sum(pmf.get(jn.id, 0.0) for jn in adj)
        if zone_mass == 0.0:
            return False
        return zone_mass >= effective_threshold or all(
            self._jack_start_distances.get(jn.id, 0) >= current_depth
            for jn in adj
            if pmf.get(jn.id, 0.0) > 0.0
        )

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
        if not cop_node.jack_neighbours:
            return CopTurn(cop_idx=cop_idx, destination=cop_node.id, search=True)

        if self._would_arrest(cop_node, pmf, effective_threshold, current_depth):
            return CopTurn(
                cop_idx=cop_idx, destination=cop_node.id, search=False, arrest_all=True
            )

        return CopTurn(cop_idx=cop_idx, destination=cop_node.id, search=True)
