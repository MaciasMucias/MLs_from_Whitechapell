from __future__ import annotations
import random
from dataclasses import dataclass, field

from agents.base import RoundCopDecisions
from engine.env import (
    CopTurn,
    end_of_round,
    legal_jack_edges,
    make_initial_state,
    step_cop,
    step_jack,
)
from engine.graph import JackEdge, Map
from engine.state import GameState


# ---------------------------------------------------------------------------
# History types
# ---------------------------------------------------------------------------

@dataclass
class CopStepRecord:
    cop_turn: CopTurn
    state_after: GameState
    terminated: bool
    winner: str | None
    search_results: dict[int, bool] = field(default_factory=dict)
    # jack_node_id -> True=hit (Jack visited) / False=miss; empty for arrest actions


@dataclass
class RoundRecord:
    """
    Complete record of one round. Sufficient for replay and PPO rollout
    construction. jack_logprob and jack_value are None for non-RL agents.
    """
    turn: int
    state_before: GameState
    jack_edge: JackEdge
    jack_logprob: float | None
    jack_value: float | None
    state_after_jack: GameState
    cop_steps: tuple[CopStepRecord, ...]
    state_after_round: GameState
    terminated: bool
    winner: str | None
    cop_decisions: RoundCopDecisions | None = None


@dataclass
class GameRecord:
    game_map: Map
    initial_state: GameState
    winner: str
    history: list[RoundRecord]

    @property
    def turns_survived(self) -> int:
        return len(self.history)


# ---------------------------------------------------------------------------
# Step context
# ---------------------------------------------------------------------------

@dataclass
class StepContext:
    """
    Mutable working surface for a live game. Shared by the server (one
    context per HTTP request, projected from GameSession) and run_game()
    (created and owned internally).
    """
    game_map: Map
    state: GameState
    terminated: bool
    winner: str | None
    history: list[RoundRecord] = field(default_factory=list)
    blocking: bool = False
    turn_limit: int | None = None
    arrest_all_enabled: bool = True


# ---------------------------------------------------------------------------
# Core step function
# ---------------------------------------------------------------------------

def step_round(
    ctx: StepContext,
    jack_edge: JackEdge,
    cop_agent,           # CopAgent
    director=None,       # Director | None
) -> tuple[list[dict], bool, str | None]:
    """
    Advance one full round given Jack's chosen edge.

    Turn order:
      1. Jack moves        → immediate win if hideout reached
      2. Director filters  → manipulates cop_knowledge.visited before cops plan
      3. Cops plan + act   → immediate win if arrest succeeds
      4. end_of_round      → checks turn limit and blocking condition

    Updates ctx in-place. Returns (events, terminated, winner) where events
    is a list of cop-action dicts for frontend animation.
    """
    state = ctx.state
    turn = state.turn
    state_before = state

    # 1. Jack moves
    state, terminated, winner = step_jack(state, jack_edge)
    state_after_jack = state

    events: list[dict] = []
    cop_steps: list[CopStepRecord] = []
    round_decisions = None

    if not terminated:
        # 2. Director filters  → manipulates cop_knowledge.visited_at before cops plan
        if director is not None:
            state = director.filter_knowledge(state, ctx.game_map)

        # 3. All cops plan simultaneously on the post-Director state
        cop_turns, round_decisions = cop_agent.act(state, ctx.game_map)

        # Execute each cop's planned turn
        for cop_turn in cop_turns:
            cop_node = ctx.game_map.cop_nodes[cop_turn.destination - 1]
            # Compute search results before advancing state (jack_trace is ground truth)
            if cop_turn.search:
                search_results = {
                    jn.id: (jn.id in state.jack_trace)
                    for jn in cop_node.jack_neighbours
                }
            else:
                search_results = {}

            state, terminated, winner = step_cop(state, cop_turn, ctx.game_map)
            events.append({
                "cop": cop_turn.cop_idx,
                "moved_to": cop_turn.destination,
                "action": "search" if cop_turn.search else "arrest",
                "jack_neighbours": [n.id for n in cop_node.jack_neighbours],
                "arrest_target": cop_turn.arrest_target,
                "arrest_all": cop_turn.arrest_all,
                "search_hits": [jn_id for jn_id, hit in search_results.items() if hit],
                "arrest_success": terminated and winner == "cops" and not cop_turn.search,
            })
            cop_steps.append(CopStepRecord(
                cop_turn=cop_turn,
                state_after=state,
                terminated=terminated,
                winner=winner,
                search_results=search_results,
            ))
            if terminated:
                break

        # 4. End of round checks (turn limit, blocking)
        if not terminated:
            effective_limit = ctx.turn_limit if ctx.turn_limit is not None else ctx.game_map.turn_limit
            state, terminated, winner = end_of_round(
                state, ctx.game_map,
                blocking=ctx.blocking,
                turn_limit=effective_limit,
            )

    record = RoundRecord(
        turn=turn,
        state_before=state_before,
        jack_edge=jack_edge,
        jack_logprob=None,
        jack_value=None,
        state_after_jack=state_after_jack,
        cop_steps=tuple(cop_steps),
        state_after_round=state,
        terminated=terminated,
        winner=winner,
        cop_decisions=round_decisions,
    )
    ctx.history.append(record)
    ctx.state = state
    ctx.terminated = terminated
    ctx.winner = winner

    return events, terminated, winner


# ---------------------------------------------------------------------------
# Run-to-completion loop
# ---------------------------------------------------------------------------

def run_game(
    game_map: Map,
    jack_agent,           # JackAgent
    cop_agent,            # CopAgent
    director=None,        # Director | None — defaults to NoOpDirector
    rng: random.Random | None = None,
    blocking: bool = False,
    turn_limit: int | None = None,
) -> GameRecord:
    """
    Run a complete game and return a GameRecord.

    jack_logprob and jack_value are captured from AgentOutput and stored in
    each RoundRecord so the PPO training loop can build advantage estimates
    without re-running inference.
    """
    from agents.random_agents import NoOpDirector  # deferred to avoid circular import
    if director is None:
        director = NoOpDirector()
    if rng is None:
        rng = random.Random()

    initial_state = make_initial_state(game_map, rng)
    ctx = StepContext(
        game_map=game_map,
        state=initial_state,
        terminated=False,
        winner=None,
        blocking=blocking,
        turn_limit=turn_limit,
    )

    jack_agent.on_episode_start(initial_state, game_map)
    cop_agent.on_episode_start(initial_state, game_map)

    while not ctx.terminated:
        legal_edges = legal_jack_edges(ctx.state, game_map, blocking=blocking)
        output = jack_agent.act(ctx.state, legal_edges, game_map)
        step_round(ctx, output.edge, cop_agent, director)
        # Patch logprob/value onto the mutable RoundRecord
        last = ctx.history[-1]
        last.jack_logprob = output.logprob
        last.jack_value = output.value

    jack_agent.on_episode_end(ctx.state, ctx.winner)
    cop_agent.on_episode_end(ctx.state, ctx.winner)
    director.on_game_end(ctx.winner, len(ctx.history))

    return GameRecord(
        game_map=game_map,
        initial_state=initial_state,
        winner=ctx.winner,
        history=ctx.history,
    )
