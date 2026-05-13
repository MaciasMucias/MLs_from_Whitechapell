// ============================================================
// Replay viewer — communicates with /api/replays
// ============================================================

let _replayList    = [];    // metadata from GET /api/replays
let _replayRecord  = null;  // full ReplayRecord currently loaded
let _replayRound   = -1;    // -1 = initial state; 0+ = index into _replayRecord.rounds
let _replaySubStep = 0;     // 0 = after Jack moved, 1 = after all cops acted
let _replayPmf     = false; // whether PMF overlay is active in replay

// ── Init ────────────────────────────────────────────────────

async function initReplay() {
  document.getElementById("replay-refresh-btn").addEventListener("click", loadReplayList);
  document.getElementById("replay-prev-btn").addEventListener("click", () => stepReplay(-1));
  document.getElementById("replay-next-btn").addEventListener("click", () => stepReplay(+1));
  document.getElementById("replay-start-btn").addEventListener("click", () => gotoStep(-1, 0));
  document.getElementById("replay-end-btn").addEventListener("click", () => {
    if (_replayRecord) {
      const lastRound = _replayRecord.rounds.length - 1;
      const lastSub   = _replayRecord.rounds[lastRound].cop_actions.length > 0 ? 1 : 0;
      gotoStep(lastRound, lastSub);
    }
  });
  document.getElementById("replay-pmf-check").addEventListener("change", e => {
    _replayPmf = e.target.checked;
    window.setPmfLayerEnabled(_replayPmf);
    renderReplayRound();
  });
}

// ── Load list ────────────────────────────────────────────────

async function forkFromCurrentTurn() {
  if (!_replayRecord) return;
  const slot = parseInt(document.getElementById("replay-list").value);
  if (isNaN(slot)) return;
  try {
    const r = await fetch(`/api/replays/${slot}/fork-at-turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turn: _replayRound }),
    });
    if (!r.ok) { console.error("fork-at-turn: HTTP", r.status); return; }
    const state = await r.json();
    window.startGameFromState(state);
  } catch (e) {
    console.error("forkFromCurrentTurn:", e);
  }
}

async function loadReplayList() {
  const listEl = document.getElementById("replay-list");
  listEl.innerHTML = "<option disabled selected>Loading…</option>";
  try {
    const r = await fetch("/api/replays");
    _replayList = await r.json();
    listEl.innerHTML = "";
    if (!_replayList.length) {
      listEl.innerHTML = "<option disabled selected>No replays saved yet</option>";
      return;
    }
    const sorted = [..._replayList].sort((a, b) => b.timestamp?.localeCompare(a.timestamp ?? "") ?? 0);
    for (const meta of sorted) {
      const opt = document.createElement("option");
      opt.value = meta.slot;
      const ts = meta.timestamp ? new Date(meta.timestamp).toLocaleString() : "?";
      opt.textContent = `[${ts}] ${meta.winner} — ${meta.turns_survived} turns (slot ${meta.slot})`;
      listEl.appendChild(opt);
    }
    listEl.onchange = () => loadReplaySlot(parseInt(listEl.value));
    loadReplaySlot(parseInt(listEl.value));  // auto-load the initially selected slot
  } catch (e) {
    listEl.innerHTML = "<option disabled selected>Error loading replays</option>";
    console.error("loadReplayList:", e);
  }
}

async function loadReplaySlot(slot) {
  try {
    const r = await fetch(`/api/replays/${slot}`);
    if (!r.ok) { console.error("loadReplaySlot: HTTP", r.status); return; }
    _replayRecord = await r.json();
    gotoStep(-1, 0);
    renderReplayMeta();
  } catch (e) {
    console.error("loadReplaySlot:", e);
  }
}

// ── Navigation ─────────────────────────────────────────────��─

function maxSubStep(roundIdx) {
  if (!_replayRecord) return 0;
  return _replayRecord.rounds[roundIdx].cop_actions.length > 0 ? 1 : 0;
}

function stepReplay(delta) {
  if (!_replayRecord) return;
  let r = _replayRound;
  let s = _replaySubStep + delta;

  if (r === -1) {
    if (delta > 0) { r = 0; s = 0; }
    else return;
  } else if (s > maxSubStep(r)) {
    r++;
    s = 0;
  } else if (s < 0) {
    r--;
    if (r < -1) return;
    s = r < 0 ? 0 : maxSubStep(r);
  }
  if (r >= _replayRecord.rounds.length) return;
  gotoStep(r, s);
}

function gotoStep(round, subStep) {
  if (!_replayRecord) return;
  _replayRound   = Math.max(-1, Math.min(_replayRecord.rounds.length - 1, round));
  _replaySubStep = subStep;
  renderReplayRound();
}

// ── Board render ─────────────────────────────────────────────

function _copPositionsAtSubStep() {
  const pos = [..._replayRecord.initial_cop_positions];
  // Apply all completed rounds
  const roundsToApply = _replaySubStep === 0 ? _replayRound : _replayRound + 1;
  for (let i = 0; i < roundsToApply; i++) {
    for (const ca of _replayRecord.rounds[i].cop_actions) {
      pos[ca.cop_idx] = ca.to_node;
    }
  }
  return pos;
}

function renderReplayRound() {
  if (!_replayRecord || !window.mapData) return;

  if (_replayRound === -1) {
    window.renderForReplay({
      game_id:             _replayRecord.game_id,
      jack_pos:            _replayRecord.initial_jack_pos,
      cop_positions:       [..._replayRecord.initial_cop_positions],
      hideout:             _replayRecord.hideout,
      hideout_zone_anchor: _replayRecord.hideout_zone_anchor,
      hideout_zone:        _replayRecord.hideout_zone,
      turn:                0,
      turn_limit:          _replayRecord.turn_limit,
      legal_moves:         [],
      visited_at:          [],
      search_misses:       [],
      arrest_misses:       [],
      jack_trace:          [],
      blocking:            _replayRecord.blocking,
      history_size:        0,
      terminated:          false,
      winner:              null,
    });
    window.clearPmfData();
    renderInitialPanel();
    updateReplayNav();
    return;
  }

  const rnd = _replayRecord.rounds[_replayRound];

  // Substep 0 = after Jack moved, before cops act → knowledge is from end of previous round
  const prevRnd = _replaySubStep === 0 && _replayRound > 0
    ? _replayRecord.rounds[_replayRound - 1]
    : null;
  const visited      = _replaySubStep === 1 ? rnd.visited_at_after      : (prevRnd ? prevRnd.visited_at_after      : []);
  const searchMisses = _replaySubStep === 1 ? rnd.search_misses_after   : (prevRnd ? prevRnd.search_misses_after   : []);
  const arrestMisses = _replaySubStep === 1 ? rnd.arrest_misses_after   : (prevRnd ? prevRnd.arrest_misses_after   : []);

  const replayState = {
    game_id:             _replayRecord.game_id,
    jack_pos:            rnd.jack_to,
    cop_positions:       _copPositionsAtSubStep(),
    hideout:             _replayRecord.hideout,
    hideout_zone_anchor: _replayRecord.hideout_zone_anchor,
    hideout_zone:        _replayRecord.hideout_zone,
    turn:                rnd.turn,
    turn_limit:          _replayRecord.turn_limit,
    legal_moves:         [],
    visited_at:          visited,
    search_misses:       searchMisses,
    arrest_misses:       arrestMisses,
    jack_trace:          [],
    blocking:            _replayRecord.blocking,
    history_size:        0,
    terminated:          rnd.terminated,
    winner:              rnd.winner,
  };

  window.renderForReplay(replayState);

  // PMF overlay — choose source based on sub-step
  if (_replayPmf) {
    const pmfSource = _replaySubStep === 0 ? rnd.position_pmf : rnd.pmf_after_cops;
    if (pmfSource) {
      const pmf = {};
      for (const [k, v] of Object.entries(pmfSource)) pmf[parseInt(k)] = v;
      window.setPmfData(pmf);
    } else {
      window.clearPmfData();
    }
  } else {
    window.clearPmfData();
  }

  renderReplayPanel(rnd);
  updateReplayNav();
}

// ── Side panel ───────────────────────────────────────────────

function renderInitialPanel() {
  const el = document.getElementById("replay-round-detail");
  const rec = _replayRecord;
  const lines = [
    `<div class="rpl-section"><strong>Initial state</strong></div>`,
    `<div class="rpl-row"><span class="rpl-jack">Jack</span> starts at <strong>${rec.initial_jack_pos + 1}</strong></div>`,
    `<div class="rpl-info">Hideout: ${rec.hideout + 1} &nbsp;·&nbsp; Turn limit: ${rec.turn_limit}</div>`,
    `<div class="rpl-info">Cops: ${rec.initial_cop_positions.map(n => n + 1).join(", ")}</div>`,
    `<div style="margin-top:8px"><button onclick="forkFromCurrentTurn()" style="font-size:11px;padding:2px 8px">▶ Play from here</button></div>`,
  ];
  el.innerHTML = lines.join("");
}

function renderReplayMeta() {
  if (!_replayRecord) return;
  const el = document.getElementById("replay-meta");
  el.innerHTML =
    `<strong>Game ${_replayRecord.game_id}</strong> &nbsp;·&nbsp; ` +
    `${_replayRecord.winner === "jack" ? "Jack escapes" : "Cops win"} &nbsp;·&nbsp; ` +
    `${_replayRecord.turns_survived} turns &nbsp;·&nbsp; ` +
    `Hideout: ${_replayRecord.hideout + 1}`;
}

function renderReplayPanel(rnd) {
  const el = document.getElementById("replay-round-detail");
  const lines = [];

  // Round header
  const subLabel = _replaySubStep === 0 ? "Jack" : "Cops";
  lines.push(`<div class="rpl-section"><strong>Round ${rnd.turn + 1} · ${subLabel}</strong></div>`);

  // Jack section — always shown
  const via = rnd.jack_via.length ? ` via cop-node ${rnd.jack_via.map(n => n + 1).join(",")}` : "";
  lines.push(
    `<div class="rpl-row"><span class="rpl-jack">Jack</span>` +
    ` <strong>${rnd.jack_from + 1}</strong> → <strong>${rnd.jack_to + 1}</strong>${via}</div>`
  );
  if (rnd.jack_legal_moves.length) {
    const alts = rnd.jack_legal_moves.filter(m => m !== rnd.jack_to);
    lines.push(`<div class="rpl-info">Alternatives: ${alts.map(n => n + 1).join(", ") || "none"}</div>`);
  }

  // Cops section — only on sub-step 1
  if (_replaySubStep === 1) {
    for (const ca of rnd.cop_actions) {
      const color = window.COP_COLORS ? window.COP_COLORS[ca.cop_idx % window.COP_COLORS.length] : "#aaa";
      const dot   = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:4px"></span>`;
      const roleTag = ca.role ? `<span class="rpl-role rpl-role-${ca.role}">${ca.role}</span>` : "";
      lines.push(`<div class="rpl-cop-header">${dot}<strong>Cop ${ca.cop_idx}</strong> ${roleTag} ${ca.from_node + 1} → ${ca.to_node + 1}</div>`);

      if (ca.action === "search") {
        const hitParts  = ca.search_hits.map(n => `<span class="rpl-hit">${n + 1}</span>`).join(" ");
        const missParts = ca.search_miss_nodes.map(n => `<span class="rpl-miss">${n + 1}</span>`).join(" ");
        lines.push(
          `<div class="rpl-info">Search: ${hitParts || "—"} hit &nbsp; ${missParts || "—"} miss</div>`
        );
      } else {
        const outcome = ca.arrest_success
          ? `<span class="rpl-hit">SUCCESS</span>`
          : `<span class="rpl-miss">miss</span>`;
        const targets = (Array.isArray(ca.arrest_target) && ca.arrest_target.length)
          ? ca.arrest_target.map(n => n + 1).join(", ")
          : "?";
        lines.push(`<div class="rpl-info">Arrest nodes ${targets}: ${outcome}</div>`);
      }

      if (ca.coverage_score !== null && ca.coverage_score !== undefined) {
        const dirStr = ca.direction_score !== null && ca.direction_score !== undefined
          ? ` · prox ${ca.direction_score.toFixed(3)}`
          : "";
        lines.push(`<div class="rpl-scores">cov ${ca.coverage_score.toFixed(3)}${dirStr}</div>`);
      }
    }

    if (rnd.visited_at_after.length) {
      lines.push(`<div class="rpl-info rpl-visited">Visited: ${rnd.visited_at_after.map(([n]) => n + 1).join(", ")}</div>`);
    }
  }

  if (rnd.terminated) {
    const color = rnd.winner === "jack" ? "#81c784" : "#e57373";
    lines.push(`<div class="rpl-result" style="color:${color}"><strong>${rnd.winner === "jack" ? "Jack escapes!" : "Cops win!"}</strong></div>`);
  }

  lines.push(`<div style="margin-top:8px"><button onclick="forkFromCurrentTurn()" style="font-size:11px;padding:2px 8px">▶ Play from here</button></div>`);

  el.innerHTML = lines.join("");
}

function updateReplayNav() {
  if (!_replayRecord) return;
  const totalRounds = _replayRecord.rounds.length;
  const lastRound   = totalRounds - 1;
  const lastSub     = maxSubStep(lastRound);
  const atStart     = _replayRound === -1;
  const atEnd       = _replayRound === lastRound && _replaySubStep === lastSub;

  if (atStart) {
    document.getElementById("replay-round-counter").textContent = `Initial · 0 / ${totalRounds}`;
  } else {
    const subLabel = _replaySubStep === 0 ? "Jack" : "Cops";
    document.getElementById("replay-round-counter").textContent =
      `Round ${_replayRound + 1} / ${totalRounds} · ${subLabel}`;
  }

  document.getElementById("replay-prev-btn").disabled  = atStart;
  document.getElementById("replay-start-btn").disabled = atStart;
  document.getElementById("replay-next-btn").disabled  = atEnd;
  document.getElementById("replay-end-btn").disabled   = atEnd;
}

// Init when DOM is ready (scripts load after board.js and admin.js)
initReplay();
