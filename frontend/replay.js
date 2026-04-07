// ============================================================
// Replay viewer — communicates with /api/replays
// ============================================================

let _replayList    = [];    // metadata from GET /api/replays
let _replayRecord  = null;  // full ReplayRecord currently loaded
let _replayRound   = 0;     // 0-based index into _replayRecord.rounds
let _replaySubStep = 0;     // 0 = after Jack moved, 1 = after all cops acted
let _replayPmf     = false; // whether PMF overlay is active in replay

// ── Init ────────────────────────────────────────────────────

async function initReplay() {
  document.getElementById("replay-refresh-btn").addEventListener("click", loadReplayList);
  document.getElementById("replay-prev-btn").addEventListener("click", () => stepReplay(-1));
  document.getElementById("replay-next-btn").addEventListener("click", () => stepReplay(+1));
  document.getElementById("replay-start-btn").addEventListener("click", () => gotoStep(0, 0));
  document.getElementById("replay-end-btn").addEventListener("click", () => {
    if (_replayRecord) {
      const lastRound = _replayRecord.rounds.length - 1;
      const lastSub   = _replayRecord.rounds[lastRound].cop_actions.length > 0 ? 1 : 0;
      gotoStep(lastRound, lastSub);
    }
  });
  document.getElementById("replay-pmf-check").addEventListener("change", e => {
    _replayPmf = e.target.checked;
    renderReplayRound();
  });
}

// ── Load list ────────────────────────────────────────────────

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
    gotoStep(0, 0);
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

  if (s > maxSubStep(r)) {
    r++;
    s = 0;
  } else if (s < 0) {
    r--;
    if (r < 0) return;
    s = maxSubStep(r);
  }
  if (r >= _replayRecord.rounds.length) return;
  gotoStep(r, s);
}

function gotoStep(round, subStep) {
  if (!_replayRecord) return;
  _replayRound   = Math.max(0, Math.min(_replayRecord.rounds.length - 1, round));
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
  const rnd = _replayRecord.rounds[_replayRound];

  const replayState = {
    game_id:       _replayRecord.game_id,
    jack_pos:      rnd.jack_to,
    cop_positions: _copPositionsAtSubStep(),
    hideout:       _replayRecord.hideout,
    turn:          rnd.turn,
    turn_limit:    _replayRecord.turn_limit,
    legal_moves:   [],
    visited:       rnd.visited_after,
    search_misses: rnd.search_misses_after,
    arrest_misses: rnd.arrest_misses_after,
    jack_trace:    [],
    blocking:      _replayRecord.blocking,
    history_size:  0,
    terminated:    rnd.terminated,
    winner:        rnd.winner,
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

function renderReplayMeta() {
  if (!_replayRecord) return;
  const el = document.getElementById("replay-meta");
  el.innerHTML =
    `<strong>Game ${_replayRecord.game_id}</strong> &nbsp;·&nbsp; ` +
    `${_replayRecord.winner === "jack" ? "Jack escapes" : "Cops win"} &nbsp;·&nbsp; ` +
    `${_replayRecord.turns_survived} turns &nbsp;·&nbsp; ` +
    `Hideout: ${_replayRecord.hideout}`;
}

function renderReplayPanel(rnd) {
  const el = document.getElementById("replay-round-detail");
  const lines = [];

  // Round header
  const subLabel = _replaySubStep === 0 ? "Jack" : "Cops";
  lines.push(`<div class="rpl-section"><strong>Round ${rnd.turn + 1} · ${subLabel}</strong></div>`);

  // Jack section — always shown
  const via = rnd.jack_via.length ? ` via cop-node ${rnd.jack_via.join(",")}` : "";
  lines.push(
    `<div class="rpl-row"><span class="rpl-jack">Jack</span>` +
    ` <strong>${rnd.jack_from}</strong> → <strong>${rnd.jack_to}</strong>${via}</div>`
  );
  if (rnd.jack_legal_moves.length) {
    const alts = rnd.jack_legal_moves.filter(m => m !== rnd.jack_to);
    lines.push(`<div class="rpl-info">Alternatives: ${alts.join(", ") || "none"}</div>`);
  }

  // Cops section — only on sub-step 1
  if (_replaySubStep === 1) {
    for (const ca of rnd.cop_actions) {
      const color = window.COP_COLORS ? window.COP_COLORS[ca.cop_idx % window.COP_COLORS.length] : "#aaa";
      const dot   = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:4px"></span>`;
      const roleTag = ca.role ? `<span class="rpl-role rpl-role-${ca.role}">${ca.role}</span>` : "";
      lines.push(`<div class="rpl-cop-header">${dot}<strong>Cop ${ca.cop_idx}</strong> ${roleTag} ${ca.from_node} → ${ca.to_node}</div>`);

      if (ca.action === "search") {
        const hitParts  = ca.search_hits.map(n => `<span class="rpl-hit">${n}</span>`).join(" ");
        const missParts = ca.search_miss_nodes.map(n => `<span class="rpl-miss">${n}</span>`).join(" ");
        lines.push(
          `<div class="rpl-info">Search: ${hitParts || "—"} hit &nbsp; ${missParts || "—"} miss</div>`
        );
      } else {
        const outcome = ca.arrest_success
          ? `<span class="rpl-hit">SUCCESS</span>`
          : `<span class="rpl-miss">miss</span>`;
        lines.push(`<div class="rpl-info">Arrest node ${ca.arrest_target ?? "?"}: ${outcome}</div>`);
      }

      if (ca.coverage_score !== null && ca.coverage_score !== undefined) {
        const dirStr = ca.direction_score !== null && ca.direction_score !== undefined
          ? ` · prox ${ca.direction_score.toFixed(3)}`
          : "";
        lines.push(`<div class="rpl-scores">cov ${ca.coverage_score.toFixed(3)}${dirStr}</div>`);
      }
    }

    if (rnd.visited_after.length) {
      lines.push(`<div class="rpl-info rpl-visited">Visited: ${rnd.visited_after.join(", ")}</div>`);
    }
  }

  if (rnd.terminated) {
    const color = rnd.winner === "jack" ? "#81c784" : "#e57373";
    lines.push(`<div class="rpl-result" style="color:${color}"><strong>${rnd.winner === "jack" ? "Jack escapes!" : "Cops win!"}</strong></div>`);
  }

  el.innerHTML = lines.join("");
}

function updateReplayNav() {
  if (!_replayRecord) return;
  const totalRounds = _replayRecord.rounds.length;
  const lastRound   = totalRounds - 1;
  const lastSub     = maxSubStep(lastRound);
  const atStart     = _replayRound === 0 && _replaySubStep === 0;
  const atEnd       = _replayRound === lastRound && _replaySubStep === lastSub;

  const subLabel = _replaySubStep === 0 ? "Jack" : "Cops";
  document.getElementById("replay-round-counter").textContent =
    `Round ${_replayRound + 1} / ${totalRounds} · ${subLabel}`;

  document.getElementById("replay-prev-btn").disabled  = atStart;
  document.getElementById("replay-start-btn").disabled = atStart;
  document.getElementById("replay-next-btn").disabled  = atEnd;
  document.getElementById("replay-end-btn").disabled   = atEnd;
}

// Init when DOM is ready (scripts load after board.js and admin.js)
initReplay();
