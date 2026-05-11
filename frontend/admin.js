// ============================================================
// Admin panel — communicates with /api/admin/*
// ============================================================

let adminGameId  = null;
let adminState   = null;
let copRowsBuilt = 0;  // number of cop rows currently in DOM
let pmfEnabled     = false;
let pmfZoneEnabled = false;

// pendingActions[i] = { destination: int|null, search: bool, arrest_target: int|null } | null
const pendingActions = [];

// ── External interface called by board.js ────────────────────
window.adminOnStateUpdate = function(state) {
  adminState  = state;
  adminGameId = state.game_id;
  renderAdmin();
  if (pmfEnabled || pmfZoneEnabled) refreshPmf();
};

async function refreshPmf() {
  if (!adminGameId) return;
  try {
    const r = await fetch(`/api/admin/${adminGameId}/pmf`);
    if (!r.ok) return;
    const pmf = await r.json();
    window.setPmfData(pmf);
  } catch (e) {
    console.error("PMF fetch failed:", e.message);
  }
}

// ── API helpers ──────────────────────────────────────────────

async function adminPost(endpoint, body = null) {
  if (!adminGameId) throw new Error("No active game");
  const opts = { method: "POST" };
  if (body !== null) {
    opts.headers = { "Content-Type": "application/json" };
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(`/api/admin/${adminGameId}/${endpoint}`, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

async function adminAction(endpoint, body = null) {
  try {
    const state = await adminPost(endpoint, body);
    window.refreshBoard(state);
  } catch (e) {
    console.error(`Admin [${endpoint}]:`, e.message);
  }
}

// ── Cop pending action helpers ───────────────────────────────

function getPending(i) {
  return pendingActions[i] ?? null;
}

function setPending(i, updates) {
  if (!pendingActions[i]) pendingActions[i] = { destination: null, search: true, arrest_target: null };
  Object.assign(pendingActions[i], updates);
  renderCopPending(i);
}

function clearPending(i) {
  pendingActions[i] = null;
  renderCopPending(i);
}

function renderCopPending(i) {
  const el = document.getElementById(`adm-cop-pending-${i}`);
  if (!el) return;
  const pa = getPending(i);
  if (!pa) {
    el.textContent = "No action (will stay + search)";
    return;
  }
  const dest = pa.destination !== null ? `→ node ${pa.destination}` : "(stay)";
  const action = pa.search ? "Search" : `Arrest node ${pa.arrest_target ?? "?"}`;
  el.textContent = `${dest}  ·  ${action}`;

  // Show/hide arrest controls
  const arrestEl = document.getElementById(`adm-cop-arrest-target-row-${i}`);
  if (arrestEl) arrestEl.style.display = !pa.search ? "flex" : "none";
}

// ── Cop rows ─────────────────────────────────────────────────

function buildCopRows(numCops) {
  const container = document.getElementById("adm-cops-list");
  if (!container) return;
  container.innerHTML = "";
  copRowsBuilt = numCops;

  for (let i = 0; i < numCops; i++) {
    pendingActions[i] = null;
    const div = document.createElement("div");
    div.className = "adm-cop-row";
    div.innerHTML = `
      <div class="adm-row">
        <span class="cop-color-dot" style="background:${window.COP_COLORS[i % window.COP_COLORS.length]}"></span><strong>Cop ${i}</strong> @ node <span class="adm-val" id="adm-cop-pos-${i}">—</span>
        <button class="adm-btn" id="adm-cop-teleport-${i}">Teleport</button>
      </div>
      <div class="adm-cop-sees" id="adm-cop-sees-${i}">Sees: —</div>
      <div class="adm-row" style="margin-top:3px">
        <button class="adm-btn" id="adm-cop-stay-${i}">Stay</button>
        <button class="adm-btn" id="adm-cop-pick-dest-${i}">Pick dest</button>
        <label style="font-size:11px">
          <input type="radio" name="adm-cop-sr-${i}" id="adm-cop-search-${i}" value="search" checked> S
        </label>
        <label style="font-size:11px">
          <input type="radio" name="adm-cop-sr-${i}" id="adm-cop-arrest-${i}" value="arrest"> A
        </label>
      </div>
      <div class="adm-row" id="adm-cop-arrest-target-row-${i}" style="display:none">
        <button class="adm-btn" id="adm-cop-pick-arrest-${i}">Pick arrest target</button>
        <span class="adm-info" id="adm-cop-arrest-node-${i}">—</span>
      </div>
      <div class="adm-cop-pending" id="adm-cop-pending-${i}">No action (will stay + search)</div>
      <button class="adm-btn" id="adm-cop-clear-${i}" style="margin-top:3px;font-size:10px">Clear</button>
    `;
    container.appendChild(div);

    // Attach listeners (closure over i)
    setupCopRow(i);
  }
}

function setupCopRow(i) {
  document.getElementById(`adm-cop-teleport-${i}`).addEventListener("click", () => {
    window.setPickMode("cop", null, `Teleport Cop ${i} — click a cop node`, async (nodeId) => {
      await adminAction("teleport-cop", { cop: i, node: nodeId });
    });
  });

  document.getElementById(`adm-cop-stay-${i}`).addEventListener("click", () => {
    setPending(i, { destination: null });
  });

  document.getElementById(`adm-cop-pick-dest-${i}`).addEventListener("click", () => {
    const startPos = adminState?.cop_positions[i];
    let reachable = null;
    if (startPos != null && window.mapData) {
      // Build jack_id → [cop_ids] reverse index so we can hop cop→jack→cop in one step
      const jackToCops = new Map();
      for (const cn of window.mapData.cop_nodes) {
        for (const jid of cn.jack_neighbours) {
          if (!jackToCops.has(jid)) jackToCops.set(jid, []);
          jackToCops.get(jid).push(cn.id);
        }
      }

      reachable = new Set([startPos]);
      let frontier = [startPos];
      for (let step = 0; step < 2; step++) {
        const next = [];
        for (const nodeId of frontier) {
          const cn = window.mapData.cop_nodes[nodeId];
          if (!cn) continue;
          // Direct cop-to-cop edges
          for (const nb of cn.edges) {
            if (!reachable.has(nb)) { reachable.add(nb); next.push(nb); }
          }
          // One step through a shared jack neighbour (cop → jack circle → cop)
          for (const jid of cn.jack_neighbours) {
            for (const nb of (jackToCops.get(jid) ?? [])) {
              if (!reachable.has(nb)) { reachable.add(nb); next.push(nb); }
            }
          }
        }
        frontier = next;
      }
    }
    window.setPickMode("cop", reachable, `Cop ${i} destination — click a cop node`, (nodeId) => {
      setPending(i, { destination: nodeId });
    });
  });

  document.getElementById(`adm-cop-search-${i}`).addEventListener("change", () => {
    setPending(i, { search: true, arrest_target: null });
  });

  document.getElementById(`adm-cop-arrest-${i}`).addEventListener("change", () => {
    setPending(i, { search: false });
  });

  document.getElementById(`adm-cop-pick-arrest-${i}`).addEventListener("click", () => {
    // Highlight jack nodes adjacent to this cop's pending destination (or current pos)
    const dest = getPending(i)?.destination ?? adminState?.cop_positions[i];
    let pickableJack = null;
    if (dest !== null && dest !== undefined) {
      const copNode = window.mapData?.cop_nodes[dest];
      if (copNode?.jack_neighbours) {
        pickableJack = new Set(copNode.jack_neighbours);
      }
    }
    window.setPickMode("jack", pickableJack, `Cop ${i} arrest target — click a jack node`, (nodeId) => {
      setPending(i, { arrest_target: nodeId });
      document.getElementById(`adm-cop-arrest-node-${i}`).textContent = `Node ${nodeId}`;
    });
  });

  document.getElementById(`adm-cop-clear-${i}`).addEventListener("click", () => {
    clearPending(i);
    // Reset radio to search
    const searchRadio = document.getElementById(`adm-cop-search-${i}`);
    if (searchRadio) searchRadio.checked = true;
  });
}

function updateCopRows(state) {
  for (let i = 0; i < state.cop_positions.length; i++) {
    const posEl  = document.getElementById(`adm-cop-pos-${i}`);
    const seesEl = document.getElementById(`adm-cop-sees-${i}`);
    if (posEl) posEl.textContent = state.cop_positions[i];
    if (seesEl) {
      const copNode = window.mapData?.cop_nodes[state.cop_positions[i]];
      const nb = copNode?.jack_neighbours ?? [];
      seesEl.textContent = nb.length ? `Sees jack: ${nb.join(", ")}` : "Sees: none";
    }
  }
}

// ── Render ───────────────────────────────────────────────────

function renderAdmin() {
  if (!adminState) return;
  const s = adminState;

  // Jack section
  setText("adm-jack-pos", s.jack_pos);
  setText("adm-hideout",  s.hideout);

  // Cops section
  if (s.cop_positions.length !== copRowsBuilt) {
    buildCopRows(s.cop_positions.length);
  }
  updateCopRows(s);
  setDisabled("adm-execute-all", false);

  // Knowledge section
  setText("adm-visited-count",      s.visited_at.length);
  setText("adm-search-misses-count", s.search_misses.length);
  setText("adm-arrest-misses-count", s.arrest_misses.length);

  // Game section
  setInputIfNotFocused("adm-turn",       s.turn);
  setInputIfNotFocused("adm-turn-limit", s.turn_limit);
  setChecked("adm-blocking", s.blocking);
  setDisabled("adm-undo",           s.history_size === 0);
  setDisabled("adm-new-from-state", false);
  setDisabled("adm-snapshot",       false);
  setDisabled("adm-restore",        !localStorage.getItem(`snapshot_${s.game_id}`));
}

// ── DOM helpers ──────────────────────────────────────────────

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val ?? "—";
}

function setDisabled(id, disabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = disabled;
}

function setChecked(id, val) {
  const el = document.getElementById(id);
  if (el) el.checked = !!val;
}

function setInputIfNotFocused(id, val) {
  const el = document.getElementById(id);
  if (el && document.activeElement !== el) el.value = val ?? "";
}

// ── Static event listeners ───────────────────────────────────

function initAdmin() {
  // Jack
  document.getElementById("adm-teleport-jack").addEventListener("click", () => {
    window.setPickMode("jack", null, "Teleport Jack — click a jack node", async (nodeId) => {
      await adminAction("teleport-jack", { node: nodeId });
    });
  });

  // Execute All Cops
  document.getElementById("adm-execute-all").addEventListener("click", async () => {
    if (!adminState) return;
    const actions = [];
    for (let i = 0; i < adminState.cop_positions.length; i++) {
      const pa = getPending(i);
      actions.push({
        cop: i,
        destination: pa?.destination ?? null,
        search: pa?.search ?? true,
        arrest_target: pa?.arrest_target ?? null,
      });
    }
    await adminAction("cop-actions", { actions });
    // Clear all pending actions after execution
    for (let i = 0; i < adminState.cop_positions.length; i++) {
      clearPending(i);
      const searchRadio = document.getElementById(`adm-cop-search-${i}`);
      if (searchRadio) searchRadio.checked = true;
    }
  });

  // Knowledge
  document.getElementById("adm-inject-visited").addEventListener("click", () => {
    if (!adminState?.jack_trace?.length) return;
    const alreadyInjected = new Set(adminState.visited_at.map(([id]) => id));
    const injectable = new Set(adminState.jack_trace.filter(id => !alreadyInjected.has(id)));
    if (!injectable.size) return;
    window.setPickMode("jack", injectable, "Inject visited — click a node on Jack's path", async (nodeId) => {
      await adminAction("inject-visited", { node: nodeId });
    });
  });

  document.getElementById("adm-remove-visited").addEventListener("click", () => {
    if (!adminState?.visited_at.length) return;
    window.setPickMode("jack", new Set(adminState.visited_at.map(([id]) => id)), "Remove visited — click a blue node", async (nodeId) => {
      await adminAction("remove-visited", { node: nodeId });
    });
  });

  document.getElementById("adm-clear-knowledge").addEventListener("click", async () => {
    await adminAction("clear-knowledge");
  });

  document.getElementById("adm-edit-raw").addEventListener("click", () => {
    if (!adminState) return;
    document.getElementById("adm-raw-knowledge").value = JSON.stringify({
      jack_start:    adminState.jack_pos,
      visited_at:    adminState.visited_at,
      search_misses: adminState.search_misses,
      arrest_misses: adminState.arrest_misses,
    }, null, 2);
    document.getElementById("adm-raw-editor").style.display = "flex";
  });

  document.getElementById("adm-apply-raw").addEventListener("click", async () => {
    try {
      const body = JSON.parse(document.getElementById("adm-raw-knowledge").value);
      await adminAction("set-knowledge", body);
      document.getElementById("adm-raw-editor").style.display = "none";
    } catch (e) {
      alert("Invalid JSON: " + e.message);
    }
  });

  document.getElementById("adm-cancel-raw").addEventListener("click", () => {
    document.getElementById("adm-raw-editor").style.display = "none";
  });

  // Game controls
  document.getElementById("adm-set-turn").addEventListener("click", async () => {
    const turn = parseInt(document.getElementById("adm-turn").value);
    if (!isNaN(turn)) await adminAction("set-turn", { turn });
  });

  document.getElementById("adm-set-turn-limit").addEventListener("click", async () => {
    const tl = parseInt(document.getElementById("adm-turn-limit").value);
    if (!isNaN(tl)) await adminAction("set-turn-limit", { turn_limit: tl });
  });

  document.getElementById("adm-blocking").addEventListener("change", async (e) => {
    await adminAction("set-blocking", { blocking: e.target.checked });
  });

  document.getElementById("adm-undo").addEventListener("click", async () => {
    await adminAction("undo");
  });

  document.getElementById("adm-new-from-state").addEventListener("click", async () => {
    if (!adminState) return;
    const sameHideout = document.getElementById("adm-same-hideout").checked;
    try {
      const newState = await adminPost("new-from-state", { same_hideout: sameHideout });
      window.refreshBoard(newState);
    } catch (e) {
      console.error("new-from-state failed:", e.message);
    }
  });

  document.getElementById("adm-snapshot").addEventListener("click", () => {
    if (!adminState) return;
    localStorage.setItem(`snapshot_${adminState.game_id}`, JSON.stringify(adminState));
    setDisabled("adm-restore", false);
  });

  document.getElementById("adm-restore").addEventListener("click", async () => {
    if (!adminState) return;
    const snap = localStorage.getItem(`snapshot_${adminState.game_id}`);
    if (!snap) return;
    try {
      const saved = JSON.parse(snap);
      // Restore positions and knowledge from snapshot
      await adminAction("teleport-jack", { node: saved.jack_pos });
      for (let i = 0; i < saved.cop_positions.length; i++) {
        await adminAction("teleport-cop", { cop: i, node: saved.cop_positions[i] });
      }
      await adminAction("set-knowledge", {
        jack_start:    saved.jack_pos,
        visited_at:    saved.visited_at,
        search_misses: saved.search_misses,
        arrest_misses: saved.arrest_misses,
      });
      await adminAction("set-turn", { turn: saved.turn });
    } catch (e) {
      console.error("Restore failed:", e.message);
    }
  });

  document.getElementById("adm-show-pmf").addEventListener("change", async (e) => {
    pmfEnabled = e.target.checked;
    window.setPmfLayerEnabled(pmfEnabled);
    if (pmfEnabled) {
      await refreshPmf();
    } else if (!pmfZoneEnabled) {
      window.clearPmfData();
    }
  });

  document.getElementById("adm-show-pmf-zone").addEventListener("change", async (e) => {
    pmfZoneEnabled = e.target.checked;
    window.setPmfZoneEnabled(pmfZoneEnabled);
    if (pmfZoneEnabled) {
      await refreshPmf();
    } else if (!pmfEnabled) {
      window.clearPmfData();
    }
  });
}

// cancelPickMode is defined in board.js and wired to the banner Cancel button.

// Init on DOMContentLoaded (scripts load in order so board.js already ran)
initAdmin();
