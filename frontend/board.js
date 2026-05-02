const SVG_NS = "http://www.w3.org/2000/svg";

const COP_COLORS = ["#ffeb3b", "#00bcd4", "#8bc34a", "#ff9800", "#e91e63", "#9c27b0"];
window.COP_COLORS = COP_COLORS;

// Game element colors — keep in sync with frontend_participant/colors.css
const COLOR_JACK        = "#22c55e";
const COLOR_HIDEOUT     = "#f43f5e";
const COLOR_LEGAL       = "#f97316";
const COLOR_VISITED     = "#60a5fa";
const COLOR_COP         = "#ef4444";
const COLOR_COP_STROKE  = "#ff6659";
const COLOR_ZONE        = "rgba(168,85,247,0.45)";
const COLOR_ZONE_STROKE = "#7c3aed";

let mapData = null;
let gameId = null;
let busy = false;
let lastState = null;

// Pick mode: set by admin.js
// { type: 'jack'|'cop', nodes: Set<int>|null, label: string, cb: function(id) }
window.adminPickMode = null;

async function init() {
  mapData = await fetchMap();
  window.mapData = mapData;
  document.getElementById("new-game-btn").addEventListener("click", startNewGame);
  document.getElementById("mode-cancel-btn").addEventListener("click", cancelPickMode);
}

// --- Public interface for admin.js ---

window.refreshBoard = function(state) {
  render(state);
};

// Switch to live game mode from an externally supplied state (e.g. forked from a replay).
window.startGameFromState = function(state) {
  document.getElementById("log").innerHTML = "";
  render(state);
  logEntry(`Forked game ${state.game_id}. Jack at node ${state.jack_pos}, turn ${state.turn}.`, "event-jack");
};

// Render a board state without touching gameId / lastState / adminOnStateUpdate.
// Used by the replay viewer so live game state is never corrupted.
window.renderForReplay = function(state) {
  const svg = document.getElementById("board");
  const old = svg.getElementById("overlay-group");
  if (old) old.remove();

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "overlay-group");

  const visitedSet  = new Set(state.visited_at.map(([id]) => id));
  const copIndexMap = new Map();
  for (let ci = 0; ci < state.cop_positions.length; ci++) {
    copIndexMap.set(state.cop_positions[ci], ci);
  }

  for (const node of mapData.cop_nodes) {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", node.x - 4);
    rect.setAttribute("y", node.y - 4);
    rect.setAttribute("width", 8);
    rect.setAttribute("height", 8);
    const copIdx = copIndexMap.get(node.id);
    if (copIdx !== undefined) {
      rect.setAttribute("fill", COLOR_COP);
      rect.setAttribute("stroke", COLOR_COP_STROKE);
      rect.setAttribute("stroke-width", 1.5);
      g.appendChild(rect);
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", node.x);
      dot.setAttribute("cy", node.y);
      dot.setAttribute("r", 2.5);
      dot.setAttribute("fill", COP_COLORS[copIdx % COP_COLORS.length]);
      g.appendChild(dot);
    } else {
      rect.setAttribute("fill", "rgba(20,20,20,0.4)");
      rect.setAttribute("stroke", "none");
      g.appendChild(rect);
    }
  }

  for (const node of mapData.jack_nodes) {
    const isJack    = node.id === state.jack_pos;
    const isHideout = node.id === state.hideout;
    const isVisited = visitedSet.has(node.id);

    const activeStates = [];
    if (isJack)    activeStates.push({ stroke: COLOR_JACK });
    if (isHideout) activeStates.push({ stroke: COLOR_HIDEOUT });
    if (isVisited) activeStates.push({ stroke: COLOR_VISITED });

    if (activeStates.length === 0) continue;

    const r  = (isJack || isHideout) ? 10 : 9;
    const sw = (isJack || isHideout) ? 4 : 3;
    const n  = activeStates.length;
    const circumference = 2 * Math.PI * r;
    const segLen = circumference / n;

    const nodeGroup = document.createElementNS(SVG_NS, "g");
    for (let i = 0; i < n; i++) {
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", node.x);
      c.setAttribute("cy", node.y);
      c.setAttribute("r", r);
      c.setAttribute("fill", "none");
      c.setAttribute("stroke", activeStates[i].stroke);
      c.setAttribute("stroke-width", sw);
      if (n > 1) {
        c.setAttribute("transform", `rotate(-90, ${node.x}, ${node.y})`);
        c.setAttribute("stroke-dasharray", `${segLen} ${circumference - segLen}`);
        c.setAttribute("stroke-dashoffset", -(i * segLen));
      }
      nodeGroup.appendChild(c);
    }
    g.appendChild(nodeGroup);
  }

  renderZoneLayer(state);
  svg.appendChild(g);
  renderPmfLayer();
  updateStatus(state);
};

window.rerenderBoard = function() {
  if (lastState) render(lastState);
};

window.setPickMode = function(type, nodeIds, label, cb) {
  window.adminPickMode = {
    type,
    nodes: nodeIds ? new Set(nodeIds) : null,
    label,
    cb,
  };
  updateBanner();
  renderPickLayer();
};

window.clearPickMode = function() {
  cancelPickMode();
};

// --- Banner ---

function updateBanner() {
  const banner = document.getElementById("mode-banner");
  const text   = document.getElementById("mode-banner-text");
  if (window.adminPickMode) {
    text.textContent = window.adminPickMode.label;
    banner.classList.add("active");
  } else {
    banner.classList.remove("active");
  }
}

function cancelPickMode() {
  window.adminPickMode = null;
  updateBanner();
  renderPickLayer();
}

// --- Game flow ---

async function startNewGame() {
  if (busy) return;
  busy = true;
  const btn = document.getElementById("new-game-btn");
  btn.disabled = true;
  document.getElementById("log").innerHTML = "";

  try {
    const state = await newGame();
    gameId = state.game_id;
    render(state);
    logEntry(`Game started. Jack begins at node ${state.jack_pos}. Hideout: ${state.hideout}.`, "event-jack");
  } catch (e) {
    logEntry(`Error: ${e.message}`, "event-result");
  } finally {
    busy = false;
    btn.disabled = false;
  }
}

async function handleJackMove(destination) {
  if (busy || !gameId) return;
  busy = true;

  try {
    const state = await jackMove(gameId, destination);
    render(state);

    logEntry(`Turn ${state.turn}: Jack moved to ${state.jack_pos}.`, "event-jack");
    if (state.events) {
      for (const ev of state.events) {
        const nb = ev.jack_neighbours.join(", ") || "none";
        logEntry(`  Cop ${ev.cop} → cop-node ${ev.moved_to} (searched: ${nb})`, "event-cop");
      }
    }
    if (state.terminated) {
      logEntry(state.winner === "jack" ? "Jack escaped! Jack wins." : "Cops win!", "event-result");
    }
  } catch (e) {
    logEntry(`Error: ${e.message}`, "event-result");
  } finally {
    busy = false;
  }
}

// --- Rendering ---

function render(state) {
  lastState = state;
  gameId = state.game_id;

  const svg = document.getElementById("board");
  const old = svg.getElementById("overlay-group");
  if (old) old.remove();

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "overlay-group");

  const legalSet   = new Set(state.legal_moves);
  const visitedSet = new Set(state.visited_at.map(([id]) => id));
  const copIndexMap = new Map();
  for (let ci = 0; ci < state.cop_positions.length; ci++) {
    copIndexMap.set(state.cop_positions[ci], ci);
  }

  // Cop nodes (drawn first so jack rings appear on top)
  for (const node of mapData.cop_nodes) {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", node.x - 4);
    rect.setAttribute("y", node.y - 4);
    rect.setAttribute("width", 8);
    rect.setAttribute("height", 8);
    const copIdx = copIndexMap.get(node.id);
    if (copIdx !== undefined) {
      rect.setAttribute("fill", COLOR_COP);
      rect.setAttribute("stroke", COLOR_COP_STROKE);
      rect.setAttribute("stroke-width", 1.5);
      g.appendChild(rect);
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", node.x);
      dot.setAttribute("cy", node.y);
      dot.setAttribute("r", 2.5);
      dot.setAttribute("fill", COP_COLORS[copIdx % COP_COLORS.length]);
      g.appendChild(dot);
    } else {
      rect.setAttribute("fill", "rgba(20,20,20,0.4)");
      rect.setAttribute("stroke", "none");
      g.appendChild(rect);
    }
  }

  // Jack nodes — segmented rings for active states only
  for (const node of mapData.jack_nodes) {
    const isJack    = node.id === state.jack_pos;
    const isHideout = node.id === state.hideout;
    const isLegal   = legalSet.has(node.id) && !state.terminated;
    const isVisited = visitedSet.has(node.id);

    const activeStates = [];
    if (isJack)    activeStates.push({ stroke: COLOR_JACK });
    if (isHideout) activeStates.push({ stroke: COLOR_HIDEOUT });
    if (isLegal)   activeStates.push({ stroke: COLOR_LEGAL });
    if (isVisited) activeStates.push({ stroke: COLOR_VISITED });

    if (activeStates.length === 0) continue;

    const r  = (isJack || isHideout) ? 10 : 9;
    const sw = (isJack || isHideout) ? 4 : 3;
    const n  = activeStates.length;
    const circumference = 2 * Math.PI * r;
    const segLen = circumference / n;

    const nodeGroup = document.createElementNS(SVG_NS, "g");
    const segCircles = [];

    for (let i = 0; i < n; i++) {
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", node.x);
      c.setAttribute("cy", node.y);
      c.setAttribute("r", r);
      c.setAttribute("fill", "none");
      c.setAttribute("stroke", activeStates[i].stroke);
      c.setAttribute("stroke-width", sw);
      if (n > 1) {
        c.setAttribute("transform", `rotate(-90, ${node.x}, ${node.y})`);
        c.setAttribute("stroke-dasharray", `${segLen} ${circumference - segLen}`);
        c.setAttribute("stroke-dashoffset", -(i * segLen));
      }
      segCircles.push(c);
      nodeGroup.appendChild(c);
    }

    if (isLegal) {
      const hit = document.createElementNS(SVG_NS, "circle");
      hit.setAttribute("cx", node.x);
      hit.setAttribute("cy", node.y);
      hit.setAttribute("r", r);
      hit.setAttribute("fill", "transparent");
      hit.setAttribute("stroke", "none");
      hit.style.cursor = "pointer";
      hit.addEventListener("click", () => {
        if (window.adminPickMode) return; // admin pick layer handles it
        handleJackMove(node.id);
      });
      hit.addEventListener("mouseenter", () => segCircles.forEach(c => c.setAttribute("stroke-width", sw + 2)));
      hit.addEventListener("mouseleave", () => segCircles.forEach(c => c.setAttribute("stroke-width", sw)));
      nodeGroup.appendChild(hit);
    }

    g.appendChild(nodeGroup);
  }

  renderZoneLayer(state);
  svg.appendChild(g);
  renderPmfLayer();
  renderPmfZoneLayer();
  renderPickLayer();
  updateStatus(state);

  window.adminOnStateUpdate?.(state);
}

// --- Zone layer ---

function renderZoneLayer(state) {
  const svg = document.getElementById("board");
  const old = svg.getElementById("zone-layer");
  if (old) old.remove();
  if (!state.hideout_zone || state.hideout_zone.length === 0 || !mapData) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "zone-layer");
  g.setAttribute("pointer-events", "none");

  for (const nodeId of state.hideout_zone) {
    const node = mapData.jack_nodes[nodeId - 1];
    if (!node) continue;
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", node.x);
    c.setAttribute("cy", node.y);
    c.setAttribute("r", 12);
    c.setAttribute("fill", COLOR_ZONE);
    c.setAttribute("stroke", COLOR_ZONE_STROKE);
    c.setAttribute("stroke-width", "2");
    g.appendChild(c);
  }

  const overlayGroup = svg.getElementById("overlay-group");
  if (overlayGroup) {
    svg.insertBefore(g, overlayGroup);
  } else {
    const mapImage = svg.querySelector("image");
    svg.insertBefore(g, mapImage ? mapImage.nextSibling : svg.firstChild);
  }
}

// --- PMF overlay ---

let _currentPmf = null;  // dict { nodeId(string): probability } or null
let _pmfLayerEnabled = false;
let _pmfZoneEnabled  = false;

window.setPmfData = function(pmf) {
  _currentPmf = pmf;
  renderPmfLayer();
  renderPmfZoneLayer();
};

window.clearPmfData = function() {
  _currentPmf = null;
  renderPmfLayer();
  renderPmfZoneLayer();
};

window.setPmfLayerEnabled = function(enabled) {
  _pmfLayerEnabled = enabled;
  renderPmfLayer();
};

window.setPmfZoneEnabled = function(enabled) {
  _pmfZoneEnabled = enabled;
  renderPmfZoneLayer();
};

function renderPmfLayer() {
  const svg = document.getElementById("board");
  const old = svg.getElementById("pmf-layer");
  if (old) old.remove();
  if (!_pmfLayerEnabled || !_currentPmf || !mapData) return;

  const entries = Object.entries(_currentPmf);
  if (entries.length === 0) return;

  const maxProb = Math.max(...entries.map(([, p]) => p));
  if (maxProb === 0) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "pmf-layer");

  for (const [nodeIdStr, prob] of entries) {
    const nodeId = parseInt(nodeIdStr);
    const node = mapData.jack_nodes[nodeId - 1];
    if (!node) continue;

    // Scale so the highest-probability node is fully opaque;
    // use sqrt to make low-probability nodes more visible.
    const alpha = Math.sqrt(prob / maxProb) * 0.75;

    const circle = document.createElementNS(SVG_NS, "circle");
    circle.setAttribute("cx", node.x);
    circle.setAttribute("cy", node.y);
    circle.setAttribute("r", 7);
    circle.setAttribute("fill", `rgba(220, 50, 50, ${alpha.toFixed(3)})`);
    circle.setAttribute("pointer-events", "none");
    g.appendChild(circle);
  }

  // Insert before overlay-group so PMF sits under game markers
  const overlay = svg.getElementById("overlay-group");
  svg.insertBefore(g, overlay);
}

// --- PMF–zone intersection overlay ---

function renderPmfZoneLayer() {
  const svg = document.getElementById("board");
  const old = svg.getElementById("pmf-zone-layer");
  if (old) old.remove();
  if (!_pmfZoneEnabled || !_currentPmf || !lastState?.hideout_zone || !mapData) return;

  const zoneSet = new Set(lastState.hideout_zone);
  const entries = Object.entries(_currentPmf).filter(([id, p]) => p > 0 && zoneSet.has(parseInt(id)));
  if (entries.length === 0) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "pmf-zone-layer");
  g.setAttribute("pointer-events", "none");

  for (const [nodeIdStr] of entries) {
    const nodeId = parseInt(nodeIdStr);
    const node = mapData.jack_nodes[nodeId - 1];
    if (!node) continue;
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", node.x);
    c.setAttribute("cy", node.y);
    c.setAttribute("r", 12);
    c.setAttribute("fill", "rgba(0,255,0,0.45)");
    c.setAttribute("stroke", "#00cc00");
    c.setAttribute("stroke-width", "2");
    g.appendChild(c);
  }

  // Insert after zone-layer so lime green sits on top of purple for intersection nodes
  const zoneLayer = svg.getElementById("zone-layer");
  const overlayGroup = svg.getElementById("overlay-group");
  if (zoneLayer) {
    svg.insertBefore(g, zoneLayer.nextSibling);
  } else if (overlayGroup) {
    svg.insertBefore(g, overlayGroup);
  } else {
    const mapImage = svg.querySelector("image");
    svg.insertBefore(g, mapImage ? mapImage.nextSibling : svg.firstChild);
  }
}

// --- Pick layer (admin mode) ---

function renderPickLayer() {
  const svg = document.getElementById("board");
  const old = svg.getElementById("pick-layer");
  if (old) old.remove();

  const pm = window.adminPickMode;
  if (!pm || !mapData) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "pick-layer");

  const nodes = pm.type === "jack" ? mapData.jack_nodes : mapData.cop_nodes;

  for (const node of nodes) {
    if (pm.nodes && !pm.nodes.has(node.id)) continue;

    if (pm.type === "jack") {
      // Dashed ring indicator
      const ring = document.createElementNS(SVG_NS, "circle");
      ring.setAttribute("cx", node.x);
      ring.setAttribute("cy", node.y);
      ring.setAttribute("r", 9);
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", "#e040fb");
      ring.setAttribute("stroke-width", 3);
      ring.setAttribute("stroke-dasharray", "5 3");
      ring.setAttribute("pointer-events", "none");
      const animJ = document.createElementNS(SVG_NS, "animate");
      animJ.setAttribute("attributeName", "stroke-dashoffset");
      animJ.setAttribute("from", "0");
      animJ.setAttribute("to", "8");
      animJ.setAttribute("dur", "0.4s");
      animJ.setAttribute("repeatCount", "indefinite");
      ring.appendChild(animJ);
      g.appendChild(ring);

      // Transparent hit circle
      const hit = document.createElementNS(SVG_NS, "circle");
      hit.setAttribute("cx", node.x);
      hit.setAttribute("cy", node.y);
      hit.setAttribute("r", 11);
      hit.setAttribute("fill", "transparent");
      hit.setAttribute("stroke", "none");
      hit.style.cursor = "crosshair";
      hit.addEventListener("click", () => pickNode(node.id));
      g.appendChild(hit);
    } else {
      // Cop node: dashed rect outline
      const ring = document.createElementNS(SVG_NS, "rect");
      ring.setAttribute("x", node.x - 6);
      ring.setAttribute("y", node.y - 6);
      ring.setAttribute("width", 12);
      ring.setAttribute("height", 12);
      ring.setAttribute("fill", "none");
      ring.setAttribute("stroke", "#e040fb");
      ring.setAttribute("stroke-width", 3);
      ring.setAttribute("stroke-dasharray", "4 2");
      ring.setAttribute("pointer-events", "none");
      const animC = document.createElementNS(SVG_NS, "animate");
      animC.setAttribute("attributeName", "stroke-dashoffset");
      animC.setAttribute("from", "0");
      animC.setAttribute("to", "6");
      animC.setAttribute("dur", "0.4s");
      animC.setAttribute("repeatCount", "indefinite");
      ring.appendChild(animC);
      g.appendChild(ring);

      const hit = document.createElementNS(SVG_NS, "rect");
      hit.setAttribute("x", node.x - 8);
      hit.setAttribute("y", node.y - 8);
      hit.setAttribute("width", 16);
      hit.setAttribute("height", 16);
      hit.setAttribute("fill", "transparent");
      hit.setAttribute("stroke", "none");
      hit.style.cursor = "crosshair";
      hit.addEventListener("click", () => pickNode(node.id));
      g.appendChild(hit);
    }
  }

  svg.appendChild(g);
}

function pickNode(id) {
  const cb = window.adminPickMode?.cb;
  window.adminPickMode = null;
  updateBanner();
  renderPickLayer();
  if (cb) cb(id);
}

// --- UI helpers ---

function updateStatus(state) {
  const el = document.getElementById("status");
  if (state.terminated) {
    const color = state.winner === "jack" ? "#81c784" : "#e57373";
    el.innerHTML = `<strong style="color:${color}">${state.winner === "jack" ? "Jack escaped!" : "Cops win!"}</strong>`;
  } else {
    el.innerHTML =
      `Turn <strong>${state.turn + 1}</strong> / ${state.turn_limit}<br>` +
      `Jack: node <strong>${state.jack_pos}</strong><br>` +
      `Hideout: node <strong>${state.hideout}</strong><br>` +
      `Legal moves: <strong>${state.legal_moves.length}</strong><br>` +
      `Cop-visited: <strong>${state.visited_at.length}</strong>`;
  }
}

function logEntry(text, cssClass) {
  const log = document.getElementById("log");
  const p = document.createElement("p");
  if (cssClass) p.className = cssClass;
  p.textContent = text;
  log.prepend(p);
}

init();
