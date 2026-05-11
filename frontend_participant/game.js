const SVG_NS = "http://www.w3.org/2000/svg";
const COP_COLORS = ["#ffeb3b", "#00bcd4", "#8bc34a", "#ff9800", "#e91e63", "#9c27b0"];

// Game element colors — keep in sync with colors.css
const COLOR_JACK        = "#22c55e";
const COLOR_HIDEOUT     = "#f43f5e";
const COLOR_LEGAL       = "#f97316";
const COLOR_VISITED     = "#60a5fa";
const COLOR_COP         = "#ef4444";
const COLOR_COP_STROKE  = "#ff6659";
const COLOR_ZONE        = "rgba(168,85,247,0.45)";
const COLOR_ZONE_STROKE = "#7c3aed";

let mapData      = null;
let gameId       = null;
let busy         = false;
let lastState    = null;
let course       = [];
let courseIndex  = 0;

// Precomputed lookups (built after mapData loads)
let copNodeById  = null;   // Map<id, {id,x,y,edges,jack_neighbours}>
let jackNodeById = null;   // Map<id, {id,x,y,edge_routes,...}>
let jackToCops   = null;   // Map<jackNodeId, copNodeId[]> — shared jack neighbour index

// Cop positions during animation (updated per-event before render)
let animCopPositions = new Map();  // cop_idx → cop node id

// Static cop group elements (rect+dot) drawn by render(); removed per-cop during animation
let copRectElems = new Map();  // cop_idx → SVG <g> element

// Paths taken this round (for trail drawing)
let roundCopPaths = new Map();  // cop_idx → [cop node ids]

// ── Init ──────────────────────────────────────────────────

async function init() {
  const storedCourse = sessionStorage.getItem("course");
  const storedIndex  = sessionStorage.getItem("course_index");
  const storedGame   = sessionStorage.getItem("game_id");

  if (!storedCourse || !storedGame) {
    window.location.href = "index.html";
    return;
  }

  course      = JSON.parse(storedCourse);
  courseIndex = parseInt(storedIndex || "0", 10);
  gameId      = storedGame;

  renderProgress();

  mapData = await fetchMap(course[courseIndex]?.name);
  buildLookups();

  const state = await fetchGame(gameId);
  render(state);
}

// ── Lookup table construction ─────────────────────────────

function buildLookups() {
  copNodeById  = new Map(mapData.cop_nodes.map(n => [n.id, n]));
  jackNodeById = new Map(mapData.jack_nodes.map(n => [n.id, n]));

  // Mirror the engine's reachable_cop_nodes mobility model:
  // two cop nodes are one step apart if they share a jack neighbour,
  // even with no direct cop-to-cop edge between them.
  jackToCops = new Map();
  for (const cn of mapData.cop_nodes) {
    for (const jnId of cn.jack_neighbours) {
      if (!jackToCops.has(jnId)) jackToCops.set(jnId, []);
      jackToCops.get(jnId).push(cn.id);
    }
  }
}

// All cop nodes reachable from copId in exactly 1 step, using the same mobility
// model as the engine: direct cop edges PLUS jack-mediated adjacency (two cop
// nodes are adjacent if they share a jack neighbour).
function copOneStepNeighbors(copId) {
  const cn = copNodeById.get(copId);
  if (!cn) return new Set();
  const result = new Set(cn.edges);
  for (const jnId of cn.jack_neighbours) {
    for (const nbId of (jackToCops.get(jnId) || [])) {
      if (nbId !== copId) result.add(nbId);
    }
  }
  return result;
}

// BFS for a cop path from fromId to toId (max 2 hops, engine mobility model).
// Returns [from] / [from, to] / [from, via, to].
function findCopPath(fromId, toId) {
  if (fromId === toId) return [fromId];

  const oneHop = copOneStepNeighbors(fromId);
  if (oneHop.has(toId)) return [fromId, toId];

  for (const nb1Id of oneHop) {
    if (copOneStepNeighbors(nb1Id).has(toId)) return [fromId, nb1Id, toId];
  }

  return [fromId, toId]; // fallback: destination unreachable in ≤2 hops
}

// ── Progress dots ─────────────────────────────────────────

function renderProgress() {
  const entry = course[courseIndex];
  document.getElementById("map-title").textContent = entry?.display_name || "Whitechapel";

  const container = document.getElementById("map-progress");
  container.querySelectorAll(".progress-dot").forEach(d => d.remove());

  for (let i = 0; i < course.length; i++) {
    const dot = document.createElement("span");
    const cls = i < courseIndex ? " done" : i === courseIndex ? " active" : "";
    dot.className = "progress-dot" + cls;
    container.appendChild(dot);
  }
}

// ── Full board render ─────────────────────────────────────

function render(state) {
  lastState = state;
  gameId = state.game_id;
  copRectElems.clear();

  const svg = document.getElementById("board");
  const old = svg.getElementById("overlay-group");
  if (old) old.remove();

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "overlay-group");

  const legalSet   = new Set(state.legal_moves);
  const visitedSet = new Set(state.visited_at.map(v => v[0]));
  const copMap     = new Map();
  state.cop_positions.forEach((pos, i) => copMap.set(pos, i));

  renderCopNodes(g, copMap);
  renderJackNodes(g, state, legalSet, visitedSet);

  renderZoneLayer(state);
  svg.appendChild(g);
  updateSidebar(state);
}

// ── Cop node squares ──────────────────────────────────────

function renderCopNodes(g, copMap) {
  for (const node of mapData.cop_nodes) {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", node.x - 4);
    rect.setAttribute("y", node.y - 4);
    rect.setAttribute("width", 8);
    rect.setAttribute("height", 8);

    const copIdx = copMap.get(node.id);
    if (copIdx !== undefined) {
      rect.setAttribute("fill", COLOR_COP);
      rect.setAttribute("stroke", COLOR_COP_STROKE);
      rect.setAttribute("stroke-width", 1.5);
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", node.x);
      dot.setAttribute("cy", node.y);
      dot.setAttribute("r", 2.5);
      dot.setAttribute("fill", COP_COLORS[copIdx % COP_COLORS.length]);
      const copGroup = document.createElementNS(SVG_NS, "g");
      copGroup.appendChild(rect);
      copGroup.appendChild(dot);
      g.appendChild(copGroup);
      copRectElems.set(copIdx, copGroup);
    } else {
      rect.setAttribute("fill", "rgba(20,20,20,0.35)");
      rect.setAttribute("stroke", "none");
      g.appendChild(rect);
    }
  }
}

// ── Jack node rings ───────────────────────────────────────

function renderJackNodes(g, state, legalSet, visitedSet) {
  for (const node of mapData.jack_nodes) {
    const isJack    = node.id === state.jack_pos;
    const isHideout = node.id === state.hideout;
    const isLegal   = legalSet.has(node.id) && !state.terminated;
    const isVisited = visitedSet.has(node.id);

    const segments = [];
    if (isJack)    segments.push(COLOR_JACK);
    if (isHideout) segments.push(COLOR_HIDEOUT);
    if (isLegal)   segments.push(COLOR_LEGAL);
    if (isVisited) segments.push(COLOR_VISITED);

    if (segments.length === 0) continue;

    const r  = (isJack || isHideout) ? 10 : 9;
    const sw = (isJack || isHideout) ? 4  : 3;
    const n  = segments.length;
    const circumference = 2 * Math.PI * r;
    const segLen = circumference / n;

    const nodeGroup  = document.createElementNS(SVG_NS, "g");
    if (isJack) nodeGroup.setAttribute("id", "jack-ring");
    const segCircles = [];

    for (let i = 0; i < n; i++) {
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", node.x);
      c.setAttribute("cy", node.y);
      c.setAttribute("r", r);
      c.setAttribute("fill", "none");
      c.setAttribute("stroke", segments[i]);
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
      hit.setAttribute("r", r + 2);
      hit.setAttribute("fill", "transparent");
      hit.setAttribute("stroke", "none");
      hit.style.cursor = "pointer";
      hit.addEventListener("click", () => { if (!busy) handleJackMove(node.id); });
      hit.addEventListener("mouseenter", () => segCircles.forEach(c => c.setAttribute("stroke-width", sw + 2)));
      hit.addEventListener("mouseleave", () => segCircles.forEach(c => c.setAttribute("stroke-width", sw)));
      nodeGroup.appendChild(hit);
    }

    g.appendChild(nodeGroup);
  }
}

// ── Zone layer ────────────────────────────────────────────

function renderZoneLayer(state) {
  const svg = document.getElementById("board");
  const old = svg.getElementById("zone-layer");
  if (old) old.remove();
  if (!state.hideout_zone?.length || !mapData) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "zone-layer");
  g.setAttribute("pointer-events", "none");

  for (const nodeId of state.hideout_zone) {
    const node = jackNodeById ? jackNodeById.get(nodeId) : mapData.jack_nodes[nodeId];
    if (!node) continue;
    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", node.x);
    c.setAttribute("cy", node.y);
    c.setAttribute("r", 13);
    c.setAttribute("fill", COLOR_ZONE);
    c.setAttribute("stroke", COLOR_ZONE_STROKE);
    c.setAttribute("stroke-width", "2.5");
    g.appendChild(c);
  }

  svg.appendChild(g);
}

// ── Cop trail layer ───────────────────────────────────────

function renderCopTrails() {
  const svg = document.getElementById("board");

  // Remove old trail layer
  const old = svg.getElementById("cop-trail-layer");
  if (old) old.remove();

  if (!roundCopPaths.size) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "cop-trail-layer");
  g.setAttribute("pointer-events", "none");

  // Draw each cop's trail below the overlay; cops with lower index are drawn first
  // so higher-index cop colors render on top where paths overlap.
  // Semi-transparent strokes blend naturally when paths share segments.
  for (const [copIdx, path] of [...roundCopPaths.entries()].sort((a, b) => a[0] - b[0])) {
    if (path.length < 2) continue;  // didn't move
    const color = COP_COLORS[copIdx % COP_COLORS.length];

    const coords = path.map(id => {
      const n = copNodeById.get(id);
      return `${n.x},${n.y}`;
    });

    const line = document.createElementNS(SVG_NS, "polyline");
    line.setAttribute("points", coords.join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", color);
    line.setAttribute("stroke-width", "2.5");
    line.setAttribute("stroke-opacity", "0.55");
    line.setAttribute("stroke-linecap", "round");
    line.setAttribute("stroke-linejoin", "round");
    g.appendChild(line);

    // Dot at each intermediate node (not start, not end)
    for (let i = 1; i < path.length - 1; i++) {
      const n = copNodeById.get(path[i]);
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", n.x);
      dot.setAttribute("cy", n.y);
      dot.setAttribute("r", 3);
      dot.setAttribute("fill", color);
      dot.setAttribute("opacity", "0.6");
      g.appendChild(dot);
    }
  }

  // Insert below overlay-group
  const overlay = svg.getElementById("overlay-group");
  if (overlay) svg.insertBefore(g, overlay);
  else svg.appendChild(g);
}

// ── Sidebar ───────────────────────────────────────────────

function updateSidebar(state) {
  document.getElementById("rc-current").textContent  = state.turn + 1;
  document.getElementById("rc-limit").textContent    = state.turn_limit;
  document.getElementById("stat-jack").textContent   = state.jack_pos;
  document.getElementById("stat-hideout").textContent = state.hideout;
  document.getElementById("stat-moves").textContent  = state.legal_moves.length;
}

// ── Cop action summary ────────────────────────────────────

function updateCopSummary(events) {
  const container = document.getElementById("cop-summary");
  container.innerHTML = "";
  if (!events?.length) return;

  const byCop = new Map();
  for (const ev of events) {
    if (!byCop.has(ev.cop)) byCop.set(ev.cop, []);
    byCop.get(ev.cop).push(ev);
  }

  for (const [copIdx, evs] of byCop.entries()) {
    const ev  = evs[evs.length - 1];
    const div = document.createElement("div");

    if (ev.action === "search") {
      const searched = ev.jack_neighbours?.join(", ") || "none";
      const hits     = ev.search_hits || [];
      div.className  = "cop-event searching";
      div.innerHTML  = hits.length > 0
        ? `<span class="cop-id">Cop ${copIdx + 1}</span> searched ${searched} → <span class="hit">found ${hits.join(", ")}</span>`
        : `<span class="cop-id">Cop ${copIdx + 1}</span> searched ${searched} → <span class="miss">no finds</span>`;
    } else {
      div.className = ev.arrest_success ? "cop-event arrest-success" : "cop-event arresting";
      const targetDesc = ev.arrest_all
        ? "all adjacent nodes"
        : `node ${ev.arrest_target ?? "?"}`;
      div.innerHTML = ev.arrest_success
        ? `<span class="cop-id">Cop ${copIdx + 1}</span> arrested ${targetDesc} → <span class="hit">CAUGHT</span>`
        : `<span class="cop-id">Cop ${copIdx + 1}</span> arrested ${targetDesc} → <span class="miss">not found</span>`;
    }

    container.appendChild(div);
  }
}

// ── Animation helpers ─────────────────────────────────────

function delay(ms) { return new Promise(res => setTimeout(res, ms)); }

// Animate a single SVG element along a sequence of {x,y} waypoints.
// For cop rects: pass setter = (x,y) => { rect.setAttribute("x", x-4); rect.setAttribute("y", y-4); }
// For circles: pass setter = (x,y) => { c.setAttribute("cx",x); c.setAttribute("cy",y); }
function animateAlongPath(setter, coordPath, durationMs) {
  return new Promise(resolve => {
    if (coordPath.length < 2) { resolve(); return; }

    // Compute segment lengths for proportional time allocation
    const segLengths = [];
    let totalLen = 0;
    for (let i = 0; i < coordPath.length - 1; i++) {
      const dx = coordPath[i+1].x - coordPath[i].x;
      const dy = coordPath[i+1].y - coordPath[i].y;
      const l  = Math.sqrt(dx*dx + dy*dy);
      segLengths.push(l);
      totalLen += l;
    }
    if (totalLen === 0) { resolve(); return; }

    const startTime = performance.now();

    function frame(now) {
      const t = Math.min((now - startTime) / durationMs, 1);

      // Find which segment we're in based on elapsed distance fraction
      const dist = t * totalLen;
      let accumulated = 0;
      let seg = 0;
      while (seg < segLengths.length - 1 && accumulated + segLengths[seg] < dist) {
        accumulated += segLengths[seg];
        seg++;
      }
      const segT = segLengths[seg] === 0 ? 0 : (dist - accumulated) / segLengths[seg];
      const from = coordPath[seg];
      const to   = coordPath[seg + 1] || coordPath[seg];
      setter(
        from.x + (to.x - from.x) * segT,
        from.y + (to.y - from.y) * segT,
      );

      if (t < 1) requestAnimationFrame(frame);
      else resolve();
    }

    requestAnimationFrame(frame);
  });
}

// ── Jack movement animation ───────────────────────────────

// Returns coordinate path [{x,y},...] for Jack moving from srcId to dstId,
// passing through via cop nodes. Picks one edge_route at random if multiple exist.
function jackCoordPath(srcId, dstId) {
  const srcNode = jackNodeById.get(srcId);
  const routes  = (srcNode?.edge_routes || []).filter(r => r.destination === dstId);
  const route   = routes.length ? routes[Math.floor(Math.random() * routes.length)] : null;

  const path = [jackNodeById.get(srcId)];
  if (route) {
    for (const viaId of route.via) {
      const vn = copNodeById.get(viaId);
      if (vn) path.push(vn);
    }
  }
  path.push(jackNodeById.get(dstId));
  return path.map(n => ({ x: n.x, y: n.y }));
}

async function animateJackMove(srcId, dstId) {
  const svg = document.getElementById("board");
  const g   = svg.getElementById("overlay-group");
  if (!g) return;

  const startNode = jackNodeById.get(srcId);
  if (!startNode) return;

  // Temporary circle for animation; authoritative ring drawn by render()
  const circ = document.createElementNS(SVG_NS, "circle");
  circ.setAttribute("cx", startNode.x);
  circ.setAttribute("cy", startNode.y);
  circ.setAttribute("r", 10);
  circ.setAttribute("fill", "none");
  circ.setAttribute("stroke", COLOR_JACK);
  circ.setAttribute("stroke-width", 4);
  circ.setAttribute("pointer-events", "none");
  g.appendChild(circ);

  // Remove the static ring before animating so there's only one green circle.
  // circ stays at destination after animation as a placeholder until render() redraws.
  const oldRing = svg.getElementById("jack-ring");
  if (oldRing) oldRing.remove();

  const coords = jackCoordPath(srcId, dstId);
  await animateAlongPath(
    (x, y) => { circ.setAttribute("cx", x); circ.setAttribute("cy", y); },
    coords,
    320,
  );
}

// ── Cop turn animation ────────────────────────────────────

async function animateCopTurn(events) {
  document.getElementById("police-moving").classList.add("visible");
  document.getElementById("cop-summary").innerHTML = "";

  roundCopPaths.clear();

  animCopPositions.clear();
  if (lastState) {
    lastState.cop_positions.forEach((pos, i) => animCopPositions.set(i, pos));
  }

  const svg = document.getElementById("board");

  // Cops are removed one at a time (per-cop) immediately before that cop starts moving,
  // so the other cops remain visible on the board throughout the animation sequence.

  // Pass 1: all cops move to their destinations
  for (const ev of events) {
    const fromId = animCopPositions.get(ev.cop);
    const toId   = ev.moved_to;

    const path = fromId !== undefined ? findCopPath(fromId, toId) : [toId];
    roundCopPaths.set(ev.cop, path);

    const g = svg.getElementById("overlay-group");
    if (g && path.length > 0 && copNodeById.has(path[0])) {
      // Remove only this cop's static group (rect + dot) so the others stay visible
      const staticGroup = copRectElems.get(ev.cop);
      if (staticGroup?.parentNode) staticGroup.remove();

      const startNode = copNodeById.get(path[0]);
      const rect = document.createElementNS(SVG_NS, "rect");
      rect.setAttribute("x", startNode.x - 4);
      rect.setAttribute("y", startNode.y - 4);
      rect.setAttribute("width", 8);
      rect.setAttribute("height", 8);
      rect.setAttribute("fill", COLOR_COP);
      rect.setAttribute("stroke", COLOR_COP_STROKE);
      rect.setAttribute("stroke-width", 1.5);
      rect.setAttribute("pointer-events", "none");
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", startNode.x);
      dot.setAttribute("cy", startNode.y);
      dot.setAttribute("r", 2.5);
      dot.setAttribute("fill", COP_COLORS[ev.cop % COP_COLORS.length]);
      dot.setAttribute("pointer-events", "none");
      g.appendChild(rect);
      g.appendChild(dot);

      // Animate segment-by-segment so the pieces follow each graph edge in sequence,
      // stopping briefly at intermediate nodes rather than cutting across diagonals.
      const STEP_DURATION = 220;
      const INTERMEDIATE_PAUSE = 90;
      if (path.length > 1) {
        for (let step = 0; step < path.length - 1; step++) {
          const fromNode = copNodeById.get(path[step]);
          const toNode   = copNodeById.get(path[step + 1]);
          if (!fromNode || !toNode) continue;
          await animateAlongPath(
            (x, y) => {
              rect.setAttribute("x", x - 4);
              rect.setAttribute("y", y - 4);
              dot.setAttribute("cx", x);
              dot.setAttribute("cy", y);
            },
            [{ x: fromNode.x, y: fromNode.y }, { x: toNode.x, y: toNode.y }],
            STEP_DURATION,
          );
          if (step < path.length - 2) await delay(INTERMEDIATE_PAUSE);
        }
      }
      // Rect and dot stay at destination — visible until render() clears the overlay.
    }

    animCopPositions.set(ev.cop, toId);
    await delay(100);
  }

  // Pass 2: all cops perform their searches/arrests
  for (const ev of events) {
    if (ev.action === "search") {
      await flashSearchedNodes(ev);
    } else {
      await flashArrestNodes(ev);
    }
    await delay(250);
  }

  document.getElementById("police-moving").classList.remove("visible");
}

async function flashSearchedNodes(ev) {
  const svg = document.getElementById("board");
  const g   = svg.getElementById("overlay-group");
  if (!g) return;

  const flashes = [];
  for (const nodeId of (ev.jack_neighbours || [])) {
    const node  = jackNodeById.get(nodeId);
    if (!node) continue;
    const isHit = (ev.search_hits || []).includes(nodeId);

    const c = document.createElementNS(SVG_NS, "circle");
    c.setAttribute("cx", node.x);
    c.setAttribute("cy", node.y);
    c.setAttribute("r", 11);
    c.setAttribute("fill",         isHit ? "rgba(239,68,68,0.35)"   : "rgba(103,232,249,0.25)");  // hit=cop-red, miss=cyan
    c.setAttribute("stroke",       isHit ? COLOR_COP               : "#67e8f9");
    c.setAttribute("stroke-width", isHit ? "2.5"                   : "1.5");
    c.setAttribute("pointer-events", "none");
    g.appendChild(c);
    flashes.push(c);
  }

  await delay(500);
  flashes.forEach(f => f.remove());
}

// Flash arrest targets — when arrest_all, flash all jack_neighbours; otherwise single target
async function flashArrestNodes(ev) {
  const svg = document.getElementById("board");
  const g   = svg.getElementById("overlay-group");
  if (!g) return;

  const targets = ev.arrest_all
    ? (ev.jack_neighbours || [])
    : (ev.arrest_target != null ? [ev.arrest_target] : []);

  if (!targets.length) return;

  const pulses = [];
  for (const nodeId of targets) {
    const node = jackNodeById.get(nodeId);
    if (!node) continue;

    const pulse = document.createElementNS(SVG_NS, "circle");
    pulse.setAttribute("cx", node.x);
    pulse.setAttribute("cy", node.y);
    pulse.setAttribute("r", 13);
    pulse.setAttribute("fill", "rgba(239,68,68,0.4)");
    pulse.setAttribute("stroke", COLOR_COP);
    pulse.setAttribute("stroke-width", 3);
    pulse.setAttribute("pointer-events", "none");
    g.appendChild(pulse);
    pulses.push(pulse);
  }

  await delay(600);
  pulses.forEach(p => p.remove());
}

// ── Jack move handler ─────────────────────────────────────

async function handleJackMove(destination) {
  if (busy || !gameId || !lastState) return;
  busy = true;
  document.getElementById("board-wrapper").classList.add("busy");

  try {
    const srcId = lastState.jack_pos;

    // Start Jack animation and server call in parallel
    const [state] = await Promise.all([
      jackMove(gameId, destination),
      animateJackMove(srcId, destination),
    ]);

    const events = state.events || [];
    await animateCopTurn(events);
    renderCopTrails();
    updateCopSummary(events);
    render(state);

    if (state.terminated) showGameOver(state);
  } catch (e) {
    console.error("Move error:", e);
  } finally {
    busy = false;
    document.getElementById("board-wrapper").classList.remove("busy");
  }
}

// ── Game over overlay ─────────────────────────────────────

function showGameOver(state) {
  const outcomes = {
    jack:       { icon: "🌙", title: "You Escaped",  sub: "You slipped through the night and reached your hideout. The police are left with nothing." },
    cops:       { icon: "🔒", title: "Caught",        sub: "A constable's hand falls on your shoulder. The game is up." },
    timeout:    { icon: "⏳", title: "Time's Up",     sub: "Dawn breaks over Whitechapel. The police seal every route — there's no way out now." },
    surrounded: { icon: "🔦", title: "Surrounded",    sub: "Every path is blocked. The police have closed the net." },
  };

  const o = outcomes[state.winner] || { icon: "?", title: state.winner, sub: "" };
  document.getElementById("gameover-icon").textContent  = o.icon;
  document.getElementById("gameover-title").textContent = o.title;
  document.getElementById("gameover-sub").textContent   = o.sub;
  document.getElementById("gameover-turns").textContent =
    `${state.turn} round${state.turn !== 1 ? "s" : ""} played`;

  const nextIndex = courseIndex + 1;
  const btn = document.getElementById("gameover-btn");

  if (nextIndex < course.length) {
    btn.textContent = "Next Map →";
    btn.addEventListener("click", async () => {
      btn.disabled    = true;
      btn.textContent = "Loading…";
      const nextEntry   = course[nextIndex];
      const gamingHabit = sessionStorage.getItem("gaming_habit") || "unknown";
      const newState    = await newGame(nextEntry.name, gamingHabit);
      sessionStorage.setItem("course_index", String(nextIndex));
      sessionStorage.setItem("game_id", newState.game_id);
      window.location.reload();
    });
  } else {
    btn.textContent = "Course Complete";
    btn.addEventListener("click", () => {
      sessionStorage.clear();
      window.location.href = "index.html";
    });
  }

  requestAnimationFrame(() => {
    document.getElementById("gameover-overlay").classList.add("visible");
  });
}

// ── Bootstrap ─────────────────────────────────────────────
init();
