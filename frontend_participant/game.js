const SVG_NS = "http://www.w3.org/2000/svg";

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

// Hover state
let lastEvents    = [];   // events from the most recent cop turn
let hoveredCopIdx = null; // cop currently highlighted via hover

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

  renderHideoutStar(state);
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
      rect.style.fill   = "var(--color-cop)";
      rect.style.stroke = "var(--color-cop-stroke)";
      rect.setAttribute("stroke-width", 1.5);
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", node.x);
      dot.setAttribute("cy", node.y);
      dot.setAttribute("r", 2.5);
      dot.style.fill = `var(--color-cop-${copIdx % 6})`;
      const copGroup = document.createElementNS(SVG_NS, "g");
      copGroup.appendChild(rect);
      copGroup.appendChild(dot);
      copGroup.style.cursor = "pointer";
      copGroup.addEventListener("mouseenter", () => { if (!busy) showCopHoverOverlay(copIdx); });
      copGroup.addEventListener("mouseleave", clearCopHoverOverlay);
      g.appendChild(copGroup);
      copRectElems.set(copIdx, copGroup);
    } else {
      rect.style.fill = "var(--color-cop-node-empty)";
      rect.setAttribute("stroke", "none");
      g.appendChild(rect);
    }
  }
}

// ── Star helper ───────────────────────────────────────────

function makeStar(cx, cy, outerR, innerR, points) {
  const coords = [];
  for (let i = 0; i < points * 2; i++) {
    const angle = (i * Math.PI) / points - Math.PI / 2;
    const r = i % 2 === 0 ? outerR : innerR;
    coords.push(`${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`);
  }
  const poly = document.createElementNS(SVG_NS, "polygon");
  poly.setAttribute("points", coords.join(" "));
  return poly;
}

// ── Jack node rings ───────────────────────────────────────

function renderJackNodes(g, state, legalSet, visitedSet) {
  for (const node of mapData.jack_nodes) {
    const isJack    = node.id === state.jack_pos;
    const isHideout = node.id === state.hideout;
    const isLegal   = legalSet.has(node.id) && !state.terminated;
    const isVisited = visitedSet.has(node.id);

    const segments = [];
    if (isJack)    segments.push("var(--color-jack)");
    if (isLegal)   segments.push("var(--color-legal)");
    if (isVisited) segments.push("var(--color-visited)");

    if (segments.length === 0) continue;

    const r  = isJack ? 10 : 9;
    const sw = isJack ? 4  : 3;
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
      c.style.stroke = segments[i];
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
    c.style.fill   = "var(--color-zone)";
    c.style.stroke = "var(--color-zone-stroke)";
    c.setAttribute("stroke-width", "2.5");
    g.appendChild(c);
  }

  svg.appendChild(g);
}

// ── Hideout star layer (behind map image) ─────────────────

function renderHideoutStar(state) {
  const svg = document.getElementById("board");
  const old = svg.getElementById("hideout-star-layer");
  if (old) old.remove();

  const node = jackNodeById?.get(state.hideout);
  if (!node) return;

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "hideout-star-layer");
  g.setAttribute("pointer-events", "none");

  const star = makeStar(node.x, node.y, 25, 15, 5);
  star.style.fill   = "var(--color-hideout)";
  star.style.stroke = "none";
  g.appendChild(star);

  const image = svg.querySelector("image");
  if (image) svg.insertBefore(g, image);
  else svg.prepend(g);
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
    const color = `var(--color-cop-${copIdx % 6})`;

    const coords = path.map(id => {
      const n = copNodeById.get(id);
      return `${n.x},${n.y}`;
    });

    const line = document.createElementNS(SVG_NS, "polyline");
    line.setAttribute("points", coords.join(" "));
    line.setAttribute("fill", "none");
    line.style.stroke = color;
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
      dot.style.fill = color;
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
  document.getElementById("stat-jack").textContent   = state.jack_pos + 1;
  document.getElementById("stat-hideout").textContent = state.hideout + 1;
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
    div.dataset.copIdx = copIdx;

    if (ev.action === "search") {
      const searched = ev.jack_neighbours?.map(n => n + 1).join(", ") || "none";
      const hits     = (ev.search_hits || []).map(n => n + 1);
      div.className  = "cop-event searching";
      div.innerHTML  = hits.length > 0
        ? `<span class="cop-id" style="color:var(--color-cop-${copIdx % 6})">Policjant ${copIdx + 1}</span> poszukiwał na ${searched} → <span class="hit">znalazł ${hits.join(", ")}</span>`
        : `<span class="cop-id" style="color:var(--color-cop-${copIdx % 6})">Policjant ${copIdx + 1}</span> poszukiwał na ${searched} → <span class="miss">nic nie znalazł</span>`;
    } else {
      div.className = ev.arrest_success ? "cop-event arrest-success" : "cop-event arresting";
      const targetDesc = ev.arrest_all
        ? "wszystkie sąsiednie pola"
        : `pole ${ev.arrest_target != null ? ev.arrest_target + 1 : "?"}`;
      div.innerHTML = ev.arrest_success
        ? `<span class="cop-id" style="color:var(--color-cop-${copIdx % 6})">Policjant ${copIdx + 1}</span> aresztował ${targetDesc} → <span class="hit">ZŁAPANY</span>`
        : `<span class="cop-id" style="color:var(--color-cop-${copIdx % 6})">Policjant ${copIdx + 1}</span> aresztował ${targetDesc} → <span class="miss">nie znaleziono</span>`;
    }

    div.addEventListener("mouseenter", () => { if (!busy) showCopHoverOverlay(copIdx); });
    div.addEventListener("mouseleave", clearCopHoverOverlay);
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
  circ.style.stroke = "var(--color-jack)";
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
      rect.style.fill   = "var(--color-cop)";
      rect.style.stroke = "var(--color-cop-stroke)";
      rect.setAttribute("stroke-width", 1.5);
      rect.setAttribute("pointer-events", "none");
      const dot = document.createElementNS(SVG_NS, "circle");
      dot.setAttribute("cx", startNode.x);
      dot.setAttribute("cy", startNode.y);
      dot.setAttribute("r", 2.5);
      dot.style.fill = `var(--color-cop-${ev.cop % 6})`;
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
    c.style.fill   = isHit ? "var(--color-search-hit-fill)"    : "var(--color-search-miss-fill)";
    c.style.stroke = isHit ? "var(--color-cop)"               : "var(--color-search-miss-stroke)";
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
    pulse.style.fill   = "var(--color-arrest-pulse-fill)";
    pulse.style.stroke = "var(--color-cop)";
    pulse.setAttribute("stroke-width", 3);
    pulse.setAttribute("pointer-events", "none");
    g.appendChild(pulse);
    pulses.push(pulse);
  }

  await delay(600);
  pulses.forEach(p => p.remove());
}

// ── Hover cross-highlight ─────────────────────────────────

function showCopHoverOverlay(copIdx) {
  clearCopHoverOverlay();
  if (!lastState) return;
  hoveredCopIdx = copIdx;

  const svg = document.getElementById("board");
  const g   = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "cop-hover-overlay");
  g.setAttribute("pointer-events", "none");

  // Ring at cop's current board position
  const posNodeId = lastState.cop_positions[copIdx];
  const posNode   = posNodeId !== undefined ? copNodeById.get(posNodeId) : null;
  if (posNode) {
    const ring = document.createElementNS(SVG_NS, "circle");
    ring.setAttribute("cx", posNode.x);
    ring.setAttribute("cy", posNode.y);
    ring.setAttribute("r", 9);
    ring.setAttribute("fill", "none");
    ring.style.stroke = `var(--color-cop-${copIdx % 6})`;
    ring.setAttribute("stroke-width", "2.5");
    ring.setAttribute("opacity", "0.9");
    g.appendChild(ring);
  }

  // Search / arrest indicator circles for this cop's last event
  for (const ev of lastEvents) {
    if (ev.cop !== copIdx) continue;
    if (ev.action === "search") {
      for (const nodeId of (ev.jack_neighbours || [])) {
        const node  = jackNodeById.get(nodeId);
        if (!node) continue;
        const isHit = (ev.search_hits || []).includes(nodeId);
        const c = document.createElementNS(SVG_NS, "circle");
        c.setAttribute("cx", node.x);
        c.setAttribute("cy", node.y);
        c.setAttribute("r", 11);
        c.style.fill   = isHit ? "var(--color-search-hit-fill)"  : "var(--color-search-miss-fill)";
        c.style.stroke = isHit ? "var(--color-cop)"              : "var(--color-search-miss-stroke)";
        c.setAttribute("stroke-width", isHit ? "2.5" : "1.5");
        g.appendChild(c);
      }
    } else {
      const targets = ev.arrest_all
        ? (ev.jack_neighbours || [])
        : (ev.arrest_target != null ? [ev.arrest_target] : []);
      for (const nodeId of targets) {
        const node = jackNodeById.get(nodeId);
        if (!node) continue;
        const pulse = document.createElementNS(SVG_NS, "circle");
        pulse.setAttribute("cx", node.x);
        pulse.setAttribute("cy", node.y);
        pulse.setAttribute("r", 13);
        pulse.style.fill   = "var(--color-arrest-pulse-fill)";
        pulse.style.stroke = "var(--color-cop)";
        pulse.setAttribute("stroke-width", "3");
        g.appendChild(pulse);
      }
    }
  }

  svg.appendChild(g);

  // Highlight the log entry
  const logDiv = document.querySelector(`#cop-summary [data-cop-idx="${copIdx}"]`);
  if (logDiv) logDiv.style.boxShadow = `inset 3px 0 0 var(--color-cop-${copIdx % 6})`;
}

function clearCopHoverOverlay() {
  const svg = document.getElementById("board");
  const old = svg?.getElementById("cop-hover-overlay");
  if (old) old.remove();

  if (hoveredCopIdx !== null) {
    const logDiv = document.querySelector(`#cop-summary [data-cop-idx="${hoveredCopIdx}"]`);
    if (logDiv) logDiv.style.boxShadow = "";
  }
  hoveredCopIdx = null;
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
    lastEvents = events;
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

// ── Helpers ───────────────────────────────────────────────

function polishRoundPlural(n) {
  if (n === 1) return "rundę";
  const mod10 = n % 10;
  const mod100 = n % 100;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20)) return "rundy";
  return "rund";
}

// ── Game over overlay ─────────────────────────────────────

function showGameOver(state) {
  const outcomes = {
    jack:       { icon: "🌙", title: "Ucieczka",         sub: "Dotarłeś/Dotarłaś do kryjówki. Policja nie zdążyła cię złapać." },
    cops:       { icon: "🔒", title: "Złapany/Złapana",  sub: "Policjant cię złapał. Koniec gry." },
    timeout:    { icon: "⏳", title: "Koniec czasu",      sub: "Przekroczono limit rund. Policja wygrała." },
    surrounded: { icon: "🔦", title: "Otoczony/Otoczona", sub: "Wszystkie drogi są zablokowane. Policja wygrała." },
  };

  const o = outcomes[state.winner] || { icon: "?", title: state.winner, sub: "" };
  document.getElementById("gameover-icon").textContent  = o.icon;
  document.getElementById("gameover-title").textContent = o.title;
  document.getElementById("gameover-sub").textContent   = o.sub;
  document.getElementById("gameover-turns").textContent =
    `Rozegrano ${state.turn} ${polishRoundPlural(state.turn)}`;

  const nextIndex = courseIndex + 1;
  const btn = document.getElementById("gameover-btn");

  if (nextIndex < course.length) {
    btn.textContent = "Następna mapa →";
    btn.addEventListener("click", async () => {
      btn.disabled    = true;
      btn.textContent = "Ładowanie…";
      const nextEntry   = course[nextIndex];
      const gamingHabit = sessionStorage.getItem("gaming_habit") || "unknown";
      const newState    = await newGame(nextEntry.name, gamingHabit);
      sessionStorage.setItem("course_index", String(nextIndex));
      sessionStorage.setItem("game_id", newState.game_id);
      window.location.reload();
    });
  } else {
    btn.textContent = "Kurs ukończony";
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
