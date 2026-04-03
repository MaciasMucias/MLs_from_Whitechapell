const SVG_NS = "http://www.w3.org/2000/svg";

let mapData = null;
let gameId = null;
let busy = false;

async function init() {
  mapData = await fetchMap();
  document.getElementById("new-game-btn").addEventListener("click", startNewGame);
}

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
        logEntry(`  Cop ${ev.cop} → cop-node ${ev.moved_to} (searched jack nodes: ${nb})`, "event-cop");
      }
    }
    if (state.terminated) {
      const msg = state.winner === "jack" ? "Jack escaped! Jack wins." : "Cops win!";
      logEntry(msg, "event-result");
    }
  } catch (e) {
    logEntry(`Error: ${e.message}`, "event-result");
  } finally {
    busy = false;
  }
}

function render(state) {
  const svg = document.getElementById("board");

  const old = svg.getElementById("overlay-group");
  if (old) old.remove();

  const g = document.createElementNS(SVG_NS, "g");
  g.setAttribute("id", "overlay-group");

  const legalSet = new Set(state.legal_moves);
  const visitedSet = new Set(state.visited);
  const copSet = new Set(state.cop_positions);

  // Cop nodes (drawn first so jack nodes appear on top)
  for (const node of mapData.cop_nodes) {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", node.x - 4);
    rect.setAttribute("y", node.y - 4);
    rect.setAttribute("width", 8);
    rect.setAttribute("height", 8);

    if (copSet.has(node.id)) {
      rect.setAttribute("fill", "#d32f2f");
      rect.setAttribute("stroke", "#ff6659");
      rect.setAttribute("stroke-width", 1.5);
    } else {
      rect.setAttribute("fill", "rgba(20,20,20,0.4)");
      rect.setAttribute("stroke", "none");
    }

    g.appendChild(rect);
  }

  // Jack nodes — segmented rings for active states, nothing for plain nodes.
  // Multiple active states produce N equal arc segments via stroke-dasharray.
  for (const node of mapData.jack_nodes) {
    const isJack    = node.id === state.jack_pos;
    const isHideout = node.id === state.hideout;
    const isLegal   = legalSet.has(node.id) && !state.terminated;
    const isVisited = visitedSet.has(node.id);

    const activeStates = [];
    if (isJack)    activeStates.push({ stroke: "#2e7d32" });
    if (isHideout) activeStates.push({ stroke: "#f9a825" });
    if (isLegal)   activeStates.push({ stroke: "#ff6d00" });
    if (isVisited) activeStates.push({ stroke: "#1565c0" });

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
        // rotate(-90) moves path start from 3 o'clock to 12 o'clock
        c.setAttribute("transform", `rotate(-90, ${node.x}, ${node.y})`);
        c.setAttribute("stroke-dasharray", `${segLen} ${circumference - segLen}`);
        c.setAttribute("stroke-dashoffset", -(i * segLen));
      }
      segCircles.push(c);
      nodeGroup.appendChild(c);
    }

    if (isLegal) {
      // Transparent filled circle so the whole node area is clickable
      const hit = document.createElementNS(SVG_NS, "circle");
      hit.setAttribute("cx", node.x);
      hit.setAttribute("cy", node.y);
      hit.setAttribute("r", r);
      hit.setAttribute("fill", "transparent");
      hit.setAttribute("stroke", "none");
      hit.style.cursor = "pointer";
      hit.addEventListener("click", () => handleJackMove(node.id));
      hit.addEventListener("mouseenter", () => {
        segCircles.forEach(c => c.setAttribute("stroke-width", sw + 2));
      });
      hit.addEventListener("mouseleave", () => {
        segCircles.forEach(c => c.setAttribute("stroke-width", sw));
      });
      nodeGroup.appendChild(hit);
    }

    g.appendChild(nodeGroup);
  }

  svg.appendChild(g);
  updateStatus(state);
}

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
      `Cop-visited nodes: <strong>${state.visited.length}</strong>`;
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
