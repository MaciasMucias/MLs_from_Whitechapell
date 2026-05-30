async function fetchMap() {
  const r = await fetch("/api/map");
  if (!r.ok) throw new Error("Failed to load map");
  return r.json();
}

async function newGame() {
  const r = await fetch("/api/game/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ map_name: "whitechapel", gaming_habit: "unknown" }),
  });
  if (!r.ok) throw new Error("Failed to start game");
  return r.json();
}

async function jackMove(gameId, destination) {
  const r = await fetch(`/api/game/${gameId}/jack-move`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ destination }),
  });
  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.detail || "Move failed");
  }
  return r.json();
}
