async function fetchCourse() {
  const r = await fetch("/api/course");
  if (!r.ok) throw new Error("Nie udało się załadować kursu");
  return r.json();
}

async function fetchMap(mapName) {
  const url = mapName ? `/api/map?map_name=${encodeURIComponent(mapName)}` : "/api/map";
  const r = await fetch(url);
  if (!r.ok) throw new Error("Nie udało się załadować mapy");
  return r.json();
}

async function fetchGame(gameId) {
  const r = await fetch(`/api/game/${gameId}`);
  if (!r.ok) throw new Error("Nie znaleziono gry");
  return r.json();
}

async function newGame(gamingHabit) {
  const r = await fetch("/api/game/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gaming_habit: gamingHabit }),
  });
  if (!r.ok) throw new Error("Nie udało się rozpocząć gry");
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
    throw new Error(err.detail || "Ruch nie powiódł się");
  }
  return r.json();
}
