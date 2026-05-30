"""
Extract a scripted-sim test-case dict from an existing replay file.

Reads a saved replay JSON (e.g. data/replays/slot_1.json) and prints a
ready-to-paste Python dict in the format expected by run_scripted_game()
in tools/scripted_sim.py.

The printed block includes:
  - A comment with the expected outcome (winner + turns_survived)
  - All fields needed to reconstruct the starting state
  - jack_script: the sequence of Jack destinations, one per round

Usage:
    uv run tools/replay_to_test.py data/replays/slot_1.json
    uv run tools/replay_to_test.py data/replays/slot_1.json --name MY_CASE

Arguments:
    replay_path     Path to the replay JSON file.
    --name NAME     Variable name for the printed dict. Defaults to the
                    filename stem uppercased with hyphens/dots replaced by
                    underscores (e.g. slot_1.json -> SLOT_1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def replay_to_test(replay_path: str, name: str | None = None) -> str:
    path = Path(replay_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    if name is None:
        name = path.stem.replace("-", "_").replace(".", "_").upper()

    jack_script = [r["jack_to"] for r in data["rounds"]]
    hideout_zone = sorted(data["hideout_zone"])
    cop_positions = tuple(data["initial_cop_positions"])

    lines = [
        f'# Expected: winner="{data["winner"]}", turns_survived={data["turns_survived"]}',
        f"{name} = dict(",
        f"    initial_jack_pos={data['initial_jack_pos']},",
        f"    initial_cop_positions={cop_positions!r},",
        f"    hideout={data['hideout']},",
        f"    hideout_zone_anchor={data['hideout_zone_anchor']},",
        f"    hideout_zone=frozenset({{{', '.join(str(n) for n in hideout_zone)}}}),",
        f"    turn_limit={data['turn_limit']},",
        f"    blocking={data['blocking']},",
        f"    jack_script={jack_script!r},",
        ")",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a replay JSON into a scripted-sim test-case dict."
    )
    parser.add_argument("replay_path", help="Path to the replay JSON file")
    parser.add_argument(
        "--name",
        default=None,
        help="Variable name for the output dict (default: derived from filename)",
    )
    args = parser.parse_args()
    print(replay_to_test(args.replay_path, args.name))


if __name__ == "__main__":
    main()
