"""Recompute every headline number in FINDINGS.md from the raw results and diff it.

The failure mode this guards against is not a wrong calculation but a stale
number surviving an edit - FINDINGS has been revised through two seed top-ups
and a retraction, and a figure quoted in one section can outlive its source.

Run before submitting, and after any re-pull of the study snapshot:

    uv run python tools/verify_findings.py

Exit status is 1 if any check fails.
"""

from __future__ import annotations

import collections
import csv
import random
import re
import statistics as st
import sys
from pathlib import Path

RESULTS = Path("docs/completion/results")
LAST_K = 5
# w11-ent003 is w10s-obj-cur flag-for-flag (same config, seeds 51-53).
ALIAS = {"w11-ent003": "w10s-obj-cur"}
WAVE10 = ("reward", "reward_topup", "reward_topup2", "entropy")

# ---- what FINDINGS claims ---------------------------------------------------
ARM_MEANS = {  # (last-5 participant score, seeds)
    "w10s-obj-cur": (1.295, 6),
    "w10s-dlt-cur": (1.306, 6),
    "w10s-shp-cur": (1.272, 9),
    "w10s-obj-off": (1.175, 6),
    "w10s-dlt-off": (1.163, 6),
    "w10s-shp-off": (1.183, 9),
}
CURRICULUM_EFFECTS = [
    ("w10s-obj-cur", "w10s-obj-off", 0.120),
    ("w10s-dlt-cur", "w10s-dlt-off", 0.143),
    ("w10s-shp-cur", "w10s-shp-off", 0.090),
]
POOLED_EFFECT = 0.114
POOLED_CI = (0.085, 0.142)
POOLED_SD = 0.0488
THRESHOLDS = {3: 0.111, 6: 0.079, 9: 0.064}
INTERACTION = 0.030
HUMAN_ARMS = {"obj-cur": 1.260, "dlt-cur": 1.257, "shp-cur": 1.248}
HUMAN_SCORE = 0.672
REEVAL = {
    "final_reeval.txt": {"director-on": 1.328, "director-off": 1.300, "sparse": 1.299},
    "reward_reeval.txt": {"w10s-obj-cur": 1.326, "w10s-obj-off": 1.159, "w10s-shp-off": 1.243},
}

failures = 0


def check(label: str, ok: bool, detail: str) -> None:
    global failures
    failures += not ok
    print(f"  {'OK ' if ok else '!! '}{label:<46} {detail}")


def close(a: float, b: float, tol: float = 0.0006) -> bool:
    return abs(a - b) < tol


# ---- load wave 10 -----------------------------------------------------------
runs: dict = collections.defaultdict(list)
for wave in WAVE10:
    for r in csv.DictReader((RESULTS / f"{wave}_eval.csv").open(encoding="utf-8")):
        if r.get("score") in (None, ""):
            continue
        arm = ALIAS.get(r["arm"], r["arm"])
        if arm.startswith("w11-"):
            continue
        runs[(arm, r["seed"])].append((int(r["step"]), float(r["score"])))
arms: dict = collections.defaultdict(dict)
for (arm, seed), pts in runs.items():
    v = [x for _, x in sorted(pts)]
    arms[arm][seed] = sum(v[-LAST_K:]) / len(v[-LAST_K:])
mean = {a: st.mean(v.values()) for a, v in arms.items()}
n = {a: len(v) for a, v in arms.items()}

print("§0 — arm means and seed counts")
for arm, (claimed, seeds) in ARM_MEANS.items():
    check(arm, close(mean[arm], claimed) and n[arm] == seeds,
          f"doc {claimed:.3f} (n={seeds})  data {mean[arm]:.3f} (n={n[arm]})")

print("\n§0 — curriculum effect per reward condition")
for a, b, claimed in CURRICULUM_EFFECTS:
    d = mean[a] - mean[b]
    check(f"{a} - {b}", close(d, claimed, 0.0011), f"doc {claimed:+.3f}  data {d:+.3f}")

print("\n§6b — pooled SD, thresholds, pooled effect")
sds = [(st.stdev(v.values()), len(v)) for v in arms.values()]
sd = (sum(s**2 * (k - 1) for s, k in sds) / sum(k - 1 for _, k in sds)) ** 0.5
check("pooled per-seed SD", close(sd, POOLED_SD), f"doc {POOLED_SD:.4f}  data {sd:.4f}")
for k, claimed in THRESHOLDS.items():
    t = 2.8 * sd * (2 / k) ** 0.5
    check(f"detection threshold at n={k}", close(t, claimed, 0.0015), f"doc {claimed:.3f}  data {t:.3f}")

cur = [v for a in ("w10s-obj-cur", "w10s-dlt-cur", "w10s-shp-cur") for v in arms[a].values()]
off = [v for a in ("w10s-obj-off", "w10s-dlt-off", "w10s-shp-off") for v in arms[a].values()]
d = st.mean(cur) - st.mean(off)
check(f"pooled curriculum effect ({len(cur)} v {len(off)})", close(d, POOLED_EFFECT, 0.0011),
      f"doc {POOLED_EFFECT:+.3f}  data {d:+.3f}")
rng = random.Random(0)
boot = sorted(
    st.mean([rng.choice(cur) for _ in cur]) - st.mean([rng.choice(off) for _ in off])
    for _ in range(20000)
)
lo, hi = boot[500], boot[19500]
check("pooled bootstrap 95% CI", close(lo, POOLED_CI[0], 0.0021) and close(hi, POOLED_CI[1], 0.0021),
      f"doc [{POOLED_CI[0]:+.3f}, {POOLED_CI[1]:+.3f}]  data [{lo:+.3f}, {hi:+.3f}]")

inter = (mean["w10s-shp-off"] - mean["w10s-obj-off"]) - (mean["w10s-shp-cur"] - mean["w10s-obj-cur"])
check("shaping x curriculum interaction", close(inter, INTERACTION, 0.0011),
      f"doc {INTERACTION:+.3f}  data {inter:+.3f}")

print("\n§6 — human comparison (comparison_20260921.txt)")
rows = [
    line.split()
    for line in (RESULTS / "comparison_20260921.txt").open(encoding="utf-8")
    if re.match(r"^w1\S+\s+15\.00M", line)
]
groups: dict = collections.defaultdict(list)
for r in rows:
    key = ("obj-cur" if ("obj-cur" in r[0] or "ent003" in r[0])
           else "dlt-cur" if "dlt" in r[0] else "shp-cur")
    groups[key].append(float(r[2]))
for arm, claimed in HUMAN_ARMS.items():
    got = st.mean(groups[arm])
    check(f"{arm} on human boards", close(got, claimed), f"doc {claimed:.3f}  data {got:.3f} (n={len(groups[arm])})")
beating = sum(1 for v in groups.values() for x in v if x > HUMAN_SCORE)
total = sum(len(v) for v in groups.values())
check("checkpoints beating humans", beating == total == 18, f"{beating}/{total}")

print("\n§0 — 2,000-game re-scorings (study cops)")
for fname, wanted in REEVAL.items():
    block = (RESULTS / fname).read_text(encoding="utf-8").split("Cops:")[1]
    got: dict = collections.defaultdict(list)
    for m in re.finditer(r"checkpoints/([\w-]+)-s\d+/agent_\S+\s+15\.00M\s+([\d.]+)", block):
        got[m.group(1)].append(float(m.group(2)))
    for arm, claimed in wanted.items():
        g = st.mean(got[arm])
        check(f"{arm} ({fname})", close(g, claimed), f"doc {claimed:.3f}  data {g:.3f}")

print(f"\n{'ALL CHECKS PASS' if not failures else f'{failures} CHECK(S) FAILED'}")
sys.exit(1 if failures else 0)
