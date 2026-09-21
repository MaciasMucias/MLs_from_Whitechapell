"""Check the numbers FINDINGS quotes against the CSVs and re-scoring outputs."""

import collections
import csv
import re
import statistics as st

LAST_K = 5
ALIAS = {"w11-ent003": "w10s-obj-cur"}

runs = collections.defaultdict(list)
for wave in ("reward", "reward_topup", "entropy"):
    for r in csv.DictReader(
        open(f"docs/completion/results/{wave}_eval.csv", encoding="utf-8")
    ):
        if r.get("score") in (None, ""):
            continue
        arm = ALIAS.get(r["arm"], r["arm"])
        if arm.startswith("w11-"):
            continue
        runs[(arm, r["seed"])].append((int(r["step"]), float(r["score"])))

arms = collections.defaultdict(dict)
for (arm, seed), pts in runs.items():
    v = [x for _, x in sorted(pts)]
    arms[arm][seed] = sum(v[-LAST_K:]) / len(v[-LAST_K:])

claims = {
    "w10s-obj-cur": 1.295,
    "w10s-dlt-cur": 1.306,
    "w10s-shp-cur": 1.278,
    "w10s-obj-off": 1.175,
    "w10s-dlt-off": 1.163,
    "w10s-shp-off": 1.185,
}
print("§0 table — 6-seed arm means")
ok = True
for arm, claimed in claims.items():
    actual = st.mean(arms[arm].values())
    n = len(arms[arm])
    flag = "OK " if abs(actual - claimed) < 0.0006 and n == 6 else "!! "
    ok &= flag == "OK "
    print(f"  {flag}{arm:<16} doc {claimed:.3f}  data {actual:.3f}  n={n}")

print("\n§0 curriculum differences")
for a, b, claimed in [
    ("w10s-obj-cur", "w10s-obj-off", 0.120),
    ("w10s-dlt-cur", "w10s-dlt-off", 0.143),
    ("w10s-shp-cur", "w10s-shp-off", 0.093),
]:
    actual = st.mean(arms[a].values()) - st.mean(arms[b].values())
    flag = "OK " if abs(actual - claimed) < 0.0011 else "!! "
    ok &= flag == "OK "
    print(f"  {flag}{a} - {b}: doc {claimed:+.3f}  data {actual:+.3f}")

print("\n§6b pooled SD and thresholds")
sds = [(st.stdev(v.values()), len(v)) for v in arms.values()]
pooled = (sum(sd**2 * (n - 1) for sd, n in sds) / sum(n - 1 for _, n in sds)) ** 0.5
print(
    f"  pooled SD data {pooled:.4f}  doc 0.0516 -> {'OK' if abs(pooled - 0.0516) < 0.0006 else '!!'}"
)
for n, claimed in ((3, 0.118), (6, 0.083), (10, 0.065)):
    mdd = 2.8 * pooled * (2 / n) ** 0.5
    flag = "OK" if abs(mdd - claimed) < 0.002 else "!!"
    print(f"  n={n}: data {mdd:.3f}  doc {claimed:.3f} -> {flag}")

print("\n§6 human comparison (from comparison_20260921.txt)")
rows = [
    l.split()
    for l in open("docs/completion/results/comparison_20260921.txt", encoding="utf-8")
    if re.match(r"^w1\S+\s+15\.00M", l)
]
groups = collections.defaultdict(list)
for r in rows:
    key = (
        "obj-cur"
        if ("obj-cur" in r[0] or "ent003" in r[0])
        else "dlt-cur"
        if "dlt" in r[0]
        else "shp-cur"
    )
    groups[key].append((float(r[2]), float(r[4].rstrip("%")), float(r[8].rstrip("%"))))
for k, claimed in (("obj-cur", 1.260), ("dlt-cur", 1.257), ("shp-cur", 1.248)):
    actual = st.mean(x[0] for x in groups[k])
    flag = "OK " if abs(actual - claimed) < 0.0006 else "!! "
    print(f"  {flag}{k}: doc {claimed:.3f}  data {actual:.3f}  n={len(groups[k])}")
print(
    f"  checkpoints beating humans: {sum(1 for v in groups.values() for x in v if x[0] > 0.672)}/18"
)
print(
    f"  score range: {min(x[0] for v in groups.values() for x in v):.3f}"
    f" - {max(x[0] for v in groups.values() for x in v):.3f}"
)

print("\n§0 2,000-game re-scoring (final_reeval.txt / reward_reeval.txt)")
for path, wanted in (
    (
        "final_reeval.txt",
        {"director-on": 1.328, "director-off": 1.300, "sparse": 1.299},
    ),
    (
        "reward_reeval.txt",
        {"w10s-obj-cur": 1.326, "w10s-obj-off": 1.159, "w10s-shp-off": 1.243},
    ),
):
    txt = open(f"docs/completion/results/{path}", encoding="utf-8").read()
    block = txt.split("Cops:")[1]  # COPS_STUDY_V2 table only
    got = collections.defaultdict(list)
    for m in re.finditer(
        r"checkpoints/([\w-]+)-s\d+/agent_\S+\s+15\.00M\s+([\d.]+)", block
    ):
        got[m.group(1)].append(float(m.group(2)))
    for arm, claimed in wanted.items():
        actual = st.mean(got[arm])
        flag = "OK " if abs(actual - claimed) < 0.0006 else "!! "
        print(f"  {flag}{arm:<16} doc {claimed:.3f}  data {actual:.3f}")
