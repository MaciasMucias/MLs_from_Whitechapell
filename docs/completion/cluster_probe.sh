#!/usr/bin/env bash
# Run this ON THE CLUSTER (login node) and paste the output back.
# It answers everything needed to write the job templates in 02-C3.
# Read-only: inspects the environment, submits nothing, installs nothing.

echo "=== scheduler ==="
for c in sbatch qsub bsub; do command -v $c >/dev/null && echo "found: $c"; done
command -v sinfo >/dev/null && sinfo -o "%20P %10l %12G %5c %10m %a" 2>/dev/null | head -20

echo; echo "=== partitions / walltime caps (slurm) ==="
command -v scontrol >/dev/null && scontrol show partition 2>/dev/null \
  | grep -E "PartitionName|MaxTime|DefaultTime|TRES=" | head -40

echo; echo "=== gpus visible from login node (may be none - normal) ==="
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no nvidia-smi on login node"

echo; echo "=== python / uv ==="
command -v uv && uv --version
for p in python3 python3.13 python3.12 python3.11; do command -v $p >/dev/null && echo "$p -> $($p --version 2>&1)"; done

echo; echo "=== module system ==="
command -v module >/dev/null && { echo "modules available"; module avail 2>&1 | grep -iE "cuda|python|anaconda|uv" | head -20; } || echo "no module command"

echo; echo "=== containers ==="
for c in singularity apptainer podman docker; do command -v $c >/dev/null && echo "found: $c"; done

echo; echo "=== internet from login node (wandb online?) ==="
curl -sS -m 8 -o /dev/null -w "api.wandb.ai -> %{http_code}\n" https://api.wandb.ai 2>&1 || echo "no outbound https"

echo; echo "=== storage / quota ==="
echo "HOME=$HOME"; df -h "$HOME" 2>/dev/null | tail -1
for v in SCRATCH WORK PROJECT TMPDIR; do [ -n "${!v}" ] && echo "$v=${!v}"; done
command -v quota >/dev/null && quota -s 2>/dev/null | head -10

echo; echo "=== job limits ==="
command -v sacctmgr >/dev/null && sacctmgr show assoc user=$USER format=Account,Partition,MaxJobs,MaxSubmit,GrpTRES 2>/dev/null | head -10
