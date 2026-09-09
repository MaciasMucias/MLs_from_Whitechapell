# 08 — Cluster access for automated work

**Status:** setup pending (one interactive step by the user)
**Blocks:** convenient execution of 02-C3 / 03 / 04 — not correctness of anything
**Blocked by:** nothing

---

## Goal

Let an assistant session deploy code to the cluster and run Slurm commands there, **without any
credential ever entering the conversation**.

## The security model

Credentials never reach the assistant. The split is:

| Who | Does what |
|---|---|
| **You**, once, interactively | Authenticate to the jump host (password / passphrase / OTP) |
| **OpenSSH** | Holds the authenticated connection open in a background master process |
| **Assistant** | Runs `ssh cluster '<command>'`, which reuses that socket with no auth |

The private key stays at `~/.ssh/id_ed25519` and is never read, printed, or transmitted. No password
or OTP is ever typed into chat.

**Be clear-eyed about what this does grant.** While the master connection is alive, any command run
through `ssh cluster` executes as you on the cluster. The boundary is *"the assistant can act on the
cluster for as long as you leave the socket open"* — not *"the assistant can't do anything"*. You
control the lifetime:

```bash
ssh -O check cluster    # is it open?
ssh -O exit cluster     # close it now
```

`ControlPersist` below also closes it automatically after an idle period.

---

## Environment findings (2026-09-09)

| | |
|---|---|
| Git Bash ssh | **OpenSSH_10.5p1** — supports `ControlMaster`. **Use this one.** |
| Windows native ssh | OpenSSH_for_Windows_9.5p1 — **no `ControlMaster` support**, do not use |
| WSL | Ubuntu present (stopped) — viable fallback |
| Key | `~/.ssh/id_ed25519` already exists |
| ssh-agent | not running |
| rsync | **not installed locally** — use `git` to deploy and `scp` to retrieve |

**The critical gotcha:** the master connection must be started **from Git Bash**, because the control
socket is an MSYS2-emulated Unix socket. A master started in PowerShell or CMD with Windows-native
ssh is invisible to Git Bash, and the assistant's shell is Git Bash. Typing `! ssh -fN cluster` in
the assistant session is the reliable way — that runs in the right shell by construction.

---

## Rejected: a gitignored script with hardcoded credentials

Considered and rejected 2026-09-09. The idea was a script holding the password, gitignored and
blocked from assistant reads, that the assistant could execute to open the channel itself.

**The read block is not a security boundary.** The assistant *executes* the script, so its behaviour
is observable — `bash -x` alone prints the password. A deny rule on `Read(path)` does not stop `cat`,
`od`, `sed`, or `python -c "open(...)"`; enforcing it would mean enumerating every possible read
path. The general principle: **if a process the assistant runs can read the file, the assistant can
read the file.** It is a speed bump against accidental exposure, not a barrier against intent — and a
dangerous one precisely because it invites more trust than it earns.

It also trades down: plaintext password on disk, when key auth already exists and is strictly better.
Most university AUPs additionally prohibit storing cluster credentials in plaintext.

**It is also unnecessary.** `~/.ssh/id_ed25519` has **no passphrase** (verified 2026-09-09 via
`ssh-keygen -y -P ""`, which exposes nothing). So if that key is authorized on both hops, `ssh
cluster '<cmd>'` is already fully non-interactive and needs no script at all.

### Decision tree

| Situation | Action |
|---|---|
| Key auth works end-to-end | **Nothing to build.** `ssh cluster` is already non-interactive. |
| Jump host requires OTP / 2FA | No script can help — TOTP is time-based. Use `ControlMaster` + `ControlPersist`: one OTP covers 8h. |
| Password-only, key not installed | `ssh-copy-id jump` (and to the cluster). Fixes it properly, no stored secret. |

Single diagnostic — after the config below is in place, run `ssh cluster hostname`. Returns a
hostname: first row, done. Prompts: OTP is in play, use multiplexing.

---

## Setup

### 1. `~/.ssh/config`

```sshconfig
Host jump
    HostName <jump.host.edu>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519

Host cluster
    HostName <login-node.cluster.edu>
    User <your-username>
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump jump
    # Multiplexing: authenticate once, reuse for everything after.
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h-%p
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

`ProxyJump` handles the jump host natively — no `nc`/`ProxyCommand` gymnastics.

### 2. Fix key permissions

The key is currently world-readable (`-rw-r--r--`). Tighten it:

```bash
chmod 600 ~/.ssh/id_ed25519
```

### 3. Open the master connection — **only if `ssh cluster hostname` prompted**

If key auth already works end-to-end, skip this: nothing needs to stay open and the assistant can
run `ssh cluster` directly. This step exists for the OTP / password case.

In the assistant session, type:

```
! ssh -fN cluster
```

Enter the password / passphrase / OTP at the prompt. `-f` backgrounds it, `-N` runs no command. It
stays alive for `ControlPersist` (8h) after the last use.

If the OTP prompt misbehaves through the harness, run the same command in a **Git Bash** window
instead — the socket is shared either way.

### 4. Verify

```bash
ssh -O check cluster        # -> "Master running (pid=NNNN)"
ssh cluster 'hostname; sinfo -s | head'
```

Once that works the assistant can run cluster commands directly, and
[`cluster_probe.sh`](cluster_probe.sh) can be executed remotely rather than pasted by hand.

### 5. Reduce permission prompts (optional)

Each `ssh` call goes through the Bash tool and prompts by default. Allowlist `Bash(ssh cluster:*)` in
project settings — the `/fewer-permission-prompts` command or the `update-config` skill will do it.

---

## Deploying code

**Use git, not file copying.** The cluster needs only code and maps; checkpoints and data travel the
other way.

```bash
# once, on the cluster
git clone <repo-url> && cd MLs_from_Whitechapel && git checkout dev

# thereafter
ssh cluster 'cd ~/MLs_from_Whitechapel && git pull'
```

**Prerequisite — this depends on [05](05-study-data.md) item A5.** `uv.lock` is currently
**untracked**, and without it `uv sync --frozen` cannot reproduce the environment on the cluster. The
`dev` branch is also 1 commit ahead of `origin/dev` and carries ~20 commits `master` lacks. So:
commit `uv.lock`, push `dev`, *then* clone.

## Retrieving results

`rsync` is not installed locally, so use `scp`. Pull only what is needed — with 02-C2's
best-checkpoint tracking that is one ~11 MB file per run, not the ~1.1 GB of periodic checkpoints:

```bash
scp 'cluster:~/MLs_from_Whitechapel/checkpoints/*/agent_best.pt' ./checkpoints/
scp -r cluster:~/MLs_from_Whitechapel/wandb/offline-run-* ./wandb/
```

Then `wandb sync wandb/offline-run-*` locally if the compute nodes had no outbound HTTPS.

## Session log

- 2026-09-09 — probed the local SSH environment; documented the ControlMaster approach. Setup not yet
  performed — awaiting the one interactive `ssh -fN cluster`.
- 2026-09-09 — rejected the hardcoded-credentials-in-a-blocked-script approach (read blocks are not
  a boundary against a process that executes the file; plaintext password is a downgrade from
  existing key auth). Verified the local key has no passphrase, so key auth is already
  non-interactive if authorized on both hops.
