# LABGENE-MOO Cloud Guide — from WSL laptop to a 64-core machine, once, reproducibly

**Goal.** Move the optimization pipeline to a rented 64-core Linux machine, accessed from your PC's terminal over SSH (Secure Shell), transferring code through GitHub, packaging everything in a Docker container so the exact same setup can later be redeployed on the UNIPI (University of Pisa) machine — with essentially zero performance penalty from the container.

**Architecture you are building:**

```
 Your PC (WSL)                GitHub                    AWS EC2 c7a.16xlarge (64 cores)
 ┌────────────────┐   git push   ┌──────────┐  git clone  ┌───────────────────────────────┐
 │ code + guide   │ ───────────► │ private  │ ──────────► │  ~/labgene-moo   (code, repo) │
 │                │              │ repo     │             │  ~/openmc_data   (6 GB, once) │
 │ ssh / rsync    │ ◄─────────────────────────────────────│  Docker: openmc-env container │
 └────────────────┘        results come back by rsync     └───────────────────────────────┘
```

Everything on the server is created by **three scripted steps** (`git clone`, `bash setup_data.sh`, `docker build`). That is the whole point: any future machine — UNIPI, another cloud, a colleague's workstation — becomes a 20-minute rebuild instead of a day of manual setup.

---

## 1. The two decisions, and why

### 1.1 Provider: AWS EC2 `c7a.16xlarge` (Amazon Web Services, Elastic Compute Cloud)

| Option | Cores you actually get | ~Price | Verdict |
|---|---|---|---|
| **AWS c7a.16xlarge** | **64 physical** AMD EPYC "Genoa" cores (SMT — Simultaneous Multi-Threading — is disabled on c7a, so every vCPU is a full core) | ~$3.28/h US, ~$3.6–3.9/h Frankfurt | **Chosen** |
| Google Cloud c3d-highcpu-60 | 60 vCPUs = only **30 physical** cores (SMT on) | ~$2.2–2.7/h | Fewer real cores per dollar for Monte Carlo; the $300 new-account trial credit is its one strong card (see 1.3) |
| Hetzner CCX (dedicated vCPU) | vCPU = thread | Raised **+113–169 %** in April + June 2026; price advantage gone. Also bills stopped servers until you *delete* them | Dropped |
| Azure F/HB series | comparable | comparable | No advantage; student subscriptions are vCPU-capped |

Why c7a.16xlarge specifically wins for **this** workload:

1. **OpenMC (Open source Monte Carlo particle transport code) is CPU-bound and memory-bandwidth-hungry; physical cores are what count.** SMT sibling threads add only ~10–25 % to Monte Carlo throughput. c7a is the rare instance family where 64 vCPUs = 64 real cores.
2. **"Configure once, keep the machine, pay ~nothing between sessions"** — exactly what you asked for — is native on AWS: a **stopped** instance costs only its disk (~$8–10/month for 100 GB), and Linux instances bill **per second** while running. You stop it after each session and start it again next week, fully configured.
3. 128 GiB RAM is far more than the depletion runs need — no memory risk.
4. The skills (SSH keys, security groups, per-second billing) are the industry standard you will meet everywhere later.

**Cost expectation** (verify with your own cloud smoke test, step 6): a 72-evaluation campaign at the raised fidelity should land around **4–12 hours ≈ $15–45**, plus a few dollars for calibration runs and ~$8–10/month of disk while the machine sleeps between sessions. Terminate the instance when the thesis compute is finished and even that stops.

**One mandatory early step:** new AWS accounts start with a small vCPU (virtual Central Processing Unit) quota — often 5–8 — for the "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances" family. You must request an increase to **64** *before* you can launch this instance (step 3.2). Do it on day one; approval takes minutes to ~2 days.

### 1.2 Package: yes, Docker — the performance penalty is negligible

Your worry was that containerizing would penalize the parallel run. It will not, and here is the reason: **a Docker container is not a virtual machine.** There is no hypervisor and no emulation layer; a container is an ordinary Linux process running directly on the host kernel, merely isolated by namespaces (its own filesystem/network view) and cgroups (resource accounting). Your OpenMC binary executes the same machine instructions on the same cores at the same clock as it would "bare". For a CPU-bound OpenMP (Open Multi-Processing) workload, measured overhead is ~0–2 %.

The only places containers *can* cost performance are I/O through Docker's layered filesystem and container start-up time — and this setup avoids both: all heavy files (your repo working dir and the 6 GB data library) are **bind mounts**, i.e. direct host-filesystem paths, and the container is started once per session, not per evaluation.

Two honest nuances, so you can defend the choice:

- The conda-forge OpenMC binary is a **generic x86-64 build**. Compiling OpenMC from source with `-march=native` + tuned flags could buy roughly another ~10 %. You are deliberately *not* doing that, because using the **identical binary** to your validated WSL environment keeps cloud results directly comparable to your baseline (pin-cell k∞ 1.296, assembly k∞ 1.288, …) — worth far more to the thesis than 10 % speed. Note it as future work.
- MPI (Message Passing Interface) adds **nothing** on a single node for OpenMC — OpenMP shared-memory threading is the efficient intra-node mode, and it duplicates no memory. MPI only matters if you ever spread one run across *multiple* machines. So: one big node + `OMP_NUM_THREADS` is the correct architecture here.

For UNIPI, where you may not get root/Docker rights: the image converts to **Apptainer** (the HPC — High Performance Computing — container standard, formerly Singularity) in one command, or you simply fall back to `mamba env create -f environment.yml`. Both paths are in step 11.

### 1.3 Budget alternative (optional)

If you want to try to make the compute nearly free: Google Cloud's new-account **$300 / 90-day trial credit** can cover the whole campaign on a `c3d-highcpu-60`. The friction: trial accounts must be upgraded to full (credits are kept) and a vCPU quota increase requested, and you get 30 physical cores instead of 64, so runs take ~2× longer. Everything in this guide except step 3 transfers unchanged — Docker doesn't care whose machine it is. This guide proceeds with AWS.

---

## 2. Part A — one-time setup on your PC (WSL)

### 2.1 Create an SSH key (if you don't already have one)

SSH (Secure Shell) authenticates with a key **pair**: the private key never leaves your PC; the public key is handed to servers (AWS) and services (GitHub).

```bash
ssh-keygen -t ed25519 -C "diogo-wsl"
#   -t ed25519   key type: modern elliptic-curve algorithm (short, fast, strong)
#   -C "..."     a comment label embedded in the public key, to recognise it later
# Press Enter to accept the default path (~/.ssh/id_ed25519); a passphrase is optional.

cat ~/.ssh/id_ed25519.pub     # this PUBLIC half is what you paste into websites
```

### 2.2 Put the project on GitHub

On github.com: **New repository** → name `labgene-moo` → **Private** → create (no README; you are pushing an existing folder). Then add your PC's key: GitHub → Settings → *SSH and GPG keys* → *New SSH key* → paste the `.pub` content.

Now in WSL, in the folder holding the project files:

```bash
cd ~/thesis_opt        # wherever reactor_model.py etc. live — adjust

# Tell git which generated/heavy files must NEVER be committed:
cat > .gitignore << 'EOF'
openmc_runs/
out*/
run_assembly_ref/
run_core_ref*/
run_core_refl_*/
*.h5
*.png
*.log
__pycache__/
*.pyc
.ipynb_checkpoints/
EOF
# (cat > file << 'EOF' ... EOF  writes everything between the markers into the
#  file — a "heredoc". The quotes around 'EOF' prevent variable expansion.)

git init -b main                      # -b main: name the initial branch "main"
git add reactor_model.py reactor_optimization.py openmc_evaluator.py \
        run_optimization.py measure_leakage_target.py sweep_ktarget.py \
        environment.yml environment.cloud.yml requirements.txt \
        Dockerfile setup_data.sh .gitignore CLOUD_GUIDE.md
git commit -m "LABGENE MOO pipeline, cloud-ready"
git remote add origin git@github.com:YOUR_USER/labgene-moo.git
git push -u origin main               # -u: remember origin/main as the default
                                      #     upstream, so future pushes are just `git push`
```

> Use the **cloud edition** `run_optimization.py` delivered with this guide (it replaces the old one; section 7 of this file and its own docstring explain the three differences). `environment.yml` is the portable export you already prepared for Prof. Giusti. The small result JSONs are *not* ignored on purpose — committing a finished checkpoint is nice provenance; just never `git add` the `openmc_runs/` scratch.

---

## 3. Part B — get the machine

### 3.1 AWS account

Create one at aws.amazon.com (email, credit card, phone verification). Sign in to the **Console**, and set the region (top-right selector) to **Europe (Frankfurt) `eu-central-1`** — close to you and one of the regions that definitely offers c7a. (Stockholm `eu-north-1` is usually a touch cheaper if you prefer.)

Set a billing guard immediately: Console → *Billing* → *Budgets* → create a monthly budget (e.g. $80) with an email alert. This is your safety net against a forgotten running instance.

### 3.2 Request the vCPU quota (do this first — it can take up to ~2 days)

Console → search **Service Quotas** → *AWS services* → **Amazon EC2** → find **"Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances"** → *Request increase at account level* → new value **64**. In the use-case box, one honest sentence works: *"MSc thesis nuclear-engineering Monte Carlo simulations (OpenMC); single c7a.16xlarge instance, intermittent use."* Quotas are **per region** — request it in the same region you will launch in.

### 3.3 Import your SSH key and launch

- EC2 → *Key pairs* → **Import key pair** → paste the content of `~/.ssh/id_ed25519.pub` → name it `diogo-wsl`.
- EC2 → **Launch instance**:
  - **Name:** `labgene-64core`
  - **AMI (Amazon Machine Image):** Ubuntu Server 24.04 LTS, 64-bit (x86). x86 matters — your `environment.yml` was solved for `linux-64`, not ARM.
  - **Instance type:** `c7a.16xlarge`
  - **Key pair:** `diogo-wsl`
  - **Network settings:** create a security group allowing **SSH** with source **My IP** (a firewall rule: port 22 only, only from your current address).
  - **Storage:** **100 GiB gp3** (data 6 GB + Docker image ~5 GB + conda + run scratch, with headroom).
  - Launch. Copy the **Public IPv4 address** from the instance page.

### 3.4 Connect

```bash
ssh ubuntu@<PUBLIC_IP>
# "ubuntu" is the default user on Ubuntu AMIs. Your imported key is used
# automatically from ~/.ssh/id_ed25519. Answer "yes" to the first-connection
# host-fingerprint question.
```

You are now in a terminal on the 64-core machine. `nproc` (print the number of processing units) should answer `64`.

---

## 4. Part C — base setup on the server (once)

```bash
sudo apt-get update && sudo apt-get upgrade -y
#   apt-get update    refresh the package index
#   upgrade -y        install available updates; -y = assume "yes"

sudo apt-get install -y git tmux htop rsync docker.io
#   git       clone your repository
#   tmux      terminal multiplexer: keeps your run alive if SSH disconnects
#   htop      live per-core CPU monitor (the satisfying 64-green-bars view)
#   rsync     efficient file synchronisation, for pulling results back
#   docker.io Docker engine from Ubuntu's own repository

sudo usermod -aG docker ubuntu
#   usermod -aG   -a append, -G to the group list: lets user "ubuntu" run
#                 docker without sudo. Takes effect on next login:
exit
```

…and `ssh ubuntu@<PUBLIC_IP>` back in. Check: `docker run --rm hello-world` should print a greeting.

---

## 5. Part D — code, data, image (the three scripted steps)

### 5.1 Give the *server* read access to your private repo

Cleanest pattern: a **deploy key** — an SSH key that lives on the server and is authorised for this one repository, read-only.

```bash
ssh-keygen -t ed25519 -C "labgene-ec2" -f ~/.ssh/id_ed25519 -N ""
#   -f  file path for the key    -N ""  empty passphrase (fine for a read-only
#                                        deploy key on a machine only you access)
cat ~/.ssh/id_ed25519.pub
```

Copy that line → GitHub → your repo → *Settings* → *Deploy keys* → **Add deploy key** (leave "write access" unticked). Then:

```bash
git clone git@github.com:YOUR_USER/labgene-moo.git
cd labgene-moo
```

### 5.2 Nuclear data (once per machine, ~10 min)

```bash
bash setup_data.sh
```

The script (commented line-by-line inside) downloads the official **ENDF/B-VII.1** (Evaluated Nuclear Data File, version B-VII.1) HDF5 cross-section library and the **thermal-spectrum PWR** (Pressurized Water Reactor) depletion chain from the openmc.org distribution links, extracts them into `~/openmc_data`, saves the chain as `chain_endfb71_pwr.xml` (the name your code already expects), and creates the `xslib` symlink the Dockerfile points at. Same library family, same chain as your WSL setup — consistency preserved.

### 5.3 Build the container image (once, ~5–10 min)

```bash
docker build -t labgene-openmc .
#   build .              use the Dockerfile in the current directory
#   -t labgene-openmc    tag (name) the resulting image
```

This recreates `openmc-env` from your `environment.yml` inside the image. If the pinned solve ever fails on a fresh machine, retry the build after editing the Dockerfile's COPY line to use `environment.cloud.yml` (the unpinned fallback — then re-verify the baseline numbers, as its header explains).

**Mental model of a run command** — you will use variations of this one line for everything:

```bash
docker run --rm -it \
  -e OMP_NUM_THREADS=64 \
  -v "$HOME/labgene-moo:/work" \
  -v "$HOME/openmc_data:/data:ro" \
  labgene-openmc  <command>
```

`--rm` deletes the container afterwards (image + your files survive); `-it` interactive terminal; `-e` sets the OpenMP (Open Multi-Processing) thread count inside; the two `-v` bind mounts expose your repo at `/work` (read-write — outputs land straight in the host folder, owned by the container) and the data at `/data` read-only (`:ro`). If output files ever show as root-owned on the host, reclaim them with `sudo chown -R $USER:$USER ~/labgene-moo` (`chown -R`: change owner, recursively).

---

## 6. Part E — smoke test on the cloud (mandatory, ~minutes, cents)

```bash
docker run --rm -it \
  -e OMP_NUM_THREADS=64 \
  -v "$HOME/labgene-moo:/work" -v "$HOME/openmc_data:/data:ro" \
  labgene-openmc \
  python run_optimization.py --smoke --out out_smoke
```

Expect the same qualitative behaviour as your WSL smoke test (flat hypervolume, single Pareto point — that is correct for 6 evaluations), just much faster. **Write down the wall-clock time**: it is your calibration for estimating the full-run cost on this machine. Watch `htop` in a second SSH session — all 64 bars should saturate during transport.

---

## 7. Part F — the physics gate (before spending money on the full run)

Two things stand between you and the overnight run, and both are **yours**, not the machine's:

1. **Route A vs. Route B for `refl_thick`** (reflector thickness). The pipeline currently ships as the hybrid in which `refl_thick` is physically inactive in real evaluations. Decide the route *before* the full run — a 72-evaluation Pareto front computed under the wrong leakage treatment is money and a week lost. If you choose Route B, `sweep_ktarget.py` runs beautifully on this machine (each core-model solve that took long on WSL is minutes here), but remember its warning: never combine `reflector=True` depletion with a `refl_thick`-dependent target.

2. **Measure the frozen K_TARGET on the frozen 32-assembly geometry** (the smoke value 1.085 came from the old geometry):

```bash
docker run --rm -it \
  -e OMP_NUM_THREADS=64 \
  -v "$HOME/labgene-moo:/work" -v "$HOME/openmc_data:/data:ro" \
  labgene-openmc \
  python measure_leakage_target.py | tee ktarget_$(date +%d%b).log
#   tee  print to screen AND save to a file at the same time
```

On 64 cores this is fast enough that you can *raise* the statistics constants at the top of the script (e.g. 5× the particles) for a tighter k_target — its uncertainty propagates directly into the EOC (End Of Cycle) crossing and hence every EFPD (Effective Full Power Days) value on the Pareto front. Paste the printed `K_TARGET` into `run_optimization.py` (or pass `--ktarget` every time — the flag always wins).

*(Note: measure_leakage_target.py's docstring still says "~21 assemblies" — a stale comment only; the code calls the current 32-assembly `make_core_model`.)*

---

## 8. Part G — the full run, in chunks, inside tmux

**Why tmux:** if your SSH connection drops (laptop sleeps, Wi-Fi hiccup), every process started in that session dies — including an 8-hour optimization. tmux (terminal multiplexer) keeps the session alive on the server; you *detach* and *re-attach* at will.

**Why chunks:** the checkpoint is written at the **end** of a session. Three chunks give you a durable save-point after ~each third, using your own `--resume` machinery exactly as designed ("run a batch, look at the HV curve, resume").

```bash
tmux new -s opt          # new session named "opt"   (-s: session name)
```

Inside tmux — session 1 (fresh: 24 DOE + 2×6 infill = 36 evaluations):

```bash
docker run --rm \
  -e OMP_NUM_THREADS=64 \
  -v "$HOME/labgene-moo:/work" -v "$HOME/openmc_data:/data:ro" \
  labgene-openmc \
  python run_optimization.py --ktarget <YOUR_MEASURED_VALUE> \
      --iters 2 --out out 2>&1 | tee -a out/run.log
#   2>&1        merge stderr into stdout so the log captures errors too
#   tee -a      -a append (don't overwrite the log across sessions)
```

Detach with **Ctrl-b then d** (tmux keeps it running). Re-attach anytime with `tmux attach -t opt` (`-t`: target session). Sessions 2 and 3 (run each after the previous finishes; check the HV plot between them):

```bash
docker run --rm \
  -e OMP_NUM_THREADS=64 \
  -v "$HOME/labgene-moo:/work" -v "$HOME/openmc_data:/data:ro" \
  labgene-openmc \
  python run_optimization.py --ktarget <SAME_VALUE> \
      --resume out/optimization_checkpoint.json --iters 3 --out out \
      2>&1 | tee -a out/run.log
```

36 + 18 + 18 = **72 evaluations**, matching your validated budget. The new driver prints (and the checkpoint records) the fidelity and thread count; if a later session's flags disagree with the checkpoint, it warns you — same guard philosophy as k_target. If, per your convergence plan, the HV is still climbing at 72 on the *real* OpenMC landscape, more `--resume --iters N` sessions extend the budget without restarting anything.

**Monitoring from a second terminal:** `htop` for cores; `tail -f ~/labgene-moo/out/run.log` (follow the log live); `ls ~/labgene-moo/openmc_runs/` to see `case_NNNN` folders appear; `df -h` (disk free, human units) if you ever worry about space.

---

## 9. Part H — pull the results to your PC

From **WSL** (not the server):

```bash
rsync -avz --progress ubuntu@<PUBLIC_IP>:~/labgene-moo/out/ ~/thesis_opt/out_cloud/
#   -a  archive: preserve times/permissions, recurse into directories
#   -v  verbose  -z  compress in transit  --progress  live per-file progress
```

That folder contains `optimization_results.json`, `optimization_checkpoint.json`, `optimization_openmc.png`, and `run.log` — everything the thesis plots and the resume feature need. The multi-gigabyte `openmc_runs/` scratch (statepoints, depletion HDF5s) normally stays on the server; rsync a specific `case_NNNN` only if you want to autopsy one design. Commit the finished checkpoint from WSL if you want provenance in git.

---

## 10. Part I — pause, restart, terminate (the money mechanics)

- **Between sessions:** EC2 console → instance → *Instance state* → **Stop**. A stopped instance bills **only the 100 GB disk** (~$8–10/month); CPU billing ends. Everything on disk — Docker image, data, repo, checkpoints — persists exactly as you left it.
- **Next session:** *Instance state* → **Start**. ⚠ The **public IP changes** on every stop/start cycle — read the new one from the console before `ssh`/`rsync`. (An "Elastic IP" would pin it, but it costs while the instance is stopped; just reading the new IP is free.)
- **Thesis compute finished:** rsync everything you want, `git push` any last commits, then *Instance state* → **Terminate**. This deletes machine *and* disk; billing goes to exactly zero. Recreating later = steps 3.3→5.3, ~30 minutes — which is precisely the reproducibility you built.

---

## 11. Part J — redeploying the identical package elsewhere (UNIPI or any machine)

The repo *is* the package. Three paths, by how much freedom the target machine gives you:

**(a) Docker allowed (root or docker group):** identical to this guide —
`git clone` → `bash setup_data.sh` → `docker build -t labgene-openmc .` → same `docker run` lines. Different CPU count? Just change `-e OMP_NUM_THREADS=<nproc>` and `--threads`.

**(b) No root, but Apptainer available (typical university HPC):** convert the image you already trust, without rebuilding:

```bash
# on the AWS box (or any machine with the image):
docker save labgene-openmc -o labgene-openmc.tar     # serialize image to a tar
# copy the ~4–5 GB tar to UNIPI (scp/rsync), then on UNIPI:
apptainer build labgene.sif docker-archive://labgene-openmc.tar
apptainer exec --bind $HOME/labgene-moo:/work,$HOME/openmc_data:/data \
    --env OMP_NUM_THREADS=$(nproc) labgene.sif \
    conda run --no-capture-output -n openmc-env \
    python /work/run_optimization.py --smoke --out /work/out_smoke
#   --bind  Apptainer's equivalent of docker -v (comma-separated pairs)
```

**(c) Neither — plain conda (your original plan with Prof. Giusti):**
`git clone` → `bash setup_data.sh` → `mamba env create -f environment.yml` → export `OPENMC_CROSS_SECTIONS=$HOME/openmc_data/xslib/cross_sections.xml` and `OPENMC_CHAIN_FILE=$HOME/openmc_data/chain_endfb71_pwr.xml` in `~/.bashrc` → run the same Python commands without the `docker run` wrapper. Same binaries from conda-forge, so results stay comparable; you lose only the OS-level encapsulation.

Whichever path: **run `--smoke` first on every new machine.** Always.

---

## 12. Troubleshooting

| Symptom | Cause → fix |
|---|---|
| Launch fails: *VcpuLimitExceeded* / "requested more vCPU capacity than your current limit" | Quota not yet granted (step 3.2). Check Service Quotas → request history; wait/nudge support. |
| `ssh: Permission denied (publickey)` | Wrong user (must be `ubuntu`), wrong key, or WSL key permissions: `chmod 600 ~/.ssh/id_ed25519` (owner read/write only — SSH refuses looser). |
| `docker: permission denied … docker.sock` | Group change not active — log out and back in after `usermod -aG docker`. |
| Container error: cross sections / chain not found | A `-v` mount missing or `setup_data.sh` not run: check `ls ~/openmc_data/xslib/cross_sections.xml` on the **host**, and that both `-v` flags are in the command. |
| `mamba env create` fails on a pin | Use `environment.cloud.yml` (fallback), then re-verify baseline k∞ before real runs. |
| Run died when laptop slept | It was started outside tmux. Restart inside `tmux new -s opt`; recover completed work via `--resume out/optimization_checkpoint.json`. |
| Output files owned by root on host | `sudo chown -R $USER:$USER ~/labgene-moo` |
| Disk filling up | `df -h`; delete old scratch: `rm -rf ~/labgene-moo/openmc_runs/case_00*` (checkpoint JSON keeps every result regardless). |
| Can't reconnect after Stop/Start | The public IP changed — read the new one in the EC2 console. |

---

## 13. One-page cost picture

| Item | Cost |
|---|---|
| c7a.16xlarge running (Frankfurt, on-demand, per-second billing) | ~$3.6–3.9 / hour |
| Cloud smoke test + K_TARGET measurement | ~$1–3 |
| Full 72-eval campaign at 20 000 × 80 fidelity | ~4–12 h → **~$15–45** (calibrate with your smoke timing) |
| Disk while stopped (100 GB gp3) | ~$8–10 / month |
| After Terminate | $0 |

Advanced later option: **Spot instances** (~60–70 % off, interruptible). Your chunked `--resume` pattern makes interruptions survivable, but learn the on-demand workflow first.
