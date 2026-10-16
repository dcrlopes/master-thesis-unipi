# Running the code on the Mac mini (Apple silicon)

Written on 16 October 2026 for a timing test of Campaign 10 on a Mac mini
2026. Nothing in the physics changes; what changes is the environment, the
thread count and the absence of the Linux tools the queue scripts use.

## What differs from wks720

| Item | wks720 | Mac mini |
|---|---|---|
| CPU threads | 64 (`--threads 64`) | the machine's cores: `sysctl -n hw.ncpu` |
| OpenMC build | conda-forge linux-64, MPI (Open MPI with CUDA hooks) | conda-forge osx-arm64 or linux-aarch64, no MPI needed |
| `environment.yml` | linux-64 build strings, will not resolve elsewhere | use `environment-portable.yml` |
| Nuclear data | `~/openmc_data/endfb-vii.1-hdf5` (6 GB) and the depletion chain | copy the same files from wks720, so that the results are comparable |
| Queue, cron, watchdog, machine monitor | used | not needed for a timing test; `run_c10.sh` still works with `ALLOW_HOST=1` |
| Sleep | never | macOS sleeps an idle machine: run under `caffeinate -i` |

## Choice: native conda or Docker

Native conda (osx-arm64) is the faster of the two and the simpler to watch.
Docker Desktop on Apple silicon runs an arm64 Linux container at near-native
speed, but the repository's image is built from the linux-64 `environment.yml`
and would run under Rosetta emulation, several times slower, so for Docker the
image must be rebuilt for `linux/arm64` from `environment-portable.yml`.
Both recipes follow. Use one.

### A. Native conda (recommended)

```bash
# 1. Miniforge for Apple silicon, once
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3.sh -b && ~/miniforge3/bin/conda init zsh && exec zsh

# 2. the environment
git clone git@github.com:dcrlopes/master-thesis-unipi.git ~/master-thesis-unipi
cd ~/master-thesis-unipi
conda env create -n openmc-env -f environment-portable.yml
conda activate openmc-env
python -c "import openmc; print(openmc.__version__)"      # 0.15.3

# 3. the nuclear data, copied from wks720 (replace the paths by what
#    `echo $OPENMC_CROSS_SECTIONS $OPENMC_CHAIN_FILE` prints there)
mkdir -p ~/openmc_data
rsync -avz diogo@10.114.180.54:~/openmc_data/ ~/openmc_data/
export OPENMC_CROSS_SECTIONS=~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml
export OPENMC_CHAIN_FILE=~/openmc_data/chain_endfb71_pwr.xml
echo 'export OPENMC_CROSS_SECTIONS=~/openmc_data/endfb-vii.1-hdf5/cross_sections.xml' >> ~/.zshrc
echo 'export OPENMC_CHAIN_FILE=~/openmc_data/chain_endfb71_pwr.xml' >> ~/.zshrc
```

### B. Docker (arm64 image)

```bash
git clone git@github.com:dcrlopes/master-thesis-unipi.git ~/master-thesis-unipi
cd ~/master-thesis-unipi
# the Dockerfile copies environment.yml; point it at the portable one for this build
sed 's/environment.yml/environment-portable.yml/' Dockerfile > Dockerfile.arm64
docker build --platform linux/arm64 -f Dockerfile.arm64 -t labgene-openmc-arm64 .
rsync -avz diogo@10.114.180.54:~/openmc_data/ ~/openmc_data/
ln -sfn ~/openmc_data/endfb-vii.1-hdf5 ~/openmc_data/xslib     # the image expects /data/xslib
docker run --rm -it --platform linux/arm64 \
  -e OMP_NUM_THREADS=$(sysctl -n hw.ncpu) \
  -v "$HOME/master-thesis-unipi:/work" -v "$HOME/openmc_data:/data:ro" \
  labgene-openmc-arm64 bash
```

Inside the container the commands below are the same, without `caffeinate`
(Docker Desktop keeps running; set the Mac not to sleep in System Settings).

## The timing test

A full Campaign 10 is 58 evaluations. The question for the Mac is the time
per evaluation, which the smoke run and two real evaluations give in a few
hours. From `~/master-thesis-unipi` with the environment active:

```bash
# 1. the pure checks and the smoke run (about 20 to 40 min)
python test_core3d_hook.py && python test_c9_wiring.py | tail -1
ALLOW_HOST=1 THREADS=$(sysctl -n hw.ncpu) caffeinate -i bash run_c10.sh --smoke

# 2. two real evaluations of the design of experiments, no infill
ALLOW_HOST=1 caffeinate -i python -u run_optimization.py --out out_c10_mac --workdir openmc_runs_c10_mac \
  --ktarget-table ktarget_table_c8.json --k-basis core --k-max 1.166 --k-min 1.02 --f-max 1.65 --enr-max 16 \
  --ctrl-margin 1000 --objective-set c9 --efpd-req 1826 --boron-objective floor --boron-step 2000 --boron-top 3000 \
  --boron-ceiling 2763 --hump-noise 400 --cycle-layers 8 --threads $(sysctl -n hw.ncpu) \
  --n-init 2 --n-infill 6 --iters 0 --cycle-core3d 10000,100,50 --cycle-max-burnup 29.5 2>&1 | tee out_c10_mac.log

# 3. the times
python -c "
import json
for v in json.load(open('openmc_runs_c10_mac/eval_cache.json')).values():
    print(f\"total {v['t_eval_s']/60:6.1f} min | assembly depletion {v['t_deplete_s']/60:5.1f} | core solves {v['t_core_bol_s']/60:5.1f} | 3D {v.get('t_dep3d_s',0)/60:5.1f} min\")
"
grep "Total time in simulation\|Calculation Rate (active)" out_c10_mac.log | tail -4
```

On wks720 an evaluation takes 45 to 90 min: 15 to 30 min for the campaign
part and 35 to 60 min for the three-dimensional depletion, of which about
20 min of transport at 106 to 117 s per solve and the rest on one CPU.
The Mac will be slower on the transport (fewer cores) and faster on the
single-CPU part. The two numbers to compare are the "Calculation Rate
(active)" of OpenMC, particles per second, and the total minutes per
evaluation.

## If the full campaign is wanted on the Mac

With the seed now reaching the sampler, a fresh run on the Mac draws its own
design of experiments, the same for every launch there, but different from
the 22 designs that wks720 is using (those came from five unseeded draws).
To compare the two machines on the same designs, copy wks720's cache and
use it as the design of experiments:

```bash
scp diogo@10.114.180.54:~/master-thesis-unipi/openmc_runs_c10/eval_cache.json ~/eval_cache_wks720.json
# ... then --doe-from-cache ~/eval_cache_wks720.json in place of --n-init 2 --iters 0, with --iters 6
```

That skips the design of experiments on the Mac (the 22 evaluations are
taken as they are) and runs only the 36 infill evaluations, so it measures
the loop with the proxy, not the design of experiments.
