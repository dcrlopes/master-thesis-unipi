# ============================================================================
# LABGENE-MOO reproducible runtime
# ----------------------------------------------------------------------------
# Recreates the SAME conda environment ("openmc-env") that was validated on
# your WSL (Windows Subsystem for Linux) laptop, from the portable
# environment.yml already generated for Prof. Giusti. Same conda-forge
# binaries = results directly comparable with the laptop runs.
#
# The ~6 GB nuclear data library is deliberately NOT baked into the image
# (it would bloat every rebuild/transfer). Instead it lives on the host at
# ~/openmc_data and is bind-mounted read-only at /data. The two environment
# variables below therefore point at the /data mount:
#   /data/xslib -> stable symlink created by setup_data.sh, so this Dockerfile
#                  never needs to know the tarball's extracted folder name.
#
# Build:  docker build -t labgene-openmc .
#           -t labgene-openmc   name ("tag") the image so you can refer to it
#
# Run:    docker run --rm -it \
#           -e OMP_NUM_THREADS=64 \
#           -v "$HOME/labgene-moo:/work" \
#           -v "$HOME/openmc_data:/data:ro" \
#           labgene-openmc \
#           python run_optimization.py --smoke --out out_smoke
#
#           --rm       delete the container when the command exits (the image
#                      and your bind-mounted files are untouched)
#           -it        interactive terminal (drop it for unattended runs)
#           -e VAR=x   set an environment variable inside the container
#           -v A:B     bind-mount host path A at container path B
#              :ro     mount read-only (the nuclear data is never written to)
# ============================================================================

# Miniforge = minimal conda + the fast mamba solver, defaulting to the
# conda-forge channel (exactly where the validated openmc build comes from).
# After the first successful build you can pin this to an immutable digest
# (docker images --digests) for bit-for-bit rebuilds.
FROM condaforge/miniforge3:latest

# ---- recreate the validated environment ------------------------------------
# environment.yml is the portable export from the WSL machine (machine-specific
# paths already stripped). `mamba env create` is the fast solver you already
# use; `-n openmc-env` fixes the name; `-f` points at the spec file.
# `mamba clean -afy` (-a all caches, -f force, -y yes) shrinks the image.
COPY environment.yml /tmp/environment.yml
RUN mamba env create -n openmc-env -f /tmp/environment.yml && \
    mamba clean -afy

# ---- nuclear data locations (provided by the /data bind mount) --------------
ENV OPENMC_CROSS_SECTIONS=/data/xslib/cross_sections.xml \
    OPENMC_CHAIN_FILE=/data/chain_endfb71_pwr.xml

# ---- default working directory = your repo bind mount ------------------------
WORKDIR /work

# Every command runs inside the activated openmc-env.
# `conda run --no-capture-output` streams stdout/stderr live (essential for
# watching the optimizer log and for `tee`).
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "openmc-env"]
CMD ["bash"]
