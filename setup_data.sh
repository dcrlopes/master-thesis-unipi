#!/usr/bin/env bash
# ============================================================================
# setup_data.sh -- download the nuclear data ONCE per machine
# ----------------------------------------------------------------------------
# Fetches the two official data files this project needs (URLs from
# https://openmc.org/data/):
#   1. ENDF/B-VII.1 HDF5 cross-section library (Evaluated Nuclear Data File,
#      version B-VII.1) -- ~2 GB download, ~6 GB extracted
#   2. ENDF/B-VII.1 THERMAL-spectrum depletion chain (the PWR -- Pressurized
#      Water Reactor -- chain, saved under the name your code expects:
#      chain_endfb71_pwr.xml)
#
# It then creates a stable symlink  <data_dir>/xslib -> <extracted folder>
# so the Dockerfile can hardcode /data/xslib/cross_sections.xml without ever
# caring what the tarball's top-level folder is called.
#
# Usage:   bash setup_data.sh [target_dir]      (default: ~/openmc_data)
# Re-run:  safe; wget -c resumes partial downloads instead of restarting.
# ============================================================================
set -euo pipefail
#   -e  exit immediately if any command fails
#   -u  treat use of an undefined variable as an error
#   -o pipefail  a pipeline fails if ANY command in it fails (not just the last)

DATA_DIR="${1:-$HOME/openmc_data}"   # first argument, or ~/openmc_data if none
mkdir -p "$DATA_DIR"                 # -p: create parents, no error if it exists
cd "$DATA_DIR"

XS_URL="https://anl.box.com/shared/static/9igk353zpy8fn9ttvtrqgzvw1vtejoz6.xz"
CHAIN_URL="https://anl.box.com/shared/static/os1u896bwsbopurpgas72bi6aij2zzdc.xml"

echo ">> [1/3] downloading ENDF/B-VII.1 HDF5 cross sections (~2 GB) ..."
wget -c -O endfb-vii.1-hdf5.tar.xz "$XS_URL"
#   -c  continue a partially downloaded file (resume-safe)
#   -O  write to this exact filename instead of the server's name

echo ">> [2/3] extracting (~6 GB, a few minutes) ..."
tar -xJf endfb-vii.1-hdf5.tar.xz
#   -x  extract   -J  the archive is xz-compressed   -f  archive filename

echo ">> [3/3] downloading ENDF/B-VII.1 thermal-spectrum (PWR) depletion chain ..."
wget -c -O chain_endfb71_pwr.xml "$CHAIN_URL"

# Locate cross_sections.xml inside whatever folder the tarball produced and
# point a RELATIVE symlink 'xslib' at that folder. Relative matters: the link
# still resolves when the whole directory is bind-mounted at /data in Docker.
XS_FILE="$(find "$DATA_DIR" -maxdepth 3 -type f -name cross_sections.xml \
           -not -path '*xslib*' | head -n 1)"
#   -maxdepth 3       do not descend deeper than 3 levels
#   -type f           regular files only
#   -name             match this filename
#   -not -path        ignore anything already behind the xslib symlink
#   head -n 1         keep only the first match
if [ -z "$XS_FILE" ]; then
    echo "ERROR: cross_sections.xml not found after extraction." >&2
    exit 1
fi
XS_SUBDIR="$(basename "$(dirname "$XS_FILE")")"
ln -sfn "$XS_SUBDIR" "$DATA_DIR/xslib"
#   -s  symbolic link   -f  replace an existing link   -n  treat existing
#       link-to-directory as a file (do not descend into it)

echo ""
echo "Done. Data layout:"
echo "  OPENMC_CROSS_SECTIONS = $DATA_DIR/xslib/cross_sections.xml"
echo "  OPENMC_CHAIN_FILE     = $DATA_DIR/chain_endfb71_pwr.xml"
echo "Sanity check sizes:"
du -sh "$DATA_DIR/$XS_SUBDIR" "$DATA_DIR/chain_endfb71_pwr.xml"
#   du -s  summarize total   -h  human-readable units
