#!/usr/bin/env bash
# preflight.sh -- environment gate. Run before ANY OpenMC launch on wks720.
# Exits non-zero if the machine, environment, branch or data files are wrong.
set -u
FAIL=0
chk() { if eval "$2" > /dev/null 2>&1; then echo "  OK    $1"; else echo "  FAIL  $1"; FAIL=1; fi; }

echo "PREFLIGHT $(date -Is)"
chk "host is wks720"                 '[ "$(hostname -s)" = "wks720" ]'
chk "conda env is openmc-env"        '[ "${CONDA_DEFAULT_ENV:-none}" = "openmc-env" ]'
chk "cwd is the working clone"       '[ "$(basename "$PWD")" = "master-thesis-unipi" ]'
chk "git branch is main"             '[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ]'
chk "no uncommitted edits to tracked files" '[ -z "$(git status --porcelain --untracked-files=no)" ]'
chk "cross-section file present"     '[ -f "$HOME/openmc_data/xslib/cross_sections.xml" ]'
chk "depletion chain present"        '[ -f "$HOME/openmc_data/chain_endfb71_pwr.xml" ]'
chk "OPENMC_CHAIN_FILE is set"       '[ -n "${OPENMC_CHAIN_FILE:-}" ]'
chk "OpenMC version is 0.15.3"       'python -c "import openmc,sys; sys.exit(0 if openmc.__version__==\"0.15.3\" else 1)"'
chk "no OpenMC job already running"  '! pgrep -f "python[0-9.]* .*(run_optimization|confirm3d|sweep_ktarget|validate_ktarget)" > /dev/null'
chk "at least 20 GB free on cwd"     '[ "$(df --output=avail -BG . | tail -1 | tr -dc 0-9)" -ge 20 ]'
echo "  INFO  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), untracked paths: $(git status --porcelain 2>/dev/null | grep -c '^??')"
echo "  INFO  cores reported by nproc: $(nproc)"
echo "  INFO  OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}"
if [ "$FAIL" -ne 0 ]; then echo "PREFLIGHT FAILED. Nothing launched."; exit 1; fi
echo "PREFLIGHT OK"
