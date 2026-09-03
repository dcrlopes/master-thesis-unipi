"""
make_architecture_figure.py -- macro architecture of the simulation code,
drawn with Graphviz from a single specification kept in this file.

The same specification prints the per-block tables, so the figure, the
appendix tables and the README cannot drift apart.

This script needs NO OpenMC and NO conda environment. It runs on any
machine with Python 3.8+ and the Graphviz binary `dot` (laptop, WSL2,
wks720). Do NOT run it inside the `lab` Docker image, which has no `dot`.

Usage (from the repository root, ~/master-thesis-unipi):

    python make_architecture_figure.py                  # writes code_architecture.pdf and .png
    python make_architecture_figure.py --check          # only verify the spec against the repo
    python make_architecture_figure.py --table latex    # print the LaTeX table rows
    python make_architecture_figure.py --table md       # print a Markdown table for README.md
    python make_architecture_figure.py --out images     # write into another directory

Flags:
    --check       verify that every listed script exists and report the
                  unclassified .py files, then exit without drawing
    --table       print the block tables in `latex` or `md` format and exit
    --out DIR     output directory (default: current directory)
    --dpi N       resolution of the PNG preview (default 200)

Copy the PDF to the dissertation repository as images/code_architecture.pdf.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 0. Machine check, printed before anything else
# --------------------------------------------------------------------------

def machine_check():
    dot = shutil.which("dot")
    print(f"host     : {platform.node()}")
    print(f"python   : {sys.version.split()[0]}")
    print(f"cwd      : {Path.cwd()}")
    if dot is None:
        print("ABORT: Graphviz `dot` not found. Install with "
              "`sudo apt install graphviz` (Ubuntu/WSL2) or "
              "`conda install -c conda-forge graphviz`.")
        sys.exit(1)
    ver = subprocess.run(["dot", "-V"], capture_output=True, text=True)
    print(f"graphviz : {(ver.stderr or ver.stdout).strip()}")
    if os.environ.get("CONDA_DEFAULT_ENV"):
        print(f"conda env: {os.environ['CONDA_DEFAULT_ENV']} (not required)")
    print()


# --------------------------------------------------------------------------
# 1. Specification: nine blocks, the scripts that compute something,
#    and the artefacts that cross each edge.
#    Roles are one line. Reads/Writes name files, not variables.
# --------------------------------------------------------------------------

BLOCKS = {
    "env": {
        "title": "Environment and nuclear data",
        "kind": "infra",
        "scripts": [],
        "note": ["conda openmc-env, OpenMC 0.15.3",
                 "ENDF/B-VII.1, chain_endfb71_pwr.xml"],
    },
    "builders": {
        "title": "Model builders",
        "kind": "physics",
        "scripts": [
            ("core_geometry.py", "Vessel envelope, geometry margin, end-of-cycle crossing, bilinear interpolation",
             "--", "--"),
            ("leu_policy.py", "Enrichment cap and search-box constants shared by all modules",
             "--", "--"),
            ("reactor_model.py", "Materials, pin, guide tube, 17x17 assembly and 32-assembly core builders",
             "--", "OpenMC model objects"),
            ("zoning.py", "Ring map, centre/middle/periphery multipliers, control-bank positions, core BOL solve",
             "--", "OpenMC model objects"),
            ("hardware3d.py", "Finite-height core with the assembly hardware stack",
             "--", "OpenMC model objects"),
        ],
    },
    "ktarget": {
        "title": "Reactivity target (Route B)",
        "kind": "physics",
        "scripts": [
            ("sweep_ktarget.py", "Tabulates k_target(pitch, t_refl) from assembly and core solves",
             "--", "ktarget_table.json"),
            ("axial_leakage_study.py", "Axial leakage factor L_ax = k_2D / k_3D on archived designs",
             "optimization_checkpoint.json", "axial_<absorber>.json"),
            ("validate_ktarget_burnup.py", "Burnup dependence of the target on front designs",
             "optimization_checkpoint.json, ktarget_table*.json", "summary.json, summary_table.tex"),
        ],
    },
    "evaluator": {
        "title": "Design evaluator",
        "kind": "physics",
        "scripts": [
            ("openmc_evaluator.py", "Design vector to EFPD, F_dH and constraint vector g (assembly depletion, core BOL solve, control solve)",
             "ktarget_table*.json", "depletion_results.h5 per case"),
        ],
    },
    "ctrl": {
        "title": "Control-rod worth and derived bounds",
        "kind": "physics",
        "scripts": [
            ("rod_bank_worth.py", "Bank worths and controllability screen of archived designs",
             "optimization_checkpoint.json", "banks_<absorber>.json"),
            ("rod_worth_ladder.py", "Rod-worth ladder and rodded peaking of the zoned core",
             "optimization_checkpoint.json", "ladder_idx<i>_<absorber>.json"),
            ("derive_kmax.py", "Derived k_max ceiling from the minimum bank worth and the margin",
             "banks_<absorber>.json", "ctrl_kmax_<group>.json"),
            ("verify_rod_pitch.py", "Guide-tube clipping study versus pitch",
             "case directories", "fig_rod_pitch_*.pdf"),
        ],
    },
    "optimiser": {
        "title": "Surrogate-assisted optimiser and driver",
        "kind": "optim",
        "scripts": [
            ("reactor_optimization.py", "Design space, GP surrogates, NSGA-II acquisition, active-learning loop, checkpoint",
             "--", "optimization_checkpoint.json, optimization_results.json"),
            ("run_optimization.py", "Command-line driver: DOE, blocks, resume, provenance",
             "ktarget_table*.json, optimization_checkpoint.json", "optimization_checkpoint.json, run.log"),
            ("nsga_sensitivity.py", "Sensitivity of the acquisition step to NSGA-II settings and seeds",
             "optimization_checkpoint.json", "nsga_sensitivity.json, .csv, .tex"),
            ("inspect_front.py", "Census of a live checkpoint: front, screens, per-block diversity",
             "optimization_checkpoint.json", "<prefix>_all.csv, <prefix>_front.csv"),
        ],
    },
    "confirm": {
        "title": "Candidate confirmation and rescoring",
        "kind": "physics",
        "scripts": [
            ("confirm3d.py", "Three-dimensional confirmation of candidates with the hardware stack",
             "optimization_checkpoint.json", "runs.json, summary.json"),
            ("boron_worth.py", "Boron worth and hold-down share across rod states and concentrations",
             "optimization_checkpoint.json", "runs.json, summary.json, boron_table.tex"),
            ("tier1_coefficients.py", "Reactivity coefficients of a candidate",
             "optimization_checkpoint.json", "tier1_idx<i>.json"),
            ("rescore_archive_core.py", "Core-basis peaking of an assembly-basis archive",
             "optimization_checkpoint.json", "runs.json, core_rescore.csv"),
            ("rescore_kmax_core.py", "Re-screens a finished campaign on the core reactivity limits",
             "optimization_checkpoint.json", "<label>_kbasis_summary.json"),
            ("rescore_zoned_core.py", "Transfer test of the zoned loading on an archive",
             "optimization_checkpoint.json", "runs.json, zoned_rescore.csv, transfer_summary.json"),
            ("refine_zoning.py", "Refinement of the ring multipliers of a champion",
             "optimization_checkpoint.json", "runs.json"),
            ("confirm_zoned_champion.py", "Depletion of the zoned champion at full fidelity",
             "optimization_checkpoint.json, ktarget_table*.json", "confirm_idx<i>.json"),
            ("confirm_c5_champion.py", "Feasibility of the Campaign 5 champion at full fidelity",
             "optimization_checkpoint.json", "runs.json, summary.json"),
            ("rank_front_peaking.py", "Minimum-F_dH design beyond Monte Carlo noise (sequential screening)",
             "optimization_checkpoint.json", "runs.json, ranking.csv"),
            ("peaking_noise_test.py", "Seed replicates of F_dH on front designs",
             "optimization_checkpoint.json", "noise_summary.csv"),
            ("validate_core_proxy.py", "Assembly-to-core peaking proxy validation",
             "optimization_checkpoint.json, ktarget_table*.json", "runs.json, core_validation.csv"),
            ("test_gd_pins.py", "Gadolinium pin-count test on a champion (assembly and core)",
             "optimization_checkpoint.json", "runs.json, summary.json"),
        ],
    },
    "post": {
        "title": "Post-processing",
        "kind": "infra",
        "scripts": [
            ("analyze_results.py", "Pareto front, hypervolume history, design-space diagnostics",
             "optimization_results.json, optimization_checkpoint.json", "fig1..fig5 PNG"),
            ("salvage_k_histories.py", "Corrects archived depletion histories for the restart defect",
             "optimization_checkpoint.json, depletion_results.h5", "case_<i>_corrected.csv, salvage_summary.csv"),
            ("variable_importance.py", "Permutation importance of the design variables on the archive",
             "optimization_checkpoint.json", "stdout"),
        ],
    },
    "thesis": {
        "title": "Dissertation repository",
        "kind": "infra",
        "scripts": [],
        "note": ["images/*.pdf, table fragments",
                 "master_numbers.json"],
    },
}

# Edges: (source, target, label, style). Unlabelled edges are Python imports.
EDGES = [
    ("env", "builders", "OpenMC 0.15.3, cross_sections.xml", "solid"),
    ("builders", "ktarget", "", "solid"),
    ("builders", "evaluator", "", "solid"),
    ("builders", "ctrl", "", "solid"),
    ("builders", "confirm", "", "solid"),
    ("ktarget", "evaluator", "ktarget_table_c8.json", "solid"),
    ("ctrl", "optimiser", "ctrl_kmax_*.json as --k-max, --ctrl-margin", "solid"),
    ("evaluator", "optimiser", "x  /  (EFPD, F_dH, g)", "both"),
    ("optimiser", "confirm", "optimization_checkpoint.json", "solid"),
    ("optimiser", "post", "optimization_checkpoint.json, optimization_results.json", "solid"),
    ("confirm", "post", "summary.json, *.csv, *.tex", "solid"),
    ("post", "thesis", "figures, tables, numbers", "solid"),
]

# Files excluded on purpose (not drawn, not tabulated). Kept here so that
# --check can tell the user what was left out and why.
EXCLUDED_PREFIXES = ("apply_", "fix_", "repair_", "verify_leu", "verify_zoned")
EXCLUDED_EXACT = {
    "make_leu_policy.py": "code generator for leu_policy.py",
    "measure_leakage_target.py": "superseded by sweep_ktarget.py",
    "rescore_pareto.py": "superseded by rescore_*_core.py",
    "test_chunking.py": "restart test of the evaluator",
    "test_constraint_norm.py": "unit test",
    "constraint_scale_audit.py": "diagnostic",
    "time_optimizer_phases.py": "timing accounting",
    "campaign_timing.py": "timing accounting",
    "cost_vs_design.py": "timing accounting",
    "fig_cost_vs_design.py": "figure only",
    "merge_nsga_sens.py": "aggregation of nsga_sensitivity.py runs",
    "parse_runlog.py": "log parser",
    "extract_k_history.py": "data extraction",
    "extract_burnup_history.py": "data extraction",
    "c6_pareto_figure.py": "figure only",
    "c6_extra_figures.py": "figure only",
    "c6_variable_influence.py": "figure only",
    "plot_feasibility_envelope.py": "figure only",
    "plot_atf_tradeoff.py": "figure only",
    "apply_atf_limit.py": "archive filter",
    "build_corrected_front.py": "archive filter",
    "c8_boron_3d_analysis.py": "analysis of outputs",
    "check_mp.py": "one-off check",
    "check_entropy.py": "diagnostic",
    "recheck_entropy.py": "diagnostic",
}

# --------------------------------------------------------------------------
# 2. Consistency check against the repository
# --------------------------------------------------------------------------

def check(repo: Path) -> bool:
    ok = True
    listed = {s[0] for b in BLOCKS.values() for s in b["scripts"]}
    for name in sorted(listed):
        if not (repo / name).exists():
            print(f"MISSING in repo: {name}")
            ok = False
    present = {p.name for p in repo.glob("*.py")}
    unclassified = []
    for name in sorted(present):
        if name in listed or name in EXCLUDED_EXACT or name == Path(__file__).name:
            continue
        if name.startswith(EXCLUDED_PREFIXES):
            continue
        unclassified.append(name)
    print(f"listed scripts : {len(listed)}  (all present: {ok})")
    print(f"excluded       : {len(EXCLUDED_EXACT)} by name, "
          f"{sum(1 for n in present if n.startswith(EXCLUDED_PREFIXES))} by prefix")
    if unclassified:
        print("UNCLASSIFIED (add to a block or to EXCLUDED_EXACT):")
        for n in unclassified:
            print(f"  {n}")
        ok = False
    return ok


# --------------------------------------------------------------------------
# 3. Tables
# --------------------------------------------------------------------------

def tex_escape(s: str) -> str:
    return (s.replace("_", "\\_").replace("<", "\\textless ").replace(">", "\\textgreater ")
             .replace("#", "\\#").replace("%", "\\%").replace("&", "\\&"))


def tables_text(fmt: str) -> str:
    """The per-block tables as one string, `latex` rows or `md` tables."""
    out = []
    for key, b in BLOCKS.items():
        if not b["scripts"]:
            continue
        out.append(f"\n%% ---- {b['title']} ({key}) ----" if fmt == "latex"
                   else f"\n### {b['title']}\n")
        if fmt == "md":
            out.append("| Script | Role | Reads | Writes |")
            out.append("|---|---|---|---|")
            for name, role, reads, writes in b["scripts"]:
                out.append(f"| `{name}` | {role} | {reads} | {writes} |")
        else:
            for name, role, reads, writes in b["scripts"]:
                out.append(f"    \\texttt{{{tex_escape(name)}}} & {tex_escape(role)} & "
                           f"{tex_escape(reads)} & {tex_escape(writes)} \\\\")
    return "\n".join(out) + "\n"


def print_tables(fmt: str):
    print(tables_text(fmt), end="")


# --------------------------------------------------------------------------
# 4. Graphviz source
# --------------------------------------------------------------------------

FILL = {"physics": ("#EEEDFE", "#534AB7", "#26215C"),
        "optim":   ("#FAECE7", "#993C1D", "#4A1B0C"),
        "infra":   ("#F1EFE8", "#5F5E5A", "#2C2C2A")}


def node_label(b) -> str:
    fill, border, text = FILL[b["kind"]]
    rows = [f'<TR><TD ALIGN="CENTER"><B><FONT POINT-SIZE="11" COLOR="{text}">{b["title"]}</FONT></B></TD></TR>']
    if b["scripts"]:
        names = [s[0] for s in b["scripts"]]
        for n in names:
            rows.append(f'<TR><TD ALIGN="LEFT"><FONT FACE="Courier" POINT-SIZE="9" COLOR="{text}">{n}</FONT></TD></TR>')
    for line in b.get("note", []):
        rows.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{text}">{line}</FONT></TD></TR>')
    return ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
            + "".join(rows) + "</TABLE>>")


def dot_source() -> str:
    lines = ["digraph code_architecture {",
             '  graph [rankdir=TB, nodesep=0.45, ranksep=0.55, fontname="Helvetica", '
             'splines=spline, pad=0.15];',
             '  node  [shape=box, style="rounded,filled", fontname="Helvetica", penwidth=0.8, margin="0.12,0.06"];',
             '  edge  [fontname="Helvetica", fontsize=8.5, color="#444441", fontcolor="#444441", '
             'arrowsize=0.7, penwidth=0.8];']
    for key, b in BLOCKS.items():
        fill, border, _ = FILL[b["kind"]]
        lines.append(f'  {key} [label={node_label(b)}, fillcolor="{fill}", color="{border}"];')
    lines.append("  { rank=same; ktarget; evaluator; ctrl; }")
    lines.append("  { rank=same; confirm; post; }")
    for src, dst, label, style in EDGES:
        attrs = []
        if label:
            attrs.append(f'label=" {label} "')
        if style == "both":
            attrs.append("dir=both")
        if (src, dst) == ("confirm", "post"):
            attrs.append("constraint=false")
        lines.append(f"  {src} -> {dst} [{', '.join(attrs)}];")
    lines.append("}")
    return "\n".join(lines)


def downgrade_pdf(path: Path):
    """Graphviz writes PDF 1.7. pdfTeX before TeX Live 2018 accepts at most
    1.5 and warns otherwise. Rewrite the header in place when possible."""
    if shutil.which("qpdf"):
        tmp = path.with_suffix(".tmp.pdf")
        r = subprocess.run(["qpdf", "--force-version=1.5", str(path), str(tmp)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            tmp.replace(path)
            return
        tmp.unlink(missing_ok=True)
    print("  note: qpdf not found, PDF left at version 1.7 "
          "(harmless warning in older pdfTeX)")


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="verify spec against repo and exit")
    ap.add_argument("--table", choices=["latex", "md"], help="print block tables and exit")
    ap.add_argument("--out", default=".", help="output directory")
    ap.add_argument("--dpi", type=int, default=200, help="PNG preview resolution")
    ap.add_argument("--repo", default=".", help="repository root to check against")
    args = ap.parse_args()

    machine_check()
    repo = Path(args.repo)
    ok = check(repo)
    if args.check:
        sys.exit(0 if ok else 1)
    if args.table:
        print_tables(args.table)
        return
    if not ok:
        print("\nWARNING: spec and repository disagree, figure drawn anyway.\n")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = dot_source()
    (out / "code_architecture.dot").write_text(src)
    for fmt, extra in (("pdf", []), ("png", [f"-Gdpi={args.dpi}"])):
        target = out / f"code_architecture.{fmt}"
        subprocess.run(["dot", f"-T{fmt}", *extra, "-o", str(target)],
                       input=src, text=True, check=True)
        if fmt == "pdf":
            downgrade_pdf(target)
        print(f"wrote {target}")


if __name__ == "__main__":
    main()
