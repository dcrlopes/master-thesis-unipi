#!/usr/bin/env python3
r"""
apply_c9_figures.py -- insert the Campaign 9 figures and the analysis that
goes with them into results_c9.tex.

Every edit is anchored on a unique string. The script refuses to run if an
anchor is missing or appears more than once, so it cannot be applied twice
and cannot be applied to a file that has moved on.

Usage, from the directory that holds results_c9.tex:

    python apply_c9_figures.py --selftest
    python apply_c9_figures.py --check
    python apply_c9_figures.py
    python apply_c9_figures.py --revert

What it changes:
  1. header comment, the figure list now names the files that exist
  2. cost paragraph, 15 per cent corrected to 14.5 per cent, figure added
  3. boron measurement, new paragraph validating the C8 proxy, figure added
  4. Pareto figure remapped to c9_front.pdf with a caption that matches it
  5. design space figure added under reading 2, the reflector at its bound
  6. convergence, hypervolume figure and infill learning figure added
  7. stability figure remapped to c9_front_stability.pdf
  8. two formulations figure remapped to c9_two_formulations.pdf
  9. mechanism figure remapped to c9_gd_trade.pdf, hump figure added
 10. limitation 1 now points at the design space figure
"""

import argparse
import datetime
import pathlib
import sys

FILE = "results_c9.tex"

EDITS = [

# ------------------------------------------------------------------ 1 header
("""%  FIGURES EXPECTED IN figs_c9/ (written by c9_figures.py and
%  c9_analysis_figures.py)
%    c9_pareto.pdf, c9_two_formulations.pdf, c9_gd_mechanism.pdf,
%    c9_front_stability.pdf""",
 """%  FIGURES EXPECTED IN figs_c9/ (all written by c9_thesis_figures.py from
%  out_c9/optimization_checkpoint.json, no OpenMC needed)
%    c9_front.pdf             objective plane, both MTC ceilings
%    c9_cost.pdf              cost by component and per design
%    c9_wb_proxy.pdf          measured boron worth against the C8 proxy law
%    c9_design_space.pdf      the four variables along the evaluation order
%    c9_hv.pdf                hypervolume and the gain per iteration
%    c9_learning.pdf          boron demand along the evaluation order
%    c9_front_stability.pdf   Pareto membership frequency
%    c9_two_formulations.pdf  the archive in the C8 objective plane
%    c9_gd_trade.pdf          boron demand against gadolinia
%    c9_hump.pdf              the hump and the penalty it carries"""),

("""%    fig:c9-pareto, fig:c9-two-formulations, fig:c9-gd-mechanism,
%    fig:c9-front-stability""",
 """%    fig:c9-pareto, fig:c9-two-formulations, fig:c9-gd-mechanism,
%    fig:c9-front-stability, fig:c9-cost, fig:c9-wb-proxy,
%    fig:c9-design-space, fig:c9-hv, fig:c9-learning, fig:c9-hump"""),

# -------------------------------------------------------------- 2 cost + fig
("""\\paragraph{Cost.}
The campaign ran 60 evaluations in \\SI{18.7}{h} on the reference
workstation. The boron measurement accounted for \\SI{2.7}{h}, that is
\\SI{15}{\\percent} of the total, at one or two extra core solves per design.
The reformulation is therefore obtained at a cost comparable to the
\\SI{3}{\\percent} increment that moved the peaking objective to core level in
Campaign 4.""",
 """\\paragraph{Cost.}
The campaign ran 60 evaluations in \\SI{18.65}{h} on the reference
workstation. The boron measurement accounted for \\SI{2.71}{h}, that is
\\SI{14.5}{\\percent} of the total, at one or two extra core solves per design.
The reformulation is therefore obtained at a cost comparable to the
\\SI{3}{\\percent} increment that moved the peaking objective to core level in
Campaign 4.

Figure~\\ref{fig:c9-cost} separates that total. Depletion at assembly level
takes \\SI{10.47}{h}, which is \\SI{56}{\\percent} of the campaign, and the
three core solves of the screens and of the reference concentration take
\\SI{5.28}{h} together. The new measurement is therefore not the expensive
part of an evaluation, and the adaptive bracket is the reason. Ninety boron
solves were needed for sixty designs, an average of 1.5 per design, because
a single solve at \\SI{2000}{ppm} already brackets the root whenever the
design is anywhere near criticality at the reference concentration. The
designs that needed the second solve are the high-enrichment ones that the
constraint set rejects in any case, visible in the right panel as the group
near 30 minutes inside the design of experiments.

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{figs_c9/c9_cost.pdf}
  \\caption{Cost of Campaign 9. Left, the aggregate wall time by component
  over the 60 evaluations. Right, the wall time of each evaluation,
  separated by the number of boron solves the adaptive bracket required.
  The shaded region is the design of experiments.}
  \\label{fig:c9-cost}
\\end{figure}"""),

# ------------------------------------------------------- 3 proxy validation
("""Two guards bound the objective where the measurement cannot reach.""",
 """That measurement also tests the correlation it replaces. The power law of
Section~\\ref{sec:res-c8-swing} was fitted on eleven Campaign 8 designs and
Campaign 9 measures the worth of sixty, so the campaign provides an
independent sample an order of magnitude larger than the fit.
Figure~\\ref{fig:c9-wb-proxy} compares them. The relative root mean square
deviation is \\SI{3.5}{\\percent} with a mean bias of \\SI{-1.2}{\\percent},
and the agreement is not uniform. Below \\SI{5.6}{wt\\%}, where every
feasible design lies, the bias is \\SI{-0.3}{\\percent} and the scatter is
\\SI{2.6}{\\percent}. Above that value the law overestimates the worth by
\\SI{3.6}{\\percent} on average.

Two consequences follow, and they point in opposite directions.

\\begin{enumerate}
  \\item Inside the converged region the correlation would have been
        adequate. A \\SI{2.6}{\\percent} error in the worth is a
        \\SI{2.6}{\\percent} error in the concentration, that is about
        \\SI{45}{ppm} at \\SI{1700}{ppm}, against a front that spans
        \\SI{502}{ppm}. The ranking of the front would not have changed.

  \\item Outside it the correlation degrades exactly where the archive is
        most uncertain. The high-enrichment designs whose concentration
        reaches the clip are the ones the law overestimates, so a
        correlation-driven campaign would have compounded an extrapolated
        worth with an extrapolated root. Measuring removes that compounding
        at a cost of \\SI{14.5}{\\percent} of the campaign.
\\end{enumerate}

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.80\\textwidth]{figs_c9/c9_wb_proxy.pdf}
  \\caption{The differential boron worth measured on the 60 Campaign 9
  designs against the power law fitted on eleven Campaign 8 designs in
  Section~\\ref{sec:res-c8-swing}. Upper panel, the worth against
  enrichment. Lower panel, the relative residual of the law, with the root
  mean square band shaded.}
  \\label{fig:c9-wb-proxy}
\\end{figure}

Two guards bound the objective where the measurement cannot reach."""),

# ----------------------------------------------------------- 4 pareto figure
("""\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.86\\textwidth]{figs_c9/c9_pareto.pdf}
  \\caption{The Campaign 9 archive in the reformulated objective plane.
  Crosses are infeasible designs, filled circles are feasible, colour is
  the gadolinia weight fraction, and the solid horizontal line is the
  measured moderator-coefficient ceiling at \\SI{12.8}{MPa}. The dashed line
  is the corresponding ceiling at \\SI{15.5}{MPa}.}
  \\label{fig:c9-pareto}
\\end{figure}""",
 """Figure~\\ref{fig:c9-pareto} shows the archive in the objective plane. Three
features are worth reading from it before the front itself.

\\begin{enumerate}
  \\item The feasible set is a band, not a cloud. The 22 feasible designs
        span \\SIrange{1.490}{1.647}{} in $F_{\\Delta H}$ and
        \\SIrange{1406}{4371}{ppm} in $c_\\mathrm{max}$, and the five front
        members are compressed into \\SI{0.055}{} of the first range. The
        boron objective discriminates between these designs while the
        peaking objective barely does, which is what makes the front
        narrow.

  \\item The designs at the clip are all infeasible. Eight designs sit on the
        \\SI{6000}{ppm} plateau and every one of them is rejected on the
        reactivity or the controllability constraints, so the plateau never
        competes for a place on the front and the guard of the previous
        subsection never influences the selection.

  \\item Two feasible designs lie above both ceilings, C9-10 at
        \\SI{4371}{ppm} and C9-12 at \\SI{3003}{ppm}. Feasibility in the
        optimisation sense and operability are therefore not the same
        property, which is the reason the ceiling is drawn on the figure but
        is not imposed as a constraint. It also sets the size of the claim
        the front can carry, since 20 of the 22 feasible designs clear the
        lower ceiling and no front member comes within \\SI{800}{ppm} of
        it.
\\end{enumerate}

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.86\\textwidth]{figs_c9/c9_front.pdf}
  \\caption{The Campaign 9 archive in the reformulated objective plane. Open
  circles are the 38 infeasible designs and filled circles are the 22
  feasible ones, coloured by cycle length. The connected line is the
  five-member Pareto front, labelled by archive index. The dashed and
  dotted horizontal lines are the moderator-coefficient ceilings measured
  in Section~\\ref{sec:res-c8-swing} at \\SI{12.8}{MPa} and
  \\SI{15.5}{MPa}. The shaded band at the top is the \\SI{6000}{ppm} clip of
  the boron objective. Drawn by \\texttt{c9\\_thesis\\_figures.py}.}
  \\label{fig:c9-pareto}
\\end{figure}"""),

# --------------------------------------------------- 5 design space figure
("""Table~\\ref{tab:c9-constraints} shows which constraints bind.""",
 """The second reading deserves the evidence of the whole campaign rather than
of the front alone. Figure~\\ref{fig:c9-design-space} follows each design
variable along the evaluation order. The reflector reaches its bound at the
first infill iteration and never leaves it, and the enrichment collapses
from the full \\SIrange{2.0}{17.2}{wt\\%} box to a band narrower than
\\SI{0.5}{wt\\%} around \\SI{4.3}{wt\\%}. The gadolinia content and the pin
count stay spread over most of their ranges to the last evaluation, which
says that the optimiser found no single best absorber loading and is
trading it against the peaking objective inside an otherwise fixed design.

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{figs_c9/c9_design_space.pdf}
  \\caption{The four design variables along the evaluation order, with the
  box bounds drawn as dotted lines and the design of experiments shaded.
  Filled circles are feasible designs. The reflector sits on its upper
  bound throughout the infill and the enrichment collapses into a narrow
  band, while the two gadolinia variables remain spread.}
  \\label{fig:c9-design-space}
\\end{figure}

Table~\\ref{tab:c9-constraints} shows which constraints bind."""),

# ------------------------------------------- 6 convergence, hv and learning
("""\\noindent identical over the last three iterations. The stopping rule of
Section~\\ref{sec:meth-stats}, a gain below \\SI{1}{\\percent} over three
consecutive iterations, was therefore satisfied before the campaign ended.
The run continued only because the iteration count was fixed in advance.""",
 """\\noindent identical over the last three iterations. The stopping rule of
Section~\\ref{sec:meth-stats}, a gain below \\SI{1}{\\percent} over three
consecutive iterations, was therefore satisfied before the campaign ended.
The run continued only because the iteration count was fixed in advance.

Figure~\\ref{fig:c9-hv} gives the same history with the gain of each
iteration. The first infill iteration is worth \\SI{41.6}{\\percent}, the
second \\SI{3.8}{\\percent} and the fourth \\SI{2.1}{\\percent}, while
iterations three, five and six contribute \\SI{0.01}{\\percent} or less. The
sequence is not monotone in its gains, which matters for the stopping rule,
because a rule that stopped at the first iteration below the threshold would
have stopped at iteration three and would have missed the
\\SI{2.1}{\\percent} improvement that follows it. The three-iteration window
of Section~\\ref{sec:meth-stats} is what prevents that error.

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{figs_c9/c9_hv.pdf}
  \\caption{Hypervolume history of Campaign 9. Left, the absolute
  hypervolume against iteration, iteration zero being the design of
  experiments. Right, the relative gain of each iteration against the
  \\SI{1}{\\percent} threshold of the stopping rule.}
  \\label{fig:c9-hv}
\\end{figure}

The infill is also visible directly in the objective.
Figure~\\ref{fig:c9-learning} follows the boron requirement along the
evaluation order. The mean falls from \\SI{3977}{ppm} over the 24 designs of
the Latin hypercube to \\SI{1861}{ppm} over the 36 infill designs, a factor
of \\num{2.1}, and the best feasible value improves in seven steps from
\\SI{2615}{ppm} to \\SI{1406}{ppm}. Four of those steps fall in the first
six infill evaluations and the last occurs at evaluation 47 of 60, which is
consistent with the flat tail of the hypervolume.

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.86\\textwidth]{figs_c9/c9_learning.pdf}
  \\caption{Boron requirement along the evaluation order. The shaded region
  is the design of experiments and the clear region the surrogate-driven
  infill. The step line is the best feasible value found so far and the
  horizontal dashed lines are the phase means.}
  \\label{fig:c9-learning}
\\end{figure}"""),

# -------------------------------------------------------- 7 stability figure
("""\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.76\\textwidth]{figs_c9/c9_front_stability.pdf}
  \\caption{Pareto membership frequency under the measurement noise. Dark
  bars are members of the nominal front.}
  \\label{fig:c9-front-stability}
\\end{figure}""",
 """\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.80\\textwidth]{figs_c9/c9_front_stability.pdf}
  \\caption{Pareto membership frequency under the measurement noise, 2000
  perturbations with \\SI{0.010}{} in $F_{\\Delta H}$ and \\SI{10}{ppm} in
  $c_\\mathrm{max}$. Coloured bars are members of the nominal front and grey
  bars are designs that join it only under the noise. The dotted line is a
  simple majority. Designs below \\SI{5}{\\percent} are omitted, as in
  Table~\\ref{tab:c9-stability}.}
  \\label{fig:c9-front-stability}
\\end{figure}

The figure separates the front into three groups rather than two. C9-47 and
C9-34 are on the front in every resample. C9-44, C9-40 and C9-35 sit around
the majority line, so their membership is a coin toss at the measurement
precision of this work. C9-29 and C9-30 are excluded from the nominal front
by margins smaller than the noise and appear on it in about a quarter of the
resamples, which is why the candidate set carried forward from this campaign
should be the seven designs of the figure rather than the five of
Table~\\ref{tab:c9-front}."""),

# -------------------------------------------------- 8 two formulations figure
("""\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.88\\textwidth]{figs_c9/c9_two_formulations.pdf}
  \\caption{The Campaign 9 archive drawn in the Campaign 8 objective plane.
  Open squares are the designs the Campaign 8 objectives would select from
  this archive. Circles outlined in red are the Campaign 9 front, coloured
  by their boron requirement. The dashed vertical line is the mission
  constraint.}
  \\label{fig:c9-two-formulations}
\\end{figure}""",
 """\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.92\\textwidth]{figs_c9/c9_two_formulations.pdf}
  \\caption{The Campaign 9 archive drawn in the Campaign 8 objective plane,
  maximise cycle length and minimise peaking. Crosses are designs the
  Campaign 8 constraint set rejects, grey circles are the 35 it accepts and
  open squares are the seven the Campaign 8 objectives would select from
  them. Circles outlined in colour are the Campaign 9 front, coloured by
  boron requirement. The dashed vertical line is the mission floor. The
  inset enlarges the cluster in which every Campaign 9 front member sits.}
  \\label{fig:c9-two-formulations}
\\end{figure}

The inset makes the separation concrete. The five Campaign 9 front members
occupy \\SI{106}{d} of cycle length, from \\SI{1841}{d} to \\SI{1947}{d},
which the Campaign 8 objectives read as the worst corner of the feasible set
in one objective while the peaking objective barely separates them at all.
Only C9-35 is picked up by both formulations, and it is the front member
with the highest boron requirement of the five, \\SI{1908}{ppm}.

The seven designs the Campaign 8 objectives would select span
\\SIrange{1813}{4163}{d} and require \\SIrange{1452}{4371}{ppm}. Three of
the seven, C9-55 at \\SI{2923}{ppm}, C9-12 at \\SI{3003}{ppm} and C9-10 at
\\SI{4371}{ppm}, exceed the \\SI{12.8}{MPa} ceiling. An objective set built
on cycle length would therefore again have placed a substantial part of its
selection outside the operable region, which is the Campaign 8 outcome
reproduced on a different archive.""")

,

# ------------------------------------------- 9 mechanism figure and the hump
("""\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.78\\textwidth]{figs_c9/c9_gd_mechanism.pdf}
  \\caption{The mechanism of the campaign. Upper panel, the critical boron at
  beginning of life and at the operating maximum for the 16 feasible designs
  in the enrichment band, joined by a vertical bar whose length is the
  hump expressed in ppm. Lower panel, the hump itself. Designs on the
  Pareto front are circled.}
  \\label{fig:c9-gd-mechanism}
\\end{figure}""",
 """\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{figs_c9/c9_gd_trade.pdf}
  \\caption{The mechanism of the campaign. Left, the boron requirement of
  the 22 feasible designs against gadolinia content, coloured by
  enrichment, where the two effects are mixed. Right, the 16 feasible
  designs of the band $e = 4.31 \\pm \\SI{0.35}{wt\\%}$, with the
  beginning-of-life concentration as an open circle, the objective as a
  filled circle and the vertical bar between them equal to the hump
  expressed in ppm. The dashed line is a least squares fit to
  $c_\\mathrm{max}$ and the horizontal lines are the two
  moderator-coefficient ceilings.}
  \\label{fig:c9-gd-mechanism}
\\end{figure}

Read across the band in Figure~\\ref{fig:c9-gd-mechanism}, three things
happen at once as the gadolinia rises. The beginning-of-life concentration
falls, because the poison holds down reactivity that boron would otherwise
have to hold. The vertical bars shorten and then vanish, because the hump
disappears above roughly \\SI{2.5}{wt\\%}. And the two quantities converge,
so that above that loading the objective and its beginning-of-life value are
the same number. One design of the band sits above the \\SI{12.8}{MPa} ceiling, C9-12 at
\\SI{3003}{ppm}, and it is the most dilute of the sixteen at
\\SI{0.14}{wt\\%}. Its beginning-of-life requirement is \\SI{2482}{ppm},
below both ceilings, so it is the hump alone that carries it above them.
Evaluating the objective at beginning of life would have declared this
design operable.

Figure~\\ref{fig:c9-hump} isolates that second effect over the whole
feasible set. The hump reaches \\SI{3430}{pcm} at negligible loading and
falls below the \\SI{400}{pcm} noise floor above \\SI{2.5}{wt\\%}. Only two
feasible designs above that loading retain a resolved hump, of
\\SI{1093}{pcm} at \\SI{3.15}{wt\\%} and \\SI{512}{pcm} at
\\SI{4.10}{wt\\%}, and both carry their gadolinia on 20 pins rather than on
16, which is the matched comparison of Table~\\ref{tab:c9-pairs} seen from
the other side.

The right panel confirms that the penalty enters the objective as
Equation~\\eqref{eq:c9-cmax} prescribes. It is proportional to the hump
with a slope of \\SI{0.153}{ppm\\per pcm}, that is a boron worth of
\\SI{6.5}{pcm\\per ppm}, against \\SI{6.35}{pcm\\per ppm} measured as the
mean over the eight designs that carry a hump. The agreement is a
consistency check on the implementation rather than a physical result, and
it is reported because it is the only direct verification that the hump term
and the measured worth are combined correctly.

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\textwidth]{figs_c9/c9_hump.pdf}
  \\caption{The mid-cycle hump over the feasible set. Left, the hump against
  gadolinia content, with filled circles for the designs inside the
  enrichment band and the \\SI{400}{pcm} noise floor marked. Right, the
  penalty the hump carries into the objective, against the hump itself.}
  \\label{fig:c9-hump}
\\end{figure}"""),

# ------------------------------------------------------------- 10 limitation
("""  \\item \\textbf{The reflector is active at its bound.} Every front member
        carries the maximum reflector thickness the vessel admits, and
        $g_\\mathrm{geom}$ reaches $-0.009$. The solution is a boundary
        optimum. A larger vessel would move the front, so the result is
        conditional on the hull assumed in Section~\\ref{sec:meth-zoning}.""",
 """  \\item \\textbf{The reflector is active at its bound.} Every front member
        carries the maximum reflector thickness the vessel admits, and
        $g_\\mathrm{geom}$ reaches $-0.009$. The solution is a boundary
        optimum, and Figure~\\ref{fig:c9-design-space} shows that the
        optimiser reached that bound in the first infill iteration and never
        tested a design away from it afterwards. A larger vessel would move
        the front, so the result is conditional on the hull assumed in
        Section~\\ref{sec:meth-zoning}."""),
]


def apply(mode):
    p = pathlib.Path(FILE)
    if not p.exists():
        print(f"ABORT: {FILE} not found in {pathlib.Path.cwd()}")
        return 1
    text = p.read_text(encoding="utf-8")
    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor {i} found {n} times, expected 1. Nothing written.")
            return 1
        text = text.replace(old, new, 1)
    print(f"  {len(EDITS)} anchors matched exactly once")
    if mode == "check":
        print("  check only, nothing written")
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = pathlib.Path(f"{FILE}.bak_{stamp}")
    bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(text, encoding="utf-8")
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))
    print(f"  wrote {FILE} (backup {bak.name})")
    print(f"  semicolons in new body: {body.count(';')}, em dashes: {body.count(chr(8212))}")
    print(f"  includegraphics: {text.count('includegraphics')}")
    return 0


def revert():
    baks = sorted(pathlib.Path(".").glob(f"{FILE}.bak_*"))
    if not baks:
        print(f"  no backup for {FILE}")
        return 1
    pathlib.Path(FILE).write_text(baks[-1].read_text(encoding="utf-8"), encoding="utf-8")
    print(f"  restored {FILE} from {baks[-1].name}")
    return 0


def selftest():
    """Every anchor must be unique in the pristine file and absent after."""
    p = pathlib.Path(FILE)
    text = p.read_text(encoding="utf-8")
    bad = 0
    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            print(f"  anchor {i}: {n} matches, expected 1")
            bad += 1
        if new in text:
            print(f"  anchor {i}: replacement already present, file looks patched")
            bad += 1
    print("  selftest failed" if bad else "  selftest passed")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.revert:
        sys.exit(revert())
    sys.exit(apply("check" if a.check else "apply"))
