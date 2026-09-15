#!/usr/bin/env python3
r"""
apply_c9_ceilings.py -- fold the measured boron ceilings into results_c9.tex.

Run this AFTER apply_c9_figures.py. Every edit is anchored on a unique
string, so it refuses to run twice or on a file that has moved on.

    python apply_c9_ceilings.py --selftest
    python apply_c9_ceilings.py --check
    python apply_c9_ceilings.py
    python apply_c9_ceilings.py --revert

What changes:
  1. header, the new figure and the new labels
  2. the opening count of the front subsection, which cited the C8 ceiling
  3. Pareto reading 3, now judged against each design's own ceiling
  4. the Pareto figure caption, the lines are the champion's own ceilings
  5. the two-formulations paragraph, the ceiling number
  6. the mechanism paragraph on C9-12
  7. the peaking subsection, the five-design study is now a sixty-design one
  8. NEW subsection, the boron ceiling of each candidate lattice
  9. limitation 3, the ceiling is measured, the residual limitation restated
 10. the closing open point

Numbers come from c9_post/mtc_ceiling_table.txt, the local weighted refit
with the uncertainty inflated by sqrt(chi2/dof), and from the four scans on
the champion lattice.
"""

import argparse
import datetime
import pathlib
import sys

FILE = "results_c9.tex"

NEW_SUBSECTION = r"""\subsection{The boron ceiling of each candidate lattice}
\label{sec:res-c9-ceiling}

Campaign 8 measured one moderator temperature coefficient ceiling, on the
lattice of its own champion, and Campaign 9 inherited it as a single
reference line. That inheritance is not sound. The coefficient responds to a
change in moderator density, and how it responds depends on the absorber
loading of the lattice, which is one of the design variables. Eight lattices
were therefore scanned on the hardware model at \SI{12.8}{MPa}, covering
every candidate on either peaking front of
Section~\ref{sec:res-c9-peaking}, plus C9-57 as a diagnostic design.

\paragraph{Method.}
Each scan solves the core at two moderator temperatures and four boron
concentrations, two seeds each, sixteen solves per design. The states are

\begin{itemize}
  \item \SI{547}{K} and \SI{567}{K} at \SI{12.8}{MPa}, densities
        \SI{0.77109}{\gram\per\cubic\centi\metre} and
        \SI{0.73403}{\gram\per\cubic\centi\metre}, subcooling
        \SI{55.8}{K} and \SI{35.8}{K}
  \item \SI{570}{K} and \SI{590}{K} at \SI{15.5}{MPa}, used for the
        champion only
\end{itemize}

\noindent with densities from IAPWS-IF97 and saturation at
\SI{602.8}{K} and \SI{617.9}{K}. The window at \SI{12.8}{MPa} is the one
used in Section~\ref{sec:res-c8-swing}, so the Campaign 8 and Campaign 9
ceilings are directly comparable.

The ceiling is the concentration at which the coefficient changes sign. It
is obtained here by a weighted straight-line fit of the coefficient against
concentration over the three points nearest the sign change, with the
uncertainty propagated from the fit covariance and inflated by
$\sqrt{\chi^2/\nu}$ where that exceeds unity. Three of the eight scans have
$\chi^2/\nu$ above two, which is the signature of the mild curvature of the
coefficient in boron, and the inflation is what keeps their error bars
honest.

\paragraph{The champion.}
The ceiling of C9-47 is

\begin{itemize}
  \item \SI{12.8}{MPa}, two dimensions: \SI{2415}{ppm}
  \item \SI{12.8}{MPa}, three dimensions: \SI{2897}{ppm}, refit
        $\pm \SI{99}{ppm}$
  \item \SI{15.5}{MPa}, two dimensions: \SI{2653}{ppm}
  \item \SI{15.5}{MPa}, three dimensions: \SI{3244}{ppm}
\end{itemize}

\noindent The move from two to three dimensions raises the ceiling by
\SI{482}{ppm} at \SI{12.8}{MPa} and \SI{591}{ppm} at \SI{15.5}{MPa}. On the
Campaign 8 champion the same shift was \SI{508}{ppm}, so the axial leakage
feedback of Section~\ref{sec:res-c8-3d} reproduces on an independent
lattice. Any ceiling computed in two dimensions is conservative by roughly
that amount, which is a useful statement because the two-dimensional scan
costs a quarter of the three-dimensional one.

\paragraph{The set.}
Table~\ref{tab:c9-ceilings} gives the eight lattices and
Figure~\ref{fig:c9-ceiling} draws them against the gadolinia inventory,
defined as the weight fraction multiplied by the number of gadolinia pins
in the assembly, which is proportional to the mass of gadolinia per
assembly.

Three readings follow.

\begin{enumerate}
  \item The ceiling rises with the inventory and not with either variable
        alone. The rank correlation is \num{1.000} against the inventory,
        \num{0.976} against the weight fraction and \num{-0.071} against
        the pin count. C9-44 and C9-34 carry almost the same weight
        fraction, \SI{3.15}{wt\%} and \SI{3.32}{wt\%}, but 20 pins against
        16, and the ceiling differs by \SI{106}{ppm} in favour of the
        larger pin count. C9-12 carries the most pins of the eight and has
        the lowest ceiling, because its weight fraction is negligible.

  \item The dependence is logarithmic rather than linear. A fit in
        $\ln(\text{inventory})$ leaves a residual of \SI{60}{ppm} root mean
        square against \SI{96}{ppm} for a straight line in the inventory
        and \SI{78}{ppm} for a straight line in the weight fraction. The
        \SI{60}{ppm} residual is the size of the individual error bars, so
        the logarithmic form describes the data as well as the data can
        distinguish. Saturation of an already strongly absorbing pin is the
        expected explanation, and C9-44 and C9-47 differ by \SI{40}{\percent}
        in inventory and \SI{39}{ppm} in ceiling.

  \item The ceiling and the demand move in opposite directions with the
        same variable. Across the eight lattices the ceiling spans
        \SI{806}{ppm} while the operating margin spans \SI{2403}{ppm},
        from \SI{-912}{ppm} to \SI{+1491}{ppm}. The demand does most of the
        work and the ceiling reinforces it, so a design is either
        comfortably operable or clearly not, with little middle ground.
\end{enumerate}

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{figs_c9/c9_ceiling_inventory.pdf}
  \caption{The boron ceiling of eight lattices at \SI{12.8}{MPa} on the
  hardware model. Left, the measured ceiling and the boron demand of the
  same design against gadolinia inventory on a logarithmic axis, with the
  vertical bar joining each pair and the dashed line the logarithmic fit.
  Right, the operating margin, ceiling minus demand, with the two designs
  that fail their own ceiling in a separate colour. Error bars are the
  refit uncertainty. Drawn by \texttt{mtc\_front\_table.py}.}
  \label{fig:c9-ceiling}
\end{figure}

\input{figs_c9/mtc_ceiling_table}

\paragraph{Verdicts.}
Of the seven candidates on either peaking front, six clear their own ceiling
and one does not.

\begin{itemize}
  \item C9-47, C9-44, C9-34, C9-40 and C9-35 clear by \SI{710}{ppm} to
        \SI{1491}{ppm}, that is by \num{3.7} standard deviations or more.
        These verdicts are not in question.

  \item C9-58 clears by \SI{173}{ppm}, which is \num{2.2} standard
        deviations. It is the weakest verdict of the six and a finer scan
        with four points around its crossing is the way to settle it.

  \item C9-12 fails by \SI{912}{ppm}, which is \num{13.2} standard
        deviations. It is the one design that entered the front only
        through the three-dimensional peaking of
        Section~\ref{sec:res-c9-peaking}, and it is inoperable whatever its
        peaking.
\end{itemize}

C9-57, the diagnostic design, fails by \SI{193}{ppm} at \num{1.1} standard
deviations and is therefore undecided. It is not a front candidate and
nothing in this chapter depends on it, but it marks where the compounding of
demand and ceiling puts the boundary, at roughly \SI{1}{wt\%} of gadolinia
in this design space.

\openpoint{The \SI{15.5}{MPa} ceiling is measured on the champion only, and
its three-point scan has no refit uncertainty. Extend the scan to C9-34 and
C9-44 before the operability claim is made at both pressures rather than at
one. The eight lattices also confound the inventory with the reflector
thickness, which is at its bound for seven of the eight, so the logarithmic
law should not be extrapolated outside the converged region of this
campaign.}

"""

EDITS = [

# ------------------------------------------------------------------ 1 header
("""%    c9_gd_trade.pdf          boron demand against gadolinia
%    c9_hump.pdf              the hump and the penalty it carries""",
 """%    c9_gd_trade.pdf          boron demand against gadolinia
%    c9_hump.pdf              the hump and the penalty it carries
%    c9_ceiling_inventory.pdf ceiling and demand against gadolinia inventory
%                             (written by mtc_front_table.py, with
%                              figs_c9/mtc_ceiling_table.tex)"""),

("""%    fig:c9-front-stability, fig:c9-cost, fig:c9-wb-proxy,
%    fig:c9-design-space, fig:c9-hv, fig:c9-learning, fig:c9-hump""",
 """%    fig:c9-front-stability, fig:c9-cost, fig:c9-wb-proxy,
%    fig:c9-design-space, fig:c9-hv, fig:c9-learning, fig:c9-hump,
%    fig:c9-ceiling, tab:c9-ceilings, sec:res-c9-ceiling"""),

# ------------------------------------------------- 2 opening of the front §
("""Of the 60 designs evaluated, 22 satisfy every constraint and 20 of those
require less boron than the moderator-coefficient ceiling of
\\SI{2763}{ppm} measured in Section~\\ref{sec:res-c8-swing}. Thirteen are
controllable with the first two regulating banks alone.""",
 """Of the 60 designs evaluated, 22 satisfy every constraint and 20 of those
require less boron than the moderator-coefficient ceiling of the champion
lattice, \\SI{2897}{ppm}, measured in
Section~\\ref{sec:res-c9-ceiling}. That ceiling belongs to one lattice and
is used here only as a reference line, because
Section~\\ref{sec:res-c9-ceiling} shows it to be a property of the design
rather than of the core. Thirteen designs are controllable with the first
two regulating banks alone."""),

# ------------------------------------------------------ 3 Pareto reading 3
("""  \\item Two feasible designs lie above both ceilings, C9-10 at
        \\SI{4371}{ppm} and C9-12 at \\SI{3003}{ppm}. Feasibility in the
        optimisation sense and operability are therefore not the same
        property, which is the reason the ceiling is drawn on the figure but
        is not imposed as a constraint. It also sets the size of the claim
        the front can carry, since 20 of the 22 feasible designs clear the
        lower ceiling and no front member comes within \\SI{800}{ppm} of
        it.""",
 """  \\item Two feasible designs lie above both reference lines, C9-10 at
        \\SI{4371}{ppm} and C9-12 at \\SI{3003}{ppm}. Feasibility in the
        optimisation sense and operability are therefore not the same
        property, which is the reason the ceiling is drawn on the figure but
        is not imposed as a constraint. C9-12 was scanned on its own lattice
        in Section~\\ref{sec:res-c9-ceiling} and fails its own ceiling by
        \\SI{912}{ppm}, so the reference line understates its problem rather
        than overstating it. No front member comes within \\SI{980}{ppm} of
        the champion ceiling."""),

# ------------------------------------------------------- 4 Pareto caption
("""  dotted horizontal lines are the moderator-coefficient ceilings measured
  in Section~\\ref{sec:res-c8-swing} at \\SI{12.8}{MPa} and
  \\SI{15.5}{MPa}. The shaded band at the top is the \\SI{6000}{ppm} clip of
  the boron objective. Drawn by \\texttt{c9\\_thesis\\_figures.py}.}""",
 """  dotted horizontal lines are the moderator-coefficient ceilings of the
  champion lattice, \\SI{2897}{ppm} at \\SI{12.8}{MPa} and \\SI{3244}{ppm}
  at \\SI{15.5}{MPa}, measured in Section~\\ref{sec:res-c9-ceiling}. They are
  reference lines and not constraints, since each design has its own
  ceiling. The shaded band at the top is the \\SI{6000}{ppm} clip of the
  boron objective. Drawn by \\texttt{c9\\_thesis\\_figures.py}.}"""),

# ---------------------------------------------------- 5 two formulations
("""The seven designs the Campaign 8 objectives would select span
\\SIrange{1813}{4163}{d} and require \\SIrange{1452}{4371}{ppm}. Three of
the seven, C9-55 at \\SI{2923}{ppm}, C9-12 at \\SI{3003}{ppm} and C9-10 at
\\SI{4371}{ppm}, exceed the \\SI{12.8}{MPa} ceiling.""",
 """The seven designs the Campaign 8 objectives would select span
\\SIrange{1813}{4163}{d} and require \\SIrange{1452}{4371}{ppm}. Three of
the seven, C9-55 at \\SI{2923}{ppm}, C9-12 at \\SI{3003}{ppm} and C9-10 at
\\SI{4371}{ppm}, exceed the champion ceiling of \\SI{2897}{ppm}, and C9-12
also fails the ceiling of its own lattice by \\SI{912}{ppm}."""),

# ------------------------------------------------------- 6 mechanism, C9-12
("""One design of the band sits above the \\SI{12.8}{MPa} ceiling, C9-12 at
\\SI{3003}{ppm}, and it is the most dilute of the sixteen at
\\SI{0.14}{wt\\%}. Its beginning-of-life requirement is \\SI{2482}{ppm},
below both ceilings, so it is the hump alone that carries it above them.
Evaluating the objective at beginning of life would have declared this
design operable.""",
 """One design of the band sits above the champion ceiling, C9-12 at
\\SI{3003}{ppm}, and it is the most dilute of the sixteen at
\\SI{0.14}{wt\\%}. Its beginning-of-life requirement is \\SI{2482}{ppm},
below that line, so it is the hump alone that carries it above.

The lattice scan of Section~\\ref{sec:res-c9-ceiling} makes the case
stronger than the reference line allows. The ceiling of C9-12 is
\\SI{2091}{ppm}, not \\SI{2897}{ppm}, because the same dilution that raises
its demand also lowers what it can tolerate. Judged against its own lattice
the design fails by \\SI{912}{ppm} rather than by \\SI{106}{ppm}, and even
its beginning-of-life requirement is \\SI{391}{ppm} too high. Evaluating the
objective at beginning of life would still have rejected this design, but
only once its own ceiling was known."""),

# ----------------------------------------- 7 peaking, five designs to sixty
("""Two cautions apply to this subsection. The comparison rests on five
designs spanning \\SI{0.23}{wt\\%} in enrichment, which is a narrow basis for
a claim about the design space. And C9-40, the design that drives the front
change, has a two-seed spread of \\num{0.017} in the two-dimensional solve
against \\num{0.001} to \\num{0.004} for the other four, so its own
convergence is the least well established of the set.

\\openpoint{Re-solve C9-40 in the unrodded state with six seeds in both
models, twelve solves and about fifteen minutes, before the front change is
asserted. Extend the comparison to all sixty archived designs, one unrodded
solve per design in each model, which converts the five-design observation
into a measurement over the whole space and decides whether a
three-dimensional campaign is warranted.}""",
 """One caution applies to the five-design comparison. C9-40, the design that
drives the front change, has a two-seed spread of \\num{0.017} in the
two-dimensional solve against \\num{0.001} to \\num{0.004} for the other
four, so its own convergence is the least well established of the set.

\\paragraph{The comparison over the whole archive.}
The unrodded comparison was extended to all sixty designs, one solve per
design in each model with two seeds. The three-dimensional hot channel
factor is lower than the two-dimensional one for almost every design, by
\\num{0.023} on average with a standard deviation of \\num{0.017}, a ratio
of \\num{0.9851} with a relative spread of \\SI{1.10}{\\percent}. The
correction is therefore not a constant, and the rank correlation between the
two orderings is \\num{0.941} with a largest rank shift of 14 places out of
60.

Recomputed over the 22 confirmed feasible designs, the non-dominated set
moves from
$\\{\\mathrm{C9\\text{-}34}, \\mathrm{C9\\text{-}35}, \\mathrm{C9\\text{-}44},
\\mathrm{C9\\text{-}47}, \\mathrm{C9\\text{-}58}\\}$ with the
two-dimensional factor to
$\\{\\mathrm{C9\\text{-}12}, \\mathrm{C9\\text{-}34},
\\mathrm{C9\\text{-}40}, \\mathrm{C9\\text{-}47}\\}$ with the
three-dimensional one. Three observations bound what that means.

\\begin{enumerate}
  \\item The champion is on both fronts. C9-47 is selected whichever factor
        is used, so the reordering does not reach the design the campaign
        selects.

  \\item The newcomer is inoperable. C9-12 fails the ceiling of its own
        lattice by \\SI{912}{ppm}, Section~\\ref{sec:res-c9-ceiling}, so it
        cannot be a candidate whatever its peaking.

  \\item Part of the change is fidelity rather than dimensionality. The
        two-dimensional column of this comparison is measured at
        \\num{150000} particles over 200 batches, while the archive value
        is measured at \\num{100000} over 170. Over the 41 designs common
        to both, the higher-fidelity value is lower by \\num{0.016} on
        average with a standard deviation of \\num{0.017}, reaching
        \\num{0.068} on C9-42 and \\num{0.037} on C9-58. A hot channel
        factor is a maximum over channels, and a maximum over noisier
        estimates is biased upward, so the campaign-fidelity value reads
        high. C9-58 enters the two-dimensional front of this comparison
        largely through that shift.
\\end{enumerate}

The three quantities that matter are of the same size. The peaking spread
across the whole front is \\num{0.055}, the dimensional shift is
\\num{0.023}, the fidelity shift is \\num{0.016} and the leave-one-out error
of the surrogate on the feasible subset is \\num{0.039},
Table~\\ref{tab:c9-cv}. A campaign that optimised this objective more
finely would not resolve differences of this size, which is the argument
against a further campaign on the peaking objective, and it is a stronger
argument than the front change taken alone.

\\openpoint{Re-solve C9-40 in the unrodded state with six seeds in both
models, twelve solves and about fifteen minutes, before the front change of
the five-design comparison is asserted. Report the three-dimensional
peaking of the selected design as a verified value rather than an optimised
one, \\num{1.501} against \\num{1.527} in two dimensions for C9-47, both
well inside the \\num{1.65} constraint.}"""),

# ------------------------------------------------------- 9 limitation 3
("""  \\item \\textbf{The boron ceiling was measured on a different lattice.}
        The \\SI{2763}{ppm} limit of Section~\\ref{sec:res-c8-swing} was
        measured on C8-47, which carries \\SI{2.88}{wt\\%} gadolinia on 12
        pins. The Campaign 9 champion carries \\SI{4.42}{wt\\%} on 20 pins,
        roughly four times the inventory. Gadolinium hardens the spectrum
        and makes the moderator coefficient less negative, so the ceiling
        applicable to the Campaign 9 front has not yet been established.""",
 """  \\item \\textbf{The boron ceiling is measured, and it is measured at one
        pressure.} Section~\\ref{sec:res-c9-ceiling} scans eight lattices at
        \\SI{12.8}{MPa} and every candidate on either peaking front except
        C9-12 clears its own ceiling. Only the champion is scanned at
        \\SI{15.5}{MPa}, and that scan has three points rather than four, so
        it carries no refit uncertainty. The ceiling at the higher pressure
        is therefore known for one design and assumed for the rest. The
        weakest verdict of the set, C9-58 at \\num{2.2} standard deviations,
        also remains to be confirmed with a finer scan."""),

# ------------------------------------------------------ 10 closing openpoint
("""\\openpoint{Limitation 3 is the one that can change a conclusion rather than
qualify it. The moderator temperature coefficient of C9-47 must be scanned on
the hardware model at both operating pressures before the feasibility of the
Campaign 9 front is asserted. Limitation 5 is quantified in
Section~\\ref{sec:res-c9-peaking} on the five front designs and is being
extended to the full archive, one unrodded solve per design, which is what
decides whether a three-dimensional campaign is warranted.}""",
 """\\openpoint{Limitations 3 and 5 were the two that could change a conclusion
rather than qualify it, and both are now measured.
Section~\\ref{sec:res-c9-ceiling} gives the ceiling of eight lattices at
\\SI{12.8}{MPa} and of the champion at both pressures, and
Section~\\ref{sec:res-c9-peaking} gives the two-dimensional to
three-dimensional peaking comparison over all sixty designs. Neither
displaces the selected design. What remains is narrower: the
\\SI{15.5}{MPa} ceiling on two further lattices, a finer scan on C9-58, and
six seeds on C9-40.}"""),
]


def apply(mode):
    p = pathlib.Path(FILE)
    if not p.exists():
        print(f"ABORT: {FILE} not found in {pathlib.Path.cwd()}")
        return 1
    text = p.read_text(encoding="utf-8")

    marker = "\\subsection{Limitations}"
    if text.count(marker) != 1:
        print(f"ABORT: '{marker}' found {text.count(marker)} times, expected 1")
        return 1
    if "sec:res-c9-ceiling" in text:
        print("ABORT: the ceiling subsection is already present. Nothing written.")
        return 1

    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            print(f"ABORT: anchor {i} found {n} times, expected 1. Nothing written.")
            return 1
        text = text.replace(old, new, 1)

    text = text.replace(marker, NEW_SUBSECTION + marker, 1)
    print(f"  {len(EDITS)} anchors matched, new subsection inserted before Limitations")

    if mode == "check":
        print("  check only, nothing written")
        return 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = pathlib.Path(f"{FILE}.bak_{stamp}")
    bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    p.write_text(text, encoding="utf-8")
    body = "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))
    print(f"  wrote {FILE} (backup {bak.name})")
    print(f"  semicolons {body.count(';')}, em dashes {body.count(chr(8212))}, "
          f"includegraphics {text.count('includegraphics')}")
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
    text = pathlib.Path(FILE).read_text(encoding="utf-8")
    bad = 0
    for i, (old, new) in enumerate(EDITS, 1):
        n = text.count(old)
        if n != 1:
            print(f"  anchor {i}: {n} matches, expected 1")
            bad += 1
        if new in text:
            print(f"  anchor {i}: replacement already present")
            bad += 1
    if "sec:res-c9-ceiling" in text:
        print("  the new subsection is already present")
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
