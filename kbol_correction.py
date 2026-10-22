"""kbol_correction.py -- the eight-layer cycle length with the end-of-cycle
criterion evaluated at a high-statistics fresh-assembly k_inf.

A stored eight-layer depletion ends its cycle at k_3D(0) x k_target / k_bol,
with k_bol the campaign solve at 4000 x 60 (about 264 pcm between seeds, about
43 d per cycle length, paper_finds.md Section 12). repro_c9/kbol_<c9|c10>_<i>_hifi.json
holds the same k_inf at 20 000 x 160 (c9_repro_check.py). The corrected cycle
length is read from the stored k history, no transport. Index i is the index of
the design in out_c9a (Campaign 9 and its continuations, whose indices agree
with out_c9 and out_c9f) or in out_c10.
"""
import json
from pathlib import Path

import core_geometry as cg


def hifi_kbol(campaign, idx, root="."):
    p = Path(root) / "repro_c9" / f"kbol_{campaign}_{int(idx)}_hifi.json"
    return float(json.loads(p.read_text())["k"][0]) if p.exists() else None


def corrected_efpd(run, k_bol_campaign, k_bol_hifi):
    """Cycle length [d] of one stored depletion (a runs.json entry) with the
    campaign k_bol replaced by the high-statistics one in its criterion."""
    bu, k, sp = run["bu_hist"], run["k_hist"], run["spec_power"]
    k_eoc = k[0] * run["k_target_ratio"] * k_bol_campaign / k_bol_hifi
    b = cg.eoc_crossing_burnup(bu, k, k_eoc)
    if b is None:
        b = bu[-1] if k[-1] > k_eoc else 0.0
    return b * 1000.0 / sp


VARS = ("enrich", "gd_wt", "refl_thick", "gd_pins")
_CKPT = {"c9": "out_c9a/optimization_checkpoint.json", "c10": "out_c10/optimization_checkpoint.json"}
_cache = {}


def _same_design(rec, campaign, idx, root):
    """True when `rec` is design idx of the checkpoint the hifi files are named
    after: an index of another checkpoint (the gadolinia study) is another design."""
    if campaign not in _cache:
        _cache[campaign] = json.loads((Path(root) / _CKPT[campaign]).read_text())["all_raw"]
    raw = _cache[campaign]
    return idx < len(raw) and all(abs(float(raw[idx][v]) - float(rec[v])) < 1e-9 for v in VARS)


def efpd_of(run, rec, campaign, idx, root="."):
    """The corrected cycle length where a high-statistics k_bol of this design
    exists, else the stored one. `rec` is the checkpoint record of the design.
    Raises if the stored criterion does not belong to this record."""
    if abs(run["k_target_ratio"] * rec["k_bol"] - rec["k_target"]) > 1e-6:
        raise ValueError(f"runs.json entry {idx} does not match the record of {campaign} design {idx}")
    kh = hifi_kbol(campaign, idx, root) if _same_design(rec, campaign, int(idx), root) else None
    return (corrected_efpd(run, rec["k_bol"], kh), True) if kh else (float(run["efpd"]), False)
