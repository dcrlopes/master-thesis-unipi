"""
measure_leakage_target.py
=========================
Run ONCE for your finalized reference geometry to obtain the frozen
leakage-corrected end-of-cycle target used by the optimizer (this is your
Section-4 correction that replaces the physically-wrong hard-coded 1.03).

Physics: the assembly depletion runs at INFINITE MEDIUM (reflective BC, zero
leakage). The real finite core loses reactivity to leakage. For the core to
stay critical (k_eff = 1.0), the assembly k_inf must hold the leakage factor:

        k_eff_core = k_inf_assembly / leakage_factor
        =>  assembly must satisfy  k_inf >= 1.0 * leakage_factor  at EOC

So:     leakage_factor = k_inf(assembly) / k_eff(core)
        K_TARGET       = 1.0 * leakage_factor

Paste the printed K_TARGET into run_optimization.py (and openmc_evaluator's
__main__ check). Re-run this whenever your reference pitch/reflector/geometry
changes, because the leakage factor depends on them.
"""
import openmc
import reactor_model as rm

# --- your finalized reference design (edit to match your firmed-up geometry) ---
design = {
    "enrich_inner": 4.55, "enrich_outer": 4.05, "gd_wt": 0.0,
    "pitch": 1.26, "refl_thick": 15.0,
}
op = rm.Operating()
geo = rm.Geometry17x17()

# Assembly k_inf (reflective) -- run with tight statistics; this is a one-off.
asm, _fc, _lat = rm.make_assembly_model(
    design, op, geo, bc="reflective",
    particles=10000, batches=120, inactive=30)
sp = asm.run(cwd="run_assembly_ref", output=False)
with openmc.StatePoint(sp) as s:
    k_inf = s.keff.nominal_value

# Small-core k_eff (vacuum BC, ~21 assemblies + reflector)
core, _fc = rm.make_core_model(
    design, op, geo, particles=20000, batches=150, inactive=40)
sp = core.run(cwd="run_core_ref", output=False)
with openmc.StatePoint(sp) as s:
    k_eff = s.keff.nominal_value

leak = k_inf / k_eff
print("-" * 56)
print(f"k_inf (assembly, reflective) = {k_inf:.4f}")
print(f"k_eff (core, vacuum)         = {k_eff:.4f}")
print(f"leakage factor (k_inf/k_eff) = {leak:.4f}")
print(f"K_TARGET = {1.0 * leak:.4f}   <-- paste into run_optimization.py")
print("-" * 56)
