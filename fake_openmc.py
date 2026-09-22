"""
fake_openmc.py -- a minimal stand-in for the parts of the openmc API that
reactor_model.py, zoning.py and hardware3d.py touch when BUILDING a model.
No transport. Used only by the tests of the depletion checks, so the real
builders (materials, universes, lattices, cells, axial segments) run here
and the counting, volume and tagging code is exercised on the real
geometry. Install with fake_openmc.install() BEFORE importing the builders.
"""
import sys
import types
from collections import OrderedDict

import numpy as np

_ids = {"n": 0}


def _next():
    _ids["n"] += 1
    return _ids["n"]


class Material:
    def __init__(self, name="", material_id=None, temperature=None):
        self.id = material_id or _next()
        self.name = name
        self.nuclides = []
        self.density = None
        self.temperature = temperature
        self.depletable = False
        self.volume = None
        self._sab = []

    def add_nuclide(self, n, f, pt="ao"):
        self.nuclides.append((n, float(f), pt))
        # real OpenMC hands these materials to the builder already flagged;
        # reproduce it so the tests exercise the unmark_unused guard
        if str(n).startswith(("U2", "U3", "Pu")):
            self.depletable = True

    def add_element(self, e, f, pt="ao", **kw):
        # crude expansion: uranium keeps its isotopes named, others as-is
        self.nuclides.append((e, float(f), pt))

    def add_elements_from_formula(self, formula, pt="ao"):
        self.nuclides.append((formula, 1.0, pt))

    def set_density(self, units, v=None):
        self.density = (units, None if v is None else float(v))

    def add_s_alpha_beta(self, name, fraction=1.0):
        self._sab.append((name, float(fraction)))

    def get_nuclides(self):
        return [n for n, *_ in self.nuclides]

    @classmethod
    def mix_materials(cls, mats, fracs, pt="ao", name=""):
        m = cls(name=name)
        for mat, f in zip(mats, fracs):
            for n, v, p in mat.nuclides:
                m.nuclides.append((n, v * f, p))
        m.density = mats[0].density
        m.depletable = any(getattr(x, "depletable", False) for x in mats)
        return m


class Materials(list):
    def __init__(self, it=()):
        super().__init__(it)


class _Region:
    def __init__(self, s):
        self.s = s

    def __and__(self, o):
        return _Region(f"({self.s}&{o.s})")

    def __or__(self, o):
        return _Region(f"({self.s}|{o.s})")

    def __invert__(self):
        return _Region(f"~{self.s}")


class _Surface:
    def __init__(self, **kw):
        self.id = _next()
        self.boundary_type = kw.pop("boundary_type", "transmission")
        self.__dict__.update(kw)

    def __neg__(self):
        return _Region(f"-{self.id}")

    def __pos__(self):
        return _Region(f"+{self.id}")


class ZCylinder(_Surface):
    pass


class ZPlane(_Surface):
    pass


class XPlane(_Surface):
    pass


class YPlane(_Surface):
    pass


class Cell:
    def __init__(self, cell_id=None, name="", fill=None, region=None):
        self.id = cell_id or _next()
        self.name, self.fill, self.region = name, fill, region
        self.temperature = None


class Universe:
    def __init__(self, universe_id=None, name="", cells=()):
        self.id = universe_id or _next()
        self.name = name
        self.cells = OrderedDict((c.id, c) for c in cells)

    def add_cell(self, c):
        self.cells[c.id] = c

    def get_all_materials(self):
        out = OrderedDict()
        for c in self.cells.values():
            out.update(_mats_of(c.fill))
        return out


class RectLattice:
    def __init__(self, lattice_id=None, name=""):
        self.id = lattice_id or _next()
        self.name = name
        self.lower_left = None
        self.pitch = None
        self.universes = None
        self.outer = None

    def get_all_materials(self):
        out = OrderedDict()
        for u in np.asarray(self.universes, dtype=object).ravel():
            out.update(u.get_all_materials())
        if self.outer is not None:
            out.update(self.outer.get_all_materials())
        return out


def _mats_of(fill):
    if fill is None:
        return {}
    if isinstance(fill, Material):
        return {fill.id: fill}
    return fill.get_all_materials()


class Geometry:
    def __init__(self, root=None):
        self.root_universe = root if isinstance(root, Universe) else Universe(cells=list(root or []))

    def get_all_materials(self):
        return self.root_universe.get_all_materials()

    def get_all_cells(self):
        return dict(self.root_universe.cells)


class RegularMesh:
    def __init__(self, mesh_id=None, name=""):
        self.id = mesh_id or _next()
        self.name = name
        self.dimension = self.lower_left = self.upper_right = None


class RectilinearMesh(RegularMesh):
    def __init__(self, mesh_id=None, name=""):
        super().__init__(mesh_id, name)
        self.x_grid = self.y_grid = self.z_grid = None


class MeshFilter:
    def __init__(self, mesh):
        self.mesh = mesh


class Tally:
    def __init__(self, tally_id=None, name=""):
        self.id = tally_id or _next()
        self.name, self.filters, self.scores = name, [], []


class Tallies(list):
    pass


class IndependentSource:
    def __init__(self, space=None, constraints=None, **kw):
        self.space, self.constraints = space, constraints


class Settings:
    def __init__(self):
        self.particles = self.batches = self.inactive = None
        self.run_mode = None
        self.temperature = None
        self.source = None
        self.entropy_mesh = None
        self.seed = 1


class Model:
    def __init__(self, geometry=None, materials=None, settings=None, tallies=None):
        self.geometry, self.materials, self.settings = geometry, materials, settings
        self.tallies = tallies or Tallies()


stats = types.SimpleNamespace(Box=lambda lo, hi: ("box", lo, hi))


class _RectangularPrism:
    def __init__(self, width, height, axis="z", origin=(0, 0), boundary_type="transmission"):
        self.id = _next()
        self.boundary_type = boundary_type

    def __neg__(self):
        return _Region(f"-prism{self.id}")

    def __pos__(self):
        return _Region(f"+prism{self.id}")


model = types.SimpleNamespace(RectangularPrism=_RectangularPrism)
config = {}
__version__ = "fake"


def install():
    """Register this module as `openmc` (and empty `openmc.deplete`,
    `openmc.model`, `openmc.stats`) in sys.modules."""
    me = sys.modules[__name__]
    sys.modules["openmc"] = me
    dep = types.ModuleType("openmc.deplete")
    sys.modules["openmc.deplete"] = dep
    me.deplete = dep
    sys.modules["openmc.model"] = types.ModuleType("openmc.model")
    sys.modules["openmc.model"].RectangularPrism = _RectangularPrism
    sys.modules["openmc.stats"] = types.ModuleType("openmc.stats")
    sys.modules["openmc.stats"].Box = stats.Box
    return me
