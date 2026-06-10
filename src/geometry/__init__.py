# Geometry registry: each geometry exports Program + Residual
# per_layer=True → creates independent Program per layer (needed for pairing patterns, etc.)
from .o2 import O2Program, o2_residual
from .sp import SpProgram, sp_residual
from .u1 import U1Program, u1_residual
from .su2 import SU2Program, su2_residual
from .time_crystal import TCProgram, tc_residual
from .gauge import GaugeProgram, gauge_residual
from .trop import TropProgram, trop_residual
from .lorentz import LorentzProgram, lorentz_residual
from .sl import SLProgram, sl_residual
from .cobordism import CobProgram, cob_residual
from .tqft import TQFTProgram, tqft_residual
from .soliton import SolitonProgram, soliton_residual
from .weyl import WeylProgram, weyl_residual
from .weyl_soliton import weyl_soliton_residual

GEOMETRIES = {
    'o2':     {'Program': O2Program,      'residual': o2_residual,       'per_layer': False},
    'sp':     {'Program': SpProgram,      'residual': sp_residual,       'per_layer': True},
    'u1':     {'Program': U1Program,      'residual': u1_residual,       'per_layer': False},
    'su2':    {'Program': SU2Program,     'residual': su2_residual,      'per_layer': False},
    'tc':     {'Program': TCProgram,      'residual': tc_residual,       'per_layer': False},
    'gauge':  {'Program': GaugeProgram,   'residual': gauge_residual,    'per_layer': False},
    'trop':   {'Program': TropProgram,    'residual': trop_residual,     'per_layer': False},
    'lorentz':{'Program': LorentzProgram, 'residual': lorentz_residual,  'per_layer': False},
    'sl':     {'Program': SLProgram,      'residual': sl_residual,       'per_layer': False},
    'cob':    {'Program': CobProgram,     'residual': cob_residual,      'per_layer': False},
    'tqft':   {'Program': TQFTProgram,    'residual': tqft_residual,     'per_layer': False},
    'soliton':{'Program': SolitonProgram, 'residual': soliton_residual,  'per_layer': False},
    'weyl':   {'Program': WeylProgram,    'residual': weyl_residual,     'per_layer': False},
    'weyl_soliton': {'Program': WeylProgram, 'residual': weyl_soliton_residual, 'per_layer': False},
}

def get_geometry(geometry='o2'):
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown: {geometry}. Available: {list(GEOMETRIES.keys())}")
    return GEOMETRIES[geometry]

RESIDUAL_MAP = {k: v['residual'] for k, v in GEOMETRIES.items()}

def get_residual(geometry='o2'):
    return get_geometry(geometry)['residual']