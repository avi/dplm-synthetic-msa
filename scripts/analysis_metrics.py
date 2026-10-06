"""Compare all OF3 samples, fixed-correspondence CA geometry and confidence."""
from pathlib import Path
import csv
import itertools
import json
import re
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1
from tmtools import tm_align

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARMS = ["natural_deep", "no_msa", "natural_50", "synthetic_50"]


def get_ca(path):
    d = MMCIF2Dict(str(path))
    # OF3 omits optional occupancy columns required by Bio.PDB's full parser.
    n = len(d['_atom_site.label_atom_id'])
    indices = [i for i in range(n) if d['_atom_site.label_atom_id'][i] == 'CA'
               and d['_atom_site.label_asym_id'][i] == 'A'
               and d.get('_atom_site.label_alt_id', ['.']*n)[i] in ('.','A')
               and d.get('_atom_site.pdbx_PDB_model_num', ['1']*n)[i] == '1']
    seq = ''.join(seq1(d['_atom_site.label_comp_id'][i], custom_map={'MSE':'M'}) for i in indices)
    xyz = np.array([[float(d['_atom_site.Cartn_'+axis][i]) for axis in 'xyz'] for i in indices])
    b = np.array([float(d.get('_atom_site.B_iso_or_equiv', ['nan']*n)[i]) for i in indices])
    return seq, xyz, b


def fit(mobile, target):
    x, y = mobile-mobile.mean(0), target-target.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correct = np.eye(3)
    correct[-1, -1] = np.linalg.det(u @ vt)
    aligned = x @ (u @ correct @ vt) + target.mean(0)
    return aligned, float(np.sqrt(np.mean(np.sum((aligned-target)**2, axis=1))))


def ca_lddt(mobile, reference):
    ref = np.linalg.norm(reference[:, None]-reference[None, :], axis=-1)
    pred = np.linalg.norm(mobile[:, None]-mobile[None, :], axis=-1)
    mask = (ref < 15) & (ref > 0) & ~np.eye(len(ref), dtype=bool)
    errors = np.abs(pred-ref)[mask]
    return float(np.mean([np.mean(errors < t) for t in (.5, 1, 2, 4)]))

