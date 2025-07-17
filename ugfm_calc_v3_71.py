"""ugfm_calc_v3_71.py
================================
First public reference implementation of the *Unified Geometric Fork Model* (UGFM)
mass‑prediction toy code.

The goal is two‑fold:
1. provide a *clear, pythonic* example – every variable name should reveal its meaning;
2. show how UGFM turns string tensions on a 6D hypersphere into baryon masses.

--------------------------------------------------------
THEORETICAL BACKGROUND  (ultra‑concise cheat‑sheet)
--------------------------------------------------------
• In UGFM each baryon is a triple‑junction (Y‑node) where three world‑strings meet.
  A flavour letter (u,d,s,c,b) labels the tension τ of an outgoing string segment.

• Dynamics of small oscillations around the node are governed by an N×N   
  *stiffness matrix*  K  with diagonal terms τᵢ  and universal spring‑like   
  coupling  κ  between any two strings (here treated in mean‑field fashion).

• Eigen‑frequencies ωₖ of K are promoted to quantum energies ħωₖ.   
  The total rest energy of the node is approximated by Σₖ ħωₖ  (ground state).

• Finally we calibrate the overall energy scale to the experimental proton mass   
  – everything else is a parameter‑free UGFM prediction within this toy picture.

This 200‑line script is obviously *not* a full UGFM simulator, but it captures   
its core flavour→mass mechanism and is self‑contained.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants (SI units unless stated otherwise)
# ---------------------------------------------------------------------------
HBAR: float = 1.054_571_817e-34  # J·s
LIGHT_SPEED: float = 2.997_924_58e8  # m/s  (currently unused but kept for reference)

# ---------------------------------------------------------------------------
# Model parameters (MeV units for energies / tensions)
# ---------------------------------------------------------------------------
quark_tension: Dict[str, float] = {
    "u": 33.28,
    "d": 33.71,
    "s": 146.70,
    "c": 1360.00,
    "b": 12990.00,
}

# Spring‑like coupling between any two string segments in a node (isotropic).
coupling_strength_mev: float = 10.0

# Effective oscillator mass (dimensionless factor) – kept =1 so that    
# ω ∝ √K; adjusting it rescales *all* eigen‑frequencies equally.
oscillator_mass: float = 1.0

# ---------------------------------------------------------------------------
# Quark composition of known baryons + hypothetical test nodes
# ---------------------------------------------------------------------------
# "baryon_quark_content" maps human‑readable names to a plain 3‑char flavour key.
# NB: the order of letters is irrelevant for the model because K is symmetric.

baryon_quark_content: Dict[str, str] = {
    "proton (uud)": "uud",
    "neutron (udd)": "udd",
    "lambda (uds)": "uds",
    "sigma+ (uus)": "uus",
    "xi0 (uss)": "uss",
    "omega- (sss)": "sss",
    "Lambda_c (udc)": "udc",
    "Xi_c+ (usc)": "usc",
    "Xi_c0 (dsc)": "dsc",
    "Omega_c (ssc)": "ssc",
    "Xi_cc++ (ccu)": "ccu",
    "Xi_cc+ (ccd)": "ccd",
    "Lambda_b (udb)": "udb",
    "Xi_b0 (usb)": "usb",
    "Xi_b- (dsb)": "dsb",
    "Omega_b (ssb)": "ssb",
}

hypothetical_nodes: Dict[str, str] = {
    "node_ccc (ccc)": "ccc",   # pure charm triple‑string
    "node_bbb (bbb)": "bbb",   # pure beauty triple‑string
}

# Merge both dicts – convenient single access point for the whole script.
node_quark_content: Dict[str, str] = {
    **baryon_quark_content,
    **hypothetical_nodes,
}

# ---------------------------------------------------------------------------
# Experimental masses (PDG 2025, MeV).  NaN means "not measured" yet.
# ---------------------------------------------------------------------------

pdg_mass_mev: Dict[str, float] = {
    "proton (uud)": 938.272,
    "neutron (udd)": 939.565,
    "lambda (uds)": 1115.683,
    "sigma+ (uus)": 1189.37,
    "xi0 (uss)": 1314.86,
    "omega- (sss)": 1672.45,
    "Lambda_c (udc)": 2286.46,
    "Xi_c+ (usc)": 2467.88,
    "Xi_c0 (dsc)": 2471.0,
    "Omega_c (ssc)": 2695.1,
    "Xi_cc++ (ccu)": 3621.4,
    "Xi_cc+ (ccd)": 3519.0,
    "Lambda_b (udb)": 5619.6,
    "Xi_b0 (usb)": 5794.7,
    "Xi_b- (dsb)": 5791.9,
    "Omega_b (ssb)": 6046.1,
}

# ---------------------------------------------------------------------------
# UGFM core numerical routines
# ---------------------------------------------------------------------------

def assemble_stiffness_matrix(tensions: List[float],
                              coupling: float = coupling_strength_mev) -> np.ndarray:
    """Return the NxN stiffness matrix K for a node with *isotropic* coupling.

    K_ij = τ_i                  if i == j
         = −κ                   otherwise

    where τ_i is the individual string tension and κ the universal coupling.
    """
    size = len(tensions)
    matrix = np.full((size, size), -coupling, dtype=float)
    for i, τ in enumerate(tensions):
        matrix[i, i] = τ + coupling * (size - 1)
    return matrix


def calc_raw_mode_energy(tensions: List[float],
                         coupling: float = coupling_strength_mev,
                         eff_mass: float = oscillator_mass) -> float:
    """Compute summed ground‑state energy  Σ ħω_k  for a given tension list.

    1. Build K.
    2. Diagonalise K → eigen‑values λ_k (units: MeV).
    3. Convert to angular frequencies ω_k = √(λ_k / m).
       (m is a dummy oscillator mass parameter).
    4. Sum all positive ω_k.
    """
    K = assemble_stiffness_matrix(tensions, coupling)
    eigen_vals = np.linalg.eigvals(K)
    # keep only non‑negative real parts to avoid numerical noise
    omega_k = np.sqrt(np.clip(eigen_vals.real, 0.0, None) / eff_mass)
    return float(np.sum(omega_k))


def predict_mass_spectrum() -> pd.DataFrame:
    """Return a DataFrame with UGFM mass predictions for every known/hypo node."""

    # 1) Build numeric tension list per node name
    quark_tension_by_node: Dict[str, List[float]] = {
        name: [quark_tension[q] for q in flavour]
        for name, flavour in node_quark_content.items()
    }

    # 2) Calibrate energy scale so that proton matches experiment exactly
    proton_tension = quark_tension_by_node["proton (uud)"]
    proton_energy_raw = calc_raw_mode_energy(proton_tension)
    energy_scale = pdg_mass_mev["proton (uud)"] / proton_energy_raw

    # 3) Compute predictions
    records = []
    for name, tension_list in quark_tension_by_node.items():
        raw_energy = calc_raw_mode_energy(tension_list)
        model_mass = raw_energy * energy_scale  # MeV
        exp_mass = pdg_mass_mev.get(name, math.nan)
        deviation = (
            100.0 * (model_mass - exp_mass) / exp_mass
            if not math.isnan(exp_mass)
            else math.nan
        )
        records.append(
            {
                "node": name,
                "model_mass_MeV": round(model_mass, 2),
                "pdg_mass_MeV": None if math.isnan(exp_mass) else exp_mass,
                "delta_%": None if math.isnan(deviation) else round(deviation, 2),
            }
        )

    df = pd.DataFrame(records)
    return df.sort_values("pdg_mass_MeV").reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Print pretty 4‑column mass table to stdout."""
    df = predict_mass_spectrum()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()