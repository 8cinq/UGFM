"""
UGFM harmonic scan on n‑sphere (n = 4, 5, 6).
v.3.71
Fixed
Changes requested by the user:
  1. Print *one* table per dimension that already contains both photon and baryon‑candidate rows.
  2. Round every numeric column to two decimals.
  3. Add a Russian column "Роль" that qualitatively labels the harmonic:
       • "photon" if ℓ = 0,
       • "nucleus" if ℓ ∈ {1,2,3} and R ≈ 1 fm,
       • otherwise one‑word verdict "нестабильно" (unstable).
  4. Extend the scan from the original 4‑sphere to S⁵ and S⁶, so three tables are printed.

Column meaning (theory logic):
  R (m)         – compactification radius of the n‑sphere in metres.
  ℓ             – Laplacian quantum number (harmonic index).
  degeneracy    – # of independent eigenfunctions with the same ℓ (spin degeneracy neglected).
  ω_ℓ (rad/s)   – Klein‑Gordon angular frequency for the given ℓ and radius.
  E_ℓ (J)       – One‑quantum energy ħω.
  Role          – Qualitative UGFM assignment (see above).

"""

import math
import numpy as np
import pandas as pd
from typing import List

ħ = 1.054_571_817e-34  # J·s
c = 2.997_924_58e8     # m/s

# Radii to scan (metres)
R_LIST: List[float] = [1.0e-15, 5.0e-15, 1.0e-14]

ℓ_max: int = 10        # maximum harmonic index to consider
m_particle: float = 0.0  # scalar mass (kg) – set to zero for pure KG photon‑like modes

def degeneracy_nsphere(n: int, ℓ: int) -> int:
    """Return Laplacian eigenfunction degeneracy on S^n (scalar harmonics).
    Formula: g_n(ℓ) = (2ℓ + n − 1) * (ℓ + n − 2)! / (ℓ! * (n − 1)!)."""
    numerator = (2 * ℓ + n - 1)
    denom_factorial = math.factorial(n - 1) * math.factorial(ℓ)
    return (numerator * math.factorial(ℓ + n - 2)) // denom_factorial

def laplacian_eigenvalue(n: int, ℓ: int, R: float) -> float:
    """Eigenvalue of minus‑Laplacian on S^n: λ = ℓ(ℓ + n − 1) / R²"""
    return ℓ * (ℓ + n - 1) / (R ** 2)

def klein_gordon_frequency(n: int, ℓ: int, R: float, m: float = 0.0) -> float:
    """Relativistic frequency √[(ck)² + (mc²/ħ)²] with k = √λ."""
    λ = laplacian_eigenvalue(n, ℓ, R)
    k = math.sqrt(λ)
    ω2 = (c * k) ** 2 + (m * c ** 2 / ħ) ** 2
    return math.sqrt(ω2)

def energy_quantum(ω: float) -> float:
    return ħ * ω

def classify_role(ℓ: int, R: float) -> str:
    """Assign heuristic UGFM role label."""
    if ℓ == 0:
        return "photon"
    # 1 fm ≃ 1.0e‑15 m; allow 5 % tolerance
    if ℓ in (1, 2, 3) and math.isclose(R, 1.0e-15, rel_tol=0.05):
        return "nucleus"
    return "unstable"

def build_table_for_dimension(n: int) -> pd.DataFrame:
    """Compute harmonics for given n and return rounded DataFrame with role column."""
    rows = []
    for R in R_LIST:
        for ℓ in range(ℓ_max + 1):
            ω = klein_gordon_frequency(n, ℓ, R, m_particle)
            rows.append({
                "R (m)": R,
                "ℓ": ℓ,
                "degeneracy": degeneracy_nsphere(n, ℓ),
                "ω_ℓ (rad/s)": ω,
                "E_ℓ (J)": energy_quantum(ω),
                "Role": classify_role(ℓ, R)
            })
    df = pd.DataFrame(rows)
    num_cols = ["ω_ℓ (rad/s)", "E_ℓ (J)"]
    df[num_cols] = df[num_cols].round(2)
    return df

def print_tables():
    for n in (4, 5, 6):
        print(f"\n=== Harmonics on S^{n} ===")
        df = build_table_for_dimension(n)
        print(df)

if __name__ == "__main__":
    print_tables()
