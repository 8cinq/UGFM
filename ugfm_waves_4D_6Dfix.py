"""
UGFM harmonic scan on n‑sphere (n = 4, 5, 6).
v.3.71
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
"""
v3.71 — radii & energies fixed
"""

import math, numpy as np, pandas as pd
ħ = 1.054_571_817e-34     # J·s
c = 2.997_924_58e8        # m/s

R_LIST = [1.0e-15, 5.0e-15, 1.0e-14]   # metres
ℓ_max   = 10
m_particle = 0.0                       # kg (set 0 → photon‑like)

def degeneracy(n, ℓ):
    return (2*ℓ + n - 1) * math.factorial(ℓ + n - 2) // (
           math.factorial(ℓ) * math.factorial(n - 1))

def laplace_eval(n, ℓ, R):
    return ℓ * (ℓ + n - 1) / R**2

def ω_klein(n, ℓ, R, m=0.0):
    k = math.sqrt(laplace_eval(n, ℓ, R))
    return math.sqrt((c*k)**2 + (m*c**2/ħ)**2)

def classify(ℓ, R):
    if ℓ == 0: return "photon"
    if ℓ in (1,2,3) and math.isclose(R, 1e-15, rel_tol=0.05):
        return "nucleus"
    return "unstable"

def table_Sn(n):
    rows = []
    for R in R_LIST:
        for ℓ in range(ℓ_max + 1):
            ω  = ω_klein(n, ℓ, R, m_particle)
            E  = ħ * ω
            rows.append(dict(
                R_m = R,
                l   = ℓ,
                degeneracy = degeneracy(n, ℓ),
                omega_rad_s = ω,
                E_J = E,
                Role = classify(ℓ, R)
            ))
    df = pd.DataFrame(rows)

    # форматируем число радиуса и энергии в научном виде
    df["R_m"]  = df["R_m" ].apply(lambda x: f"{x:.2e}")
    df["E_J"]  = df["E_J" ].apply(lambda x: f"{x:.2e}")
    df["omega_rad_s"] = df["omega_rad_s"].apply(lambda x: f"{x:.2e}")
    return df

def main():
    for n in (4,5,6):
        print(f"\n=== Harmonics on S^{n} ===")
        print(table_Sn(n))

if __name__ == "__main__":
    main()
