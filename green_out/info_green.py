import json
import numpy as np


K_cd = np.load(f"K_cd_disp.npy")
K_ij = np.load(f"K_ij_tau.npy")

def stats(name, A, unit):
    print(f"\n{name}")
    print("shape     :", A.shape)
    print("unit      :", unit)
    print("min       :", float(A.min()))
    print("max       :", float(A.max()))
    print("mean      :", float(A.mean()))
    print("mean_abs  :", float(np.abs(A).mean()))
    print("std       :", float(A.std()))
    print("p95_abs   :", float(np.quantile(np.abs(A), 0.95)))

stats("K_cd", K_cd, "m per m slip")
stats("K_ij_tau", K_ij, "Pa per m slip")
