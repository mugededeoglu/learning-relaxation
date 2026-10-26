import hashlib
"""
============================================================
SYNTHETIC FEATURE EXTRACTION - PRUNED + EPS-SAFE + SPECTRUM-SCALED
(ALIGNED WITH PRUNED MINLPLIB VERSION)
============================================================
- Same column naming scheme as MINLPLIB pruned version
- Prunes saturated/redundant extras by default
- Works with QCQPInstance objects OR dicts
- DI only (stable for mixed sizes)
============================================================

Default pruning policy:
  ✅ Keep:  comm_mean, comm_max
  ✅ Keep:  h_{avg,IQR,skewness,CV}_LogCond
  ❌ Drop:  comm_min
  ❌ Drop:  diag_off_ratio extras by default
  ❌ Drop:  log10_cond_A0 by default
  ✅ Add:   g_overlap_Theta, g_indef_Theta, g_linear_mass
  ✅ Add:   spectrum-scaled eigen features (A0 + constraints)

CHANGES vs previous version:
  1) Raw lambda_min_A0 / lambda_max_A0 removed; scaled versions kept
  2) Raw Lambda_min / Lambda_max removed from constraint aggregations;
     Lambda_min_scaled / Lambda_max_scaled kept
  3) mu3 (raw 3rd central moment) replaced by
     skewness = mu3 / (std^3 + EPS_SKEW)  — dimensionless, scale-invariant
  4) CV = std / (|mean| + EPS_CV)  — always non-negative denominator
============================================================
"""

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# GLOBAL SWITCHES / PRUNING DEFAULTS
# ============================================================

INCLUDE_NM_COLUMNS = False
THETA_CLASS_EPS    = 1e-8
MASS_TOL           = 1e-10

DEFAULT_EXTRA_FEATURES = ("cond", "commutator")
INCLUDE_COND_A0        = False
INCLUDE_DIAG_OFF_RATIO = False
INCLUDE_SCALE_FEATURES = False

KEEP_LOGCOND_STATS    = ("avg", "IQR", "skewness", "CV")
KEEP_LOGDIAGOFF_STATS = ("avg", "IQR", "skewness", "CV")


# ============================================================
# EPS / TINY CONSTANTS
# ============================================================

EPS_NUM   = 1e-6
EPS_THETA = 1e-6
EPS_CV    = 1e-6      # stabilizer for CV denominator  |mean| + EPS_CV
EPS_SKEW  = 1e-9      # stabilizer for skewness denominator  std^3 + EPS_SKEW
EPS_SPEC  = 1e-6
EPS_EIG   = 1e-12


# ============================================================
# HELPERS
# ============================================================

def _get(inst: Any, key: str, default=None):
    if isinstance(inst, dict):
        return inst.get(key, default)
    return getattr(inst, key, default)


def _to_dict(inst: Any) -> Dict[str, Any]:
    if isinstance(inst, dict):
        return inst
    if hasattr(inst, "__dict__"):
        return dict(inst.__dict__)
    raise TypeError(f"Unsupported instance type: {type(inst)}")


def _sym(A):
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)


# ============================================================
# SPARSITY PATTERN DETECTION
# ============================================================

def detect_sparsity_patterns(A, tol=1e-8):
    patterns = {}
    A = _sym(A)
    n = A.shape[0]
    off_diag = A - np.diag(np.diag(A))

    patterns["Diagonal"] = bool(np.allclose(off_diag, 0, atol=tol))
    patterns["Hollow"]   = bool(np.allclose(np.diag(A), 0, atol=tol))

    patterns["Bipartite"] = False
    patterns["Tree"]      = False
    patterns["Chordal"]   = False
    patterns["Planar"]    = False

    mask = (np.abs(np.triu(off_diag, 1)) > tol)
    if np.any(mask):
        rows, cols = np.where(mask)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(zip(rows.tolist(), cols.tolist()))

        patterns["Bipartite"] = bool(bipartite.is_bipartite(G))
        patterns["Tree"]      = bool(nx.is_tree(G))
        planar, _             = nx.check_planarity(G)
        patterns["Planar"]    = bool(planar)
        patterns["Chordal"]   = bool(nx.is_chordal(G))

    return patterns


# ============================================================
# STATISTICS
#
#   CHANGE 1: mu3 replaced by skewness = mu3 / (std^3 + EPS_SKEW)
#             skewness is dimensionless and scale-invariant; mu3 has units
#             of (feature_value)^3 making it incomparable across instances
#             with different scales.
#
#   CHANGE 2: CV = std / (|mean| + EPS_CV)  [always non-negative denominator]
#             The original std / mean is signed when mean < 0 (common for
#             eigenvalue distributions of indefinite matrices), making CV
#             meaningless as a "relative variability" measure in that regime.
# ============================================================

def compute_statistics(vector):
    if vector is None or len(vector) == 0:
        return {
            "min": 0.0, "max": 0.0, "avg": 0.0,
            "skewness": 0.0, "IQR": 0.0, "Pout": 0.0, "CV": 0.0,
        }

    v = np.asarray(vector, dtype=float).reshape(-1)
    mean_val = float(np.mean(v))
    std_val  = float(np.std(v))

    # --- skewness (dimensionless, scale-invariant) ---
    if v.size > 1:
        mu3 = float(np.mean((v - mean_val) ** 3))
        skewness = float(mu3 / (std_val ** 3 + EPS_SKEW))
    else:
        skewness = 0.0

    q1, q3 = np.percentile(v, [25, 75])
    iqr = float(q3 - q1)

    if iqr > 0:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        pout = float(np.mean((v < lower) | (v > upper)))
    else:
        pout = 0.0

    # --- CV with non-negative denominator ---
    cv = float(std_val / (abs(mean_val) + EPS_CV))

    return {
        "min":      float(np.min(v)),
        "max":      float(np.max(v)),
        "avg":      float(mean_val),
        "skewness": skewness,
        "IQR":      float(iqr),
        "Pout":     float(pout),
        "CV":       cv,
    }


# ============================================================
# BOUNDS FLAG
# ============================================================

def _bounds_exist_flag(inst: Any) -> int:
    l = np.asarray(_get(inst, "l", []), dtype=float).reshape(-1)
    u = np.asarray(_get(inst, "u", []), dtype=float).reshape(-1)
    if l.size == 0 or u.size == 0 or l.size != u.size:
        return 0
    return int(np.all(np.isfinite(l)) and np.all(np.isfinite(u)))


# ============================================================
# EXTRA FEATURES
# ============================================================

def _log10_cond_from_eigs(eigs, eig_eps=EPS_EIG):
    a = np.abs(np.asarray(eigs, dtype=float))
    amax = float(np.max(a)) if a.size else 1.0
    nz = a[a > eig_eps]
    if nz.size == 0:
        cond = 1e16
    else:
        amin = float(np.min(nz))
        cond = amax / (amin + EPS_NUM)
    cond = max(1.0, min(cond, 1e16))
    return float(np.log10(cond))


def _log10_diag_off_energy_ratio(A, tiny=EPS_NUM):
    A = _sym(A)
    d = np.diag(A)
    diag_e = float(np.sum(d * d))
    tot_e  = float(np.sum(A * A))
    off_e  = max(tot_e - diag_e, 0.0)

    if off_e < tiny:
        ratio = 1e16
    elif diag_e < tiny:
        ratio = 1e-6
    else:
        ratio = diag_e / (off_e + tiny)

    ratio = max(1e-6, min(ratio, 1e16))
    return float(np.log10(ratio))


def _commutator_norm(A, B, tiny=EPS_NUM):
    A = _sym(A)
    B = _sym(B)
    C = A @ B - B @ A
    num = float(np.linalg.norm(C, ord="fro"))
    den = float(np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro"))
    return num / (den + tiny)


def _compute_commutator_features(A_list, max_pairs=40, seed_key=""):
    # Pruned: keep only mean/max (drop comm_min)
    if len(A_list) < 2:
        return {"comm_mean": 0.0, "comm_max": 0.0}

    mats = [_sym(A) for A in A_list]
    norms = []

    for k in range(1, len(mats)):
        norms.append(_commutator_norm(mats[0], mats[k], tiny=EPS_NUM))

    if len(mats) > 2:
        total_pairs = (len(mats) - 1) * (len(mats) - 2) // 2
        if total_pairs <= max_pairs:
            for i in range(1, len(mats)):
                for j in range(i + 1, len(mats)):
                    norms.append(_commutator_norm(mats[i], mats[j], tiny=EPS_NUM))
        else:
            all_pairs = [
                (i, j)
                for i in range(1, len(mats))
                for j in range(i + 1, len(mats))
            ]
            # Local, stable seed; no dependence on traversal/order or global RNG.
            stable = int.from_bytes(hashlib.sha256(str(seed_key).encode("utf-8")).digest()[:8], "little")
            sampled = random.Random(stable).sample(all_pairs, min(max_pairs, len(all_pairs)))
            for i, j in sampled:
                norms.append(_commutator_norm(mats[i], mats[j], tiny=EPS_NUM))

    if len(norms) == 0:
        return {"comm_mean": 0.0, "comm_max": 0.0}

    norms = np.asarray(norms, dtype=float)
    return {"comm_mean": float(np.mean(norms)), "comm_max": float(np.max(norms))}


def _sample_abs_entries(arr, cap=200000, tiny=EPS_NUM):
    v = np.abs(np.asarray(arr, dtype=float).ravel())
    v = v[np.isfinite(v)]
    v = v[v > tiny]
    if v.size == 0:
        return v
    if v.size <= cap:
        return v
    step = int(np.ceil(v.size / cap))
    return v[::step]


def _scale_features_from_values(v, tiny=EPS_NUM):
    if v.size == 0:
        return {
            "scale_max_abs": 0.0,
            "scale_min_pos_abs": 0.0,
            "scale_median_abs": 0.0,
            "scale_log10_range": 0.0,
            "scale_max_over_median": 0.0,
            "scale_p95_over_p05": 0.0,
        }

    vmax = float(np.max(v))
    vmin = float(np.min(v))
    vmed = float(np.median(v))
    p05  = float(np.percentile(v, 5))  if v.size >= 20 else vmin
    p95  = float(np.percentile(v, 95)) if v.size >= 20 else vmax

    return {
        "scale_max_abs":         vmax,
        "scale_min_pos_abs":     vmin,
        "scale_median_abs":      vmed,
        "scale_log10_range":     float(np.log10((vmax + tiny) / (vmin + tiny))),
        "scale_max_over_median": float((vmax + tiny) / (vmed + tiny)),
        "scale_p95_over_p05":    float((p95 + tiny) / (p05 + tiny)),
    }


def _theta_and_masses_from_eigs(eigs, tol, eps_denom=EPS_THETA):
    eigs = np.asarray(eigs, dtype=float)
    a = np.abs(eigs)
    total_mass = float(np.sum(a))
    neg_mass   = float(np.sum(a[eigs < -tol]))
    pos_mass   = float(np.sum(a[eigs >  tol]))
    theta      = float(neg_mass / (total_mass + eps_denom))
    return theta, total_mass, neg_mass, pos_mass


def _spec_scale_denom(eigs, eps=EPS_SPEC):
    """sum(|eigenvalues|) + eps — used to normalise individual eigenvalues."""
    return float(np.sum(np.abs(np.asarray(eigs, dtype=float))) + eps)


# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_features_DI(
    instances,
    tol=1e-8,
    extra_feature_sets=None,
    comm_max_pairs=40,
    scale_cap=200000,
):
    if extra_feature_sets is None:
        extra_feature_sets = DEFAULT_EXTRA_FEATURES

    extra_feature_sets = set(extra_feature_sets or [])
    if not INCLUDE_DIAG_OFF_RATIO:
        extra_feature_sets.discard("diag_off_ratio")
    if not INCLUDE_SCALE_FEATURES:
        extra_feature_sets.discard("scale")

    rows = []
    sparsity_patterns = ["Diagonal", "Hollow", "Bipartite", "Tree", "Chordal", "Planar"]
    # "mu3" replaced with "skewness" everywhere
    stat_names = ["min", "max", "avg", "skewness", "IQR", "Pout", "CV"]

    print(f"  📊 Extracting features from {len(instances)} instances...")
    print(f"  ✅ Extra features: {sorted(extra_feature_sets) if extra_feature_sets else 'None'}")

    for inst0 in instances:
        inst = _to_dict(inst0)

        A_list = _get(inst, "A_list", None)
        if A_list is None or len(A_list) == 0:
            continue

        b_list = _get(inst, "b_list", None)
        c_list = _get(inst, "c_list", None)

        A0 = _sym(A_list[0])
        n  = int(A0.shape[0])
        m  = int(len(A_list) - 1)

        row = {
            "id": _get(inst, "instance_id", _get(inst, "id", -1)),
            "instance_name": str(_get(inst, "instance_name", "synthetic")),
            "BoundsExist": _bounds_exist_flag(inst),
        }
        if INCLUDE_NM_COLUMNS:
            row["n"] = n
            row["m"] = m

        # ----------------------------------------
        # A0 features
        # ----------------------------------------
        eigs0 = np.linalg.eigvalsh(A0)
        neg0  = np.sum(eigs0 < -tol)
        row["eta_A0_over_n"] = float(neg0 / n)

        theta0, total_mass0, neg_mass0, pos_mass0 = _theta_and_masses_from_eigs(
            eigs0, tol=tol, eps_denom=EPS_THETA
        )
        row["theta_A0"] = float(theta0)

        # SCALED extremal eigenvalues only (raw removed)
        den_spec0 = _spec_scale_denom(eigs0, eps=EPS_SPEC)
        row["lambda_min_A0_scaled"] = float(np.min(eigs0) / den_spec0)
        row["lambda_max_A0_scaled"] = float(np.max(eigs0) / den_spec0)

        row["rho_A0_over_n"] = float(np.linalg.matrix_rank(A0) / n)

        patt0 = detect_sparsity_patterns(A0, tol=tol)
        for pat in sparsity_patterns:
            row[f"P{pat[0]}_A0"] = int(patt0[pat])

        if ("cond" in extra_feature_sets) and INCLUDE_COND_A0:
            row["log10_cond_A0"] = _log10_cond_from_eigs(eigs0, eig_eps=EPS_EIG)

        if "diag_off_ratio" in extra_feature_sets:
            row["log10_diagoff_A0"] = _log10_diag_off_energy_ratio(A0, tiny=EPS_NUM)

        # ----------------------------------------
        # Constraints aggregation
        # ----------------------------------------
        if m > 0:
            Xi, Theta = [], []
            Lambda_min_scaled, Lambda_max_scaled = [], []  # raw removed
            Rho = []

            LogConds, LogDiagOffs = [], []
            sparsity_vectors = {pat: [] for pat in sparsity_patterns}
            scale_vals = []

            convex_flags, concave_flags = [], []
            overlap_flags, indef_flags, linear_flags = [], [], []

            if "scale" in extra_feature_sets:
                scale_vals.append(_sample_abs_entries(A0, cap=scale_cap, tiny=EPS_NUM))

            for k in range(1, len(A_list)):
                Ak   = _sym(A_list[k])
                eigs = np.linalg.eigvalsh(Ak)

                neg = np.sum(eigs < -tol)
                Xi.append(float(neg / n))

                th, total_mass, neg_mass, pos_mass = _theta_and_masses_from_eigs(
                    eigs, tol=tol, eps_denom=EPS_THETA
                )
                Theta.append(float(th))

                is_linear  = (total_mass <= MASS_TOL)
                is_convex  = (neg_mass  <= THETA_CLASS_EPS * (total_mass + EPS_THETA))
                is_concave = (pos_mass  <= THETA_CLASS_EPS * (total_mass + EPS_THETA))

                linear_flags.append(bool(is_linear))
                convex_flags.append(bool(is_convex))
                concave_flags.append(bool(is_concave))
                overlap_flags.append(bool(is_convex and is_concave))
                indef_flags.append(bool((not is_convex) and (not is_concave)))

                # SCALED extremal eigenvalues only (raw removed)
                den_spec = _spec_scale_denom(eigs, eps=EPS_SPEC)
                Lambda_min_scaled.append(float(np.min(eigs) / den_spec))
                Lambda_max_scaled.append(float(np.max(eigs) / den_spec))

                Rho.append(float(np.linalg.matrix_rank(Ak) / n))

                if "cond" in extra_feature_sets:
                    LogConds.append(_log10_cond_from_eigs(eigs, eig_eps=EPS_EIG))

                if "diag_off_ratio" in extra_feature_sets:
                    LogDiagOffs.append(
                        _log10_diag_off_energy_ratio(Ak, tiny=EPS_NUM)
                    )

                if "scale" in extra_feature_sets:
                    scale_vals.append(
                        _sample_abs_entries(Ak, cap=scale_cap, tiny=EPS_NUM)
                    )

                patt = detect_sparsity_patterns(Ak, tol=tol)
                for pat in sparsity_patterns:
                    sparsity_vectors[pat].append(int(patt[pat]))

            row["g0_Theta"]        = float(np.mean(convex_flags))  if convex_flags  else 0.0
            row["g1_Theta"]        = float(np.mean(concave_flags)) if concave_flags else 0.0
            row["g_overlap_Theta"] = float(np.mean(overlap_flags)) if overlap_flags else 0.0
            row["g_indef_Theta"]   = float(np.mean(indef_flags))   if indef_flags   else 0.0
            row["g_linear_mass"]   = float(np.mean(linear_flags))  if linear_flags  else 0.0

            # Raw Lambda_min / Lambda_max removed; only scaled retained
            properties = {
                "Xi":                Xi,
                "Theta":             Theta,
                "Lambda_min_scaled": Lambda_min_scaled,
                "Lambda_max_scaled": Lambda_max_scaled,
                "Rho":               Rho,
            }
            for prop_name, vec in properties.items():
                stats = compute_statistics(vec)
                for s in stat_names:
                    row[f"h_{s}_{prop_name}"] = stats[s]

            if "cond" in extra_feature_sets:
                cstats = compute_statistics(LogConds)
                for s in KEEP_LOGCOND_STATS:
                    row[f"h_{s}_LogCond"] = cstats[s]

            if "diag_off_ratio" in extra_feature_sets:
                rstats = compute_statistics(LogDiagOffs)
                for s in KEEP_LOGDIAGOFF_STATS:
                    row[f"h_{s}_LogDiagOff"] = rstats[s]

            for pat in sparsity_patterns:
                row[f"g{pat[0]}_Phi{pat[0]}"] = float(sum(sparsity_vectors[pat]) / m)

            if "scale" in extra_feature_sets:
                if b_list is not None:
                    for bk in b_list:
                        scale_vals.append(
                            _sample_abs_entries(bk, cap=scale_cap, tiny=EPS_NUM)
                        )
                if c_list is not None:
                    scale_vals.append(
                        _sample_abs_entries(
                            np.asarray(c_list, dtype=float), cap=scale_cap, tiny=EPS_NUM
                        )
                    )

                v = (
                    np.concatenate(
                        [x for x in scale_vals if x.size > 0], axis=0
                    )
                    if any(x.size > 0 for x in scale_vals)
                    else np.array([])
                )
                row.update(_scale_features_from_values(v, tiny=EPS_NUM))

        else:
            # m == 0 — zero-fill all aggregated columns
            row["g0_Theta"]        = 0.0
            row["g1_Theta"]        = 0.0
            row["g_overlap_Theta"] = 0.0
            row["g_indef_Theta"]   = 0.0
            row["g_linear_mass"]   = 0.0

            for prop_name in [
                "Xi", "Theta",
                "Lambda_min_scaled", "Lambda_max_scaled",
                "Rho",
            ]:
                for s in stat_names:
                    row[f"h_{s}_{prop_name}"] = 0.0

            if "cond" in extra_feature_sets:
                for s in KEEP_LOGCOND_STATS:
                    row[f"h_{s}_LogCond"] = 0.0

            if "diag_off_ratio" in extra_feature_sets:
                for s in KEEP_LOGDIAGOFF_STATS:
                    row[f"h_{s}_LogDiagOff"] = 0.0

            for pat in sparsity_patterns:
                row[f"g{pat[0]}_Phi{pat[0]}"] = 0.0

            if "scale" in extra_feature_sets:
                row.update(_scale_features_from_values(np.array([]), tiny=EPS_NUM))

        if "commutator" in extra_feature_sets:
            row.update(_compute_commutator_features(
                A_list, max_pairs=comm_max_pairs,
                seed_key=str(_get(inst, "id_global", _get(inst, "instance_id", -1)))
            ))

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  ✅ Extracted {len(df)} instances | {df.shape[1]} features")
    return df


# ============================================================
# MAIN RUNNER
# ============================================================

if __name__ == "__main__":

    # The original notebook pickles __main__.QCQPInstance (not a Python module).
    # This shim is needed for standalone extraction; it does NOT change the PKL.
    class QCQPInstance:
        pass
    PKL_PATH = r"C:\Users\suuser\Desktop\minlplib\paperexperiments\synthetic_prime_minimal_revision\instances_mixed_pilot.pkl"
    OUT_XLSX = r"C:\Users\suuser\Desktop\minlplib\paperexperiments\synthetic_prime_minimal_revision\features_mixed_pilot.xlsx"

    EXTRA_FEATURES = ("cond", "commutator")

    print("=" * 60)
    print("SYNTHETIC FEATURE EXTRACTION - PRUNED + ALIGNED")
    print("=" * 60)
    print(f"Input:  {PKL_PATH}")
    print(f"Output: {OUT_XLSX}")
    print(f"Extra features: {EXTRA_FEATURES if EXTRA_FEATURES else 'None'}")
    print("=" * 60)

    with open(PKL_PATH, "rb") as f:
        instances = pickle.load(f)

    print(f"\n📦 Loaded {len(instances)} instances from {PKL_PATH}")

    df = extract_features_DI(
        instances,
        tol=1e-8,
        extra_feature_sets=EXTRA_FEATURES,
        comm_max_pairs=40,
        scale_cap=200000,
    )

    df.to_excel(OUT_XLSX, index=False)

    print("\n" + "=" * 60)
    print("✅ COMPLETE")
    print("=" * 60)
    print(f"📁 Output file: {OUT_XLSX}")
    print(f"📊 Shape: {df.shape}")
    print(f"🔢 Features (excluding id): {df.shape[1] - 1}")