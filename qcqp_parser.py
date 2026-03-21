# qcqp_parser.py
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import re
from pathlib import Path

# ---------------------------- Helpers ---------------------------- #

def _symmetrize(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _sense_char(s: Union[str, int, None]) -> str:
    if s in (GRB.LESS_EQUAL, '<', '<='):
        return '<'
    if s in (GRB.GREATER_EQUAL, '>', '>='):
        return '>'
    if s in (GRB.EQUAL, '=', '=='):
        return '='
    return '<'

def _lin_size(lin: gp.LinExpr) -> int:
    try:
        return lin.size()
    except Exception:
        k = 0
        while True:
            try:
                _ = lin.getVar(k)  # type: ignore[attr-defined]
                k += 1
            except Exception:
                break
        return k

def _quad_size(quad: gp.QuadExpr) -> int:
    try:
        return quad.size()
    except Exception:
        k = 0
        while True:
            try:
                _ = quad.getVar1(k)  # type: ignore[attr-defined]
                k += 1
            except Exception:
                break
        return k

def _accum_quad(A: np.ndarray, i: int, j: int, val: float) -> None:
    """Fill A so that 0.5 * x^T A x == sum val * x_i * x_j (Gurobi convention)."""
    if i == j:
        A[i, i] += 2.0 * val
    else:
        A[i, j] += val
        A[j, i] += val

def _extract_qc_linear_coeffs_from_lp_text(lp_path: str, qc_name: str) -> Optional[Dict[str, float]]:
    """
    Fallback for old Gurobi builds: parse QC linear terms directly from the LP text.
    We capture the part after 'qc_name:' up to the first '[' (quadratic block),
    and extract terms like '+ 7.66e-4 x1', '- x4', 'x4', etc.
    Returns a dict {var_name -> coeff} or None if not found.
    """
    try:
        txt = Path(lp_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Grab from '<name>:' up to '[' (start of quadratic block)
    m = re.search(rf'^\s*{re.escape(qc_name)}\s*:\s*(.*?)\[', txt, flags=re.M | re.S)
    if not m:
        return None

    linear_str = m.group(1)

    # Terms: optional signed number then a var name (skip "Constant" or similar)
    # Examples that should match:
    #   "+ 7.66e-4 x1", "- x4", "x4", "1 x4"
    terms = re.findall(r'([+\-]?\s*(?:\d+(?:\.\d+)?(?:e[+\-]?\d+)?)?)\s*([A-Za-z_]\w*)\b', linear_str)
    if not terms:
        return None

    coeffs: Dict[str, float] = {}
    for coef_str, vname in terms:
        if vname.lower() in ("constant", "objconstant"):
            continue
        token = coef_str.strip().replace(" ", "")
        if token in ("", "+"):
            coef = 1.0
        elif token == "-":
            coef = -1.0
        else:
            try:
                coef = float(token)
            except Exception:
                continue
        coeffs[vname] = coeffs.get(vname, 0.0) + coef

    return coeffs if coeffs else None

def _normalize_qcrow(
    lp_path: str,
    model: gp.Model,
    qc: gp.QConstr
) -> Tuple[Optional[gp.QuadExpr], Optional[gp.LinExpr], str, float, Optional[Dict[str, float]]]:
    """
    Return (quad, lin, sense_char, rhs, lin_from_text) robustly across Gurobi versions.
    lin_from_text is a dict var_name->coeff if we had to parse from LP text; else None.
    """
    quad: Optional[gp.QuadExpr] = None
    lin: Optional[gp.LinExpr] = None
    sense: Optional[Union[str, int]] = None
    rhs: Optional[float] = None
    lin_from_text: Optional[Dict[str, float]] = None

    row = model.getQCRow(qc)
    if isinstance(row, tuple) and len(row) == 4:
        a, b, s, r = row
        sense = s
        rhs = float(r)
        if isinstance(a, gp.QuadExpr): quad = a
        if isinstance(a, gp.LinExpr):  lin  = a
        if isinstance(b, gp.QuadExpr): quad = b
        if isinstance(b, gp.LinExpr):  lin  = b
    elif isinstance(row, gp.QuadExpr):
        quad = row
        sense = getattr(qc, "QCSense", None)
        rhs = float(getattr(qc, "QCRHS", 0.0))
    else:
        sense = getattr(qc, "QCSense", None)
        rhs = float(getattr(qc, "QCRHS", 0.0))

    # Fallback: some versions put the linear part inside QuadExpr
    if lin is None and isinstance(quad, gp.QuadExpr):
        try:
            lin = quad.getLinExpr()
        except Exception:
            lin = None

    # Last resort: text parse if still no linear info
    need_text_fallback = (lin is None)
    if not need_text_fallback:
        try:
            need_text_fallback = (_lin_size(lin) == 0)
        except Exception:
            need_text_fallback = True

    if need_text_fallback:
        qc_name = getattr(qc, "QCName", None) or getattr(qc, "ConstrName", None) or ""
        if qc_name:
            lin_from_text = _extract_qc_linear_coeffs_from_lp_text(lp_path, qc_name)

    s_char = _sense_char(sense)
    if rhs is None:
        rhs = 0.0
    return quad, lin, s_char, float(rhs), lin_from_text

# ---------------------------- Parser ---------------------------- #

def parse_qcqp_from_file(path: str) -> Dict[str, Any]:
    """
    Returns data under 0.5*x^T A x + b^T x + c.
    Each constraint stored as g(x) sense 0 with c = -rhs (+ folded contributions).
    """
    model = gp.read(path)
    model.update()

    # Fold fixed AMPL-style 'objconstant' into constants
    all_vars = model.getVars()
    fold_vals: Dict[str, float] = {}
    for v in all_vars:
        if v.VarName.lower() == "objconstant" and v.LB is not None and v.UB is not None and abs(v.LB - v.UB) < 1e-12:
            fold_vals[v.VarName] = float(v.LB)

    vars_ = [v for v in all_vars if v.VarName not in fold_vals]
    n = len(vars_)
    name_to_idx = {v.VarName: i for i, v in enumerate(vars_)}
    idx_of = lambda v: name_to_idx[v.VarName]

    l = np.array([v.LB if v.LB is not None else -np.inf for v in vars_], dtype=float)
    u = np.array([v.UB if v.UB is not None else  np.inf for v in vars_], dtype=float)

    # Objective
    A_obj = np.zeros((n, n), dtype=float)
    b_obj = np.zeros(n, dtype=float)
    try:
        c_obj = float(getattr(model, "ObjCon", 0.0))
    except Exception:
        c_obj = 0.0

    obj = model.getObjective()

    def _add_lin_to_b_or_c(linexpr: Optional[gp.LinExpr]):
        nonlocal c_obj
        if linexpr is None:
            return
        sz = _lin_size(linexpr)
        for k in range(sz):
            v = linexpr.getVar(k)
            coef = linexpr.getCoeff(k)
            if v.VarName in fold_vals:
                c_obj += coef * fold_vals[v.VarName]
            else:
                b_obj[idx_of(v)] += coef

    def _add_quad_to_A(quadexpr: Optional[gp.QuadExpr]):
        if quadexpr is None:
            return
        qsz = _quad_size(quadexpr)
        for k in range(qsz):
            v1 = quadexpr.getVar1(k)
            v2 = quadexpr.getVar2(k)
            val = quadexpr.getCoeff(k)
            if (v1.VarName in fold_vals) or (v2.VarName in fold_vals):
                continue
            i, j = idx_of(v1), idx_of(v2)
            _accum_quad(A_obj, i, j, val)

    if isinstance(obj, gp.LinExpr):
        _add_lin_to_b_or_c(obj)
    elif isinstance(obj, gp.QuadExpr):
        try:
            _add_lin_to_b_or_c(obj.getLinExpr())
        except Exception:
            pass
        _add_quad_to_A(obj)
    else:
        try:
            _add_lin_to_b_or_c(obj.getLinExpr())  # type: ignore[attr-defined]
        except Exception:
            pass

    A_obj = _symmetrize(A_obj)

    # Constraints
    A_list: List[np.ndarray] = []
    b_list: List[np.ndarray] = []
    c_list: List[float] = []
    senses: List[str] = []

    # Linear rows
    for cons in model.getConstrs():
        row = model.getRow(cons)
        b_i = np.zeros(n, dtype=float)
        c_shift = 0.0
        sz = _lin_size(row)
        for k in range(sz):
            v = row.getVar(k)
            coef = row.getCoeff(k)
            if v.VarName in fold_vals:
                c_shift += coef * fold_vals[v.VarName]
            else:
                b_i[idx_of(v)] += coef

        A_i = np.zeros((n, n), dtype=float)
        rhs = float(cons.RHS)
        sense = _sense_char(cons.Sense)

        A_list.append(A_i)
        b_list.append(b_i)
        c_list.append(-rhs + c_shift)
        senses.append(sense)

    # Quadratic rows
    for qc in model.getQConstrs():
        quad, lin, s_char, rhs, lin_from_text = _normalize_qcrow(path, model, qc)

        b_i = np.zeros(n, dtype=float)
        c_shift = 0.0

        # Use native LinExpr when available
        if lin is not None:
            sz = _lin_size(lin)
            for k in range(sz):
                v = lin.getVar(k)
                coef = lin.getCoeff(k)
                if v.VarName in fold_vals:
                    c_shift += coef * fold_vals[v.VarName]
                else:
                    b_i[idx_of(v)] += coef
        # Otherwise, use parsed text coeffs
        elif lin_from_text:
            for vname, coef in lin_from_text.items():
                if vname in fold_vals:
                    c_shift += coef * fold_vals[vname]
                elif vname in name_to_idx:
                    b_i[name_to_idx[vname]] += coef
                else:
                    # If names are x1.. and variables are x1.., this won't trigger.
                    # Add more mapping here if your naming differs.
                    pass

        A_i = np.zeros((n, n), dtype=float)
        if quad is not None:
            qsz = _quad_size(quad)
            for k in range(qsz):
                v1 = quad.getVar1(k)
                v2 = quad.getVar2(k)
                val = quad.getCoeff(k)
                if (v1.VarName in fold_vals) or (v2.VarName in fold_vals):
                    continue
                i, j = idx_of(v1), idx_of(v2)
                _accum_quad(A_i, i, j, val)
        A_i = _symmetrize(A_i)

        A_list.append(A_i)
        b_list.append(b_i)
        c_list.append(float(-rhs + c_shift))
        senses.append(s_char)

    return {
        "var_names": [v.VarName for v in vars_],
        "l": l,
        "u": u,
        "A_obj": A_obj,
        "b_obj": b_obj,
        "c_obj": c_obj,
        "A_list": A_list,
        "b_list": b_list,
        "c_list": c_list,
        "senses": senses,
    }

# ------------------------ Post-processing ------------------------ #

def canonicalize_leq(
    A_list: List[np.ndarray],
    b_list: List[np.ndarray],
    c_list: List[float],
    senses: List[str],
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Convert all constraints to <= form.
    Assumes each is 0.5 x^T A x + b^T x + c sense 0 (with c already = -rhs + folds).
    """
    A_leq: List[np.ndarray] = []
    b_leq: List[np.ndarray] = []
    c_leq: List[float] = []
    for A, b, c, s in zip(A_list, b_list, c_list, senses):
        sc = _sense_char(s)
        if sc == '>':
            A_leq.append(-A); b_leq.append(-b); c_leq.append(-c)
        else:
            A_leq.append(A);  b_leq.append(b);  c_leq.append(c)
    return A_leq, b_leq, c_leq
