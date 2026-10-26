import time
import sys
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

import mosek
from mosek.fusion import (
    Model, Domain, Expr, ObjectiveSense, Matrix,
    SolutionStatus, ProblemStatus, AccSolutionStatus
)


def _np_to_fusion_matrix(A: np.ndarray) -> Matrix:
    return Matrix.dense(A.tolist())


class QCQPSolver:
    """
    Canonical form assumed:
        0.5 * x^T A_k x + b_k^T x + c_k
    k=0 is objective, k>=1 are constraints <= 0.
    """

    def __init__(
        self,
        A_list,
        b_list,
        c_list,
        l,
        u,
        use_blocks=True,
        block_tol=1e-8,
        verbose_blocks=True,
        prune_matrices=False,
        min_block_edges=0,
    ):
        self.A_list_raw = [np.asarray(A, dtype=float) for A in A_list]
        self.b_list = [np.asarray(b, dtype=float).reshape(-1) for b in b_list]
        self.c_list = np.array([float(np.asarray(c).reshape(-1)[0]) for c in c_list], dtype=float)

        self.l = np.asarray(l, dtype=float)
        self.u = np.asarray(u, dtype=float)

        self.n = self.l.size
        self.m = len(self.A_list_raw)
        self.results = {}

        self.use_blocks = bool(use_blocks)
        self.verbose_blocks = bool(verbose_blocks)
        self.min_block_edges = int(min_block_edges)

        self.A_model = self._symmetrize_A_list(self.A_list_raw)

        self.graph_tol = float(block_tol) if prune_matrices else 0.0
        if prune_matrices and self.graph_tol > 0:
            self.A_graph = self._prune_A_list(self.A_model, self.graph_tol)
        else:
            self.A_graph = self.A_model

        self._blocks = None
        self._used = None
        self._total_edges = None

    # ------------------------- utilities ------------------------- #
    def _symmetrize_A_list(self, A_list):
        out = []
        for A in A_list:
            if A.size == 0:
                out.append(A)
            else:
                out.append(0.5 * (A + A.T))
        return out

    def _prune_A_list(self, A_list, tol):
        out = []
        for A in A_list:
            if A.size == 0:
                out.append(A)
            else:
                B = A.copy()
                B[np.abs(B) <= tol] = 0.0
                out.append(B)
        return out

    def _compute_blocks(self):
        if self._blocks is not None:
            return self._blocks, self._used, self._total_edges

        n = self.n
        G = nx.Graph()
        G.add_nodes_from(range(n))

        used = np.zeros(n, dtype=bool)
        for A in self.A_model:
            if A.size == 0:
                continue
            rows, cols = np.where(A != 0.0)
            for i, j in zip(rows, cols):
                used[int(i)] = True
                used[int(j)] = True

        for A in self.A_graph:
            if A.size == 0:
                continue
            rows, cols = np.where(A != 0.0)
            for i, j in zip(rows, cols):
                i = int(i)
                j = int(j)
                if i != j:
                    G.add_edge(i, j)

        total_edges = G.number_of_edges()
        components = [sorted(comp) for comp in nx.connected_components(G)]
        blocks = [comp for comp in components if any(used[i] for i in comp)]

        if self.min_block_edges > 0 and total_edges < self.min_block_edges:
            if self.verbose_blocks:
                print("========== PSD BLOCK INFO ==========")
                print(f"#vars:                      {n}")
                print(f"#edges (graph):             {total_edges}")
                print(f"min_block_edges threshold:  {self.min_block_edges}")
                print("Block PSD DISABLED (graph too sparse)")
                print("====================================\n")
            blocks = []

        if self.verbose_blocks and not (self.min_block_edges > 0 and total_edges < self.min_block_edges):
            print("========== PSD BLOCK INFO ==========")
            print(f"#vars:                      {n}")
            print(f"#edges (graph):             {total_edges}")
            print(f"#connected components:      {len(components)}")
            print(f"#blocks used as PSD blocks: {len(blocks)}")
            print(f"Block sizes:                {[len(b) for b in blocks]}")
            if self.graph_tol > 0:
                print(f"graph pruning tol:          {self.graph_tol}")
            print("====================================\n")

        self._blocks = blocks
        self._used = used
        self._total_edges = total_edges
        return blocks, used, total_edges

    def _blocks_compatible_with_model(self, blocks):
        if not blocks:
            return False

        block_id = -np.ones(self.n, dtype=int)
        for bi, B in enumerate(blocks):
            for i in B:
                block_id[int(i)] = bi

        for Ak in self.A_model:
            if Ak.size == 0:
                continue
            rows, cols = np.where(Ak != 0.0)
            for i, j in zip(rows, cols):
                i = int(i)
                j = int(j)
                bi = block_id[i]
                bj = block_id[j]
                if bi != -1 and bj != -1 and bi != bj:
                    return False
        return True

    def _has_square_anywhere(self, i: int, tol: float = 0.0) -> bool:
        for Ak in self.A_model:
            if Ak.size == 0:
                continue
            if abs(Ak[i, i]) > tol:
                return True
        return False

    def _collect_quadratic_pairs(self, tol: float = 0.0):
        diag_idx = set()
        off_pairs = set()  # store (i,j) with i<j
    
        for A in self.A_model:  # IMPORTANT: use A_model (not pruned) so we don't drop real terms
            if A.size == 0:
                continue
    
            # diagonal terms (x_i^2)
            d = np.diag(A)
            for i in np.where(np.abs(d) > tol)[0]:
                diag_idx.add(int(i))
    
            # off-diagonal terms (x_i x_j), take upper triangle only
            rows, cols = np.where(np.abs(np.triu(A, 1)) > tol)
            for i, j in zip(rows, cols):
                off_pairs.add((int(i), int(j)))
    
        diag_idx = sorted(diag_idx)
        off_pairs = sorted(off_pairs)
        return diag_idx, off_pairs


    def _quad_expr_sparse_gurobi(self, A: np.ndarray, X, diag_idx, off_pairs):
        expr = gp.LinExpr()
    
        # 0.5 * sum_i A_ii * X_ii
        for i in diag_idx:
            aii = float(A[i, i])
            if aii != 0.0:
                expr += 0.5 * aii * X[i, i]
    
        # sum_{i<j} A_ij * X_ij
        for (i, j) in off_pairs:
            aij = float(A[i, j])
            if aij != 0.0:
                expr += aij * X[i, j]
    
        return expr

    # ------------------------- LP (McCormick) ------------------------- #
    def solve_lp(self):
        if not (np.all(np.isfinite(self.l)) and np.all(np.isfinite(self.u))):
            raise ValueError("LP requires finite bounds for all variables.")
    
        # sparsity-aware pair collection
        diag_idx, off_pairs = self._collect_quadratic_pairs(tol=0.0)
    
        model = gp.Model()
        model.setParam("OutputFlag", 1)
        model.setParam("TimeLimit", 50)
    
        x = model.addVars(self.n, lb=self.l.tolist(), ub=self.u.tolist(), name="x")
    
        # linear-only shortcut (no quadratic anywhere)
        if len(diag_idx) == 0 and len(off_pairs) == 0:
            b0 = self.b_list[0]
            c0 = float(self.c_list[0])
    
            obj_expr = gp.quicksum(float(b0[i]) * x[i] for i in range(self.n)) + c0
            model.setObjective(obj_expr, GRB.MINIMIZE)
    
            for k in range(1, self.m):
                bk = self.b_list[k]
                ck = float(self.c_list[k])
                expr = gp.quicksum(float(bk[i]) * x[i] for i in range(self.n)) + ck
                model.addConstr(expr <= 0, name=f"qc_{k}")
    
            start = time.perf_counter()
            model.optimize()
            elapsed = time.perf_counter() - start
    
            status = model.Status
            if status == GRB.OPTIMAL:
                res = {"objective": model.ObjVal, "time": elapsed, "status": "optimal"}
            elif status == GRB.UNBOUNDED:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded"}
            elif status == GRB.INFEASIBLE:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible"}
            else:
                res = {"objective": float("nan"), "time": elapsed, "status": f"gurobi_status_{status}"}
    
            self.results["LP"] = res
            return res
    
        # create only needed X_{ij} variables
        keys = [(i, i) for i in diag_idx] + list(off_pairs)
        X = model.addVars(keys, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="X")
    
        # enforce X_ii >= 0 only for diagonal variables we actually created
        for i in diag_idx:
            X[i, i].LB = 0.0
    
        # objective: 0.5*x^T A0 x + b0^T x + c0  (sparsity-aware)
        A0 = self.A_model[0]
        b0 = self.b_list[0]
        c0 = float(self.c_list[0])
    
        quad0 = self._quad_expr_sparse_gurobi(A0, X, diag_idx, off_pairs)
        lin0 = gp.quicksum(float(b0[i]) * x[i] for i in range(self.n))
        model.setObjective(quad0 + lin0 + c0, GRB.MINIMIZE)
    
        # constraints: 0.5*x^T Ak x + bk^T x + ck <= 0  (sparsity-aware)
        for k in range(1, self.m):
            Ak = self.A_model[k]
            bk = self.b_list[k]
            ck = float(self.c_list[k])
    
            quadk = self._quad_expr_sparse_gurobi(Ak, X, diag_idx, off_pairs)
            link = gp.quicksum(float(bk[i]) * x[i] for i in range(self.n))
            model.addConstr(quadk + link + ck <= 0, name=f"qc_{k}")
    
        # McCormick/RLT linking constraints (sparsity-aware, separate diagonal vs bilinear)
    
        # diagonal (x_i^2): tangents + secant envelope
        for i in diag_idx:
            li, ui = float(self.l[i]), float(self.u[i])
            Xi = X[i, i]
    
            # Xi >= 2*li*x - li^2
            model.addConstr(Xi >= 2.0 * li * x[i] - li * li, name=f"sq_tan_l_{i}")
            # Xi >= 2*ui*x - ui^2
            model.addConstr(Xi >= 2.0 * ui * x[i] - ui * ui, name=f"sq_tan_u_{i}")
            # Xi <= (ui+li)*x - ui*li
            model.addConstr(Xi <= (ui + li) * x[i] - ui * li, name=f"sq_env_{i}")
    
        # bilinear (x_i x_j): 4 McCormick inequalities
        for (i, j) in off_pairs:
            li, ui = float(self.l[i]), float(self.u[i])
            lj, uj = float(self.l[j]), float(self.u[j])
            Xij = X[i, j]
    
            model.addConstr(Xij >= lj * x[i] + li * x[j] - li * lj, name=f"mcc1_{i}_{j}")
            model.addConstr(Xij >= uj * x[i] + ui * x[j] - ui * uj, name=f"mcc2_{i}_{j}")
            model.addConstr(Xij <= uj * x[i] + li * x[j] - li * uj, name=f"mcc3_{i}_{j}")
            model.addConstr(Xij <= lj * x[i] + ui * x[j] - ui * lj, name=f"mcc4_{i}_{j}")
    
        start = time.perf_counter()
        model.optimize()
        elapsed = time.perf_counter() - start
    
        status = model.Status
        if status == GRB.OPTIMAL:
            res = {"objective": model.ObjVal, "time": elapsed, "status": "optimal"}
        elif status == GRB.UNBOUNDED:
            res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded"}
        elif status == GRB.INFEASIBLE:
            res = {"objective": float("nan"), "time": elapsed, "status": "infeasible"}
        else:
            res = {"objective": float("nan"), "time": elapsed, "status": f"gurobi_status_{status}"}
    
        self.results["LP"] = res
        return res

    # ------------------------- SDP (Shor) - MOSEK Fusion ------------------------- #
    def solve_sdp(self):
        blocks, used, _ = self._compute_blocks()

        if self.use_blocks and len(blocks) > 0 and not self._blocks_compatible_with_model(blocks):
            if self.verbose_blocks:
                print("Block mode DISABLED: original A_k has cross-block quadratic terms -> monolithic SDP.\n")
            blocks = []

        # linear-only shortcut
        if not np.any(used):
            M = Model("SDP_linear_shortcut")
            if self.verbose_blocks:
                M.setLogHandler(sys.stdout)

            x = M.variable("x", self.n)
            for i in range(self.n):
                if np.isfinite(self.l[i]):
                    M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
                if np.isfinite(self.u[i]):
                    M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

            for k in range(1, self.m):
                bk = self.b_list[k]
                ck = float(self.c_list[k])
                M.constraint(f"qc_{k}", Expr.add(Expr.dot(bk.tolist(), x), ck), Domain.lessThan(0.0))

            b0 = self.b_list[0]
            c0 = float(self.c_list[0])
            M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(b0.tolist(), x), c0))

            M.setSolverParam("optimizerMaxTime", 50.0)
            M.setSolverParam("intpntMaxIterations", 1000)
            M.setSolverParam("intpntCoTolRelGap", 1e-4)

            start = time.perf_counter()
            M.solve()
            elapsed = time.perf_counter() - start

            ps = M.getProblemStatus()
            if ps == ProblemStatus.DualInfeasible:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
            elif ps == ProblemStatus.PrimalInfeasible:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
            else:
                try:
                    obj_val = M.primalObjValue()
                    x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                    res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": None}
                except:
                    res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

            M.dispose()
            self.results["SDP"] = res
            return res

        # ---- Block SDP ----
        if self.use_blocks and len(blocks) > 0:
            big_blocks = [B for B in blocks if len(B) > 1]
            singletons = [B[0] for B in blocks if len(B) == 1]

            M = Model("SDP_block")
            if self.verbose_blocks:
                M.setLogHandler(sys.stdout)

            x = M.variable("x", self.n)

            for i in range(self.n):
                if np.isfinite(self.l[i]):
                    M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
                if np.isfinite(self.u[i]):
                    M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

            Z_blocks = []
            for bi, B in enumerate(big_blocks):
                idx = list(B)
                p = len(idx)
                Zb = M.variable(f"Zb_{bi}", Domain.inPSDCone(p + 1))
                Z_blocks.append((idx, Zb, p))

                M.constraint(f"Zb_{bi}_one", Zb.index(p, p), Domain.equalsTo(1.0))
                for t, i in enumerate(idx):
                    M.constraint(f"link_{bi}_{t}", Expr.sub(Zb.index(t, p), x.index(i)), Domain.equalsTo(0.0))

            S = {}
            for i in singletons:
                if self._has_square_anywhere(i, tol=0.0):
                    s = M.variable(f"s_{i}", 1, Domain.greaterThan(0.0))
                    M.constraint(
                        f"rot_{i}",
                        Expr.vstack(s, Expr.constTerm(1, 0.5), x.index(i)),
                        Domain.inRotatedQCone(3),
                    )
                    S[i] = s

            for k in range(1, self.m):
                Ak = self.A_model[k]
                bk = self.b_list[k]
                ck = float(self.c_list[k])

                quad = None
                for idx, Zb, p in Z_blocks:
                    A_bb = Ak[np.ix_(idx, idx)]
                    if A_bb.size == 0:
                        continue
                    Ck = np.zeros((p + 1, p + 1), dtype=float)
                    Ck[:p, :p] = A_bb
                    term = Expr.dot(_np_to_fusion_matrix(Ck), Zb)
                    quad = term if quad is None else Expr.add(quad, term)

                for i, s in S.items():
                    aii = Ak[i, i]
                    if aii != 0.0:
                        term = Expr.mul(float(aii), s)
                        quad = term if quad is None else Expr.add(quad, term)

                if quad is None:
                    quad = Expr.constTerm(1, 0.0)

                expr = Expr.add(Expr.add(Expr.mul(0.5, quad), Expr.dot(bk.tolist(), x)), ck)
                M.constraint(f"qc_{k}", expr, Domain.lessThan(0.0))

            A0 = self.A_model[0]
            b0 = self.b_list[0]
            c0 = float(self.c_list[0])

            quad0 = None
            for idx, Zb, p in Z_blocks:
                A0_bb = A0[np.ix_(idx, idx)]
                if A0_bb.size == 0:
                    continue
                C0 = np.zeros((p + 1, p + 1), dtype=float)
                C0[:p, :p] = A0_bb
                term = Expr.dot(_np_to_fusion_matrix(C0), Zb)
                quad0 = term if quad0 is None else Expr.add(quad0, term)

            for i, s in S.items():
                aii = A0[i, i]
                if aii != 0.0:
                    term = Expr.mul(float(aii), s)
                    quad0 = term if quad0 is None else Expr.add(quad0, term)

            if quad0 is None:
                quad0 = Expr.constTerm(1, 0.0)

            obj = Expr.add(Expr.add(Expr.mul(0.5, quad0), Expr.dot(b0.tolist(), x)), c0)
            M.objective(ObjectiveSense.Minimize, obj)

            M.setSolverParam("optimizerMaxTime", 50.0)
            M.setSolverParam("intpntMaxIterations", 1000)
            M.setSolverParam("intpntCoTolRelGap", 1e-4)

            start = time.perf_counter()
            M.solve()
            elapsed = time.perf_counter() - start

            ps = M.getProblemStatus()
            if ps == ProblemStatus.DualInfeasible:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
            elif ps == ProblemStatus.PrimalInfeasible:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
            else:
                try:
                    obj_val = M.primalObjValue()
                    x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                    X_full = np.zeros((self.n, self.n), dtype=float)
                    for idx, Zb, p in Z_blocks:
                        Zb_val = np.asarray(Zb.level(), dtype=float).reshape(p + 1, p + 1)
                        X_full[np.ix_(idx, idx)] = Zb_val[:p, :p]
                    for i, s in S.items():
                        X_full[i, i] = float(np.asarray(s.level(), dtype=float).reshape(-1)[0])
                    res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": X_full}
                except:
                    res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

            M.dispose()
            self.results["SDP"] = res
            return res

        # ---- Monolithic SDP ----
        M = Model("SDP_mono")
        if self.verbose_blocks:
            M.setLogHandler(sys.stdout)

        x = M.variable("x", self.n)
        Z = M.variable("Z", Domain.inPSDCone(self.n + 1))

        M.constraint("Z_one", Z.index(self.n, self.n), Domain.equalsTo(1.0))
        for i in range(self.n):
            M.constraint(f"link_{i}", Expr.sub(Z.index(i, self.n), x.index(i)), Domain.equalsTo(0.0))

        for i in range(self.n):
            if np.isfinite(self.l[i]):
                M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
            if np.isfinite(self.u[i]):
                M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

        for k in range(1, self.m):
            Ak = self.A_model[k]
            bk = self.b_list[k]
            ck = float(self.c_list[k])

            Ck = np.zeros((self.n + 1, self.n + 1), dtype=float)
            Ck[:self.n, :self.n] = Ak

            quad = Expr.dot(_np_to_fusion_matrix(Ck), Z)
            expr = Expr.add(Expr.add(Expr.mul(0.5, quad), Expr.dot(bk.tolist(), x)), ck)
            M.constraint(f"qc_{k}", expr, Domain.lessThan(0.0))

        A0 = self.A_model[0]
        b0 = self.b_list[0]
        c0 = float(self.c_list[0])

        C0 = np.zeros((self.n + 1, self.n + 1), dtype=float)
        C0[:self.n, :self.n] = A0

        quad0 = Expr.dot(_np_to_fusion_matrix(C0), Z)
        obj = Expr.add(Expr.add(Expr.mul(0.5, quad0), Expr.dot(b0.tolist(), x)), c0)
        M.objective(ObjectiveSense.Minimize, obj)

        M.setSolverParam("optimizerMaxTime", 50.0)
        M.setSolverParam("intpntMaxIterations", 1000)
        M.setSolverParam("intpntCoTolRelGap", 1e-4)

        start = time.perf_counter()
        M.solve()
        elapsed = time.perf_counter() - start

        ps = M.getProblemStatus()
        if ps == ProblemStatus.DualInfeasible:
            res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
        elif ps == ProblemStatus.PrimalInfeasible:
            res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
        else:
            try:
                obj_val = M.primalObjValue()
                x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                Z_val = np.asarray(M.getVariable("Z").level(), dtype=float).reshape(self.n + 1, self.n + 1)
                res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": Z_val[:self.n, :self.n]}
            except:
                res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

        M.dispose()
        self.results["SDP"] = res
        return res

    # ------------------------- SDP' (Shor + diagonal envelope) - MOSEK Fusion ------------------------- #
    def solve_sdp_prime(self):
        blocks, used, _ = self._compute_blocks()

        if self.use_blocks and len(blocks) > 0 and not self._blocks_compatible_with_model(blocks):
            if self.verbose_blocks:
                print("Block mode DISABLED: original A_k has cross-block quadratic terms -> monolithic SDP'.\n")
            blocks = []

        # linear-only shortcut
        if not np.any(used):
            M = Model("SDPp_linear_shortcut")
            if self.verbose_blocks:
                M.setLogHandler(sys.stdout)

            x = M.variable("x", self.n)

            for i in range(self.n):
                if np.isfinite(self.l[i]):
                    M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
                if np.isfinite(self.u[i]):
                    M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

            for k in range(1, self.m):
                bk = self.b_list[k]
                ck = float(self.c_list[k])
                M.constraint(f"qc_{k}", Expr.add(Expr.dot(bk.tolist(), x), ck), Domain.lessThan(0.0))

            b0 = self.b_list[0]
            c0 = float(self.c_list[0])
            M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(b0.tolist(), x), c0))

            M.setSolverParam("optimizerMaxTime", 50.0)
            M.setSolverParam("intpntMaxIterations", 1000)
            M.setSolverParam("intpntCoTolRelGap", 1e-4)

            start = time.perf_counter()
            M.solve()
            elapsed = time.perf_counter() - start

            ps = M.getProblemStatus()
            if ps == ProblemStatus.DualInfeasible:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
            elif ps == ProblemStatus.PrimalInfeasible:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
            else:
                try:
                    obj_val = M.primalObjValue()
                    x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                    res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": None}
                except:
                    res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

            M.dispose()
            self.results["SDP_PRIME"] = res
            return res

        # ---- Block SDP' ----
        if self.use_blocks and len(blocks) > 0:
            big_blocks = [B for B in blocks if len(B) > 1]
            singletons = [B[0] for B in blocks if len(B) == 1]

            M = Model("SDPp_block")
            if self.verbose_blocks:
                M.setLogHandler(sys.stdout)

            x = M.variable("x", self.n)

            for i in range(self.n):
                if np.isfinite(self.l[i]):
                    M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
                if np.isfinite(self.u[i]):
                    M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

            Z_blocks = []
            for bi, B in enumerate(big_blocks):
                idx = list(B)
                p = len(idx)
                Zb = M.variable(f"Zb_{bi}", Domain.inPSDCone(p + 1))
                Z_blocks.append((idx, Zb, p))

                M.constraint(f"Zb_{bi}_one", Zb.index(p, p), Domain.equalsTo(1.0))
                for t, i in enumerate(idx):
                    M.constraint(f"link_{bi}_{t}", Expr.sub(Zb.index(t, p), x.index(i)), Domain.equalsTo(0.0))

                for t, i in enumerate(idx):
                    li, ui = self.l[i], self.u[i]
                    if np.isfinite(li) and np.isfinite(ui):
                        rhs = Expr.add(Expr.mul(float(ui + li), x.index(i)), -float(ui * li))
                        M.constraint(f"env_{bi}_{t}", Expr.sub(Zb.index(t, t), rhs), Domain.lessThan(0.0))

            S = {}
            for i in singletons:
                if self._has_square_anywhere(i, tol=0.0):
                    s = M.variable(f"s_{i}", 1, Domain.greaterThan(0.0))
                    M.constraint(
                        f"rot_{i}",
                        Expr.vstack(s, Expr.constTerm(1, 0.5), x.index(i)),
                        Domain.inRotatedQCone(3),
                    )
                    li, ui = self.l[i], self.u[i]
                    if np.isfinite(li) and np.isfinite(ui):
                        rhs = Expr.add(Expr.mul(float(ui + li), x.index(i)), -float(ui * li))
                        M.constraint(f"env_s_{i}", Expr.sub(s, rhs), Domain.lessThan(0.0))
                    S[i] = s

            for k in range(1, self.m):
                Ak = self.A_model[k]
                bk = self.b_list[k]
                ck = float(self.c_list[k])

                quad = None
                for idx, Zb, p in Z_blocks:
                    A_bb = Ak[np.ix_(idx, idx)]
                    if A_bb.size == 0:
                        continue
                    Ck = np.zeros((p + 1, p + 1), dtype=float)
                    Ck[:p, :p] = A_bb
                    term = Expr.dot(_np_to_fusion_matrix(Ck), Zb)
                    quad = term if quad is None else Expr.add(quad, term)

                for i, s in S.items():
                    aii = Ak[i, i]
                    if aii != 0.0:
                        term = Expr.mul(float(aii), s)
                        quad = term if quad is None else Expr.add(quad, term)

                if quad is None:
                    quad = Expr.constTerm(1, 0.0)

                expr = Expr.add(Expr.add(Expr.mul(0.5, quad), Expr.dot(bk.tolist(), x)), ck)
                M.constraint(f"qc_{k}", expr, Domain.lessThan(0.0))

            A0 = self.A_model[0]
            b0 = self.b_list[0]
            c0 = float(self.c_list[0])

            quad0 = None
            for idx, Zb, p in Z_blocks:
                A0_bb = A0[np.ix_(idx, idx)]
                if A0_bb.size == 0:
                    continue
                C0 = np.zeros((p + 1, p + 1), dtype=float)
                C0[:p, :p] = A0_bb
                term = Expr.dot(_np_to_fusion_matrix(C0), Zb)
                quad0 = term if quad0 is None else Expr.add(quad0, term)

            for i, s in S.items():
                aii = A0[i, i]
                if aii != 0.0:
                    term = Expr.mul(float(aii), s)
                    quad0 = term if quad0 is None else Expr.add(quad0, term)

            if quad0 is None:
                quad0 = Expr.constTerm(1, 0.0)

            obj = Expr.add(Expr.add(Expr.mul(0.5, quad0), Expr.dot(b0.tolist(), x)), c0)
            M.objective(ObjectiveSense.Minimize, obj)

            M.setSolverParam("optimizerMaxTime", 50.0)
            M.setSolverParam("intpntMaxIterations", 1000)
            M.setSolverParam("intpntCoTolRelGap", 1e-4)

            start = time.perf_counter()
            M.solve()
            elapsed = time.perf_counter() - start

            ps = M.getProblemStatus()
            if ps == ProblemStatus.DualInfeasible:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
            elif ps == ProblemStatus.PrimalInfeasible:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
            else:
                try:
                    obj_val = M.primalObjValue()
                    x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                    X_full = np.zeros((self.n, self.n), dtype=float)
                    for idx, Zb, p in Z_blocks:
                        Zb_val = np.asarray(Zb.level(), dtype=float).reshape(p + 1, p + 1)
                        X_full[np.ix_(idx, idx)] = Zb_val[:p, :p]
                    for i, s in S.items():
                        X_full[i, i] = float(np.asarray(s.level(), dtype=float).reshape(-1)[0])
                    res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": X_full}
                except:
                    res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

            M.dispose()
            self.results["SDP_PRIME"] = res
            return res

        # ---- Monolithic SDP' ----
        M = Model("SDPp_mono")
        if self.verbose_blocks:
            M.setLogHandler(sys.stdout)

        x = M.variable("x", self.n)
        Z = M.variable("Z", Domain.inPSDCone(self.n + 1))

        M.constraint("Z_one", Z.index(self.n, self.n), Domain.equalsTo(1.0))
        for i in range(self.n):
            M.constraint(f"link_{i}", Expr.sub(Z.index(i, self.n), x.index(i)), Domain.equalsTo(0.0))

        for i in range(self.n):
            if np.isfinite(self.l[i]):
                M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
            if np.isfinite(self.u[i]):
                M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

        for i in range(self.n):
            li, ui = self.l[i], self.u[i]
            if np.isfinite(li) and np.isfinite(ui):
                rhs = Expr.add(Expr.mul(float(ui + li), x.index(i)), -float(ui * li))
                M.constraint(f"env_{i}", Expr.sub(Z.index(i, i), rhs), Domain.lessThan(0.0))

        for k in range(1, self.m):
            Ak = self.A_model[k]
            bk = self.b_list[k]
            ck = float(self.c_list[k])

            Ck = np.zeros((self.n + 1, self.n + 1), dtype=float)
            Ck[:self.n, :self.n] = Ak

            quad = Expr.dot(_np_to_fusion_matrix(Ck), Z)
            expr = Expr.add(Expr.add(Expr.mul(0.5, quad), Expr.dot(bk.tolist(), x)), ck)
            M.constraint(f"qc_{k}", expr, Domain.lessThan(0.0))

        A0 = self.A_model[0]
        b0 = self.b_list[0]
        c0 = float(self.c_list[0])

        C0 = np.zeros((self.n + 1, self.n + 1), dtype=float)
        C0[:self.n, :self.n] = A0

        quad0 = Expr.dot(_np_to_fusion_matrix(C0), Z)
        obj = Expr.add(Expr.add(Expr.mul(0.5, quad0), Expr.dot(b0.tolist(), x)), c0)
        M.objective(ObjectiveSense.Minimize, obj)

        M.setSolverParam("optimizerMaxTime", 50.0)
        M.setSolverParam("intpntMaxIterations", 1000)
        M.setSolverParam("intpntCoTolRelGap", 1e-4)

        start = time.perf_counter()
        M.solve()
        elapsed = time.perf_counter() - start

        ps = M.getProblemStatus()
        if ps == ProblemStatus.DualInfeasible:
            res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
        elif ps == ProblemStatus.PrimalInfeasible:
            res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
        else:
            try:
                obj_val = M.primalObjValue()
                x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                Z_val = np.asarray(M.getVariable("Z").level(), dtype=float).reshape(self.n + 1, self.n + 1)
                res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": Z_val[:self.n, :self.n]}
            except:
                res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

        M.dispose()
        self.results["SDP_PRIME"] = res
        return res

        # ------------------------- SDP-RLT (Shor SDP + McCormick/RLT) - MOSEK Fusion ------------------------- #
    def solve_sdp_rlt(self):
        blocks, used, _ = self._compute_blocks()

        if self.use_blocks and len(blocks) > 0 and not self._blocks_compatible_with_model(blocks):
            if self.verbose_blocks:
                print("Block mode DISABLED: original A_k has cross-block quadratic terms -> monolithic SDP-RLT.\n")
            blocks = []

        # collect pairs that actually appear in any A_k (objective+constraints)
        off_pairs = set()
        diag_idx = set()
        for Ak in self.A_model:
            if Ak.size == 0:
                continue
            rows, cols = np.where(Ak != 0.0)
            for i, j in zip(rows, cols):
                i = int(i); j = int(j)
                if i == j:
                    diag_idx.add(i)
                else:
                    a, b = (i, j) if i < j else (j, i)
                    off_pairs.add((a, b))

        # linear-only shortcut (same as others)
        if not np.any(used):
            M = Model("SDPRLT_linear_shortcut")
            if self.verbose_blocks:
                M.setLogHandler(sys.stdout)

            x = M.variable("x", self.n)
            for i in range(self.n):
                if np.isfinite(self.l[i]):
                    M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
                if np.isfinite(self.u[i]):
                    M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

            for k in range(1, self.m):
                bk = self.b_list[k]
                ck = float(self.c_list[k])
                M.constraint(f"qc_{k}", Expr.add(Expr.dot(bk.tolist(), x), ck), Domain.lessThan(0.0))

            b0 = self.b_list[0]
            c0 = float(self.c_list[0])
            M.objective(ObjectiveSense.Minimize, Expr.add(Expr.dot(b0.tolist(), x), c0))

            M.setSolverParam("optimizerMaxTime", 50.0)
            M.setSolverParam("intpntMaxIterations", 1000)
            M.setSolverParam("intpntCoTolRelGap", 1e-4)

            start = time.perf_counter()
            M.solve()
            elapsed = time.perf_counter() - start

            ps = M.getProblemStatus()
            if ps == ProblemStatus.DualInfeasible:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
            elif ps == ProblemStatus.PrimalInfeasible:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
            else:
                try:
                    obj_val = M.primalObjValue()
                    x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                    res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": None}
                except:
                    res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

            M.dispose()
            self.results["SDP_RLT"] = res
            return res

        # ---- Block SDP-RLT ----
        if self.use_blocks and len(blocks) > 0:
            big_blocks = [B for B in blocks if len(B) > 1]
            singletons = [B[0] for B in blocks if len(B) == 1]

            M = Model("SDPRLT_block")
            if self.verbose_blocks:
                M.setLogHandler(sys.stdout)

            x = M.variable("x", self.n)

            for i in range(self.n):
                if np.isfinite(self.l[i]):
                    M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
                if np.isfinite(self.u[i]):
                    M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

            Z_blocks = []
            for bi, B in enumerate(big_blocks):
                idx = list(B)
                p = len(idx)
                pos = {int(v): t for t, v in enumerate(idx)}

                Zb = M.variable(f"Zb_{bi}", Domain.inPSDCone(p + 1))
                Z_blocks.append((idx, pos, Zb, p))

                M.constraint(f"Zb_{bi}_one", Zb.index(p, p), Domain.equalsTo(1.0))
                for t, gi in enumerate(idx):
                    M.constraint(f"link_{bi}_{t}", Expr.sub(Zb.index(t, p), x.index(int(gi))), Domain.equalsTo(0.0))

                # RLT diagonal envelope (x_i^2 <= (u+l)x - ul) on Zb_tt
                for t, gi in enumerate(idx):
                    gi = int(gi)
                    li, ui = self.l[gi], self.u[gi]
                    if np.isfinite(li) and np.isfinite(ui):
                        rhs = Expr.add(Expr.mul(float(ui + li), x.index(gi)), -float(ui * li))
                        M.constraint(f"env_{bi}_{t}", Expr.sub(Zb.index(t, t), rhs), Domain.lessThan(0.0))

                # McCormick on off-diagonals that appear in A_k
                for (i, j) in off_pairs:
                    if i in pos and j in pos:
                        ti, tj = pos[i], pos[j]
                        li, ui = self.l[i], self.u[i]
                        lj, uj = self.l[j], self.u[j]
                        if not (np.isfinite(li) and np.isfinite(ui) and np.isfinite(lj) and np.isfinite(uj)):
                            continue

                        Zij = Zb.index(ti, tj)

                        rhs1 = Expr.add(Expr.add(Expr.mul(float(lj), x.index(i)),
                                                 Expr.mul(float(li), x.index(j))),
                                        -float(li * lj))
                        rhs2 = Expr.add(Expr.add(Expr.mul(float(uj), x.index(i)),
                                                 Expr.mul(float(ui), x.index(j))),
                                        -float(ui * uj))
                        rhs3 = Expr.add(Expr.add(Expr.mul(float(uj), x.index(i)),
                                                 Expr.mul(float(li), x.index(j))),
                                        -float(li * uj))
                        rhs4 = Expr.add(Expr.add(Expr.mul(float(lj), x.index(i)),
                                                 Expr.mul(float(ui), x.index(j))),
                                        -float(ui * lj))

                        M.constraint(f"mcc1_{bi}_{ti}_{tj}", Expr.sub(Zij, rhs1), Domain.greaterThan(0.0))
                        M.constraint(f"mcc2_{bi}_{ti}_{tj}", Expr.sub(Zij, rhs2), Domain.greaterThan(0.0))
                        M.constraint(f"mcc3_{bi}_{ti}_{tj}", Expr.sub(Zij, rhs3), Domain.lessThan(0.0))
                        M.constraint(f"mcc4_{bi}_{ti}_{tj}", Expr.sub(Zij, rhs4), Domain.lessThan(0.0))

            S = {}
            for i in singletons:
                i = int(i)
                if self._has_square_anywhere(i, tol=0.0):
                    s = M.variable(f"s_{i}", 1, Domain.greaterThan(0.0))
                    M.constraint(
                        f"rot_{i}",
                        Expr.vstack(s, Expr.constTerm(1, 0.5), x.index(i)),
                        Domain.inRotatedQCone(3),
                    )
                    # RLT diagonal envelope on s (x_i^2 <= (u+l)x - ul)
                    li, ui = self.l[i], self.u[i]
                    if np.isfinite(li) and np.isfinite(ui):
                        rhs = Expr.add(Expr.mul(float(ui + li), x.index(i)), -float(ui * li))
                        M.constraint(f"env_s_{i}", Expr.sub(s, rhs), Domain.lessThan(0.0))
                    S[i] = s

            for k in range(1, self.m):
                Ak = self.A_model[k]
                bk = self.b_list[k]
                ck = float(self.c_list[k])

                quad = None
                for idx, pos, Zb, p in Z_blocks:
                    A_bb = Ak[np.ix_(idx, idx)]
                    if A_bb.size == 0:
                        continue
                    Ck = np.zeros((p + 1, p + 1), dtype=float)
                    Ck[:p, :p] = A_bb
                    term = Expr.dot(_np_to_fusion_matrix(Ck), Zb)
                    quad = term if quad is None else Expr.add(quad, term)

                for i, s in S.items():
                    aii = Ak[i, i]
                    if aii != 0.0:
                        term = Expr.mul(float(aii), s)
                        quad = term if quad is None else Expr.add(quad, term)

                if quad is None:
                    quad = Expr.constTerm(1, 0.0)

                expr = Expr.add(Expr.add(Expr.mul(0.5, quad), Expr.dot(bk.tolist(), x)), ck)
                M.constraint(f"qc_{k}", expr, Domain.lessThan(0.0))

            A0 = self.A_model[0]
            b0 = self.b_list[0]
            c0 = float(self.c_list[0])

            quad0 = None
            for idx, pos, Zb, p in Z_blocks:
                A0_bb = A0[np.ix_(idx, idx)]
                if A0_bb.size == 0:
                    continue
                C0 = np.zeros((p + 1, p + 1), dtype=float)
                C0[:p, :p] = A0_bb
                term = Expr.dot(_np_to_fusion_matrix(C0), Zb)
                quad0 = term if quad0 is None else Expr.add(quad0, term)

            for i, s in S.items():
                aii = A0[i, i]
                if aii != 0.0:
                    term = Expr.mul(float(aii), s)
                    quad0 = term if quad0 is None else Expr.add(quad0, term)

            if quad0 is None:
                quad0 = Expr.constTerm(1, 0.0)

            obj = Expr.add(Expr.add(Expr.mul(0.5, quad0), Expr.dot(b0.tolist(), x)), c0)
            M.objective(ObjectiveSense.Minimize, obj)

            M.setSolverParam("optimizerMaxTime", 50.0)
            M.setSolverParam("intpntMaxIterations", 1000)
            M.setSolverParam("intpntCoTolRelGap", 1e-4)

            start = time.perf_counter()
            M.solve()
            elapsed = time.perf_counter() - start

            ps = M.getProblemStatus()
            if ps == ProblemStatus.DualInfeasible:
                res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
            elif ps == ProblemStatus.PrimalInfeasible:
                res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
            else:
                try:
                    obj_val = M.primalObjValue()
                    x_val = np.asarray(M.getVariable("x").level(), dtype=float)

                    X_full = np.zeros((self.n, self.n), dtype=float)
                    for idx, pos, Zb, p in Z_blocks:
                        Zb_val = np.asarray(Zb.level(), dtype=float).reshape(p + 1, p + 1)
                        X_full[np.ix_(idx, idx)] = Zb_val[:p, :p]
                    for i, s in S.items():
                        X_full[i, i] = float(np.asarray(s.level(), dtype=float).reshape(-1)[0])

                    res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": X_full}
                except:
                    res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

            M.dispose()
            self.results["SDP_RLT"] = res
            return res

        # ---- Monolithic SDP-RLT ----
        M = Model("SDPRLT_mono")
        if self.verbose_blocks:
            M.setLogHandler(sys.stdout)

        x = M.variable("x", self.n)
        Z = M.variable("Z", Domain.inPSDCone(self.n + 1))

        M.constraint("Z_one", Z.index(self.n, self.n), Domain.equalsTo(1.0))
        for i in range(self.n):
            M.constraint(f"link_{i}", Expr.sub(Z.index(i, self.n), x.index(i)), Domain.equalsTo(0.0))

        for i in range(self.n):
            if np.isfinite(self.l[i]):
                M.constraint(f"l_{i}", x.index(i), Domain.greaterThan(float(self.l[i])))
            if np.isfinite(self.u[i]):
                M.constraint(f"u_{i}", x.index(i), Domain.lessThan(float(self.u[i])))

        # RLT diagonal envelope on Z_ii
        for i in diag_idx:
            li, ui = self.l[i], self.u[i]
            if np.isfinite(li) and np.isfinite(ui):
                rhs = Expr.add(Expr.mul(float(ui + li), x.index(i)), -float(ui * li))
                M.constraint(f"env_{i}", Expr.sub(Z.index(i, i), rhs), Domain.lessThan(0.0))

        # McCormick on off-diagonals appearing in A_k
        for (i, j) in off_pairs:
            li, ui = self.l[i], self.u[i]
            lj, uj = self.l[j], self.u[j]
            if not (np.isfinite(li) and np.isfinite(ui) and np.isfinite(lj) and np.isfinite(uj)):
                continue

            Zij = Z.index(i, j)

            rhs1 = Expr.add(Expr.add(Expr.mul(float(lj), x.index(i)),
                                     Expr.mul(float(li), x.index(j))),
                            -float(li * lj))
            rhs2 = Expr.add(Expr.add(Expr.mul(float(uj), x.index(i)),
                                     Expr.mul(float(ui), x.index(j))),
                            -float(ui * uj))
            rhs3 = Expr.add(Expr.add(Expr.mul(float(uj), x.index(i)),
                                     Expr.mul(float(li), x.index(j))),
                            -float(li * uj))
            rhs4 = Expr.add(Expr.add(Expr.mul(float(lj), x.index(i)),
                                     Expr.mul(float(ui), x.index(j))),
                            -float(ui * lj))

            M.constraint(f"mcc1_{i}_{j}", Expr.sub(Zij, rhs1), Domain.greaterThan(0.0))
            M.constraint(f"mcc2_{i}_{j}", Expr.sub(Zij, rhs2), Domain.greaterThan(0.0))
            M.constraint(f"mcc3_{i}_{j}", Expr.sub(Zij, rhs3), Domain.lessThan(0.0))
            M.constraint(f"mcc4_{i}_{j}", Expr.sub(Zij, rhs4), Domain.lessThan(0.0))

        for k in range(1, self.m):
            Ak = self.A_model[k]
            bk = self.b_list[k]
            ck = float(self.c_list[k])

            Ck = np.zeros((self.n + 1, self.n + 1), dtype=float)
            Ck[:self.n, :self.n] = Ak

            quad = Expr.dot(_np_to_fusion_matrix(Ck), Z)
            expr = Expr.add(Expr.add(Expr.mul(0.5, quad), Expr.dot(bk.tolist(), x)), ck)
            M.constraint(f"qc_{k}", expr, Domain.lessThan(0.0))

        A0 = self.A_model[0]
        b0 = self.b_list[0]
        c0 = float(self.c_list[0])

        C0 = np.zeros((self.n + 1, self.n + 1), dtype=float)
        C0[:self.n, :self.n] = A0

        quad0 = Expr.dot(_np_to_fusion_matrix(C0), Z)
        obj = Expr.add(Expr.add(Expr.mul(0.5, quad0), Expr.dot(b0.tolist(), x)), c0)
        M.objective(ObjectiveSense.Minimize, obj)

        M.setSolverParam("optimizerMaxTime", 50.0)
        M.setSolverParam("intpntMaxIterations", 1000)
        M.setSolverParam("intpntCoTolRelGap", 1e-4)

        start = time.perf_counter()
        M.solve()
        elapsed = time.perf_counter() - start

        ps = M.getProblemStatus()
        if ps == ProblemStatus.DualInfeasible:
            res = {"objective": float("-inf"), "time": elapsed, "status": "unbounded", "x": None, "X": None}
        elif ps == ProblemStatus.PrimalInfeasible:
            res = {"objective": float("nan"), "time": elapsed, "status": "infeasible", "x": None, "X": None}
        else:
            try:
                obj_val = M.primalObjValue()
                x_val = np.asarray(M.getVariable("x").level(), dtype=float)
                Z_val = np.asarray(M.getVariable("Z").level(), dtype=float).reshape(self.n + 1, self.n + 1)
                res = {"objective": obj_val, "time": elapsed, "status": "optimal", "x": x_val, "X": Z_val[:self.n, :self.n]}
            except:
                res = {"objective": float("nan"), "time": elapsed, "status": "unknown", "x": None, "X": None}

        M.dispose()
        self.results["SDP_RLT"] = res
        return res

    # ------------------------- dispatch ------------------------- #
    def solve(self, method="LP"):
        m = method.upper()
        if m == "LP":
            return self.solve_lp()
        if m == "SDP":
            return self.solve_sdp()
        if m in ("SDP_PRIME", "SDP2"):
            return self.solve_sdp_prime()
        if m in ("SDP_RLT", "RLT_SDP", "SDPRLT"):
            return self.solve_sdp_rlt()
        raise ValueError(f"Unknown relaxation method '{method}'")
