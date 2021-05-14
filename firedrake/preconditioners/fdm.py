from functools import lru_cache
import numpy as np

from pyop2 import op2
from pyop2.sparsity import get_preallocation

from ufl import FiniteElement, VectorElement, TensorElement, Jacobian, JacobianInverse
from ufl import as_tensor, diag_vector, dot, dx, indices, inner, inv
from ufl.algorithms.ad import expand_derivatives

from firedrake.petsc import PETSc
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.base import PCBase
from firedrake.utils import IntType_c
from firedrake.dmhooks import get_function_space, get_appctx
import firedrake


class FDMPC(PCBase):

    _prefix = "fdm_"

    def initialize(self, pc):
        A, P = pc.getOperators()

        # Read options
        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options(options_prefix)
        fdm_type = opts.getString("type", default="affine")

        dm = pc.getDM()
        V = get_function_space(dm)

        ele = V.ufl_element()
        if isinstance(ele, firedrake.TensorProductElement):
            family = set(e.family() for e in ele.sub_elements())
        else:
            family = {ele.family()}
        needs_interior_facet = family <= {"DQ", "Discontinuous Lagrange"}

        self.mesh = V.mesh()
        self.uf = firedrake.Function(V)
        self.uc = firedrake.Function(V)

        ndim = self.mesh.topological_dimension()
        nscal = V.value_size
        N = V.ufl_element().degree()
        try:
            N, = set(N)
        except TypeError:
            pass
        Nq = 2 * N + 1

        # Get problem solution and bcs
        solverctx = get_appctx(dm)
        self.u = solverctx._problem.u
        self.bcs = solverctx.bcs_F

        if len(self.bcs) > 0:
            self.bc_nodes = np.unique(np.concatenate([bcdofs(bc, ghost=False)
                                                      for bc in self.bcs]))
        else:
            self.bc_nodes = np.empty(0, dtype=PETSc.IntType)

        bcflags = self.get_bc_flags(V, self.mesh, self.bcs, solverctx._problem.J)

        self.weight = self.multiplicity(V)
        with self.weight.dat.vec as w:
            w.reciprocal()

        # Get problem coefficients
        appctx = self.get_appctx(pc)
        mu = appctx.get("viscosity", None)  # sets the viscosity
        helm = appctx.get("helm", None)  # sets the potential
        hflag = helm is not None

        self.restrict_kernel, self.prolong_kernel, self.stencil_kernel, Afdm, Dfdm = self.assemble_matfree(ndim, nscal, N, Nq, needs_interior_facet, hflag)

        self.stencil = None
        if fdm_type == "stencil":
            # Compute high-order PDE coefficients and only extract
            # nonzeros from the diagonal and interface neighbors
            # Vertex-vertex couplings are ignored here,
            # so this should work as direct solver only on star patches
            W = firedrake.VectorFunctionSpace(self.mesh, "DG" if ndim == 1 else "DQ", N, dim=2*ndim+1)
            self.stencil = firedrake.Function(W)
            Gq, Bq = self.assemble_coef(mu, helm, Nq)
            Pmat = self.assemble_stencil(A, V, Gq, Bq, N, bcflags)
        elif fdm_type == "affine":
            # Compute low-order PDE coefficients, such that the FDM
            # sparsifies the assembled matrix
            Gq, Bq = self.assemble_coef(mu, helm, 0, diagonal=True)
            Pmat = self.assemble_affine(A, V, Gq, Bq, Afdm, Dfdm, bcflags)
        else:
            raise ValueError("Unknown fdm_type")

        opc = pc
        # Internally, we just set up a PC object that the user can configure
        # however from the PETSc command line.  Since PC allows the user to specify
        # a KSP, we can do iterative by -fdm_pc_type ksp.
        pc = PETSc.PC().create(comm=opc.comm)
        pc.incrementTabLevel(1, parent=opc)

        dm = opc.getDM()
        pc.setDM(dm)
        pc.setOptionsPrefix(options_prefix)
        pc.setOperators(Pmat, Pmat)
        self.pc = pc
        pc.setFromOptions()

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        pass

    def apply(self, pc, x, y):
        self.uc.assign(firedrake.zero())

        with self.uf.dat.vec_wo as xf:
            x.copy(xf)

        op2.par_loop(self.restrict_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.INC, self.uc.cell_node_map()),
                     self.uf.dat(op2.READ, self.uf.cell_node_map()),
                     self.weight.dat(op2.READ, self.weight.cell_node_map()))

        for bc in self.bcs:
            bc.zero(self.uc)

        with self.uc.dat.vec as x_, self.uf.dat.vec as y_:
            self.pc.apply(x_, y_)

        for bc in self.bcs:
            bc.zero(self.uf)

        op2.par_loop(self.prolong_kernel, self.mesh.cell_set,
                     self.uc.dat(op2.WRITE, self.uc.cell_node_map()),
                     self.uf.dat(op2.READ, self.uf.cell_node_map()))

        with self.uc.dat.vec_ro as xc:
            xc.copy(y)

        y.array_w[self.bc_nodes] = x.array_r[self.bc_nodes]

    def view(self, pc, viewer=None):
        super(FDMPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("PC to apply inverse\n")
            self.pc.view(viewer)

    @staticmethod
    def pull_facet(x, pshape, fdata):
        return np.reshape(np.moveaxis(np.reshape(x.copy(), pshape), fdata // 2, 0), x.shape)

    @staticmethod
    def index_bcs(x, pshape, bc, val):
        xshape = x.shape
        x.shape = pshape
        if bc[0]:
            x[0, ...] = val
        if bc[1]:
            x[-1, ...] = val
        if len(pshape) >= 2:
            if bc[2]:
                x[:, 0, ...] = val
            if bc[3]:
                x[:, -1, ...] = val
        if len(pshape) >= 3:
            if bc[4]:
                x[:, :, 0, ...] = val
            if bc[5]:
                x[:, :, -1, ...] = val
        x.shape = xshape
        return

    def assemble_stencil(self, A, V, Gq, Bq, N, bcflags):
        assert V.value_size == 1

        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map([])
        # TODO implement stencil for IPDG
        strong = 1

        lexico_cg, nel = self.glonum_fun(V)
        lexico_dg, _ = self.glonum_fun(self.stencil)

        ndim = V.mesh().topological_dimension()
        ndof_cell = V.cell_node_list.shape[1]
        nx1 = N + 1
        pshape = (nx1,)*ndim

        self.stencil.assign(firedrake.zero())
        # FIXME I don't know how to use optional arguments here, maybe a MixedFunctionSpace
        if Bq is not None:
            op2.par_loop(self.stencil_kernel, self.mesh.cell_set,
                         self.stencil.dat(op2.WRITE, self.stencil.cell_node_map()),
                         Gq.dat(op2.READ, Gq.cell_node_map()),
                         Bq.dat(op2.READ, Bq.cell_node_map()))
        else:
            op2.par_loop(self.stencil_kernel, self.mesh.cell_set,
                         self.stencil.dat(op2.WRITE, self.stencil.cell_node_map()),
                         Gq.dat(op2.READ, Gq.cell_node_map()))

        # Connectivity graph between the nodes within a cell
        i = np.arange(ndof_cell, dtype=PETSc.IntType)
        sx = i - (i % nx1)
        sy = i - ((i // nx1) % nx1) * nx1
        if ndim == 1:
            graph = np.array([sx, sx+(nx1-1)])
        elif ndim == 2:
            graph = np.array([sy, sy+(nx1-1)*nx1, sx, sx+(nx1-1)])
        else:
            sz = i - (((i // nx1) // nx1) % nx1) * nx1 * nx1
            graph = np.array([sz, sz+(nx1-1)*nx1*nx1, sy, sy+(nx1-1)*nx1, sx, sx+(nx1-1)])

        ondiag = (graph == i).T
        graph = graph.T

        prealloc = PETSc.Mat().create(comm=A.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(A.getSizes())
        prealloc.setUp()

        aones = np.ones(graph.shape[1], dtype=PETSc.RealType)
        for e in range(nel):
            ie = lgmap.apply(lexico_cg(e))

            # Preallocate diagonal
            for row in ie:
                prealloc.setValue(row, row, 1.0E0)

            # Preallocate off-diagonal
            self.index_bcs(ie, pshape, bcflags[e] == strong, -1)
            je = ie[graph]
            je[ondiag] = -1
            for row, cols in zip(ie, je):
                prealloc.setValues(row, cols, aones)
                prealloc.setValues(cols, row, aones)

        prealloc.assemble()
        nnz = get_preallocation(prealloc, V.dof_dset.set.size)
        Pmat = PETSc.Mat().createAIJ(A.getSizes(), nnz=nnz, comm=A.comm)
        Pmat.setLGMap(lgmap, lgmap)
        Pmat.zeroEntries()
        for e in range(nel):
            ie = lgmap.apply(lexico_cg(e))
            vals = self.stencil.dat.data_ro[lexico_dg(e)]

            # Assemble diagonal
            for row, aij in zip(ie, vals):
                Pmat.setValue(row, row, aij[0], imode)

            # Assemble off-diagonal
            self.index_bcs(ie, pshape, bcflags[e] == strong, -1)
            je = ie[graph]
            je[ondiag] = -1
            for row, cols, aij in zip(ie, je, vals):
                Pmat.setValues(row, cols, aij[1:], imode)
                Pmat.setValues(cols, row, aij[1:], imode)

        Pmat.assemble()
        return Pmat

    def assemble_affine(self, A, V, Gq, Bq, Afdm, Dfdm, bcflags):
        from scipy.sparse import kron, csr_matrix

        needs_interior_facet = Dfdm is not None
        strong = -1 if needs_interior_facet else 1

        imode = PETSc.InsertMode.ADD_VALUES
        lgmap = V.local_to_global_map([])

        lexico_cell, nel = self.glonum_fun(V)
        gid, _ = self.glonum_fun(Gq)
        bid, _ = self.glonum_fun(Bq) if Bq is not None else (None, nel)

        ele = V.ufl_element()
        ncomp = ele.value_size()
        bsize = V.value_size
        needs_hdiv = bsize != ncomp

        ndof = bsize * V.dof_dset.set.size
        ndim = V.mesh().topological_dimension()
        idsym = None

        gele = Gq.ufl_element()
        if gele.reference_value_size() != gele.value_size():
            # We need to extract the diagonal of a symmetric tensor
            idsym = [0, ndim, 2*ndim-1]
            idsym = idsym[:ndim]

        nx1 = Afdm[0][0].shape[0]
        pshape = (nx1,)*ndim

        prealloc = PETSc.Mat().create(comm=A.comm)
        prealloc.setType(PETSc.Mat.Type.PREALLOCATOR)
        prealloc.setSizes(A.getSizes())
        prealloc.setUp()

        # Build elemental sparse matrices and preallocate matrix
        cell_csr = []
        facet_csr = []
        flag2id = np.kron(np.eye(ndim, ndim, dtype=PETSc.IntType), [[1], [3]])

        for e in range(nel):
            ie = lexico_cell(e)
            if needs_hdiv:
                ie = np.reshape(ie, (ncomp, -1))

            mue = np.atleast_1d(np.sum(Gq.dat.data_ro[gid(e)], axis=0))
            bce = bcflags[e]
            for j in range(ncomp):
                bcj = bce[j] if len(bce.shape) == 2 else bce
                muj = mue[j] if len(mue.shape) == 2 else mue
                if idsym:
                    muj = muj[idsym]

                fbc = bcj @ flag2id
                be = Afdm[fbc[0]][1]
                ae = Afdm[fbc[0]][0] * muj[0]
                if Bq is not None:
                    ae += be * sum(Bq.dat.data_ro[bid(e)])

                if ndim > 1:
                    ae = kron(ae, Afdm[fbc[1]][1], format="csr")
                    ae += kron(be, Afdm[fbc[1]][0] * muj[1], format="csr")
                    if ndim > 2:
                        be = kron(be, Afdm[fbc[1]][1], format="csr")
                        ae = kron(ae, Afdm[fbc[2]][1], format="csr")
                        ae += kron(be, Afdm[fbc[2]][0] * muj[2], format="csr")

                cell_csr.append(ae)

                rows = lgmap.apply(ie[j] if needs_hdiv else j+bsize*ie)

                for row in rows:
                    prealloc.setValue(row, row, 1.0E0, imode)

                self.index_bcs(rows, pshape, bcj == strong, -1)
                cols = rows[ae.indices]
                for i, row in enumerate(rows):
                    i0 = ae.indptr[i]
                    i1 = ae.indptr[i+1]
                    prealloc.setValues(row, cols[i0:i1], ae.data[i0:i1], imode)

        if needs_interior_facet:
            eta = nx1*(nx1+1)
            adense = np.zeros((nx1, nx1))

            # TODO extrude these arrays
            lexico_facet, nfacet = self.glonum_fun(V, interior_facet=True)
            facet_cells = self.mesh.interior_facets.facet_cell_map.values
            facet_data = self.mesh.interior_facets.local_facet_dat.data

            for f in range(nfacet):
                e0, e1 = facet_cells[f]
                k0 = -(facet_data[f, 0] % 2)
                k1 = -(facet_data[f, 1] % 2)
                idir = facet_data[f] // 2
                if idsym:
                    idir = idsym[idir]

                mu0 = np.atleast_1d(np.sum(Gq.dat.data_ro[gid(e0)], axis=0))
                mu1 = np.atleast_1d(np.sum(Gq.dat.data_ro[gid(e1)], axis=0))
                ie = lexico_facet(f)
                if needs_hdiv:
                    ie = np.reshape(ie, (ncomp, -1))

                for j in range(ncomp):
                    mu0j = mu0[j][idir[0]] if len(mu0.shape) > 1 else mu0[idir[0]]
                    mu1j = mu1[j][idir[1]] if len(mu1.shape) > 1 else mu1[idir[1]]

                    adense.fill(0.0E0)
                    adense[:, k1] += (0.5E0*mu0j) * Dfdm[:, k0]
                    adense[k0, :] += (0.5E0*mu1j) * Dfdm[:, k1]
                    adense[k0, k1] -= eta * 0.5E0*(mu0j + mu1j)
                    ae = csr_matrix(adense)
                    if ndim > 1:
                        # Here we are assuming that the mesh is oriented
                        ae = kron(ae, Afdm[0][1], format="csr")
                        if ndim > 2:
                            ae = kron(ae, Afdm[0][1], format="csr")

                    facet_csr.append(ae)
                    icell = np.reshape(lgmap.apply(ie[j] if needs_hdiv else j+bsize*ie), (2, -1))
                    rows = self.pull_facet(icell[0], pshape, facet_data[f, 0])
                    cols = self.pull_facet(icell[1], pshape, facet_data[f, 1])
                    cols = cols[ae.indices]
                    for i, row in enumerate(rows):
                        i0 = ae.indptr[i]
                        i1 = ae.indptr[i+1]
                        prealloc.setValues(row, cols[i0:i1], ae.data[i0:i1], imode)
                        prealloc.setValues(cols[i0:i1], row, ae.data[i0:i1], imode)

        prealloc.assemble()
        nnz = get_preallocation(prealloc, ndof)
        Pmat = PETSc.Mat().createAIJ(A.getSizes(), nnz=nnz, comm=A.comm)
        Pmat.setBlockSize(bsize)
        Pmat.setLGMap(lgmap, lgmap)
        Pmat.zeroEntries()

        # Assemble global matrix
        for e, ae in enumerate(cell_csr):
            j = e % ncomp
            if j == 0:
                bce = bcflags[e // ncomp]
                ie = lexico_cell(e // ncomp)
                if needs_hdiv:
                    ie = np.reshape(ie, (ncomp, -1))

            bcj = bce[e % ncomp] if len(bce.shape) == 2 else bce
            rows = lgmap.apply(ie[j] if needs_hdiv else j+bsize*ie)
            ibc = rows.copy()

            self.index_bcs(rows, pshape, bcj == strong, -1)
            adiag = ae.diagonal()
            adiag = adiag[rows == -1]
            ibc = ibc[rows == -1]
            for row, val in zip(ibc, adiag):
                Pmat.setValue(row, row, val, imode)

            cols = rows[ae.indices]
            for i, row in enumerate(rows):
                i0 = ae.indptr[i]
                i1 = ae.indptr[i+1]
                Pmat.setValues(row, cols[i0:i1], ae.data[i0:i1], imode)

        if needs_interior_facet:
            for f, ae in enumerate(facet_csr):
                j = f % ncomp
                if j == 0:
                    bce = bcflags[f // ncomp]
                    ie = lexico_facet(f // ncomp)
                    if needs_hdiv:
                        ie = np.reshape(ie, (ncomp, -1))

                icell = np.reshape(lgmap.apply(ie[j] if needs_hdiv else j+bsize*ie), (2, -1))
                rows = self.pull_facet(icell[0], pshape, facet_data[f // ncomp, 0])
                cols = self.pull_facet(icell[1], pshape, facet_data[f // ncomp, 1])
                cols = cols[ae.indices]
                for i, row in enumerate(rows):
                    i0 = ae.indptr[i]
                    i1 = ae.indptr[i+1]
                    Pmat.setValues(row, cols[i0:i1], ae.data[i0:i1], imode)
                    Pmat.setValues(cols[i0:i1], row, ae.data[i0:i1], imode)

        Pmat.assemble()
        return Pmat

    def assemble_coef(self, mu, helm, Nq=0, diagonal=False):
        ndim = self.mesh.topological_dimension()
        gdim = self.mesh.geometric_dimension()
        gshape = (ndim, ndim)

        if gdim == ndim:
            Finv = JacobianInverse(self.mesh)
            if mu is None:
                G = dot(Finv, Finv.T)
            elif mu.ufl_shape == gshape:
                G = dot(dot(Finv, mu), Finv.T)
            elif mu.ufl_shape == gshape + gshape:
                i1, i2, i3, i4, j1, j3 = indices(6)
                G = as_tensor(Finv[i1, j1] * Finv[i3, j3] * mu[j1, i2, j3, i4], (i1, i2, i3, i4))
            elif mu.ufl_shape == ():
                G = mu * dot(Finv, Finv.T)
            else:
                raise ValueError("I don't know how to ")
        else:
            F = Jacobian(self.mesh)
            G = inv(dot(F.T, F))
            if mu:
                G = mu * G
            # I don't know how to use tensor viscosity on embedded manifolds

        if diagonal:
            if len(G.ufl_shape) == 2:
                G = diag_vector(G)
                Qe = VectorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default", dim=np.prod(G.ufl_shape))
            elif len(G.ufl_shape) == 4:
                G = as_tensor([[G[i, j, i, j] for i in range(G.ufl_shape[0])] for j in range(G.ufl_shape[1])])
                Qe = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                                   quad_scheme="default", shape=G.ufl_shape)
            else:
                raise ValueError("I don't know how to get the diagonal of a tensor of shape ", G.ufl_shape)
        else:
            Qe = TensorElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                               quad_scheme="default", shape=G.ufl_shape, symmetry=True)

        Q = firedrake.FunctionSpace(self.mesh, Qe)
        q = firedrake.TestFunction(Q)
        Gq = firedrake.assemble(inner(G, q)*dx(degree=Nq))

        if helm is None:
            Bq = None
        else:
            Qe = FiniteElement("Quadrature", self.mesh.ufl_cell(), degree=Nq,
                               quad_scheme="default")
            Q = firedrake.FunctionSpace(self.mesh, Qe)
            q = firedrake.TestFunction(Q)
            Bq = firedrake.assemble(inner(helm, q)*dx(degree=Nq))

        return Gq, Bq

    @staticmethod
    @lru_cache(maxsize=10)
    def semhat(N, Nq):
        from FIAT.reference_element import UFCInterval
        from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
        from FIAT.quadrature import GaussLegendreQuadratureLineRule
        cell = UFCInterval()
        elem = GaussLobattoLegendre(cell, N)
        rule = GaussLegendreQuadratureLineRule(cell, (Nq + 2) // 2)
        basis = elem.tabulate(1, rule.get_points())
        Jhat = basis[(0,)]
        Dhat = basis[(1,)]
        what = rule.get_weights()
        Ahat = Dhat @ np.diag(what) @ Dhat.T
        Bhat = Jhat @ np.diag(what) @ Jhat.T
        return Ahat, Bhat, Jhat, Dhat, what

    @staticmethod
    def fdm_cg(Ahat, Bhat):
        from scipy.linalg import eigh
        from scipy.sparse import csr_matrix
        rd = (0, -1)
        kd = slice(1, -1)
        Vfdm = np.eye(Ahat.shape[0])
        _, Vfdm[kd, kd] = eigh(Ahat[kd, kd], Bhat[kd, kd])
        Vfdm[kd, rd] = -Vfdm[kd, kd] @ ((Vfdm[kd, kd].T @ Bhat[kd, rd]) @ Vfdm[np.ix_(rd, rd)])

        def apply_strong_bcs(Ahat, Bhat, bc0, bc1):
            k0 = 0 if bc0 == 1 else 1
            k1 = Ahat.shape[0] if bc1 == 1 else -1
            ki = slice(k0, k1)
            A = Ahat.copy()
            a = A.diagonal().copy()
            A[ki, ki] = 0.0E0
            np.fill_diagonal(A, a)

            B = Bhat.copy()
            b = B.diagonal().copy()
            B[ki, ki] = 0.0E0
            np.fill_diagonal(B, b)
            return [csr_matrix(A), csr_matrix(B)]

        Afdm = []
        Ak = Vfdm.T @ Ahat @ Vfdm
        Bk = Vfdm.T @ Bhat @ Vfdm
        Bk[rd, kd] = 0.0E0
        Bk[kd, rd] = 0.0E0
        for bc1 in range(3):
            for bc0 in range(3):
                Afdm.append(apply_strong_bcs(Ak, Bk, bc0, bc1))

        return Afdm, Vfdm, None

    @staticmethod
    def fdm_ipdg(Ahat, Bhat, N, eta=None):
        from scipy.linalg import eigh
        from scipy.sparse import csr_matrix
        from FIAT.reference_element import UFCInterval
        from FIAT.gauss_lobatto_legendre import GaussLobattoLegendre
        from FIAT.quadrature import GaussLegendreQuadratureLineRule

        if eta is None:
            eta = (N+1)*(N+2)

        cell = UFCInterval()
        elem = GaussLobattoLegendre(cell, N)

        # Interpolation onto GL nodes
        rule = GaussLegendreQuadratureLineRule(cell, N + 1)
        basis = elem.tabulate(0, rule.get_points())
        Jipdg = basis[(0,)]

        # Facet normal derivatives
        basis = elem.tabulate(1, cell.get_vertices())
        Dfacet = basis[(1,)]
        Dfacet[:, 0] = -Dfacet[:, 0]

        rd = (0, -1)
        kd = slice(1, -1)
        Vfdm = np.eye(Ahat.shape[0])
        _, Vfdm[kd, kd] = eigh(Ahat[kd, kd], Bhat[kd, kd])
        Vfdm[kd, rd] = -Vfdm[kd, kd] @ ((Vfdm[kd, kd].T @ Bhat[kd, rd]) @ Vfdm[np.ix_(rd, rd)])

        def apply_weak_bcs(Ahat, Bhat, Dfacet, bcs, eta):
            Abc = Ahat.copy()
            Bbc = Bhat.copy()
            for j in (0, -1):
                if bcs[j]:
                    mult = 1.0E0 / bcs[j]
                    Abc[:, j] -= mult * Dfacet[:, j]
                    Abc[j, :] -= mult * Dfacet[:, j]
                    Abc[j, j] += eta

            return [csr_matrix(Abc), csr_matrix(Bbc)]

        A = Vfdm.T @ Ahat @ Vfdm
        a = A.diagonal().copy()
        A[kd, kd] = 0.0E0
        np.fill_diagonal(A, a)

        B = Vfdm.T @ Bhat @ Vfdm
        b = B.diagonal().copy()
        B[kd, kd] = 0.0E0
        B[rd, kd] = 0.0E0
        B[kd, rd] = 0.0E0
        np.fill_diagonal(B, b)

        Dfdm = Vfdm.T @ Dfacet
        Afdm = []
        for bc1 in range(3):
            for bc0 in range(3):
                bcs = (bc0, bc1)
                Afdm.append(apply_weak_bcs(A, B, Dfdm, bcs, eta))

        # Vbc rotates GL residuals into modes obtained from GLL
        Vfdm = Jipdg.T @ Vfdm

        return Afdm, Vfdm, Dfdm

    @staticmethod
    @lru_cache(maxsize=10)
    def assemble_matfree(ndim, nscal, N, Nq, needs_interior_facet, helm=False):
        # Assemble sparse 1D matrices and matrix-free kernels for basis transformation and stencil computation
        from firedrake.slate.slac.compiler import BLASLAPACK_LIB, BLASLAPACK_INCLUDE

        nsym = (ndim * (ndim+1)) // 2

        Ahat, Bhat, Jhat, Dhat, _ = FDMPC.semhat(N, Nq)
        nx = Ahat.shape[0]
        ny = nx if ndim >= 2 else 1
        nz = nx if ndim >= 3 else 1
        nxyz = nx*ny*nz
        ntot = nscal*nxyz
        lwork = ntot
        nv = nx * nx

        nxq = Jhat.shape[1]
        nyq = nxq if ndim >= 2 else 1
        nzq = nxq if ndim >= 3 else 1
        nquad = nxq * nyq * nzq

        if needs_interior_facet:
            Afdm, Vbc, Dbc = FDMPC.fdm_ipdg(Ahat, Bhat, N)
        else:
            Afdm, Vbc, Dbc = FDMPC.fdm_cg(Ahat, Bhat)

        Vsize = nv
        Vhex = ', '.join(map(float.hex, np.asarray(Vbc).flatten()))
        VX = "V"
        VY = "V" if ndim > 1 else "&one"
        VZ = "V" if ndim > 2 else "&one"

        kronmxv_code = """
        #include <petscsys.h>
        #include <petscblaslapack.h>

        static void kronmxv(PetscBLASInt tflag,
            PetscBLASInt mx, PetscBLASInt my, PetscBLASInt mz,
            PetscBLASInt nx, PetscBLASInt ny, PetscBLASInt nz, PetscBLASInt nel,
            PetscScalar  *A1, PetscScalar *A2, PetscScalar *A3,
            PetscScalar  *x , PetscScalar *y){

        PetscBLASInt m,n,k,s,p,lda;
        char TA1, TA2, TA3;
        char tran='T', notr='N';
        PetscScalar zero=0.0E0, one=1.0E0;

        if(tflag>0){
           TA1 = tran;
           TA2 = notr;
        }else{
           TA1 = notr;
           TA2 = tran;
        }
        TA3 = TA2;

        m = mx;  k = nx;  n = ny*nz*nel;
        lda = (tflag>0)? nx : mx;

        BLASgemm_(&TA1, &notr, &m,&n,&k, &one, A1,&lda, x,&k, &zero, y,&m);

        p = 0;  s = 0;
        m = mx;  k = ny;  n = my;
        lda = (tflag>0)? ny : my;
        for(PetscBLASInt i=0; i<nz*nel; i++){
           BLASgemm_(&notr, &TA2, &m,&n,&k, &one, y+p,&m, A2,&lda, &zero, x+s,&m);
           p += m*k;
           s += m*n;
        }

        p = 0;  s = 0;
        m = mx*my;  k = nz;  n = mz;
        lda = (tflag>0)? nz : mz;
        for(PetscBLASInt i=0; i<nel; i++){
           BLASgemm_(&notr, &TA3, &m,&n,&k, &one, x+p,&m, A3,&lda, &zero, y+s,&m);
           p += m*k;
           s += m*n;
        }
        return;
        }
        """

        transfer_code = f"""
        {kronmxv_code}

        void prolongation(PetscScalar *y,
                      PetscScalar *x){{
            PetscScalar V[{Vsize}] = {{ {Vhex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                    t0[j + {nxyz}*i] = x[i + {nscal}*j];

            kronmxv(1, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {VX},{VY},{VZ}, t0, t1);

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                   y[i + {nscal}*j] = t1[j + {nxyz}*i];
            return;
        }}

        void restriction(PetscScalar *y,
                      PetscScalar *x,
                      PetscScalar *w){{
            PetscScalar V[{Vsize}] = {{ {Vhex} }};
            PetscScalar t0[{lwork}], t1[{lwork}];
            PetscScalar one = 1.0E0;

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                    t0[j + {nxyz}*i] = x[i + {nscal}*j] * w[i + {nscal}*j];

            kronmxv(0, {nx},{ny},{nz}, {nx},{ny},{nz}, {nscal}, {VX},{VY},{VZ}, t0, t1);

            for({IntType_c} j=0; j<{nxyz}; j++)
                for({IntType_c} i=0; i<{nscal}; i++)
                   y[i + {nscal}*j] += t1[j + {nxyz}*i];
            return;
        }}
        """

        restrict_kernel = op2.Kernel(transfer_code, "restriction", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())
        prolong_kernel = op2.Kernel(transfer_code, "prolongation", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())

        nb = Jhat.size
        VJ = Vbc.T @ Jhat
        VD = Vbc.T @ Dhat
        Jhex = ', '.join(map(float.hex, np.asarray(VJ).flatten()))
        Dhex = ', '.join(map(float.hex, np.asarray(VD).flatten()))

        JX = "J"
        JY = "J" if ndim > 1 else "&one"
        JZ = "J" if ndim > 2 else "&one"

        DX = "D"
        DY = "D" if ndim > 1 else "&one"
        DZ = "D" if ndim > 2 else "&one"

        # FIXME I don't know how to use optional arguments here
        bcoef = "bcoef" if helm else "NULL"
        cargs = "PetscScalar *diag"
        cargs += ", PetscScalar *gcoef"
        if helm:
            cargs += ", PetscScalar *bcoef"

        stencil_code = f"""
        {kronmxv_code}

        void mult3(PetscBLASInt n, PetscScalar *A, PetscScalar *B, PetscScalar *C){{
            for({IntType_c} i=0; i<n; i++)
                C[i] = A[i] * B[i];
            return;
        }}

        void mult_diag(PetscBLASInt m, PetscBLASInt n,
                       PetscScalar *A, PetscScalar *B, PetscScalar *C){{
            for({IntType_c} j=0; j<n; j++)
                for({IntType_c} i=0; i<m; i++)
                    C[i+m*j] = A[i] * B[i+m*j];
            return;
        }}

        void get_basis(PetscBLASInt dom, PetscScalar *J, PetscScalar *D, PetscScalar *B){{
            PetscScalar *basis[2] = {{J, D}};
            if(dom)
                for({IntType_c} j=0; j<2; j++)
                    for({IntType_c} i=0; i<2; i++)
                        mult_diag({nxq}, {nx}, basis[i]+{nxq*(nx-1)}*(dom-1), basis[j], B+(i+2*j)*{nb});
            else
                for({IntType_c} j=0; j<2; j++)
                    for({IntType_c} i=0; i<2; i++)
                        mult3({nb}, basis[i], basis[j], B+(i+2*j)*{nb});
            return;
        }}

        void get_band(PetscBLASInt dom1, PetscBLASInt dom2, PetscBLASInt dom3,
                      PetscScalar *JX, PetscScalar *DX,
                      PetscScalar *JY, PetscScalar *DY,
                      PetscScalar *JZ, PetscScalar *DZ,
                      PetscScalar *gcoef,
                      PetscScalar *bcoef,
                      PetscScalar *band){{

            PetscScalar BX[{4 * nb}];
            PetscScalar BY[{4 * nb}];
            PetscScalar BZ[{4 * nb}];
            PetscScalar t0[{nquad}], t1[{nquad}], t2[{nxyz}] = {{0.0E0}};
            PetscScalar scal;
            {IntType_c} k, ix, iy, iz;
            {IntType_c} ndiag = {nxyz}, nquad = {nquad}, nstencil = {2*ndim+1}, inc = 1;

            get_basis(dom1, JX, DX, BX);
            get_basis(dom2, JY, DY, BY);
            if({ndim}==3)
                get_basis(dom3, JZ, DZ, BZ);
            else
                BZ[0] = 1.0E0;

            if(bcoef){{
                BLAScopy_(&nquad, bcoef, &inc, t0, &inc);
                kronmxv(1, {nx}, {ny}, {nz}, {nxq}, {nyq}, {nzq}, 1, BX, BY, BZ, t0, t2);
            }}

            for({IntType_c} j=0; j<{ndim}; j++)
                for({IntType_c} i=0; i<{ndim}; i++){{
                    k = i + j + (i>0 && j>0 && {ndim}==3);
                    ix = (i == {ndim-1}) + 2 * (j == {ndim-1});
                    iy = (i == {ndim-2}) + 2 * (j == {ndim-2});
                    iz = (i == {ndim-3}) + 2 * (j == {ndim-3});
                    scal = (i == j) ? 1.0E0 : 0.5E0;
                    BLAScopy_(&nquad, gcoef+k*nquad, &inc, t0, &inc);
                    kronmxv(1, {nx}, {ny}, {nz}, {nxq}, {nyq}, {nzq}, 1, BX+ix*{nb}, BY+iy*{nb}, BZ+iz*{nb}, t0, t1);
                    BLASaxpy_(&ndiag, &scal, t1, &inc, t2, &inc);
                }}

            BLAScopy_(&ndiag, t2, &inc, band, &nstencil);
            return;
        }}

        void stencil({cargs}){{
            PetscScalar J[{nb}] = {{ {Jhex} }};
            PetscScalar D[{nb}] = {{ {Dhex} }};
            PetscScalar tcoef[{nquad * nsym}];
            PetscScalar one = 1.0E0;
            {IntType_c} i1, i2, i3;

            for({IntType_c} j=0; j<{nquad}; j++)
                for({IntType_c} i=0; i<{nsym}; i++)
                    tcoef[j + {nquad}*i] = gcoef[i + {nsym}*j];

            get_band(0, 0, 0, {JX}, {DX}, {JY}, {DY}, {JZ}, {DZ}, tcoef, {bcoef}, diag);

            for({IntType_c} j=0; j<{2*ndim}; j++){{
                i1 = (j/2 == {ndim-1}) * (1 + (j%2));
                i2 = (j/2 == {ndim-2}) * (1 + (j%2));
                i3 = (j/2 == {ndim-3}) * (1 + (j%2));
                get_band(i1, i2, i3, {JX}, {DX}, {JY}, {DY}, {JZ}, {DZ}, tcoef, {bcoef}, diag + (j+1));
            }}
            return;
        }}
        """

        stencil_kernel = op2.Kernel(stencil_code, "stencil", include_dirs=BLASLAPACK_INCLUDE.split(), ldargs=BLASLAPACK_LIB.split())
        return restrict_kernel, prolong_kernel, stencil_kernel, Afdm, Dbc

    @staticmethod
    def multiplicity(V):
        # Lawrence's magic code for calculating dof multiplicities
        shapes = (V.finat_element.space_dimension(),
                  np.prod(V.shape))
        domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
        instructions = """
        for i, j
            w[i,j] = w[i,j] + 1
        end
        """
        weight = firedrake.Function(V)
        firedrake.par_loop((domain, instructions), firedrake.dx,
                           {"w": (weight, op2.INC)}, is_loopy_kernel=True)
        return weight

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum_fun(V, interior_facet=False):
        node_map = V.interior_facet_node_map() if interior_facet else V.cell_node_map()
        nelh = node_map.values.shape[0]
        if node_map.offset is None:
            return lambda e: node_map.values[e], nelh
        else:
            layers = node_map.iterset.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1] - layers[0, 0] - 1
                nel = nelz * nelh
                return lambda e: node_map.values[e % nelh] + (e//nelh)*node_map.offset, nel
            else:
                k = 0
                nelz = layers[:nelh, 1] - layers[:nelh, 0] - 1
                nel = sum(nelz)
                layer_id = np.zeros((nel, 2))
                for e in range(0, nelh):
                    for l in range(0, nelz[e]):
                        layer_id[k, :] = [e, l]
                        k += 1
                return lambda e: node_map.values[layer_id[e, 0]] + layer_id[e, 1]*node_map.offset, nel

    @staticmethod
    @lru_cache(maxsize=10)
    def glonum(V):
        node_map = V.cell_node_map()
        if node_map.offset is None:
            return node_map.values
        else:
            nelh = node_map.values.shape[0]
            layers = node_map.iterset.layers_array
            if(layers.shape[0] == 1):
                nelz = layers[0, 1]-layers[0, 0]-1
                nel = nelz * nelh
                gl = np.zeros((nelz,)+node_map.values.shape, dtype=PETSc.IntType)
                for k in range(0, nelz):
                    gl[k] = node_map.values + k*node_map.offset
                gl = np.reshape(gl, (nel, -1))
            else:
                k = 0
                nelz = layers[:nelh, 1]-layers[:nelh, 0]-1
                nel = sum(nelz)
                gl = np.zeros((nel, node_map.values.shape[1]), dtype=PETSc.IntType)
                for e in range(0, nelh):
                    for l in range(0, nelz[e]):
                        gl[k] = node_map.values[e] + l*node_map.offset
                        k += 1
            return gl

    @staticmethod
    @lru_cache(maxsize=10)
    def get_bc_flags(V, mesh, bcs, J):
        extruded = mesh.cell_set._extruded
        ndim = mesh.topological_dimension()
        nface = 2*ndim

        # Partition of unity at interior facets (fraction of volumes)
        DG0 = firedrake.FunctionSpace(mesh, 'DG', 0)
        if ndim == 1:
            DGT = firedrake.FunctionSpace(mesh, 'Lagrange', 1)
        else:
            DGT = firedrake.FunctionSpace(mesh, 'DGT', 0)
        cell2cell = FDMPC.glonum(DG0)
        face2cell = FDMPC.glonum(DGT)

        area = firedrake.FacetArea(mesh)
        vol = firedrake.CellVolume(mesh)
        tau = firedrake.interpolate(vol, DG0)
        v = firedrake.TestFunction(DGT)

        dFacet = firedrake.dS_h + firedrake.dS_v if extruded else firedrake.dS
        w = firedrake.assemble(((v('-') * tau('-') + v('+') * tau('+')) / area) * dFacet)

        rho = w.dat.data_ro_with_halos[face2cell] / tau.dat.data_ro[cell2cell]

        if extruded:
            ibot = 4
            itop = 5
            ivert = [0, 1, 2, 3]
            nelh = mesh.cell_set.sizes[1]
            layers = mesh.cell_set.layers_array
            if layers.shape[0] == 1:
                nelz = layers[0, 1] - layers[0, 0] - 1
                nel = nelh * nelz
                facetdata = np.zeros([nel, nface, 2], dtype=PETSc.IntType)
                facetdata[:, ivert, :] = np.tile(mesh.cell_to_facets.data, (nelz, 1, 1))
            else:
                nelz = layers[:nelh, 1] - layers[:nelh, 0] - 1
                nel = sum(nelz)
                facetdata = np.zeros([nel, nface, 2], dtype=PETSc.IntType)
                facetdata[:, ivert, :] = np.repeat(mesh.cell_to_facets.data, nelz, axis=0)
                for f in ivert:
                    bnd = np.isclose(rho[:, f], 0.0E0)
                    bnd &= (facetdata[:, f, 0] != 0)
                    facetdata[bnd, f, :] = [0, -8]

            bot = np.isclose(rho[:, ibot], 0.0E0)
            top = np.isclose(rho[:, itop], 0.0E0)
            facetdata[:, [ibot, itop], :] = -1
            facetdata[bot, ibot, :] = [0, -2]
            facetdata[top, itop, :] = [0, -4]
        else:
            facetdata = mesh.cell_to_facets.data

        flags = facetdata[:, :, 0]
        sub = facetdata[:, :, 1]

        # Boundary condition flags
        # 0 => Natural, do nothing
        # 1 => Strong Dirichlet
        # 2 => Interior facet
        maskall = []
        comp = dict()
        for bc in bcs:
            if isinstance(bc, firedrake.DirichletBC):
                labels = comp.get(bc._indices, ())
                bs = bc.sub_domain
                if bs == "on_boundary":
                    maskall.append(bc._indices)
                elif bs == "bottom":
                    labels += (-2,)
                elif bs == "top":
                    labels += (-4,)
                else:
                    labels += bs if type(bs) == tuple else (bs,)
                comp[bc._indices] = labels

        # TODO add support for weak component BCs
        # The Neumann integral may still be present but it's zero
        J = expand_derivatives(J)
        for it in J.integrals():
            itype = it.integral_type()
            if itype.startswith("exterior_facet"):
                labels = comp.get((), ())
                bs = it.subdomain_id()
                if bs == "everywhere":
                    if itype == "exterior_facet_bottom":
                        labels += (-2,)
                    elif itype == "exterior_facet_top":
                        labels += (-4,)
                    else:
                        maskall.append(())
                else:
                    labels += bs if type(bs) == tuple else (bs,)
                comp[()] = labels

        labels = comp.get((), ())
        labels = list(set(labels))
        fbc = np.isin(sub, labels).astype(PETSc.IntType)

        if () in maskall:
            fbc[sub >= -1] = 1
        fbc[flags != 0] = 2

        others = set(comp.keys()) - {()}
        if others:
            # We have bcs on individual vector components
            fbc = np.tile(fbc, (V.value_size, 1, 1))
            for j in range(V.value_size):
                key = (j,)
                labels = comp.get(key, ())
                labels = list(set(labels))
                fbc[j] |= np.isin(sub, labels)
                if key in maskall:
                    fbc[j][sub >= -1] = 1

            fbc = np.transpose(fbc, (1, 0, 2))
        return fbc
