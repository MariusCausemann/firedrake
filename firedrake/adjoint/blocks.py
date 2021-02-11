from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks
from pyadjoint.block import Block
from ufl.algorithms.analysis import extract_arguments_and_coefficients, extract_coefficients
from ufl import replace

import firedrake
import firedrake.utils as utils

import ufl


class Backend:
    @utils.cached_property
    def backend(self):
        import firedrake
        return firedrake

    @utils.cached_property
    def compat(self):
        import firedrake
        return compat(firedrake)


class DirichletBCBlock(blocks.DirichletBCBlock, Backend):
    pass


class ConstantAssignBlock(blocks.ConstantAssignBlock, Backend):
    pass


class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
    pass


class AssembleBlock(blocks.AssembleBlock, Backend):
    pass


def solve_init_params(self, args, kwargs, varform):
    if len(self.forward_args) <= 0:
        self.forward_args = args
    if len(self.forward_kwargs) <= 0:
        self.forward_kwargs = kwargs.copy()

    if len(self.adj_args) <= 0:
        self.adj_args = self.forward_args

    if len(self.adj_kwargs) <= 0:
        self.adj_kwargs = self.forward_kwargs.copy()

        if varform:
            if "J" in self.forward_kwargs:
                self.adj_kwargs["J"] = self.backend.adjoint(self.forward_kwargs["J"])
            if "Jp" in self.forward_kwargs:
                self.adj_kwargs["Jp"] = self.backend.adjoint(self.forward_kwargs["Jp"])

            if "M" in self.forward_kwargs:
                raise NotImplementedError("Annotation of adaptive solves not implemented.")
            self.adj_kwargs.pop("appctx", None)

    if "solver_parameters" in kwargs and "mat_type" in kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = kwargs["solver_parameters"]["mat_type"]

    if varform:
        if "appctx" in kwargs:
            self.assemble_kwargs["appctx"] = kwargs["appctx"]


class GenericSolveBlock(blocks.GenericSolveBlock, Backend):
    pass


class SolveLinearSystemBlock(GenericSolveBlock):
    def __init__(self, A, u, b, *args, **kwargs):
        lhs = A.form
        func = u.function
        rhs = b.form
        bcs = A.bcs if hasattr(A, "bcs") else []
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

        # Set up parameters initialization
        self.ident_zeros_tol = A.ident_zeros_tol if hasattr(A, "ident_zeros_tol") else None
        self.assemble_system = A.assemble_system if hasattr(A, "assemble_system") else False

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=False)


class SolveVarFormBlock(GenericSolveBlock):
    def __init__(self, equation, func, bcs=[], *args, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)


class NonlinearVariationalSolveBlock(GenericSolveBlock):
    def __init__(self, equation, func, bcs, problem_J, solver_params, solver_kwargs, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs

        self.problem_J = problem_J
        self.solver_params = solver_params.copy()
        self.solver_kwargs = solver_kwargs

        super().__init__(lhs, rhs, func, bcs, **{**solver_kwargs, **kwargs})

        if self.problem_J is not None:
            for coeff in self.problem_J.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        J = self.problem_J
        if J is not None:
            J = self._replace_form(J, func)
        problem = self.backend.NonlinearVariationalProblem(lhs, func, bcs, J=J)
        solver = self.backend.NonlinearVariationalSolver(problem, **self.solver_kwargs)
        solver.parameters.update(self.solver_params)
        solver.solve()
        return func


class ProjectBlock(SolveVarFormBlock):
    def __init__(self, v, V, output, bcs=[], *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = V.mesh()
        dx = self.backend.dx(mesh)
        w = self.backend.TestFunction(V)
        Pv = self.backend.TrialFunction(V)
        a = self.backend.inner(w, Pv) * dx
        L = self.backend.inner(w, v) * dx

        super().__init__(a == L, output, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)


class MeshInputBlock(Block):
    """
    Block which links a MeshGeometry to its coordinates, which is a firedrake
    function.
    """
    def __init__(self, mesh):
        super().__init__()
        self.add_dependency(mesh)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        mesh = self.get_dependencies()[0].saved_output
        return mesh.coordinates


class FunctionSplitBlock(Block, Backend):
    def __init__(self, func, idx):
        super().__init__()
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        eval_adj = self.backend.Function(block_variable.output.function_space())
        eval_adj.sub(self.idx).assign(adj_inputs[0].function)
        return eval_adj.vector()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return self.backend.Function.sub(tlm_inputs[0], self.idx)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        eval_hessian = self.backend.Function(block_variable.output.function_space())
        eval_hessian.sub(self.idx).assign(hessian_inputs[0].function)
        return eval_hessian.vector()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.backend.Function.sub(inputs[0], self.idx)


class FunctionMergeBlock(Block, Backend):
    def __init__(self, func, idx):
        super().__init__()
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return adj_inputs[0]

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        output = self.get_outputs()[0]
        fs = output.output.function_space()
        f = self.backend.Function(fs)
        output.add_tlm_output(self.backend.Function.assign(f.sub(self.idx), tlm_input))

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute(self):
        dep = self.get_dependencies()[0].checkpoint
        output = self.get_outputs()[0].checkpoint
        self.backend.Function.assign(self.backend.Function.sub(output, self.idx), dep)


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh are changed.
    """
    def __init__(self, func, mesh):
        super().__init__()
        self.add_dependency(func)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        vector = self.get_dependencies()[0].saved_output
        mesh = vector.function_space().mesh()
        mesh.coordinates.assign(vector, annotate=False)
        return mesh._ad_create_checkpoint()


class PointwiseOperatorBlock(Block, Backend):
    def __init__(self, point_op, *args, **kwargs):
        super(PointwiseOperatorBlock, self).__init__()
        self.point_op = point_op
        for c in self.point_op.ufl_operands:
            coeff_c = ufl.algorithms.extract_coefficients(c)
            for ci in coeff_c:
                self.add_dependency(ci, no_duplicates=True)
        self.add_dependency(self.point_op, no_duplicates=True)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        return ufl.replace(self.point_op, self._replace_map())

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, N=None):
        if self.point_op.coefficient == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            return None

        q_rep = block_variable.saved_output
        if isinstance(N, self.backend.PointnetOperator) and hasattr(q_rep.dat, '_dat_version'):
            N._weights_version['version'] = q_rep.dat._dat_version

        i_ops = list(i for i, e in enumerate(N.ufl_operands) if q_rep in extract_coefficients(e))[0]
        dNdm_adj = N.evaluate_adj_component_control(adj_inputs[0], i_ops)
        return dNdm_adj

    def _replace_map(self):
        coeffs_point_op = tuple(ci for opi in self.point_op.ufl_operands for ci in extract_coefficients(opi)) + (self.point_op.coefficient,)
        replace_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in coeffs_point_op:
                replace_coeffs[coeff] = block_variable.saved_output
        return replace_coeffs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return ufl.replace(self.point_op, self._replace_map())

    def recompute_component(self, inputs, block_variable, idx, prepared):
        p = prepared
        q = type(p).copy(p)
        return q.evaluate()


class InterpolateBlock(Block, Backend):
    def __init__(self, interpolator, *functions, **kwargs):
        super().__init__()

        self.expr = interpolator.expr
        self.arguments, self.coefficients = extract_arguments_and_coefficients(self.expr)

        if isinstance(interpolator.V, firedrake.Function):
            self.V = interpolator.V.function_space()
        else:
            self.V = interpolator.V

        for coefficient in self.coefficients:
            self.add_dependency(coefficient, no_duplicates=True)

        for function in functions:
            self.add_dependency(function, no_duplicates=True)

    def _replace_map(self):
        # Replace the dependencies with checkpointed values
        replace_map = {}
        args = 0
        for block_variable in self.get_dependencies():
            output = block_variable.output
            if output in self.coefficients:
                replace_map[output] = block_variable.saved_output
            else:
                replace_map[self.arguments[args]] = block_variable.saved_output
                args += 1
        return replace_map

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        dJdm = self.backend.derivative(prepared, inputs[idx])
        return self.backend.Interpolator(dJdm, self.V).interpolate(adj_inputs[0], transpose=True)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        dJdm = 0.

        for i, input in enumerate(inputs):
            if tlm_inputs[i] is None:
                continue
            dJdm += self.backend.derivative(prepared, input, tlm_inputs[i])

        return self.backend.Interpolator(dJdm, self.V).interpolate()

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.backend.interpolate(prepared, self.V)
