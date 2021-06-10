from functools import partial
import numpy as np

from ufl.referencevalue import ReferenceValue
from ufl.log import error

from firedrake.external_operators import AbstractExternalOperator
from firedrake.function import Function
from firedrake.constant import Constant
from firedrake import utils

from pyop2.datatypes import ScalarType

from firedrake.external_operators.pytorch_model_parameters import PytorchModelParameters

import torch
from torch import nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector 


class PytorchOperator(AbstractExternalOperator):
    r"""A :class:`PointnetOperator`: is an implementation of ExternalOperator that is defined through
    a given neural network model N and whose values correspond to the output of the neural network represented by N.
     """

    def __init__(self, *operands, function_space, derivatives=None, val=None,
                 name=None, coefficient=None, arguments=(), dtype=ScalarType,operator_data=None, 
                 local_operands=()):
        r"""
        :param operands: operands on which act the :class:`PointnetOperator`.
        :param function_space: the :class:`.FunctionSpace`,
        :model: a pytorch model,
        """

        self.model = operator_data["model"]
        self.model.double()
        self.model_parameters = PytorchModelParameters(self.model)

        AbstractExternalOperator.__init__(self, *operands, function_space=function_space,
         derivatives=derivatives, val=val, name=name, coefficient=coefficient, arguments=arguments, dtype=dtype,
         operator_data=operator_data, local_operands=operands)

        
    @property
    def framework(self):
        # PyTorch by default
        return 'PyTorch'

    # @property
    def operator_inputs(self):
        return self.ufl_operands

    def _evaluate_jacobian(self, N, x, **kwargs):

        jac, = torch.autograd.grad(outputs=N, inputs=x,
                                  grad_outputs=torch.ones_like(N),
                                  retain_graph=True)

        return jac
   
    def _evaluate(self, model_tape=False):
        """
        Evaluate the neural network by performing a forward pass through the network
        """
        model = self.model

        # Explictly set the eval mode does matter for
        # networks having different behaviours for training/evaluating (e.g. Dropout)
        model.eval()

        # Process the inputs
        space = self.ufl_function_space()
        ops = tuple(Function(space).interpolate(op) for op in self.operator_inputs())

        torch_op = torch.tensor([op.dat.data_ro for op in ops], requires_grad=True, dtype=torch.float64).T
        # Vectorized forward pass
        val = model(torch_op)

        # Compute the jacobian
        if self.derivatives != (0,)*len(self.ufl_operands):
            val = self._evaluate_jacobian(val, torch_op)

        # We return a list instead of assigning to keep track of the PyTorch tape contained in the torch variables
        if model_tape:
            return val

        res = val.squeeze(-1).detach().numpy()
        result = Function(space)
        result.dat.data[:] = res

        # Explictly set the train mode does matter for
        # networks having different behaviours for training/evaluating (e.g. Dropout)
        model.train()
        return self.assign(result)

    def evaluate_backprop(self, x):
        outputs = self.evaluate(model_tape=True)
        params = self.model.parameters()
        grad_outputs = torch.tensor(x.dat.data_ro).unsqueeze(-1)
        grad_W = torch.autograd.grad(outputs, params,
                                     grad_outputs=grad_outputs,
                                     retain_graph=True)
        grad_W = parameters_to_vector(grad_W)
        return [grad_W]


def neuralnet(model, function_space, inputs_format=0):

    torch_module = type(None)
    tensorflow_module = type(None)

    # Checks
    try:
        import torch
        torch_module = torch.nn.modules.module.Module
    except ImportError:
        pass
    if inputs_format not in (0, 1):
        raise ValueError('Expecting inputs_format to be 0 or 1')

    if isinstance(model, torch_module):
        operator_data = {'framework': 'PyTorch', 'model': model, 'inputs_format': inputs_format}
        return partial(PytorchOperator, function_space=function_space, operator_data=operator_data)
    elif isinstance(model, tensorflow_module):
        operator_data = {'framework': 'TensorFlow', 'model': model, 'inputs_format': inputs_format}
        return partial(TensorFlowOperator, function_space=function_space, operator_data=operator_data)
    else:
        error("Expecting one of the following library : PyTorch, TensorFlow (or Keras) and that the library has been installed")
