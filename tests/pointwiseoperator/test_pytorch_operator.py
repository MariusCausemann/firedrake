from firedrake import *
from firedrake_adjoint import *
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector 
import copy

torch.manual_seed(20)

class MLP(nn.Module):

  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(1,10),
      #nn.Sigmoid(),
      nn.Linear(10,1),
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


mesh = UnitSquareMesh(4,4)
V1 = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
w = TestFunction(V1)
f = Function(V1, name="f").interpolate(cos(x)*sin(y))
u = Function(V1, name="u")
mlp = MLP()
mlp.double()

net = PytorchOperator(u, function_space=V1, operator_data={"model":mlp})

F = inner(grad(w), 10*net*grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
solve(F == 0, u)

loss = assemble(u**2*dx)
print(loss)
weights = net.model_parameters
J_hat = ReducedFunctional(loss, Control(weights))
d = J_hat.derivative()

mlp_copy = copy.deepcopy(mlp)
h = torch.ones_like(parameters_to_vector(mlp.parameters()))* 1e-3
vector_to_parameters(h, mlp_copy.parameters())
h = PytorchModelParameters(mlp_copy)
#get_working_tape().visualise(output="tape.dot")
conv_rate = taylor_test(J_hat, weights._ad_copy(), h)
assert 1.9 < conv_rate
