from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.reduced_functional_numpy import gather

import numpy

import torch
from torch.nn.utils import vector_to_parameters, parameters_to_vector 

import copy 

#@register_overloaded_type
class PytorchModelParameters(OverloadedType):
    def __init__(self, model, *args, **kwargs):

        self.model = model
        self.n_params = sum([p.numel() for p in model.parameters()])

        OverloadedType.__init__(self, *args, **kwargs)

    def _ad_convert_type(self, value, options={}):
        model_copy = copy.deepcopy(self.model)
        vector_to_parameters(value, model_copy.parameters())
        return PytorchModelParameters(model_copy)

    def _ad_create_checkpoint(self):
        return self #copy.deepcopy(self.model)

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint #PytorchModelParameters(checkpoint)

    def _ad_mul(self, other):
        model_copy = copy.deepcopy(self.model)
        self_params = parameters_to_vector(model_copy.parameters())
        if isinstance(other, PytorchModelParameters):  
            other = parameters_to_vector(other.model.parameters())
        vector_to_parameters(self_params * other, model_copy.parameters())
        return PytorchModelParameters(model_copy)

    def _assign_val(self, values):
        self_params = parameters_to_vector(self.model.parameters())
        self_params[:] = values
        vector_to_parameters(self_params, self.model.parameters())
        return self

    def _ad_add(self, other):
        model_copy = copy.deepcopy(self.model)
        self_params = parameters_to_vector(model_copy.parameters())
        other_params = parameters_to_vector(other.model.parameters())
        vector_to_parameters(self_params + other_params, model_copy.parameters())
        return PytorchModelParameters(model_copy)

    def _ad_dot(self, other, options=None):
        self_params = parameters_to_vector(self.model.parameters())
        other_params = parameters_to_vector(other.model.parameters())
        return (self_params * other_params).sum().detach().numpy()

    def _ad_copy(self):
        return PytorchModelParameters(copy.deepcopy(self.model))

