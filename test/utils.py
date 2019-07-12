import torch
import math

class DummyNetwork:
    def __init__(self, list_of_weights=None, list_of_grads=None):
        if not list_of_weights:
            list_of_weights = []
        if not list_of_grads:
            list_of_grads = []
        self.gradients = [torch.Tensor(grad) for grad in list_of_grads]
        self.data = [torch.Tensor(weight) for weight in list_of_weights]

    def parameters(self):
        for grad, data in zip(self.gradients, self.data):
            param = torch.nn.Parameter(data, requires_grad=True)
            # TODO find more elegant way to do this
            param.grad = grad
            param.grad.data = grad
            yield param

    def get_data_as_list(self):
        return [list(datapoint.numpy()) for datapoint in self.data]


def is_float_list_equal(l1, l2, delta):
    l1 = _flatten(l1)
    l2 = _flatten(l2)

    for el1, el2 in zip(l1, l2):
        if not math.isclose(el1, el2, rel_tol=delta):
            return False
    return True


def _flatten(l):
    for elem in l:
        if not isinstance(elem, list):
            yield elem
        else:
            yield from _flatten(elem)
