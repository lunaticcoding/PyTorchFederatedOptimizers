import torch
import numpy as np
from optimizer import Optimizer, required


class FSVRGServer(Optimizer):
    r"""Implements the server side of the stochastic variance reduced gradient descent algorithm presented in the paper
    'Federated Optimization: Distributed Machine Learning for On-Device Intelligence' (https://www.maths.ed.ac.uk/~prichtar/papers/federated_optimization.pdf).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

    Example:
        >>> optimizer = optimizers.FSVRGServer(model.parameters())
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>>
        >>> # On every client then do
        >>> optimizer_client = optimizers.FSVRGClient(model.parameters(), lr=0.1)
        >>> optimizer_client.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> s = optimizer_client.compute_stochastic_gradient_scaling_matrix_s()
        >>>
        >>> # send a to clients (and then on Clients)
        >>> optimizer_client.step(a)
        >>> optimizer.compute_aggregated_scaling_matrix_a(list_of_s)

        >>> # Send nk_grad from clients (1 to l) to the server
        >>> list_nk_grad = [nk_grad1, ..., nk_gradl]
        >>> optimizers.step(list_nk_grad)
        >>> # Redistribute updated model.parameters() from server to clients
    """

    def __init__(self, params):
        self.A = None
        super(FSVRGServer, self).__init__(params, dict())

    def __setstate__(self, state):
        super(FSVRGServer, self).__setstate__(state)

    def get_phi_and_compute_A(self, phi_ks):
        phi = sum(phi_ks)
        #w_j = torch.Tensor(phi_ks).apply_(lambda x: 0.0 if x == 0.0 else 1.0)
        #K = torch.full_like(w_j, len(phi_ks))
        #a_j = K.div_(w_j)
        #self.A = np.diag(a_j)
        return phi

    def step(self, list_nk_grad, closure=None):
        """Performs a single optimization step.

        Arguments:
            list_nk_grad: A list of the number of datapoints and the calculated gradient for each device k.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # set old weights to 0
                p.data.add_(-p.data)

        n = 0
        for group in self.param_groups:
            for n_k, _ in list_nk_grad:
                n += n_k
            for n_k, w_k in list_nk_grad:
                for p, w_k_p in zip(group['params'], list(w_k)):
                    if w_k_p.grad is None:
                        continue
                    p.data.add_(w_k_p.data * n_k/n)
        print(self.param_groups)

        return loss
