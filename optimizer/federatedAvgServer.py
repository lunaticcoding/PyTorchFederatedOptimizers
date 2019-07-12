import torch
from optimizer import Optimizer, required


class FederatedAvgServer(Optimizer):
    r"""Implements federated averaging.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = torch.optim.FederatedAvgServer(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step(list_nk_grad)
    """

    def __init__(self, params, dampening=0,
                 weight_decay=0):
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(dampening=dampening,
                        weight_decay=weight_decay)
        super(FederatedAvgServer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FederatedAvgServer, self).__setstate__(state)

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
                p.data = torch.zeros_like(p.data)

        n = 0
        for group in self.param_groups:
            for n_k, w_k in list_nk_grad:
                for p, w_k_p in zip(group['params'], list(w_k)):
                    if w_k_p.grad is None:
                        continue
                    p.data += w_k_p.data * n_k
                n += n_k

        for p in group['params']:
            p.data /= n
        return loss
