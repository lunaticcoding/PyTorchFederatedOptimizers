import torch
from optimizer import Optimizer, required


class FederatedAvgClient(Optimizer):
    r"""Implements federated averaging.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizers = torch.optim.FederatedAvgClient(model.parameters(), lr=0.1)
        >>> optimizers.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizers.step()
    """

    def __init__(self, params, lr=required, dampening=0,
                 weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, dampening=dampening,
                        weight_decay=weight_decay)
        super(FederatedAvgClient, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FederatedAvgClient, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_((weight_decay*(-group['lr'])), p.data)
                else:
                    p.data.add_(-group['lr'], d_p)

        return loss
