import torch
from optimizer import Optimizer, required


class FSVRGClient(Optimizer):
    r"""Implements federated averaging.


    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    Example:
        >>> optimizer = optimizers.FSVRGClient(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizers.step()
    """

    def __init__(self, params, lr=1, dampening=0,
                 weight_decay=0):
        self.n_k = None

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, dampening=dampening,
                        weight_decay=weight_decay)
        super(FSVRGClient, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FSVRGClient, self).__setstate__(state)

    def compute_scaling_matrix_s(self, data):
        data_shape = data.shape
        self.n_k = data_shape[0]
        data.apply_(lambda x: 0.0 if x == 0.0 else 1.0)
        s_k = sum([t for t in data])
        return s_k


        # sum over first shape column

    def step(self, n_k=required, closure=None):
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
                local_lr = group['lr']/n_k

                if weight_decay != 0:
                    d_p.add_((weight_decay*(-group['lr'])), p.data)
                p.data.add_(-group['lr'], d_p)

        return loss
