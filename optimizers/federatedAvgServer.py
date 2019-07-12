import torch
from optimizer import Optimizer, required


class FederatedAvgServer(Optimizer):
    r"""Implements the server side of the federated averaging algorithm presented in the paper
    'Communication-Efficient Learning of Deep Networks from Decentralized Data' (https://arxiv.org/pdf/1602.05629.pdf).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups

    Example:
        >>> optimizers = torch.optim.FederatedAvgServer(model.parameters())
        >>> optimizers.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>>
        >>> # On every client then do
        >>> optimizer_client = torch.optim.FederatedAvgClient(model.parameters(), lr=0.1)
        >>> optimizer_client.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer_client.step()
        >>> nk_grad = (n_training_examples, model.parameters())
        >>>
        >>> # Send nk_grad from clients (1 to l) to the server
        >>> list_nk_grad = [nk_grad1, ..., nk_gradl]
        >>> optimizers.step(list_nk_grad)
        >>> # Redistribute updated model.parameters() from server to clients
    """

    def __init__(self, params):
        super(FederatedAvgServer, self).__init__(params, dict())

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
