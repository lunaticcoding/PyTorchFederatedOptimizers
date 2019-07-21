import unittest
import torch
import numpy as np
from test.utils import DummyNetwork, is_float_list_equal

from optimizers.FSVRGClient import FSVRGClient as Client
from optimizers.FSVRGServer import FSVRGServer as Server


class TestFSVRG(unittest.TestCase):
    def test_passes(self):
        self.assertEqual(1+1, 2)

    def test_computes_phi_k(self):
        data = torch.Tensor([[[12.321, -3.21, 5.1], [12, -3, 5], [12, 0, 0]],
                             [[6, 3, 6], [0, -3, 5], [12, 0.0, 0]],
                             [[6, 3, 6], [0, -3, 5], [12, 0.0, 0]],
                             [[12, -3, 5], [12, 0, 5], [0, 0, 5.523]]])
        client = Client(DummyNetwork().parameters())
        s = client.compute_nonzero_features_on_node_phi_k(data)
        s_correct = torch.Tensor([[1, 1, 1], [.5, .75, 1], [.75, 0, .25]])
        self.assertTrue(torch.Tensor.allclose(s, s_correct))

    def test_computes_phi(self):
        phi_ks = torch.Tensor([[1, 1, 1], [.5, .75, 1], [.75, 0, .25],
                               [.4, .1, 1], [.35, .5, .1], [1.0, 0, .15],
                               [.1, 0.2, 1], [.5, .75, 0], [.32, 0, .46]
                               ])
        server = Server(DummyNetwork().parameters())
        phi = server.get_phi_and_compute_A(phi_ks)
        phi_correct = torch.Tensor([[1.5, 1.3, 3], [1.35, 2, 2], [2.07, 0, .86]])
        self.assertTrue(torch.Tensor.allclose(phi, phi_correct))

    def test_computes_A(self):
        phi_ks = torch.Tensor([[1, 1, 1], [.5, .75, 1], [.75, 0, .25],
                            [.4, .1, 1], [.35, .5, .1], [1.0, 0, .15],
                            [.1, 0.2, 1], [.5, .75, 0], [.32, .3, .46]
                               ])
        server = Server(DummyNetwork().parameters())
        server.get_phi_and_compute_A(phi_ks)
        A = server.A
        A_correct = np.diag(torch.Tensor([[1, 1, 1], [1, 1, 1.5], [1, 3, 1]]))
        self.assertTrue(torch.Tensor.allclose(A, A_correct))

    def test_computes_A_with_inf(self):
        phi_ks = torch.Tensor([[1, 1, 1], [.5, .75, 1], [.75, 0, .25],
                            [.4, .1, 1], [.35, .5, .1], [1.0, 0, .15],
                            [.1, 0.2, 1], [.5, .75, 0], [.32, 0, .46]
                               ])
        server = Server(DummyNetwork().parameters())
        server.get_phi_and_compute_A(phi_ks)
        A = server.A
        A_correct = np.diag(torch.Tensor([[1, 1, 1], [1, 1, 1.5], [1, int('inf'), 1]]))
        self.assertTrue(torch.Tensor.allclose(A, A_correct))


if __name__ == '__main__':
    unittest.main()


