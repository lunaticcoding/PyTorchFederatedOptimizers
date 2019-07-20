import unittest
import torch
from test.utils import DummyNetwork, is_float_list_equal

from optimizers.FSVRGClient import FSVRGClient as Client
from optimizers.FSVRGServer import FSVRGServer as Server


class TestFSVRG(unittest.TestCase):
    def test_passes(self):
        self.assertEqual(1+1, 2)

    def test_computes_scaling_matrix_S(self):
        data = torch.Tensor([[[12.321, -3.21, 5.1], [12, -3, 5], [12, 0, 0]],
                             [[6, 3, 6], [0, -3, 5], [12, 0.0, 0]],
                             [[12, -3, 5], [12, 0, 5], [0, 0, 5.523]]])
        client = Client(DummyNetwork().parameters())
        s = client.compute_scaling_matrix_s(data)
        s_correct = torch.Tensor([[3, 3, 3], [2, 2, 3], [2, 0, 1]])
        self.assertTrue(torch.Tensor.allclose(s, s_correct))


if __name__ == '__main__':
    unittest.main()


