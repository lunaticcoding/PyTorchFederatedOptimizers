import unittest
from test.utils import DummyNetwork, is_float_list_equal

from optimizers.federatedAvgClient import FederatedAvgClient as Client
from optimizers.federatedAvgServer import FederatedAvgServer as Server


class TestFederatedAvg(unittest.TestCase):
    def test_passes(self):
        self.assertEqual(1+1, 2)

    def test_client_runs_correct_calculation(self):
        network = DummyNetwork(list_of_weights=[[2.3, 3.4, 0.32], [4.54, 8.21, -6.54], [32.5, 0.3, 1.342]],
                              list_of_grads=[[0.1, 0.23, 0.32], [0.45, 0.93, 0.01], [0.31, 0.87, 0.74]])
        optim_client = Client(params=network.parameters(), lr=0.1)
        optim_client.step()

        correct = [[2.29, 3.377, 0.288], [4.495, 8.117, -6.5410], [32.469, 0.213, 1.268]]
        actual = network.get_data_as_list()
        self.assertTrue(is_float_list_equal(actual, correct, 0.000001))

    def test_server_runs_correct_calculation(self):
        network_server = DummyNetwork(list_of_weights=[[2.3, 3.4, 0.32], [4.54, 8.21, -6.54], [32.5, 0.3, 1.342]],
                                      list_of_grads=[[0.1, 0.23, 0.32], [0.45, 0.93, 0.01], [0.31, 0.87, 0.74]])
        networks_clients = [
            (12, DummyNetwork(list_of_weights=[[2.3, 32.4, 0.32], [4.54, 8.21, -6.54], [32.5, 0.3, 1.342]],
                              list_of_grads=[[0.1, 0.23, 0.32], [0.45, 0.93, 0.01], [0.31, 0.87, 0.74]]).parameters()),
            (3, DummyNetwork(list_of_weights=[[12.3, 3.4, 0.32], [4.54, 8.21, -6.54], [32.5, 0.3, 1.342]],
                              list_of_grads=[[0.1, 0.23, 0.32], [0.45, 0.93, 0.01], [0.31, 0.87, 0.74]]).parameters()),
            (67, DummyNetwork(list_of_weights=[[2.3, 3.4, 0.32], [4.54, 8.21, -6.54], [32.5, 0.3, 1543.342]],
                              list_of_grads=[[0.1, 0.23, 0.32], [0.45, 0.93, 0.01], [0.31, 0.87, 888.74]]).parameters())
        ]

        optim_server = Server(network_server.parameters())
        optim_server.step(networks_clients)

        correct = [[2.6658536585365, 7.6439024390243, 0.32], [4.54, 8.21, -6.54], [32.5, 0.3, 1261.2688292682]]
        actual = network_server.get_data_as_list()
        self.assertTrue(is_float_list_equal(actual, correct, 0.000001))


if __name__ == '__main__':
    unittest.main()


