import torch
import unittest
from bayesian_pasa.activations import PASA, BayesianPASA, Mish, Swish


class TestActivations(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.channels = 3
        self.height = 32
        self.width = 32
        self.x = torch.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_pasa_shape(self):
        pasa = PASA()
        output = pasa(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_pasa_weights(self):
        pasa = PASA()
        output, weights = pasa(self.x, return_weights=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(weights.shape[-1], 3)  # 3 branches
    
    def test_bayesian_pasa_shape(self):
        bpasa = BayesianPASA()
        output = bpasa(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_bayesian_pasa_weights(self):
        bpasa = BayesianPASA()
        output, weights = bpasa(self.x, return_weights=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(weights.shape[-1], 3)
    
    def test_mish_shape(self):
        mish = Mish()
        output = mish(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_swish_shape(self):
        swish = Swish()
        output = swish(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_pasa_forward(self):
        pasa = PASA()
        output = pasa(self.x)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


if __name__ == '__main__':
    unittest.main()
