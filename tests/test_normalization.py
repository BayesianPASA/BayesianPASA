import torch
import unittest
from bayesian_pasa.normalization import StandardLayerNorm, RLayerNorm, BayesianRLayerNorm


class TestNormalization(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 4
        self.channels = 16
        self.height = 32
        self.width = 32
        self.x = torch.randn(self.batch_size, self.channels, self.height, self.width)
    
    def test_standard_layernorm_shape(self):
        ln = StandardLayerNorm(self.channels)
        output = ln(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_r_layernorm_shape(self):
        rln = RLayerNorm(self.channels)
        output = rln(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_bayesian_r_layernorm_shape(self):
        brln = BayesianRLayerNorm(self.channels)
        output = brln(self.x)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_bayesian_r_layernorm_uncertainty(self):
        brln = BayesianRLayerNorm(self.channels)
        output, uncertainty = brln(self.x, return_uncertainty=True)
        self.assertEqual(output.shape, self.x.shape)
        self.assertEqual(uncertainty.shape, self.x.shape)
    
    def test_psi_function(self):
        brln = BayesianRLayerNorm(self.channels)
        t = torch.tensor([0.1, 1.0, 10.0])
        psi = brln.psi(t)
        self.assertEqual(psi.shape, t.shape)
        self.assertFalse(torch.isnan(psi).any())


if __name__ == '__main__':
    unittest.main()
