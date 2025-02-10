import unittest
import torch
from chronoflux import Generator, Discriminator

class TestModels(unittest.TestCase):
    def test_generator(self):
        model = Generator()
        x = torch.randn(1, 3, 128, 128)
        age = torch.rand(1)
        output = model(x, age)
        self.assertEqual(output.shape, (1, 3, 128, 128))

    def test_discriminator(self):
        model = Discriminator()
        x = torch.randn(1, 3, 128, 128)
        age = torch.rand(1)
        validity, age_pred = model(x, age)
        self.assertEqual(validity.shape, (1, 1))
        self.assertEqual(age_pred.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()