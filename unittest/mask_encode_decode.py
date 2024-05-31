import unittest
import torch

from dp_gfn.utils.masking import encode, decode


class TestMasking(unittest.TestCase):

    def test_encode_decode(self):
        decoded = torch.tensor([[[True, False, True], [False, True, False], [True, False, True]]])
        encoded = encode(decoded)
        decoded_again = decode(encoded, 3)
        self.assertTrue(torch.all(decoded == decoded_again))


if __name__ == '__main__':
    unittest.main()
