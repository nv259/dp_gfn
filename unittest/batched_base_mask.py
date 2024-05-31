import unittest
import torch

from dp_gfn.utils.masking import batched_base_mask

class TestBatchedBaseMask(unittest.TestCase):

    def test_empty_batch(self):
        num_words = torch.tensor([])
        num_variables = 5
        root_first = True
        device = torch.device('cpu')

        expected_mask = torch.zeros(0, num_variables, num_variables, dtype=torch.bool, device=device)

        actual_mask = batched_base_mask(num_words, num_variables, root_first, device)

        self.assertEqual(actual_mask.shape, expected_mask.shape)
        self.assertTrue(torch.all(actual_mask == expected_mask))

    def test_single_sentence(self):
        num_words = torch.tensor([3])
        num_variables = 5
        root_first = True
        device = torch.device('cpu')

        expected_mask = torch.zeros(1, num_variables, num_variables, dtype=torch.bool, device=device)
        expected_mask[0, 0, 1:4] = True

        actual_mask = batched_base_mask(num_words, num_variables, root_first, device)

        self.assertEqual(actual_mask.shape, expected_mask.shape)
        self.assertTrue(torch.all(actual_mask == expected_mask))

    def test_multiple_sentences(self):
        num_words = torch.tensor([2, 4])
        num_variables = 5
        root_first = True
        device = torch.device('cpu')

        expected_mask = torch.zeros(2, num_variables, num_variables, dtype=torch.bool, device=device)
        expected_mask[0, 0, 1:3] = True
        expected_mask[1, 0, 1:5] = True

        actual_mask = batched_base_mask(num_words, num_variables, root_first, device)

        self.assertEqual(actual_mask.shape, expected_mask.shape)
        self.assertTrue(torch.all(actual_mask == expected_mask))

    def test_root_last(self):
        num_words = torch.tensor([3])
        num_variables = 5
        root_first = False
        device = torch.device('cpu')

        expected_mask = torch.zeros(1, num_variables, num_variables, dtype=torch.bool, device=device)
        expected_mask[0, 4, 1:4] = True

        actual_mask = batched_base_mask(num_words, num_variables, root_first, device)

        self.assertEqual(actual_mask.shape, expected_mask.shape)
        self.assertTrue(torch.all(actual_mask == expected_mask))

if __name__ == '__main__':
    unittest.main()
