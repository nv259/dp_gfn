import unittest
import torch

from dp_gfn.nets.initial_encoders import LabelScorer

class TestLabelScorer(unittest.TestCase):
    def test_forward(self):
        # Create a LabelScorer instance
        scorer = LabelScorer(num_tags=5)

        # Create some dummy data
        heads = torch.randn(16, 10, 768)
        deps = torch.randn(16, 10, 768)

        # Run the scorer
        lab_scores = scorer(heads, deps)

        # Check the output shape
        self.assertEqual(lab_scores.shape, (16, 5))

print(f"Test LabelScorer")
TestLabelScorer().test_forward()
print("All tests passed!")