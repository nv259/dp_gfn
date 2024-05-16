import unittest
import torch

from dp_gfn.nets.initial_encoders import PrefEncoder

class TestPrefEncoder(unittest.TestCase):
    def test_forward(self):
        # Create a PrefEncoder instance
        encoder = PrefEncoder(pretrained_path="bert-base-uncased")

        # Create some dummy data
        batch = ["This is a sentence.", "This is another sentence."]

        # Run the encoder
        word_embeddings = encoder(batch)

        # Check the output shape
        self.assertEqual(word_embeddings.shape, (2, 160, 768))

print(f"Test PrefEncoder")
TestPrefEncoder().test_forward()
print("All tests passed!")