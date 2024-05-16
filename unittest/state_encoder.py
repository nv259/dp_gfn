import unittest
import torch
from dp_gfn.nets.initial_encoders import StateEncoder

class TestStateEncoder(unittest.TestCase):
    def test_forward(self):
        # Create a StateEncoder instance
        encoder = StateEncoder(num_variables=10, num_tags=5)

        # Create some dummy data
        word_embeddings = torch.randn(16, 10, 768)
        adjacency = torch.randint(0, 5, (16, 10, 10))

        # Run the encoder
        state_embeddings = encoder(word_embeddings, adjacency)

        # Check the output shape
        self.assertEqual(state_embeddings.shape, (16, 100, 128 + 128 + 128))


print("Test StateEncoder")
TestStateEncoder().test_forward()
print("All tests passed!")




