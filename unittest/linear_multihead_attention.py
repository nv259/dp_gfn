import unittest
import torch

from dp_gfn.nets.attention import LinearMultiHeadAttention


class TestLinearMultiHeadAttention(unittest.TestCase):

    def test_forward_pass(self):
        # Define input tensors
        batch_size = 1
        target_len = 160 ** 2
        source_len = 160 ** 2           
        input_size = 10 * 3    # | source_dim | target_dim | label_dim | 
        num_heads = 4
        d_k = 64
        d_model = d_k * num_heads

        query = torch.randn(batch_size, target_len, input_size)
        key = torch.randn(batch_size, source_len, input_size)
        value = torch.randn(batch_size, source_len, input_size)

        # Create LinearMultiHeadAttention instance
        model = LinearMultiHeadAttention(input_size, num_heads, d_k, d_model=d_model)

        # Compute attention output
        output = model(query, key, value)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, target_len, d_model))

    def test_projection(self):
        # Define input tensor
        input_size = 10
        x = torch.randn(1, input_size)

        # Create LinearMultiHeadAttention instance
        model = LinearMultiHeadAttention(input_size, 1, 8)

        # Apply projection
        projected_x = model.projection(x)

        # Check output shape
        self.assertEqual(projected_x.shape, x.shape)

    def test_invalid_mask(self):
        # Define input tensors
        batch_size = 1
        target_len = 160 ** 2
        source_len = 160 ** 2           
        input_size = 10 * 3    # | source_dim | target_dim | label_dim | 
        num_heads = 4
        d_k = 64

        query = torch.randn(batch_size, target_len, input_size)
        key = torch.randn(batch_size, source_len, input_size)
        value = torch.randn(batch_size, source_len, input_size)
        mask = torch.randn(batch_size, target_len, source_len)

        # Create LinearMultiHeadAttention instance
        model = LinearMultiHeadAttention(input_size, num_heads, d_k)

        # Test with mask
        with self.assertRaises(NotImplementedError):
            model(query, key, value, mask)

if __name__ == '__main__':
    unittest.main()
