import unittest
import torch

from dp_gfn.nets.encoders import LinearTransformer, MLP
from dp_gfn.nets.attention import LinearMultiHeadAttention



class TestLinearTransformer(unittest.TestCase):

    def test_init(self):
        # Test with label embedding
        model = LinearTransformer(
            input_dim=128,
            num_heads=4,
            d_k=32,
            d_v=32,
            d_model=128,
            label_embedded=True,
        )
        self.assertEqual(model.input_dim, 128)
        self.assertEqual(model.d_model, 128)
        self.assertEqual(len(model.layer_norms), 2)
        self.assertIsInstance(model.attention, LinearMultiHeadAttention)
        self.assertIsInstance(model.dense, MLP)

        # Test without label embedding
        model = LinearTransformer(
            input_dim=128,
            num_heads=4,
            d_k=32,
            d_v=32,
            d_model=64,
            label_embedded=False,
            num_tags=10,
            label_embedding_dim=64,
        )
        self.assertEqual(model.input_dim, 128)
        self.assertEqual(model.d_model, 64)
        self.assertEqual(model.label_embedding_dim, 64)
        self.assertEqual(len(model.label_embeddings), 2)
        self.assertEqual(len(model.layer_norms), 2)
        self.assertIsInstance(model.attention, LinearMultiHeadAttention)
        self.assertIsInstance(model.dense, MLP)

    def test_forward_with_labels(self):
        model = LinearTransformer(
            input_dim=32 * 3,
            num_heads=4,
            d_k=16,
            d_model=32 * 2,
            label_embedded=False,
            num_tags=10,
            label_embedding_dim=32,
        )
        x = torch.randn(2, 9, 32 * 2)
        labels = torch.randint(0, 10, (2, 9))
        output = model(x, labels)
        self.assertEqual(output.shape, (2, 9, 32 * 2))

    def test_forward_without_labels(self):
        model = LinearTransformer(
            input_dim=32 * 3,
            num_heads=4,
            d_k=16,
            d_model=32 * 3,
            label_embedded=True,
        )
        x = torch.randn(2, 9, 32 * 3)
        output = model(x)
        self.assertEqual(output.shape, (2, 9, 32 * 3))

if __name__ == '__main__':
    unittest.main()
