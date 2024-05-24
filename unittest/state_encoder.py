import unittest
import torch

from dp_gfn.nets.initial_encoders import StateEncoder


class TestStateEncoder(unittest.TestCase):

    def setUp(self):
        self.num_variables = 3
        self.num_tags = 5
        self.word_embedding_dim = 768
        self.node_embedding_dim = 128
        self.label_embedding_dim = 128
        self.hidden_layers = [512, 256]
        self.dropout_rate = 0.1
        self.activation = 'ReLU'
        self.encode_label = True

        self.encoder = StateEncoder(
            self.num_variables,
            self.num_tags,
            self.word_embedding_dim,
            self.node_embedding_dim,
            self.label_embedding_dim,
            self.hidden_layers,
            self.dropout_rate,
            self.activation,
            self.encode_label
        )

    def test_forward_pass(self):
        batch_size = 10
        word_embeddings = torch.randn(batch_size, self.num_variables, self.word_embedding_dim)
        adjacency = torch.randint(0, 2, (batch_size, self.num_variables, self.num_variables))

        state_embeddings, adjacency = self.encoder(word_embeddings, adjacency)

        self.assertEqual(state_embeddings.shape, (batch_size, self.num_variables ** 2, self.node_embedding_dim + self.node_embedding_dim + self.label_embedding_dim))
        self.assertEqual(adjacency.shape, (batch_size, self.num_variables * self.num_variables))

    def test_forward_pass_without_label_encoding(self):
        self.encoder = StateEncoder(
            self.num_variables,
            self.num_tags,
            self.word_embedding_dim,
            self.node_embedding_dim,
            self.label_embedding_dim,
            self.hidden_layers,
            self.dropout_rate,
            self.activation,
            encode_label=False
        )

        batch_size = 10
        word_embeddings = torch.randn(batch_size, self.num_variables, self.word_embedding_dim)
        adjacency = torch.randint(0, 2, (batch_size, self.num_variables, self.num_variables))

        state_embeddings, adjacency = self.encoder(word_embeddings, adjacency)

        self.assertEqual(state_embeddings.shape, (batch_size, self.num_variables ** 2, self.node_embedding_dim * 2))
        self.assertEqual(adjacency.shape, (batch_size, self.num_variables * self.num_variables))

    def test_forward_pass_with_invalid_input_shape(self):
        batch_size = 10
        word_embeddings = torch.randn(batch_size, self.num_variables + 1, self.word_embedding_dim)
        adjacency = torch.randint(0, 2, (batch_size, self.num_variables, self.num_variables))

        with self.assertRaises(AssertionError):
            self.encoder(word_embeddings, adjacency)

    def test_forward_pass_with_invalid_adjacency_shape(self):
        batch_size = 10
        word_embeddings = torch.randn(batch_size, self.num_variables, self.word_embedding_dim)
        adjacency = torch.randint(0, 2, (batch_size, self.num_variables + 1, self.num_variables))

        with self.assertRaises(AssertionError):
            self.encoder(word_embeddings, adjacency)

if __name__ == '__main__':
    unittest.main()
