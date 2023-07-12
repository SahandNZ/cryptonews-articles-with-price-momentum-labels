import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_embed = n_embed

        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)

        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_target(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size, n_samples=5):
        noise_dist = torch.ones(self.n_vocab)

        current_device = next(self.parameters()).device
        noise_words = torch.multinomial(input=noise_dist, num_samples=batch_size * n_samples, replacement=True)
        noise_words = noise_words.to(current_device)

        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        return noise_vectors
