import torch
from torch import nn


class Actor(nn.Module):
	def __init__(self, m, n, h, num_neurons):
		super().__init__()
		self.n = n
		self.num_neurons = num_neurons
		self.pos_embed = nn.Parameter(torch.randn(1, n * num_neurons, h))
		feat = m + num_neurons  # vi + wi (incoming)
		self.q_ffn, self.k_ffn, self.v_ffn = (
			nn.Sequential(
				nn.Linear(feat, h),
				nn.LeakyReLU(),
				nn.Linear(h, h),
				nn.LeakyReLU(),
				nn.Linear(h, h),
			)
			for _ in range(3)
		)
		self.attn = nn.MultiheadAttention(embed_dim=h, num_heads=4, batch_first=True)
		self.out_ffn = nn.Sequential(
			nn.Linear(h, h),
			nn.LeakyReLU(),
			nn.Linear(h, h),
			nn.LeakyReLU(),
			nn.Linear(h, num_neurons * num_neurons),
			nn.Tanh(),
		)

	def forward(self, vi, wi):
		"""
		vi, wi: (..., n * num_neurons, m) and (..., n * num_neurons, num_neurons)
		returns next-chunk weight matrix (..., num_neurons, num_neurons) in (-1, 1)
		"""
		N = self.num_neurons
		x = torch.cat([vi, wi], dim=-1)
		q = self.q_ffn(x) + self.pos_embed
		k = self.k_ffn(x) + self.pos_embed
		v = self.v_ffn(x) + self.pos_embed
		attn, _ = self.attn(q, k, v)
		attn = attn.mean(dim=1)
		return self.out_ffn(attn).view(*attn.shape[:-1], N, N)
