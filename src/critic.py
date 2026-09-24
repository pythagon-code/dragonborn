import torch
from torch import nn


class Critic(nn.Module):
	def __init__(self, m, n, h, num_neurons):
		super().__init__()
		self.n = n
		self.num_neurons = num_neurons
		self.pos_embed = nn.Parameter(torch.randn(1, n * num_neurons, h))
		feat = m + 2 * num_neurons  # vi + wi + next_wi (incoming)
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
			nn.Linear(h + 1, h),
			nn.LeakyReLU(),
			nn.Linear(h, h),
			nn.LeakyReLU(),
			nn.Linear(h, 1),
		)

	def forward(self, vi, wi, next_wi, t):
		"""
		vi, wi: (..., n * num_neurons, m) and (..., n * num_neurons, num_neurons)
		next_wi: proposed next-chunk weights (..., num_neurons, num_neurons)
		t: episode progress in [0, 1], shape (..., 1)
		"""
		N = self.num_neurons
		# token (c, i) gets incoming row i of the proposed next W
		next_tok = next_wi.unsqueeze(-3).expand(*next_wi.shape[:-2], self.n, N, N)
		next_tok = next_tok.reshape(*next_wi.shape[:-2], self.n * N, N)

		x = torch.cat([vi, wi, next_tok], dim=-1)
		q = self.q_ffn(x) + self.pos_embed
		k = self.k_ffn(x) + self.pos_embed
		v = self.v_ffn(x) + self.pos_embed
		attn, _ = self.attn(q, k, v)
		attn = attn.mean(dim=1)
		if t.ndim == attn.ndim - 1:
			t = t.unsqueeze(-1)
		return self.out_ffn(torch.cat([attn, t], dim=-1))
