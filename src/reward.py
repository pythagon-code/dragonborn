import math

import torch
from config import a0, a1, a2, a3, reward_switch_chunks


class RewardModel:
	"""
	f(x) = (a0 + a1 x + a2 x^2 + a3 x^3)/8 + 0.5,  x in [-1, 1].
	Coeffs live in [-1, 1]; landscape resamples every reward_switch_chunks.
	"""

	def __init__(self, coeffs=None):
		if coeffs is None:
			coeffs = (a0, a1, a2, a3)
		self.a = torch.tensor(coeffs, dtype=torch.float32)
		self.chunks_on_landscape = 0

	@torch.no_grad()
	def resample(self):
		self.a = torch.empty(4).uniform_(-1.0, 1.0)
		self.chunks_on_landscape = 0

	@torch.no_grad()
	def __call__(self, x):
		x = torch.as_tensor(x, dtype=torch.float32).clamp(-1.0, 1.0)
		a0_, a1_, a2_, a3_ = self.a
		return (a0_ + a1_ * x + a2_ * x ** 2 + a3_ * x ** 3) / 8.0 + 0.5

	@torch.no_grad()
	def optimum(self):
		"""
		Maximize f on [-1, 1] via critical points of f' = 0 and endpoints.
		f'(x) ∝ a1 + 2 a2 x + 3 a3 x^2.
		Returns (x_star, f_star).
		"""
		a0_, a1_, a2_, a3_ = (float(v) for v in self.a)
		candidates = [-1.0, 1.0]

		A, B, C = 3.0 * a3_, 2.0 * a2_, a1_
		if abs(A) < 1e-12:
			if abs(B) > 1e-12:
				x = -C / B
				if -1.0 <= x <= 1.0:
					candidates.append(x)
		else:
			disc = B * B - 4.0 * A * C
			if disc >= 0.0:
				sqrt_d = math.sqrt(disc)
				for x in ((-B + sqrt_d) / (2.0 * A), (-B - sqrt_d) / (2.0 * A)):
					if -1.0 <= x <= 1.0:
						candidates.append(x)

		best_x, best_f = candidates[0], float(self(candidates[0]))
		for x in candidates[1:]:
			fx = float(self(x))
			if fx > best_f:
				best_x, best_f = x, fx
		return best_x, best_f

	@torch.no_grad()
	def on_chunk_end(self):
		"""Count a finished chunk; switch landscape when due. Returns True if switched."""
		self.chunks_on_landscape += 1
		if self.chunks_on_landscape >= reward_switch_chunks:
			self.resample()
			return True
		return False
