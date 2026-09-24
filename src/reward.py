import math

import torch
from config import a0, a1, a2, a3, reward_switch_chunks


class RewardModel:
	"""
	Cubic landscape on [-1, 1], affine-normalized to [0, 1]:
	  raw(x) = a0 + a1 x + a2 x^2 + a3 x^3
	  f(x) = (raw(x) - raw_min) / (raw_max - raw_min)

	So r* is always 1 at the interior mode. Coeffs in [-1, 1];
	resamples reject endpoint optima and flat landscapes.
	"""

	def __init__(self, coeffs=None):
		if coeffs is None:
			coeffs = (a0, a1, a2, a3)
		self.a = torch.tensor(coeffs, dtype=torch.float32)
		self.chunks_on_landscape = 0
		self._raw_min = 0.0
		self._raw_max = 1.0
		self._refresh_bounds()
		x_star, _ = self.optimum()
		if abs(x_star) >= 1.0 - 1e-9 or self._raw_max - self._raw_min < 1e-6:
			self.resample()

	def _critical_xs(self):
		a0_, a1_, a2_, a3_ = (float(v) for v in self.a)
		xs = [-1.0, 1.0]
		A, B, C = 3.0 * a3_, 2.0 * a2_, a1_
		if abs(A) < 1e-12:
			if abs(B) > 1e-12:
				x = -C / B
				if -1.0 <= x <= 1.0:
					xs.append(x)
		else:
			disc = B * B - 4.0 * A * C
			if disc >= 0.0:
				sqrt_d = math.sqrt(disc)
				for x in ((-B + sqrt_d) / (2.0 * A), (-B - sqrt_d) / (2.0 * A)):
					if -1.0 <= x <= 1.0:
						xs.append(x)
		return xs

	def _raw(self, x):
		x = torch.as_tensor(x, dtype=torch.float32)
		a0_, a1_, a2_, a3_ = self.a
		return a0_ + a1_ * x + a2_ * x ** 2 + a3_ * x ** 3

	@torch.no_grad()
	def _refresh_bounds(self):
		vals = [float(self._raw(x)) for x in self._critical_xs()]
		self._raw_min = min(vals)
		self._raw_max = max(vals)

	@torch.no_grad()
	def resample(self):
		"""Draw coeffs until interior mode and non-flat range on [-1, 1]."""
		while True:
			self.a = torch.empty(4).uniform_(-1.0, 1.0)
			self._refresh_bounds()
			if self._raw_max - self._raw_min < 1e-6:
				continue
			x_star, _ = self.optimum()
			if abs(x_star) < 1.0 - 1e-9:
				break
		self.chunks_on_landscape = 0

	@torch.no_grad()
	def __call__(self, x):
		x = torch.as_tensor(x, dtype=torch.float32).clamp(-1.0, 1.0)
		span = self._raw_max - self._raw_min
		return (self._raw(x) - self._raw_min) / span

	@torch.no_grad()
	def optimum(self):
		"""
		Maximize f on [-1, 1] via critical points of raw' = 0 and endpoints.
		Returns (x_star, f_star) with f_star == 1 after normalization.
		"""
		candidates = self._critical_xs()
		best_x = candidates[0]
		best_raw = float(self._raw(best_x))
		for x in candidates[1:]:
			rx = float(self._raw(x))
			if rx > best_raw:
				best_x, best_raw = x, rx
		span = self._raw_max - self._raw_min
		f_star = (best_raw - self._raw_min) / span if span > 1e-12 else 1.0
		return best_x, f_star

	@torch.no_grad()
	def on_chunk_end(self):
		"""Count a finished chunk; switch landscape when due. Returns True if switched."""
		self.chunks_on_landscape += 1
		if self.chunks_on_landscape >= reward_switch_chunks:
			self.resample()
			return True
		return False
