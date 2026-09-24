import torch
from config import m, n, num_neurons, V_threshold, beta

INPUT_IDX = 0
OUTPUT_IDX = 1


class Brain:
	def __init__(self):
		# Ring buffers. When full, _w_pos is the oldest chunk index (next write).
		self.V = torch.zeros(n, m, num_neurons)           # [chunk, frame, neuron]
		self.W = torch.zeros(n, num_neurons, num_neurons)  # [chunk, post, pre]
		self.v = torch.zeros(num_neurons)
		self.w = torch.zeros(num_neurons, num_neurons)
		self.last_spike = torch.zeros(num_neurons, dtype=torch.bool)
		self._frame_in_chunk = 0
		self._w_pos = 0
		self._chunks_filled = 0  # for ready (capped at n)
		self.total_chunks = 0    # lifetime chunk count
		self._pending_w = None   # applied at the start of the next chunk

	@torch.no_grad()
	def set_next_w(self, next_wi):
		"""Queue actor output to become self.w when the next chunk begins."""
		self._pending_w = next_wi.detach().clone()

	@torch.no_grad()
	def step(self):
		"""
		Advance one frame. Caller may hard-set v[INPUT_IDX] before calling.

		Returns (actor_due, chunk_ended).
		actor_due: history window full and a chunk just finished — run actor.
		"""
		c = self._w_pos

		if self._frame_in_chunk == 0:
			if self._pending_w is not None:
				self.w = self._pending_w
				self._pending_w = None
			self.W[c] = self.w

		# Input is hard-set by Env; preserve it and skip LIF/reset on that neuron.
		input_v = self.v[INPUT_IDX].clone()

		self.last_spike = self.v >= V_threshold
		# w[i, j] = weight from j -> i (incoming to i)
		self.v = beta * self.v + self.w @ self.last_spike.float()

		# Hard-reset hidden neurons only (not input, not output).
		reset = self.last_spike.clone()
		reset[INPUT_IDX] = False
		reset[OUTPUT_IDX] = False
		self.v = self.v * (~reset).float()
		self.v[INPUT_IDX] = input_v

		self.V[c, self._frame_in_chunk] = self.v

		self._frame_in_chunk = (self._frame_in_chunk + 1) % m
		chunk_ended = self._frame_in_chunk == 0
		if chunk_ended:
			self._w_pos = (self._w_pos + 1) % n
			self._chunks_filled = min(self._chunks_filled + 1, n)
			self.total_chunks += 1

		actor_due = chunk_ended and self.ready
		return actor_due, chunk_ended

	@property
	def ready(self):
		return self._chunks_filled == n

	def _chunk_order(self):
		"""Oldest → newest chunk indices into the ring buffers."""
		return (torch.arange(n) + self._w_pos) % n

	@torch.no_grad()
	def actor_inputs(self):
		"""
		Build (vi, wi) matching Actor.forward (views/indexing, no stack).

		vi: (n * num_neurons, m)       — neuron i's voltages over chunk c
		wi: (n * num_neurons, num_neurons) — incoming weights to i during chunk c

		Token order: chunk-major, then neuron:
		  token = c * num_neurons + i
		"""
		if not self.ready:
			raise RuntimeError("Need n full chunks of history before actor_inputs()")

		order = self._chunk_order()
		vi = self.V[order].permute(0, 2, 1).reshape(n * num_neurons, m)
		wi = self.W[order].reshape(n * num_neurons, num_neurons)
		return vi, wi
