import torch
from config import V_threshold, alpha, input_rate, reward_switch_chunks
from brain import Brain, INPUT_IDX, OUTPUT_IDX
from reward import RewardModel


class Env:
	"""
	LIF network + Bernoulli input + output-rate reward.

	Input neuron (0): hard-set to V_threshold or 0 ~ Bernoulli(input_rate).
	Output neuron (1): EMA of spikes with decay alpha; x = 2*ema - 1 feeds f.
	"""

	def __init__(self, reward_model=None):
		self.brain = Brain()
		self.reward = reward_model if reward_model is not None else RewardModel()
		self.output_ema = torch.tensor(0.0)
		self.spike_frac = torch.tensor(0.0)  # hidden-only spike fraction this frame
		self._chunk_reward_sum = torch.tensor(0.0)
		self._chunk_spike_sum = torch.tensor(0.0)
		self._chunk_frames = 0
		self.last_chunk_reward = None
		self.last_chunk_spike_frac = None
		self.reward_switched = False

	@torch.no_grad()
	def reset(self):
		"""Start (or restart) the environment from a clean state."""
		self.brain = Brain()
		self.output_ema = torch.tensor(0.0)
		self.spike_frac = torch.tensor(0.0)
		self._chunk_reward_sum = torch.tensor(0.0)
		self._chunk_spike_sum = torch.tensor(0.0)
		self._chunk_frames = 0
		self.last_chunk_reward = None
		self.last_chunk_spike_frac = None
		self.reward_switched = False
		return self

	@torch.no_grad()
	def episode_frac(self):
		"""Chunks completed in this landscape, scaled to [0, 1]."""
		return self.reward.chunks_on_landscape / reward_switch_chunks

	@torch.no_grad()
	def reset_episode(self):
		"""Clear neural state between landscapes. Keeps the current reward model."""
		reward = self.reward
		self.reset()
		self.reward = reward
		return self

	@torch.no_grad()
	def step(self):
		"""
		One frame. Returns (actor_due, chunk_reward).
		chunk_reward is set only when a chunk ends; otherwise None.
		"""
		fired = torch.rand(()) < input_rate
		self.brain.v[INPUT_IDX] = V_threshold if fired else 0.0

		actor_due, chunk_ended = self.brain.step()

		# Hidden neurons only (exclude hard-set input and readout output).
		hidden_spikes = self.brain.last_spike[2:].float()
		self.spike_frac = hidden_spikes.mean() if hidden_spikes.numel() else torch.tensor(0.0)
		out_spike = self.brain.last_spike[OUTPUT_IDX].float()
		self.output_ema = alpha * self.output_ema + (1.0 - alpha) * out_spike

		x = 2.0 * self.output_ema - 1.0
		r = self.reward(x)
		self._chunk_reward_sum = self._chunk_reward_sum + r
		self._chunk_spike_sum = self._chunk_spike_sum + self.spike_frac
		self._chunk_frames += 1

		chunk_reward = None
		self.reward_switched = False
		if chunk_ended:
			chunk_reward = self._chunk_reward_sum / self._chunk_frames
			self.last_chunk_reward = chunk_reward
			self.last_chunk_spike_frac = self._chunk_spike_sum / self._chunk_frames
			self._chunk_reward_sum = torch.tensor(0.0)
			self._chunk_spike_sum = torch.tensor(0.0)
			self._chunk_frames = 0
			self.reward_switched = self.reward.on_chunk_end()

		return actor_due, chunk_reward
