import copy

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.data import Bounded, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import AdditiveGaussianModule

from actor import Actor
from config import (
	actor_lr,
	batch_size,
	critic_lr,
	device,
	gamma,
	h,
	m,
	n,
	num_neurons,
	replay_memory,
	sigma_decay,
	sigma_final,
	sigma_initial,
	tau,
	train_chunks,
	updates_per_chunk,
	warmup_chunks,
)
from critic import Critic
from env import Env


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau_: float):
	with torch.no_grad():
		for tp, p in zip(target.parameters(), source.parameters()):
			tp.data.lerp_(p.data, tau_)


def make_noise(dev: torch.device) -> AdditiveGaussianModule:
	spec = Bounded(
		low=-1.0,
		high=1.0,
		shape=(num_neurons, num_neurons),
		dtype=torch.float32,
		device=dev,
	)
	return AdditiveGaussianModule(
		spec=spec,
		sigma_init=sigma_initial,
		sigma_end=sigma_final,
		annealing_num_steps=sigma_decay,
		action_key="action",
		device=dev,
	)


@torch.no_grad()
def explore_action(actor, noise, vi, wi):
	action = actor(vi, wi)
	td = TensorDict({"action": action}, batch_size=action.shape[:-2], device=action.device)
	return noise(td)["action"]


def ddpg_update(actor, critic, actor_t, critic_t, actor_opt, critic_opt, batch):
	vi = batch["vi"]
	wi = batch["wi"]
	action = batch["action"]
	reward = batch["reward"]
	next_vi = batch["next_vi"]
	next_wi = batch["next_wi"]

	with torch.no_grad():
		next_action = actor_t(next_vi, next_wi)
		target_q = reward + gamma * critic_t(next_vi, next_wi, next_action)

	q = critic(vi, wi, action)
	critic_loss = F.mse_loss(q, target_q)
	critic_opt.zero_grad()
	critic_loss.backward()
	critic_opt.step()

	actor_loss = -critic(vi, wi, actor(vi, wi)).mean()
	actor_opt.zero_grad()
	actor_loss.backward()
	actor_opt.step()

	soft_update(actor_t, actor, tau)
	soft_update(critic_t, critic, tau)
	return float(critic_loss.item()), float(actor_loss.item())


def train():
	dev = torch.device(device if torch.cuda.is_available() else "cpu")
	print(f"device: {dev}")

	env = Env().reset()
	actor = Actor(m, n, h, num_neurons).to(dev)
	critic = Critic(m, n, h, num_neurons).to(dev)
	actor_t = copy.deepcopy(actor)
	critic_t = copy.deepcopy(critic)
	for p in actor_t.parameters():
		p.requires_grad_(False)
	for p in critic_t.parameters():
		p.requires_grad_(False)

	actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)
	critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr)
	noise = make_noise(dev)

	rb = TensorDictReplayBuffer(
		storage=LazyTensorStorage(replay_memory, device=dev),
		batch_size=batch_size,
	)

	pending = None  # (vi, wi, action) waiting for next chunk reward + next state
	chunks_done = 0
	log_every = 64  # divides reward_switch_chunks (128)

	while chunks_done < train_chunks:
		actor_due, chunk_reward = env.step()
		if not actor_due:
			continue

		vi, wi = env.brain.actor_inputs()
		vi = vi.to(dev)
		wi = wi.to(dev)

		if pending is not None:
			p_vi, p_wi, p_action = pending
			transition = TensorDict(
				{
					"vi": p_vi,
					"wi": p_wi,
					"action": p_action,
					"reward": chunk_reward.to(dev).view(1),
					"next_vi": vi,
					"next_wi": wi,
				},
				batch_size=[],
				device=dev,
			)
			rb.add(transition)

			if len(rb) >= warmup_chunks:
				for _ in range(updates_per_chunk):
					batch = rb.sample()
					c_loss, a_loss = ddpg_update(
						actor, critic, actor_t, critic_t, actor_opt, critic_opt, batch
					)
			else:
				c_loss = a_loss = float("nan")

			r = float(chunk_reward)
			chunks_done += 1
			if chunks_done % log_every == 0:
				x_star, r_star = env.reward.optimum()
				spike = float(env.last_chunk_spike_frac)
				print(
					f"chunk {chunks_done}/{train_chunks}  "
					f"r={r:.4f}  r*={r_star:.4f}  x*={x_star:.4f}  "
					f"spike={spike:.4f}  "
					f"sigma={float(noise.sigma):.4f}  "
					f"buffer={len(rb)}  "
					f"c_loss={c_loss:.4f}  a_loss={a_loss:.4f}"
				)


		action = explore_action(actor, noise, vi.unsqueeze(0), wi.unsqueeze(0)).squeeze(0)
		env.brain.set_next_w(action.cpu())
		noise.step(1)
		pending = (vi, wi, action.detach())

	return actor, critic


if __name__ == "__main__":
	train()
