"""Dragonborn DDPG entrypoint. Prefer: conda run -n cuda python train.py"""

from train import train


if __name__ == "__main__":
	train()
