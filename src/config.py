m = 32
n = 16
h = 256
num_heads = 4
num_neurons = 10

V_threshold = 1
beta = 0.98         # LIF leak (higher → slower leak, more spikes)
alpha = 0.98        # output spike EMA decay

# input neuron fires ~ Bernoulli(input_rate) each frame
input_rate = 0.5

# reward f(x) = (a0 + a1 x + a2 x^2 + a3 x^3)/8 + 0.5, x in (-1, 1)
# initial coeffs; resampled uniformly in [-1, 1] every reward_switch_chunks
a0 = 0.0
a1 = 1.0
a2 = 0.0
a3 = 0.0
reward_switch_chunks = 128

replay_memory = 50_000
batch_size = 32
gamma = 0.99
tau = 0.005

# DDPG exploration (TorchRL AdditiveGaussianModule: sigma_init/end, annealing_num_steps)
sigma_initial = 0.2
sigma_final = 0.05
sigma_decay = 10_000  # chunks until sigma reaches sigma_final

actor_lr = 1e-4
critic_lr = 1e-3
warmup_chunks = 64
train_chunks = 5_000
updates_per_chunk = 1
device = "cuda"
