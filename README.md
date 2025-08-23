# Ramsey-Guided Quantum Optimization for Exploration–Exploitation in Multi-Agent Reinforcement Learning (QIO-MARL)

QIO-MARL is a research codebase that implements **quantum-inspired operator updates**
and **entropy-regularized control** for multi-agent reinforcement learning under partial
observability. It includes:
- A simple cooperative UAV forest-monitoring environment (grid, partial obs, limited comms)
- Quantum-inspired operator `𝒬` (amplitude-amplification–style reweighting of logits)
- Entropy annealing (classical + quantum-inspired decay law)
- A minimal Actor–Critic (shared parameters) training loop for N agents
- Reproducible config, logging, and unit test for `𝒬`.

> This repo is intentionally small and pedagogical—ideal as a starting point to reproduce
> trends and extend for larger benchmarks (SMAC/MPE).

## Quickstart

```bash
# 1) Clone or copy this repository
python -m venv .venv && source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
pip install -r requirements.txt
Artifacts (plots, CSV logs, checkpoints) are saved under runs/<timestamp>/.

Core Ideas

Random operators + contraction in expectation.
We model each update as a random operator; the expected operator is contractive,
yielding almost-sure convergence under standard boundedness and measurability assumptions.

Quantum-inspired operator (𝒬).
We reweight action logits with an amplitude-style map on probabilities:

Accentuates high-probability actions while preserving a min entropy floor

Plays well with entropy regularization to avoid premature collapse

Entropy control.
We provide both classical exponential decay and a “quantum-inspired” aggregated decay:

𝛼
𝑡
=
𝛼
0
exp
⁡
 ⁣
(
−
𝜆
∑
𝑘
=
1
𝑡
1
−
𝐻
𝑘
2
)
α
t
	​

=α
0
	​

exp(−λ
k=1
∑
t
	​

1−H
k
2
	​

	​

)
Repository Contents

qio_marl/agents/policy.py — Shared-parameter Actor–Critic (MLP)

qio_marl/agents/qio_operator.py — Quantum-inspired operator apply_q_operator

qio_marl/algos/qio_marl.py — Training update (entropy-regularized A2C + 𝒬)

qio_marl/envs/forest_uav.py — Toy UAV grid env (partial obs, coverage reward)

qio_marl/utils/* — Logger, schedules, replay buffer

scripts/train_uav.py — CLI experiment runner

configs/uav_default.yaml — Default configuration

tests/test_q_operator.py — Unit test for 𝒬

Configuration

See configs/uav_default.yaml for:

env: grid size, #agents, obs radius, episode length

algo: learning rates, gamma, entropy schedule, 𝒬-operator hyperparams

train: total steps, log interval, seed
# 2) Train on the UAV forest environment (10 agents, small grid)
python scripts/train_uav.py --config configs/uav_default.yaml

# 3) (Optional) Run unit test for the quantum-inspired operator
pytest -q

