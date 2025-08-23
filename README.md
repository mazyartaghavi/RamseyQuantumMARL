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

# 2) Train on the UAV forest environment (10 agents, small grid)
python scripts/train_uav.py --config configs/uav_default.yaml

# 3) (Optional) Run unit test for the quantum-inspired operator
pytest -q

