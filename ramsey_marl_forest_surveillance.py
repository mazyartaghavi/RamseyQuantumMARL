
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import random
import itertools

# Set plotting style
sns.set(style="whitegrid")

# Constants
NUM_AGENTS = 10
NUM_SECTORS = 25
CLIQUE_THRESHOLD = 4
PENALTY_LAMBDA = 1.0
ENTROPY_THRESHOLD = 0.1
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Simulated entropy values for sectors (representing uncertainty)
entropy_values = np.random.rand(NUM_SECTORS)

# Function to visualize entropy map
def plot_entropy_map(entropy_values, title="Forest Sector Entropy Map"):
    grid_size = int(np.ceil(np.sqrt(len(entropy_values))))
    padded_entropy = np.pad(entropy_values, (0, grid_size**2 - len(entropy_values)), constant_values=np.nan)
    entropy_grid = padded_entropy.reshape((grid_size, grid_size))

    plt.figure(figsize=(8, 6))
    sns.heatmap(entropy_grid, annot=True, cmap="YlGnBu", cbar=True, square=True, linewidths=0.5, linecolor='gray')
    plt.title(title)
    plt.tight_layout()
    plt.savefig("entropy_map.png")
    plt.close()

# Simulate agent observations (subset of sectors)
def simulate_agent_observations(num_agents, num_sectors, obs_per_agent=3):
    return {i: random.sample(range(num_sectors), obs_per_agent) for i in range(num_agents)}

# Construct similarity graph and visualize Ramsey cliques
def plot_similarity_graph(agent_observations, entropy_values, threshold=ENTROPY_THRESHOLD):
    G = nx.Graph()
    G.add_nodes_from(agent_observations.keys())

    for i, j in itertools.combinations(agent_observations.keys(), 2):
        hi = np.mean([entropy_values[s] for s in agent_observations[i]])
        hj = np.mean([entropy_values[s] for s in agent_observations[j]])
        if abs(hi - hj) < threshold:
            G.add_edge(i, j)

    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=SEED)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800)
    plt.title("Agent Entropy Similarity Graph (Potential Ramsey Cliques)")
    plt.tight_layout()
    plt.savefig("similarity_graph.png")
    plt.close()

# Generate and visualize QUBO matrix
def construct_qubo_matrix(entropy_values, agent_observations, num_agents, num_sectors, lambda_penalty):
    Q = np.zeros((num_agents * num_sectors, num_agents * num_sectors))

    for i in range(num_agents):
        for s in agent_observations[i]:
            idx = i * num_sectors + s
            Q[idx, idx] = -entropy_values[s]

    for i, j in itertools.combinations(agent_observations.keys(), 2):
        hi = np.mean([entropy_values[s] for s in agent_observations[i]])
        hj = np.mean([entropy_values[s] for s in agent_observations[j]])
        if abs(hi - hj) < ENTROPY_THRESHOLD:
            for s1 in agent_observations[i]:
                for s2 in agent_observations[j]:
                    idx1 = i * num_sectors + s1
                    idx2 = j * num_sectors + s2
                    Q[idx1, idx2] += lambda_penalty

    return Q

# Plot heatmap of QUBO matrix
def plot_qubo_matrix(Q):
    plt.figure(figsize=(10, 8))
    sns.heatmap(Q, cmap="coolwarm", center=0)
    plt.title("QUBO Matrix Visualization")
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.savefig("qubo_matrix.png")
    plt.close()

# Driver function for the whole experiment
def run_simulation():
    plot_entropy_map(entropy_values)
    agent_obs = simulate_agent_observations(NUM_AGENTS, NUM_SECTORS)
    plot_similarity_graph(agent_obs, entropy_values)
    qubo_matrix = construct_qubo_matrix(entropy_values, agent_obs, NUM_AGENTS, NUM_SECTORS, PENALTY_LAMBDA)
    plot_qubo_matrix(qubo_matrix)

    return {
        "entropy_values": entropy_values,
        "agent_observations": agent_obs,
        "qubo_matrix": qubo_matrix
    }

# Run and save results
if __name__ == "__main__":
    results = run_simulation()
