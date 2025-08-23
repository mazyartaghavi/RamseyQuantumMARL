import numpy as np
from numpy.random import default_rng

class ForestUAV:
    """
    Toy cooperative coverage environment:
    - Grid world, N agents move in {stay, up, down, left, right}
    - Partial observation: local (2*obs_radius+1)^2 crop + local vegetation density
    - Reward: +1 for visiting previously unmonitored cell; small penalty for collisions
    - Limited comms: with probability comm_budget, agents share their positions (toy)
    - Episode ends at fixed horizon.

    Observation: concatenated local map crop + agent (x,y) normalized coords.
    Action space: 5 discrete actions.
    """
    def __init__(self, grid_size=20, n_agents=10, obs_radius=2, episode_len=200, seed=42, comm_budget=0.1):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.obs_radius = obs_radius
        self.episode_len = episode_len
        self.comm_budget = comm_budget
        self.rng = default_rng(seed)
        self.t = 0

        self.action_dim = 5
        crop = (2*obs_radius+1) ** 2
        self.observation_dim = crop + 2  # local map + (x,y)
        self.reset()

    def _rand_positions(self):
        pos = set()
        coords = []
        while len(coords) < self.n_agents:
            x = self.rng.integers(0, self.grid_size)
            y = self.rng.integers(0, self.grid_size)
            if (x,y) not in pos:
                pos.add((x,y))
                coords.append([x,y])
        return np.array(coords, dtype=np.int32)

    def reset(self, eval_mode=False):
        self.t = 0
        self.map = self.rng.random((self.grid_size, self.grid_size))  # vegetation density proxy
        self.cover = np.zeros_like(self.map, dtype=np.int32)          # visited flags
        self.agents = self._rand_positions()
        obs = self._get_obs()
        return obs, {}

    def _crop(self, x, y):
        r = self.obs_radius
        xs = np.clip(np.arange(x - r, x + r + 1), 0, self.grid_size-1)
        ys = np.clip(np.arange(y - r, y + r + 1), 0, self.grid_size-1)
        patch = self.map[np.ix_(xs, ys)]
        return patch

    def _get_obs(self):
        obs = []
        for i in range(self.n_agents):
            x, y = self.agents[i]
            patch = self._crop(x, y).flatten()
            xy = np.array([x / (self.grid_size-1), y / (self.grid_size-1)], dtype=np.float32)
            obs.append(np.concatenate([patch, xy]).astype(np.float32))
        return np.stack(obs, axis=0)

    def step(self, actions):
        actions = np.asarray(actions)
        assert actions.shape == (self.n_agents,)
        self.t += 1

        # movement
        deltas = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, -1),
            4: (0, 1),
        }
        next_pos = self.agents.copy()
        for i, a in enumerate(actions):
            dx, dy = deltas[int(a)]
            nx = np.clip(next_pos[i,0] + dx, 0, self.grid_size-1)
            ny = np.clip(next_pos[i,1] + dy, 0, self.grid_size-1)
            next_pos[i] = [nx, ny]

        # collisions penalty
        uniq, counts = np.unique(next_pos, axis=0, return_counts=True)
        collision_cells = {tuple(u) for u,c in zip(uniq, counts) if c > 1}

        # coverage reward: +1 for first visit (team)
        reward = 0.0
        for i in range(self.n_agents):
            x, y = next_pos[i]
            if self.cover[x, y] == 0:
                reward += 1.0
                self.cover[x, y] = 1

        # collisions penalize
        reward -= 0.25 * len(collision_cells)

        self.agents = next_pos

        done = (self.t >= self.episode_len)
        obs = self._get_obs()
        info = {}
        return obs, reward, done, info
