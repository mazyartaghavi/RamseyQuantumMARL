import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
from qio_marl.agents.policy import ActorCritic
from qio_marl.agents.qio_operator import apply_q_operator
from qio_marl.utils.schedules import EntropyController
from qio_marl.utils.logger import safe_mean

class QIOMARL:
    def __init__(self, cfg, obs_dim, act_dim, n_agents, out_dir):
        self.device = torch.device(cfg["train"]["device"])
        self.n_agents = n_agents
        self.gamma = cfg["algo"]["gamma"]
        self.ent = EntropyController(cfg["algo"]["entropy"])
        self.q_cfg = cfg["algo"]["q_operator"]

        self.net = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt_pi = optim.Adam(self.net.pi.parameters(), lr=cfg["algo"]["lr_actor"])
        self.opt_v  = optim.Adam(self.net.v.parameters(),  lr=cfg["algo"]["lr_critic"])

        self.rollout_len = cfg["train"]["rollout_len"]
        self.reset_buffers()
        self.last_entropy = 0.0
        self.last_alpha = self.ent.alpha

    def reset_buffers(self):
        self.buf = {"obs": [], "act": [], "rew": [], "done": [], "logp": [], "val": [], "extra": []}

    @torch.no_grad()
    def act(self, obs_np, greedy=False):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        logits, value = self.net(obs)

        if self.q_cfg["enabled"]:
            q_logits, probs = apply_q_operator(
                logits,
                eta=self.q_cfg["eta"],
                min_entropy=self.q_cfg["min_entropy"],
                temperature=self.q_cfg["temperature"],
            )
            logits = q_logits
        else:
            probs = torch.softmax(logits, dim=-1)

        H = -(probs * (probs.add(1e-8).log())).sum(dim=-1).mean().item()
        self.last_entropy = H
        self.last_alpha = self.ent.update(H)  # may use quantum-inspired decay

        # entropy-tempered sampling
        temperature = max(1e-6, 1.0)  # fixed here; alpha used in loss
        if greedy:
            actions = torch.argmax(logits, dim=-1)
            logp = torch.zeros_like(value)
        else:
            dist = torch.distributions.Categorical(logits=logits / temperature)
            actions = dist.sample()
            logp = dist.log_prob(actions)

        return actions.cpu().numpy(), logp.detach(), value.detach(), {"logits": logits.detach()}

    def store(self, obs, act, rew, done, logp, val, extra):
        self.buf["obs"].append(torch.tensor(obs, dtype=torch.float32))
        self.buf["act"].append(torch.tensor(act))
        self.buf["rew"].append(torch.tensor(rew, dtype=torch.float32))
        self.buf["done"].append(torch.tensor(done, dtype=torch.float32))
        self.buf["logp"].append(logp.cpu())
        self.buf["val"].append(val.cpu())
        self.buf["extra"].append(extra)

    def ready_to_update(self):
        return len(self.buf["obs"]) >= self.rollout_len

    def update(self):
        obs = torch.stack(self.buf["obs"]).to(self.device)              # (T, n_agents, obs_dim)
        act = torch.stack(self.buf["act"]).to(self.device)              # (T, n_agents)
        rew = torch.stack(self.buf["rew"]).to(self.device)              # (T,)
        done= torch.stack(self.buf["done"]).to(self.device)             # (T,)
        old_logp = torch.stack(self.buf["logp"]).to(self.device)        # (T, n_agents)
        val_t = torch.stack(self.buf["val"]).to(self.device)            # (T, n_agents)

        # bootstrap value with zero at episode end (simple on-policy A2C)
        with torch.no_grad():
            returns = []
            g = torch.zeros(self.n_agents, device=self.device)
            for t in reversed(range(len(rew))):
                g = rew[t].unsqueeze(0).repeat(self.n_agents) + self.gamma * g * (1.0 - done[t])
                returns.append(g)
            returns = torch.stack(list(reversed(returns)))              # (T, n_agents)
            adv = returns - val_t

        # policy loss with entropy
        logits, value = self.net(obs.view(-1, obs.size(-1)))
        if self.q_cfg["enabled"]:
            logits, _ = apply_q_operator(logits, **{k: self.q_cfg[k] for k in ["eta","min_entropy","temperature"]})

        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(act.view(-1))
        entropy = dist.entropy().mean()

        alpha = self.last_alpha
        policy_loss = -(logp * adv.view(-1).detach()).mean() - alpha * entropy

        value_loss = F.mse_loss(value.view_as(adv), returns.view_as(adv))
        loss = policy_loss + value_loss

        self.opt_pi.zero_grad(); self.opt_v.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.opt_pi.step(); self.opt_v.step()

        metrics = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "alpha": float(alpha),
            "adv_mean": adv.mean().item(),
            "ret_mean": returns.mean().item(),
        }
        self.reset_buffers()
        return metrics
