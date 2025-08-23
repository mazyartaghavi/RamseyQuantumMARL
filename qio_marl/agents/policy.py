import torch, torch.nn as nn

def mlp(sizes, activation=nn.ReLU, out_act=None):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if act is not None:
            layers += [act()]
    return nn.Sequential(*layers)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.pi = mlp([obs_dim, hidden, hidden, act_dim])
        self.v  = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs):
        logits = self.pi(obs)     # (B, act_dim)
        value  = self.v(obs).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def action(self, obs, temperature=1.0, greedy=False):
        logits, value = self.forward(obs)
        if greedy:
            act = torch.argmax(logits, dim=-1)
            logp = torch.zeros_like(value)
        else:
            dist = torch.distributions.Categorical(logits=logits / max(1e-6, temperature))
            act  = dist.sample()
            logp = dist.log_prob(act)
        return act, logp, value, {"logits": logits}
