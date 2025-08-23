import torch
from qio_marl.agents.qio_operator import apply_q_operator

def test_entropy_floor():
    logits = torch.zeros(4, 5)  # uniform
    new_logits, p = apply_q_operator(logits, eta=0.5, min_entropy=0.5)
    H = -(p * (p+1e-8).log()).sum(dim=-1)
    Hn = H / torch.log(torch.tensor(p.size(-1), dtype=H.dtype))
    assert (Hn >= 0.49).all(), "Entropy floor not respected (allow small tolerance)"

def test_prob_normalization():
    logits = torch.randn(3, 7)
    new_logits, p = apply_q_operator(logits, eta=0.2, min_entropy=0.1)
    s = p.sum(dim=-1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-6)
