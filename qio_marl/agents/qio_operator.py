import torch
import torch.nn.functional as F

def _normalized_entropy(probs, eps=1e-8):
    H = -(probs * (probs.add(eps).log())).sum(dim=-1)          # natural log
    H_max = torch.log(torch.tensor(probs.size(-1), device=probs.device, dtype=probs.dtype))
    return (H / (H_max + eps)).clamp(0.0, 1.0)                 # in [0,1]

def apply_q_operator(logits, eta=0.1, min_entropy=0.05, temperature=1.0):
    """
    Quantum-inspired amplitude reweighting on logits.
    Steps:
      1) Convert logits -> probs
      2) Amplitude map: a_i = sqrt(p_i), amplify: a'_i ∝ a_i + eta*(a_i - mean(a))
      3) Back to probs: p'_i ∝ (a'_i)^2
      4) Optional temperature on final logits
      5) Enforce min entropy via convex combo with uniform if needed
    """
    probs = F.softmax(logits, dim=-1)
    a = torch.sqrt((probs + 1e-8))
    a_centered = a - a.mean(dim=-1, keepdim=True)
    a_prime = (a + eta * a_centered).clamp(min=1e-9)
    p_prime = (a_prime ** 2)
    p_prime = p_prime / p_prime.sum(dim=-1, keepdim=True)

    # Enforce minimum normalized entropy by mixing with uniform
    Hn = _normalized_entropy(p_prime)
    mask = (Hn < min_entropy).float().unsqueeze(-1)
    uniform = torch.full_like(p_prime, 1.0 / p_prime.size(-1))
    mix_weight = ((min_entropy - Hn).clamp(min=0.0) / (min_entropy + 1e-8)).unsqueeze(-1)
    p_tilde = (1 - mix_weight) * p_prime + mix_weight * uniform
    p_tilde = p_tilde / p_tilde.sum(dim=-1, keepdim=True)

    new_logits = (p_tilde + 1e-8).log() * temperature
    return new_logits, p_tilde
