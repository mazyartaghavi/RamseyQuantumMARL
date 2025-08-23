import math

class EntropyController:
    """
    Controls alpha_t, the entropy regularization weight.

    Modes:
      - "exp": alpha_t = max(alpha_min, alpha0 * exp(-lambda * t))
      - "quantum_inspired": alpha_t = alpha0 * exp(-lambda * sum_{k<=t} sqrt(1 - H_k^2))
        where H_k is the normalized policy entropy in [0,1].
    """
    def __init__(self, cfg):
        self.mode = cfg.get("type", "exp")
        self.alpha0 = float(cfg.get("alpha0", 0.1))
        self.alpha_min = float(cfg.get("alpha_min", 0.01))
        self.lmbda = float(cfg.get("lambda", 5e-4))
        self.t = 0
        self.cum = 0.0
        self.alpha = self.alpha0

    def update(self, normalized_entropy):
        self.t += 1
        if self.mode == "exp":
            self.alpha = max(self.alpha_min, self.alpha0 * math.exp(-self.lmbda * self.t))
        else:
            # quantum-inspired
            h = float(max(0.0, min(1.0, normalized_entropy)))
            term = math.sqrt(max(0.0, 1.0 - h**2))
            self.cum += term
            self.alpha = max(self.alpha_min, self.alpha0 * math.exp(-self.lmbda * self.cum))
        return self.alpha
