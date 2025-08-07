import torch
import torch.nn as nn
import torch.nn.functional as F

GATE_SCOPE = "l0_gates"
P_ZERO_LOSS_FACTOR = 0.05
P_ONE_LOSS_FACTOR = 0.95
LOSS_MODE = "lagrangian_prob"  # 'binary_weighted' # 'prob_target_MSE'
BINARY_MSE_TARGETS = [0.02, 0.98]
LAGRANGIAN_NONBINARY_MAX = 0.005
LAGRANGIAN_P_1_TARGET = 0.98

class LZeroGate(nn.Module):
    def __init__(self, shape, gamma=0.1, zeta=1.1, temperature=0.4):
        super().__init__()
        self.shape = shape
        self.gamma = gamma
        self.zeta = zeta
        self.temperature = temperature
        self.log_alpha = nn.Parameter(torch.zeros(*shape))
        self.beta = self.temperature

    def forward(self, x, gate_activated=True, test=False):
        if gate_activated:
            if not test:
                u = torch.rand(self.shape, device=x.device)
                s = torch.sigmoid(
                    (torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta
                )
                s_scaled = s * (self.zeta - self.gamma) + self.gamma
                z = torch.clamp(s_scaled, 0.0, 1.0)
            else:
                z = torch.round(
                    torch.clamp(
                        torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma,
                        0.0, 1.0
                    )
                )
            x = x * z
            # In PyTorch, add regularization loss manually to your total loss
            # self._last_reg_loss = self.regularization_loss()
            return x
        else:
            return x

    def get_expected_gate(self):
        z = torch.round(
            torch.clamp(
                torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma,
                0.0, 1.0
            )
        )
        return z

    def open_gate_probability(self):
        ones = torch.ones(self.shape, device=self.log_alpha.device)
        probs_one = self.stretched_concrete_cdf(ones)
        return probs_one


    def stretched_concrete_cdf(self, s_dash):
        s = (s_dash - self.gamma) / (self.zeta - self.gamma)
        cdf = torch.sigmoid(
            (torch.log(s) - torch.log(1 - s)) * self.beta - self.log_alpha
        )
        return cdf

    def regularization_loss(self):
        if LOSS_MODE == "binary_weighted":
            p_leq_0 = self.stretched_concrete_cdf(torch.tensor(0., device=self.log_alpha.device))
            p_geq_1 = 1 - self.stretched_concrete_cdf(torch.tensor(1., device=self.log_alpha.device))
            sum_binary = P_ZERO_LOSS_FACTOR * p_leq_0 + P_ONE_LOSS_FACTOR * p_geq_1
            binarization_loss = -sum_binary
            return torch.mean(binarization_loss)
        elif LOSS_MODE == "prob_target_MSE":
            p_leq_0 = self.stretched_concrete_cdf(torch.tensor(0., device=self.log_alpha.device))
            p_geq_1 = 1 - self.stretched_concrete_cdf(torch.tensor(1., device=self.log_alpha.device))
            loss_leq = (torch.mean(p_leq_0) - BINARY_MSE_TARGETS[0]) ** 2
            loss_geq = (torch.mean(p_geq_1) - BINARY_MSE_TARGETS[1]) ** 2
            binarization_loss = loss_leq + loss_geq
            return binarization_loss
        elif LOSS_MODE == "lagrangian_prob":
            p_geq_1 = torch.mean(1 - self.stretched_concrete_cdf(torch.tensor(1., device=self.log_alpha.device)))
            geq_1_loss = LAGRANGIAN_P_1_TARGET - p_geq_1
            p_leq_0 = torch.mean(self.stretched_concrete_cdf(torch.tensor(0., device=self.log_alpha.device)))
            p_non_binary = 1.0 - (p_geq_1 + p_leq_0)
            non_binary_loss = p_non_binary - LAGRANGIAN_NONBINARY_MAX
            loss_vector = torch.stack((geq_1_loss, non_binary_loss))
            return loss_vector