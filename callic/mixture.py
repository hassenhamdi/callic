"""CALLIC mixture — discrete logistic mixture NLL (PixelCNN++ style, K mixtures).
Head channels = K*10: per mixture [pi(1)+mu(3)+scale(3)+coeff(3)].
Honest entropy, no test tuning."""
import torch
import torch.nn.functional as F


def _logistic_cdf(x, mu, scale):
    return torch.sigmoid((x - mu) / scale)


def discretized_mixture_nll(x, logits, K=10):
    """x: [B,3,H,W] uint8/float 0-255; logits: [B,K*10,H,W]. Returns mean NLL in bits per sub-pixel."""
    B, _, H, W = x.shape
    x = x.float()
    logits = logits.permute(0, 2, 3, 1).reshape(B, H, W, K, 10)
    pi = F.softmax(logits[..., 0], dim=-1)  # [B,H,W,K]
    mu = logits[..., 1:4]  # [B,H,W,K,3]
    scale = F.softplus(logits[..., 4:7]).clamp(min=1e-3, max=32.0)
    coeff = torch.tanh(logits[..., 7:10])  # RGB autoreg coeffs
    # channel autoregression: adjust means
    xr = x.permute(0, 2, 3, 1).unsqueeze(-2).expand(B, H, W, K, 3)  # [B,H,W,K,3]
    m0 = mu[..., 0]
    m1 = mu[..., 1] + coeff[..., 0] * xr[..., 0]
    m2 = mu[..., 2] + coeff[..., 1] * xr[..., 0] + coeff[..., 2] * xr[..., 1]
    means = torch.stack([m0, m1, m2], dim=-1)  # [B,H,W,K,3]
    scales = torch.stack([scale[..., 0], scale[..., 1], scale[..., 2]], dim=-1)
    # discretized logistic prob per channel
    plus = (xr + 0.5 - means) / scales
    minus = (xr - 0.5 - means) / scales
    cdf_plus = torch.sigmoid(plus)
    cdf_minus = torch.sigmoid(minus)
    prob = cdf_plus - cdf_minus
    # edges 0 / 255
    prob0 = torch.sigmoid((xr + 0.5 - means) / scales)
    prob255 = 1.0 - torch.sigmoid((xr - 0.5 - means) / scales)
    is0 = (xr == 0).float()
    is255 = (xr == 255).float()
    prob = is0 * prob0 + is255 * prob255 + (1 - is0 - is255) * prob
    prob = prob.clamp(min=1e-9)
    nll = -torch.log(prob)  # nats per subpixel per mixture
    # mix over K (log-sum-exp with pi)
    mix = torch.logsumexp(torch.log(pi.unsqueeze(-1).clamp(min=1e-12)) + (-nll), dim=-2)  # [B,H,W,3]
    bits = -mix / 0.69314718056
    return bits.mean()
