"""Separate reference refresh from continuous action memory; no new backbone."""
import torch

from .models import Base


class ProtectedReference(Base):
    """Reference receives a single hold plan; candidate controls only enter R4.

    block: nominal reference refreshes every32, R4 response never resets.
    native: both components run uninterrupted. Neither sees future true T.
    """
    native_protocol = 'uninterrupted nominal reference + uninterrupted R4 carrier'
    block_protocol = 'nominal reference block32 + uninterrupted R4 carrier'

    def __init__(self, predictor, response):
        super().__init__(predictor.mean, predictor.scale)
        self.predictor, self.response = predictor, response

    def forward(self, history, actions, boundaries):
        nominal = history[:, -1:, 5:7].expand_as(actions)
        reference = self.predictor(history, nominal, boundaries)
        return reference + self.response.core(history, actions, boundaries)['action_response_C']

    def forecast_plan(self, history, actions, boundaries, *, mode):
        from .evaluation import forecast
        nominal = history[:, -1:, 5:7].expand_as(actions)
        reference = forecast(self.predictor, history, nominal, boundaries, mode=mode)
        response = self.response.core(history, actions[:, :-1], boundaries[:, :-1])['action_response_C']
        return reference + response


def response_matching_loss(model, teacher, history, boundaries, update):
    """Small training-only perturbations; R4 is a structural teacher, not truth.

    Cycle 2 valves x 2 signs x 3 onsets x 3 shapes independently.
    Scale response error by 10% of training temperature std (weight 1).
    """
    h, d = history[:8], boundaries[:8]
    horizon = d.shape[1]
    valve, sign = update % 2, (1 if (update // 2) % 2 == 0 else -1)
    onset = (0, 8, 24)[(update // 4) % 3]
    shape = (update // 12) % 3
    x = torch.arange(horizon, device=h.device) - onset
    profile = (x >= 0).to(h.dtype)
    if shape == 1: profile = profile * (x / 8).clamp(0, 1)
    if shape == 2: profile = profile * (x < 8)
    nominal = h[:, -1:, 5:7].expand(-1, horizon, -1)
    candidate = nominal.clone()
    candidate[:, :, valve] += sign * .03 * profile
    valid = ((candidate >= 0) & (candidate <= 1)).all((1, 2))
    if not valid.any():
        return h.new_zeros(())
    h, d, nominal, candidate = h[valid], d[valid], nominal[valid], candidate[valid]
    with torch.no_grad():
        target = teacher.core(h, candidate, d)['action_response_C']
    # Shared paired batch and same shape as evaluation; training stays float32.
    p = model(torch.cat((h, h)), torch.cat((candidate, nominal)), torch.cat((d, d)))
    delta = p[:len(h)] - p[len(h):]
    weights = h.new_tensor([.125, .125, .125, .125, .5])
    return (((delta-target) / (.1 * model.scale[:5])).square() * weights).sum(-1).mean()
