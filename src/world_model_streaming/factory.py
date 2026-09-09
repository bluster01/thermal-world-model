"""Matched initialization when callers supply deterministic, fresh CPU backends."""
import torch

from src.world_model_vnext.backends import PhysicsBackend
from .model import StreamingPhysicsWorldModel


def build_streaming_model(backend_factory, *, seed, read_mode, device='cpu', **model_kwargs):
    """Build fresh CPU parameters under an isolated RNG, then move to device.

    Every read mode constructs identical stored parameter shapes in the same
    order. With one seed and a deterministic backend factory, all initial
    state_dict tensors match. Modes differ in active computation and trainable
    parameter count. This is not a parameter-matched scientific study.

    The caller MUST construct a new backend with independent parameter/buffer
    storage on every invocation. The type/CPU checks below do not certify that
    responsibility: returning a captured instance or shared inner modules can
    make models share state, and .to(device) would mutate that shared backend.
    Only CPU Torch RNG is isolated here; custom factories must not depend on
    uncontrolled NumPy/Python RNG, external mutable state, or existing weights.
    """
    if not callable(backend_factory):
        raise ValueError('backend_factory must create a fresh PhysicsBackend')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise ValueError('seed must be a nonnegative signed 64-bit integer')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        backend = backend_factory()
        if not isinstance(backend, PhysicsBackend) or backend.state_loc.device.type != 'cpu':
            raise ValueError('backend_factory must return a CPU PhysicsBackend; freshness is a caller contract')
        model = StreamingPhysicsWorldModel(backend, read_mode=read_mode, **model_kwargs)
    return model.to(device)
