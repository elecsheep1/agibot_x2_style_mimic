"""Validate a supervised world-model checkpoint before online initialization."""
import math

import torch


def read_world_checkpoint(path, expected_metadata, limiter_values, context_dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {'model', 'metadata', 'hidden_dims', 'limiter', 'context_dim'}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError('Expected a supervised world_model.pt checkpoint')
    metadata = checkpoint['metadata']
    for name, expected in expected_metadata.items():
        actual = metadata.get(name)
        if name in ('step_dt', 'physics_dt', 'decimation'):
            compatible = actual is not None and math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=1e-9)
        else:
            compatible = actual == expected
        if not compatible:
            raise ValueError(f'World checkpoint incompatible: {name}')
    if checkpoint['context_dim'] != context_dim:
        raise ValueError('World checkpoint incompatible: context_dim')
    dimensions = checkpoint['hidden_dims']
    if not dimensions or any(not isinstance(v, int) or v < 1 for v in dimensions):
        raise ValueError('Invalid hidden_dims')
    for name, current in limiter_values.items():
        saved = torch.as_tensor(checkpoint['limiter'][name], device=current.device, dtype=current.dtype)
        torch.testing.assert_close(saved, current, atol=1e-6, rtol=1e-6, msg=f'Limiter mismatch: {name}')
        # The duplicated calibration and the actual state_dict must also agree.
        torch.testing.assert_close(checkpoint['model'][f'limiter.{name}'], current,
                                   atol=1e-6, rtol=1e-6, msg=f'State-dict limiter mismatch: {name}')
    if not bool(checkpoint['model'].get('stats_ready', False)):
        raise ValueError('Checkpoint has no fitted input statistics')
    for name, value in checkpoint['model'].items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite world checkpoint tensor: {name}')
    return checkpoint
