"""Read-only checks for production reset states, including dynamic recovery births."""
import numpy as np
import torch


def check_births(env, bank, recovery_bank=None, env_ids=None):
    robot = env.scene['robot']
    ids = (torch.arange(env.num_envs, device=env.device) if env_ids is None
           else torch.as_tensor(env_ids, dtype=torch.long, device=env.device))
    types = env._getup_birth_types[ids]
    rows = env._getup_bank_row_ids[ids]
    dynamic_rows = env._getup_recovery_row_ids[ids]
    assert bool(((types >= 0) & (types < 5)).all()), 'Invalid birth type'
    static = (types == 1) | (types == 3)
    dynamic = types == 4
    assert torch.equal(rows >= 0, static), 'Static bank row/type mismatch'
    assert torch.equal(dynamic_rows >= 0, dynamic), 'Dynamic bank row/type mismatch'
    assert bool((rows[~static] == -1).all())
    assert bool((dynamic_rows[~dynamic] == -1).all())
    errors = {}
    config = env.cfg.events.reset_from_reference.params
    dataset = env._getup_motion_dataset_cache[1]
    for kind in range(5):
        selected = types == kind
        if not bool(selected.any()):
            continue
        selected_ids = ids[selected]
        motion_ids = env._getup_reference_motion_ids[selected_ids]
        frame_ids = env._getup_reference_frame_ids[selected_ids]
        limits = robot.data.joint_pos_limits.torch[selected_ids]
        velocity_limits = robot.data.joint_vel_limits.torch[selected_ids]
        if kind in (0, 2):
            assert bool((frame_ids >= 0).all())
            if kind == 0:
                assert bool((frame_ids == 0).all()), 'Reference initial frame is not zero'
            expected = {
                name: torch.stack([dataset.clips[int(m)][name][int(f)] for m, f in zip(motion_ids, frame_ids)]).clone()
                for name in ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel')
            }
            expected['root_pose'][:, 2] += config['height_offset']
            if not config.get('use_reference_velocity', False):
                expected['root_velocity'].zero_(); expected['joint_vel'].zero_()
            elif config.get('zero_initial_frame_velocity', False):
                initial = frame_ids == 0
                expected['root_velocity'][initial] = 0.; expected['joint_vel'][initial] = 0.
            expected['joint_pos'] = expected['joint_pos'].clamp(limits[..., 0], limits[..., 1])
            expected['joint_vel'] = expected['joint_vel'].clamp(-velocity_limits, velocity_limits)
        else:
            source = recovery_bank if kind == 4 else bank
            assert source is not None, 'Dynamic bank needed for validation'
            index = (dynamic_rows if kind == 4 else rows)[selected].cpu().numpy()
            assert (index >= 0).all() and (index < len(source['root_pose'])).all()
            assert (source['split'][index] == 0).all(), 'Holdout sampled'
            if kind in (1, 3):
                assert (source['source_type'][index] == (0 if kind == 1 else 1)).all()
                assert np.array_equal(source['motion_id'][index], motion_ids.cpu().numpy())
            if kind in (3, 4):
                assert bool((motion_ids == -1).all())
            assert bool((frame_ids == -1).all())
            expected = {
                name: torch.as_tensor(source[name][index], device=env.device, dtype=torch.float32).clone()
                for name in ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel')
            }
            expected['root_pose'][:, :2] = 0.
            if kind != 4:
                expected['joint_vel'] = expected['joint_vel'].clamp(-velocity_limits, velocity_limits)
        pose = robot.data.root_link_pose_w.torch[selected_ids].clone()
        pose[:, :3] -= env.scene.env_origins[selected_ids]
        measured = {
            'position': (pose[:, :3] - expected['root_pose'][:, :3]).abs().max(),
            'orientation': torch.minimum(
                (pose[:, 3:] - expected['root_pose'][:, 3:]).abs().amax(-1),
                (pose[:, 3:] + expected['root_pose'][:, 3:]).abs().amax(-1)).max(),
            'root_velocity': (robot.data.root_link_vel_w.torch[selected_ids] - expected['root_velocity']).abs().max(),
            'joint_position': (robot.data.joint_pos.torch[selected_ids] - expected['joint_pos']).abs().max(),
            'joint_velocity': (robot.data.joint_vel.torch[selected_ids] - expected['joint_vel']).abs().max(),
        }
        for name, error in measured.items():
            value = float(error)
            errors[f'{kind}/{name}'] = value
            assert np.isfinite(value) and value < 1e-4, f'Birth {kind} {name} mismatch: {value}'
    if len(ids):
        assert bool((env.action_manager.action[ids] == 0).all()), 'Last action not reset'
        term = env.action_manager.get_term('joint_pos')
        q = robot.data.joint_pos.torch[ids][:, term._joint_ids]
        limits = robot.data.joint_pos_limits.torch[ids][:, term._joint_ids]
        target = q.clamp(limits[..., 0], limits[..., 1])
        assert torch.allclose(term._previous_target[ids], target, atol=1e-4), 'Action limiter retained old target'
        monitor = env.scene.sensors['self_contact']
        for value in (monitor.duration, monitor.strong_duration, monitor._strong_active, monitor.peak_force, monitor.pressure_cost_sum, monitor._updates):
            assert bool((value[ids] == 0).all()), 'Self-contact state not reset'
    return dict(counts={i: int((types == i).sum()) for i in range(5)}, max_errors=errors)
