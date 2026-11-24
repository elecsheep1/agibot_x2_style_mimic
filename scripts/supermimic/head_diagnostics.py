"""头部几何代理与当前／历史接触的只读诊断。"""
import torch

SENSORS = ('head_yaw_ground_contact', 'head_pitch_ground_contact')


def head_snapshot(env, task, following, valid):
    """自动重置会清空接触历史，因此终止步不用于接触归因。"""
    cfg = env.reward_manager.get_term_cfg('head_contact')
    threshold = float(cfg.params['force_threshold'])
    current, historical = [], []
    for name in SENSORS:
        data = env.scene.sensors[name].data
        now = data.force_matrix_w
        history = data.force_matrix_w_history
        now = now.torch if hasattr(now, 'torch') else now
        history = history.torch if hasattr(history, 'torch') else history
        if now.shape != (env.num_envs, 1, 1, 3) or history.shape[0] != env.num_envs or history.shape[2:] != (1, 1, 3):
            raise RuntimeError(f'头部地面接触传感器形状不兼容：{name}')
        current.append(now[:, 0, 0].norm(dim=-1))
        historical.append(history.norm(dim=-1).amax(dim=(1, 2, 3)))
    clearance = task.contact_geometry(following)[:, 4:6]
    proxy = task.head_contact_cost(following)
    current = torch.stack(current, -1)
    historical = torch.stack(historical, -1)
    if not all(bool(torch.isfinite(x).all()) for x in (clearance, proxy, current, historical)):
        raise RuntimeError('头部诊断数据包含 NaN/Inf')
    return dict(valid=valid.tolist(), clearance_m=clearance.tolist(), proxy_cost=proxy.tolist(),
                current_force_n=current.tolist(), historical_force_n=historical.tolist(),
                current_contact=(current > threshold).any(-1).tolist(),
                historical_contact=(historical > threshold).any(-1).tolist(),
                force_threshold_n=threshold)


def summarize_head_trace(trace):
    valid_count = current_count = history_count = history_only = missed_current = missed_history = proxy_only = 0
    for row in trace:
        for valid, current, history, proxy in zip(row['valid'], row['current_contact'], row['historical_contact'], row['proxy_cost']):
            if not valid:
                continue
            valid_count += 1
            current_count += int(current)
            history_count += int(history)
            history_only += int(history and not current)
            missed_current += int(current and proxy <= 1e-6)
            missed_history += int(history and proxy <= 1e-6)
            proxy_only += int(proxy > 1e-6 and not history)
    return dict(valid_samples=valid_count, current_contact_samples=current_count,
                historical_contact_samples=history_count, historical_only_samples=history_only,
                zero_proxy_on_current_contact=missed_current,
                zero_proxy_on_historical_contact=missed_history, proxy_without_historical_contact=proxy_only)


def physical_head_geometry(env, task, following):
    """同一组 USD 顶点分别使用 FK／仿真实际 link 姿态，隔离运动学误差。"""
    from x2_style_mimic.algorithms.supermimic.state import unpack_state
    from x2_style_mimic.utils.rotation import inverse_rotate_vector, normalize_quaternion, quaternion_angle_error
    geometry = task.contact_geometry
    robot = env.scene['robot']
    names = tuple(geometry.kinematics.body_names)[4:6]
    body_ids = [robot.body_names.index(name) for name in names]
    physical = robot.data.body_link_pose_w
    physical = physical.torch if hasattr(physical, 'torch') else physical
    physical = physical[:, body_ids].clone()
    physical[:, :, :3] -= env.scene.env_origins[:, None, :]
    fields = unpack_state(following, task.spec)
    fk, _ = geometry.kinematics(fields['root_pose'], fields['root_velocity'],
                                fields['joint_position'], fields['joint_velocity'])
    fk = fk[:, 4:6]
    world_up = torch.zeros_like(physical[..., :3])
    world_up[..., 2] = 1.
    up = inverse_rotate_vector(normalize_quaternion(physical[..., 3:7]), world_up)
    distances = []
    for i in range(2):
        points = getattr(geometry, f'vertices_{i + 4}').to(physical)
        distances.append(physical[:, i, 2] + (up[:, i] @ points.T).amin(-1))
    direct = torch.stack(distances, -1)
    delta = geometry(following)[:, 4:6] - direct
    return dict(physical_clearance_m=direct.tolist(), fk_minus_physical_clearance_m=delta.tolist(),
                fk_position_error_m=(fk[..., :3] - physical[..., :3]).norm(dim=-1).tolist(),
                fk_rotation_error_rad=quaternion_angle_error(fk[..., 3:7], physical[..., 3:7]).tolist())


def reset_evaluation_sensors(env, ids):
    """固定出生后的脚本补丁：当前 PhysX reset 未清空过滤接触历史。"""
    for sensor in env.scene.sensors.values():
        sensor.reset(ids)
        history = getattr(sensor.data, 'force_matrix_w_history', None)
        if history is not None:
            values = history.torch if hasattr(history, 'torch') else history
            values[ids] = 0.
        # 读取 data 可能触发旧物理帧的懒更新，再重置当前缓冲及时间戳。
        sensor.reset(ids)
