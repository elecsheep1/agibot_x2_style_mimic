"""Collect train-family states before strong self-contact, then replay-check every candidate.

Does not modify source code or activate a training reset branch. Existing holdout is untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import load_bank, reset_from_bank_for_eval, sha256, TRAIN
from common.pre_entrapment import save


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay-bank', type=Path, default=None, help='Revalidate an existing dynamic candidate bank without collecting new trajectories')
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--max-parents', type=int, default=512)
    parser.add_argument('--target-families', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261110)
    parser.add_argument('--trigger-seconds', type=float, default=.10)
    parser.add_argument('--window-seconds', type=float, default=6.)
    parser.add_argument('--leads', type=float, nargs='+', default=[.30, .15, 0.])
    parser.add_argument('--max-penetration', type=float, default=.005)
    parser.add_argument('--max-linear', type=float, default=1.5)
    parser.add_argument('--max-angular', type=float, default=6.)
    parser.add_argument('--replay-seconds', type=float, default=.20)
    required = [a for a in parser._actions if a.required]
    for a in required: a.required = False
    try: add_launcher_args(parser)
    finally:
        for a in required: a.required = True
    args, rest = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + rest
    if min(args.num_envs, args.max_parents, args.target_families) < 1:
        parser.error('Counts must be positive')
    if not all(np.isfinite(x) and x > 0 for x in (args.trigger_seconds, args.window_seconds, args.max_penetration, args.max_linear, args.max_angular, args.replay_seconds)) or not args.leads or not all(np.isfinite(x) and x >= 0 for x in args.leads):
        parser.error('Invalid thresholds')
    report_path = args.output.with_suffix('.report.json')
    if args.output.exists() or report_path.exists(): parser.error('Output already exists')
    if not args.checkpoint.is_file(): parser.error('Checkpoint missing')
    parent_hash = sha256(args.bank)
    parent = load_bank(args.bank)
    if parent['metadata'].get('collection_report', {}).get('training_ready') is not True:
        parser.error('Parent bank is not training ready')
    rows = np.flatnonzero(parent['split'] == TRAIN)
    np.random.default_rng(args.seed).shuffle(rows)
    rows = rows[:args.max_parents]
    if not len(rows): parser.error('No train rows')
    import x2_style_mimic.tasks
    cfg, agent = resolve_task_config('X2-Prone-MultiMotion-v0', 'rsl_rl_cfg_entry_point')
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from importlib.metadata import version
    agent = handle_deprecated_rsl_rl_cfg(agent, version('rsl-rl-lib'))
    cfg.scene.num_envs = min(args.num_envs, len(rows)); cfg.seed = args.seed
    if args.device is not None: cfg.sim.device = args.device; agent.device = args.device
    cfg.assist_curriculum_enabled = False
    cfg.assist_initial_stage = len(cfg.assist_force_levels) - 1
    cfg.assist_zero_probability = 1.
    params = dict(cfg.events.reset_from_reference.params)
    params.update(bank_path=str(args.bank.resolve()), bank_split='train', bank_source='all', bank_rows=rows.tolist(), bank_center_xy=True)
    cfg.events.reset_from_reference.func = reset_from_bank_for_eval
    cfg.events.reset_from_reference.params = params
    candidates = []; accepted = []; replay_audit = []; reasons = Counter(); families = set(); seen = set()
    with launch_simulation(cfg, args):
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env import X2MultiMotionGetUpEnv
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from checks.build_fallen_state_bank import Contacts
        from pxr import Usd, UsdPhysics
        from isaaclab_physx.physics import PhysxManager
        torch.manual_seed(args.seed)
        env = X2MultiMotionGetUpEnv(cfg=cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        try:
            runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=agent.device)
            runner.load(str(args.checkpoint.resolve()), load_cfg={'actor': True, 'critic': True, 'optimizer': False, 'iteration': False, 'rnd': False, 'amp': False, 'assist_curriculum': False})
            policy = runner.get_inference_policy(device=env.device)
            robot = env.scene['robot']; monitor = env.scene.sensors['self_contact']
            # Use exactly the sensor's articulation subtree, not the visual Geometry hierarchy.
            root_path = monitor.cfg.prim_path.replace('{ENV_REGEX_NS}', '/World/envs/env_0').replace('env_.*', 'env_0')
            paths = sorted(str(prim.GetPath()) for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath(root_path), Usd.TraverseInstanceProxies()) if prim.HasAPI(UsdPhysics.RigidBodyAPI))
            if [p.rsplit('/', 1)[-1] for p in paths] != monitor.body_names:
                raise RuntimeError('Contact source mapping differs from production sensor')
            class CollectionContacts(Contacts):
                def metrics(self):
                    ground, gd = self.read(self.ground_view, 1)
                    own, sd = self.read(self.self_view, self.b - 1)
                    return ground, own, gd, sd
            contacts = CollectionContacts.__new__(CollectionContacts)
            contacts.dt, contacts.n, contacts.b, contacts.device = float(cfg.sim.dt), env.num_envs, len(paths), env.device
            contacts.self_view = monitor._view
            sources = [p.replace('/env_0/', f'/env_{e}/') for e in range(env.num_envs) for p in paths]
            contacts.ground_view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(sources, filter_patterns=[['/World/ground/GroundPlane/CollisionPlane'] for _ in sources], max_contact_data_count=max(262144, env.num_envs * len(paths) * 8))
            if contacts.ground_view.sensor_count != len(sources) or contacts.ground_view.filter_count != 1:
                raise RuntimeError('Invalid ground contact layout')
            # Replay impact screening must see real physics substeps, including terminal ones.
            replay_probe = dict(active=False)
            original_update = env.scene.update
            def observed_update(*update_args, **update_kwargs):
                result = original_update(*update_args, **update_kwargs)
                if replay_probe['active']:
                    n = replay_probe['count']
                    v = robot.data.root_link_vel_w.torch[:n]
                    metric = contacts.metrics()
                    values = torch.stack((v[:,:3].norm(dim=-1), v[:,3:].norm(dim=-1), metric[0][:n], metric[1][:n]), dim=-1)
                    depth = torch.stack((metric[2][:n], metric[3][:n]), dim=-1)
                    replay_probe['peak'] = torch.maximum(replay_probe['peak'], values)
                    replay_probe['depth'] = torch.minimum(replay_probe['depth'], depth)
                    replay_probe['valid'] &= torch.isfinite(v).all(-1) & torch.isfinite(robot.data.joint_pos.torch[:n]).all(-1) & torch.isfinite(robot.data.joint_vel.torch[:n]).all(-1)
                    replay_probe['valid'] &= (values < values.new_tensor([3., 12., 6000., 3000.])).all(-1) & (depth >= -.015).all(-1)
                    replay_probe['updates'] += 1
                return result
            env.scene.update = observed_update
            rings = [deque(maxlen=int(np.ceil(max(args.leads) / env.step_dt)) + 4) for _ in range(env.num_envs)]
            serials = [-1] * env.num_envs
            obs = wrapped.get_observations()
            if args.replay_bank is not None:
                from common.pre_entrapment import load, FIELDS
                prior = load(args.replay_bank, args.bank, robot.joint_names, require_reset_protocol=False)
                if prior['metadata'].get('checkpoint_sha256') != sha256(args.checkpoint):
                    raise ValueError('Revalidation requires the collection policy checkpoint')
                candidates = [{key: prior[key][i].copy() for key in FIELDS} for i in range(len(prior['root_pose']))]
                families = {int(row['family_id']) for row in candidates}
                seen = {int(row['parent_row']) for row in candidates}

            def state(e, t):
                pose = robot.data.root_link_pose_w.torch[e].clone()
                pose[:3] -= env.scene.env_origins[e]; pose[:2] = 0.
                return dict(root_pose=pose.cpu().numpy(), root_velocity=robot.data.root_link_vel_w.torch[e].cpu().numpy().copy(),
                            joint_pos=robot.data.joint_pos.torch[e].cpu().numpy().copy(), joint_vel=robot.data.joint_vel.torch[e].cpu().numpy().copy(), time_s=np.asarray(t))

            with torch.inference_mode():
                # At most ceil(parent_count/env_count) full episodes; scanning stops per parent at window_seconds.
                max_steps = (int(np.ceil(len(rows) / env.num_envs)) + 1) * int(np.ceil(cfg.episode_length_s / env.step_dt))
                for step in range(0 if args.replay_bank is not None else max_steps):
                    index = env._fallen_eval_index.clone(); serial = env._fallen_eval_serial.clone()
                    obs, _, dones, extras = wrapped.step(policy(obs))
                    if bool((extras['getup_progress']['assist_applied_force_n'].abs() > 1e-6).any()):
                        raise RuntimeError('Nonzero assist')
                    metrics = contacts.metrics()
                    limits = robot.data.joint_pos_limits.torch
                    velocity_limits = robot.data.joint_vel_limits.torch
                    for e in range(env.num_envs):
                        row = int(index[e]); number = int(serial[e])
                        if number >= len(rows) or row in seen: continue
                        if serials[e] != number:
                            rings[e].clear(); serials[e] = number
                        t = float(env.episode_length_buf[e]) * env.step_dt
                        if bool(dones[e]) or t > args.window_seconds:
                            seen.add(row); continue
                        item = state(e, t)
                        item.update(ground_min_separation_m=np.asarray(float(metrics[2][e])), self_min_separation_m=np.asarray(float(metrics[3][e])))
                        q = robot.data.joint_pos.torch[e]; qd = robot.data.joint_vel.torch[e]
                        item['valid'] = all(np.isfinite(item[k]).all() for k in ('root_pose', 'root_velocity', 'joint_pos', 'joint_vel')) and float((q-q.clamp(limits[e,:,0], limits[e,:,1])).abs().max()) <= .003 and bool((qd.abs() <= velocity_limits[e] + 1e-4).all()) and float(metrics[2][e]) >= -args.max_penetration and float(metrics[3][e]) >= -args.max_penetration and np.linalg.norm(item['root_velocity'][:3]) <= args.max_linear and np.linalg.norm(item['root_velocity'][3:]) <= args.max_angular
                        rings[e].append(item)
                        duration, pair = monitor.strong_duration[e].max(dim=-1)
                        if float(duration) + 1e-7 < args.trigger_seconds: continue
                        seen.add(row)
                        family = int(parent['family_id'][row])
                        if family in families: continue
                        saved = 0; used_times = set()
                        for lead in args.leads:
                            eligible = [x for x in rings[e] if float(x['time_s']) <= t - lead + 1e-6]
                            if not eligible: reasons['insufficient_history'] += 1; continue
                            entry = eligible[-1]
                            if not entry['valid']: reasons['invalid_candidate'] += 1; continue
                            if float(entry['time_s']) in used_times: continue
                            used_times.add(float(entry['time_s']))
                            entry = {k: v.copy() for k, v in entry.items() if k != 'valid'}
                            entry.update(parent_row=np.asarray(row), family_id=np.asarray(family), split=np.asarray(TRAIN), lead_seconds=np.asarray(t-float(entry['time_s'])), trigger_pair=np.asarray([int(monitor._i[pair]), int(monitor._j[pair])]))
                            candidates.append(entry); saved += 1
                        if saved: families.add(family)
                    policy.reset(dones)
                    if step % 100 == 0:
                        print(f'[pre-entrapment] scanned={len(seen)}/{len(rows)} families={len(families)} candidates={len(candidates)}', flush=True)
                    if len(families) >= args.target_families or len(seen) >= len(rows): break
                    # Once every active parent has been scanned, skip the remaining standing tail.
                    active_rows = [int(index[e]) for e in range(env.num_envs) if int(serial[e]) < len(rows)]
                    if active_rows and all(row in seen for row in active_rows):
                        obs, _ = wrapped.reset()
                        policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

                # Reproduce the training reset: clear caches/actions, restore real velocities and pose,
                # start fresh observations. No AMP reference history or previous action is injected.
                replay_steps = max(1, int(np.ceil(args.replay_seconds / env.step_dt)))
                for begin in range(0, len(candidates), env.num_envs):
                    batch = candidates[begin:begin+env.num_envs]; count = len(batch)
                    wrapped.reset()
                    ids = torch.arange(count, device=env.device)
                    def tensor(key): return torch.as_tensor(np.stack([r[key] for r in batch]), device=env.device, dtype=torch.float32)
                    pose = tensor('root_pose'); pose[:,:3] += env.scene.env_origins[ids]
                    velocity = tensor('root_velocity'); q = tensor('joint_pos'); qd = tensor('joint_vel')
                    _write_reference_state(env, ids, robot, pose, velocity, q, qd)
                    errors = [(robot.data.root_link_pose_w.torch[ids]-pose).abs().max(), (robot.data.root_link_vel_w.torch[ids]-velocity).abs().max(), (robot.data.joint_pos.torch[ids]-q).abs().max(), (robot.data.joint_vel.torch[ids]-qd).abs().max()]
                    if max(float(x) for x in errors) > 1e-4: raise RuntimeError('Restored state differs from saved state')
                    # The saved state overwrites the ordinary bank reset. Reinitialize the action
                    # limiter from this pose and clear observation history again, just as production
                    # reset does after its reset event. Never retain the parent's target/history.
                    env.action_manager.reset(ids)
                    env.observation_manager.reset(ids)
                    env.scene.write_data_to_sim(); env.sim.forward()
                    obs = env.observation_manager.compute()
                    policy.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
                    replay_probe.update(active=True, count=count, valid=torch.ones(count, dtype=torch.bool, device=env.device), peak=torch.zeros((count, 4), device=env.device), depth=torch.zeros((count, 2), device=env.device), updates=0)
                    try:
                        for _ in range(replay_steps):
                            obs, _, dones, extras = wrapped.step(policy(obs))
                            if bool((extras['getup_progress']['assist_applied_force_n'].abs() > 1e-6).any()): raise RuntimeError('Nonzero replay assist')
                            replay_probe['valid'] &= ~dones[:count].bool()
                            policy.reset(dones)
                    finally:
                        replay_probe['active'] = False
                    if replay_probe['updates'] != replay_steps * cfg.decimation:
                        raise RuntimeError('Replay physics sampling count mismatch')
                    valid, peak, deepest = replay_probe['valid'], replay_probe['peak'], replay_probe['depth']
                    for i, entry in enumerate(batch):
                        replay_audit.append(dict(parent_row=int(entry['parent_row']), time_s=float(entry['time_s']), accepted=bool(valid[i]), physics_peak_linear_speed=float(peak[i,0]), physics_peak_angular_speed=float(peak[i,1]), physics_peak_ground_force_n=float(peak[i,2]), physics_peak_self_force_n=float(peak[i,3]), physics_min_ground_separation_m=float(deepest[i,0]), physics_min_self_separation_m=float(deepest[i,1])))
                        if bool(valid[i]): accepted.append(entry)
                        else: reasons['replay_invalid'] += 1
            if sha256(args.bank) != parent_hash: raise RuntimeError('Parent bank changed during collection')
            meta = dict(parent_bank=str(args.bank.resolve()), parent_sha256=parent_hash, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint), asset_hashes=parent['metadata']['asset_hashes'], seed=args.seed, quat_order='xyzw', root_frame='link_origin', position_frame='environment_local', velocity_frame='world_axes', replay_validated=True, reset_protocol='clear_action_and_observation_history_after_state_write_v1', collector_sha256=sha256(__file__), replay_audit=replay_audit, replay_physics_dt=float(cfg.sim.dt), body_names=monitor.body_names, thresholds=vars(args).copy())
            meta['thresholds'] = {k: str(v) if isinstance(v, Path) else v for k, v in meta['thresholds'].items()}
            report = dict(scanned=len(seen), target_parents=len(rows), trigger_families=len(families), candidates=len(candidates), accepted=len(accepted), accepted_families=len({int(r['family_id']) for r in accepted}), rejected=dict(reasons), replay_audit=replay_audit, replay_physics_dt=float(cfg.sim.dt), zero_assist_verified=True, holdout_unchanged=True, ready_for_source_integration=len({int(r['family_id']) for r in accepted}) >= 32)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2))
            if accepted: save(args.output, accepted, robot.joint_names, meta)
            print(json.dumps(report), flush=True)
        finally:
            wrapped.close()


if __name__ == '__main__': main()
