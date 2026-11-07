"""Collect and replay-validate X2 fallen states in isolated Isaac/PhysX scenes.

No policy training, no edits to training configuration, no dependency on getup_births.py.
"""
from __future__ import annotations
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.fallen_bank import (asset_hashes, dedup_key, family_split, save_bank,
                                sha256, summary, load_bank, SOURCE_NEAR, SOURCE_FALL)


def rotate(q, v):
    t = 2 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:4] * t + torch.cross(q[..., :3], t, dim=-1)


def yaw_rotate(pose, angle):
    """Rotate root orientation about world Z; root XY is centered separately."""
    q = pose[:, 3:7].clone(); s = torch.sin(angle / 2); c = torch.cos(angle / 2)
    pose[:, 3] = c*q[:, 0] - s*q[:, 1]
    pose[:, 4] = c*q[:, 1] + s*q[:, 0]
    pose[:, 5] = c*q[:, 2] + s*q[:, 3]
    pose[:, 6] = c*q[:, 3] - s*q[:, 2]


class Geometry:
    """Conservative floor clearance from authored collision mesh vertices, not ankle origins."""
    def __init__(self, stage, body_names, device):
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        cache = UsdGeom.XformCache(); values, indexes = [], []
        paths = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if '/env_0/Robot/' in path and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                paths[prim.GetName()] = path
        if set(body_names) - set(paths):
            raise ValueError('Cannot resolve all articulation rigid-body paths')
        self.paths = [paths[n] for n in body_names]
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if '/env_0/Robot/' not in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            parent = prim
            while parent and not parent.HasAPI(UsdPhysics.RigidBodyAPI):
                parent = parent.GetParent()
            if not parent or parent.GetName() not in body_names:
                raise ValueError(f'Collision shape without known body: {path}')
            if prim.IsA(UsdGeom.Mesh):
                points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            else:
                extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
                if extent is None:
                    raise ValueError(f'Unsupported shape without extent: {path}')
                points = [Gf.Vec3f(*v) for v in itertools.product(*zip(extent[0], extent[1]))]
            if points is None or not len(points):
                raise ValueError(f'Empty collision geometry: {path}')
            matrix = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(parent).GetInverse()
            # Gf uses row-vector transform composition.
            local = [matrix.Transform(Gf.Vec3d(*p)) for p in points]
            values.extend([list(p) for p in local]); indexes.extend([body_names.index(parent.GetName())]*len(local))
        if not values:
            raise ValueError('No collision geometry found')
        self.points = torch.tensor(values, dtype=torch.float32, device=device)
        self.indexes = torch.tensor(indexes, device=device)

    def minimum_z(self, robot):
        pose = robot.data.body_link_pose_w.torch
        minimum = torch.full((len(pose),), float('inf'), device=pose.device)
        for begin in range(0, len(self.points), 4096):
            points = self.points[begin:begin+4096]
            body = pose[:, self.indexes[begin:begin+4096]]
            world = rotate(body[..., 3:7], points[None].expand(len(pose), -1, -1)) + body[..., :3]
            minimum = torch.minimum(minimum, world[..., 2].amin(1))
        return minimum


class Contacts:
    def __init__(self, geometry, count, dt, device, capacity):
        from isaaclab_physx.physics import PhysxManager
        self.dt, self.n, self.b, self.device = dt, count, len(geometry.paths), device
        sources = [p.replace('/env_0/', f'/env_{e}/') for e in range(count) for p in geometry.paths]
        filters = [[p.replace('/env_0/', f'/env_{e}/') for p in geometry.paths]
                   for e in range(count) for _ in geometry.paths]
        physics = PhysxManager.get_physics_sim_view()
        self.self_view = physics.create_rigid_contact_view(sources, filter_patterns=filters,
                                                         max_contact_data_count=capacity)
        self.ground_view = physics.create_rigid_contact_view(sources,
            filter_patterns=[['/World/ground/GroundPlane/CollisionPlane'] for _ in sources],
            max_contact_data_count=capacity)
        for view, f in [(self.self_view, self.b), (self.ground_view, 1)]:
            if view.sensor_count != count*self.b or view.filter_count != f:
                raise RuntimeError('Unexpected contact matrix layout')

    def read(self, view, filters):
        import warp as wp
        forces = wp.to_torch(view.get_contact_force_matrix(self.dt)).reshape(self.n, self.b, filters, 3)
        force = forces.norm(dim=-1).flatten(1).amax(1)
        if filters == 1:
            force = forces[..., 2].clamp_min(0).sum((1, 2))
        _, _, _, separations, counts, starts = view.get_contact_data(self.dt)
        counts = wp.to_torch(counts).long().flatten()
        starts = wp.to_torch(starts).long().flatten()
        separation = wp.to_torch(separations).flatten()
        total = int(counts.sum())
        if total >= len(separation):
            raise RuntimeError('Contact buffer reached capacity; increase --contact-capacity')
        minimum = torch.zeros(self.n, device=self.device)
        if total:
            pairs = torch.repeat_interleave(torch.arange(len(counts), device=self.device), counts)
            offsets = counts.cumsum(0)-counts
            indices = starts[pairs] + torch.arange(total, device=self.device)-offsets[pairs]
            if (indices < 0).any() or (indices >= len(separation)).any():
                raise RuntimeError('Invalid contact-data offsets')
            minimum.scatter_reduce_(0, pairs//(self.b*filters), separation[indices], reduce='amin')
        if not torch.isfinite(force).all() or not torch.isfinite(minimum).all():
            raise RuntimeError('Nonfinite contact data')
        return force, minimum

    def metrics(self):
        ground, ground_depth = self.read(self.ground_view, 1)
        self_force, self_depth = self.read(self.self_view, self.b)
        return ground, self_force, ground_depth, self_depth


def main():
    from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--near-count', type=int, default=1024)
    parser.add_argument('--fall-count', type=int, default=1024)
    parser.add_argument('--drop-fraction', type=float, default=0., help='Fraction of fall batches using independent side-oriented passive placement')
    parser.add_argument('--drop-joint-noise', type=float, default=.5)
    parser.add_argument('--min-side-count', type=int, default=0)
    parser.add_argument('--exclude-bank', type=Path, default=None)
    parser.add_argument('--max-batches' , type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--holdout-fraction', type=float, default=.2)
    parser.add_argument('--joint-noise', type=float, default=.03)
    parser.add_argument('--settle-seconds', type=float, default=8.)
    parser.add_argument('--stable-seconds', type=float, default=.5)
    parser.add_argument('--replay-seconds', type=float, default=1.)
    parser.add_argument('--push-force-min', type=float, default=150.)
    parser.add_argument('--push-force-max', type=float, default=450.)
    parser.add_argument('--push-seconds', type=float, default=.25)
    parser.add_argument('--max-fallen-height', type=float, default=.35)
    parser.add_argument('--max-root-up', type=float, default=.7)
    parser.add_argument('--max-linear-speed', type=float, default=.10)
    parser.add_argument('--max-angular-speed', type=float, default=.30)
    parser.add_argument('--max-joint-speed-rms', type=float, default=.30)
    parser.add_argument('--max-penetration', type=float, default=.015)
    parser.add_argument('--max-self-force', type=float, default=150.)
    parser.add_argument('--max-replay-linear', type=float, default=3.)
    parser.add_argument('--max-replay-angular', type=float, default=12.)
    parser.add_argument('--max-replay-ground-force', type=float, default=6000.)
    parser.add_argument('--contact-capacity', type=int, default=262144)
    # Launcher examines argv when registering; temporarily suppress required actions.
    required = [x for x in parser._actions if x.required]
    for x in required: x.required = False
    try: add_launcher_args(parser)
    finally:
        for x in required: x.required = True
    args, rest = setup_preset_cli(parser); sys.argv = [sys.argv[0]] + rest
    if args.output.exists() or args.output.with_suffix('.report.json').exists():
        parser.error('Output/report exists; choose a new output path')
    positive = ['settle_seconds','stable_seconds','replay_seconds','push_seconds','max_fallen_height',
                'max_linear_speed','max_angular_speed','max_joint_speed_rms','max_penetration',
                'max_self_force','max_replay_linear','max_replay_angular','max_replay_ground_force']
    if any(not np.isfinite(getattr(args,k)) or getattr(args,k)<=0 for k in positive):
        parser.error('Durations and thresholds must be finite and positive')
    if (args.num_envs < 1 or args.max_batches < 1 or min(args.near_count,args.fall_count)<0
        or args.near_count+args.fall_count==0 or not 0<args.holdout_fraction<1
        or not 0<=args.joint_noise<=.2 or not 0<=args.push_force_min<=args.push_force_max<float('inf')):
        parser.error('Invalid counts, perturbation, split or force range')
    if not 0 <= args.drop_fraction <= 1 or not 0 <= args.drop_joint_noise <= 1 or not 0 <= args.min_side_count <= args.fall_count:
        parser.error('Invalid drop coverage settings')
    import x2_style_mimic.tasks
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_style_mimic_env_cfg import POLICY_JOINT_NAMES
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.motion_dataset import MotionDataset
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.getup import _write_reference_state
    from x2_style_mimic.tasks.manager_based.x2_style_mimic.x2_getup_env_cfg import MOTION_MANIFEST
    from x2_style_mimic.assets.x2_ultra_cfg import X2_ULTRA_USD
    cfg, _ = resolve_task_config('X2-Prone-MultiMotion-v0', '')
    if args.device is not None: cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    cfg.scene.env_spacing = 4.
    # Retain only the generic ground/rigid-body reporter sensor for asset setup.
    from isaaclab.sensors import ContactSensorCfg
    for key, value in vars(cfg.scene).items():
        if isinstance(value, ContactSensorCfg) and key != 'contact_forces': setattr(cfg.scene,key,None)
    rows, rejection, seen = [], Counter(), set()
    accepted = {SOURCE_NEAR: 0, SOURCE_FALL: 0}
    targets = {SOURCE_NEAR: args.near_count, SOURCE_FALL: args.fall_count}
    attempts = {SOURCE_NEAR: 0, SOURCE_FALL: 0}
    report = {'settings': vars(args).copy(), 'complete': False}
    report['settings']['output'] = str(args.output)
    if args.exclude_bank is not None: report['settings']['exclude_bank'] = str(args.exclude_bank)
    with launch_simulation(cfg,args):
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        sim = sim_utils.SimulationContext(cfg.sim)
        scene = InteractiveScene(cfg.scene); sim.reset()
        robot = scene['robot']; device = sim.device; dt = float(cfg.sim.dt); n = args.num_envs
        rng = torch.Generator(device=device).manual_seed(args.seed)
        geometry = Geometry(sim.stage, list(robot.body_names), device)
        contact = Contacts(geometry,n,dt,device,args.contact_capacity)
        dataset = MotionDataset(str(MOTION_MANIFEST),robot.joint_names,POLICY_JOINT_NAMES,device,float(cfg.sim.dt*cfg.decimation))
        active = (dataset.weights>0).nonzero().flatten().tolist()
        near_counts = {m: 0 for m in active}
        near_targets = {m: args.near_count//len(active) + int(i < args.near_count%len(active))
                        for i,m in enumerate(active)}
        controlled = torch.tensor([robot.joint_names.index(x) for x in POLICY_JOINT_NAMES],device=device)
        torso = torch.tensor(robot.find_bodies('torso_link')[0],device=device,dtype=torch.long)
        stiffness = robot.data.joint_stiffness.torch.clone()
        damping = robot.data.joint_damping.torch.clone()
        limits = robot.data.joint_pos_limits.torch.clone()
        zero = torch.zeros(n,len(robot.joint_names),device=device)
        env_ids = torch.arange(n,device=device)
        def gains(passive):
            robot.write_joint_stiffness_to_sim_index(stiffness=torch.zeros_like(stiffness) if passive else stiffness)
            robot.write_joint_damping_to_sim_index(damping=torch.zeros_like(damping) if passive else damping)
            robot.set_joint_effort_target_index(target=zero)
        def write(pose,q,rv=None,jv=None):
            robot.permanent_wrench_composer.reset()
            scene.reset()
            world = pose.clone(); world[:,:3] += scene.env_origins
            _write_reference_state(None,env_ids,robot,world,
                torch.zeros(n,6,device=device) if rv is None else rv,q,zero if jv is None else jv)
            sim.forward();scene.update(dt)
        def step():
            robot.set_joint_effort_target_index(target=zero)
            scene.write_data_to_sim();sim.step(render=False);scene.update(dt)
        def measures():
            pose=robot.data.root_link_pose_w.torch
            velocity=robot.data.root_link_vel_w.torch
            q=robot.data.joint_pos.torch;jv=robot.data.joint_vel.torch
            conjugate=pose[:,3:7].clone();conjugate[:,:3]*=-1
            gravity=rotate(conjugate,torch.tensor([0.,0.,-1.],device=device).expand(n,-1))
            linear=velocity[:,:3].norm(dim=-1);angular=velocity[:,3:].norm(dim=-1)
            jspeed=jv.square().mean(-1).sqrt()
            gf,sf,gd,sd=contact.metrics()
            local_height=pose[:,2]-scene.env_origins[:,2]
            valid=(torch.isfinite(pose).all(-1)&torch.isfinite(velocity).all(-1)&torch.isfinite(q).all(-1)
                   &(q>=limits[:,:,0]-.003).all(-1)&(q<=limits[:,:,1]+.003).all(-1)
                   &(local_height>0)&(local_height<args.max_fallen_height)&(-gravity[:,2]<args.max_root_up)
                   &(linear<args.max_linear_speed)&(angular<args.max_angular_speed)
                   &(jspeed<args.max_joint_speed_rms)&(gf>20)&(sf<args.max_self_force)
                   &(gd>=-args.max_penetration)&(sd>=-args.max_penetration))
            return valid,gravity,torch.stack((gf,sf,gd,sd,linear,angular,jspeed),-1)
        if args.exclude_bank is not None:
            old=load_bank(args.exclude_bank,joint_names=robot.joint_names)
            for old_i in range(len(old['root_pose'])):
                seen.add(dedup_key(old['joint_pos'][old_i],old['root_gravity_b'][old_i],old['root_pose'][old_i,2]))
        side_count=0; mode_counts=Counter(); family_modes={}
        family=0
        # Separate modes ensure one source cannot silently replace another's quota.
        for source in [SOURCE_NEAR,SOURCE_FALL]:
            for batch in range(args.max_batches):
                if accepted[source]>=targets[source]:break
                attempts[source]+=n
                families=list(range(family,family+n));family+=n
                mids=[active[(batch*n+i)%len(active)] for i in range(n)] if source==SOURCE_NEAR else [-1]*n
                # Deterministic batch schedule avoids random coverage starvation.
                drop = source==SOURCE_FALL and ((batch+1)*args.drop_fraction//1 > batch*args.drop_fraction//1)
                for f in families: family_modes[str(f)]='side_placement' if drop else ('push' if source==SOURCE_FALL else 'near_reference')
                gains(False)
                if source==SOURCE_NEAR:
                    pose=torch.stack([dataset.clips[i]['root_pose'][0] for i in mids]).clone()
                    q=torch.stack([dataset.clips[i]['joint_pos'][0] for i in mids]).clone()
                    # Match the existing reset's hard-limit alignment before adding noise.
                    q=q.clamp(limits[:,:,0],limits[:,:,1])
                    lower=(limits[:,controlled,0]-q[:,controlled]).clamp(-args.joint_noise,args.joint_noise)
                    upper=(limits[:,controlled,1]-q[:,controlled]).clamp(-args.joint_noise,args.joint_noise)
                    noise=lower+torch.rand((n,len(controlled)),device=device,generator=rng)*(upper-lower)
                    q[:,controlled]+=noise
                    seed_valid=(q>=limits[:,:,0]-1e-6).all(-1)&(q<=limits[:,:,1]+1e-6).all(-1)
                    # Sample within the feasible interval, instead of clipping noisy candidates.
                    q=q.clamp(limits[:,:,0],limits[:,:,1])
                else:
                    pose=robot.data.default_root_state.torch[:,:7].clone()
                    pose[:,2]=.68
                    q=robot.data.default_joint_pos.torch.clone()
                    seed_valid=torch.ones(n,dtype=torch.bool,device=device)
                    if drop:
                        # Independent side placement; no holdout pose is used as seed.
                        signs=torch.where(torch.rand(n,device=device,generator=rng)<.5,-1.,1.)
                        roll=signs*(np.pi/2)+(torch.rand(n,device=device,generator=rng)-.5)*.4
                        pitch=(torch.rand(n,device=device,generator=rng)-.5)*.6
                        cr,sr=torch.cos(roll/2),torch.sin(roll/2)
                        cp,sp=torch.cos(pitch/2),torch.sin(pitch/2)
                        pose[:,3:7]=torch.stack((sr*cp,cr*sp,-sr*sp,cr*cp),-1)
                        lower=(limits[:,controlled,0]-q[:,controlled]).clamp(-args.drop_joint_noise,args.drop_joint_noise)
                        upper=(limits[:,controlled,1]-q[:,controlled]).clamp(-args.drop_joint_noise,args.drop_joint_noise)
                        q[:,controlled]+=lower+torch.rand((n,len(controlled)),device=device,generator=rng)*(upper-lower)
                        # Folded legs and asymmetric abduction create broader side support.
                        def set_joint(name,values):
                            j=robot.joint_names.index(name)
                            q[:,j]=values.clamp(limits[:,j,0],limits[:,j,1])
                        random=lambda lo,hi: lo+(hi-lo)*torch.rand(n,device=device,generator=rng)
                        for name in ['left_hip_pitch_joint','right_hip_pitch_joint']:set_joint(name,random(-1.5,-.4))
                        for name in ['left_knee_joint','right_knee_joint']:set_joint(name,random(.8,2.2))
                        for name in ['left_hip_yaw_joint','right_hip_yaw_joint']:set_joint(name,random(-.6,.6))
                        set_joint('left_hip_roll_joint',torch.where(signs>0,random(.05,.3),random(.5,1.1)))
                        set_joint('right_hip_roll_joint',torch.where(signs>0,random(-1.1,-.5),random(-.3,-.05)))
                        for name in ['waist_yaw_joint','waist_roll_joint']:set_joint(name,random(-.25,.25))
                        for name in ['left_elbow_joint','right_elbow_joint']:set_joint(name,random(-1.6,-.3))
                pose[:,:2]=0
                yaw_rotate(pose,(torch.rand(n,device=device,generator=rng)*2-1)*np.pi)
                write(pose,q)
                # Use collision mesh vertices to lift clear of the ground before physics.
                bottom=geometry.minimum_z(robot)-scene.env_origins[:,2]
                pose[:,2]+=.01-bottom
                write(pose,q)
                if source==SOURCE_FALL and not drop:
                    for _ in range(round(1.5/dt)):step()
                    p0=robot.data.root_link_pose_w.torch;v0=robot.data.root_link_vel_w.torch
                    upright=1-2*(p0[:,3].square()+p0[:,4].square())
                    seed_valid&=(p0[:,2]-scene.env_origins[:,2]>.45)&(upright>.85)&(v0[:,:3].norm(dim=-1)<.2)
                    theta=torch.rand(n,device=device,generator=rng)*2*np.pi
                    magnitude=args.push_force_min+torch.rand(n,device=device,generator=rng)*(args.push_force_max-args.push_force_min)
                    force=torch.zeros(n,1,3,device=device)
                    force[:,0,0]=magnitude*theta.cos();force[:,0,1]=magnitude*theta.sin()
                    # Physically push while normal PD still holds the standing target.
                    robot.permanent_wrench_composer.reset()
                    robot.permanent_wrench_composer.add_forces_and_torques_index(
                        forces=force,torques=torch.zeros_like(force),body_ids=torso,is_global=True)
                    for _ in range(round(args.push_seconds/dt)):step()
                    robot.permanent_wrench_composer.reset()
                gains(True)
                streak=torch.zeros(n,dtype=torch.long,device=device)
                found=torch.zeros(n,dtype=torch.bool,device=device)
                snap={k:torch.zeros((n,width),device=device) for k,width in
                      [('root_pose',7),('root_velocity',6),('joint_pos',len(robot.joint_names)),
                       ('joint_vel',len(robot.joint_names)),('root_gravity_b',3),('quality',7)]}
                # Unaccepted slots still need valid poses during batched replay.
                snap['root_pose'].copy_(pose)
                snap['joint_pos'].copy_(q)
                for _ in range(round(args.settle_seconds/dt)):
                    step();valid,gravity,quality=measures()
                    streak=torch.where(valid&seed_valid,streak+1,torch.zeros_like(streak))
                    fresh=(streak>=round(args.stable_seconds/dt))&~found
                    if fresh.any():
                        local=robot.data.root_link_pose_w.torch.clone();local[:,:3]-=scene.env_origins
                        for k,value in [('root_pose',local),('root_velocity',robot.data.root_link_vel_w.torch),
                                        ('joint_pos',robot.data.joint_pos.torch),('joint_vel',robot.data.joint_vel.torch),
                                        ('root_gravity_b',gravity),('quality',quality)]:snap[k][fresh]=value[fresh]
                        found|=fresh
                    if found.all():break
                rejection['seed_invalid']+=int((~seed_valid).sum())
                rejection['not_settled_or_invalid']+=int((seed_valid&~found).sum())
                # Validate the actual training reset condition, centered XY.
                snap['root_pose'][:,:2]=0.
                # Replay the exact captured state with normal PD targeting that same pose.
                gains(False);write(snap['root_pose'],snap['joint_pos'],snap['root_velocity'],snap['joint_vel'])
                replay_ok=found.clone();peak_lin=torch.zeros(n,device=device);peak_ang=peak_lin.clone()
                for _ in range(round(args.replay_seconds/dt)):
                    step();pose_now=robot.data.root_link_pose_w.torch;v=robot.data.root_link_vel_w.torch;qnow=robot.data.joint_pos.torch
                    gf,sf,gd,sd=contact.metrics()
                    lin=v[:,:3].norm(dim=-1);ang=v[:,3:].norm(dim=-1)
                    peak_lin=torch.maximum(peak_lin,lin);peak_ang=torch.maximum(peak_ang,ang)
                    replay_ok&=(torch.isfinite(pose_now).all(-1)&torch.isfinite(v).all(-1)&torch.isfinite(qnow).all(-1)
                        &(lin<args.max_replay_linear)&(ang<args.max_replay_angular)
                        &(gf<args.max_replay_ground_force)&(gd>=-args.max_penetration)&(sd>=-args.max_penetration)
                        &(qnow>=limits[:,:,0]-.003).all(-1)&(qnow<=limits[:,:,1]+.003).all(-1))
                rejection['replay_invalid']+=int((found&~replay_ok).sum())
                for i in replay_ok.nonzero().flatten().tolist():
                    if accepted[source]>=targets[source]:break
                    if source==SOURCE_NEAR and near_counts[mids[i]]>=near_targets[mids[i]]:continue
                    side=bool(snap['root_gravity_b'][i,1].abs()>snap['root_gravity_b'][i,[0,2]].abs().max())
                    if source==SOURCE_FALL and not side and targets[source]-accepted[source] <= max(0,args.min_side_count-side_count):
                        rejection['side_quota_reserved']+=1;continue
                    key=dedup_key(snap['joint_pos'][i].cpu().numpy(),snap['root_gravity_b'][i].cpu().numpy(),snap['root_pose'][i,2].item())
                    if key in seen:rejection['quantized_duplicate']+=1;continue
                    seen.add(key)
                    row={k:snap[k][i].cpu().numpy().copy() for k in snap if k!='quality'}
                    row.update(source_type=np.int64(source),motion_id=np.int64(mids[i]),family_id=np.int64(families[i]),
                        split=np.int64(family_split(args.seed,families[i],args.holdout_fraction)),
                        quality=np.r_[snap['quality'][i].cpu().numpy(),peak_lin[i].item(),peak_ang[i].item()].astype(np.float32))
                    rows.append(row);accepted[source]+=1
                    if source==SOURCE_FALL:
                        side_count+=int(side)
                        mode_counts['side_placement' if drop else 'push']+=1
                    if source==SOURCE_NEAR:near_counts[mids[i]]+=1
                # Atomic partial checkpoint is explicitly not training ready.
                checkpoint=args.output.with_name(args.output.stem+'.partial.npz')
                temp=args.output.with_name(args.output.stem+'.partial.tmp.npz')
                if temp.exists():temp.unlink()
                if rows:
                    save_bank(temp,rows,list(robot.joint_names),dict(collection_report={'complete':False,'training_ready':False},
                        settings=report['settings'],side_count=side_count,family_collection_modes=family_modes))
                    temp.replace(checkpoint)
                print(f'[fallen bank] source={source} batch={batch+1} accepted={accepted} side={side_count} rejected={dict(rejection)}',flush=True)
        report.update(accepted=accepted,attempts=attempts,rejections=dict(rejection),
                      near_counts=near_counts,near_targets=near_targets,
                      side_count=side_count,mode_counts=dict(mode_counts),
                      complete=all(accepted[s]>=targets[s] for s in targets) and side_count>=args.min_side_count)
        coverage = {f'{source}/{split}': sum(int(r['source_type'])==source and int(r['split'])==split for r in rows)
                    for source in [SOURCE_NEAR,SOURCE_FALL] for split in [0,1]}
        near_coverage = {f'{motion}/{split}': sum(int(r['source_type'])==SOURCE_NEAR and int(r['motion_id'])==motion
                        and int(r['split'])==split for r in rows) for motion in active for split in [0,1]}
        report.update(coverage=coverage,near_coverage=near_coverage,
                      training_ready=report['complete'] and all(coverage.values()) and all(near_coverage.values()))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2,default=str))
        if rows:
            meta={'generator':str(Path(__file__).resolve()),'generator_sha256':sha256(__file__),
                  'settings':report['settings'],'collection_report':report,'asset_hashes':asset_hashes(X2_ULTRA_USD.parent),
                  'reference_files':[str(p) for p in dataset.paths],
                  'reference_sha256':[sha256(p) for p in dataset.paths],
                  'physics_dt':dt,'collection_control':'PD=0 after perturbation/push; normal PD for replay',
                  'velocities':'actual captured link-origin world velocities, not artificially zeroed',
                  'validation':'0.5s passive rest + normal-PD same-pose replay; not a get-up success filter',
                  'dedup':'quantized joints/gravity/height, ignores world XY and yaw',
                  'family_collection_modes':family_modes,'centered_before_replay':True}
            bank=save_bank(args.output,rows,list(robot.joint_names),meta)
            print(json.dumps(summary(bank),indent=2),flush=True)
        else:
            raise RuntimeError('No states passed; see report. Do not enable training bank sampling.')
        if not report['complete']:
            raise RuntimeError('Partial bank saved; quotas unmet. Inspect report before use.')


if __name__=='__main__':
    with torch.inference_mode():main()
