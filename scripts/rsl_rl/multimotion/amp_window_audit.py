"""Inference-only AMP capture; uses true terminal states and training history semantics."""
import json
from pathlib import Path
import torch
from x2_style_mimic.algorithms.amp.history import AMPHistory

BLOCKS = {"root_height": (0, 1), "root_gravity": (1, 4),
          "root_linear": (4, 7), "root_angular_scaled": (7, 10),
          "joint_position": (10, 39), "joint_velocity_scaled": (39, 68),
          "body_position": (68, 101), "body_rotation6d": (101, 167),
          "body_linear": (167, 200), "body_angular_scaled": (200, 233)}

class AMPWindowAudit:
    def __init__(self, env, algorithm, output, checkpoint, quota):
        self.env, self.alg = env, algorithm
        self.output = Path(output).resolve()
        self.checkpoint = str(Path(checkpoint).resolve())
        self.quota = quota
        self.history = AMPHistory(env.num_envs, 233, algorithm.amp_num_steps, env.device)
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.times = torch.zeros(env.num_envs, device=env.device)
        self.rows = []

    def begin(self, obs):
        self.history.begin(obs["amp"])

    def advance(self, dones, extras):
        dones = dones.bool()
        if not torch.equal(dones, extras["amp_terminal_mask"].bool()):
            raise RuntimeError("Terminal masks differ")
        windows, valid = self.history.advance(extras["amp_next_state"], dones)
        self.times += self.env.step_dt
        if self.quota:
            valid &= self.episodes < self.quota
        ids = valid.nonzero().flatten()
        progress = extras["getup_progress"]
        if len(ids):
            w = windows[ids]
            score = self.alg.amp_discriminator(w)
            self.rows.append({"windows": w.cpu(), "score": score.cpu(),
                "env_id": ids.cpu(), "episode": self.episodes[ids].cpu().clone(),
                "motion_id": progress["motion_ids"][ids].cpu(),
                "time_s": self.times[ids].cpu().clone(),
                "hold_s": progress["hold_seconds"][ids].cpu(),
                "terminal": dones[ids].cpu()})
        self.episodes += dones.long()
        self.times[dones] = 0

    def close(self):
        if not self.rows:
            return
        data = {k: torch.cat([r[k] for r in self.rows]) for k in self.rows[0]}
        # A five-state window spans four control intervals (0.08 s).
        # Require every state in the window to belong to the standing interval.
        span = (self.alg.amp_num_steps - 1) * self.env.step_dt
        standing = data["hold_s"] > span + 1e-6
        data["standing"] = standing
        data["style_reward"] = (1 - .25 * (data["score"] - 1).square()).clamp(0, 1)
        data["checkpoint"] = self.checkpoint
        data["feature_version"] = "x2_amp_relative_body_v1_233_window_v1"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, self.output)
        device = self.alg.device
        reference = torch.cat([s.unfold(0, self.alg.amp_num_steps, 1).permute(0, 2, 1)
                               for s in self.alg.amp_reference.amp_states]).to(device)
        d = self.alg.amp_discriminator
        # Self-contained offline analysis, using exactly the captured checkpoint statistics.
        data["reference_windows"] = reference.cpu()
        data["state_mean"] = d.state_mean.cpu()
        data["state_std"] = d.state_std.cpu()
        data["joint_names"] = list(self.alg.amp_cfg["joint_names"])
        from x2_style_mimic.tasks.manager_based.x2_style_mimic.mdp.amp import AMP_KEY_BODY_NAMES
        data["body_names"] = list(AMP_KEY_BODY_NAMES)
        data["step_dt"] = self.env.step_dt
        ref = d.normalize(reference)
        ref_norm = ref.square().sum(-1)
        nearest, distances = [], []
        # Match entire windows to the closest window anywhere in the reference library.
        for chunk in data["windows"].split(128):
            z = d.normalize(chunk.to(device))
            dist = (z.square().sum(-1, keepdim=True) + ref_norm[None] - 2*z@ref.T).clamp_min(0)
            value, index = dist.min(-1)
            nearest.append(index.cpu()); distances.append((value / z.shape[-1]).sqrt().cpu())
        match = torch.cat(nearest)
        data_distance = torch.cat(distances)
        data["nearest_reference_window_index"] = match
        data["reference_window_counts"] = [len(s) - self.alg.amp_num_steps + 1 for s in self.alg.amp_reference.amp_states]
        data["reference_files"] = [str(p) for p in self.alg.amp_reference.paths]
        torch.save(data, self.output)
        errors = {}
        for name, (lo, hi) in BLOCKS.items():
            delta = (data["windows"][:, :, lo:hi] - reference[match.to(device), :, lo:hi].cpu()) / d.state_std[lo:hi].cpu()
            errors[name] = delta.square().mean((1, 2)).sqrt()
        def stats(mask):
            if not mask.any(): return {"count": 0}
            score, reward = data["score"][mask], data["style_reward"][mask]
            return {"count": int(mask.sum()), "score_mean": float(score.mean()),
                "score_p10_p50_p90": score.quantile(torch.tensor([.1,.5,.9])).tolist(),
                "style_mean": float(reward.mean()), "zero_style_fraction": float((reward == 0).float().mean()),
                "nearest_reference_normalized_rms": float(data_distance[mask].mean()),
                "block_normalized_rms": {k: float(v[mask].mean()) for k,v in errors.items()}}
        report = {"checkpoint": self.checkpoint, "capture": str(self.output),
            "phase_definition": "standing: current uninterrupted success-predicate hold exceeds full window span; other windows: getup_or_transition (including boundary/recovery)",
            "comparison": "Nearest full five-state reference window, all clips/all phases; normalized with checkpoint reference std (floor 0.05). Descriptive mismatch, not causal attribution.",
            "all": stats(torch.ones_like(standing)), "standing": stats(standing), "getup_or_transition": stats(~standing),
            "by_motion": {str(int(i)): {"standing": stats(standing & (data["motion_id"]==i)),
                "getup_or_transition": stats(~standing & (data["motion_id"]==i))} for i in data["motion_id"].unique()}}
        report_path = self.output.with_suffix(".json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("[AMP 窗口分析]", report_path, json.dumps({k: report[k] for k in ("standing", "getup_or_transition")}), flush=True)
