"""Audit reward-to-actor gradients without taking optimizer steps."""
import json
import math
from pathlib import Path

import torch


def reward_gae(rewards, dones, gamma, lam):
    """Reward-only GAE component: zero value baseline, terminal boundaries respected."""
    result = torch.zeros_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        tail = rewards[t] + gamma * lam * (1 - dones[t].float()) * tail
        result[t] = tail
    return result


def gradient_vectors(alg, total_adv, amp_adv):
    """Use common advantage normalization so task + AMP remains exactly additive."""
    st = alg.storage
    obs = st.observations.flatten(0, 1)
    actions = st.actions.flatten(0, 1)
    old_logp = st.actions_log_prob.flatten().detach()
    total = total_adv.flatten().detach()
    amp = amp_adv.flatten().detach()
    scale = total.std().clamp_min(1e-8)
    task = total - amp
    normalized = [(v - v.mean()) / scale for v in (total, task, amp)]
    params = [p for p in alg.actor.parameters() if p.requires_grad]
    vectors = [torch.zeros(sum(p.numel() for p in params), device=alg.device) for _ in range(3)]
    ratios = []
    for start in range(0, len(total), 512):
        end = min(start + 512, len(total))
        alg.actor(obs[start:end], stochastic_output=True)
        logp = alg.actor.get_output_log_prob(actions[start:end])
        ratio = (logp - old_logp[start:end]).exp()
        ratios.append(ratio.detach())
        for i, adv in enumerate(normalized):
            a = adv[start:end]
            # Components use the unclipped surrogate for exact additive attribution.
            # Total uses the same unclipped surrogate; clipping deviation is reported separately.
            loss = -(a * ratio).sum() / len(total)
            grads = torch.autograd.grad(loss, params, retain_graph=i < 2, allow_unused=True)
            vectors[i] += torch.cat([g.detach().flatten() if g is not None else torch.zeros_like(p).flatten()
                                     for g, p in zip(grads, params)])
    ratio = torch.cat(ratios)
    return vectors, {
        "ratio_mean": float(ratio.mean()),
        "ratio_outside_clip_fraction": float(((ratio - 1).abs() > alg.clip_param).float().mean()),
        "advantage_std_total": float(scale),
        "advantage_std_amp": float(amp.std()),
        "advantage_std_task": float(task.std()),
    }


def summarize_vectors(vectors):
    total, task, amp = vectors
    tn, an = task.norm(), amp.norm()
    return {
        "total_gradient_norm": float(total.norm()),
        "task_gradient_norm": float(tn),
        "amp_gradient_norm": float(an),
        "amp_to_task_gradient_norm_ratio": float(an / tn.clamp_min(1e-12)),
        "task_amp_cosine": float(torch.dot(task, amp) / (tn * an).clamp_min(1e-12)),
        "additivity_relative_error": float((total-task-amp).norm() / total.norm().clamp_min(1e-12)),
    }


def run_audit(env, runner, output, checkpoint):
    alg = runner.alg
    if alg.actor.is_recurrent or alg.normalize_advantage_per_mini_batch:
        raise ValueError("Audit requires feedforward actor and rollout-wide advantage normalization")
    # Freeze observation statistics to isolate the checkpoint policy. Otherwise even without
    # optimizer steps normalization updates change the behavior policy within each rollout.
    alg.eval_mode()
    before = {k: v.detach().clone() for k, v in alg.actor.state_dict().items()}
    d_before = {k: v.detach().clone() for k, v in alg.amp_discriminator.state_dict().items()}
    obs = env.get_observations().to(alg.device)
    horizon = runner.cfg["num_steps_per_env"]
    iterations = math.ceil(env.unwrapped.max_episode_length / horizon)
    rows, totals, score_rows = [], None, []
    phase_rows = {"getup": [], "post_hold": []}
    for iteration in range(iterations):
        amp_rewards, holds, task_rewards = [], [], []
        with torch.inference_mode():
            for _ in range(horizon):
                actions = alg.act(obs)
                obs, reward, done, extras = env.step(actions.to(env.device))
                obs, reward, done = obs.to(alg.device), reward.to(alg.device), done.to(alg.device)
                task = reward.clone()
                alg.process_env_step(obs, reward, done, extras)
                amp_rewards.append((reward-task).clone())
                task_rewards.append(task)
                holds.append(extras["getup_progress"]["hold_seconds"].to(alg.device).clone())
            alg.compute_returns(obs)
        # Copies outside inference_mode are ordinary tensors for autograd operations.
        amp_r = torch.stack(amp_rewards).clone()
        task_r = torch.stack(task_rewards).clone()
        hold = torch.stack(holds).clone()
        raw_total = (alg.storage.returns-alg.storage.values).squeeze(-1).clone()
        amp_adv = reward_gae(amp_r, alg.storage.dones.squeeze(-1), alg.gamma, alg.lam)
        vectors, row = gradient_vectors(alg, raw_total, amp_adv)
        row.update(summarize_vectors(vectors))
        row.update(iteration=iteration, time_end_s=(iteration+1)*horizon*env.unwrapped.step_dt,
                   getup_fraction=float((hold <= .1).float().mean()))
        rows.append(row)
        if totals is None:
            totals = [v.clone() for v in vectors]
        else:
            for accum, vector in zip(totals, vectors):
                accum += vector
        for name, mask in (("getup", hold <= .1), ("post_hold", hold > .1)):
            phase_rows[name].append({"count": int(mask.sum()), "task_sum": float(task_r[mask].sum()),
                                     "amp_sum": float(amp_r[mask].sum()),
                                     "amp_zero_count": int((amp_r[mask] <= 0).sum())})
        if alg._amp_getup_scores:
            score_rows.extend([x.cpu() for x in alg._amp_getup_scores])
        alg._amp_getup_scores.clear()
        alg._amp_getup_rewards.clear()
        alg.storage.clear()
        print(f"[AMP gradient audit] {iteration+1}/{iterations}: "
              f"AMP/task gradient={row['amp_to_task_gradient_norm_ratio']:.4f}, "
              f"cos={row['task_amp_cosine']:+.4f}", flush=True)
    if not all(torch.equal(v, before[k]) for k,v in alg.actor.state_dict().items()):
        raise RuntimeError("Actor state changed during read-only audit")
    if not all(torch.equal(v, d_before[k]) for k,v in alg.amp_discriminator.state_dict().items()):
        raise RuntimeError("Discriminator changed during read-only audit")
    phases = {}
    for name, values in phase_rows.items():
        n = sum(r["count"] for r in values)
        phases[name] = {"count": n,
            "task_reward_mean": sum(r["task_sum"] for r in values)/max(n,1),
            "amp_reward_scaled_mean": sum(r["amp_sum"] for r in values)/max(n,1),
            "amp_zero_fraction": sum(r["amp_zero_count"] for r in values)/max(n,1)}
    scores = torch.cat(score_rows) if score_rows else torch.empty(0)
    report = {"checkpoint": str(Path(checkpoint).resolve()), "num_envs": env.num_envs,
        "actor_and_discriminator_unchanged": True,
        "method": "Fixed checkpoint; stochastic on-policy actions; zero assistance; frame-zero births; "
                  "frozen observation normalizers; training-length GAE rollouts with existing critic and timeout bootstrap. "
                  "AMP component has zero value baseline; task = total - AMP. Common centering and total advantage std. "
                  "Unclipped actor surrogate gradients at unchanged weights, excluding entropy/critic; "
                  "ratio outside PPO clip range is reported to assess applicability. No optimizer updates. "
                  "Last rollout can extend just beyond the first episode.",
        "phases": phases, "mean_gradient_vector": summarize_vectors([v/iterations for v in totals]),
        "getup_score_p10_p50_p90": scores.quantile(torch.tensor([.1,.5,.9])).tolist() if len(scores) else [],
        "rollouts": rows}
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2),encoding="utf-8")
    print("[AMP gradient audit saved]", path, flush=True)
