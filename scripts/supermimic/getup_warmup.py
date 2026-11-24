"""脚本层世界模型预热：只更新动力学，不更新 actor、critic 或判别器。"""
import torch


def warmup_world_model(runtime, args, record):
    if not runtime.warmup_complete and not bool(runtime.trainer.world_model.stats_ready) and args.warmup_updates < 1:
        raise ValueError('未初始化的世界模型必须完成监督预热，不能直接更新策略')
    # 断点恢复也要重新采集，因为 checkpoint 不包含 replay 或物理场景。
    print('[预热] 采集真实转移并更新真实观测统计量', flush=True)
    runtime.runner.collect(args.warmup_steps)
    if runtime.warmup_complete:
        if not bool(runtime.trainer.world_model.stats_ready):
            raise RuntimeError('已完成预热的检查点缺少世界模型输入统计量')
        print('[预热] 已恢复模型，跳过额外的世界模型预热更新', flush=True)
        return

    frozen = {name: {key: value.detach().clone() for key, value in runtime.modules[name].state_dict().items()}
              for name in ('actor', 'critic', 'style')}
    world = runtime.trainer.world_model
    if not bool(world.stats_ready):
        batch = runtime.runner.sample_world_batch(
            args.world_batch_size, args.world_horizons[0] - 1, runtime.offline, args.offline_world_fraction)
        world.fit_input_stats(batch)
        print(f'[预热] 已由真实数据初始化世界模型输入统计量，维度={world.input_mean.numel()}', flush=True)

    for update in range(args.warmup_updates):
        if update > 0 and update % args.warmup_collect_interval == 0:
            runtime.runner.collect(args.collection_steps)
        horizon = args.world_horizons[update % len(args.world_horizons)] - 1
        batch = runtime.runner.sample_world_batch(
            args.world_batch_size, horizon, runtime.offline, args.offline_world_fraction)
        metrics = runtime.trainer.update_world_model(batch)
        record('warmup', update + 1, metrics)
        if update == 0 or (update + 1) % 25 == 0 or update + 1 == args.warmup_updates:
            print(f"[世界模型预热 {update + 1}/{args.warmup_updates}] 损失={metrics['world_loss']:.6f}", flush=True)

    for name, before in frozen.items():
        for key, value in runtime.modules[name].state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    runtime.warmup_complete = True
    print('[预热完成] actor、critic 和判别器未更新；接下来开始联合训练', flush=True)
