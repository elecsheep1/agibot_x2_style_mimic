"""世界模型监督批次的校验与合并。"""

import torch


def detach_world_batch(batch, device):
    """递归保留字段，切断数据梯度并移动到指定设备。"""

    result = {}

    for name, value in batch.items():
        if isinstance(value, dict):
            result[name] = detach_world_batch(
                value,
                device,
            )
        elif isinstance(value, torch.Tensor):
            result[name] = (
                value.detach().to(device)
            )
        else:
            raise ValueError(f"世界模型批次字段必须为张量或字典：{name}")

    return result


def merge_world_batches(
    batches,
    batch_size,
    horizon,
    state_dim,
    action_dim,
    context_dim,
    extra_fields,
    device,
):
    """把在线和离线采样得到的多批轨迹数据，检查后合并成一个训练批次，再统一打乱样本顺序。"""

    if not batches:
        raise ValueError("世界模型批次不能为空")

    # shape 不包含批量维；None 表示要求浮点型，
    # 不额外限制为 float32。
    specifications = {
        ("states",): (
            (horizon + 1, state_dim),
            None,
        ),
        ("actions",): (
            (horizon, action_dim),
            None,
        ),
        ("context",): (
            (context_dim,),
            None,
        ),
    }

    for path, spec in extra_fields.items():
        field_shape, dtype = spec
        specifications[path] = (
            (horizon, *field_shape),
            dtype,
        )

    merged = {}

    for path, spec in specifications.items():
        expected_tail, expected_dtype = spec
        values = []

        for source_index, batch in enumerate(batches):
            value = batch

            for name in path:
                if (
                    not isinstance(value, dict)
                    or name not in value
                ):
                    raise ValueError(
                        f"第 {source_index} 个数据批次缺少字段："
                        f"{'.'.join(path)}。"
                        "旧离线数据不能用补零代替。"
                    )
                value = value[name]

            if not isinstance(value, torch.Tensor):
                raise ValueError(f"批次字段必须为张量：{'.'.join(path)}")
            if (
                value.ndim != len(expected_tail) + 1
                or tuple(value.shape[1:])
                != expected_tail
            ):
                raise ValueError(
                    f"批次字段形状错误：{'.'.join(path)}，"
                    f"实际={tuple(value.shape)}，"
                    f"要求=[B, {expected_tail}]"
                )
            if expected_dtype is None:
                if not value.is_floating_point():
                    raise ValueError(
                        f"批次字段必须为浮点型："
                        f"{'.'.join(path)}"
                    )
            elif value.dtype != expected_dtype:
                raise ValueError(
                    f"批次字段类型错误：{'.'.join(path)}，"
                    f"要求={expected_dtype}，"
                    f"实际={value.dtype}"
                )
            if (
                value.is_floating_point()
                and not bool(
                    torch.isfinite(value).all()
                )
            ):
                raise ValueError(
                    f"批次字段包含 NaN/Inf："
                    f"{'.'.join(path)}"
                )
            # 每个字段必须与所属来源的 states 批量相同。
            if value.shape[0] != batch["states"].shape[0]:
                raise ValueError(
                    f"同一批次的样本数不一致："
                    f"{'.'.join(path)}"
                )

            values.append(
                value.detach().to(device)
            )

        combined = torch.cat(
            values,
            dim=0,
        )

        if combined.shape[0] != batch_size:
            raise ValueError(f"合并后的样本数错误：{'.'.join(path)}")

        merged[path] = combined

    order = torch.randperm(
        batch_size,
        device=device,
    )

    result = {}

    for path, value in merged.items():
        target = result

        for name in path[:-1]:
            target = target.setdefault(
                name,
                {},
            )

        target[path[-1]] = value[order]

    return result