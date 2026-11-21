"""基于32步真实窗口的高度变化分层；这些标签不等于起身成功或接触状态。"""
import torch

PHASE_NAMES = ('low_other', 'rising', 'middle_other', 'falling', 'high_other')


def labels_from_heights(heights):
    if heights.ndim != 2 or heights.shape[1] < 2 or not bool(torch.isfinite(heights).all()):
        raise ValueError('窗口高度必须为有限的 [窗口数量, 时间长度] 张量')
    median = heights.median(dim=1).values
    labels = torch.full((len(heights),), 2, dtype=torch.long, device=heights.device)
    labels[median < .3] = 0
    labels[median >= .5] = 4
    change = heights[:, -1] - heights[:, 0]
    span = heights.amax(1) - heights.amin(1)
    moving = span >= .1
    labels[moving & (change >= .05)] = 1
    labels[moving & (change <= -.05)] = 3
    return labels


def candidate_labels(data, candidates, horizon=32):
    start, env = candidates.unbind(-1)
    time = start[:, None] + torch.arange(horizon)
    heights = torch.cat((data['state'][start, env, 2, None],
                         data['next_state'][time, env[:, None], 2]), dim=1)
    return labels_from_heights(heights)


def choose_indexes(labels, batch_size, balanced):
    if batch_size < 1 or len(labels) == 0:
        raise ValueError('批量必须为正数，候选窗口不能为空')
    if not balanced:
        return torch.randint(len(labels), (batch_size,))
    groups = [(labels == index).nonzero().flatten() for index in range(len(PHASE_NAMES))]
    available = [group for group in groups if len(group)]
    quotient, remainder = divmod(batch_size, len(available))
    extra = set(torch.randperm(len(available))[:remainder].tolist())
    chosen = []
    for index, group in enumerate(available):
        count = quotient + int(index in extra)
        if count:
            chosen.append(group[torch.randint(len(group), (count,))])
    indexes = torch.cat(chosen)
    return indexes[torch.randperm(len(indexes))]


def gather_windows(data, candidates, horizon, device):
    start, env = candidates.unbind(-1)
    time = start[:, None] + torch.arange(horizon)
    return dict(states=torch.cat((data['state'][start, env, None],
                                  data['next_state'][time, env[:, None]]), dim=1).to(device),
                actions=data['action'][time, env[:, None]].to(device),
                context=data['contexts'][env].to(device))


def label_counts(labels):
    return {name: int((labels == index).sum()) for index, name in enumerate(PHASE_NAMES)}
