import torch  # 导入 PyTorch，提供张量存储、索引和随机采样操作。


class SequenceReplayBuffer:  # 定义按时间步和并行环境组织的序列回放缓冲区。
    """
    用于存储和采样序列的经验回放缓冲区。
    """

    def __init__(  # 初始化多环境环形存储、固定上下文以及回合和步计数。
        self,  # 当前缓冲区实例。
        num_envs,  # 并行环境数量 N。
        capacity_steps,  # 最多保留的全环境控制步数。
        state_dim,  # 单个状态向量的维度。
        action_dim,  # 单个动作向量的维度。
        contexts,  # 各环境对应的固定上下文，形状为 [N, Dc]。
        device="cpu",  # 张量存储设备，默认使用 CPU。
        command_dim=0,  # 可选的命令向量维度，默认为零。
        extra_fields=None,  # 可选的额外字段字典，键为字段名，值为形状元组。
    ):  # 结束参数或遍历项列表，开始执行对应代码块。
        '''
        初始化用于多环境序列采样的环形经验回放缓冲区。

        校验环境数量、存储容量、状态和动作维度，以及上下文的形状；
        将每个环境的固定上下文复制到指定设备，并分配状态、后继状态、
        动作、奖励、终止标志、截断标志及回合元数据的存储空间。
        容量以一次覆盖全部环境的控制步为单位，写满后循环覆盖最旧数据。
        各环境的回合编号、回合内步编号和累计写入步数均从零开始。
        '''
        if min(  # 检查环境数、容量和向量维度是否均为正数。
            num_envs, capacity_steps, state_dim, action_dim  # 参与正数检查的四个维度参数。
        ) <= 0:  # 任意维度不为正时进入错误分支。
            raise ValueError("buffer 维度必须为正")  # 输入或数据不满足要求时抛出异常并说明原因。

        self.num_envs = int(num_envs)  # 保存并行环境数量。
        self.capacity = int(capacity_steps)  # 保存环形缓冲区的时间步容量。
        self.state_dim = int(state_dim)  # 保存状态维度。
        self.action_dim = int(action_dim)  # 保存动作维度。
        self.device = torch.device(device)  # 将设备参数转换为 PyTorch 设备对象。
        if contexts.ndim != 2 or contexts.shape[0] != num_envs:  # 检查上下文是否为二维且每个环境恰有一行。
            raise ValueError("contexts 必须为 [N, Dc]")  # 输入或数据不满足要求时抛出异常并说明原因。

        # 当前版本 context_id 就是环境编号。
        self.contexts = (  # 创建缓冲区独立持有的上下文张量。
            contexts.detach()  # 切断上下文与原计算图的关联。
            .to(device=self.device, dtype=torch.float32)  # 将上下文移到目标设备并统一为 float32。
            .clone()  # 复制数据，避免外部原地修改影响缓冲区。
        )  # 结束当前多行表达式或函数调用。
        shape = (self.capacity, self.num_envs)  # 定义存储的前两维：[时间槽位, 环境]。

        self.command_dim = int(command_dim)  # 保存命令向量维度。
        if self.command_dim < 0:
            raise ValueError("command_dim 不能为负")
        self.commands = torch.zeros(
            *shape,
            self.command_dim,
            device=self.device,
        )

        self.state = torch.empty(  # 分配转移起始状态存储，写入前内容未初始化。
            *shape, self.state_dim,  # 状态存储形状为 [容量, N, 状态维度]。
            device=self.device,  # 在缓冲区指定的设备上分配或操作张量。
        )  # 结束当前多行表达式或函数调用。
        self.next_state = torch.empty_like(self.state)  # 分配与起始状态同形状的真实后继状态存储。

        self.action = torch.empty(  # 分配动作存储，写入前内容未初始化。
            *shape, self.action_dim,  # 动作存储形状为 [容量, N, 动作维度]。
            device=self.device,  # 在缓冲区指定的设备上分配或操作张量。
        )  # 结束当前多行表达式或函数调用。

        self.rewards = torch.empty(*shape, device=self.device)  # 为每个时间槽位和环境存储标量奖励。
        self.terminated = torch.empty(*shape, device=self.device, dtype=torch.bool)  # 存储任务终止标志。
        self.truncated = torch.empty_like(self.terminated)  # 存储超时等原因导致的截断标志。

        self.episode_ids = torch.empty(*shape, device=self.device, dtype=torch.long)  # 记录每条转移所属的回合编号。
        self.step_ids = torch.empty_like(self.episode_ids)  # 记录每条转移在回合内的步编号。

        self._episode = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)  # 初始化各环境当前回合编号为零。
        self._step = torch.zeros_like(self._episode)  # 初始化各环境回合内步编号为零。

        # 累计写入了多少个“全环境控制步”。
        self.total_steps = 0  # 初始化累计写入的全环境控制步数。

        self.extra_fields = (
            {}
            if extra_fields is None
            else dict(extra_fields)
        )
        self.extra_buffers = {}

        for path, field_spec in self.extra_fields.items():
            if (
                not isinstance(path, tuple)
                or not path
                or not all(
                    isinstance(name, str) and name
                    for name in path
                )
            ):
                raise ValueError("附加字段路径必须为非空字符串元组")

            field_shape, dtype = field_spec
            field_shape = tuple(field_shape)

            if any(
                not isinstance(size, int) or size < 1
                for size in field_shape
            ):
                raise ValueError(f"附加字段形状不合法：{path}")

            self.extra_fields[path] = (
                field_shape,
                dtype,
            )
            # 为每个附加字段分配环形存储，形状为 [容量, N, *field_shape]。
            self.extra_buffers[path] = torch.empty(
                self.capacity,
                self.num_envs,
                *field_shape,
                dtype=dtype,
                device=self.device,
            )

    @property  # 将下方方法暴露为只读属性。
    def size_steps(self):  # 返回当前仍被缓冲区保留的有效全环境控制步数。
        '''
        返回缓冲区当前保留的有效时间步数。

        取累计写入步数与容量的较小值，结果以全环境控制步为单位，
        而不是所有环境的转移条数之和；缓冲区写满后该值保持为容量。
        '''
        return min(self.total_steps, self.capacity)  # 有效时间步数不超过缓冲区容量。

    @torch.no_grad()  # 关闭下方方法中的梯度记录。
    def mark_reset(self, env_ids):  # 记录未通过终止或截断转移表达的手动环境重置。
        '''
        记录未通过 add 的结束标志表达的手动环境重置。

        将 env_ids 转换为设备上的长整型索引，为指定环境递增回合编号，
        并将回合内步编号归零，使后续写入的数据与旧回合区分开。
        该操作只更新计数，不清空已存转移；已由 terminated 或 truncated
        表达的结束无需再次调用本方法，以免重复递增回合编号。
        '''
        ids = torch.as_tensor(  # 将手动重置的环境编号转换为索引张量。
            env_ids, device=self.device, dtype=torch.long,  # 使用目标设备上的长整型环境索引。
        )  # 结束当前多行表达式或函数调用。
        self._episode[ids] += 1  # 为手动重置的环境开启新回合。
        self._step[ids] = 0  # 将这些环境的回合内步编号归零。

    def _check(self, value, shape, name):  # 校验一个输入张量的形状，不匹配时报告字段名和形状。
        '''
        检查单个输入张量是否具有指定形状。

        value 为待检查张量，shape 为期望的形状元组，name 为字段名称。
        形状一致时正常返回 None；不一致时抛出 ValueError，
        错误信息包含字段名称、期望形状和实际形状，便于定位输入问题。
        '''
        if tuple(value.shape) != shape:  # 比较输入张量的实际形状与要求的形状。
            raise ValueError(f"{name}: 期望 {shape}，实际 {tuple(value.shape)}")  # 输入或数据不满足要求时抛出异常并说明原因。

    def _validate_extra_fields(self, transition):
        """先检查全部字段，避免写入一半才发现错误。
        transition 中的额外字段
            → 逐个取出并校验
            → 整理成 {字段路径: 数据} 的 values 字典
            → 返回
        """

        values = {}

        for path, spec in self.extra_fields.items():
            field_shape, dtype = spec
            # 变量 value 指向 transition 所指向的同一个字典，用于逐级索引嵌套字段。
            value = transition

            for name in path:
                if (
                    not isinstance(value, dict)
                    or name not in value
                ):
                    raise ValueError(f"转移缺少附加字段：{'.'.join(path)}")
                value = value[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"附加字段必须为张量：{'.'.join(path)}")

            expected_shape = (
                self.num_envs,
                *field_shape,
            )

            self._check(
                value,
                expected_shape,
                ".".join(path),
            )

            if value.dtype != dtype:
                raise ValueError(
                    f"附加字段类型错误：{'.'.join(path)}，"
                    f"期望 {dtype}，实际 {value.dtype}"
                )

            if (
                value.is_floating_point()
                and not bool(
                    torch.isfinite(value).all()
                )
            ):
                raise ValueError(
                    f"附加字段包含 NaN/Inf："
                    f"{'.'.join(path)}"
                )

            values[path] = value

        return values

    def _sample_extra_fields(self, slots, env_grid):
        """用相同索引读取附加字段，构造嵌套字典。
        slots 选择时间，env_grid 选择环境
        """

        result = {}

        for path, storage in self.extra_buffers.items():
            # target 都重新指向最外层的 result
            target = result

            # 逐级索引嵌套字典，确保每一级都是字典，最后一级是张量。
            for name in path[:-1]:
                target = target.setdefault(
                    name,
                    {},
                )

            target[path[-1]] = (
                storage[slots, env_grid].clone()
            )

        return result

    def add(self, transition):  # 校验并写入一次所有环境的转移，再更新各环境的回合计数。
        '''
        向环形缓冲区写入所有并行环境在同一控制步的转移。

        transition 必须包含 state、next_state、action、reward、terminated
        和 truncated；先校验各字段形状及结束标志的布尔类型，再将脱离
        计算图的数据复制到当前槽位，并保存转移所属的回合和步编号。
        随后推进步计数，为终止或截断的环境递增回合编号并重置步编号。
        next_state 应由调用方提供真实后继状态，结束时也不能用重置状态替代。
        '''
        n = self.num_envs  # 读取并行环境数量以构造输入形状。

        shapes = {  # 列出一次全环境转移所需各字段的形状。
            "state": (n, self.state_dim),  # 每个环境提供一个起始状态。
            "next_state": (n, self.state_dim),  # 每个环境提供一个真实后继状态。
            "action": (n, self.action_dim),  # 每个环境提供一个动作向量。
            "reward": (n,),  # 每个环境提供一个标量奖励。
            "terminated": (n,),  # 每个环境提供一个任务终止标志。
            "truncated": (n,),  # 每个环境提供一个截断标志。
        }  # 结束当前字典。

        for name, shape in shapes.items():  # 逐项检查所有必需字段的形状。
            self._check(transition[name], shape, name)  # 校验当前字段，缺失字段会触发键错误。
        for name in ("terminated", "truncated"):  # 依次检查两类结束标志。
            if transition[name].dtype != torch.bool:  # 结束标志必须使用布尔类型。
                raise ValueError(f"{name} 必须为布尔型")  # 输入或数据不满足要求时抛出异常并说明原因。

        commands = None
        if self.command_dim > 0:
            commands = transition.get("commands")
            if commands is None:
                raise ValueError("当前 buffer 要求提供 commands")
            self._check(
                commands,
                (n, self.command_dim),
                "commands",
            )
            if not commands.is_floating_point():
                raise ValueError("commands 必须为浮点张量")
            if not bool(torch.isfinite(commands).all()):
                raise ValueError("commands 包含 NaN/Inf")

        # 所有校验必须发生在写入和推进计数之前。
        extra_values = (
            self._validate_extra_fields(
                transition
            )
        )

        index = self.total_steps % self.capacity  # 通过取模计算本次写入的环形槽位。
        # 保存本次真实转移。
        for name, destination in (  # 遍历输入字段与对应存储张量的映射。
            ("state", self.state),
            ("next_state", self.next_state),
            ("action", self.action),
            ("reward", self.rewards),
            ("terminated", self.terminated),
            ("truncated", self.truncated),
        ):  # 结束参数或遍历项列表，开始执行对应代码块。
            destination[index].copy_(transition[name].detach())  # 脱离计算图后将字段复制到当前时间槽位。

        if commands is not None:
            self.commands[index].copy_(commands.detach())  # 脱离计算图后将 commands 复制到当前时间槽位。

        for path, value in extra_values.items():
            self.extra_buffers[path][index].copy_(
                value.detach()
            )

        # 保存的是本次转移所属旧回合的编号。
        self.episode_ids[index].copy_(self._episode)  # 在更新回合计数前保存本次转移的回合编号。
        self.step_ids[index].copy_(self._step)  # 保存本次转移在回合内的步编号。

        done = (transition["terminated"] | transition["truncated"]).to(self.device)  # 合并终止和截断标志，并移到缓冲区设备。
        self._step += 1  # 所有环境的回合内步计数前进一步。
        self._episode += done.long()  # 结束的环境进入下一回合编号。
        self._step[done] = 0  # 将结束环境的下一回合步计数归零。

        self.total_steps += 1  # 累计一次全环境控制步写入。

    def _valid_starts(self, horizon):  # 枚举指定长度的有效窗口，要求时间连续且不跨回合。
        '''
        查找指定窗口长度下所有时间连续且不跨回合的起点。

        horizon 必须位于 [1, capacity]；使用绝对时间枚举仍被保留的起点，
        再映射到环形槽位，按环境检查回合编号一致、步编号连续，
        并要求除最后一条转移以外均未终止或截断。
        返回绝对起点 starts 及有效的 [起点索引, 环境编号] 组合 candidates；
        数据不足或没有有效窗口时抛出 RuntimeError。
        当前实现存在 first_step 与 first_steps 命名不一致的问题，
        会在连续性检查时触发 NameError。
        '''
        if not 1 <= horizon <= self.capacity:  # 窗口长度必须为正且不超过存储容量。
            raise ValueError("horizon 必须在 [1, capacity]")  # 输入或数据不满足要求时抛出异常并说明原因。

        count = self.size_steps - horizon + 1  # 计算当前保留数据中可能的时间起点数量。
        if count <= 0:  # 现有数据不足以组成指定长度的窗口。
            raise RuntimeError("buffer 中没有足够的连续数据")  # 输入或数据不满足要求时抛出异常并说明原因。

        # 绝对时间编号避免跨越环形覆盖边界。
        oldest = self.total_steps - self.size_steps  # 计算仍被保留的最早绝对时间步。
        starts = torch.arange(oldest, oldest + count, device=self.device)  # 枚举不越过最新已写入数据的绝对起点。

        first_slots = starts % self.capacity  # 将各绝对起点转换为环形存储槽位。

        valid = torch.ones(  # 初始化每个起点与环境组合的有效性掩码。
            count, self.num_envs,  # 掩码形状为 [候选起点数量, N]。
            device=self.device, dtype=torch.bool,  # 使用目标设备上的布尔掩码。
        )  # 结束当前多行表达式或函数调用。

        first_episode = self.episode_ids[first_slots]  # 读取每个候选起点所属的回合编号。
        first_steps = self.step_ids[first_slots]  # 读取每个候选起点的回合内步编号。

        for offset in range(horizon):  # 逐个检查窗口中的时间偏移。
            slots = (starts + offset) % self.capacity  # 计算当前偏移对应的环形槽位。
            valid &= (  # 将当前连续性条件合并到有效性掩码。
                self.episode_ids[slots] == first_episode  # 要求窗口内所有转移属于起点所在回合。
            )  # 结束当前多行表达式或函数调用。
            valid &= (  # 将当前连续性条件合并到有效性掩码。
                self.step_ids[slots] == first_steps + offset  # 要求回合内步编号连续；原代码 first_step 未定义，此处保留原逻辑。
            )  # 结束当前多行表达式或函数调用。

            # 最后一条转移允许终止，前面的不允许。
            if offset < horizon - 1:  # 仅对窗口最后一条之前的转移检查结束标志。
                valid &= ~(  # 排除已经终止或截断的中间转移。
                    self.terminated[slots]  # 读取当前转移的任务终止标志。
                    | self.truncated[slots]  # 合并当前转移的截断标志。
                )  # 结束当前多行表达式或函数调用。

        candidates = valid.nonzero(as_tuple=False)  # 提取有效的 [起点索引, 环境编号] 组合。
        if candidates.shape[0] == 0:  # 检查是否存在满足连续性要求的窗口。
            raise RuntimeError("没有不跨回合的有效窗口")  # 输入或数据不满足要求时抛出异常并说明原因。

        return starts, candidates  # 返回绝对起点列表和有效组合索引。

    @torch.no_grad()  # 关闭下方方法中的梯度记录。
    def sample_windows(self, batch_size, horizon=8):  # 有放回地采样连续转移窗口，返回状态、动作和回合元数据。
        '''
        从有效的起点与环境组合中有放回地均匀采样连续窗口。

        batch_size 为正的批量大小，horizon 为每个窗口包含的转移数量。
        先筛选有效窗口，再读取动作、奖励和结束标志；状态序列由首个
        起始状态和每条转移的真实后继状态拼接，长度为 horizon + 1。
        返回字典中 state 形状为 [B, H+1, Ds]，action 为 [B, H, Da]，
        reward、terminated 和 truncated 为 [B, H]，context 为 [B, Dc]；
        同时返回形状为 [B] 的上下文编号、窗口起始回合编号和步编号。
        本方法不记录梯度，并会传播有效窗口检查中的异常。
        '''
        if batch_size <= 0:  # 采样批量大小必须为正。
            raise ValueError("batch_size 必须为正")  # 输入或数据不满足要求时抛出异常并说明原因。

        starts, candidates = self._valid_starts(horizon)  # 枚举当前缓冲区中不跨回合的有效窗口。
        selection = torch.randint(  # 有放回地均匀抽取有效组合的索引。
            candidates.shape[0],  # 随机索引的上界为有效组合数量，不包含上界。
            (batch_size,),  # 生成 batch_size 个随机索引。
            device=self.device,  # 在缓冲区指定的设备上分配或操作张量。
        )  # 结束当前多行表达式或函数调用。

        chosen = candidates[selection]  # 取出选中的起点与环境组合。
        absolute_starts = starts[chosen[:, 0]]  # 将候选起点索引转换为绝对时间步。
        env_ids = chosen[:, 1]  # 读取各采样窗口对应的环境编号。

        offsets = torch.arange(horizon, device=self.device)  # 生成窗口内从零开始的时间偏移。
        slots = (absolute_starts[:, None] + offsets[None, :]) % self.capacity  # 广播生成 [批量, 窗口长度] 的环形槽位索引。
        env_grid = env_ids[:, None]  # 增加环境索引维度以沿窗口时间轴广播。

        before = self.state[slots, env_grid]  # 读取每条采样转移的起始状态。
        following = self.next_state[slots, env_grid]  # 读取每条采样转移保存的真实后继状态。

        # 注意：最后一个状态来自真实 next_state，
        # 不能拿下一条转移的 reset 后 state 替代。
        states = torch.cat(  # 拼接出长度为 horizon + 1 的状态序列。
            (before[:, :1], following),  # 使用窗口首状态以及全部真实后继状态。
            dim=1,  # 沿时间维度拼接。
        )  # 结束当前多行表达式或函数调用。
        first_slot = slots[:, 0]  # 读取每个采样窗口的首槽位。

        result = {
            "states": states,  # 状态形状为 [批量, horizon + 1, 状态维度]。
            "actions": self.action[slots, env_grid],  # 读取窗口内动作，形状为 [批量, horizon, 动作维度]。
            "rewards": self.rewards[slots, env_grid],  # 读取窗口内奖励，形状为 [批量, horizon]。
            "terminated": self.terminated[slots, env_grid],  # 读取窗口内任务终止标志。
            "truncated": self.truncated[slots, env_grid],  # 读取窗口内截断标志。
            "commands": self.commands[slots, env_grid].clone(),  # 读取窗口内命令，形状为 [批量, horizon, 命令维度]。
            "context": self.contexts[env_ids].clone(),  # 返回每个窗口对应环境的独立上下文副本。
            "context_id": env_ids.clone(),  # 以环境编号作为上下文编号并复制返回。
            "episode_id": self.episode_ids[first_slot, env_ids].clone(),  # 返回各窗口首条转移所属的回合编号。
            "step_id": self.step_ids[first_slot, env_ids].clone(),  # 返回各窗口在回合内的起始步编号。
        }  # 结束当前字典。

        result.update(
            self._sample_extra_fields(
                slots,
                env_grid,
            )
        )

        return result

    @torch.no_grad()  # 关闭下方方法中的梯度记录。
    def sample_initial_states(self, batch_size):  # 采样已存转移的起始状态及元数据，并非仅采样回合初始状态。
        '''
        采样已存转移的起始状态及其上下文和回合元数据。

        通过 sample_windows 采样 batch_size 个长度为一的窗口，
        取每个窗口的首状态，返回 state、context、context_id、episode_id
        和 step_id，其中 state 的形状为 [B, Ds]。
        这里的起始状态是任意已存转移的起点，不保证是回合第零步。
        本方法不记录梯度，并会传播窗口采样中的参数或数据异常。
        '''
        batch = self.sample_windows(batch_size=batch_size, horizon=1)  # 通过长度为一的窗口采样任意已存转移的起始状态。

        return {  # 组装并返回采样结果字典。
            "state": batch["states"][:, 0],  # 取出各窗口的首状态；不保证是回合第零步。
            "context": batch["context"],  # 传递对应环境的上下文。
            "context_id": batch["context_id"],  # 传递对应的上下文编号。
            "episode_id": batch["episode_id"],  # 传递状态所属的回合编号。
            "step_id": batch["step_id"],  # 传递状态在回合内的步编号。
        }  # 结束当前字典。

    @torch.no_grad()
    def sample_policy_histories(
        self,
        batch_size,
        history_length=5,
    ):
        # 采样用于策略训练的历史状态序列及对应的上下文和命令。
        batch = self.sample_windows(
            batch_size=batch_size,
            horizon=history_length,
        )

        result = {
            "history": batch["states"][:, :-1],
            "context": batch["context"],
            "commands": batch["commands"][:, -1],
        }

        if "task_memory" in batch:
            # history 最后一帧是最后一条转移的步前状态，
            # 对应的必须也是该条转移的步前任务记忆。
            result["task_memory"] = {
                name: value[:, -1].clone()
                for name, value
                in batch["task_memory"].items()
            }

        return result