from dataclasses import dataclass  # 导入数据类装饰器，用于声明上下文结构规格。

import torch  # 导入 PyTorch，执行张量拼接、拆分和数值检查。


@dataclass(frozen=True)  # 生成初始化等数据类方法，并禁止常规字段赋值。
class ContextSpec:  # 定义刚体、关节和碰撞形状的上下文布局信息。
    body_names:tuple[str, ...]  # 按上下文存储顺序记录刚体名称。
    joint_names:tuple[str, ...]  # 按上下文存储顺序记录关节名称。
    num_shapes:int  # 记录碰撞形状数量。

    @property  # 将下方方法作为只读属性访问。
    def num_bodies(self):  # 返回上下文规格中的刚体数量。
        '''
        返回上下文规格中的刚体数量。

        通过 body_names 的长度确定刚体数，用于质量和惯量字段的形状计算。
        '''
        return len(self.body_names)  # 由刚体名称数量得到刚体数。

    @property  # 将下方方法作为只读属性访问。
    def num_joints(self):  # 返回上下文规格中的关节数量。
        '''
        返回上下文规格中的关节数量。

        通过 joint_names 的长度确定关节数，用于刚度和阻尼字段的形状计算。
        '''
        return len(self.joint_names)  # 由关节名称数量得到关节数。

    @property  # 将下方方法作为只读属性访问。
    def dim(self):  # 计算上下文向量的特征维度。
        # 每个碰撞形状：静摩擦、动摩擦、恢复系数。
        # 每个刚体：质量 + 3x3惯量。
        # 每个关节：刚度、阻尼。
        '''
        计算上下文向量的特征维度。

        当前公式为三个材质参数乘以碰撞形状数、十乘以刚体数，
        再加两个关节参数乘以关节数，与物理参数打包后的总宽度一致。
        '''
        return (  # 开始计算并返回上下文维度或字段副本。
            3 * self.num_shapes  # 每个碰撞形状包含三个材质参数。
            + 10 * self.num_bodies  # 每个刚体包含一个质量和九个惯量特征，共十维。
            + 2 * self.num_joints  # 每个关节包含刚度和阻尼两个参数。
        )  # 结束当前多行表达式或调用参数组。


def pack_context(  # 将多环境的物理参数打包为二维上下文张量。
    material,  # 按环境和碰撞形状排列的三个材质参数。
    mass,  # 按环境和刚体排列的质量。
    inertia,  # 按环境和刚体排列的惯量矩阵。
    stiffness,  # 按环境和关节排列的刚度。
    damping,  # 按环境和关节排列的阻尼。
    spec: ContextSpec,  # 描述字段数量和排列顺序的上下文规格。
):  # 结束条件或函数参数列表，开始执行代码块。
    '''
    将多环境的物理参数打包为二维上下文张量。

    输入依次为材质 [N, S, 3]、质量 [N, L]、惯量 [N, L, 9]、
    刚度 [N, J]、阻尼 [N, J]，spec 描述形状数、刚体数和关节数。
    检查字段形状和设备一致性后，保留环境维，按上述顺序展平并拼接。
    返回形状为 [N, 3S + 10L + 2J] 的张量；校验失败时抛出 ValueError。
    '''
    if mass.ndim != 2:  # 要求质量张量具有环境和刚体两个维度。
        raise ValueError("mass 必须为 [N, L]")  # 输入、控制模式或物理参数不符合要求时抛出异常。


    num_envs = mass.shape[0]  # 由质量张量读取并行环境数量。

    expected = {  # 定义各输入字段必须满足的形状。
        "material": (num_envs, spec.num_shapes, 3),  # 材质形状为 [环境数, 碰撞形状数, 三个材质参数]。
        "mass": (num_envs, spec.num_bodies),  # 质量形状为 [环境数, 刚体数]。
        "inertia": (num_envs, spec.num_bodies, 9),  # 惯量形状为 [环境数, 刚体数, 九个矩阵元素]。
        "stiffness": (num_envs, spec.num_joints),  # 刚度形状为 [环境数, 关节数]。
        "damping": (num_envs, spec.num_joints),  # 阻尼形状为 [环境数, 关节数]。
    }  # 结束字段字典。
    fields = {  # 将字段名称与输入张量关联，便于统一校验。
        "material": material,  # 保存材质张量。
        "mass": mass,  # 保存质量张量。
        "inertia": inertia,  # 保存惯量张量。
        "stiffness": stiffness,  # 保存关节刚度张量。
        "damping": damping,  # 保存关节阻尼张量。
    }  # 结束字段字典。

    for name, value in fields.items():  # 逐一校验所有上下文字段。
        if tuple(value.shape) != expected[name]:  # 比较当前字段的实际形状与要求的形状。
            raise ValueError(f"{name}: 期望 {expected[name]}，实际 {tuple(value.shape)}")  # 输入、控制模式或物理参数不符合要求时抛出异常。
        if value.device != mass.device:  # 要求所有字段与质量张量位于相同设备。
            raise ValueError("context 各字段必须在同一设备")  # 输入、控制模式或物理参数不符合要求时抛出异常。

    return torch.cat(  # 沿特征维拼接所有物理参数，返回每个环境的上下文向量。
        [  # 开始列出按固定顺序拼接的张量。
            material.flatten(1),  # 保留环境维，将碰撞形状和材质参数展平。
            mass,  # 按环境和刚体排列的质量。
            inertia.flatten(1),  # 保留环境维，将全部刚体的惯量矩阵展平。
            stiffness,  # 按环境和关节排列的刚度。
            damping,  # 按环境和关节排列的阻尼。
        ],  # 结束待拼接张量列表。
        dim=-1,  # 沿最后一个特征维度执行拼接或拆分。
    )  # 结束当前多行表达式或调用参数组。


def unpack_context(context, spec: ContextSpec):  # 按照上下文规格将二维向量拆分为五组物理参数。
    '''
    按照上下文规格将二维向量拆分为五组物理参数。

    先检查 context 的维数和宽度，再按材质、质量、惯量、刚度、阻尼
    的固定顺序拆分，返回包含相应字段的字典。惯量返回为 [N, L, 9]。
    返回字段形状与 pack_context 的输入要求一致，可用于重新打包。
    输入维数或宽度不符合规格时抛出 ValueError。
    '''
    if context.ndim != 2 or context.shape[1] != spec.dim:  # 检查输入是否为二维且特征宽度符合规格。
        raise ValueError("context 形状与 ContextSpec 不一致")  # 输入、控制模式或物理参数不符合要求时抛出异常。

    num_envs = context.shape[0]  # 从上下文批量维读取环境数量。

    material, mass, inertia, stiffness, damping = torch.split(  # 按打包顺序将上下文拆为五组物理参数。
        context,  # 待拆分的二维上下文张量。
        (  # 开始列出五组字段在特征维上的宽度。
            3 * spec.num_shapes,  # 每个碰撞形状占三个材质特征。
            spec.num_bodies,  # 每个刚体占一个质量特征。
            9 * spec.num_bodies,  # 每个刚体的 3×3 惯量矩阵占九个特征。
            spec.num_joints,  # 当前关节参数组中每个关节占一个特征。
            spec.num_joints,  # 当前关节参数组中每个关节占一个特征。
        ),  # 结束当前多行表达式或调用参数组。
        dim=-1,  # 沿最后一个特征维度执行拼接或拆分。
    )  # 结束当前多行表达式或调用参数组。

    return {  # 将拆分并恢复形状的字段组成字典返回。
        "material": material.reshape(num_envs, spec.num_shapes, 3),  # 恢复每个环境中各碰撞形状的三个材质参数。
        "mass": mass,  # 保存质量张量。
        "inertia": inertia.reshape(num_envs, spec.num_bodies, 9),  # 恢复每个刚体的九维惯量向量，与打包形状一致。
        "stiffness": stiffness,  # 保存关节刚度张量。
        "damping": damping,  # 保存关节阻尼张量。
    }  # 结束字段字典。


@torch.no_grad()  # 关闭读取和复制物理参数时的梯度记录。
def read_sim_context(env):  # 在启动随机化完成后读取各环境的物理参数及布局规格。
    '''
    在启动随机化完成后读取各环境的物理参数及布局规格。

    针对本项目的 PhysX 和隐式 PD 执行器，读取碰撞材质、刚体质量、
    惯量以及关节刚度和阻尼，将其复制为目标设备上的 float32 张量。
    保留碰撞形状原始顺序，打包后检查是否包含 NaN 或 Inf，
    返回 (context, spec)。材质读取可能涉及 CPU/GPU 传输，应启动时
    调用一次，避免每个控制步读取；不支持的执行器会触发异常。
    '''
    import warp as wp  # 延迟导入 Warp，用于将物理引擎数组转为 PyTorch 张量。
    from isaaclab.actuators import ImplicitActuator  # 导入隐式执行器类型以检查控制模式。

    robot = env.scene["robot"]  # 从场景中获取名为 robot 的机器人对象。

    if not all(  # 检查是否有执行器不属于隐式 PD 类型。
        isinstance(actuator, ImplicitActuator)  # 判断当前执行器是否为隐式执行器。
        for actuator in robot.actuators.values()  # 遍历机器人配置的全部执行器。
    ):  # 结束条件或函数参数列表，开始执行代码块。
        raise NotImplementedError("当前 context 读取器只支持 implicit PD")  # 输入、控制模式或物理参数不符合要求时抛出异常。

    device = robot.data.joint_pos.torch.device  # 使用机器人关节位置张量所在的计算设备。
    num_envs = env.num_envs  # 读取仿真环境的并行实例数量。

    # 保留 PhysX 返回的碰撞形状顺序，不取平均值。
    material = wp.to_torch(  # 将 PhysX 返回的材质数组转换为 PyTorch 张量。
        robot.root_view.get_material_properties()  # 读取所有碰撞形状的静摩擦、动摩擦及恢复系数。
    ).detach().to(device=device, dtype=torch.float32).clone()  # 脱离计算图、转为目标设备的 float32，并复制独立数据。

    if (  # 开始检查材质数组的维数和关键维度。
        material.ndim != 3  # 材质数组必须具有三个维度。
        or material.shape[0] != num_envs  # 第一维必须等于并行环境数量。
        or material.shape[-1] != 3  # 最后一维必须包含三个材质参数。
    ):  # 结束条件或函数参数列表，开始执行代码块。
        raise RuntimeError(f"材质数组形状不符合预期：{tuple(material.shape)}")  # 输入、控制模式或物理参数不符合要求时抛出异常。

    spec = ContextSpec(  # 依据机器人名称顺序和材质数组构建上下文规格。
        body_names=tuple(robot.body_names),  # 保存仿真器给出的刚体名称顺序。
        joint_names=tuple(robot.joint_names),  # 保存仿真器给出的关节名称顺序。
        num_shapes=material.shape[1],  # 由材质数组第二维确定碰撞形状数量。
    )  # 结束当前多行表达式或调用参数组。

    def copy_field(name):  # 复制指定机器人数据字段，得到独立的物理参数张量。
        '''
        复制指定机器人数据字段，得到独立的物理参数张量。

        name 为 robot.data 中的字段名称。通过外层作用域取得机器人
        和目标设备，将字段脱离计算图并转换为 float32 后复制返回，
        避免返回数据与仿真内部存储共享。
        '''
        return (  # 开始计算并返回上下文维度或字段副本。
            getattr(robot.data, name).torch  # 按字段名读取机器人数据，并获取其 PyTorch 张量。
            .detach()  # 断开与原计算图的关联。
            .to(device=device, dtype=torch.float32)  # 将字段转换为目标设备上的 float32 张量。
            .clone()  # 复制字段，避免与仿真内部数据共享存储。
        )  # 结束当前多行表达式或调用参数组。

    context = pack_context(  # 将读取的物理参数按固定顺序打包为上下文向量。
        material=material,  # 传入保留碰撞形状顺序的材质参数。
        mass=copy_field("body_mass"),  # 复制并传入各刚体质量。
        inertia=copy_field("body_inertia").reshape(  # 复制惯量，并将每个刚体的矩阵元素排列为九维向量。
            num_envs, spec.num_bodies, 9  # 将惯量形状设为 [环境数, 刚体数, 9]，保留原实现。
        ),  # 结束当前多行表达式或调用参数组。
        stiffness=copy_field("joint_stiffness"),  # 复制并传入各关节刚度。
        damping=copy_field("joint_damping"),  # 复制并传入各关节阻尼。
        spec=spec,  # 使用刚构建的字段布局规格。
    )  # 结束当前多行表达式或调用参数组。

    if not bool(torch.isfinite(context).all()):  # 确认所有上下文元素均为有限数值。
        raise RuntimeError("context 存在 NaN/Inf")  # 输入、控制模式或物理参数不符合要求时抛出异常。

    return context, spec  # 返回物理参数上下文及解释其布局所需的规格。
