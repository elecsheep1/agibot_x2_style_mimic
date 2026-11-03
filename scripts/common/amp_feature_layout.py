"""AMP feature layouts for current 175D and archived 233D diagnostic captures."""


def feature_blocks(state_dim, body_count=11, joint_count=29):
    blocks = {
        "root_height": (0, 1), "root_gravity": (1, 4),
        "root_linear": (4, 7), "root_angular_scaled": (7, 10),
    }
    offset = 10
    if state_dim == 10 + 2 * joint_count + 15 * body_count:
        blocks["joint_position"] = (offset, offset + joint_count)
        offset += joint_count
        blocks["joint_velocity_scaled"] = (offset, offset + joint_count)
        offset += joint_count
    elif state_dim != 10 + 15 * body_count:
        raise ValueError(f"Unsupported AMP layout: {state_dim}D, {body_count} bodies, {joint_count} joints")
    for name, width in (("body_position", 3), ("body_rotation6d", 6),
                        ("body_linear", 3), ("body_angular_scaled", 3)):
        end = offset + width * body_count
        blocks[name] = (offset, end)
        offset = end
    if offset != state_dim:
        raise ValueError("AMP blocks do not cover the feature vector")
    return blocks
