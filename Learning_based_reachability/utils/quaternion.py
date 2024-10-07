"""
Created by Zeyuan in Dec. 14th 2023. Copied from pytorch3D since there is no conda implementation 
"""
import torch
import numpy as np


def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    qx = torch.sin(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) - \
        torch.cos(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)
    qy = torch.cos(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2) + \
        torch.sin(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2)
    qz = torch.cos(roll/2) * torch.cos(pitch/2) * torch.sin(yaw/2) - \
        torch.sin(roll/2) * torch.sin(pitch/2) * torch.cos(yaw/2)
    qw = torch.cos(roll/2) * torch.cos(pitch/2) * torch.cos(yaw/2) + \
        torch.sin(roll/2) * torch.sin(pitch/2) * torch.sin(yaw/2)

    return torch.cat([qw[..., None], qx[..., None], qy[..., None], qz[..., None]], dim=-1)


def get_euler_from_quaternion(w, x, y, z):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, min=-1, max=1)

    pitch_y = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    return torch.cat([roll_x[..., None], pitch_y[..., None], yaw_z[..., None]], dim=-1)
