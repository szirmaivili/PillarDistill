import math
import numpy as np
import tensorflow as tf
from center_utils import circle_nms

# ---------------------------------------------------------
# Segédfüggvények
# ---------------------------------------------------------

def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.
    Args:
        dims: (N, ndim) tensor
        origin: float or list
    Returns:
        (N, 2**ndim, ndim) tensor
    """
    ndim = int(dims.shape[1])
    if isinstance(origin, float):
        origin = [origin] * ndim

    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(np.float32)

    if ndim == 2:
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]

    corners_norm = corners_norm - np.array(origin, dtype=np.float32)
    corners_norm = tf.convert_to_tensor(corners_norm, dtype=dims.dtype)
    corners = tf.expand_dims(dims, 1) * tf.expand_dims(corners_norm, 0)
    return corners


def corners_2d(dims, origin=0.5):
    return corners_nd(dims, origin)


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    mins, maxs = [], []
    for i in range(ndim):
        mins.append(tf.reduce_min(boxes_corner[:, :, i], axis=1))
    for i in range(ndim):
        maxs.append(tf.reduce_max(boxes_corner[:, :, i], axis=1))
    return tf.stack(mins + maxs, axis=1)


# ---------------------------------------------------------
# Rotációk
# ---------------------------------------------------------

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3], angles: [N]
    sin, cos = tf.sin(angles), tf.cos(angles)
    ones, zeros = tf.ones_like(cos), tf.zeros_like(cos)

    if axis == 1:
        rot_mat_T = tf.stack([
            tf.stack([cos, zeros, -sin]),
            tf.stack([zeros, ones, zeros]),
            tf.stack([sin, zeros, cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = tf.stack([
            tf.stack([cos, -sin, zeros]),
            tf.stack([sin, cos, zeros]),
            tf.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = tf.stack([
            tf.stack([zeros, cos, -sin]),
            tf.stack([zeros, sin, cos]),
            tf.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError("axis should be 0,1,2")

    return tf.einsum("aij,jka->aik", points, rot_mat_T)


def rotate_points_along_z(points, angle):
    """points: (B, N, 3 + C), angle: (B)"""
    cosa, sina = tf.cos(angle), tf.sin(angle)
    zeros, ones = tf.zeros_like(angle), tf.ones_like(angle)
    rot_matrix = tf.stack([
        cosa, -sina, zeros,
        sina,  cosa, zeros,
        zeros, zeros, ones
    ], axis=1)
    rot_matrix = tf.reshape(rot_matrix, [-1, 3, 3])
    pts_rot = tf.matmul(points[:, :, 0:3], rot_matrix)
    return tf.concat([pts_rot, points[:, :, 3:]], axis=-1)


def rotate_points2d_along_z(points, angle):
    """points: (B, N, 2 + C), angle: (B)"""
    cosa, sina = tf.cos(angle), tf.sin(angle)
    rot_matrix = tf.stack([cosa, -sina, sina, cosa], axis=1)
    rot_matrix = tf.reshape(rot_matrix, [-1, 2, 2])
    pts_rot = tf.matmul(points[:, :, 0:2], rot_matrix)
    return tf.concat([pts_rot, points[:, :, 2:]], axis=-1)


def rotation_2d(points, angles):
    sin, cos = tf.sin(angles), tf.cos(angles)
    rot_mat_T = tf.stack([
        tf.stack([cos, -sin]),
        tf.stack([sin,  cos])
    ])
    return tf.einsum("aij,jka->aik", points, rot_mat_T)


# ---------------------------------------------------------
# Box átalakítások
# ---------------------------------------------------------

def center_to_corner_box3d(centers, dims, angles, origin=(0.5, 0.5, 0.5), axis=1):
    corners = corners_nd(dims, origin=origin)  # [N, 8, 3]
    corners = rotation_3d_in_axis(corners, angles, axis=axis)
    return corners + tf.expand_dims(centers, 1)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    corners = corners_nd(dims, origin=origin)  # [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    return corners + tf.expand_dims(centers, 1)


def get_dense_roi_grid_points(dims, grid_size):
    batch_size = tf.shape(dims)[0]
    dense_idx = tf.where(tf.ones(grid_size))
    dense_idx = tf.cast(tf.expand_dims(dense_idx, 0), tf.float32)
    dense_idx = tf.tile(dense_idx, [batch_size, 1, 1])

    dims_exp = tf.expand_dims(dims, 1)
    roi_grid = (dense_idx + 0.5) / tf.constant(grid_size, dtype=tf.float32) * dims_exp - (dims_exp / 2.0)
    return roi_grid


def center_to_grid_box2d(centers, dims, angles=None, grid_size=(6, 6)):
    corners = get_dense_roi_grid_points(dims, grid_size)
    if angles is not None:
        corners = rotation_2d(corners, angles)
    return corners + tf.expand_dims(centers, 1)


# ---------------------------------------------------------
# Koordináta transzformációk
# ---------------------------------------------------------

def project_to_image(points_3d, proj_mat):
    batch_shape = tf.shape(points_3d)[:-1]
    ones = tf.ones(list(batch_shape) + [1], dtype=points_3d.dtype)
    points_4 = tf.concat([points_3d, ones], axis=-1)
    point_2d = tf.matmul(points_4, proj_mat, transpose_b=True)
    return point_2d[..., :2] / point_2d[..., 2:3]


def camera_to_lidar(points, r_rect, velo2cam):
    num_points = tf.shape(points)[0]
    ones = tf.ones((num_points, 1), dtype=points.dtype)
    points_h = tf.concat([points, ones], axis=-1)
    mat = tf.linalg.inv(tf.matmul(r_rect, velo2cam))
    lidar_pts = tf.matmul(points_h, mat, transpose_b=True)
    return lidar_pts[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    num_points = tf.shape(points)[0]
    ones = tf.ones((num_points, 1), dtype=points.dtype)
    points_h = tf.concat([points, ones], axis=-1)
    cam_pts = tf.matmul(points_h, tf.matmul(r_rect, velo2cam), transpose_b=True)
    return cam_pts[..., :3]


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[..., 0:3]
    l, h, w, r = data[..., 3:4], data[..., 4:5], data[..., 5:6], data[..., 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return tf.concat([xyz_lidar, w, l, h, r], axis=-1)


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[..., 0:3]
    w, l, h, r = data[..., 3:4], data[..., 4:5], data[..., 5:6], data[..., 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return tf.concat([xyz, l, h, w, r], axis=-1)

def rotate_nms_pcdet(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """
    :param boxes: (N, 7) [x, y, z, l, w, h, theta]
    :param scores: (N,)
    :param thresh: float (distance threshold for circle_nms)
    """
    # transform back to pcdet's coordinate
    boxes = tf.concat(
        [boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3],
         boxes[:, 4:5], boxes[:, 3:4], boxes[:, 5:6],
         -boxes[:, -1:] - math.pi / 2],
        axis=-1
    )

    def _circle_nms_np(dets_np):

        return circle_nms(dets_np, thresh).astype(np.int32)

    # rendezés score szerint
    order = tf.argsort(scores, direction="DESCENDING")
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = tf.gather(boxes, order)
    scores = tf.gather(scores, order)

    if tf.shape(boxes)[0] == 0:
        return tf.constant([], dtype=tf.int32)

    # (x, y, score) → circle_nms input
    dets = tf.stack([boxes[:, 0], boxes[:, 1], scores], axis=-1)
    dets_np = dets.numpy()  # circle_nms numba függvény numpy-t vár

    keep = circle_nms(dets_np, thresh=thresh)

    selected = tf.gather(order, keep)
    if post_max_size is not None:
        selected = selected[:post_max_size]

    return selected