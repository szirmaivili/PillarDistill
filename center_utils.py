import numpy as np
import tensorflow as tf

def circle_nms(dets, thresh): 
    x1 = dets[:, 0] 
    y1 = dets[:, 1] 
    scores = dets[:, 2] 
    order = scores.argsort()[::-1]
    #order = scores.argsort()[::-1] # highest->lowest 
    ndets = dets.shape[0]
    suppressed = np.zeros(ndets, dtype=np.int32) 
    keep = [] 
    for _i in range(ndets): 
        i = order[_i] # start with highest score box 
        if suppressed[i] == 1: # if any box have enough iou with this, remove it 
            continue 
        keep.append(i) 
        for _j in range(_i + 1, ndets): 
            j = order[_j] 
            if suppressed[j] == 1: 
                continue # calculate center distance between i and j box 
            dist = (x1[i]-x1[j])**2 + (y1[i]-y1[j])**2 # ovr = inter / areas[j] 
            if dist <= thresh: suppressed[j] = 1 
    return keep


def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size
    a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _gather_feat(feat, ind, mask=None):
    dim = tf.shape(feat)[2]
    ind = tf.squeeze(ind, axis=0)
    ind = tf.reshape(ind, [tf.shape(feat)[0], -1])  # erőszakosan (B, M)
    ind = tf.expand_dims(ind, -1)
    ind = tf.tile(ind, [1, 1, dim])
    feat = tf.gather(feat, ind, batch_dims=1)
    if mask is not None:
        mask = tf.expand_dims(mask, 2)
        mask = tf.tile(mask, [1, 1, dim])
        feat = tf.boolean_mask(feat, mask)
        feat = tf.reshape(feat, [-1, dim])
    return feat


def _transpose_and_gather_feat(feat, ind):
    #feat = tf.transpose(feat, [0, 2, 3, 1])
    feat = tf.reshape(feat, [tf.shape(feat)[0], -1, tf.shape(feat)[-1]])
    feat = _gather_feat(feat, ind)
    return feat


def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    Args:
        boxes: (N, 3) tensor [x, y, score]
        min_radius: float
    """
    if isinstance(boxes, tf.Tensor):
        boxes_np = boxes.numpy()
    else:
        boxes_np = boxes
    keep = np.array(circle_nms(boxes_np, thresh=min_radius))[:post_max_size]
    return tf.convert_to_tensor(keep, dtype=tf.int64)


def bilinear_interpolate_tf(im, x, y):
    H = tf.shape(im)[0]
    W = tf.shape(im)[1]

    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    x0 = tf.clip_by_value(x0, 0, W - 1)
    x1 = tf.clip_by_value(x1, 0, W - 1)
    y0 = tf.clip_by_value(y0, 0, H - 1)
    y1 = tf.clip_by_value(y1, 0, H - 1)

    Ia = tf.gather_nd(im, tf.stack([y0, x0], axis=1))
    Ib = tf.gather_nd(im, tf.stack([y1, x0], axis=1))
    Ic = tf.gather_nd(im, tf.stack([y0, x1], axis=1))
    Id = tf.gather_nd(im, tf.stack([y1, x1], axis=1))

    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)

    wa = (tf.cast(x1, tf.float32) - x) * (tf.cast(y1, tf.float32) - y)
    wb = (tf.cast(x1, tf.float32) - x) * (y - tf.cast(y0, tf.float32))
    wc = (x - tf.cast(x0, tf.float32)) * (tf.cast(y1, tf.float32) - y)
    wd = (x - tf.cast(x0, tf.float32)) * (y - tf.cast(y0, tf.float32))

    return Ia * tf.expand_dims(wa, -1) + Ib * tf.expand_dims(wb, -1) + Ic * tf.expand_dims(wc, -1) + Id * tf.expand_dims(wd, -1)

def reorganize_test_cfg_for_multi_tasks(test_cfg, task_num_classes):
    def reorganize_param(param):
        if isinstance(param, (float, int)):
            return [param] * len(task_num_classes)

        assert isinstance(param, (list, tuple))
        assert len(param) == sum(task_num_classes)

        ret_list = [[] for _ in range(len(task_num_classes))]
        flag = 0
        for k, num in enumerate(task_num_classes):
            ret_list[k] = list(param[flag:flag + num])
            flag += num
        return ret_list

    if test_cfg.get('rectifier', False) is not None:
        test_cfg['rectifier'] = reorganize_param(test_cfg['rectifier'])

    test_cfg['nms']['nms_pre_max_size'] = reorganize_param(test_cfg['nms']['nms_pre_max_size'])
    test_cfg['nms']['nms_post_max_size'] = reorganize_param(test_cfg['nms']['nms_post_max_size'])
    test_cfg['nms']['nms_iou_threshold'] = reorganize_param(test_cfg['nms']['nms_iou_threshold'])

    return test_cfg


def center_to_corner2d(center, dim):
    corners_norm = tf.constant([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], dtype=tf.float32)
    corners = tf.expand_dims(dim, 1) * tf.expand_dims(corners_norm, 0)
    corners = corners + tf.expand_dims(center, 1)
    return corners


def bbox3d_overlaps_iou(pred_boxes, gt_boxes):
    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = tf.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = tf.maximum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = tf.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              tf.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = tf.maximum(inter_h, 0)

    inter = tf.maximum(inter_max_xy - inter_min_xy, 0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    return tf.clip_by_value(volume_inter / volume_union, 0.0, 1.0)


def bbox3d_overlaps_giou(pred_boxes, gt_boxes):
    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = tf.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = tf.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = tf.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = tf.minimum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = tf.minimum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              tf.maximum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    inter_h = tf.maximum(inter_h, 0)

    inter = tf.maximum(inter_max_xy - inter_min_xy, 0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    outer_h = tf.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              tf.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = tf.maximum(outer_h, 0)
    outer = tf.maximum(out_max_xy - out_min_xy, 0)
    closure = outer[:, 0] * outer[:, 1] * outer_h

    gious = volume_inter / volume_union - (closure - volume_union) / closure
    return tf.clip_by_value(gious, -1.0, 1.0)


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])

    inter_max_xy = tf.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = tf.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = tf.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = tf.minimum(qcorners[:, 0], gcorners[:, 0])

    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = tf.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              tf.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = tf.maximum(inter_h, 0)

    inter = tf.maximum(inter_max_xy - inter_min_xy, 0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    inter_diag = tf.reduce_sum(tf.square(gt_boxes[:, 0:3] - pred_boxes[:, 0:3]), axis=-1)

    outer_h = tf.maximum(gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]) - \
              tf.minimum(gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5])
    outer_h = tf.maximum(outer_h, 0)
    outer = tf.maximum(out_max_xy - out_min_xy, 0)
    outer_diag = tf.square(outer[:, 0]) + tf.square(outer[:, 1]) + tf.square(outer_h)

    dious = volume_inter / volume_union - inter_diag / outer_diag
    return tf.clip_by_value(dious, -1.0, 1.0)


def bboxes_overlaps_ciou_bev(pred_boxes, gt_boxes):
    cious = tf.zeros([tf.shape(pred_boxes)[0]], dtype=tf.float32)
    if tf.shape(pred_boxes)[0] == 0:
        return cious

    d = tf.reduce_sum(tf.square(gt_boxes[:, 0:2] - pred_boxes[:, 0:2]), axis=1)
    d2 = d * d
    gr = gt_boxes[:, 3]
    gr2 = gr * gr

    inside_mask = d <= tf.abs(pred_boxes[:, 3] - gr)
    idx_inside = tf.where(inside_mask)
    if tf.size(idx_inside) > 0:
        cious = tf.tensor_scatter_nd_update(
            cious, idx_inside,
            tf.minimum(pred_boxes[:, 3][inside_mask] ** 2, gr2[inside_mask]) /
            tf.maximum(pred_boxes[:, 3][inside_mask] ** 2, gr2[inside_mask])
        )

    insec_mask = (d > tf.abs(pred_boxes[:, 3] - gr)) & (d < (pred_boxes[:, 3] + gr))
    idx_insec = tf.where(insec_mask)
    if tf.size(idx_insec) > 0:
        x_t = (pred_boxes[:, 3][insec_mask] ** 2 - gr2[insec_mask] + d2[insec_mask]) / (2 * d[insec_mask])
        z_t = x_t * x_t
        y_t = tf.sqrt(tf.maximum(pred_boxes[:, 3][insec_mask] ** 2 - z_t, 0))
        overlaps = (pred_boxes[:, 3][insec_mask] ** 2 * tf.asin(tf.clip_by_value(y_t / gr[insec_mask], -1, 1)) +
                    gr2[insec_mask] * tf.asin(tf.clip_by_value(y_t / gr[insec_mask], -1, 1)) -
                    y_t * (x_t + tf.sqrt(z_t + gr2[insec_mask] - pred_boxes[:, 3][insec_mask] ** 2)))
        cious = tf.tensor_scatter_nd_update(
            cious, idx_insec,
            overlaps / (np.pi * pred_boxes[:, 3][insec_mask] ** 2 + np.pi * gr2[insec_mask] - overlaps)
        )
    return tf.clip_by_value(cious, 0.0, 1.0)

def bboxes_overlaps_ciou(pred_boxes, gt_boxes):
    """ Calculate the circle IoU in bev space formed by 3D bounding boxes """
    cious = tf.zeros([tf.shape(pred_boxes)[0]], dtype=tf.float32)
    if tf.shape(pred_boxes)[0] == 0:
        return cious

    d2 = tf.reduce_sum(tf.square(pred_boxes[:, 0:2] - gt_boxes[:, 0:2]), axis=-1)
    qr2 = tf.reduce_sum(tf.square(pred_boxes[:, 3:5]), axis=-1)
    gr2 = tf.reduce_sum(tf.square(gt_boxes[:, 3:5]), axis=-1)

    d = tf.sqrt(d2)
    qr = tf.sqrt(qr2)
    gr = tf.sqrt(gr2)

    inter_h = tf.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              tf.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = tf.maximum(inter_h, 0)

    # teljesen benne van az egyik a másikban
    inside_mask = d <= tf.abs(qr - gr)
    idx_inside = tf.where(inside_mask)
    if tf.size(idx_inside) > 0:
        overlaps_inside = tf.minimum(qr2[inside_mask], gr2[inside_mask]) * inter_h[inside_mask]
        denom_inside = qr2[inside_mask] * gt_boxes[:, 5][inside_mask] + \
                       gr2[inside_mask] * gt_boxes[:, 5][inside_mask] - overlaps_inside + 1e-4
        cious = tf.tensor_scatter_nd_update(
            cious, idx_inside, overlaps_inside / denom_inside
        )

    # részben metszik egymást
    insec_mask = (d > tf.abs(qr - gr)) & (d < (qr + gr))
    idx_insec = tf.where(insec_mask)
    if tf.size(idx_insec) > 0:
        qr_t = qr[insec_mask]
        gr_t = gr[insec_mask]
        d_t = d[insec_mask]

        qr2_t = qr_t * qr_t
        gr2_t = gr_t * gr_t
        x_t = (qr2_t - gr2_t + d_t * d_t) / (2 * d_t)
        z_t = x_t * x_t
        y_t = tf.sqrt(tf.maximum(qr2_t - z_t, 0.))

        overlaps = (qr2_t * tf.asin(tf.clip_by_value(y_t / qr_t, -1., 1.)) +
                    gr2_t * tf.asin(tf.clip_by_value(y_t / gr_t, -1., 1.)) -
                    y_t * (x_t + tf.sqrt(tf.maximum(z_t + gr2_t - qr2_t, 0.)))) * inter_h[insec_mask]

        denom_insec = np.pi * qr2_t * pred_boxes[:, 5][insec_mask] + \
                      np.pi * gr2_t * gt_boxes[:, 5][insec_mask] - overlaps + 1e-4

        cious = tf.tensor_scatter_nd_update(
            cious, idx_insec, overlaps / denom_insec
        )

    return tf.clip_by_value(cious, 0.0, 1.0)


def bboxes_overlaps_cdiou(pred_boxes, gt_boxes):
    """ Calculate the circle DIoU in bev space formed by 3D bounding boxes """
    d2 = tf.reduce_sum(tf.square(pred_boxes[:, 0:2] - gt_boxes[:, 0:2]), axis=-1)
    qr2 = tf.reduce_sum(tf.square(pred_boxes[:, 3:5]), axis=-1)
    gr2 = tf.reduce_sum(tf.square(gt_boxes[:, 3:5]), axis=-1)

    d = tf.sqrt(d2)
    qr = tf.sqrt(qr2)
    gr = tf.sqrt(gr2)

    cdious = -1.0 * d2 / tf.square(d + qr + gr)

    inter_h = tf.minimum(pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]) - \
              tf.maximum(pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5])
    inter_h = tf.maximum(inter_h, 0)

    # teljesen benne van az egyik
    inside_mask = d <= tf.abs(qr - gr)
    idx_inside = tf.where(inside_mask)
    if tf.size(idx_inside) > 0:
        overlaps_inside = tf.minimum(qr2[inside_mask], gr2[inside_mask]) * inter_h[inside_mask]
        denom_inside = qr2[inside_mask] * gt_boxes[:, 5][inside_mask] + \
                       gr2[inside_mask] * gt_boxes[:, 5][inside_mask] - overlaps_inside
        cdious = tf.tensor_scatter_nd_update(
            cdious, idx_inside,
            overlaps_inside / denom_inside - d2[inside_mask] / tf.maximum(qr2[inside_mask], gr2[inside_mask])
        )

    # részben metszik egymást
    insec_mask = (~inside_mask) & (d < (qr + gr))
    idx_insec = tf.where(insec_mask)
    if tf.size(idx_insec) > 0:
        qr_t = qr[insec_mask]
        gr_t = gr[insec_mask]
        d_t = d[insec_mask]

        d2_t = d_t * d_t
        qr2_t = qr_t * qr_t
        gr2_t = gr_t * gr_t
        x_t = (qr2_t - gr2_t + d2_t) / (2 * d_t)
        z_t = x_t * x_t
        y_t = tf.sqrt(tf.maximum(qr2_t - z_t, 0.))

        overlaps = (qr2_t * tf.asin(y_t / qr_t) +
                    gr2_t * tf.asin(y_t / gr_t) -
                    y_t * (x_t + tf.sqrt(z_t + gr2_t - qr2_t))) * inter_h[insec_mask]

        denom_insec = np.pi * qr2_t * pred_boxes[:, 5][insec_mask] + \
                      np.pi * gr2_t * gt_boxes[:, 5][insec_mask] - overlaps

        cdious = tf.tensor_scatter_nd_update(
            cdious, idx_insec,
            overlaps / denom_insec - d2_t / tf.square(d_t + qr_t + gr_t)
        )

    return tf.clip_by_value(cdious, -1.0, 1.0)

def rotate_boxes_2d(center, dims, heading):
    """
    Args:
        center: (N, 2) [x, y]
        dims: (N, 2) [dx, dy]
        heading: (N,) rotation in radians
    Returns:
        corners: (N, 4, 2)
    """
    dx = dims[:, 0] / 2.0
    dy = dims[:, 1] / 2.0

    # 4 sarkot origin körül
    corners = tf.stack([
        tf.stack([-dx, -dy], axis=1),
        tf.stack([-dx,  dy], axis=1),
        tf.stack([ dx,  dy], axis=1),
        tf.stack([ dx, -dy], axis=1)
    ], axis=1)  # (N, 4, 2)

    # forgatás mátrix
    cos_h = tf.cos(heading)
    sin_h = tf.sin(heading)
    rot_mat = tf.stack([tf.stack([cos_h, -sin_h], axis=1),
                        tf.stack([sin_h,  cos_h], axis=1)], axis=1)  # (N, 2, 2)

    # elforgatás és eltolás
    rotated = tf.einsum('nij,nkj->nki', rot_mat, corners)
    rotated = rotated + tf.expand_dims(center, axis=1)
    return rotated  # (N, 4, 2)


# def polygon_area(poly):
#     """
#     Shoelace formula for convex polygon area.
#     poly: (M, 2)
#     """
#     x = poly[:, 0]
#     y = poly[:, 1]
#     return 0.5 * tf.abs(tf.reduce_sum(x * tf.roll(y, shift=-1, axis=0) -
#                                       y * tf.roll(x, shift=-1, axis=0)))

@tf.function
def polygon_area(poly):
    """
    Compute signed polygon area with the shoelace formula.
    poly: (N, 2)
    """
    n = tf.shape(poly)[0]
    if tf.less(n, 3):
        return tf.constant(0.0, dtype=tf.float32)

    x = poly[:, 0]
    y = poly[:, 1]

    # shift by one (x_{i+1}, y_{i+1})
    x_next = tf.concat([x[1:], x[:1]], axis=0)
    y_next = tf.concat([y[1:], y[:1]], axis=0)

    area = 0.5 * tf.abs(tf.reduce_sum(x * y_next - x_next * y))
    return area



# def convex_polygon_intersection_area(poly1, poly2):
#     """
#     Compute intersection area of two convex polygons using Sutherland–Hodgman clipping.
#     poly1, poly2: (4, 2)
#     Returns:
#         area (scalar tf.float32)
#     """
#     def clip(subjectPolygon, clipEdgeStart, clipEdgeEnd):
#         outputList = []
#         A = subjectPolygon[-1]
#         for B in subjectPolygon:
#             # vektorok
#             edge = clipEdgeEnd - clipEdgeStart
#             AB = B - A
#             AP = A - clipEdgeStart
#             BP = B - clipEdgeStart

#             # oldalak meghatározása
#             crossA = edge[0] * AP[1] - edge[1] * AP[0]
#             crossB = edge[0] * BP[1] - edge[1] * BP[0]

#             if crossB >= 0:
#                 if crossA < 0:
#                     t = crossA / (crossA - crossB + 1e-9)
#                     I = A + t * AB
#                     outputList.append(I)
#                 outputList.append(B)
#             elif crossA >= 0:
#                 t = crossA / (crossA - crossB + 1e-9)
#                 I = A + t * AB
#                 outputList.append(I)
#             A = B
#         return tf.stack(outputList) if outputList else tf.zeros((0, 2))

#     outputList = poly1
#     for i in range(tf.shape(poly2)[0]):
#         clipEdgeStart = poly2[i]
#         clipEdgeEnd = poly2[(i + 1) % tf.shape(poly2)[0]]
#         if tf.shape(outputList)[0] == 0:
#             return tf.constant(0.0, dtype=tf.float32)
#         outputList = clip(outputList, clipEdgeStart, clipEdgeEnd)

#     if tf.shape(outputList)[0] < 3:
#         return tf.constant(0.0, dtype=tf.float32)
#     return polygon_area(outputList)

@tf.function
def convex_polygon_intersection_area(poly1, poly2):
    """
    Compute intersection area of two convex polygons using Sutherland–Hodgman clipping.
    poly1, poly2: (N, 2) convex poligon pontjai
    Returns:
        area (scalar tf.float32)
    """

    def clip(subjectPolygon, clipEdgeStart, clipEdgeEnd):
        output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        edge = clipEdgeEnd - clipEdgeStart

        def body(j, A, output):
            B = subjectPolygon[j]
            AB = B - A
            AP = A - clipEdgeStart
            BP = B - clipEdgeStart

            crossA = edge[0] * AP[1] - edge[1] * AP[0]
            crossB = edge[0] * BP[1] - edge[1] * BP[0]

            def write_inside(output):
                return output.write(output.size(), B)

            def write_cross(output, A=A, B=B, AB=AB, crossA=crossA, crossB=crossB):
                t = crossA / (crossA - crossB + 1e-9)
                I = A + t * AB
                output = output.write(output.size(), I)
                output = output.write(output.size(), B)
                return output

            def write_entry(output, A=A, B=B, AB=AB, crossA=crossA, crossB=crossB):
                t = crossA / (crossA - crossB + 1e-9)
                I = A + t * AB
                output = output.write(output.size(), I)
                return output

            output = tf.case([
                (crossB >= 0, lambda: tf.cond(crossA < 0, 
                                              lambda: write_cross(output), 
                                              lambda: write_inside(output)))
            ], default=lambda: tf.cond(crossA >= 0, 
                                       lambda: write_entry(output), 
                                       lambda: output))

            return j+1, B, output

        def cond(j, A, output):
            return j < tf.shape(subjectPolygon)[0]

        A0 = subjectPolygon[-1]
        _, _, output = tf.while_loop(cond, body, [0, A0, output])
        return output.stack()

    # iterálunk poly2 élein
    def body(i, out_poly):
        clipEdgeStart = poly2[i]
        clipEdgeEnd = poly2[(i+1) % tf.shape(poly2)[0]]
        out_poly = tf.cond(tf.shape(out_poly)[0] > 0,
                           lambda: clip(out_poly, clipEdgeStart, clipEdgeEnd),
                           lambda: tf.zeros((0,2), dtype=tf.float32))
        return i+1, out_poly

    def cond(i, out_poly):
        return i < tf.shape(poly2)[0]

    _, outputList = tf.while_loop(cond, body, loop_vars=[0, poly1],
        shape_invariants=[tf.TensorShape([]),tf.TensorShape([None, 2])])

    area = tf.cond(
    tf.shape(outputList)[0] < 3,
    lambda: tf.constant(0.0, tf.float32),
    lambda: polygon_area(outputList)
    )
    return area



# def boxes_aligned_overlap_bev_tf(boxes_a, boxes_b):
#     """
#     Args:
#         boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
#         boxes_b: (N, 7)
#     Returns:
#         overlaps_bev: (N, 1)
#     """
#     centers_a = boxes_a[:, :2]
#     dims_a = boxes_a[:, 3:5]
#     heading_a = boxes_a[:, 6]

#     centers_b = boxes_b[:, :2]
#     dims_b = boxes_b[:, 3:5]
#     heading_b = boxes_b[:, 6]

#     corners_a = rotate_boxes_2d(centers_a, dims_a, heading_a)
#     corners_b = rotate_boxes_2d(centers_b, dims_b, heading_b)

#     overlaps = []
#     for i in range(tf.shape(corners_a)[0]):
#         area = convex_polygon_intersection_area(corners_a[i], corners_b[i])
#         overlaps.append(area)
#     overlaps = tf.stack(overlaps)
#     return tf.expand_dims(overlaps, axis=1)

def boxes_aligned_overlap_bev_tf(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7)
    Returns:
        overlaps_bev: (N, 1)
    """
    centers_a = boxes_a[:, :2]
    dims_a = boxes_a[:, 3:5]
    heading_a = boxes_a[:, 6]

    centers_b = boxes_b[:, :2]
    dims_b = boxes_b[:, 3:5]
    heading_b = boxes_b[:, 6]

    corners_a = rotate_boxes_2d(centers_a, dims_a, heading_a)  # (N, 4, 2)
    corners_b = rotate_boxes_2d(centers_b, dims_b, heading_b)  # (N, 4, 2)

    def compute_overlap(args):
        ca, cb = args
        return convex_polygon_intersection_area(ca, cb)  # scalar

    overlaps = tf.map_fn(
        compute_overlap,
        (corners_a, corners_b),
        fn_output_signature=tf.float32
    )  # (N,)

    return tf.expand_dims(overlaps, axis=1)  # (N, 1)


def to_pcdet_tf(boxes):
    """
    Transform boxes back to pcdet's coordinate system.

    Args:
        boxes: (N, 7) tf.Tensor [x, y, z, dx, dy, dz, heading]
    Returns:
        boxes: (N, 7) tf.Tensor
    """
    # új sorrend: [0, 1, 2, 4, 3, 5, -1]
    idx = tf.constant([0, 1, 2, 4, 3, 5, 6], dtype=tf.int32)
    boxes = tf.gather(boxes, idx, axis=1)

    # heading: -heading - pi/2
    heading = boxes[:, -1]
    heading = -heading - np.pi / 2.0
    boxes = tf.concat([boxes[:, :-1], tf.expand_dims(heading, axis=1)], axis=1)

    return boxes

def boxes_aligned_iou3d_tf(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        iou3d: (N, 1)
    """
    # ellenőrzés
    #tf.debugging.assert_equal(tf.shape(boxes_a)[0], tf.shape(boxes_b)[0])
    if tf.shape(boxes_a)[0] != tf.shape(boxes_b)[0]:

        limit = tf.minimum(tf.shape(boxes_a)[0], tf.shape(boxes_b)[0])

    else:

        limit = tf.shape(boxes_a)[0]

    tf.debugging.assert_equal(tf.shape(boxes_a)[1], 7)
    tf.debugging.assert_equal(tf.shape(boxes_b)[1], 7)

    # transform back to pcdet's coordinate
    boxes_a = to_pcdet_tf(boxes_a)[:limit]
    boxes_b = to_pcdet_tf(boxes_b)[:limit]

    # height overlap
    boxes_a_height_max = tf.expand_dims(boxes_a[:, 2] + boxes_a[:, 5] / 2, axis=1)
    boxes_a_height_min = tf.expand_dims(boxes_a[:, 2] - boxes_a[:, 5] / 2, axis=1)
    boxes_b_height_max = tf.expand_dims(boxes_b[:, 2] + boxes_b[:, 5] / 2, axis=1)
    boxes_b_height_min = tf.expand_dims(boxes_b[:, 2] - boxes_b[:, 5] / 2, axis=1)

    # bev overlap (N, 1)
    # TODO: ezt neked kell implementálni TensorFlow-ban
    overlaps_bev = boxes_aligned_overlap_bev_tf(boxes_a, boxes_b)
    #overlaps_bev = tf.zeros([tf.shape(boxes_a)[0], 1], dtype=tf.float32)

    # height overlap
    max_of_min = tf.maximum(boxes_a_height_min, boxes_b_height_min)
    min_of_max = tf.minimum(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = tf.maximum(min_of_max - max_of_min, 0.0)

    # 3D overlap
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = tf.expand_dims(boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5], axis=1)
    vol_b = tf.expand_dims(boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5], axis=1)

    iou3d = overlaps_3d / tf.maximum(vol_a + vol_b - overlaps_3d, 1e-6)

    return iou3d