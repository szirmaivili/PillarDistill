import tensorflow as tf
from center_utils import (
    _transpose_and_gather_feat,
    bbox3d_overlaps_iou,
    bbox3d_overlaps_giou,
    bbox3d_overlaps_diou,
    boxes_aligned_iou3d_tf,
)


class RegLoss(tf.keras.layers.Layer):
    """Regression loss for an output tensor."""

    def __init__(self, name="RegLoss"):
        super().__init__(name=name)

    def call(self, output, mask, ind, target):
        if tf.reduce_sum(mask) == 0:
            return tf.zeros(tf.shape(target)[-1], dtype=output.dtype)

        #pred = _transpose_and_gather_feat(output, ind)
        #print(pred.shape)
        #print(mask.shape)
        mask_new = tf.tile(mask, [1, 1, 1, 10])

        loss = tf.abs(output * mask_new - target * mask_new)  # L1 loss
        loss = loss / (tf.reduce_sum(mask) + 1e-4)
        # (batch, max_objects, dim) → (dim,)
        loss = tf.reduce_sum(loss, axis=[0, 1])
        return loss


class FastFocalLoss(tf.keras.layers.Layer):
    """Reimplemented focal loss, same as CornerNet version."""

    def __init__(self, name="FastFocalLoss"):
        super().__init__(name=name)

    def call(self, out, target, ind, mask, cat):
        """
        Args:
            out, target: B x C x H x W
            ind, mask: B x M
            cat: B x M (category id for peaks)
        """

        if len(target.shape) == 3:
            target = tf.expand_dims(target, 0)  # (1, H, W, C)
        elif len(target.shape) > 4:
            # pl. (B, 6, H, W, C) esetben taskonként lesz szétválasztva a hívásnál
            pass
        mask = tf.cast(mask, tf.float32)
        gt = tf.pow(1 - target, 4)

        if tf.rank(gt) == 4:  # (B, 1, H, W)
            gt = tf.transpose(gt, [0, 2, 3, 1])        # -> (B, H, W, 1)

        gt = tf.broadcast_to(gt, tf.shape(out))  # ugyanakkora legyen, mint out

        neg_loss = tf.math.log(tf.clip_by_value(1 - out, 1e-6, 1.0)) * tf.pow(out, 2) * gt
        neg_loss = tf.reduce_sum(neg_loss)

        pos_pred_pix = _transpose_and_gather_feat(out, ind)  # B x M x C
        # gather a category id channel
        cat = tf.expand_dims(cat, -1)  # B x M x 1

        # Biztonság kedvéért alakítsuk át a cat-et megfelelő formára
        cat = tf.cast(cat, tf.int32)

        # Ha hiányzik a batch dimenzió:
        if len(cat.shape) == 1:
            cat = tf.expand_dims(cat, 0)  # (1, M)

        # Ha hiányzik a 'csatorna' dimenzió:
        if len(cat.shape) == 2:
            cat = tf.expand_dims(cat, -1)  # (B, M, 1)

        if tf.rank(pos_pred_pix) == 4:
            pos_pred_pix = tf.squeeze(pos_pred_pix, -1)  # (1, 500, 1)

        if len(cat.shape) == 3 and cat.shape[0] != tf.shape(pos_pred_pix)[0]:
            cat = tf.expand_dims(cat, 0)  # (1, 500, 1)

        # Mindig legyen (B, M, 1)
        if tf.rank(cat) == 2:      # (M, 1)
            cat = tf.expand_dims(cat, 0)
        elif tf.rank(cat) == 1:    # (M,)
            cat = tf.reshape(cat, [1, -1, 1])
        elif tf.rank(cat) == 3 and cat.shape[0] != tf.shape(pos_pred_pix)[0]:
            cat = tf.expand_dims(cat, 0)

        pos_pred = tf.gather(pos_pred_pix, cat, batch_dims=1, axis=2)  # B x M x 1
        if pos_pred.shape[-1] == 1:
            pos_pred = tf.squeeze(pos_pred, -1)
        #pos_pred = tf.squeeze(pos_pred, -1)  # B x M

        num_pos = tf.reduce_sum(mask)
        mask = tf.expand_dims(mask, -1)
        #mask = tf.reshape(mask, [tf.shape(mask)[0], -1])
        pos_loss = tf.math.log(tf.clip_by_value(pos_pred, 1e-6, 1.0)) * tf.pow(1 - pos_pred, 2) * mask
        pos_loss = tf.reduce_sum(pos_loss)

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos

    # def call(self, out, target, ind, mask, cat):
    #     """
    #     Args:
    #         out:    (B, H, W, C)   - model output logits/probs
    #         target: (B, H, W, C)   - heatmap target
    #         ind:    (B, M)         - flattened indices of positive peaks
    #         mask:   (B, M)         - mask for valid peaks
    #         cat:    (B, M)         - category ids for each peak
    #     """
    #     mask = tf.cast(mask, tf.float32)
    #     gt = tf.pow(1 - target, 4)

    #     # ---------- neg loss ----------
    #     neg_loss = tf.math.log(tf.clip_by_value(1 - out, 1e-6, 1.0)) * tf.pow(out, 2) * gt
    #     neg_loss = tf.reduce_sum(neg_loss)

    #     # ---------- pos loss ----------
    #     pos_pred_pix = _transpose_and_gather_feat(out, ind)  # (B, M, C)

    #     B = tf.shape(pos_pred_pix)[0]
    #     M = tf.shape(pos_pred_pix)[1]
    #     C = tf.shape(pos_pred_pix)[2]

    #     # batch index előállítása
    #     batch_idx = tf.tile(tf.reshape(tf.range(B), [B, 1]), [1, M])  # (B, M)
    #     batch_idx = tf.cast(batch_idx, tf.int32)

    #     # peak index előállítása (0..M-1 minden batchben)
    #     peak_idx = tf.tile(tf.reshape(tf.range(M), [1, M]), [B, 1])  # (B, M)
    #     peak_idx = tf.cast(peak_idx, tf.int32)

    #     # (B,M,3): [batch, peak, class_id]
    #     H = tf.shape(peak_idx)[0]
    #     W = tf.shape(peak_idx)[1]
    #     cat = tf.reshape(cat, [H, W])
    #     gather_idx = tf.stack([batch_idx, peak_idx, cat], axis=-1)

    #     # csak a megfelelő class logit kiválasztása
    #     pos_pred = tf.gather_nd(pos_pred_pix, gather_idx)  # (B, M)
        
    #     # mask alakra igazítása
    #     mask_new = tf.tile(mask, [1, 1, 1, C])
    #     pos_loss = tf.math.log(tf.clip_by_value(pos_pred, 1e-6, 1.0)) * tf.pow(1 - pos_pred, 2) * mask_new
    #     pos_loss = tf.reduce_sum(pos_loss)

    #     num_pos = tf.reduce_sum(mask)

    #     if tf.equal(num_pos, 0):
    #         return -neg_loss
    #     return -(pos_loss + neg_loss) / num_pos

class IouLoss(tf.keras.layers.Layer):
    """IoU prediction loss for an output tensor."""

    def __init__(self, name="IouLoss"):
        super().__init__(name=name)

    def call(self, iou_pred, mask, ind, box_pred, box_gt):
        if tf.reduce_sum(mask) == 0:
            return tf.zeros((1,), dtype=iou_pred.dtype)

        mask = tf.cast(mask, tf.bool)
        #mask = tf.reshape(mask, [mask.shape[0], -1])
        #pred = tf.boolean_mask(_transpose_and_gather_feat(iou_pred, ind), mask)
        pred_box = tf.transpose(box_pred, perm=[0, 2, 3, 1])
        #box = tf.transpose(box_pred, perm=[0, 2, 3, 1])
        mask_new = tf.tile(mask, [1, 1, 1, 7])
        mask_new_2 = tf.tile(mask, [1, 1, 1, 7])
        
        boxes_a = tf.reshape(tf.boolean_mask(pred_box, mask_new), [-1, 7])
        boxes_b = tf.reshape(tf.boolean_mask(box_gt, mask_new_2), [-1, 7])
        
        target = boxes_aligned_iou3d_tf(boxes_a, boxes_b)
        target = 2 * target - 1  # scale from [0,1] → [-1,1]

        loss = tf.reduce_sum(tf.abs(iou_pred - target))  # L1 loss
        loss = loss / (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-4)
        return loss


class IouRegLoss(tf.keras.layers.Layer):
    """Distance IoU loss for output boxes."""

    def __init__(self, type="IoU", name="IouRegLoss"):
        super().__init__(name=name)
        if type == "IoU":
            self.bbox3d_iou_func = bbox3d_overlaps_iou
        elif type == "GIoU":
            self.bbox3d_iou_func = bbox3d_overlaps_giou
        elif type == "DIoU":
            self.bbox3d_iou_func = bbox3d_overlaps_diou
        else:
            #raise NotImplementedError
            pass

    def call(self, box_pred, mask, ind, box_gt):
        if tf.reduce_sum(mask) == 0:
            return tf.zeros((1,), dtype=box_pred.dtype)

        mask = tf.cast(mask, tf.bool)
        #mask = tf.reshape(mask, [mask.shape[0], -1])
        pred_box = _transpose_and_gather_feat(box_pred, ind)
        iou = self.bbox3d_iou_func(tf.boolean_mask(pred_box, mask), tf.boolean_mask(box_gt, mask))
        loss = tf.reduce_sum(1.0 - iou) / (tf.reduce_sum(tf.cast(mask, tf.float32)) + 1e-4)
        return loss
