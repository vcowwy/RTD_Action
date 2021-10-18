import numpy as np

import paddle
from x2paddle import torch2paddle


def prop_cl_to_se(x):
    c, l = x.unbind(-1)
    b = [c - 0.5 * l, c + 0.5 * l]
    return paddle.stack(b, axis=-1).clip(0, 1)


def prop_se_to_cl(x):
    s, e = x.unbind(-1)
    b = [(s + e) / 2, e - s]
    return paddle.stack(b, axis=-1)


def prop_relative_to_absolute(x, base, window_size, interval):
    s, e = x.unbind(-1)
    num_samples = s.shape[1]
    base = base.unsqueeze(1).repeat(1, num_samples).cuda()
    b = [s * window_size * interval + base, e * window_size * interval + base]
    return paddle.stack(b, axis=-1)


def segment_tiou(box_a, box_b):
    N = box_a.shape[0]
    M = box_b.shape[0]

    tiou = paddle.zeros((N, M)).requires_grad_(False).to(box_a.cuda)
    for i in range(N):
        inter_max_xy = torch2paddle.min(box_a[i, 1], box_b[:, 1])
        inter_min_xy = torch2paddle.max(box_a[i, 0], box_b[:, 0])

        inter = paddle.clip(inter_max_xy - inter_min_xy, min=0)

        union = box_b[:, 1] - box_b[:, 0] + (box_a[i, 1] - box_a[i, 0]) - inter

        tiou[i, :] = inter / union

    return tiou


def pairwise_temporal_iou(candidate_segments, target_segments):
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    n, m = target_segments.shape[0], candidate_segments.shape[0]
    t_iou = np.empty((n, m), dtype=np.float32)
    for i in range(m):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])

        segments_intersection = (tt2 - tt1).clip(0)

        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)

        t_iou[:, i] = segments_intersection.astype(float) / segments_union

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)

    return t_iou


def generalized_prop_iou(props1, props2):
    return segment_tiou(props1, props2)
