import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import paddle
from paddle import nn
from x2paddle import torch2paddle

from util.box_ops import generalized_prop_iou
from util.box_ops import pairwise_temporal_iou
from util.box_ops import prop_cl_to_se
from util.box_ops import prop_se_to_cl


class HungarianMatcher(nn.Layer):
    def __init__(self,
                 cost_class,
                 cost_bbox,
                 cost_giou,
                 stage,
                 relax_rule,
                 relax_thresh,
                 relax_topk=1):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'
        self.stage = stage
        self.relax_rule = relax_rule
        self.relax_thresh = relax_thresh
        self.relax_topk = relax_topk

    @paddle.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].flatten(0, 1)
        out_prob = paddle.nn.functional.softmax(out_prob)
        out_bbox = outputs['pred_boxes'].flatten(0, 1)

        try:
            tgt_ids = paddle.concat([v['labels'] for v in targets])
        except:
            tgt_ids = paddle.to_tensor([])
        try:
            tgt_bbox = paddle.concat([v['boxes'] for v in targets])
        except:
            tgt_bbox = paddle.to_tensor([])

        if len(tgt_bbox) == 0:
            return None

        # cost_class = -out_prob[:, tgt_ids]
        cost_class = -paddle.index_select(out_prob, tgt_ids, axis=1)
        prop_data = prop_se_to_cl(tgt_bbox)

        out_bbox_np = out_bbox.numpy()
        prop_data_np = prop_data.numpy()
        cost_bbox_np = cdist(out_bbox_np, prop_data_np, metric='minkowski', p=1)
        cost_bbox = paddle.to_tensor(cost_bbox_np)

        cost_giou = -generalized_prop_iou(prop_cl_to_se(out_bbox), tgt_bbox)

        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]
        indices = []
        for i, c in enumerate(C.split(sizes, -1)):
            if c.shape[2] == 0:
                np_data = np.empty([c.shape[1], c.shape[2]])
                pd_data = paddle.to_tensor(np_data)
                indices.append(linear_sum_assignment(pd_data))
                continue
            item = c[i]
            indices.append(linear_sum_assignment(item))

        if self.stage == 2:
            result_indices = []
            for batch_id in range(len(targets)):
                result_indices.append(list(indices[batch_id]))
                gt_boxes = targets[batch_id]['boxes']
                pred_boxes = prop_cl_to_se(outputs['pred_boxes'][batch_id])

                if len(gt_boxes) == 0:
                    tiou = np.zeros((len(pred_boxes), 1))
                    continue
                else:
                    tiou = pairwise_temporal_iou(
                        pred_boxes.detach().cpu().numpy(),
                        gt_boxes.detach().cpu().numpy())

                if self.relax_rule == 'thresh':
                    max_tiou = tiou.max(axis=0)
                    max_tiou_indices = tiou.argmax(axis=0)
                    pred_idx = np.where(max_tiou >= self.relax_thresh)[0]
                    gt_idx = max_tiou_indices[pred_idx]

                    for i, j in zip(gt_idx, pred_idx):
                        if j not in result_indices[batch_id][0]:
                            result_indices[batch_id][0] = np.append(
                                result_indices[batch_id][0], j)
                            result_indices[batch_id][1] = np.append(
                                result_indices[batch_id][1], i)

                elif self.relax_rule == 'topk':
                    pred_idx = paddle.to_tensor(tiou).argsort(dim=1)[:, -self.relax_topk:].reshape(-1).tolist()
                    for i in range(len(pred_idx)):
                        if pred_idx[i] not in result_indices[batch_id][0]:
                            result_indices[batch_id][0] = np.append(
                                result_indices[batch_id][0], pred_idx[i])
                            result_indices[batch_id][1] = np.append(
                                result_indices[batch_id][1], i)

            return [(paddle.to_tensor(dtype=paddle.int64, data=i),
                     paddle.to_tensor(dtype=paddle.int64, data=j))
                    for i, j in result_indices]
        else:
            return [(paddle.to_tensor(dtype=paddle.int64, data=i),
                     paddle.to_tensor(dtype=paddle.int64, data=j))
                    for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou,
                            stage=args.stage,
                            relax_rule=args.relax_rule,
                            relax_thresh=args.relax_thresh)
