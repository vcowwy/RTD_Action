import numpy as np

import paddle
from paddle import nn
import paddle.nn.functional as F
import x2paddle
from x2paddle import torch2paddle

from util import box_ops
from util.misc import accuracy
from util.misc import get_world_size
from util.misc import is_dist_avail_and_initialized
from models.matcher import build_matcher
from models.position_embedding import build_position_embedding
from models.transformer import build_transformer

from util.t2p import Linear, Conv1d, Conv2d


class RTD(nn.Layer):

    def __init__(self,
                 position_embedding,
                 transformer,
                 num_classes,
                 num_queries,
                 stage,
                 aux_loss=False):
        super().__init__()
        self.num_queries = num_queries

        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        self.class_embed = Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        input_dim = 2048
        self.input_proj = Conv2d(input_dim, hidden_dim // 2, kernel_size=1)

        self.iou_conv = nn.Sequential(
            Conv1d(self.hidden_dim,
                      self.hidden_dim * 2,
                      kernel_size=3,
                      padding=1),
            x2paddle.torch2paddle.ReLU(inplace=True),
            Conv1d(self.hidden_dim * 2,
                      self.hidden_dim,
                      kernel_size=3,
                      padding=1))
        self.iou_embed = MLP(hidden_dim, hidden_dim * 2, 1, 3)
        self.stage = stage

        self.aux_loss = aux_loss
        self.position_embedding = position_embedding

    def forward(self, locations, samples, s_e_scores):
        bs = s_e_scores.shape[0]

        features_flatten = samples.flatten(0, 1)
        projected_fts = self.input_proj(
            features_flatten.unsqueeze(-1).unsqueeze(-1))
        projected_fts = projected_fts.view(bs, -1, self.hidden_dim // 2)
        scaling_factor = 2
        s = s_e_scores[:, :, 0] * scaling_factor
        e = s_e_scores[:, :, 1] * scaling_factor
        features_s = paddle.multiply(projected_fts, s.unsqueeze(-1))
        features_e = paddle.multiply(projected_fts, e.unsqueeze(-1))
        features = paddle.concat((features_s, features_e), axis=2)

        pos = self.position_embedding(locations)

        hs = self.transformer(features, self.query_embed.weight, pos)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs)
        outputs_coord = paddle.nn.functional.sigmoid(outputs_coord)

        proposal_fts = hs[-1, :, :, :].permute(0, 2, 1)
        proposal_fts = self.iou_conv(proposal_fts)
        proposal_fts = proposal_fts.permute(0, 2, 1)
        outputs_iou = paddle.nn.functional.sigmoid(self.iou_embed(proposal_fts))

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'pred_iou': outputs_iou}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                    outputs_coord, outputs_iou)
        return out

    #@paddle.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_iou):
        return [{'pred_logits': a,
                 'pred_boxes': b,
                 'pred_iou': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],
                                   outputs_iou[:-1])]


class SetCriterion(nn.Layer):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):

        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        shape = paddle.tolist(paddle.to_tensor(self.num_classes + 1))
        empty_weight = paddle.ones(shape)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert 'pred_logits' in outputs
        if indices is None:
            losses = {'loss_ce': 0}
            if log:
                losses['class_error'] = 0
            return losses

        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = paddle.concat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = paddle.full(src_logits.shape[:2],
                                     self.num_classes,
                                     dtype=paddle.int64).requires_grad_(False)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        idx_tensor = paddle.transpose(paddle.stack(x=idx), perm=[1, 0])
        idx_list = list()
        for i in range(int(idx_tensor.shape[0])):
            a = src_logits[int(idx_tensor[int(i)][0])][int(idx_tensor[int(i)][1])]
            idx_list.append(a)

        if log:
            losses['class_error'] = 100 - accuracy(paddle.stack(x=idx_list), target_classes_o)[0]
        return losses

    @paddle.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        if indices is None:
            losses = {'cardinality_error': 0}
            return losses

        pred_logits = outputs['pred_logits']
        device = pred_logits.place
        tgt_lengths = paddle.to_tensor(data=[len(v['labels']) for v in targets],
                                       place=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = paddle.nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        if indices is None:
            return {'loss_bbox': 0, 'loss_giou': 0}

        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        idx_tensor = paddle.transpose(paddle.stack(x=idx), perm=[1, 0])
        idx_list = list()
        for i in range(int(idx_tensor.shape[0])):
            a = outputs['pred_boxes'][int(idx_tensor[int(i)][0])][int(idx_tensor[int(i)][1])]
            idx_list.append(a)
        src_boxes = paddle.stack(idx_list)

        #[t['boxes'][i] for t, (_, i) in zip(targets, indices)]
        target_boxes = paddle.concat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices) if t['boxes'][i].shape[0] != 0], axis=0)

        loss_bbox = paddle.nn.functional.l1_loss(src_boxes,
                                                 box_ops.prop_se_to_cl(target_boxes),
                                                 reduction='none')

        losses = {}
        losses['loss_bbox'] = paddle.sum(loss_bbox, axis=None) / num_boxes

        loss_giou = 1 - paddle.diag(
            box_ops.generalized_prop_iou(box_ops.prop_cl_to_se(src_boxes),
                                         target_boxes))
        losses['loss_giou'] = paddle.sum(loss_giou, axis=None) / num_boxes
        return losses

    def loss_iou(self, outputs, targets, indices, num_boxes):
        assert 'pred_iou' in outputs
        assert 'pred_boxes' in outputs

        bs = outputs['pred_boxes'].shape[0]

        pred_boxes = outputs['pred_boxes']
        preds_iou = outputs['pred_iou']

        tgt_iou = []
        for i in range(bs):
            pred_boxes_per_seg = pred_boxes[i, :, :]
            target_boxes_per_seg = targets[i]['boxes']
            if len(target_boxes_per_seg) == 0:
                tiou = paddle.zeros(
                    [len(pred_boxes_per_seg)]).requires_grad_(False).to(pred_boxes_per_seg.place)
            else:
                tiou = box_ops.generalized_prop_iou(
                    box_ops.prop_cl_to_se(pred_boxes_per_seg),
                    target_boxes_per_seg)
                tiou = paddle.max(tiou, axis=1)
            tgt_iou.append(tiou)

        tgt_iou = paddle.stack(tgt_iou, axis=0).view(-1)
        preds_iou = preds_iou.view(-1)

        pos_ind = paddle.nonzero(tgt_iou > 0.7, as_tuple=True)[0]
        m_ind = paddle.nonzero((tgt_iou <= 0.7).logical_and(tgt_iou > 0.3),
                               as_tuple=True)[0].squeeze().cpu().detach().numpy()
        neg_ind = paddle.nonzero(tgt_iou < 0.3,
                                 as_tuple=True)[0].squeeze().cpu().detach().numpy()

        sampled_m_ind = np.random.choice(m_ind, len(pos_ind))
        sampled_neg_ind = np.random.choice(neg_ind, 2 * len(pos_ind))

        iou_mask = (tgt_iou > 0.7).float()
        iou_mask[sampled_m_ind] = 1.0
        iou_mask[sampled_neg_ind] = 1.0
        iou_loss = F.smooth_l1_loss(preds_iou, tgt_iou.squeeze()).float()
        iou_loss = torch2paddle.sum(
            iou_loss * iou_mask) / (1e-06 + torch2paddle.sum(iou_mask)).float()

        losses = {'loss_iou': iou_loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = paddle.concat(
            [paddle.full_like(src, i).requires_grad_(False) for i, (src, _) in enumerate(indices)])
        src_idx = paddle.concat([src for src, _ in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = paddle.concat(
            [paddle.full_like(tgt, i).requires_grad_(False) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = paddle.concat([tgt for _, tgt in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {'labels': self.loss_labels,
                    'cardinality': self.loss_cardinality,
                    'boxes': self.loss_boxes,
                    'iou': self.loss_iou}
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = paddle.to_tensor(dtype=paddle.float32,
                                     data=[num_boxes],
                                     place=next(iter(outputs.values())).place)
        if is_dist_avail_and_initialized():
            paddle.distributed.all_reduce(num_boxes)
        num_boxes = paddle.clip(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs and 'iou' not in self.losses:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets,
                                           indices, num_boxes, **kwargs)
                    l_dict = {(k + f'_{i}'): v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Layer):

    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size
        self.interval = args.interval
        self.absolute_position = args.absolute_position
        self.stage = args.stage

    @paddle.no_grad()
    def forward(self, outputs, num_frames, base):
        out_logits, out_bbox, out_iou = outputs['pred_logits'], outputs[
            'pred_boxes'], outputs['pred_iou']

        assert len(out_logits) == len(num_frames)
        num_frames = num_frames.reshape(len(out_logits), 1)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.prop_cl_to_se(out_bbox)
        if not self.absolute_position:
            boxes = box_ops.prop_relative_to_absolute(boxes, base,
                                                      self.window_size,
                                                      self.interval)
        else:
            bs, ws, _ = boxes.shape
            scale_fct = num_frames.unsqueeze(-1).repeat((1, ws, 2)).to(boxes.device)

            boxes = boxes * scale_fct

        if self.stage != 3:
            results = [{
                'scores': s,
                'labels': l,
                'boxes': b,
                'iou_score': i
            } for s, i, l, b in zip(scores, out_iou, labels, boxes)]
        if self.stage == 3:
            new_scores = 0.5 * (scores.squeeze() + out_iou.squeeze())
            results = [{
                'scores': s,
                'labels': l,
                'boxes': b,
                'iou_score': i
            } for s, i, l, b in zip(new_scores, out_iou, labels, boxes)]

        return results


class MLP(nn.Layer):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 1
    device = args.device
    device = device.replace('cuda', 'gpu')
    device = paddle.set_device(device)

    position_embedding = build_position_embedding(args)

    transformer = build_transformer(args)

    model = RTD(position_embedding=position_embedding,
                transformer=transformer,
                num_classes=num_classes,
                num_queries=args.num_queries,
                stage=args.stage,
                aux_loss=args.aux_loss)

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_iou': args.iou_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({
                (k + f'_{i}'): v
                for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    if args.stage != 3:
        losses = ['labels', 'boxes', 'cardinality']
    else:
        losses = ['iou']

    criterion = SetCriterion(num_classes,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=args.eos_coef,
                             losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, postprocessors
