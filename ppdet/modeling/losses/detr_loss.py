# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .iou_loss import GIoULoss
from ..bbox_utils import bbox_cxcywh_to_xyxy, bbox_overlaps

from scipy.optimize import linear_sum_assignment

__all__ = ['DETRLoss']


@register
class DETRLoss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 matcher_coeff={'class': 1,
                                'bbox': 5,
                                'giou': 2},
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 log_metric=False):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher_coeff (dict): The coefficient of hungarian matcher cost.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
        """
        super(DETRLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher_coeff = matcher_coeff
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.log_metric = log_metric

        self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                               loss_coeff['class'])
        self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    @paddle.no_grad()
    def _hungarian_matcher(self, boxes, scores, gt_bbox, gt_class):
        r"""
        Args:
            boxes (Tensor): [b, query, 4]
            scores (Tensor): [b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, num_classes]
        out_prob = F.softmax(scores.flatten(0, 1))
        # [batch_size * num_queries, 4]
        out_bbox = boxes.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = paddle.concat(gt_class).flatten()
        tgt_bbox = paddle.concat(gt_bbox)

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -paddle.gather(out_prob, tgt_ids, axis=1)

        # Compute the L1 cost between boxes
        cost_bbox = (
            out_bbox.unsqueeze(1) - tgt_bbox.unsqueeze(0)).abs().sum(-1)

        # Compute the giou cost betwen boxes
        cost_giou = self.giou_loss(
            bbox_cxcywh_to_xyxy(out_bbox.unsqueeze(1)),
            bbox_cxcywh_to_xyxy(tgt_bbox.unsqueeze(0))).squeeze(-1) - 1

        # Final cost matrix
        C = self.matcher_coeff['bbox'] * cost_bbox + self.matcher_coeff['class'] * cost_class + \
            self.matcher_coeff['giou'] * cost_giou
        C = C.reshape([bs, num_queries, -1])

        sizes = [a.shape[0] for a in gt_bbox]
        indices = [
            linear_sum_assignment(c[i].numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [(paddle.to_tensor(
            i, dtype='int64'), paddle.to_tensor(
                j, dtype='int64')) for i, j in indices]

    def _get_loss_class(self, scores, gt_class, match_indices, bg_index):
        # scores: [b, query, 81], gt_class: list[[n, 1]]
        target_label = paddle.full(scores.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        index, updates = self._get_index_updates(num_query_objects, gt_class,
                                                 match_indices)
        target_label = paddle.scatter(
            target_label.reshape([-1, 1]), index, updates.astype('int64'))
        target_label.stop_gradient = True 

        return {
            'loss_class': F.cross_entropy(
                scores,
                target_label.reshape([bs, num_query_objects, 1]),
                weight=self.loss_coeff['class'])
        }

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        num_gts = sum(len(a) for a in gt_bbox)
        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        target_bbox.stop_gradient = True 

        loss = dict()
        loss['loss_bbox'] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts

        loss['loss_giou'] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss['loss_giou'] = loss['loss_giou'].sum() / num_gts
        loss['loss_giou'] = self.loss_coeff['giou'] * loss['loss_giou']
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        num_gts = sum(len(a) for a in gt_mask)
        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        target_masks.stop_gradient = True 

        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss = dict()
        loss['loss_mask'] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss['loss_dice'] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self, boxes, scores, gt_bbox, gt_class, bg_index):
        loss_class = []
        loss_bbox = []
        loss_giou = []
        for aux_boxes, aux_scores in zip(boxes, scores):
            match_indices = self._hungarian_matcher(aux_boxes, aux_scores,
                                                    gt_bbox, gt_class)
            loss_class.append(
                self._get_loss_class(aux_scores, gt_class, match_indices,
                                     bg_index)['loss_class'])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices)
            loss_bbox.append(loss_['loss_bbox'])
            loss_giou.append(loss_['loss_giou'])
        loss = {
            'loss_class_aux': paddle.add_n(loss_class),
            'loss_bbox_aux': paddle.add_n(loss_bbox),
            'loss_giou_aux': paddle.add_n(loss_giou)
        }
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) for t, (_, J) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    @paddle.no_grad()
    def _log_precision_recall(self,
                              boxes,
                              scores,
                              gt_bbox,
                              gt_class,
                              threshold=0.5):
        scores = F.softmax(scores, -1)
        scores, labels = scores[:, :, :-1].max(-1), scores[:, :, :-1].argmax(-1)

        masked_objects = (scores > threshold).astype('float32')
        num_pred = masked_objects.sum()
        if num_pred > 0:
            num_gt = paddle.to_tensor([sum(len(a) for a in gt_class)])
            num_correct = 0
            for masked_object, label, box, gt_box, gt_label in \
                zip(masked_objects, labels, boxes, gt_bbox, gt_class):
                if masked_object.sum() > 0:
                    label = paddle.masked_select(label,
                                                 masked_object.astype('bool'))
                    box = paddle.masked_select(
                        box,
                        masked_object.astype('bool').unsqueeze(-1).tile(
                            [1, 4])).reshape([-1, 4])
                    ious = bbox_overlaps(
                        bbox_cxcywh_to_xyxy(gt_box), bbox_cxcywh_to_xyxy(box))
                    masked_pred = paddle.zeros_like(label)
                    for i in range(len(ious)):
                        iou, iou_index = ious[i].sort(
                            descending=True), ious[i].argsort(descending=True)
                        for j in range(len(iou)):
                            if iou[j] <= threshold:
                                break
                            if masked_pred[int(iou_index[j])] == 0:
                                if iou[j] > threshold and label[int(iou_index[
                                        j])] == gt_label[i]:
                                    num_correct += 1
                                    masked_pred[int(iou_index[j])] = 1
                                    break
            return {
                'log_recall': num_correct / num_gt,
                'log_precision': num_correct / num_pred
            }
        else:
            return {
                'log_recall': paddle.zeros([1]),
                'log_precision': paddle.zeros([1])
            }

    def forward(self,
                boxes,
                scores,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            scores (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
        """
        match_indices = self._hungarian_matcher(
            boxes[-1].detach(), scores[-1].detach(), gt_bbox, gt_class)

        total_loss = dict()
        total_loss.update(
            self._get_loss_class(scores[-1], gt_class, match_indices,
                                 self.num_classes))
        total_loss.update(
            self._get_loss_bbox(boxes[-1], gt_bbox, match_indices))
        if masks is not None and gt_mask is not None:
            total_loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices))

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(boxes[:-1], scores[:-1], gt_bbox, gt_class,
                                   self.num_classes))

        if self.log_metric:
            total_loss.update(
                self._log_precision_recall(boxes[-1].detach(), scores[-1]
                                           .detach(), gt_bbox, gt_class))

        return total_loss
