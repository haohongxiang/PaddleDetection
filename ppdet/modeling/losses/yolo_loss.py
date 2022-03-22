# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity, bbox_iou
from IPython import embed

__all__ = ['YOLOv3Loss', 'YOLOv5Loss']


def bbox_transform(pbox, anchor, downsample):
    pbox = decode_yolo(pbox, anchor, downsample)
    pbox = xywh2xyxy(pbox)
    return pbox


@register
class YOLOv3Loss(nn.Layer):

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 ignore_thresh=0.7,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):
        """
        YOLOv3Loss layer

        Args:
            num_calsses (int): number of foreground classes
            ignore_thresh (float): threshold to ignore confidence loss
            label_smooth (bool): whether to use label smoothing
            downsample (list): downsample ratio for each detection block
            scale_x_y (float): scale_x_y factor
            iou_loss (object): IoULoss instance
            iou_aware_loss (object): IouAwareLoss instance  
        """
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.label_smooth = label_smooth
        self.downsample = downsample
        self.scale_x_y = scale_x_y
        self.iou_loss = iou_loss
        self.iou_aware_loss = iou_aware_loss
        self.distill_pairs = []

    def obj_loss(self, pbox, gbox, pobj, tobj, anchor, downsample):
        # pbox
        pbox = decode_yolo(pbox, anchor, downsample)
        pbox = xywh2xyxy(pbox)
        pbox = paddle.concat(pbox, axis=-1)
        b = pbox.shape[0]
        pbox = pbox.reshape((b, -1, 4))
        # gbox
        gxy = gbox[:, :, 0:2] - gbox[:, :, 2:4] * 0.5
        gwh = gbox[:, :, 0:2] + gbox[:, :, 2:4] * 0.5
        gbox = paddle.concat([gxy, gwh], axis=-1)

        iou = iou_similarity(pbox, gbox)
        iou.stop_gradient = True
        iou_max = iou.max(2)  # [N, M1]
        iou_mask = paddle.cast(iou_max <= self.ignore_thresh, dtype=pbox.dtype)
        iou_mask.stop_gradient = True

        pobj = pobj.reshape((b, -1))
        tobj = tobj.reshape((b, -1))
        obj_mask = paddle.cast(tobj > 0, dtype=pbox.dtype)
        obj_mask.stop_gradient = True

        loss_obj = F.binary_cross_entropy_with_logits(
            pobj, obj_mask, reduction='none')
        loss_obj_pos = (loss_obj * tobj)
        loss_obj_neg = (loss_obj * (1 - obj_mask) * iou_mask)
        return loss_obj_pos + loss_obj_neg

    def cls_loss(self, pcls, tcls):
        if self.label_smooth:
            delta = min(1. / self.num_classes, 1. / 40)
            pos, neg = 1 - delta, delta
            # 1 for positive, 0 for negative
            tcls = pos * paddle.cast(
                tcls > 0., dtype=tcls.dtype) + neg * paddle.cast(
                    tcls <= 0., dtype=tcls.dtype)

        loss_cls = F.binary_cross_entropy_with_logits(
            pcls, tcls, reduction='none')
        return loss_cls

    def yolov3_loss(self, p, t, gt_box, anchor, downsample, scale=1.,
                    eps=1e-10):
        na = len(anchor)
        b, c, h, w = p.shape
        if self.iou_aware_loss:
            ioup, p = p[:, 0:na, :, :], p[:, na:, :, :]
            ioup = ioup.unsqueeze(-1)
        p = p.reshape((b, na, -1, h, w)).transpose((0, 1, 3, 4, 2))
        x, y = p[:, :, :, :, 0:1], p[:, :, :, :, 1:2]
        w, h = p[:, :, :, :, 2:3], p[:, :, :, :, 3:4]
        obj, pcls = p[:, :, :, :, 4:5], p[:, :, :, :, 5:]
        self.distill_pairs.append([x, y, w, h, obj, pcls])

        t = t.transpose((0, 1, 3, 4, 2))
        tx, ty = t[:, :, :, :, 0:1], t[:, :, :, :, 1:2]
        tw, th = t[:, :, :, :, 2:3], t[:, :, :, :, 3:4]
        tscale = t[:, :, :, :, 4:5]
        tobj, tcls = t[:, :, :, :, 5:6], t[:, :, :, :, 6:]

        tscale_obj = tscale * tobj
        loss = dict()

        x = scale * F.sigmoid(x) - 0.5 * (scale - 1.)
        y = scale * F.sigmoid(y) - 0.5 * (scale - 1.)

        if abs(scale - 1.) < eps:
            loss_x = F.binary_cross_entropy(x, tx, reduction='none')
            loss_y = F.binary_cross_entropy(y, ty, reduction='none')
            loss_xy = tscale_obj * (loss_x + loss_y)
        else:
            loss_x = paddle.abs(x - tx)
            loss_y = paddle.abs(y - ty)
            loss_xy = tscale_obj * (loss_x + loss_y)

        loss_xy = loss_xy.sum([1, 2, 3, 4]).mean()

        loss_w = paddle.abs(w - tw)
        loss_h = paddle.abs(h - th)
        loss_wh = tscale_obj * (loss_w + loss_h)
        loss_wh = loss_wh.sum([1, 2, 3, 4]).mean()

        loss['loss_xy'] = loss_xy
        loss['loss_wh'] = loss_wh

        if self.iou_loss is not None:
            # warn: do not modify x, y, w, h in place
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou = self.iou_loss(pbox, gbox)
            loss_iou = loss_iou * tscale_obj
            loss_iou = loss_iou.sum([1, 2, 3, 4]).mean()
            loss['loss_iou'] = loss_iou

        if self.iou_aware_loss is not None:
            box, tbox = [x, y, w, h], [tx, ty, tw, th]
            pbox = bbox_transform(box, anchor, downsample)
            gbox = bbox_transform(tbox, anchor, downsample)
            loss_iou_aware = self.iou_aware_loss(ioup, pbox, gbox)
            loss_iou_aware = loss_iou_aware * tobj
            loss_iou_aware = loss_iou_aware.sum([1, 2, 3, 4]).mean()
            loss['loss_iou_aware'] = loss_iou_aware

        box = [x, y, w, h]
        loss_obj = self.obj_loss(box, gt_box, obj, tobj, anchor, downsample)
        loss_obj = loss_obj.sum(-1).mean()
        loss['loss_obj'] = loss_obj
        loss_cls = self.cls_loss(pcls, tcls) * tobj
        loss_cls = loss_cls.sum([1, 2, 3, 4]).mean()
        loss['loss_cls'] = loss_cls
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        gt_box = targets['gt_bbox']
        yolo_losses = dict()
        self.distill_pairs.clear()
        for x, t, anchor, downsample in zip(inputs, gt_targets, anchors,
                                            self.downsample):
            yolo_loss = self.yolov3_loss(x, t, gt_box, anchor, downsample,
                                         self.scale_x_y)
            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss
        return yolo_losses


@register
class YOLOv5Loss(nn.Layer):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 downsample_ratios=[8, 16, 32],
                 bias=0.5,
                 anchor_t=4.0,
                 balance=[4.0, 1.0, 0.4],
                 box_weight=0.05,
                 obj_weight=1.0,
                 cls_weght=0.5,
                 label_smooth_eps=0.):
        super(YOLOv5Loss, self).__init__()
        self.num_classes = num_classes
        self.balance = balance
        self.na = 3
        self.no = self.num_classes + 4 + 1
        self.gr = 1.0

        self.BCEcls = nn.BCEWithLogitsLoss(reduction="none")
        self.BCEobj = nn.BCEWithLogitsLoss(reduction="none")

        self.loss_weights = {
            'box': box_weight,
            'obj': obj_weight,
            'cls': cls_weght,
        }

        eps = label_smooth_eps if label_smooth_eps > 0 else 0.
        self.cp = 1.0 - 0.5 * eps
        self.cn = 0.5 * eps

        self.downsample_ratios = downsample_ratios
        self.bias = bias
        self.off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype=np.float32) * self.bias
        self.anchor_t = anchor_t

    def build_targets(self, outputs, targets, anchors):
        '''
        [[[116, 90], [156, 198], [373, 326]],
        [[30, 61], [62, 45], [59, 119]],
        [[10, 13], [16, 30], [33, 23]]]
        '''
        h, w = targets['image'].shape[2:]
        gt_nums = [len(bbox) for bbox in targets['gt_bbox']]
        # nt =
        nt = int(sum(gt_nums))
        na = len(anchors)
        tcls, tbox, indices, anch = [], [], [], []

        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.repeat(np.arange(na).reshape(na, 1), nt, axis=1)
        ai = paddle.to_tensor(ai, dtype='float64').unsqueeze(-1)

        batch_size = outputs[0].shape[0]
        gt_labels = []
        for idx in range(batch_size):
            gt_num = gt_nums[idx]
            if gt_num == 0:
                continue
                print('没有gt')
                print(targets['gt_bbox'][idx])
                print(targets['gt_class'][idx])
            gt_bbox = targets['gt_bbox'][idx][:gt_num]
            gt_class = targets['gt_class'][idx][:gt_num] * 1.0
            img_idx = np.repeat(np.array([[idx]]), gt_num, axis=0)
            img_idx = paddle.to_tensor(img_idx, dtype='float64')
            gt_labels.append(
                paddle.concat(
                    (img_idx, gt_class, gt_bbox), axis=-1))
        gt_labels = paddle.concat(gt_labels)
        targets_labels = paddle.concat(
            (paddle.tile(gt_labels.unsqueeze(0), [na, 1, 1]), ai), axis=2)
        # targets_labels.shape (3, 13, 7)
        targets_labels = targets_labels.numpy()
        g = 0.5  # bias
        off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            dtype=np.float32) * g  # offsets
        # anchors = anchors[::-1]
        for i in range(len(anchors)):
            anchor = np.array(anchors[i]) / self.downsample_ratios[i]  #
            gain[2:6] = np.array(
                outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to 
            t = targets_labels * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchor[:, None]  # wh ratio
                j = np.maximum(
                    r, 1 /
                    r).max(2) < self.anchor_t  #self.hyp['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = np.stack((np.ones_like(j), j, k, l, m))
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets_labels[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1), gi.clip(
                0, gain[2] - 1)  # add line make result same as clamp_
            indices.append(
                (paddle.to_tensor(b), paddle.to_tensor(a), paddle.to_tensor(
                    gj, dtype=paddle.int64), paddle.to_tensor(
                        gi, dtype=paddle.int64)))  # image, anchor, grid indices
            tbox.append(
                paddle.to_tensor(
                    np.concatenate((gxy - gij, gwh), 1),
                    dtype=paddle.float32))  # box
            anch.append(paddle.to_tensor(anchor[a]))  # 
            tcls.append(paddle.to_tensor(c))  # class
        return tcls, tbox, indices, anch

    def yolov5_loss(self, pi, t_cls, t_box, t_indices, t_anchor, balance):
        loss = dict()
        b, a, gj, gi = t_indices  # image, anchor, gridy, gridx
        n = b.shape[0]  # number of targets
        tobj = paddle.zeros_like(pi[:, :, :, :, 0])
        tobj.stop_gradient = True
        loss_box = paddle.to_tensor([0.])
        loss_cls = paddle.to_tensor([0.])
        if n:
            # if n <= 0:
            #     # print('---------------------------------------------------')
            #     loss['loss_box'] = paddle.to_tensor([0.])
            #     loss['loss_obj'] = paddle.to_tensor([0.])
            #     loss['loss_cls'] = paddle.to_tensor([0.])
            #     return loss

            # tobj = paddle.zeros_like(pi[:, :, :, :, 0])  # [4, 3, 80, 80]
            # tobj.stop_gradient = True

            ps = pi[b, a, gj, gi]  # TODO, fix in paddle 2.2.1
            # [4, 3, 80, 80, 85] -> [21, 85]

            # Regression
            pxy = F.sigmoid(ps[:, :2]) * 2 - 0.5
            pwh = (F.sigmoid(ps[:, 2:4]) * 2)**2 * t_anchor
            pbox = paddle.concat((pxy, pwh), 1)  # predicted box # [21, 4]
            iou = bbox_iou(
                pbox.T, t_box.T, x1y1x2y2=False,
                ciou=True)  # iou(prediction, target)
            # iou.stop_gradient = True
            loss_box = (1.0 - iou).mean()

            # Objectness
            score_iou = iou.detach().clip(0)
            tobj[b, a, gj, gi] = (1.0 - self.gr
                                  ) + self.gr * score_iou  # iou ratio

            # Classification
            t = paddle.full_like(ps[:, 5:], self.cn)
            t[range(n), t_cls] = self.cp
            t.stop_gradient = True
            loss_cls = self.BCEcls(ps[:, 5:], t).mean()

        obji = self.BCEobj(pi[:, :, :, :, 4], tobj).mean()  # [4, 3, 80, 80]
        # print(pi[:, :, :, :, 4].mean(), tobj.mean())

        loss_obj = obji * balance

        loss['loss_box'] = loss_box * self.loss_weights['box']
        loss['loss_obj'] = loss_obj * self.loss_weights['obj']
        loss['loss_cls'] = loss_cls * self.loss_weights['cls']

        return loss

    def forward(self, outputs, targets, anchors):
        assert len(outputs) == len(anchors)
        batch_size = outputs[0].shape[0]
        yolo_losses = dict()
        #print('pred shape', [x.shape for x in outputs])
        #print('pred sum ', [x.sum() for x in outputs])

        tcls, tbox, indices, anch = self.build_targets(outputs, targets,
                                                       anchors)

        for i, (p_det, balance) in enumerate(zip(outputs, self.balance)):
            t_cls = tcls[i]  #targets['tcls{}'.format(i)][0]
            t_box = tbox[i]  #targets['tbox{}'.format(i)][0]
            t_anchor = anch[i]  #targets['anchors{}'.format(i)][0]
            # TODO, now each sample has all targets of the batch

            #num_indices = len(indices[i]) #len(targets['indices{}'.format(i)])
            #t_indices = [targets['indices{}'.format(i)][j][0] for j in range(num_indices)]
            t_indices = indices[
                i]  #[targets['indices{}'.format(i)][j][0] for j in range(num_indices)]

            bs, ch, h, w = p_det.shape
            pi = p_det.reshape((bs, self.na, -1, h, w)).transpose(
                (0, 1, 3, 4, 2))

            yolo_loss = self.yolov5_loss(pi, t_cls, t_box, t_indices, t_anchor,
                                         balance)

            for k, v in yolo_loss.items():
                if k in yolo_losses:
                    yolo_losses[k] += v
                else:
                    yolo_losses[k] = v

        loss = 0
        for k, v in yolo_losses.items():
            loss += v

        yolo_losses['loss'] = loss * batch_size
        return yolo_losses
