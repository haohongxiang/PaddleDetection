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

import numpy as np

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register

from ..bbox_utils import decode_yolo, xywh2xyxy, iou_similarity, bbox_iou

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

    __inject__ = ['iou_loss', 'iou_aware_loss']
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 label_smooth=False,
                 downsample=[32, 16, 8],
                 giou_ratio=1.0,
                 balance=[4., 1., 0.4],
                 ignore_thresh=0.7,
                 scale_x_y=1.,
                 iou_loss=None,
                 iou_aware_loss=None):

        super(YOLOv5Loss, self).__init__()
        self.giou_ratio = giou_ratio
        self.num_classes = num_classes
        self.balance = balance
        self.downsample = downsample
        
        self.num_classes = 80
        self.no = self.num_classes + 4 + 1 
        
    def obj_loss(self, pobj, giou, mask):
        # giou_mask = giou * mask
        # giou_mask = paddle.max(giou_mask, axis=1)
        # giou_mask = paddle.clip(giou_mask.detach(), min=0)
        giou_mask = (giou.detach() * mask).max(axis=1).clip(min=0)

        # mask = paddle.sum(mask, axis=1)
        # mask = (mask >= 1)
        # mask = mask.sum(axis=1) >= 1
        
        # pobj = paddle.reshape(pobj, (-1)
        # tobj = ((1 - self.giou_ratio) + self.giou_ratio * giou_mask) * mask
        tobj = (1 - self.giou_ratio) + self.giou_ratio * giou_mask

        loss_obj = F.binary_cross_entropy_with_logits(pobj[:, 0, :, :, :], tobj, reduction='mean')
        
        return loss_obj

    def cls_loss(self, pcls, tcls, mask):
        #pcls = pcls.reshape((-1, self.num_classes))
        #tcls = tcls.reshape((-1, self.num_classes))
        
        # print(pcls.shape, tcls.shape, mask.shape)
        
        # TODO
        # mask = mask.unsqueeze(-1)
        index = (mask > 0).nonzero()
        tcls = tcls.gather_nd(index)
        pcls = pcls.gather_nd(index)
        print(tcls.shape, pcls.shape)
        
        loss_cls = F.binary_cross_entropy_with_logits(pcls, tcls, reduction='mean')

        # loss_cls = F.binary_cross_entropy_with_logits(pcls, tcls, reduction='none')
        # loss_cls = paddle.mean(loss_cls, axis=-1)
        # loss_cls = paddle.sum(loss_cls * mask)
        
        return loss_cls

    def yolov5_loss(self, p, t, anchor, downsample, balance, eps=1e-8):
        loss_boxes, loss_objs, loss_clss = [], [], []

#         anchor = [[i / downsample for i in xy] for xy in anchor]
#         anchor = np.array(anchor).reshape((1, 3, 1, 1, 2)).astype(np.float32)
#         anchor = np.tile(anchor[:, None, :, :, :, :], (1, self.nt_max, 1, 1, 1, 1))
#         anchor_w = anchor[:, :, :, :, :, 0]
#         anchor_h = anchor[:, :, :, :, :, 1]
#         anchor_w = paddle.to_tensor(anchor_w)
#         anchor_h = paddle.to_tensor(anchor_h)
        
        anchor = anchor.reshape((1, 1, 3, 1, 1, 2)) / downsample
        anchor = paddle.expand(anchor, [1, self.nt_max, 3, 1, 1, 2])
        # anchor_w = anchor[:, :, :, :, :, 0]
        # anchor_h = anchor[:, :, :, :, :, 1]

        x = F.sigmoid(p[:, :, :, :, :, 0]) * 2 - 0.5
        y = F.sigmoid(p[:, :, :, :, :, 1]) * 2 - 0.5

        # w = (F.sigmoid(p[:, :, :, :, :, 2]) * 2) ** 2 * anchor_w
        # h = (F.sigmoid(p[:, :, :, :, :, 3]) * 2) ** 2 * anchor_h
        w = (F.sigmoid(p[:, :, :, :, :, 2]) * 2) ** 2 * anchor[:, :, :, :, :, 0]
        h = (F.sigmoid(p[:, :, :, :, :, 3]) * 2) ** 2 * anchor[:, :, :, :, :, 1]

        pobj, pcls = p[:, :, :, :, :, 4], p[:, :, :, :, :, 5:]

        tx = t[:, :, :, :, :, 0]
        ty = t[:, :, :, :, :, 1]
        tw = t[:, :, :, :, :, 2]
        th = t[:, :, :, :, :, 3]
        mask = t[:, :, :, :, :, 4]
        tcls = t[:, :, :, :, :, 5:]
        

        loss = dict()

        # pbox = [x, y, w, h]
        # tbox = [tx, ty, tw, th]

        pbox = xywh2xyxy([x, y, w, h])
        tbox = xywh2xyxy([tx, ty, tw, th])
        
        nm = mask.sum()
        
        giou = bbox_iou(pbox, tbox, ciou=True)
        # giou = bbox_iou(pbox, tbox, ciou=True).detach()
        # giou.stop_gradient = True
        
        loss_obj = self.obj_loss(pobj, giou, mask)
        loss['loss_obj'] = loss_obj * 1.0 * balance

        if nm.item():
            index = (mask > 0).nonzero()
            tcls = tcls.gather_nd(index)
            pcls = pcls.gather_nd(index)
            
            loss_cls = F.binary_cross_entropy_with_logits(pcls, tcls, reduction='mean')
            loss['loss_cls'] = loss_cls * 0.5
        
#             tbox = tbox.gather_nd(index)
#             pbox = tbox.gather_nd(index)
#             print(tcls.shape, pcls.shape, tbox.shape, pbox.shape)

            # loss_box = paddle.sum((1 - giou) * mask)
            # loss['loss_box'] =  loss_box / nm * 0.05
            loss['loss_box'] = ((1 - giou).gather_nd(index)).mean() * 0.05
            
#         nm = paddle.sum(mask) + eps

#         giou = bbox_iou(pbox, tbox, ciou=True)
#         loss_box = paddle.sum((1 - giou) * mask)
#         loss['loss_box'] = loss_box / nm * 0.05

#         loss_cls = self.cls_loss(pcls, tcls, mask)
#         loss['loss_cls'] = loss_cls / nm * 0.5
        
        return loss

    def forward(self, inputs, targets, anchors):
        np = len(inputs)
        gt_targets = [targets['target{}'.format(i)] for i in range(np)]
        
        yolo_losses = dict()
        for i, (x, t, anchor, downsample) in enumerate(zip(inputs, gt_targets, anchors, self.downsample)):
                        
            na = len(anchor)
            b, c, h, w = x.shape
            self.nt_max = t.shape[1]

            # x = paddle.unsqueeze(x, axis=1)
            # x = paddle.expand(x.unsqueeze(axis=1), [b, self.nt_max, c, h, w])
            x = x.unsqueeze(axis=1).expand(shape=[b, self.nt_max, c, h, w])

            x = x.reshape((b, self.nt_max, na, self.no, h, w)).transpose((0, 1, 2, 4, 5, 3))
            t = t.reshape((b, self.nt_max, na, self.no, h, w)).transpose((0, 1, 2, 4, 5, 3))
            
            yolo_loss = self.yolov5_loss(x, t, anchor, downsample, self.balance[i])
            
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

