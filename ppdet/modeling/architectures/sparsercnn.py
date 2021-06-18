'''
    @Author: feizzhang
    Created on: 05.20.2021
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

__all__ = ["SparseRCNN"]


@register
class SparseRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = ["postprocess"]

    def __init__(self,
                backbone,
                neck,
                head = "SparsercnnHead",
                postprocess = "SparsePostProcess"):
        super(SparseRCNN, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.postprocess = postprocess

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        backbone = create(cfg['backbone'])

        kwargs = {'input_shape': backbone.out_shape}
        neck = create(cfg['neck'], **kwargs)

        kwargs = {'roi_input_shape': neck.out_shape}
        head = create(cfg['head'], **kwargs)

        return {
            'backbone': backbone,
            'neck': neck,
            "head": head,
        }
    
    def _forward(self):
        body_feats = self.backbone(self.inputs)
        fpn_feats =  self.neck(body_feats)
        head_outs = self.head(fpn_feats, self.inputs["img_whwh"])

        if not self.training:
            bboxes = self.postprocess( head_outs["pred_logits"], 
                                        head_outs["pred_boxes"], 
                                        self.inputs["scale_factor_wh"],
                                        self.inputs["img_whwh"]
                                       )
            return bboxes
        else:
            return head_outs
    
    def get_loss(self):
        batch_gt_class = self.inputs["gt_class"]
        batch_gt_box = self.inputs["gt_bbox"]
        batch_whwh = self.inputs["img_whwh"]
        targets = []

        for i in range(len(batch_gt_class)):
            boxes = batch_gt_box[i]
            labels = batch_gt_class[i].squeeze(-1)
            img_whwh = batch_whwh[i]
            img_whwh_tgt = img_whwh.unsqueeze(0).tile([int(boxes.shape[0]), 1])
            targets.append({"boxes":boxes, "labels":labels, "img_whwh": img_whwh, "img_whwh_tgt": img_whwh_tgt})
        
        outputs = self._forward()
        loss_dict = self.head.get_loss(outputs, targets)
        acc = loss_dict["acc"]
        loss_dict.pop("acc")
        total_loss = sum(loss_dict.values())
        loss_dict.update({"loss": total_loss, "acc": acc})
        return loss_dict
    
    def get_pred(self):
        bbox_pred, bbox_num = self._forward()
        output = {'bbox': bbox_pred, 'bbox_num': bbox_num}
        return output



 
































    
































