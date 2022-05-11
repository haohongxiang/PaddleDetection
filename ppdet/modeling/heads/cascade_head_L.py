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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal

from ppdet.core.workspace import register
from .bbox_head import BBoxHead, TwoFCHead, XConvNormHead
from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import delta2bbox, clip_bbox, nonempty_bbox

__all__ = ['CascadeHeadL']


@register
class CascadeHeadL(BBoxHead):
    __shared__ = ['num_classes', 'num_cascade_stages']
    __inject__ = ['bbox_assigner', 'bbox_loss']
    """
    Cascade RCNN bbox head

    Args:
        head (nn.Layer): Extract feature in bbox head
        in_channel (int): Input channel after RoI extractor
        roi_extractor (object): The module of RoI Extractor
        bbox_assigner (object): The module of Box Assigner, label and sample the 
            box.
        num_classes (int): The number of classes
        bbox_weight (List[List[float]]): The weight to get the decode box and the 
            length of weight is the number of cascade stage
        num_cascade_stages (int): THe number of stage to refine the box
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 num_classes=80,
                 bbox_weight=[[10., 10., 5., 5.], [20.0, 20.0, 10.0, 10.0],
                              [30.0, 30.0, 15.0, 15.0]],
                 num_cascade_stages=3,
                 reg_class_agnostic=False,
                 bbox_loss=None,
                 stage_loss_weights=[1 / 3., 1 / 3., 1 / 3.]):

        nn.Layer.__init__(self, )
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner

        self.num_classes = num_classes
        self.bbox_weight = bbox_weight
        self.num_cascade_stages = num_cascade_stages
        self.bbox_loss = bbox_loss
        self.stage_loss_weights = stage_loss_weights

        self.reg_class_agnostic = reg_class_agnostic
        num_bbox_delta = 4 if reg_class_agnostic else 4 * num_classes

        self.bbox_score_list = []
        self.bbox_delta_list = []
        for i in range(num_cascade_stages):
            score_name = 'bbox_score_stage{}'.format(i)
            delta_name = 'bbox_delta_stage{}'.format(i)
            bbox_score = self.add_sublayer(
                score_name,
                nn.Linear(
                    in_channel,
                    self.num_classes + 1,
                    weight_attr=paddle.ParamAttr(initializer=Normal(
                        mean=0.0, std=0.01))))

            bbox_delta = self.add_sublayer(
                delta_name,
                nn.Linear(
                    in_channel,
                    num_bbox_delta,
                    weight_attr=paddle.ParamAttr(initializer=Normal(
                        mean=0.0, std=0.001))))
            self.bbox_score_list.append(bbox_score)
            self.bbox_delta_list.append(bbox_delta)
        self.assigned_label = None
        self.assigned_rois = None

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (Tensor): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        targets = []
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            targets_list = [targets]
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        pred_bbox = None
        head_out_list = []
        for i in range(self.num_cascade_stages):
            if i > 0:
                rois, rois_num = self._get_rois_from_boxes(pred_bbox,
                                                           inputs['im_shape'])
                if self.training:
                    rois, rois_num, targets = self.bbox_assigner(
                        rois, rois_num, inputs, i, is_cascade=True)
                    targets_list.append(targets)

            rois_feat = self.roi_extractor(body_feats, rois, rois_num)
            bbox_feat = self.head(rois_feat, i)
            scores = self.bbox_score_list[i](bbox_feat)  # MX80
            deltas = self.bbox_delta_list[i](bbox_feat)  # MX320

            # TODO
            if not self.reg_class_agnostic and i < self.num_cascade_stages - 1:
                deltas = deltas.reshape([-1, self.num_classes, 4])
                # deltas = deltas.reshape([-1, 4, self.num_classes]).transpose([0, 2, 1])
                labels = scores[:, :-1].argmax(axis=-1)
                deltas = deltas[paddle.arange(deltas.shape[0]), labels]

            head_out_list.append([scores, deltas, rois])
            pred_bbox = self._get_pred_bbox(deltas, rois, self.bbox_weight[i])

        if self.training:
            loss = {}
            for stage, value in enumerate(zip(head_out_list, targets_list)):
                (scores, deltas, rois), targets = value
                loss_stage = self.get_loss(scores, deltas, targets, rois,
                                           self.bbox_weight[stage])
                # TODO
                # for k, v in loss_stage.items():
                #     loss[k + "_stage{}".format(
                #         stage)] = v / self.num_cascade_stages

                for k, v in loss_stage.items():
                    loss[k + "_stage{}".format(
                        stage)] = v * self.stage_loss_weights[stage]

            return loss, bbox_feat

        else:

            scores, deltas, self.refined_rois = self.get_prediction(
                head_out_list)
            return (deltas, scores), self.head

    def _get_rois_from_boxes(self, boxes, im_shape):
        rois = []
        for i, boxes_per_image in enumerate(boxes):
            clip_box = clip_bbox(boxes_per_image, im_shape[i])
            if self.training:
                keep = nonempty_bbox(clip_box)
                if keep.shape[0] == 0:
                    keep = paddle.zeros([1], dtype='int32')
                clip_box = paddle.gather(clip_box, keep)
            rois.append(clip_box)
        rois_num = paddle.concat([paddle.shape(r)[0] for r in rois])
        return rois, rois_num

    def _get_pred_bbox(self, deltas, proposals, weights):
        pred_proposals = paddle.concat(proposals) if len(
            proposals) > 1 else proposals[0]
        pred_bbox = delta2bbox(deltas, pred_proposals, weights)
        pred_bbox = paddle.reshape(pred_bbox, [-1, deltas.shape[-1]])
        num_prop = []
        for p in proposals:
            num_prop.append(p.shape[0])
        return pred_bbox.split(num_prop)

    def get_prediction(self, head_out_list):
        """
        head_out_list(List[Tensor]): scores, deltas, rois
        """
        pred_list = []
        scores_list = [F.softmax(head[0]) for head in head_out_list]
        scores = paddle.add_n(scores_list) / self.num_cascade_stages
        # Get deltas and rois from the last stage
        _, deltas, rois = head_out_list[-1]
        return scores, deltas, rois

    def get_refined_rois(self, ):
        return self.refined_rois


# @register
# class TESTCONVHK(nn.Layer):
#     def __init__(self, cin, cout, k=3, s=1, p=0) -> None:
#         super().__init__()
#         self.conv = nn.Conv2D(cin, cout, kernel_size=k, stride=1, padding=p)

#     def forward(self, data):
#         return self.conv(data)

# from .cascade_head import CascadeTwoFCHead, CascadeXConvNormHead

# @register
# class CascadeTwoFCHead(nn.Layer):
#     __shared__ = ['num_cascade_stage']
#     """
#     Cascade RCNN bbox head  with Two fc layers to extract feature

#     Args:
#         in_channel (int): Input channel which can be derived by from_config
#         out_channel (int): Output channel
#         resolution (int): Resolution of input feature map, default 7
#         num_cascade_stage (int): The number of cascade stage, default 3
#     """

#     def __init__(self,
#                  in_channel=256,
#                  out_channel=1024,
#                  resolution=7,
#                  num_cascade_stage=3):
#         super(CascadeTwoFCHead, self).__init__()

#         self.in_channel = in_channel
#         self.out_channel = out_channel

#         self.head_list = []
#         for stage in range(num_cascade_stage):
#             head_per_stage = self.add_sublayer(
#                 str(stage), TwoFCHead(in_channel, out_channel, resolution))
#             self.head_list.append(head_per_stage)

#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         s = input_shape
#         s = s[0] if isinstance(s, (list, tuple)) else s
#         return {'in_channel': s.channels}

#     @property
#     def out_shape(self):
#         return [ShapeSpec(channels=self.out_channel, )]

#     def forward(self, rois_feat, stage=0):
#         out = self.head_list[stage](rois_feat)
#         return out

# @register
# class CascadeXConvNormHead(nn.Layer):
#     __shared__ = ['norm_type', 'freeze_norm', 'num_cascade_stage']
#     """
#     Cascade RCNN bbox head with serveral convolution layers

#     Args:
#         in_channel (int): Input channels which can be derived by from_config
#         num_convs (int): The number of conv layers
#         conv_dim (int): The number of channels for the conv layers
#         out_channel (int): Output channels
#         resolution (int): Resolution of input feature map
#         norm_type (string): Norm type, bn, gn, sync_bn are available, 
#             default `gn`
#         freeze_norm (bool): Whether to freeze the norm
#         num_cascade_stage (int): The number of cascade stage, default 3
#     """

#     def __init__(self,
#                  in_channel=256,
#                  num_convs=4,
#                  conv_dim=256,
#                  out_channel=1024,
#                  resolution=7,
#                  norm_type='gn',
#                  freeze_norm=False,
#                  num_cascade_stage=3):
#         super(CascadeXConvNormHead, self).__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.head_list = []
#         for stage in range(num_cascade_stage):
#             head_per_stage = self.add_sublayer(
#                 str(stage),
#                 XConvNormHead(
#                     in_channel,
#                     num_convs,
#                     conv_dim,
#                     out_channel,
#                     resolution,
#                     norm_type,
#                     freeze_norm,
#                     stage_name='stage{}_'.format(stage)))
#             self.head_list.append(head_per_stage)

#     @classmethod
#     def from_config(cls, cfg, input_shape):
#         s = input_shape
#         s = s[0] if isinstance(s, (list, tuple)) else s
#         return {'in_channel': s.channels}

#     @property
#     def out_shape(self):
#         return [ShapeSpec(channels=self.out_channel, )]

#     def forward(self, rois_feat, stage=0):
#         out = self.head_list[stage](rois_feat)
#         return out

# import torch
# import torch.nn as nn

# from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
#                         build_sampler, merge_aug_bboxes, merge_aug_masks,
#                         multiclass_nms)
# from ..builder import HEADS, build_head, build_roi_extractor
# from .base_roi_head import BaseRoIHead
# from .test_mixins import BBoxTestMixin, MaskTestMixin

# @HEADS.register_module()
# class CascadeRoIHead(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
#     """Cascade roi head including one bbox head and one mask head.

#     https://arxiv.org/abs/1712.00726
#     """

#     def __init__(self,
#                  num_stages,
#                  stage_loss_weights,
#                  bbox_roi_extractor=None,
#                  bbox_head=None,
#                  mask_roi_extractor=None,
#                  mask_head=None,
#                  shared_head=None,
#                  train_cfg=None,
#                  test_cfg=None):
#         assert bbox_roi_extractor is not None
#         assert bbox_head is not None
#         assert shared_head is None, \
#             'Shared head is not supported in Cascade RCNN anymore'
#         self.num_stages = num_stages
#         self.stage_loss_weights = stage_loss_weights
#         super(CascadeRoIHead, self).__init__(
#             bbox_roi_extractor=bbox_roi_extractor,
#             bbox_head=bbox_head,
#             mask_roi_extractor=mask_roi_extractor,
#             mask_head=mask_head,
#             shared_head=shared_head,
#             train_cfg=train_cfg,
#             test_cfg=test_cfg)

#     def init_bbox_head(self, bbox_roi_extractor, bbox_head):
#         """Initialize box head and box roi extractor.

#         Args:
#             bbox_roi_extractor (dict): Config of box roi extractor.
#             bbox_head (dict): Config of box in box head.
#         """
#         self.bbox_roi_extractor = nn.ModuleList()
#         self.bbox_head = nn.ModuleList()
#         if not isinstance(bbox_roi_extractor, list):
#             bbox_roi_extractor = [
#                 bbox_roi_extractor for _ in range(self.num_stages)
#             ]
#         if not isinstance(bbox_head, list):
#             bbox_head = [bbox_head for _ in range(self.num_stages)]
#         assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
#         for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
#             self.bbox_roi_extractor.append(build_roi_extractor(roi_extractor))
#             self.bbox_head.append(build_head(head))

#     def init_mask_head(self, mask_roi_extractor, mask_head):
#         """Initialize mask head and mask roi extractor.

#         Args:
#             mask_roi_extractor (dict): Config of mask roi extractor.
#             mask_head (dict): Config of mask in mask head.
#         """
#         self.mask_head = nn.ModuleList()
#         if not isinstance(mask_head, list):
#             mask_head = [mask_head for _ in range(self.num_stages)]
#         assert len(mask_head) == self.num_stages
#         for head in mask_head:
#             self.mask_head.append(build_head(head))
#         if mask_roi_extractor is not None:
#             self.share_roi_extractor = False
#             self.mask_roi_extractor = nn.ModuleList()
#             if not isinstance(mask_roi_extractor, list):
#                 mask_roi_extractor = [
#                     mask_roi_extractor for _ in range(self.num_stages)
#                 ]
#             assert len(mask_roi_extractor) == self.num_stages
#             for roi_extractor in mask_roi_extractor:
#                 self.mask_roi_extractor.append(
#                     build_roi_extractor(roi_extractor))
#         else:
#             self.share_roi_extractor = True
#             self.mask_roi_extractor = self.bbox_roi_extractor

#     def init_assigner_sampler(self):
#         """Initialize assigner and sampler for each stage."""
#         self.bbox_assigner = []
#         self.bbox_sampler = []
#         if self.train_cfg is not None:
#             for idx, rcnn_train_cfg in enumerate(self.train_cfg):
#                 self.bbox_assigner.append(
#                     build_assigner(rcnn_train_cfg.assigner))
#                 self.current_stage = idx
#                 self.bbox_sampler.append(
#                     build_sampler(rcnn_train_cfg.sampler, context=self))

#     def init_weights(self, pretrained):
#         """Initialize the weights in head.

#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """
#         if self.with_shared_head:
#             self.shared_head.init_weights(pretrained=pretrained)
#         for i in range(self.num_stages):
#             if self.with_bbox:
#                 self.bbox_roi_extractor[i].init_weights()
#                 self.bbox_head[i].init_weights()
#             if self.with_mask:
#                 if not self.share_roi_extractor:
#                     self.mask_roi_extractor[i].init_weights()
#                 self.mask_head[i].init_weights()

#     def forward_dummy(self, x, proposals):
#         """Dummy forward function."""
#         # bbox head
#         outs = ()
#         rois = bbox2roi([proposals])
#         if self.with_bbox:
#             for i in range(self.num_stages):
#                 bbox_results = self._bbox_forward(i, x, rois)
#                 outs = outs + (bbox_results['cls_score'],
#                                bbox_results['bbox_pred'])
#         # mask heads
#         if self.with_mask:
#             mask_rois = rois[:100]
#             for i in range(self.num_stages):
#                 mask_results = self._mask_forward(i, x, mask_rois)
#                 outs = outs + (mask_results['mask_pred'], )
#         return outs

#     def _bbox_forward(self, stage, x, rois):
#         """Box head forward function used in both training and testing."""
#         bbox_roi_extractor = self.bbox_roi_extractor[stage]
#         bbox_head = self.bbox_head[stage]
#         bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
#                                         rois)
#         # do not support caffe_c4 model anymore
#         cls_score, bbox_pred = bbox_head(bbox_feats)

#         bbox_results = dict(
#             cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
#         return bbox_results

#     def _bbox_forward_train(self, stage, x, sampling_results, gt_bboxes,
#                             gt_labels, rcnn_train_cfg):
#         """Run forward function and calculate loss for box head in training."""
#         rois = bbox2roi([res.bboxes for res in sampling_results])
#         bbox_results = self._bbox_forward(stage, x, rois)
#         bbox_targets = self.bbox_head[stage].get_targets(
#             sampling_results, gt_bboxes, gt_labels, rcnn_train_cfg)
#         loss_bbox = self.bbox_head[stage].loss(bbox_results['cls_score'],
#                                                bbox_results['bbox_pred'], rois,
#                                                *bbox_targets)

#         bbox_results.update(
#             loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
#         return bbox_results

#     def forward_train(self,
#                       x,
#                       img_metas,
#                       proposal_list,
#                       gt_bboxes,
#                       gt_labels,
#                       gt_bboxes_ignore=None,
#                       gt_masks=None):
#         """
#         Args:
#             x (list[Tensor]): list of multi-level img features.
#             img_metas (list[dict]): list of image info dict where each dict
#                 has: 'img_shape', 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmdet/datasets/pipelines/formatting.py:Collect`.
#             proposals (list[Tensors]): list of region proposals.
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss.
#             gt_masks (None | Tensor) : true segmentation masks for each box
#                 used if the architecture supports a segmentation task.

#         Returns:
#             dict[str, Tensor]: a dictionary of loss components
#         """
#         losses = dict()
#         for i in range(self.num_stages):
#             self.current_stage = i
#             rcnn_train_cfg = self.train_cfg[i]
#             lw = self.stage_loss_weights[i]

#             # assign gts and sample proposals
#             sampling_results = []
#             if self.with_bbox or self.with_mask:
#                 bbox_assigner = self.bbox_assigner[i]
#                 bbox_sampler = self.bbox_sampler[i]
#                 num_imgs = len(img_metas)
#                 if gt_bboxes_ignore is None:
#                     gt_bboxes_ignore = [None for _ in range(num_imgs)]

#                 for j in range(num_imgs):
#                     assign_result = bbox_assigner.assign(
#                         proposal_list[j], gt_bboxes[j], gt_bboxes_ignore[j],
#                         gt_labels[j])
#                     sampling_result = bbox_sampler.sample(
#                         assign_result,
#                         proposal_list[j],
#                         gt_bboxes[j],
#                         gt_labels[j],
#                         feats=[lvl_feat[j][None] for lvl_feat in x])
#                     sampling_results.append(sampling_result)

#             # bbox head forward and loss
#             bbox_results = self._bbox_forward_train(i, x, sampling_results,
#                                                     gt_bboxes, gt_labels,
#                                                     rcnn_train_cfg)

#             for name, value in bbox_results['loss_bbox'].items():
#                 losses[f's{i}.{name}'] = (
#                     value * lw if 'loss' in name else value)

#             # mask head forward and loss
#             if self.with_mask:
#                 mask_results = self._mask_forward_train(
#                     i, x, sampling_results, gt_masks, rcnn_train_cfg,
#                     bbox_results['bbox_feats'])
#                 for name, value in mask_results['loss_mask'].items():
#                     losses[f's{i}.{name}'] = (
#                         value * lw if 'loss' in name else value)

#             # refine bboxes
#             if i < self.num_stages - 1:
#                 pos_is_gts = [res.pos_is_gt for res in sampling_results]
#                 # bbox_targets is a tuple
#                 roi_labels = bbox_results['bbox_targets'][0]
#                 with torch.no_grad():
#                     roi_labels = torch.where(
#                         roi_labels == self.bbox_head[i].num_classes,
#                         bbox_results['cls_score'][:, :-1].argmax(1),
#                         roi_labels)
#                     proposal_list = self.bbox_head[i].refine_bboxes(
#                         bbox_results['rois'], roi_labels,
#                         bbox_results['bbox_pred'], pos_is_gts, img_metas)

#         return losses

#     def simple_test(self, x, proposal_list, img_metas, rescale=False):
#         """Test without augmentation."""
#         assert self.with_bbox, 'Bbox head must be implemented.'
#         num_imgs = len(proposal_list)
#         img_shapes = tuple(meta['img_shape'] for meta in img_metas)
#         ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
#         scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

#         # "ms" in variable names means multi-stage
#         ms_bbox_result = {}
#         ms_segm_result = {}
#         ms_scores = []
#         rcnn_test_cfg = self.test_cfg

#         rois = bbox2roi(proposal_list)
#         for i in range(self.num_stages):
#             bbox_results = self._bbox_forward(i, x, rois)

#             # split batch bbox prediction back to each image
#             cls_score = bbox_results['cls_score']
#             bbox_pred = bbox_results['bbox_pred']
#             num_proposals_per_img = tuple(
#                 len(proposals) for proposals in proposal_list)
#             rois = rois.split(num_proposals_per_img, 0)
#             cls_score = cls_score.split(num_proposals_per_img, 0)
#             if isinstance(bbox_pred, torch.Tensor):
#                 bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
#             else:
#                 bbox_pred = self.bbox_head[i].bbox_pred_split(
#                     bbox_pred, num_proposals_per_img)
#             ms_scores.append(cls_score)

#             if i < self.num_stages - 1:
#                 bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
#                 rois = torch.cat([
#                     self.bbox_head[i].regress_by_class(rois[j], bbox_label[j],
#                                                        bbox_pred[j],
#                                                        img_metas[j])
#                     for j in range(num_imgs)
#                 ])

#         # average scores of each image by stages
#         cls_score = [
#             sum([score[i] for score in ms_scores]) / float(len(ms_scores))
#             for i in range(num_imgs)
#         ]

#         # apply bbox post-processing to each image individually
#         det_bboxes = []
#         det_labels = []
#         for i in range(num_imgs):
#             det_bbox, det_label = self.bbox_head[-1].get_bboxes(
#                 rois[i],
#                 cls_score[i],
#                 bbox_pred[i],
#                 img_shapes[i],
#                 scale_factors[i],
#                 rescale=rescale,
#                 cfg=rcnn_test_cfg)
#             det_bboxes.append(det_bbox)
#             det_labels.append(det_label)

#         if torch.onnx.is_in_onnx_export():
#             return det_bboxes, det_labels
#         bbox_results = [
#             bbox2result(det_bboxes[i], det_labels[i],
#                         self.bbox_head[-1].num_classes)
#             for i in range(num_imgs)
#         ]
#         ms_bbox_result['ensemble'] = bbox_results

#         if self.with_mask:
#             if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
#                 mask_classes = self.mask_head[-1].num_classes
#                 segm_results = [[[] for _ in range(mask_classes)]
#                                 for _ in range(num_imgs)]
#             else:
#                 if rescale and not isinstance(scale_factors[0], float):
#                     scale_factors = [
#                         torch.from_numpy(scale_factor).to(det_bboxes[0].device)
#                         for scale_factor in scale_factors
#                     ]
#                 _bboxes = [
#                     det_bboxes[i][:, :4] *
#                     scale_factors[i] if rescale else det_bboxes[i][:, :4]
#                     for i in range(len(det_bboxes))
#                 ]
#                 mask_rois = bbox2roi(_bboxes)
#                 num_mask_rois_per_img = tuple(
#                     _bbox.size(0) for _bbox in _bboxes)
#                 aug_masks = []
#                 for i in range(self.num_stages):
#                     mask_results = self._mask_forward(i, x, mask_rois)
#                     mask_pred = mask_results['mask_pred']
#                     # split batch mask prediction back to each image
#                     mask_pred = mask_pred.split(num_mask_rois_per_img, 0)
#                     aug_masks.append(
#                         [m.sigmoid().cpu().numpy() for m in mask_pred])

#                 # apply mask post-processing to each image individually
#                 segm_results = []
#                 for i in range(num_imgs):
#                     if det_bboxes[i].shape[0] == 0:
#                         segm_results.append(
#                             [[]
#                              for _ in range(self.mask_head[-1].num_classes)])
#                     else:
#                         aug_mask = [mask[i] for mask in aug_masks]
#                         merged_masks = merge_aug_masks(
#                             aug_mask, [[img_metas[i]]] * self.num_stages,
#                             rcnn_test_cfg)
#                         segm_result = self.mask_head[-1].get_seg_masks(
#                             merged_masks, _bboxes[i], det_labels[i],
#                             rcnn_test_cfg, ori_shapes[i], scale_factors[i],
#                             rescale)
#                         segm_results.append(segm_result)
#             ms_segm_result['ensemble'] = segm_results

#         if self.with_mask:
#             results = list(
#                 zip(ms_bbox_result['ensemble'], ms_segm_result['ensemble']))
#         else:
#             results = ms_bbox_result['ensemble']

#         return results

#     def aug_test(self, features, proposal_list, img_metas, rescale=False):
#         """Test with augmentations.

#         If rescale is False, then returned bboxes and masks will fit the scale
#         of imgs[0].
#         """
#         rcnn_test_cfg = self.test_cfg
#         aug_bboxes = []
#         aug_scores = []
#         for x, img_meta in zip(features, img_metas):
#             # only one image in the batch
#             img_shape = img_meta[0]['img_shape']
#             scale_factor = img_meta[0]['scale_factor']
#             flip = img_meta[0]['flip']
#             flip_direction = img_meta[0]['flip_direction']

#             proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
#                                      scale_factor, flip, flip_direction)
#             # "ms" in variable names means multi-stage
#             ms_scores = []

#             rois = bbox2roi([proposals])
#             for i in range(self.num_stages):
#                 bbox_results = self._bbox_forward(i, x, rois)
#                 ms_scores.append(bbox_results['cls_score'])

#                 if i < self.num_stages - 1:
#                     bbox_label = bbox_results['cls_score'][:, :-1].argmax(
#                         dim=1)
#                     rois = self.bbox_head[i].regress_by_class(
#                         rois, bbox_label, bbox_results['bbox_pred'],
#                         img_meta[0])

#             cls_score = sum(ms_scores) / float(len(ms_scores))
#             bboxes, scores = self.bbox_head[-1].get_bboxes(
#                 rois,
#                 cls_score,
#                 bbox_results['bbox_pred'],
#                 img_shape,
#                 scale_factor,
#                 rescale=False,
#                 cfg=None)
#             aug_bboxes.append(bboxes)
#             aug_scores.append(scores)

#         # after merging, bboxes will be rescaled to the original image size
#         merged_bboxes, merged_scores = merge_aug_bboxes(
#             aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
#         det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
#                                                 rcnn_test_cfg.score_thr,
#                                                 rcnn_test_cfg.nms,
#                                                 rcnn_test_cfg.max_per_img)

#         bbox_result = bbox2result(det_bboxes, det_labels,
#                                   self.bbox_head[-1].num_classes)

#         if self.with_mask:
#             if det_bboxes.shape[0] == 0:
#                 segm_result = [[[]
#                                 for _ in range(self.mask_head[-1].num_classes)]
#                                ]
#             else:
#                 aug_masks = []
#                 aug_img_metas = []
#                 for x, img_meta in zip(features, img_metas):
#                     img_shape = img_meta[0]['img_shape']
#                     scale_factor = img_meta[0]['scale_factor']
#                     flip = img_meta[0]['flip']
#                     flip_direction = img_meta[0]['flip_direction']
#                     _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
#                                            scale_factor, flip, flip_direction)
#                     mask_rois = bbox2roi([_bboxes])
#                     for i in range(self.num_stages):
#                         mask_results = self._mask_forward(i, x, mask_rois)
#                         aug_masks.append(
#                             mask_results['mask_pred'].sigmoid().cpu().numpy())
#                         aug_img_metas.append(img_meta)
#                 merged_masks = merge_aug_masks(aug_masks, aug_img_metas,
#                                                self.test_cfg)

#                 ori_shape = img_metas[0][0]['ori_shape']
#                 segm_result = self.mask_head[-1].get_seg_masks(
#                     merged_masks,
#                     det_bboxes,
#                     det_labels,
#                     rcnn_test_cfg,
#                     ori_shape,
#                     scale_factor=1.0,
#                     rescale=False)
#             return [(bbox_result, segm_result)]
#         else:
#             return [bbox_result]
