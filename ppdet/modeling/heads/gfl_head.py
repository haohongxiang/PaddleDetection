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

import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Normal, Constant

from ppdet.core.workspace import register
from ppdet.modeling.layers import ConvNormLayer
from ppdet.modeling.bbox_utils import distance2bbox, bbox2distance
from ppdet.data.transform.atss_assigner import bbox_overlaps


class ScaleReg(nn.Layer):
    """
    Parameter for scaling the regression outputs.
    """

    def __init__(self):
        super(ScaleReg, self).__init__()
        self.scale_reg = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype="float32")

    def forward(self, inputs):
        out = inputs * self.scale_reg
        return out


class Integral(nn.Layer):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             paddle.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape([-1, self.reg_max + 1]), axis=1)
        x = F.linear(x, self.project).reshape([-1, 4])
        return x


@register
class DGQP(nn.Layer):
    """Distribution-Guided Quality Predictor of GFocal head

    Args:
        reg_topk (int): top-k statistics of distribution to guide LQE
        reg_channels (int): hidden layer unit to generate LQE
        add_mean (bool): Whether to calculate the mean of top-k statistics
    """

    def __init__(self, reg_topk=4, reg_channels=64, add_mean=True):
        super(DGQP, self).__init__()
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        if add_mean:
            self.total_dim += 1
        self.reg_conv1 = self.add_sublayer(
            'dgqp_reg_conv1',
            nn.Conv2D(
                in_channels=4 * self.total_dim,
                out_channels=self.reg_channels,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))
        self.reg_conv2 = self.add_sublayer(
            'dgqp_reg_conv2',
            nn.Conv2D(
                in_channels=self.reg_channels,
                out_channels=1,
                kernel_size=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        N, _, H, W = x.shape[:]
        prob = F.softmax(x.reshape([N, 4, -1, H, W]), axis=2)
        prob_topk, _ = prob.topk(self.reg_topk, axis=2)
        if self.add_mean:
            stat = paddle.concat(
                [prob_topk, prob_topk.mean(
                    axis=2, keepdim=True)], axis=2)
        else:
            stat = prob_topk
        y = F.relu(self.reg_conv1(stat.reshape([N, -1, H, W])))
        y = F.sigmoid(self.reg_conv2(y))
        return y

    
@register
class GFLHeadPre(nn.Layer):
    """
    FCOSFeat of FCOS

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the FCOSFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        use_dcn (bool): Whether to use dcn in tower or not.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=256,
                 num_cls_convs=4,
                 num_reg_convs=4,
                 norm_type='bn',
                 lr_scale=2.0,
                 bias_on=True,
                 use_dcn=False):
        super(GFLHeadPre, self).__init__()
        
        self.num_cls_convs = num_cls_convs
        self.num_reg_convs = num_reg_convs
        
        self.norm_type = norm_type
        self.cls_subnet_convs = []
        self.reg_subnet_convs = []
        
        for i in range(self.num_cls_convs):
            in_c = feat_in if i == 0 else feat_out

            cls_conv_name = 'fcos_head_cls_tower_conv_{}'.format(i)
            cls_conv = self.add_sublayer(
                cls_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    use_dcn=use_dcn,
                    bias_on=bias_on,
                    lr_scale=lr_scale))
            self.cls_subnet_convs.append(cls_conv)

            
        for i in range(self.num_cls_convs):
            in_c = feat_in if i == 0 else feat_out

            reg_conv_name = 'fcos_head_reg_tower_conv_{}'.format(i)
            reg_conv = self.add_sublayer(
                reg_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    use_dcn=use_dcn,
                    bias_on=bias_on,
                    lr_scale=lr_scale))
            self.reg_subnet_convs.append(reg_conv)

    def forward(self, fpn_feat):
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        
#         for i in range(self.num_cls_convs):
#             if i == self.num_cls_convs - 1:
#                 cls_feat = self.cls_subnet_convs[i](cls_feat)
#             else:
#                 cls_feat = F.relu(self.cls_subnet_convs[i](cls_feat))

        for i in range(self.num_cls_convs):
            cls_feat = F.relu(self.reg_subnet_convs[i](cls_feat))

        cls_feat = cls_feat + fpn_feat
        
        
        for i in range(self.num_reg_convs):
            reg_feat = F.relu(self.reg_subnet_convs[i](reg_feat))
            
        return cls_feat, reg_feat




@register
class FCOSFeatLastDCN(nn.Layer):
    """
    FCOSFeat of FCOS

    Args:
        feat_in (int): The channel number of input Tensor.
        feat_out (int): The channel number of output Tensor.
        num_convs (int): The convolution number of the FCOSFeat.
        norm_type (str): Normalization type, 'bn'/'sync_bn'/'gn'.
        use_dcn (bool): Whether to use dcn in tower or not.
    """

    def __init__(self,
                 feat_in=256,
                 feat_out=256,
                 num_convs=4,
                 norm_type='bn',
                 use_dcn=False):
        super(FCOSFeatLastDCN, self).__init__()
        self.num_convs = num_convs
        self.norm_type = norm_type
        self.cls_subnet_convs = []
        self.reg_subnet_convs = []
        self.use_dcn = use_dcn
        
        for i in range(self.num_convs):
            in_c = feat_in if i == 0 else feat_out
            
            if i < self.num_convs - 1:
                use_dcn = False
            else:
                use_dcn = self.use_dcn
                
            cls_conv_name = 'fcos_head_cls_tower_conv_{}'.format(i)
            cls_conv = self.add_sublayer(
                cls_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    use_dcn=use_dcn,
                    bias_on=True,
                    lr_scale=2.))
            self.cls_subnet_convs.append(cls_conv)

            reg_conv_name = 'fcos_head_reg_tower_conv_{}'.format(i)
            reg_conv = self.add_sublayer(
                reg_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=feat_out,
                    filter_size=3,
                    stride=1,
                    norm_type=norm_type,
                    use_dcn=use_dcn,
                    bias_on=True,
                    lr_scale=2.))
            self.reg_subnet_convs.append(reg_conv)

    def forward(self, fpn_feat):
        cls_feat = fpn_feat
        reg_feat = fpn_feat
        for i in range(self.num_convs):
            cls_feat = F.relu(self.cls_subnet_convs[i](cls_feat))
            reg_feat = F.relu(self.reg_subnet_convs[i](reg_feat))
        return cls_feat, reg_feat

    
    
    
@register
class GFLHead(nn.Layer):
    """
    GFLHead
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_qfl (object):
        loss_dfl (object):
        loss_bbox (object):
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_qfl', 'loss_dfl', 'loss_bbox', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_qfl='QualityFocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(GFLHead, self).__init__()
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_qfl = loss_qfl
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_qfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        conv_cls_name = "gfl_head_cls"
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.gfl_head_cls = self.add_sublayer(
            conv_cls_name,
            nn.Conv2D(
                in_channels=self.feat_in_chan,
                out_channels=self.cls_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    initializer=Constant(value=bias_init_value))))

        conv_reg_name = "gfl_head_reg"
        self.gfl_head_reg = self.add_sublayer(
            conv_reg_name,
            nn.Conv2D(
                in_channels=self.feat_in_chan,
                out_channels=4 * (self.reg_max + 1),
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

        self.scales_regs = []
        for i in range(len(self.fpn_stride)):
            lvl = int(math.log(int(self.fpn_stride[i]), 2))
            feat_name = 'p{}_feat'.format(lvl)
            scale_reg = self.add_sublayer(feat_name, ScaleReg())
            self.scales_regs.append(scale_reg)

        self.distribution_project = Integral(self.reg_max)

    def forward(self, fpn_feats):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        for scale_reg, fpn_feat in zip(self.scales_regs, fpn_feats):
            
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat)
            
            cls_logits = self.gfl_head_cls(conv_cls_feat)
            bbox_reg = scale_reg(self.gfl_head_reg(conv_reg_feat))
            
            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_reg)
                cls_logits = F.sigmoid(cls_logits) * quality_score
            if not self.training:
                cls_logits = F.sigmoid(cls_logits.transpose([0, 2, 3, 1]))
                bbox_reg = bbox_reg.transpose([0, 2, 3, 1])
                
            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)

        return (cls_logits_list, bboxes_reg_list)

    def _images_to_levels(self, target, num_level_anchors):
        """
        Convert targets by image to targets by feature level.
        """
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets

    def _grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        Args:
            grid_cells: grid cells of a feature map
        Returns:
            center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return paddle.stack([cells_cx, cells_cy], axis=-1)

    def get_loss(self, gfl_head_outs, gt_meta):
        cls_logits, bboxes_reg = gfl_head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
        ]
        grid_cells_list = self._images_to_levels(gt_meta['grid_cells'],
                                                 num_level_anchors)
        labels_list = self._images_to_levels(gt_meta['labels'],
                                             num_level_anchors)
        label_weights_list = self._images_to_levels(gt_meta['label_weights'],
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(gt_meta['bbox_targets'],
                                                   num_level_anchors)
        num_total_pos = sum(gt_meta['pos_num'])
        try:
            num_total_pos = paddle.distributed.all_reduce(num_total_pos.clone(
            )) / paddle.distributed.get_world_size()
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_qfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, grid_cells, labels, label_weights, bbox_targets, stride in zip(
                cls_logits, bboxes_reg, grid_cells_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            grid_cells = grid_cells.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])
            label_weights = label_weights.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = paddle.nonzero(
                paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            score = np.zeros(labels.shape)
            if len(pos_inds) > 0:
                pos_bbox_targets = paddle.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = paddle.gather(bbox_pred, pos_inds, axis=0)
                pos_grid_cells = paddle.gather(grid_cells, pos_inds, axis=0)
                pos_grid_cell_centers = self._grid_cells_to_center(
                    pos_grid_cells) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = paddle.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)
                score[pos_inds.numpy()] = bbox_iou
                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_grid_cell_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = paddle.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = paddle.to_tensor([0], dtype='float32')

            # qfl loss
            score = paddle.to_tensor(score)
            loss_qfl = self.loss_qfl(
                cls_score, (labels, score),
                weight=label_weights,
                avg_factor=num_total_pos)
            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_qfl_list.append(loss_qfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            avg_factor = paddle.distributed.all_reduce(avg_factor.clone())
            avg_factor = paddle.clip(
                avg_factor / paddle.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_qfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_qfl = sum(loss_qfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states

    def get_single_level_center_point(self, featmap_size, stride,
                                      cell_offset=0):
        """
        Generate pixel centers of a single stage feature map.
        Args:
            featmap_size: height and width of the feature map
            stride: down sample stride of the feature map
        Returns:
            y and x of the center points
        """
        h, w = featmap_size
        x_range = (paddle.arange(w, dtype='float32') + cell_offset) * stride
        y_range = (paddle.arange(h, dtype='float32') + cell_offset) * stride
        y, x = paddle.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        return y, x

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          img_shape,
                          scale_factor,
                          rescale=True,
                          cell_offset=0):
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred in zip(self.fpn_stride, cls_scores,
                                                bbox_preds):
            featmap_size = [
                paddle.shape(cls_score)[0], paddle.shape(cls_score)[1]
            ]
            y, x = self.get_single_level_center_point(
                featmap_size, stride, cell_offset=cell_offset)
            center_points = paddle.stack([x, y], axis=-1)
            scores = cls_score.reshape([-1, self.cls_out_channels])
            bbox_pred = self.distribution_project(bbox_pred) * stride

            if scores.shape[0] > self.nms_pre:
                max_scores = scores.max(axis=1)
                _, topk_inds = max_scores.topk(self.nms_pre)
                center_points = center_points.gather(topk_inds)
                bbox_pred = bbox_pred.gather(topk_inds)
                scores = scores.gather(topk_inds)

            bboxes = distance2bbox(
                center_points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        if rescale:
            # [h_scale, w_scale] to [w_scale, h_scale, w_scale, h_scale]
            im_scale = paddle.concat([scale_factor[::-1], scale_factor[::-1]])
            mlvl_bboxes /= im_scale
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    def decode(self, cls_scores, bbox_preds, im_shape, scale_factor,
               cell_offset):
        batch_bboxes = []
        batch_scores = []
        for img_id in range(cls_scores[0].shape[0]):
            num_levels = len(cls_scores)
            cls_score_list = [cls_scores[i][img_id] for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_levels)]
            bboxes, scores = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                im_shape[img_id],
                scale_factor[img_id],
                cell_offset=cell_offset)
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, axis=0)
        batch_scores = paddle.stack(batch_scores, axis=0)

        return batch_bboxes, batch_scores

    def post_process(self, gfl_head_outs, im_shape, scale_factor):
        cls_scores, bboxes_reg = gfl_head_outs
        bboxes, score = self.decode(cls_scores, bboxes_reg, im_shape,
                                    scale_factor, self.cell_offset)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num

    
    
    

@register
class VFLHead(nn.Layer):
    """
    GFLHead
    Args:
        conv_feat (object): Instance of 'FCOSFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_vfl (object):
        loss_dfl (object):
        loss_bbox (object):
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 16.
    """
    __inject__ = [
        'conv_feat', 'dgqp_module', 'loss_vfl', 'loss_dfl', 'loss_bbox', 'nms'
    ]
    __shared__ = ['num_classes']

    def __init__(self,
                 conv_feat='FCOSFeat',
                 dgqp_module=None,
                 num_classes=80,
                 fpn_stride=[8, 16, 32, 64, 128],
                 prior_prob=0.01,
                 loss_vfl='VarifocalLoss',
                 loss_dfl='DistributionFocalLoss',
                 loss_bbox='GIoULoss',
                 reg_max=16,
                 feat_in_chan=256,
                 nms=None,
                 nms_pre=1000,
                 cell_offset=0):
        super(GFLHead, self).__init__()
        self.conv_feat = conv_feat
        self.dgqp_module = dgqp_module
        self.num_classes = num_classes
        self.fpn_stride = fpn_stride
        self.prior_prob = prior_prob
        self.loss_vfl = loss_vfl
        self.loss_dfl = loss_dfl
        self.loss_bbox = loss_bbox
        self.reg_max = reg_max
        self.feat_in_chan = feat_in_chan
        self.nms = nms
        self.nms_pre = nms_pre
        self.cell_offset = cell_offset
        self.use_sigmoid = self.loss_vfl.use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = self.num_classes
        else:
            self.cls_out_channels = self.num_classes + 1

        conv_cls_name = "vfl_head_cls"
        bias_init_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        self.gfl_head_cls = self.add_sublayer(
            conv_cls_name,
            nn.Conv2D(
                in_channels=self.feat_in_chan,
                out_channels=self.cls_out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(
                    initializer=Constant(value=bias_init_value))))

        conv_reg_name = "gfl_head_reg"
        self.gfl_head_reg = self.add_sublayer(
            conv_reg_name,
            nn.Conv2D(
                in_channels=self.feat_in_chan,
                out_channels=4 * (self.reg_max + 1),
                kernel_size=3,
                stride=1,
                padding=1,
                weight_attr=ParamAttr(initializer=Normal(
                    mean=0., std=0.01)),
                bias_attr=ParamAttr(initializer=Constant(value=0))))

        self.scales_regs = []
        for i in range(len(self.fpn_stride)):
            lvl = int(math.log(int(self.fpn_stride[i]), 2))
            feat_name = 'p{}_feat'.format(lvl)
            scale_reg = self.add_sublayer(feat_name, ScaleReg())
            self.scales_regs.append(scale_reg)

        self.distribution_project = Integral(self.reg_max)

    def forward(self, fpn_feats):
        assert len(fpn_feats) == len(
            self.fpn_stride
        ), "The size of fpn_feats is not equal to size of fpn_stride"
        cls_logits_list = []
        bboxes_reg_list = []
        for scale_reg, fpn_feat in zip(self.scales_regs, fpn_feats):
            
            conv_cls_feat, conv_reg_feat = self.conv_feat(fpn_feat)
            
            cls_logits = self.gfl_head_cls(conv_cls_feat)
            bbox_reg = scale_reg(self.gfl_head_reg(conv_reg_feat))
            
            if self.dgqp_module:
                quality_score = self.dgqp_module(bbox_reg)
                cls_logits = F.sigmoid(cls_logits) * quality_score
            if not self.training:
                cls_logits = F.sigmoid(cls_logits.transpose([0, 2, 3, 1]))
                bbox_reg = bbox_reg.transpose([0, 2, 3, 1])
                
            cls_logits_list.append(cls_logits)
            bboxes_reg_list.append(bbox_reg)

        return (cls_logits_list, bboxes_reg_list)

    def _images_to_levels(self, target, num_level_anchors):
        """
        Convert targets by image to targets by feature level.
        """
        level_targets = []
        start = 0
        for n in num_level_anchors:
            end = start + n
            level_targets.append(target[:, start:end].squeeze(0))
            start = end
        return level_targets

    def _grid_cells_to_center(self, grid_cells):
        """
        Get center location of each gird cell
        Args:
            grid_cells: grid cells of a feature map
        Returns:
            center points
        """
        cells_cx = (grid_cells[:, 2] + grid_cells[:, 0]) / 2
        cells_cy = (grid_cells[:, 3] + grid_cells[:, 1]) / 2
        return paddle.stack([cells_cx, cells_cy], axis=-1)

    def get_loss(self, gfl_head_outs, gt_meta):
        cls_logits, bboxes_reg = gfl_head_outs
        num_level_anchors = [
            featmap.shape[-2] * featmap.shape[-1] for featmap in cls_logits
        ]
        grid_cells_list = self._images_to_levels(gt_meta['grid_cells'],
                                                 num_level_anchors)
        labels_list = self._images_to_levels(gt_meta['labels'],
                                             num_level_anchors)
        label_weights_list = self._images_to_levels(gt_meta['label_weights'],
                                                    num_level_anchors)
        bbox_targets_list = self._images_to_levels(gt_meta['bbox_targets'],
                                                   num_level_anchors)
        num_total_pos = sum(gt_meta['pos_num'])
        try:
            num_total_pos = paddle.distributed.all_reduce(num_total_pos.clone(
            )) / paddle.distributed.get_world_size()
        except:
            num_total_pos = max(num_total_pos, 1)

        loss_bbox_list, loss_dfl_list, loss_vfl_list, avg_factor = [], [], [], []
        for cls_score, bbox_pred, grid_cells, labels, label_weights, bbox_targets, stride in zip(
                cls_logits, bboxes_reg, grid_cells_list, labels_list,
                label_weights_list, bbox_targets_list, self.fpn_stride):
            grid_cells = grid_cells.reshape([-1, 4])
            cls_score = cls_score.transpose([0, 2, 3, 1]).reshape(
                [-1, self.cls_out_channels])
            bbox_pred = bbox_pred.transpose([0, 2, 3, 1]).reshape(
                [-1, 4 * (self.reg_max + 1)])
            bbox_targets = bbox_targets.reshape([-1, 4])
            labels = labels.reshape([-1])
            label_weights = label_weights.reshape([-1])

            bg_class_ind = self.num_classes
            pos_inds = paddle.nonzero(
                paddle.logical_and((labels >= 0), (labels < bg_class_ind)),
                as_tuple=False).squeeze(1)
            
            # score = np.zeros(labels.shape)
            vfl_score = np.zeors(cls_score.shape)
            
            if len(pos_inds) > 0:
                pos_bbox_targets = paddle.gather(bbox_targets, pos_inds, axis=0)
                pos_bbox_pred = paddle.gather(bbox_pred, pos_inds, axis=0)
                pos_grid_cells = paddle.gather(grid_cells, pos_inds, axis=0)
                pos_grid_cell_centers = self._grid_cells_to_center(
                    pos_grid_cells) / stride

                weight_targets = F.sigmoid(cls_score.detach())
                weight_targets = paddle.gather(
                    weight_targets.max(axis=1, keepdim=True), pos_inds, axis=0)
                pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
                pos_decode_bbox_pred = distance2bbox(pos_grid_cell_centers,
                                                     pos_bbox_pred_corners)
                pos_decode_bbox_targets = pos_bbox_targets / stride
                bbox_iou = bbox_overlaps(
                    pos_decode_bbox_pred.detach().numpy(),
                    pos_decode_bbox_targets.detach().numpy(),
                    is_aligned=True)
                
                # score[pos_inds.numpy()] = bbox_iou
                
                pred_corners = pos_bbox_pred.reshape([-1, self.reg_max + 1])
                target_corners = bbox2distance(pos_grid_cell_centers,
                                               pos_decode_bbox_targets,
                                               self.reg_max).reshape([-1])
                # regression loss
                loss_bbox = paddle.sum(
                    self.loss_bbox(pos_decode_bbox_pred,
                                   pos_decode_bbox_targets) * weight_targets)

                # dfl loss
                loss_dfl = self.loss_dfl(
                    pred_corners,
                    target_corners,
                    weight=weight_targets.expand([-1, 4]).reshape([-1]),
                    avg_factor=4.0)
                
                
                pos_labels = paddle.gather(labels, pos_inds, axis=0)
                vfl_score[pos_inds.numpy(), pos_labels] = bbox_iou
                
            else:
                loss_bbox = bbox_pred.sum() * 0
                loss_dfl = bbox_pred.sum() * 0
                weight_targets = paddle.to_tensor([0], dtype='float32')

            # qfl loss
#             score = paddle.to_tensor(score)
#             loss_qfl = self.loss_qfl(
#                 cls_score, (labels, score),
#                 weight=label_weights,
#                 avg_factor=num_total_pos)

            vfl_score = paddle.to_tensor(vfl_score)
            loss_vfl = self.loss_vfl(cls_score, vfl_score, avg_factor=num_total_pos)
    
            loss_bbox_list.append(loss_bbox)
            loss_dfl_list.append(loss_dfl)
            loss_vfl_list.append(loss_vfl)
            avg_factor.append(weight_targets.sum())

        avg_factor = sum(avg_factor)
        try:
            avg_factor = paddle.distributed.all_reduce(avg_factor.clone())
            avg_factor = paddle.clip(
                avg_factor / paddle.distributed.get_world_size(), min=1)
        except:
            avg_factor = max(avg_factor.item(), 1)
        if avg_factor <= 0:
            loss_vfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
            loss_bbox = paddle.to_tensor(
                0, dtype='float32', stop_gradient=False)
            loss_dfl = paddle.to_tensor(0, dtype='float32', stop_gradient=False)
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, loss_bbox_list))
            losses_dfl = list(map(lambda x: x / avg_factor, loss_dfl_list))
            loss_vfl = sum(loss_vfl_list)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss_states = dict(
            loss_vfl=loss_vfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)

        return loss_states

    def get_single_level_center_point(self, featmap_size, stride,
                                      cell_offset=0):
        """
        Generate pixel centers of a single stage feature map.
        Args:
            featmap_size: height and width of the feature map
            stride: down sample stride of the feature map
        Returns:
            y and x of the center points
        """
        h, w = featmap_size
        x_range = (paddle.arange(w, dtype='float32') + cell_offset) * stride
        y_range = (paddle.arange(h, dtype='float32') + cell_offset) * stride
        y, x = paddle.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        return y, x

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          img_shape,
                          scale_factor,
                          rescale=True,
                          cell_offset=0):
        assert len(cls_scores) == len(bbox_preds)
        mlvl_bboxes = []
        mlvl_scores = []
        for stride, cls_score, bbox_pred in zip(self.fpn_stride, cls_scores,
                                                bbox_preds):
            featmap_size = [
                paddle.shape(cls_score)[0], paddle.shape(cls_score)[1]
            ]
            y, x = self.get_single_level_center_point(
                featmap_size, stride, cell_offset=cell_offset)
            center_points = paddle.stack([x, y], axis=-1)
            scores = cls_score.reshape([-1, self.cls_out_channels])
            bbox_pred = self.distribution_project(bbox_pred) * stride

            if scores.shape[0] > self.nms_pre:
                max_scores = scores.max(axis=1)
                _, topk_inds = max_scores.topk(self.nms_pre)
                center_points = center_points.gather(topk_inds)
                bbox_pred = bbox_pred.gather(topk_inds)
                scores = scores.gather(topk_inds)

            bboxes = distance2bbox(
                center_points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = paddle.concat(mlvl_bboxes)
        if rescale:
            # [h_scale, w_scale] to [w_scale, h_scale, w_scale, h_scale]
            im_scale = paddle.concat([scale_factor[::-1], scale_factor[::-1]])
            mlvl_bboxes /= im_scale
        mlvl_scores = paddle.concat(mlvl_scores)
        mlvl_scores = mlvl_scores.transpose([1, 0])
        return mlvl_bboxes, mlvl_scores

    def decode(self, cls_scores, bbox_preds, im_shape, scale_factor,
               cell_offset):
        batch_bboxes = []
        batch_scores = []
        for img_id in range(cls_scores[0].shape[0]):
            num_levels = len(cls_scores)
            cls_score_list = [cls_scores[i][img_id] for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id] for i in range(num_levels)]
            bboxes, scores = self.get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                im_shape[img_id],
                scale_factor[img_id],
                cell_offset=cell_offset)
            batch_bboxes.append(bboxes)
            batch_scores.append(scores)
        batch_bboxes = paddle.stack(batch_bboxes, axis=0)
        batch_scores = paddle.stack(batch_scores, axis=0)

        return batch_bboxes, batch_scores

    def post_process(self, gfl_head_outs, im_shape, scale_factor):
        cls_scores, bboxes_reg = gfl_head_outs
        bboxes, score = self.decode(cls_scores, bboxes_reg, im_shape,
                                    scale_factor, self.cell_offset)
        bbox_pred, bbox_num, _ = self.nms(bboxes, score)
        return bbox_pred, bbox_num
