import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register
from ..backbones.darknet import ConvBNLayer


def _de_sigmoid(x, eps=1e-7):
    x = paddle.clip(x, eps, 1. / eps)
    x = paddle.clip(1. / x - 1., eps, 1. / eps)
    x = -paddle.log(x)
    return x


def _conv_bn_relu(in_channels,
                  out_channels,
                  kernel,
                  stride,
                  padding=0,
                  dilation=1):
    '''
    '''
    return nn.Sequential(
        ('conv', nn.Conv2D(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            bias_attr=False)), ('bn', nn.BatchNorm2D(out_channels)),
        ('relu', nn.ReLU()))


def conv_bn_relu(in_channels,
                 out_channels,
                 kernel,
                 stride,
                 padding=0,
                 dilation=1):
    '''
    '''
    return nn.Sequential(
        ('conv', nn.Conv2D(
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            bias_attr=False)), ('bn', nn.SyncBatchNorm(out_channels)),
        ('relu', nn.ReLU()))


def conv_bn_relu_v2(in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding=0,
                    dilation=1):
    '''
    '''
    return nn.Sequential(
        ('conv.1', nn.Conv2D(
            in_channels, in_channels // 4, 1, 1, 0, 1, bias_attr=False)),
        ('bn.1', nn.SyncBatchNorm(in_channels // 4)),
        ('relu.1', nn.ReLU()),
        ('conv.2', nn.Conv2D(
            in_channels // 4,
            in_channels // 4,
            kernel,
            stride,
            padding,
            dilation,
            bias_attr=False)),
        ('bn.2', nn.SyncBatchNorm(in_channels // 4)),
        ('relu.2', nn.ReLU()),
        ('conv.3', nn.Conv2D(
            in_channels // 4, in_channels, 1, 1, 0, 1, bias_attr=False)),
        ('bn.3', nn.SyncBatchNorm(in_channels)),
        ('relu.3', nn.ReLU()), )


@register
class YOLOv3Head(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['loss']

    def __init__(self,
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 num_classes=80,
                 loss='YOLOv3Loss',
                 iou_aware=False,
                 iou_aware_factor=0.4):
        super(YOLOv3Head, self).__init__()
        self.num_classes = num_classes
        self.loss = loss

        self.iou_aware = iou_aware
        self.iou_aware_factor = iou_aware_factor

        self.parse_anchor(anchors, anchor_masks)
        self.num_outputs = len(self.anchors)

        self.yolo_outputs = nn.LayerList()

        for i in range(len(self.anchors)):

            if self.iou_aware:
                num_filters = len(self.anchors[i]) * (self.num_classes + 6)
            else:
                num_filters = len(self.anchors[i]) * (self.num_classes + 5)

            name = 'yolo_output.{}'.format(i)

            in_channels = 128 * (2**self.num_outputs) // (2**i)

            # decode_conv = self.add_sublayer(
            #     name,
            #     nn.Conv2D(
            #         in_channels=128 * (2**self.num_outputs) // (2**i),
            #         out_channels=num_filters,
            #         kernel_size=1,
            #         stride=1,
            #         padding=0,
            #         weight_attr=ParamAttr(name=name + '.conv.weights'),
            #         bias_attr=ParamAttr(
            #             name=name + '.conv.bias', regularizer=L2Decay(0.))))

            # self.yolo_output = nn.LayerList([
            #     conv_bn_relu_v2(
            #         in_channels, in_channels, 3, 1, padding=2, dilation=2),
            #     conv_bn_relu_v2(
            #         in_channels, in_channels, 3, 1, padding=4, dilation=4),
            #     conv_bn_relu_v2(
            #         in_channels, in_channels, 3, 1, padding=6, dilation=6),
            #     decode_conv,
            # ])

            # yolo_output = nn.Sequential(
            #     (name + '.a', conv_bn_relu(in_channels, in_channels, 3, 1, 1)),
            #     (name + '.b', conv_bn_relu(in_channels, in_channels, 3, 1, 1)),
            #     (name + '.c', conv_bn_relu(in_channels, in_channels, 3, 1, 1)),
            #     (name, decode_conv), )

            yolo_output = nn.LayerList([
                conv_bn_relu_v2(
                    in_channels, in_channels, 3, 1, padding=2,
                    dilation=2), conv_bn_relu_v2(
                        in_channels, in_channels, 3, 1, padding=4,
                        dilation=4), conv_bn_relu_v2(
                            in_channels,
                            in_channels,
                            3,
                            1,
                            padding=6,
                            dilation=6), conv_bn_relu_v2(
                                in_channels,
                                in_channels,
                                3,
                                1,
                                padding=8,
                                dilation=8),
                nn.Conv2D(128 * (2**self.num_outputs) // (2**i), num_filters, 1,
                          1, 0)
            ])

            self.yolo_outputs.append(yolo_output)

        for n, p in self.named_parameters():
            print(n, p.shape)

        print(len(self.parameters()))

    def parse_anchor(self, anchors, anchor_masks):
        self.anchors = [[anchors[i] for i in mask] for mask in anchor_masks]
        self.mask_anchors = []
        anchor_num = len(anchors)
        for masks in anchor_masks:
            self.mask_anchors.append([])
            for mask in masks:
                assert mask < anchor_num, "anchor mask index overflow"
                self.mask_anchors[-1].extend(anchors[mask])

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.anchors)
        yolo_outputs = []
        for i, feat in enumerate(feats):

            for layer in self.yolo_outputs[i][:4]:
                feat += layer(feat)
            feat = self.yolo_outputs[i][4](feat)
            yolo_outputs.append(feat)

            # yolo_output = self.yolo_outputs[i](feat)
            # yolo_outputs.append(yolo_output)

        if self.training:
            return self.loss(yolo_outputs, targets, self.anchors)
        else:
            if self.iou_aware:
                y = []
                for i, out in enumerate(yolo_outputs):
                    na = len(self.anchors[i])
                    ioup, x = out[:, 0:na, :, :], out[:, na:, :, :]
                    b, c, h, w = x.shape
                    no = c // na
                    x = x.reshape((b, na, no, h * w))
                    ioup = ioup.reshape((b, na, 1, h * w))
                    obj = x[:, :, 4:5, :]
                    ioup = F.sigmoid(ioup)
                    obj = F.sigmoid(obj)
                    obj_t = (obj**(1 - self.iou_aware_factor)) * (
                        ioup**self.iou_aware_factor)
                    obj_t = _de_sigmoid(obj_t)
                    loc_t = x[:, :, :4, :]
                    cls_t = x[:, :, 5:, :]
                    y_t = paddle.concat([loc_t, obj_t, cls_t], axis=2)
                    y_t = y_t.reshape((b, c, h, w))
                    y.append(y_t)
                return y
            else:
                return yolo_outputs
