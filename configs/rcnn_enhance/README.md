## 服务器端实用目标检测方案

### 简介

* 近年来，学术界和工业界广泛关注图像中目标检测任务。基于[PaddleClas](https://github.com/PaddlePaddle/PaddleClas)中SSLD蒸馏方案训练得到的ResNet50_vd预训练模型(ImageNet1k验证集上Top1 Acc为82.39%)，结合PaddleDetection中的丰富算子，飞桨提供了一种面向服务器端实用的目标检测方案PSS-DET(Practical Server Side Detection)。基于COCO2017目标检测数据集，V100单卡预测速度为为61FPS时，COCO mAP可达41.2%。


### 模型库

| 骨架网络             | 网络类型       | 每张GPU图片个数 | 学习率策略 |推理时间(fps) | Box AP | Mask AP |                           下载                          | 配置文件 |
| :---------------------- | :-------------:  | :-------: | :-----: | :------------: | :----: | :-----: | :-------------: | :-----: |
<<<<<<< HEAD
| ResNet50-vd-FPN-Dcnv2         | Faster     |     2     |   3x    |     61.425     |  41.6  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) |  [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_enhance/faster_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
| ResNet50-vd-FPN-Dcnv2         | Cascade Faster     |     2     |   3x    |     20.001     |  47.8  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.tar) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_enhance/cascade_rcnn_dcn_r50_vd_fpn_3x_server_side.yml) |
| ResNet101-vd-FPN-Dcnv2         | Cascade Faster     |     2     |   3x    |     19.523     |  49.4  |    -    | [下载链接](https://paddlemodels.bj.bcebos.com/object_detection/cascade_rcnn_dcn_r101_vd_fpn_3x_server_side.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/master/configs/rcnn_enhance/cascade_rcnn_dcn_r101_vd_fpn_3x_server_side.yml) |


**注**：generic文件夹下面的配置文件对应的预训练模型均只支持预测，不支持训练与评估。
=======
| ResNet50-vd-FPN-Dcnv2         | Faster     |     2     |   3x    |     61.425     |  41.5  |    -    | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_enhance_3x_coco.pdparams) | [配置文件](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/rcnn_enhance/faster_rcnn_enhance_3x_coco.yml) |
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
