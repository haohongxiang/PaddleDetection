English | [简体中文](README_cn.md)

### PaddleDetection 2.0 is ready! Dygraph mode is set by default and static graph code base is [here](static)

### [Keypoint detection](configs/keypoint) and [Multi-Object Tracking](configs/mot) are released!

### Highly effective PPYOLO v2 and ultra lightweight PPYOLO tiny are released! [link](configs/ppyolo/README.md)

<<<<<<< HEAD
[PP-YOLO](https://arxiv.org/abs/2007.12099), which is faster and has higer performance than YOLOv4,
has been released, it reached mAP(0.5:0.95) as 45.2%(newest 45.9%) on COCO test2019 dataset and
72.9 FPS on single Test V100. Please refer to [PP-YOLO](configs/ppyolo/README.md) for details.

**Now all models in PaddleDetection require PaddlePaddle version 1.8 or higher, or suitable develop version.**
=======
### SOTA Anchor Free model -- PAFNet is released! [link](configs/ttfnet/README.md)
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

# Introduction

PaddleDetection is an end-to-end object detection development kit based on PaddlePaddle, which aims to help developers in the whole development of constructing, training, optimizing and deploying detection models in a faster and better way.

PaddleDetection implements varied mainstream object detection algorithms in modular design, and provides wealthy data augmentation methods, network components(such as backbones), loss functions, etc., and integrates abilities of model compression and cross-platform high-performance deployment.

After a long time of industry practice polishing, PaddleDetection has had smooth and excellent user experience, it has been widely used by developers in more than ten industries such as industrial quality inspection, remote sensing image object detection, automatic inspection, new retail, Internet, and scientific research.

<div align="center">
  <img src="static/docs/images/football.gif" width='800'/>
  <img src="docs/images/mot_pose_demo_640x360.gif" width='800'/>
</div>

### Product news

- 2021.05.20: Release `release/2.1` version. Release [Keypoint Detection](configs/keypoint), including HigherHRNet and HRNet, [Multi-Object Tracking](configs/mot), including DeepSORT，JDE and FairMOT. Release model compression for PPYOLO series models.Update documents such as [EXPORT ONNX MODEL](deploy/EXPORT_ONNX_MODEL.md). Please refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1) for details.
- 2021.04.14: Release `release/2.0` version. Dygraph mode in PaddleDetection is fully supported. Cover all the algorithm of static graph and update the performance of mainstream detection models. Release [`PP-YOLO v2` and `PP-YOLO tiny`](configs/ppyolo/README.md), enhanced anchor free model [PAFNet](configs/ttfnet/README.md) and [`S2ANet`](configs/dota/README.md) which is aimed at rotation object detection.Please refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0) for details.
- 2020.02.07: Release `release/2.0-rc` version, Please refer to [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0-rc) for details.


### Features

- **Rich Models**
PaddleDetection provides rich of models, including **100+ pre-trained models** such as **object detection**, **instance segmentation**, **face detection** etc. It covers a variety of **global competition champion** schemes.

- **Highly Flexible:**
Components are designed to be modular. Model architectures, as well as data preprocess pipelines and optimization strategies, can be easily customized with simple configuration changes.

- **Production Ready:**
From data augmentation, constructing models, training, compression, depolyment, get through end to end, and complete support for multi-architecture, multi-device deployment for **cloud and edge device**.

- **High Performance:**
Based on the high performance core of PaddlePaddle, advantages of training speed and memory occupation are obvious. FP16 training and multi-machine training are supported as well.

#### Overview of Kit Structures

<table>
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Architectures</b>
      </td>
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Components</b>
      </td>
      <td>
        <b>Data Augmentation</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul><li><b>Two-Stage Detection</b></li>
          <ul>
            <li>Faster RCNN</li>
            <li>FPN</li>
            <li>Cascade-RCNN</li>
            <li>Libra RCNN</li>
            <li>Hybrid Task RCNN</li>
            <li>PSS-Det RCNN</li>
          </ul>
        </ul>
        <ul><li><b>One-Stage Detection</b></li>
          <ul>
            <li>RetinaNet</li>
            <li>YOLOv3</li>
            <li>YOLOv4</li>  
            <li>PP-YOLO</li>
            <li>SSD</li>
          </ul>
        </ul>
        <ul><li><b>Anchor Free</b></li>
          <ul>
            <li>CornerNet-Squeeze</li>
            <li>FCOS</li>  
            <li>TTFNet</li>
          </ul>
        </ul>
        <ul>
          <li><b>Instance Segmentation</b></li>
            <ul>
             <li>Mask RCNN</li>
             <li>SOLOv2</li>
            </ul>
        </ul>
        <ul>
          <li><b>Face-Detction</b></li>
            <ul>
             <li>FaceBoxes</li>
             <li>BlazeFace</li>
             <li>BlazeFace-NAS</li>
            </ul>
        </ul>
      </td>
      <td>
        <ul>
          <li>ResNet(&vd)</li>
          <li>ResNeXt(&vd)</li>
          <li>SENet</li>
          <li>Res2Net</li>
          <li>HRNet</li>
          <li>Hourglass</li>
          <li>CBNet</li>
          <li>GCNet</li>
          <li>DarkNet</li>
          <li>CSPDarkNet</li>
          <li>VGG</li>
          <li>MobileNetv1/v3</li>  
          <li>GhostNet</li>
          <li>Efficientnet</li>  
        </ul>
      </td>
      <td>
        <ul><li><b>Common</b></li>
          <ul>
            <li>Sync-BN</li>
            <li>Group Norm</li>
            <li>DCNv2</li>
            <li>Non-local</li>
          </ul>  
        </ul>
        <ul><li><b>FPN</b></li>
          <ul>
            <li>BiFPN</li>
            <li>BFP</li>  
            <li>HRFPN</li>
            <li>ACFPN</li>
          </ul>  
        </ul>  
        <ul><li><b>Loss</b></li>
          <ul>
            <li>Smooth-L1</li>
            <li>GIoU/DIoU/CIoU</li>  
            <li>IoUAware</li>
          </ul>  
        </ul>  
        <ul><li><b>Post-processing</b></li>
          <ul>
            <li>SoftNMS</li>
            <li>MatrixNMS</li>  
          </ul>  
        </ul>
        <ul><li><b>Speed</b></li>
          <ul>
            <li>FP16 training</li>
            <li>Multi-machine training </li>  
          </ul>  
        </ul>  
      </td>
      <td>
        <ul>
          <li>Resize</li>  
          <li>Flipping</li>  
          <li>Expand</li>
          <li>Crop</li>
          <li>Color Distort</li>  
          <li>Random Erasing</li>  
          <li>Mixup </li>
          <li>Cutmix </li>
          <li>Grid Mask</li>
          <li>Auto Augment</li>  
        </ul>  
      </td>  
    </tr>


</td>
    </tr>
  </tbody>
</table>

#### Overview of Model Performance
The relationship between COCO mAP and FPS on Tesla V100 of representative models of each architectures and backbones.

<div align="center">
  <img src="docs/images/fps_map.png" />
</div>

**NOTE:**

- `CBResNet stands` for `Cascade-Faster-RCNN-CBResNet200vd-FPN`, which has highest mAP on COCO as 53.3%

- `Cascade-Faster-RCNN` stands for `Cascade-Faster-RCNN-ResNet50vd-DCN`, which has been optimized to 20 FPS inference speed when COCO mAP as 47.8% in PaddleDetection models

- `PP-YOLO` achieves mAP of 45.9% on COCO and 72.9FPS on Tesla V100. Both precision and speed surpass [YOLOv4](https://arxiv.org/abs/2004.10934)

- `PP-YOLO v2` is optimized version of `PP-YOLO` which has mAP of 49.5% and 68.9FPS on Tesla V100

<<<<<<< HEAD
- EfficientDet
- FCOS
- CornerNet-Squeeze
- YOLOv4
- PP-YOLO
=======
- All these models can be get in [Model Zoo](#ModelZoo)
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9


## Tutorials

### Get Started

- [Installation guide](docs/tutorials/INSTALL_en.md)
- [Prepare dataset](docs/tutorials/PrepareDataSet.md)
- [Quick start on PaddleDetection](docs/tutorials/GETTING_STARTED_cn.md)


### Advanced Tutorials

- Parameter configuration
  - [Parameter configuration for RCNN model](docs/tutorials/config_annotation/faster_rcnn_r50_fpn_1x_coco_annotation.md)
  - [Parameter configuration for PP-YOLO model](docs/tutorials/config_annotation/ppyolo_r50vd_dcn_1x_coco_annotation.md)

- Model Compression(Based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim))
  - [Prune/Quant/Distill](configs/slim)

<<<<<<< HEAD
The following is the relationship between COCO mAP and FPS on Tesla V100 of SOTA object detecters and PP-YOLO, which is faster and has better performance than YOLOv4, and reached mAP(0.5:0.95) as 45.2% on COCO test2019 dataset and 72.9 FPS on single Test V100. Please refer to [PP-YOLO](configs/ppyolo/README.md) for details.
=======
- Inference and deployment
  - [Export model for inference](deploy/EXPORT_MODEL.md)
  - [Paddle Inference](deploy/README.md)
      - [Python inference](deploy/python)
      - [C++ inference](deploy/cpp)
  - [Paddle-Lite](deploy/lite)
  - [Paddle Serving](deploy/serving)
  - [Export ONNX model](deploy/EXPORT_ONNX_MODEL.md)
  - [Inference benchmark](deploy/BENCHMARK_INFER.md)
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

- Advanced development
  - [New data augmentations](docs/advanced_tutorials/READER.md)
  - [New detection algorithms](docs/advanced_tutorials/MODEL_TECHNICAL.md)


## Model Zoo

- Universal object detection
  - [Model library and baselines](docs/MODEL_ZOO_cn.md)
  - [PP-YOLO](configs/ppyolo/README.md)
  - [Enhanced Anchor Free model--TTFNet](configs/ttfnet/README.md)
  - [Mobile models](static/configs/mobile/README.md)
  - [676 classes of object detection](static/docs/featured_model/LARGE_SCALE_DET_MODEL.md)
  - [Two-stage practical PSS-Det](configs/rcnn_enhance/README.md)
  - [SSLD pretrained models](docs/feature_models/SSLD_PRETRAINED_MODEL_en.md)
- Universal instance segmentation
  - [SOLOv2](configs/solov2/README.md)
- Rotation object detection
  - [S2ANet](configs/dota/README.md)
- [Keypoint detection](configs/keypoint)
  - HigherHRNet
  - HRNet
  - LiteHRNet
- [Multi-Object Tracking](configs/mot/README.md)
  - [DeepSORT](configs/mot/deepsort/README.md)
  - [JDE](configs/mot/jde/README.md)
  - [FairMOT](configs/mot/fairmot/README.md)
- Vertical field
  - [Face detection](configs/face_detection/README.md)
  - [Pedestrian detection](configs/pedestrian/README.md)
  - [Vehicle detection](configs/vehicle/README.md)
- Competition Plan
  - [Objects365 2019 Challenge champion model](static/docs/featured_model/champion_model/CACascadeRCNN.md)
  - [Best single model of Open Images 2019-Object Detction](static/docs/featured_model/champion_model/OIDV5_BASELINE_MODEL.md)

## Applications

- [Christmas portrait automatic generation tool](static/application/christmas)

<<<<<<< HEAD
- [Installation guide](docs/tutorials/INSTALL.md)
- [Quick start on small dataset](docs/tutorials/QUICK_STARTED.md)
- [Prepare dataset](docs/tutorials/PrepareDataSet.md)
- [Train/Evaluation/Inference/Deploy](docs/tutorials/DetectionPipeline.md)
- [How to train a custom dataset](docs/tutorials/Custom_DataSet.md)
- [FAQ](docs/FAQ.md)

### Advanced Tutorial

- [Guide to preprocess pipeline and dataset definition](docs/advanced_tutorials/READER.md)
- [Models technical](docs/advanced_tutorials/MODEL_TECHNICAL.md)
- [Transfer learning document](docs/advanced_tutorials/TRANSFER_LEARNING.md)
- [Parameter configuration](docs/advanced_tutorials/config_doc):
  - [Introduction to the configuration workflow](docs/advanced_tutorials/config_doc/CONFIG.md)
  - [Parameter configuration for RCNN model](docs/advanced_tutorials/config_doc/RCNN_PARAMS_DOC.md)
  - [Parameter configuration for YOLOv3 model](docs/advanced_tutorials/config_doc/yolov3_mobilenet_v1.md)
- [IPython Notebook demo](demo/mask_rcnn_demo.ipynb)
- [Model compression](slim)
    - [Model compression benchmark](slim)
    - [Quantization](slim/quantization)
    - [Model pruning](slim/prune)
    - [Model distillation](slim/distillation)
    - [Neural Architecture Search](slim/nas)
- [Deployment](deploy)
    - [Export model for inference](docs/advanced_tutorials/deploy/EXPORT_MODEL.md)
    - [Python inference](deploy/python)
    - [C++ inference](deploy/cpp)
    - [Mobile](https://github.com/PaddlePaddle/Paddle-Lite-Demo)
    - [Serving](deploy/serving)
    - [Inference benchmark](docs/advanced_tutorials/deploy/BENCHMARK_INFER_cn.md)
=======
## Updates
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

v2.1 was released at `05/2021`, Release Keypoint Detection and Multi-Object Tracking. Release model compression for PPYOLO series. Update documents such as export ONNX model. Please refer to [change log](docs/CHANGELOG.md) for details.

<<<<<<< HEAD
- Pretrained models are available in the [PaddleDetection model zoo](docs/MODEL_ZOO.md).
- [Mobile models](configs/mobile/README.md)
- [Anchor free models](configs/anchor_free/README.md)
- [Face detection models](docs/featured_model/FACE_DETECTION_en.md)
- [Pretrained models for pedestrian detection](docs/featured_model/CONTRIB.md)
- [Pretrained models for vehicle detection](docs/featured_model/CONTRIB.md)
- [YOLOv3 enhanced model](docs/featured_model/YOLOv3_ENHANCEMENT.md): Compared to MAP of 33.0% in paper, enhanced YOLOv3 reaches the MAP of 43.6%, and inference speed is improved as well
- [PP-YOLO](configs/ppyolo/README.md): PP-YOLO reeached mAP as 45.3% on COCO dataset，and 72.9 FPS on single Tesla V100
- [Objects365 2019 Challenge champion model](docs/featured_model/champion_model/CACascadeRCNN.md)
- [Best single model of Open Images 2019-Object Detction](docs/featured_model/champion_model/OIDV5_BASELINE_MODEL.md)
- [Practical Server-side detection method](configs/rcnn_enhance/README_en.md): Inference speed on single V100 GPU can reach 20FPS when COCO mAP is 47.8%.
- [Large-scale practical object detection models](docs/featured_model/LARGE_SCALE_DET_MODEL_en.md): Large-scale practical server-side detection pretrained models with 676 categories are provided for most application scenarios, which can be used not only for direct inference but also finetuning on other datasets.
=======
v2.0 was released at `04/2021`, fully support dygraph version, which add BlazeFace, PSS-Det and plenty backbones, release `PP-YOLOv2`, `PP-YOLO tiny` and `S2ANet`, support model distillation and VisualDL, add inference benchmark, etc. Please refer to [change log](docs/CHANGELOG.md) for details.
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9


## License

PaddleDetection is released under the [Apache 2.0 license](LICENSE).

<<<<<<< HEAD
## Updates
v0.4.0 was released at `05/2020`, add PP-YOLO, TTFNet, HTC, ACFPN, etc. And add BlaceFace face landmark detection model, add a series of optimized SSDLite models on mobile side, add data augmentations GridMask and RandomErasing, add Matrix NMS and EMA training, and improved ease of use, fix many known bugs, etc.
Please refer to [版本更新文档](docs/CHANGELOG.md) for details.
=======
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

## Contributing

Contributions are highly welcomed and we would really appreciate your feedback!!
- Thanks [Mandroide](https://github.com/Mandroide) for cleaning the code and unifying some function interface.
- Thanks [FL77N](https://github.com/FL77N/) for contributing the code of `Sparse-RCNN` model.

## Citation

```
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}
```
