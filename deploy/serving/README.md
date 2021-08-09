# 服务端预测部署

`PaddleDetection`训练出来的模型可以使用[Serving](https://github.com/PaddlePaddle/Serving) 部署在服务端。  
<<<<<<< HEAD
本教程以在路标数据集[roadsign_voc](https://paddlemodels.bj.bcebos.com/object_detection/roadsign_voc.tar) 使用`configs/yolov3_mobilenet_v1_roadsign.yml`算法训练的模型进行部署。  
预训练模型权重文件为[yolov3_mobilenet_v1_roadsign.pdparams](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_roadsign.pdparams) 。

## 1. 首先验证模型
```
python tools/infer.py -c configs/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_roadsign.pdparams --infer_img=demo/road554.png
```

## 2. 安装 paddle serving
```
# 安装 paddle-serving-client
pip install paddle-serving-client -i https://mirror.baidu.com/pypi/simple

# 安装 paddle-serving-server
pip install paddle-serving-server -i https://mirror.baidu.com/pypi/simple

# 安装 paddle-serving-server-gpu
pip install paddle-serving-server-gpu -i https://mirror.baidu.com/pypi/simple
```

## 3. 导出模型
PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/master/docs/advanced_tutorials/deploy/EXPORT_MODEL.md)

```
python tools/export_serving_model.py -c configs/yolov3_mobilenet_v1_roadsign.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_roadsign.pdparams --output_dir=./inference_model
```

以上命令会在./inference_model文件夹下生成一个`yolov3_mobilenet_v1_roadsign`文件夹：
```
inference_model
│   ├── yolov3_mobilenet_v1_roadsign
│   │   ├── infer_cfg.yml
=======
本教程以在COCO数据集上用`configs/yolov3/yolov3_darknet53_270e_coco.yml`算法训练的模型进行部署。  
预训练模型权重文件为[yolov3_darknet53_270e_coco.pdparams](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) 。

## 1. 首先验证模型
```
python tools/infer.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml --infer_img=demo/000000014439.jpg -o use_gpu=True weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams --infer_img=demo/000000014439.jpg
```

## 2. 安装 paddle serving
请参考[PaddleServing](https://github.com/PaddlePaddle/Serving/tree/v0.6.0) 中安装教程安装（版本>=0.6.0）。

## 3. 导出模型
PaddleDetection在训练过程包括网络的前向和优化器相关参数，而在部署过程中，我们只需要前向参数，具体参考:[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/EXPORT_MODEL.md)

```
python tools/export_model.py -c configs/yolov3/yolov3_darknet53_270e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams --export_serving_model=True
```

以上命令会在`output_inference/`文件夹下生成一个`yolov3_darknet53_270e_coco`文件夹：
```
output_inference
│   ├── yolov3_darknet53_270e_coco
│   │   ├── infer_cfg.yml
│   │   ├── model.pdiparams
│   │   ├── model.pdiparams.info
│   │   ├── model.pdmodel
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
│   │   ├── serving_client
│   │   │   ├── serving_client_conf.prototxt
│   │   │   ├── serving_client_conf.stream.prototxt
│   │   ├── serving_server
<<<<<<< HEAD
│   │   │   ├── conv1_bn_mean
│   │   │   ├── conv1_bn_offset
│   │   │   ├── conv1_bn_scale
=======
│   │   │   ├── __model__
│   │   │   ├── __params__
│   │   │   ├── serving_server_conf.prototxt
│   │   │   ├── serving_server_conf.stream.prototxt
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
│   │   │   ├── ...
```

`serving_client`文件夹下`serving_client_conf.prototxt`详细说明了模型输入输出信息
`serving_client_conf.prototxt`文件内容为：
```
<<<<<<< HEAD
=======
lient_conf.prototxt
feed_var {
  name: "im_shape"
  alias_name: "im_shape"
  is_lod_tensor: false
  feed_type: 1
  shape: 2
}
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
feed_var {
  name: "image"
  alias_name: "image"
  is_lod_tensor: false
  feed_type: 1
  shape: 3
  shape: 608
  shape: 608
}
feed_var {
<<<<<<< HEAD
  name: "im_size"
  alias_name: "im_size"
  is_lod_tensor: false
  feed_type: 2
  shape: 2
}
fetch_var {
  name: "multiclass_nms_0.tmp_0"
  alias_name: "multiclass_nms_0.tmp_0"
=======
  name: "scale_factor"
  alias_name: "scale_factor"
  is_lod_tensor: false
  feed_type: 1
  shape: 2
}
fetch_var {
  name: "save_infer_model/scale_0.tmp_1"
  alias_name: "save_infer_model/scale_0.tmp_1"
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
  is_lod_tensor: true
  fetch_type: 1
  shape: -1
}
<<<<<<< HEAD
=======
fetch_var {
  name: "save_infer_model/scale_1.tmp_1"
  alias_name: "save_infer_model/scale_1.tmp_1"
  is_lod_tensor: true
  fetch_type: 2
  shape: -1
}
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
```

## 4. 启动PaddleServing服务

```
<<<<<<< HEAD
cd inference_model/yolov3_mobilenet_v1_roadsign/

# GPU
python -m paddle_serving_server_gpu.serve --model serving_server --port 9393 --gpu_ids 0
=======
cd output_inference/yolov3_darknet53_270e_coco/

# GPU
python -m paddle_serving_server.serve --model serving_server --port 9393 --gpu_ids 0
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

# CPU
python -m paddle_serving_server.serve --model serving_server --port 9393
```

## 5. 测试部署的服务
准备`label_list.txt`文件
```
# 进入到导出模型文件夹
<<<<<<< HEAD
cd inference_model/yolov3_mobilenet_v1_roadsign/

# 将数据集对应的label_list.txt文件拷贝到当前文件夹下
cp ../../dataset/roadsign_voc/label_list.txt .
```

设置`prototxt`文件路径为`serving_client/serving_client_conf.prototxt` 。  
设置`fetch`为`fetch=["multiclass_nms_0.tmp_0"])`
=======
cd output_inference/yolov3_darknet53_270e_coco/

# 将数据集对应的label_list.txt文件放到当前文件夹下
```

设置`prototxt`文件路径为`serving_client/serving_client_conf.prototxt` 。  
设置`fetch`为`fetch=["save_infer_model/scale_0.tmp_1"])`
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

测试
```
# 进入目录
<<<<<<< HEAD
cd inference_model/yolov3_mobilenet_v1_roadsign/

# 测试代码 test_client.py 会自动创建output文件夹，并在output下生成`bbox.json`和`road554.png`两个文件
python ../../deploy/serving/test_client.py ../../demo/road554.png
=======
cd output_inference/yolov3_darknet53_270e_coco/

# 测试代码 test_client.py 会自动创建output文件夹，并在output下生成`bbox.json`和`000000014439.jpg`两个文件
python ../../deploy/serving/test_client.py ../../demo/000000014439.jpg
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
```
