[English](INSTALL.md) | 简体中文


<<<<<<< HEAD
- [简介](#简介)
- [安装PaddlePaddle](#安装PaddlePaddle)
- [安装COCO-API](#安装COCO-API)
- [PaddleDetection](#PaddleDetection)
=======
# 安装文档
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9



<<<<<<< HEAD
这份文档介绍了如何安装PaddleDetection及其依赖项(包括PaddlePaddle)。

PaddleDetection的相关信息，请参考[README.md](https://github.com/PaddlePaddle/PaddleDetection/blob/master/README.md).
=======
## 环境要求

- PaddlePaddle 2.1
- OS 64位操作系统
- Python 3(3.5.1+/3.6/3.7/3.8/3.9)，64位版本
- pip/pip3(9.0.1+)，64位版本
- CUDA >= 10.1
- cuDNN >= 7.6
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

PaddleDetection 依赖 PaddlePaddle 版本关系：

<<<<<<< HEAD
## 安装PaddlePaddle

**环境需求:**

- OS 64位操作系统
- Python2 >= 2.7.15 or Python 3(3.5.1+/3.6/3.7)，64位版本
- pip/pip3(9.0.1+)，64位版本操作系统是
- CUDA >= 9.0
- cuDNN >= 7.6

如果需要 GPU 多卡训练，请先安装NCCL(Windows暂不支持nccl)。

PaddleDetection 依赖 PaddlePaddle 版本关系：

| PaddleDetection版本 | PaddlePaddle版本  |    备注    |
| :----------------: | :---------------: | :-------: |
|      v0.3          |        >=1.7      |     --    |
|      v0.4          |       >= 1.8.4    |  PP-YOLO依赖1.8.4 |


```
# install paddlepaddle
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

# 如果您的机器安装的是CUDA9，请运行以下命令安装
python -m pip install paddlepaddle-gpu==1.8.4.post97 -i https://mirror.baidu.com/pypi/simple

如果您的机器安装的是CUDA10，请运行以下命令安装
python -m pip install paddlepaddle-gpu==1.8.4.post107 -i https://mirror.baidu.com/pypi/simple

如果您的机器是CPU，请运行以下命令安装

python -m pip install paddlepaddle==1.8.4 -i https://mirror.baidu.com/pypi/simple
```
更多的安装方式如conda, docker安装，请参考[安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作
=======
|  PaddleDetection版本  | PaddlePaddle版本  |    备注    |
| :------------------: | :---------------: | :-------: |
|    release/2.1       |       >= 2.1.0    |     默认使用动态图模式    |
|    release/2.0       |       >= 2.0.1    |     默认使用动态图模式    |
|    release/2.0-rc    |       >= 2.0.1    |     --    |
|    release/0.5       |       >= 1.8.4    |  大部分模型>=1.8.4即可运行，Cascade R-CNN系列模型与SOLOv2依赖2.0.0.rc版本 |
|    release/0.4       |       >= 1.8.4    |  PP-YOLO依赖1.8.4 |
|    release/0.3       |        >=1.7      |     --    |

## 安装说明

### 1. 安装PaddlePaddle

```
# CUDA10.1
python -m pip install paddlepaddle-gpu==2.1.0.post101 -f https://paddlepaddle.org.cn/whl/mkl/stable.html

# CPU
python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
```
- 更多CUDA版本或环境快速安装，请参考[PaddlePaddle快速安装文档](https://www.paddlepaddle.org.cn/install/quick)
- 更多安装方式例如conda或源码编译安装方法，请参考[PaddlePaddle安装文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

请确保您的PaddlePaddle安装成功并且版本不低于需求版本。使用以下命令进行验证。

```
# 在您的Python解释器中确认PaddlePaddle安装成功
>>> import paddle
>>> paddle.utils.run_check()

# 确认PaddlePaddle版本
python -c "import paddle; print(paddle.__version__)"
```
**注意**
1. 如果您希望在多卡环境下使用PaddleDetection，请首先安装NCCL

<<<<<<< HEAD

## 安装COCO-API

`PaddleDetection`在评估时若使用COCO评估标准，则需要安装[COCO-API](https://github.com/cocodataset/cocoapi) ，安装方式如下：

    # 安装pycocotools。若使用了虚拟环境，请使用虚拟环境中的pip，或者指定pip绝对路径进行安装
    pip install pycocotools

**如果windows用户按照上面方式安装COCO-API出错，可参考以下方式：**

- 安装 pycocotools-windows
    ```
    # pip install pycocotools-windows
    pip install pycocotools-windows
    ```

    - 如果您遇到ssl问题，请参考[conda issue #8273](https://github.com/conda/conda/issues/8273)  
    - 如果您的网络无法下载安装包：  
        - 设置pip源[pip issue #1736](https://github.com/pypa/pip/issues/1736) 。  
        - 将安装包下载到本地安装。从[pycocotools-windows](https://pypi.org/project/pycocotools-windows/#files) 下载对应安装包，在本地安装：  
        ```
        # 例如安装 pycocotools_windows-2.0.0.2-cp37-cp37m-win_amd64.whl
        pip install pycocotools_windows-2.0.0.2-cp37-cp37m-win_amd64.whl
        ```

- 从`pycocotools`源码安装
    ```
    # 若Cython未安装，请安装Cython
    pip install Cython

    # 由于原版cocoapi不支持windows，采用第三方实现版本，该版本仅支持Python3
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    ```

安装完成后，验证是否安装成功：
```
python -c "import pycocotools"
```

## PaddleDetection

**安装Python依赖库：**

Python依赖库在[requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/master/requirements.txt) 中给出，可通过如下命令安装：
=======
### 2. 安装PaddleDetection




**注意：** pip安装方式只支持Python3



```
# 克隆PaddleDetection仓库
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git

# 编译安装paddledet
cd PaddleDetection
python setup.py install
>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9

# 安装其他依赖
pip install -r requirements.txt
```

<<<<<<< HEAD
**克隆PaddleDetection库：**

您可以通过以下命令克隆PaddleDetection：

```
cd <path/to/clone/PaddleDetection>
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```
**提示：**

也可以通过 [https://gitee.com/paddlepaddle/PaddleDetection](https://gitee.com/paddlepaddle/PaddleDetection) 克隆。
```
cd <path/to/clone/PaddleDetection>
git clone https://gitee.com/paddlepaddle/PaddleDetection
```

**确认测试通过：**

```
python ppdet/modeling/tests/test_architectures.py
```

测试通过后会提示如下信息：
```
..........
----------------------------------------------------------------------
Ran 12 tests in 2.480s
OK (skipped=2)
```

**预训练模型预测**

使用预训练模型预测图像，快速体验模型预测效果：

```
# use_gpu参数设置是否使用GPU
python tools/infer.py -c configs/ppyolo/ppyolo.yml -o use_gpu=true weights=https://paddlemodels.bj.bcebos.com/object_detection/ppyolo.pdparams --infer_img=demo/000000014439_640x640.jpg
```

会在`output`文件夹下生成一个画有预测结果的同名图像。

结果如下图：

=======
**注意**
1. 如果github下载代码较慢，可尝试使用[gitee](https://gitee.com/PaddlePaddle/PaddleDetection.git)或者[代理加速](https://doc.fastgit.org/zh-cn/guide.html)。

1. 若您使用的是Windows系统，由于原版cocoapi不支持Windows，`pycocotools`依赖可能安装失败，可采用第三方实现版本，该版本仅支持Python3

    ```pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI```

2. 若您使用的是Python <= 3.6的版本，安装`pycocotools`可能会报错`distutils.errors.DistutilsError: Could not find suitable distribution for Requirement.parse('cython>=0.27.3')`, 您可通过先安装`cython`如`pip install cython`解决该问题


安装后确认测试通过：

```
python ppdet/modeling/tests/test_architectures.py
```

测试通过后会提示如下信息：

```
.....
----------------------------------------------------------------------
Ran 5 tests in 4.280s
OK
```

## 快速体验

**恭喜！** 您已经成功安装了PaddleDetection，接下来快速体验目标检测效果

```
# 在GPU上预测一张图片
export CUDA_VISIBLE_DEVICES=0
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

会在`output`文件夹下生成一个画有预测结果的同名图像。

结果如下图：

>>>>>>> 879c90b6d0420410973f5e22932417d174ef45a9
![](../images/000000014439.jpg)
