ps aux | grep tools/train.py | awk '{print $2}' |xargs kill -9

python -m paddle.distributed.launch --log_dir=./yolov5m/ --gpus 0,1,2,3 tools/train.py -c configs/yolov5/yolov5m_coco.yml

