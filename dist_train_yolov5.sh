ps aux | grep tools/train.py | awk '{print $2}' |xargs kill -9

python -m paddle.distributed.launch --log_dir=./yolov5$1/ --gpus 0,1,2,3,4,5,6,7,8 tools/train.py -c configs/yolov5/yolov5$1_coco.yml

