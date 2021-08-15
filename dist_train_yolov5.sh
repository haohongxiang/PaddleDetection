rm /dev/shm/* -rf

ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9

sleep 3

python -m paddle.distributed.launch --log_dir=./yolov5$1/ --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov5/yolov5$1_coco.yml --eval

