

rm /dev/shm/paddle_* -rf

ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

# python -m paddle.distributed.launch --log_dir=log   --gpus 4,5,6,7 tools/train.py -c configs/picodet/pp_fcos.yml --eval
 python -m paddle.distributed.launch --log_dir=log_yolov5 --gpus 4,5,6,7 tools/train.py -c configs/yolov5/yolov5l_coco.yml --eval

