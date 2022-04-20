ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch --log_dir=logs30 --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov5/yolov5_s_coco_30.yml --eval > trains30.log 2>&1 &


