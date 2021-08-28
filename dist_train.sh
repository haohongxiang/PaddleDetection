# python tools/train.py -c ./configs/gfl/gfl_r50_pan_1x_coco.ym
ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9
sleep 3
rm -rf /dev/shm/paddle*
sleep 3


python -m paddle.distributed.launch --log_dir=../logs/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/gfl_r50_pan_3x_coco.yml  --eval
# python -m paddle.distributed.launch --log_dir=../logs/ --selected_gpu 0,1,2,3  tools/train.py -c ./configs/gfl/gfl_r50_pan_sync_bn_1x_coco.yml  --eval


