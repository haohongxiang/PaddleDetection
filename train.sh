 
 
 
 
ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/htc/htc_swin_tiny_fpn_1x_coco.yml --eval 

# --eval > train.log 2>&1 &