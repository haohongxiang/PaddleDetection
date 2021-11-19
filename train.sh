
ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 
# python -m paddle.distributed.launch --log_dir=log   --gpus 4,5,6,7 tools/train.py -c configs/picodet/pp_fcos.yml --eval
python -m paddle.distributed.launch --log_dir=../logs/   --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/htc/htc_r50_fpn_1x_coco.yml -r ./output/htc_r50_fpn_1x_coco/10.pdparams --eval

