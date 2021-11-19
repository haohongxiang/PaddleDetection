
# wget http://10.181.196.20:8787/workspace/ssd6/lvwenyu01/workspace/dataset/CSPDarkNet53_pretrained.pdparams

rm /dev/shm/paddle_* -rf


ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

python -m paddle.distributed.launch --log_dir=log   --gpus 0,1,2,3 tools/train.py -c configs/picodet/pp_yolo.yml --eval  > train.log 2>&1 &

