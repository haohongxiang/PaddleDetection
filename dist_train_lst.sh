
ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9

sleep 3

python -m paddle.distributed.launch --log_dir=../logs --selected_gpu 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyolo/lst_ppyolo_r50vd_dcn.yml --eval

