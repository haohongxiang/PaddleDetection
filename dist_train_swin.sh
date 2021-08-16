
python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py  -c $1 -o find_unused_parameters=True --eval


