
ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 


python -m paddle.distributed.launch --log_dir=./logs --selected_gpu 0,1,2,3,4,5,6,7 tools/train.py -c configs/faster_rcnn/vit_base_16_faster_rcnn.yml  --eval &>logs.txt 2>&1 &
