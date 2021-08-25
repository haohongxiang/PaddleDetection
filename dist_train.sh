# python tools/train.py -c ./configs/gfl/gfl_r50_pan_1x_coco.ym

python -m paddle.distributed.launch --log_dir=../logs/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/gfl_r50_pan_1x_coco.yml  --eval


