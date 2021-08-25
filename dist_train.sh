
python -m paddle.distributed.launch --log_dir=../logs/ --selected_gpu 0,1,2,3  tools/train.py -c ./configs/gfl/gfl_r50_pan_1x_coco.yml 

