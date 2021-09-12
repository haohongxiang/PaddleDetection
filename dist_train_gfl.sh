# python tools/train.py -c ./configs/gfl/gfl_r50_pan_1x_coco.ym


# ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9
# sleep 3
# rm -rf /dev/shm/paddle*
# sleep 3

# # python -m paddle.distributed.launch --log_dir=../logs/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/fgl_mosaic_3x_coco.yml  --eval

# python -m paddle.distributed.launch --log_dir=../logs_bifpn/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/fgl_bifpn_3stage_mosaic_3x_coco.yml --eval



# sleep 3m

ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9
sleep 3
rm -rf /dev/shm/paddle*
sleep 3

python -m paddle.distributed.launch --log_dir=../logs_bifpn_3x/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/fgl_bifpn_3stage_mosaic_3x_coco.yml --eval




# sleep 8h
# ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9
# sleep 3
# rm -rf /dev/shm/paddle*
# sleep 3

# python -m paddle.distributed.launch --log_dir=../logs_bifpn_5x/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/fgl_bifpn_3stage_mosaic_5x_coco.yml --eval



# ps aux | grep tools/train.py | awk '{print $2}' | xargs kill -9
# sleep 3
# rm -rf /dev/shm/paddle*
# sleep 3

# # python -m paddle.distributed.launch --log_dir=../logs/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/fgl_mosaic_3x_coco.yml  --eval

# python -m paddle.distributed.launch --log_dir=../logs_bifpn_ms_cell6/ --selected_gpu 0,1,2,3,4,5,6,7  tools/train.py -c ./configs/gfl/fgl_bifpn_3stage_mosaic_ms_cell6_3x_coco.yml --eval