
# wget http://10.181.196.20:8787/workspace/ssd6/lvwenyu01/workspace/dataset/CSPDarkNet53_pretrained.pdparams


ps aux | grep "tools/train.py" | awk '{print $2}' | xargs kill -9 

# python -m paddle.distributed.launch --log_dir=log   --gpus 4,5,6,7 tools/train.py -c configs/picodet/pp_fcos.yml --eval
 python -m paddle.distributed.launch --log_dir=log_csp   --gpus 4,5,6,7 tools/train.py -c configs/picodet/pp_fcos_cspdarknet.yml --eval



# python tools/export_model.py -c configs/picodet/pp_fcos_cspdarknet.yml
# python deploy/python/infer.py --model_dir=output_inference/pp_fcos_cspdarknet/ --image_dir=./demo/ --run_mode=fluid --device=GPU --threshold=0.5 
