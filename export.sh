

python tools/export_model.py -c configs/backbone/resnet50.yml
python deploy/python/infer.py  --run_benchmark=True --run_mode=fluid --device=GPU --threshold=0.5 --output_dir=python_infer_output --image_dir=./demo --model_dir=output_inference/  
