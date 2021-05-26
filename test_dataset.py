import paddle
from ppdet.core.workspace import load_config, create, register


cfg = load_config("configs/testjson/dataset.yml")

dataset = cfg['TrainDataset']

print(dataset)
