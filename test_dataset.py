import paddle
from ppdet.core.workspace import load_config, create, register


cfg = load_config("configs/testdata/testjson.yml")
dataset = cfg['TrainDataset']
print(dataset)



cfg = load_config("configs/testdata/testvoc.yml")
dataset = cfg['TrainDataset']

print(dataset)
