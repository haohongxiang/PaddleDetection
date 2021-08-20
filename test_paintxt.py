

from ppdet.core.workspace import create, load_config


cfg = load_config("configs/paintxt/dataset.yml")

dataset = cfg['TrainDataset']

# dataset.set_kwargs()
# dataset.set_epoch()

# print(dataset)
print(len(dataset))

print(dataset.roidbs[0])
