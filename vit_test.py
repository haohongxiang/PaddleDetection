import paddle
from ppdet.core.workspace import load_config, create

cfg = load_config('./configs/faster_rcnn/vit_base_16_faster_rcnn.yml')

model = create(cfg.architecture)
model = model.backbone

data = paddle.rand([2, 3, 640, 640])
outputs = model(data)

print(model)

for out in outputs:
    print(out.shape)
