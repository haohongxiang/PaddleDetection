from plistlib import load
import paddle
from ppdet.core.workspace import load_config, create

cfg = load_config('./configs/faster_rcnn/vit_faster_rcnn.yml')
model = create(cfg.architecture)

# print(model.backbone)

data = paddle.rand([2, 3, 640, 640])
outputs = model(data)

for out in outputs:
    print(out.shape)
