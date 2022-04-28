import paddle
from ppdet.core.workspace import load_config, create

if False:
    cfg = load_config('./configs/faster_rcnn/vit_base_16_faster_rcnn.yml')
else:
    cfg = load_config('./configs/cascade_rcnn/vit_base_16_hrfpn_coco.yml')

data = paddle.rand([2, 3, 640, 640])

model = create(cfg.architecture)
backbone = model.backbone
neck = model.neck

outputs = backbone(data)
outputs = neck(outputs)

print(backbone)
print(neck)

for out in outputs:
    print(out.shape)
