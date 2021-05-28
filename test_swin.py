import paddle
from ppdet.core.workspace import load_config, create, register



# cfg = load_config("configs/swin/swin.yml")
# model = create(cfg.architecture)



# import numpy as np
# img = np.ones([1,3,512,512]).astype("float32")
# img1 =paddle.to_tensor(img)
# input = {"image":img1}

# out_p = model.backbone(input)

cfg = load_config("configs/swin_det/swinmodel.yml")
model = create(cfg.architecture)
# print(model)
