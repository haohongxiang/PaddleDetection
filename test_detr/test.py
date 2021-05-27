import paddle

from paddledetr import DETRLoss, _hungarian_matcher
from matcher import build_matcher
from setpredict import SetCriterion

import numpy as np



# l = 2
# q = 5

# pred_boxes = [[0.4881, 0.5178, 0.4990, 0.5023],
#              [0.5042, 0.5221, 0.5051, 0.5018],
#              [0.4967, 0.5196, 0.5143, 0.5048],
#              [0.5067, 0.5003, 0.4992, 0.4960],
#              [0.4945, 0.5140, 0.5062, 0.4984],]
# pred_logits = [-0.2028,  0.1942, -0.0054, 0.1740, -0.7390]


# gt_boxes = [[0.1458, 0.7139, 0.2305, 0.5372],
#             [0.5403, 0.5516, 0.9123, 0.8744]]

# gt_class = [1, 22]





# boxes = paddle.rand((l, 1, q, 4))
# scores = paddle.rand((l, 1, q, 1))

# # boxes = paddle.to_tensor(pred_boxes).reshape((1, 1, 5, 4))
# # scores = paddle.to_tensor(pred_logits).reshape((1, 1, 5, 1))

# boxes.stop_gradient = False
# scores.stop_gradient = False

# gt_bbox = paddle.to_tensor([[0.1458, 0.7139, 0.2305, 0.5372],
#                             [0.5403, 0.5516, 0.9123, 0.8744]])

# gt_bbox = [gt_bbox]

# gt_class = paddle.to_tensor([[1], [22]])
# gt_class = [gt_class]


# outputs = detrloss(boxes, scores, gt_bbox, gt_class)

# loss = sum([v for k, v in outputs.items()])
# loss.backward()






import torch

q = 10
t = 10
b = 1
num_classes = 80

detrloss = DETRLoss(num_classes=num_classes, aux_loss=False)


for _ in range(1):
    
    outputs = {}
    outputs['pred_boxes'] = torch.rand(b, q, 4) # torch.from_numpy(pred_boxes).view(1, 5, 4)
    outputs['pred_logits'] = torch.rand(b, q, num_classes + 1) # torch.from_numpy(pred_logits).view(1, 5, 1)
    
    outputs['pred_boxes'].requires_grad = True
    outputs['pred_logits'].requires_grad = True
    
    
    targets = []
    for i in range(b):
        target = {}
        target['labels'] = torch.randint(0, 80, (t, ))
        target['boxes'] = torch.rand(t, 4)
        targets.append(target)

    matcher = build_matcher()
    output1 = matcher(outputs, targets)
    output1 = [o.numpy() for o in output1[0]]
    # print(output1)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    weight_dict['loss_giou'] = 2
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)
    print(criterion)
    
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses1 = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
    print(losses1)

    sum([v for k, v in losses1.items()]).backward()
    print('pred_boxes', outputs['pred_boxes'].grad.mean(), outputs['pred_boxes'].grad.sum())
    print('pred_logits', outputs['pred_logits'].grad.mean(), outputs['pred_logits'].grad.sum(), outputs['pred_logits'].grad.max(), outputs['pred_logits'].grad.min(), )

    
    # print(outputs['pred_logits'].grad)
    
    print('-----------')
    print('-----------')
    
    
    boxes = outputs['pred_boxes'].data.numpy()
    scores = outputs['pred_logits'].data.numpy()
    gt_bbox = [t['boxes'].data.numpy() for t in targets]
    gt_class = [t['labels'].data.numpy() for t in targets]

    boxes = paddle.to_tensor(boxes)
    scores = paddle.to_tensor(scores)
    gt_bbox = [paddle.to_tensor(t) for t in gt_bbox]
    gt_class = [paddle.to_tensor(t).reshape((-1, 1)) for t in gt_class]

    boxes.stop_gradient = False
    scores.stop_gradient = False
    
    output2 = _hungarian_matcher(boxes, scores, gt_bbox, gt_class)
    output2 = [o.numpy() for o in output2[0]]
    # print(output2)
    
    np.testing.assert_array_equal(output1[0], output2[0])
    np.testing.assert_array_equal(output1[1], output2[1])
    
    
    losses2 = detrloss(boxes.unsqueeze(0), scores.unsqueeze(0), gt_bbox, gt_class)

    print({k: losses2[k].numpy() for k in losses2})
    
    sum([v for k, v in losses2.items()]).backward()
    print('boxes', boxes.grad.mean(), boxes.grad.sum())
    print('scores', scores.grad.mean(), scores.grad.sum(), scores.grad.max(), scores.grad.min())
    # print('scores', scores.grad)
    
    np.testing.assert_almost_equal(scores.grad, outputs['pred_logits'].grad.data.numpy(), decimal=7)
    
#     matcher = build_matcher(args)
#     weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
#     weight_dict['loss_giou'] = args.giou_loss_coef
#     if args.masks:
#         weight_dict["loss_mask"] = args.mask_loss_coef
#         weight_dict["loss_dice"] = args.dice_loss_coef
#     # TODO this is a hack
#     if args.aux_loss:
#         aux_weight_dict = {}
#         for i in range(args.dec_layers - 1):
#             aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#         weight_dict.update(aux_weight_dict)

#     losses = ['labels', 'boxes', 'cardinality']
#     if args.masks:
#         losses += ["masks"]
#     criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
#                              eos_coef=args.eos_coef, losses=losses)


