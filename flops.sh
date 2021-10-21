

# vim ../anaconda3/lib/python3.8/site-packages/paddle/hapi/dynamic_flops.py 
# 
#  24 def flops(net, input_size=None, data=None, custom_ops=None, print_detail=False):
# 103         if data is None:
# 104             inputs = paddle.randn(input_size)
# 105         else:
# 106             inputs = data



python tools/train.py -c ./configs/picodet/picodet_s_320_coco.yml
# python tools/train.py -c ./configs/picodet/picodet_s_416_coco.yml


# python tools/train.py -c ./configs/picodet/picodet_m_320_coco.yml
# python tools/train.py -c ./configs/picodet/picodet_m_416_coco.yml



# python tools/train.py -c ./configs/picodet/picodet_l_320_coco.yml
# python tools/train.py -c ./configs/picodet/picodet_l_416_coco.yml
# python tools/train.py -c ./configs/picodet/picodet_l_640_coco.yml
