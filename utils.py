#import onnx
from onnx import load_model, save_model
import torch
import torch.nn as nn
from float16 import convert_float_to_float16
import numpy as np
import onnxruntime as rt
from torch.nn.modules.upsampling import UpsamplingNearest2d
import time 

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)  
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=4)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.upsampling1(x)
        # x = self.upsampling2(x)
        return x

x = torch.randn((1,3,128,128))

model = TestNet()
model.eval() 

torch_out = model(x)

output_onnx_name = 'test_net.onnx'

torch.onnx.export(model, 
    x,
    output_onnx_name, 
    input_names=["input"], 
    output_names=["output"],
    opset_version=11,
    dynamic_axes={'input':{0:'batch', 2:'h', 3:'w'}, 'output':{0:'batch', 2:'h2', 3:'w2'}} 
)

#onnx_model = load_model(output_onnx_name)
onnx_model = load_model('./caffe.onnx')

trans_model = convert_float_to_float16(onnx_model,keep_io_types=True)

#save_model(trans_model, "test_net_fp16.onnx")
save_model(trans_model, "fp16.onnx")

'''
input_path = './output.onnx'
output_path = './my.onnx'
input_names = ['input_1:0']
output_names = ['functional_1/concatenate/concat:0']

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
'''