import torch

from pt_multi_input import nettest


model = nettest()

x1 = torch.randn(1, 3, 200, 300)
x2 = torch.randn(1, 3, 200, 300)

model.train()

y = model(x1, x2)

print('y:', y[0].shape)

torch.save(model,'multi_input.pth')

torch.onnx.export(
    model,
    (x1,x2),
    './test.onnx',
    opset_version=11, 
    do_constant_folding=True,   # 是否执行常量折叠优化
    input_names=["input1", "input2"],    # 模型输入名
    output_names=["output1", "output2"],  # 模型输出名
    dynamic_axes={'input1':{0:'-1'}, 'input2':{0:'-1'}, 'output1':{0:'-1'}, 'output2':{0:'-1'}}
)

