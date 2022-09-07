import paddle
import numpy as np

#静态模型
'''
# 模型加载
model = paddle.jit.load('./paddle_model/model')

# 将模型设置为评估状态
model.eval()

# 模型结构预览
paddle.summary(model, input_size=(1, 3, 320, 320))

# 获取模型得到输入和输出
input_spec = model._input_spec()
output_spec = model._output_spec()
print(input_spec)
print(output_spec)

# 准备数据
#x = paddle.randn(shape=(1, 3, 320, 320))
x = np.random.random((1, 3, 320, 320)).astype('float32')

np.save('paddle_input', x)

# 前向计算
#d0 = model(x)

d0 = model(paddle.to_tensor(x))

# 打印输出的形状
print(d0.shape)

np.save('paddle_output', d0.numpy())
'''

from mnist import LeNet

# 模型加载
model = paddle.vision.models.LeNet() #LeNet()
model.set_dict(paddle.load('./paddle_checkpoint/final.pdparams'))
model.eval()

# 将模型设置为评估状态
model.eval()

# 模型结构预览
paddle.summary(model, input_size=(1, 1, 28, 28))

# 准备数据
#x = paddle.randn(shape=(1, 3, 320, 320))
x = np.random.random((1, 1, 28, 28)).astype('float32')

np.save('paddle_input', x)

# 前向计算
#d0 = model(x)

d0 = model(paddle.to_tensor(x))

# 打印输出的形状
print(d0.shape)

np.save('paddle_output', d0.numpy())