import os
import time
import paddle

# 从模型代码中导入模型
from paddle.vision.models import mobilenet_v2

# 实例化模型
model = mobilenet_v2()

# 将模型设置为推理状态
model.eval()

# 定义输入数据
input_spec = paddle.static.InputSpec(shape=[None, 3, 320, 320], dtype='float32', name='image')

# ONNX模型导出
# enable_onnx_checker设置为True，表示使用官方ONNX工具包来check模型的正确性，需要安装ONNX（pip install onnx）

paddle.onnx.export(model, 'mobilenet_v2', input_spec=[input_spec], opset_version=12, enable_onnx_checker=True)

#paddle2onnx --model_dir inference  --model_filename model.pdmodel --params_filename model.pdiparams  --save_file mobilenet_v2.onnx  --opset_version 12
#paddle.jit.save(model, 'inference/model', input_spec=[input_spec])

########################################################
########################################################
# 动态图导出的ONNX模型测试
import time
import numpy as np
from onnxruntime import InferenceSession
# 加载ONNX模型
sess = InferenceSession('mobilenet_v2.onnx')
# 准备输入
x = np.random.random((1, 3, 320, 320)).astype('float32')
# 模型预测
start = time.time()
ort_outs = sess.run(output_names=None, input_feed={'image': x})
end = time.time()
print("Exported model has been predicted by ONNXRuntime!")
print('ONNXRuntime predict time: %.04f s' % (end - start))
# 对比ONNX Runtime 和 飞桨的结果
paddle_outs = model(paddle.to_tensor(x))
diff = ort_outs[0] - paddle_outs.numpy()
max_abs_diff = np.fabs(diff).max()
if max_abs_diff < 1e-05:
    print("The difference of results between ONNXRuntime and Paddle looks good!")
else:
    relative_diff = max_abs_diff / np.fabs(paddle_outs.numpy()).max()
    print('relative_diff: ', relative_diff)
    if relative_diff < 1e-05:
        print("The difference of results between ONNXRuntime and Paddle looks good!")
    else:
        print("The difference of results between ONNXRuntime and Paddle looks bad!")

print('max_abs_diff: ', max_abs_diff)


