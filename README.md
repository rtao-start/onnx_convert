# onnx_convert

1 caffe转onnx
   命令：python model_convert.py --model_path ./caffe_model --model_type caffe --output ./output.onnx
   参数说明：model_path：caffe模型所在的文件夹，文件夹里需要有对应的.caffemodel文件和.prototxt文件
                   model_type：模型类型，此处固定为caffe
                   output：输出onnx模型的文件路径

2 tensorflow(h5)转onnx
   命令：python model_convert.py --model_path ./test.h5 --model_type tf-h5  --output ./output.onnx
   参数说明：model_path：h5模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为tf-h5
                   output：输出onnx模型的文件路径  

3 tensorflow(savemodel)转onnx
   命令：python model_convert.py --model_path ./tfsm_model  --model_type tf-sm --output ./output.onnx
   参数说明：model_path：savemodel模型所在的文件夹，文件夹里需要有对应的assets(文件夹)/saved_model.pb(文件)/variables(文件夹)
                   model_type：模型类型，此处固定为tf-sm
                   output：输出onnx模型的文件路径

4 tensorflow(checkpoint)转onnx
   命令：python model_convert.py --model_path ./ckpt_models/test.meta --model_type tf-ckpt  --output ./output.onnx --inputs x:0,y:0  --outputs op_to_store:0
   参数说明：model_path：checkpoint .meta文件所在的路径
                   model_type：模型类型，此处固定为tf-ckpt
                   output：输出onnx模型的文件路径
                   inputs：原始模型的输入变量名称
                   outputs：原始模型的输出变量名称

5 tensorflow(graphpb)转onnx
   命令：python model_convert.py --model_path ./graph.pb  --model_type tf-graph  --output ./output.onnx --inputs x:0,y:0  --outputs op_to_store:0
   参数说明：model_path：graph模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为tf-graph
                   output：输出onnx模型的文件路径
                   inputs：原始模型的输入变量名称
                   outputs：原始模型的输出变量名称

6 pytorch转onnx
   命令：python model_convert.py --model_path ./mnist_model.pkl  --model_type pytorch --output ./output.onnx  --input_shape [1,1,28,28]
   参数说明：model_path：pytorch模型所在的路径(非文件夹)
                   model_type：模型类型，此处固定为pytorch
                   input shape：输入shape
                   output：输出onnx模型的文件路径
  注：
      (1)pytorch模型转换时，需要有模型的定义文件。
      (2)暂时只支持全量模型的转化，不支持只保存参数的模型进行转换(如有需求可以添加)。

7 darknet转onnx
   命令：python model_convert.py --model_path ./dn_models --model_type darknet  --output ./output.onnx
   参数说明：model_path：darknet模型所在的文件夹，文件夹里需要有对应的.cfg文件和.weights文件
                   model_type：模型类型，此处固定为darknet
                   output：输出onnx模型的文件路径
                    