from onnx import load_model, save_model
import torch
import torch.nn as nn
from float16 import convert_float_to_float16
import numpy as np
import onnxruntime as rt
from torch.nn.modules.upsampling import UpsamplingNearest2d
import time 
import onnx

'''
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

################ extract_model
'''
input_path = './output.onnx'
output_path = './my.onnx'
input_names = ['input_1:0']
output_names = ['functional_1/concatenate/concat:0']

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
'''

######## insert preprocess node
'''
import onnx

onnx_model = onnx.load('./v4_sub.onnx')
graph = onnx_model.graph

input_name = graph.input[0].name

sub_const_node = onnx.helper.make_tensor(name='const_sub',
                      data_type=onnx.TensorProto.FLOAT,
                      dims=[3],
                      vals=[-127.5, -127.5, -127.5])

graph.initializer.append(sub_const_node)

sub_node = onnx.helper.make_node(
                'Add',
                name='sub',
                inputs=[input_name, 'const_sub'],
                outputs=['pre_sub'])

graph.node.insert(0, sub_node)

mul_const_node = onnx.helper.make_tensor(name='const_mul',
                      data_type=onnx.TensorProto.FLOAT,
                      dims=[3],
                      vals=[1.0/127.5, 1.0/127.5, 1.0/127.5])

graph.initializer.append(mul_const_node)

sub_node = onnx.helper.make_node(
               'Mul',
               name='mul',
               inputs=['pre_sub', 'const_mul'],
               outputs=['pre_mul'])

graph.node.insert(1, sub_node)

graph.node[2].input[0]='pre_mul'

graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
info_model = onnx.helper.make_model(graph)
onnx_model = onnx.shape_inference.infer_shapes(info_model)
 
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, './nn.onnx')
'''

'''
######## insert preprocess node 2
import onnx

onnx_model = onnx.load('./mnist_model.onnx')
graph = onnx_model.graph

input_name = graph.input[0].name

mean_const_node = onnx.helper.make_tensor(name='const_mean',
                      data_type=onnx.TensorProto.FLOAT,
                      dims=[3],
                      vals=[127.5, 127.5, 127.5])

graph.initializer.append(mean_const_node)

std_const_node = onnx.helper.make_tensor(name='const_std',
                      data_type=onnx.TensorProto.FLOAT,
                      dims=[3],
                      vals=[1.0/127.5, 1.0/127.5, 1.0/127.5])

graph.initializer.append(std_const_node)

resize_const_node = onnx.helper.make_tensor(name='const_resize',
                      data_type=onnx.TensorProto.INT32,
                      dims=[2],
                      vals=[224, 224])

graph.initializer.append(resize_const_node)

crop_const_node = onnx.helper.make_tensor(name='const_crop',
                      data_type=onnx.TensorProto.INT32,
                      dims=[4],
                      vals=[100,200,300,400])

graph.initializer.append(crop_const_node)

pre_process_node = onnx.helper.make_node(
                'PreProc',
                name='preprocess',
                inputs=[input_name, 'const_mean', 'const_std', 'const_resize','const_crop'],
                outputs=['pre_process'])

graph.node.insert(0, pre_process_node)

graph.node[1].input[0]='pre_process'

graph.input[0].type.tensor_type.elem_type = 2
graph.input[0].type.tensor_type.shape.dim[2].dim_value = -1
graph.input[0].type.tensor_type.shape.dim[3].dim_value = -1

graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
info_model = onnx.helper.make_model(graph)
onnx_model = onnx.shape_inference.infer_shapes(info_model)
 
#onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, './mm.onnx')

m=onnx.load('./mm.onnx')
for node_id, node in enumerate(m.graph.node):
    print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            ", op:", node.op_type, ', len(input):', len(node.input))
'''
def add_value_info_for_constants(model : onnx.ModelProto):
    """
    Currently onnx.shape_inference doesn't use the shape of initializers, so add
    that info explicitly as ValueInfoProtos.
    Mutates the model.
    Args:
        model: The ModelProto to update.
    """
    # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
    if model.ir_version < 4:
        return

    def add_const_value_infos_to_graph(graph : onnx.GraphProto):
        inputs = {i.name for i in graph.input}
        in_ = {i.name: i for i in graph.input}
        for init in graph.initializer:
            # Check it really is a constant, not an input
            if init.name in inputs:
                continue

            # The details we want to add
            elem_type = init.data_type
            shape = init.dims

            # Get existing or create new value info for this constant
            vi = in_.get(init.name)
            if vi is None:
                vi = graph.input.add()
                vi.name = init.name

            # Even though it would be weird, we will not overwrite info even if it doesn't match
            tt = vi.type.tensor_type
            if tt.elem_type == onnx.TensorProto.UNDEFINED:
                tt.elem_type = elem_type
            if not tt.HasField("shape"):
                # Ensure we set an empty list if the const is scalar (zero dims)
                tt.shape.dim.extend([])
                for dim in shape:
                    tt.shape.dim.add().dim_value = dim

        # Handle subgraphs
        for node in graph.node:
            for attr in node.attribute:
                # Ref attrs refer to other attrs, so we don't need to do anything
                if attr.ref_attr_name != "":
                    continue

                if attr.type == onnx.AttributeProto.GRAPH:
                    add_const_value_infos_to_graph(attr.g)
                if attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        add_const_value_infos_to_graph(g)

    return add_const_value_infos_to_graph(model.graph)

m = onnx.load('./ResCNN_tf_sim.onnx')
'''
#new_model = onnx.shape_inference.infer_shapes(model)
#onnx.save(model, './vv.onnx')
for node_id, node in enumerate(m.graph.node):
    print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            ", op:", node.op_type, ', len(input):', len(node.input))

print('======================================model.ir_version:', m.ir_version)

for init in m.graph.initializer:
    print('got init: ', init.name)  

print('-------------------------------------------------------')

add_value_info_for_constants(m)

for value_info in m.graph.value_info:
    print('got value_info: ', value_info.name)

onnx.save(m, './vv.onnx')                   
'''

'''
id = 0
for node_id, node in enumerate(m.graph.node):
    print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
             ", op:", node.op_type, ', len(input):', len(node.input))
    if node.op_type == 'Transpose':
        print('Transpose, input:', node.input, node_id)
        id = node_id

del m.graph.output[:]

m.graph.output.extend([onnx.ValueInfoProto(name='output/BiasAdd_raw_output___4:0')])
#onnx.checker.check_model(m)

old_node = m.graph.node[id]
m.graph.node.remove(old_node)

onnx_model = onnx.shape_inference.infer_shapes(m)

onnx.save(onnx_model, './vv.onnx') 
'''

info_model = onnx.load('./fp16_resnet50.onnx')
onnx_model = onnx.shape_inference.infer_shapes(info_model)
 
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, './nn.onnx')