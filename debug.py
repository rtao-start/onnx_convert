from onnx import load_model, save_model
import torch
import torch.nn as nn
from float16 import convert_float_to_float16
import numpy as np
import onnxruntime as rt
from torch.nn.modules.upsampling import UpsamplingNearest2d
import time, sys 
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

'''
info_model = onnx.load('./mobilenet_v1-1.onnx')
onnx_model = onnx.shape_inference.infer_shapes(info_model)
 
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, './nn.onnx')


sys.exit()
'''

def convert_ort_type_2_np(ort_data_type):
    #logger.info("convert_ort_type_2_np")
    
    types = {
        1 : np.float32,
        2 : np.uint8,
        3 : np.int8,
        4 : np.uint16,
        5 : np.int16,
        6 : np.int32,
        7 : np.int64,
        8 : "",  #string
        9 : np.bool_,
        10 : np.float16,
        11 : np.float64,
        12 : np.uint32,
        13 : np.uint64,
        14 : np.complex64,
        15 : np.complex_,
        16 : ""
    }

    return types.get(ort_data_type, None)

def get_data_list(dtype, init):
    data_list = []

    if dtype == 2: #uint8
        data_list = init.uint8_data

    if dtype == 3: #int8
        data_list = init.int8_data    

    if dtype == 4: #uint16
        data_list = init.uint16_data

    if dtype == 5: #int16
        data_list = init.int16_data

    if dtype == 6: #int32
        data_list = init.int32_data

    if dtype == 12: #uint32
        data_list = init.uint32_data  

    if dtype == 7: #int64
        data_list = init.int64_data

    if dtype == 13: #uint64
        data_list = init.uint64_data

    return data_list         

def eliminate_reshape(onnxfile):
    model = onnx.load(onnxfile)
    reshape_input = []
    reshape_output = []

    delete_node_id = 0
    delete = False
    export_onnx = onnxfile

    for node_id, node in enumerate(model.graph.node):
        print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            ", op:", node.op_type, ', len(input):', len(node.input))

        if node.op_type == 'Reshape':
            print('got Reshape node:', node.input)
            reshape_input.extend(node.input)
            reshape_output.extend(node.output)
            delete_node_id = node_id
            break

    if len(reshape_input) > 0:
        got_value = False
        reshape_input_shape = []

        for v in model.graph.value_info:
            if v.name == reshape_input[0]:
                print('got value info:', reshape_input) 
                got_value = True
                for d in v.type.tensor_type.shape.dim:
                    reshape_input_shape.append(d.dim_value)
                    
                break

        if got_value == True:
            shape_list = []
            for init in model.graph.initializer:
                if init.name == reshape_input[1]:
                    print('-------')
                    print('init.name', init.name)
                    dtype = init.data_type
                    np_dtype = convert_ort_type_2_np(dtype)
                    if init.raw_data:
                        params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                        for p in params_list:
                            print('p:', p)
                            shape_list.append(p)
                    else:
                        data_list = get_data_list(dtype, init)
                        for p in data_list:
                            print('---p:', p)
                            shape_list.append(p)

                    if reshape_input_shape == shape_list and len(shape_list) > 0:
                        print('need eliminate_reshape')
                        delete = True

                    break            

    if delete == True:     
        print('delete: ', delete_node_id)
        delete_node = model.graph.node[delete_node_id]

        last_node = True

        for node_id, node in enumerate(model.graph.node):
            if node.input[0] == reshape_output[0]:
                print('got reshape next node:', node.name)
                next_node = model.graph.node[node_id]
                next_node.input[0] = delete_node.input[0]
                last_node = False
                break

        model.graph.node.remove(delete_node)

        if last_node == True:
            #model.graph.output.extend()
            for node_id, node in enumerate(model.graph.node):
                #print('+++++====', node.input[0], reshape_output[0])
                if node.output[0] == reshape_input[0]:
                    print('got reshape prev node:', node.name)
                    prev_node = model.graph.node[node_id]
                    prev_node.output[0] = reshape_output[0]
                    break

        export_onnx = './55.onnx'#onnxfile

        ###################
        #onnx.checker.check_model(model)
        onnx.save(model, export_onnx)

    return delete, export_onnx

#eliminate_reshape('./111.onnx')

#sys.exit()

'''
#model = onnx.load('./deart_model_sim_v11.onnx')
model = onnx.load('./6.onnx')

#new_model = onnx.shape_inference.infer_shapes(model)
#onnx.save(model, './vv.onnx')
init_list = []

for init in model.graph.initializer:
    print("init name:", init.name)
    init_list.append(init.name)   

print('==================================++++++++++++++++++')

real_input_init = []
#for node_id, node in enumerate(model.graph.node):
node = model.graph.node[0]    
for n in node.input:
    if n in init_list:
        real_input_init.append(n)

for n in real_input_init:
    print("real_input_init:", n)

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

#ValueInfoProto 
vip = []

for input in model.graph.input:
    if input.name in real_input_init:
        vip.append(input)
    elif input.name not in init_list:
        vip.append(input)  

del model.graph.input[:]

model.graph.input.extend(vip)

for input in model.graph.input:
    print("got  input name:", input.name)

onnx.checker.check_model(model)
onnx.save(model, './3.onnx')
'''

'''
from onnxsim import simplify
onnx_model = onnx.load('./mobilenet_v1-1.onnx')  # load onnx model
model_simp, check = simplify(onnx_model, skip_shape_inference=False, input_shapes={'input:0': [1,3,224,224]})
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, './zz.onnx')
print('finished exporting onnx')
'''

import argparse

def fuse_pad_to_pool(onnxfile, export_onnx):
    model = onnx.load(onnxfile)

    dict_pad = {}
    dict_pool = {}
    dict_mul = {}

    got_pad_pool = False

    for node_id, node in enumerate(model.graph.node):
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #         ", op:", node.op_type, ', len(input):', len(node.input))

        if node.op_type == 'Pad':
            dict_pad['input'] = node.input
            dict_pad['output'] = node.output
            dict_pad['id'] = node_id

        if node.op_type == 'MaxPool':
            if len(dict_pad) > 0 and node.input == dict_pad['output']:
                dict_pool['input'] = node.input
                dict_pool['output'] = node.output
                dict_pool['id'] = node_id
                print('got pad+pool pair, pad:', dict_pad['input'], dict_pad['output'])
                print('got pad+pool pair, pool:', dict_pool['input'], dict_pool['output'])
                pads = []

                got_pad_pool = True

                for init in model.graph.initializer:
                    if init.name == dict_pad['input'][1]:
                        print('got init(pads):', init.name)
                        dtype = init.data_type
                        np_dtype = convert_ort_type_2_np(dtype)
                        if init.raw_data:
                            params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                            for p in params_list:
                                print('p:', p)
                                pads.append(p)
                        else:
                            data_list = get_data_list(dtype, init)
                            for p in data_list:
                                print('---p:', p)
                                pads.append(p)
                    elif init.name == dict_pad['input'][2]:
                        print('got init(constane_value):', init.name)  

                pads_real = [pads[2], pads[3], pads[6], pads[7]]

                for attr in node.attribute:
                    #print('attr:', attr)
                    if attr.name == 'pads':
                        del attr.ints[:]
                        attr.ints.extend(pads_real)
                        print('pads:---', attr.ints)
                        break
     
                node.input[0] = dict_pad['input'][0]

                old_node = model.graph.node[dict_pad['id']] 
                model.graph.node.remove(old_node)

                dict_pad = {}
                dict_pool = {}
            else:
                #print('clear pad dict')
                dict_pad = {}    

    if got_pad_pool == True:
        print('got pad+pool node------------')

        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1
        
        onnx.save(model, export_onnx)
        

export_onnx = './tmp.onnx'
fuse_pad_to_pool('./v3-tiny.onnx', export_onnx)   






         