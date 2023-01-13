import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto

def get_shape(model, tensor_name):
    shape = []
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            if len(vi.type.tensor_type.shape.dim) > 0:
                shape = [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
                break

    if len(shape) == 0:
        for input_ in model.graph.input:
            if input_.name == tensor_name:
                if len(input_.type.tensor_type.shape.dim) > 0:
                    shape = [dim.dim_value for dim in input_.type.tensor_type.shape.dim]
                    break

    if tensor_name == 'memory':
        print('memory shape----', shape)

    return shape            

def is_constant(model, name):
    for init in model.graph.initializer:
        if init.name == name:
            shape = [dim for dim in init.dims]
            if len(shape) == 2:
                print('Got it-------', shape)
                return True

    return False

def matmul_reshape(model):
    search = True
    index = 0

    while search == True:
        search = False

        for node_id, node in enumerate(model.graph.node):
            if node.op_type == 'MatMul':
                #print('xxxxx got MatMul, name:', node.name)
                shape_in = get_shape(model, node.input[0])
                if len(shape_in) > 2:
                    #print('yyyyy got MatMul, name:', node.name)
                    if is_constant(model, node.input[1]):
                        search = True
                        shape_out = get_shape(model, node.output[0])
                        print('got MatMul, name:', node.name, ', shape:', shape_in, shape_out)

                        shape_0 = 1
                        for s in shape_in:
                            shape_0 = shape_0*s

                        s = shape_0/shape_in[-1]

                        shape_value = np.array([s, shape_in[-1]], dtype=np.int64)

                        shape_name = node.name + '_shape_' + str(index)
                        index += 1

                        reshape_output_name = node.output[0] + '__reshape__' 

                        print('shape_value:', shape_value, shape_name)

                        shape_element = helper.make_tensor(shape_name, TensorProto.INT64, [2], shape_value)

                        reshape_node = helper.make_node(
                            name = node.name + '_reshape1_',
                            op_type='Reshape', 
                            inputs=[node.input[0], shape_name],
                            outputs=[reshape_output_name]  
                            ) 

                        model.graph.initializer.append(shape_element) 

                        node.input[0] = reshape_output_name

                        model.graph.node.insert(node_id, reshape_node)

                        ####################
                        shape_name2 = node.name + '_shape_' + str(index)
                        index = index + 1

                        reshape_output_name2 = node.output[0] + '__reshape2__' 
                        shape_element2 = helper.make_tensor(shape_name2, TensorProto.INT64, [len(shape_out)], shape_out)

                        print('zzzzzzzzzzzzzzzzz node.output[0]:', node.output[0])

                        reshape_node2 = helper.make_node(
                            name = node.name + '_reshape2_',
                            op_type='Reshape', 
                            inputs=[node.output[0], shape_name2],
                            outputs=[reshape_output_name2]  
                            ) 

                        model.graph.initializer.append(shape_element2) 

                        for node_id_, node_ in enumerate(model.graph.node):
                            found = False
                            if len(node_.input) and node_.input[0] == node.output[0]:
                                print('node name:', node_.name)
                                found = True
                                print('replace', node_.input[0], reshape_output_name2)
                                node_.input[0] = reshape_output_name2
                                break

                            if found == False and len(node_.input) > 1 and node_.input[1] == node.output[0]:     
                                print('---node name:', node_.name)
                                found = True
                                node_.input[1] = reshape_output_name2
                                break

                        model.graph.node.insert(node_id+1, reshape_node2)

                        break

model = onnx.load('/home/zqiu/models/decoder_model_bs10.onnx')
model = onnx.shape_inference.infer_shapes(model)

matmul_reshape(model)

del model.graph.value_info[:]
new_model = onnx.shape_inference.infer_shapes(model)
new_model = onnx.shape_inference.infer_shapes(new_model)
onnx.save(new_model, './tt2.onnx')

