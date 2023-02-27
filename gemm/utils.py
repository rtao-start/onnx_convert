import onnx
import sys, os
import numpy as np

sys.path.append(os.path.abspath('..'))
import values

def is_shared_init(model, init, node_name):
    for node in model.graph.node:
        if node.name != node_name:
            if init in node.input:
                return True

    return False            

def is_shared_constant(model, constant):
    count = 0
    for node in model.graph.node:
        if constant in node.input:
            count = count + 1

    if count > 1:
        return True            

    return False

def got_input_shape(model, tensor):
    for vi in model.graph.input:
        if vi.name == tensor:
            dim_proto_input = vi.type.tensor_type.shape.dim[0]
            print('+++++ got input shape: ', dim_proto_input.dim_value)
            return dim_proto_input.dim_value, True

    for vi in model.graph.value_info:
        if vi.name == tensor:
            if len(vi.type.tensor_type.shape.dim) > 0:
                dim_proto_input = vi.type.tensor_type.shape.dim[0]
                print('got input shape: ', dim_proto_input.dim_value)
                return dim_proto_input.dim_value, True

    for init in model.graph.initializer:
        if tensor == init.name:
            print('---got input shape: ', init.dims[0])
            return init.dims[0], True 

    '''
    for node in model.graph.node:
        if node.op_type == 'Constant':
            if node.output[0] == tensor:
                attributes = node.attribute
                for attr in attributes:
                    if attr.name == 'value':
                        v = values.get_tensor_value(attr.t)
                        return v[0], True
    '''

    return -777, False         