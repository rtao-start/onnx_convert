import onnx
import sys
import argparse
import values
import numpy as np

def got_input_shape(model, tensor):
    for input_ in model.graph.input:
        if input_.name == tensor:
            dim_proto_input = input_.type.tensor_type.shape.dim[0]
            print('++++++got input shape: ', dim_proto_input.dim_value)
            return dim_proto_input.dim_value, True

    for vi in model.graph.value_info:
        if vi.name == tensor:
            dim_proto_input = vi.type.tensor_type.shape.dim[0]
            print('------got input shape: ', dim_proto_input.dim_value)
            return dim_proto_input.dim_value, True

    return -777, False      

def is_shared_init(model, init, node_name):
    for node in model.graph.node:
        if node.name != node_name:
            if init in node.input:
                return True

    return False

def matmul_2_gemm(model):
    index = 0

    for node in model.graph.node:
        if node.op_type == 'MatMul':
            in_shape, ret = got_input_shape(model, node.input[0])
            if ret == True and in_shape < 32:
                for init in model.graph.initializer:
                    if init.name == node.input[1]:
                        node.op_type = 'Gemm'

                        C_val = np.array([0])
                        C_name = node.name + '_gemm_c_' + str(index)
                        index = index + 1

                        CC = onnx.helper.make_tensor(name=C_name,
                                                data_type=init.data_type,
                                                dims=[1],
                                                vals=C_val.tolist())

                        model.graph.initializer.append(CC)
                        node.input.append(C_name)

                        attr = onnx.helper.make_attribute('transB', 1)
                        node.attribute.append(attr)

                        attr = onnx.helper.make_attribute('alpha', 1.0)
                        node.attribute.append(attr) 

                        attr = onnx.helper.make_attribute('beta', 1.0)
                        node.attribute.append(attr)   

                        v = values.get_init_value(model, init.name)

                        if isinstance(v, np.ndarray) == True:
                            B = v.reshape(init.dims[0], init.dims[1])
                            B = B.transpose()
                            B = B.flatten()
                        else:    
                            B = np.array(v).reshape(init.dims[0], init.dims[1])
                            B = B.transpose()
                            B = B.flatten()
                            #B = B.tolist()

                        dims_= [init.dims[1], init.dims[0]]

                        if is_shared_init(model, init.name, node.name) == True:
                            B_name = node.input[1] + '__transpose__'
                            B_ = onnx.helper.make_tensor(name=B_name,
                                                data_type=init.data_type,
                                                dims=dims_,
                                                vals=B)

                            model.graph.initializer.append(B_) 

                            node.input[1] = B_name
                        else:
                            values.set_tensor_value(init, B, dims_)            

    return model


