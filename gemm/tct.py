import onnx
import sys, os
import numpy as np
import copy
from onnx import TensorProto

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

def proc_gemm_tct(model, node_id, node, attr):
    print('proc_gemm_tct-----------')
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    length = len(node.input)
    if length == 3: 
        c_name = node.input[2]

    skip = 0

    node_index = node_id

    if transA != 0:
        print('proc_gemm_case_2, Do TransA', node_id)
        skip = skip + 1
        output = node.input[0] + '_transpose_'
        transpose_output = onnx.helper.make_tensor_value_info(output, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=output,
                    inputs=[node.input[0]],
                    outputs=[output])

        node.input[0] = output
        #model.graph.node.append(transpose_node)
        model.graph.node.insert(node_index, transpose_node)
        node_index = node_index + 1

        attributes = node.attribute
        for attr in attributes:
            if attr.name == 'transA':
                attr.i = 0

    if transB != 1 and alpha != 1.0:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[1] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                print('000 init shape:', init.dims[0], init.dims[1])
                print('000 init value:', init.name)

                if isinstance(v, np.ndarray) == True:
                    B = v.reshape(init.dims[0], init.dims[1])
                    B = B.transpose()
                    print('+++B.shape:', B.shape)
                    B = B.flatten()
                    B = v * alpha
                else:    
                    B = np.array(v).reshape(init.dims[0], init.dims[1])
                    B = B.transpose()
                    print('---B.shape:', B.shape)
                    B = B.flatten()
                    B = B * alpha
                    B = B.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    B_ = onnx.helper.make_tensor(name=node.input[1] + '__',
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=B)

                    model.graph.initializer.append(B_) 

                    node.input[1] = node.input[1] + '__'
                else:
                    values.set_tensor_value(init, B, dims_)

                attributes = node.attribute
                found = False
                for attr in attributes:
                    if attr.name == 'transB':
                        found = True
                        attr.i = 1
                
                if found == False:
                    attr = onnx.helper.make_attribute('transB', 1)
                    node.attribute.append(attr)    

                break
                #print('B:', B)
    elif alpha != 1.0:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[1] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                print('111 init shape:', init.dims[0], init.dims[1])
                print('111 init value:', init.name)

                if isinstance(v, np.ndarray) == True:
                    B = v * alpha
                else:    
                    B = np.array(v) * alpha
                    #B = B.reshape(init.dims[0], init.dims[1])
                    print('B.shape:', B.shape)
                    B = B.tolist()

                if is_shared_init(model, init.name, node.name) == True:
                    B_ = onnx.helper.make_tensor(name=node.input[1] + '__',
                                        data_type=init.data_type,
                                        dims=[init.dims[0], init.dims[1]],
                                        vals=B)

                    model.graph.initializer.append(B_) 

                    node.input[1] = node.input[1] + '__'
                else:
                    values.set_tensor_value(init, B)

                break
                #print('B:', B)

        if alpha_proc == False:
            for n in model.graph.node:
                if node.input[1] == n.name:
                    attributes = n.attribute
                    for attr in attributes:
                        if attr.name == 'value':
                            value = attr.t.dims
                            print('value:', value)

                    break
    elif transB != 1:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[1] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                print('222 init shape:', init.dims[0], init.dims[1])
                print('222 init value:', init.name)

                if isinstance(v, np.ndarray) == True:
                    B = v.reshape(init.dims[0], init.dims[1])
                    B = B.transpose()
                    print('=== B.shape:', B.shape)
                    B = B.flatten()
                else:    
                    B = np.array(v).reshape(init.dims[0], init.dims[1])
                    B = B.transpose()
                    print('!!!! B.shape:', B.shape)
                    B = B.flatten()
                    B = B.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    B_ = onnx.helper.make_tensor(name=node.input[1] + '__',
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=B)

                    model.graph.initializer.append(B_) 

                    node.input[1] = node.input[1] + '__'
                else:
                    values.set_tensor_value(init, B, dims_)

                attributes = node.attribute
                found = False
                for attr in attributes:
                    if attr.name == 'transB':
                        found = True
                        attr.i = 1
                
                if found == False:
                    attr = onnx.helper.make_attribute('transB', 1)
                    node.attribute.append(attr)

                break

    output_0 = node.output[0]

    gemm_output = node.name + '_gemm_output_'
    del node.output[:]
    node.output.append(gemm_output)

    mul_name_c = node.name + '_mul_c_'
    beta_name = node.name + '_const_beta_'
    add_name_c = node.name + '_add_c_'
    add_element_c = gemm_output

    if beta != 1.0:
        if length == 3:
            for vi in model.graph.value_info:
                if vi.name == c_name:
                    type_ = vi.elem_type

                    if len(vi.type.tensor_type.shape.dim) > 0:
                        shape_ = [s.dim_value for s in vi.type.tensor_type.shape.dim]
                        print('c_name: ', c_name, ', shape: ', shape_)

                        mul_c_output = mul_name_c + '_output_'

                        const_beta = onnx.helper.make_tensor(name=beta_name,
                                            data_type=type_,
                                            dims=(),
                                            vals=[beta])

                        model.graph.initializer.append(const_beta)                    

                        mul_node_c = onnx.helper.make_node(
                                    'Mul',
                                    name=mul_name_c,
                                    inputs=[beta_name, c_name],
                                    outputs=[onnx.helper.make_tensor_value_info(mul_c_output,
                                                                type_,
                                                                shape_)])

                        model.graph.node.insert(node_index, mul_node_c)
                        node_index = node_index + 1
                        skip = skip + 1 

                        add_node = onnx.helper.make_node(
                            'Add',
                            name=add_name_c,
                            inputs=[mul_c_output, add_element_c],
                            outputs=[output_0]) 

                        model.graph.node.insert(node_index, add_node)
                        node_index = node_index + 1
                        skip = skip + 1  

                    break       
    else:
        add_node = onnx.helper.make_node(
            'Add',
            name=add_name_c,
            inputs=[c_name, add_element_c],
            outputs=[output_0]) 

        model.graph.node.insert(node_index, add_node)
        node_index = node_index + 1
        skip = skip + 1  

    return skip