import onnx
import sys, os
import numpy as np
import copy
from .utils import got_input_shape, is_shared_init, is_shared_constant
from onnx import TensorProto

sys.path.append(os.path.abspath('..'))
import values

def proc_gemm_tct(model, node_id, node, attr):
    print('proc_gemm_tct-----------', node.name)

    in_shape, _ = got_input_shape(model, node.input[0])

    print('proc_gemm_tct, got input shape:', in_shape)

    if in_shape > 32:
        print('in_shape > 32, goto proc_gemm_tct_matmul')
        return proc_gemm_tct_matmul(model, node_id, node, attr)

    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    length = len(node.input)
    if length == 3: 
        c_name = node.input[2]

    skip = 0

    if transA != 0:
        print('transA != 0, goto proc_gemm_tct_matmul')
        return proc_gemm_tct_matmul(model, node_id, node, attr)

    node_index = node_id

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
                    B = B * alpha
                else:    
                    B = np.array(v).reshape(init.dims[0], init.dims[1])
                    B = B.transpose()
                    print('---B.shape:', B.shape)
                    B = B.flatten()
                    B = B * alpha
                    #B = B.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    B_ = onnx.helper.make_tensor(name=node.input[1] + '__',
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=B.tolist())

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

                for attr in attributes:
                    if attr.name == 'alpha':
                        attr.f = 1          

                break
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

                attributes = node.attribute
                for attr in attributes:
                    if attr.name == 'alpha':
                        attr.f = 1  
                break

        if alpha_proc == False:
            for n in model.graph.node:
                if node.input[1] == n.name:
                    attributes = n.attribute
                    for attr in attributes:
                        if attr.name == 'value':
                            value = attr.t.dims
                            print('value:', value)
                            #TBD
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
                    #B = B.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    B_ = onnx.helper.make_tensor(name=node.input[1] + '__',
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=B.tolist())

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

    mul_name_c = node.name + '_mul_c_'
    beta_name = node.name + '_const_beta_'

    if beta != 1.0:
        if length == 3:
            for vi in model.graph.value_info:
                if vi.name == c_name:
                    type_ = vi.type.tensor_type.elem_type

                    if len(vi.type.tensor_type.shape.dim) > 0:
                        shape_ = [s.dim_value for s in vi.type.tensor_type.shape.dim]
                        print('c_name: ', c_name, ', shape: ', shape_)

                        mul_c_output = mul_name_c + '_output_'

                        const_beta = onnx.helper.make_tensor(name=beta_name,
                                            data_type=type_,
                                            dims=(),
                                            vals=[beta])

                        model.graph.initializer.append(const_beta) 

                        mul_output = onnx.helper.make_tensor_value_info(mul_c_output,
                                                                type_,
                                                                shape_)                   

                        mul_node_c = onnx.helper.make_node(
                                    'Mul',
                                    name=mul_name_c,
                                    inputs=[beta_name, c_name],
                                    outputs=[mul_c_output])

                        model.graph.node.insert(node_index, mul_node_c)
                        node_index = node_index + 1
                        skip = skip + 1

                        node.input[2] = mul_c_output  

                    attributes = node.attribute
                    for attr in attributes:
                        if attr.name == 'beta':
                            attr.f = 1  

                    break       

    return skip

def proc_gemm_tct_matmul(model, node_id, node, attr): 
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    node_index = node_id

    length = len(node.input)
    c_name = ''
    if length == 3: 
        c_name = node.input[2]

    outputA = ''
    outputB = ''

    skip = 0

    if transA != 0:
        print('proc_gemm_tct_matmul, Do TransA', node_id)
        skip = skip + 1
        outputA = node.input[0] + '_transpose_'
        transpose_output = onnx.helper.make_tensor_value_info(outputA, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=outputA,
                    inputs=[node.input[0]],
                    outputs=[outputA])

        model.graph.node.insert(node_index, transpose_node)
        node_index = node_index + 1

    if transB == 1 and alpha != 1.0:
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
                    B = B * alpha
                else:    
                    B = np.array(v).reshape(init.dims[0], init.dims[1])
                    B = B.transpose()
                    print('---B.shape:', B.shape)
                    B = B.flatten()
                    B = B * alpha
                    #B = B.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    B_ = onnx.helper.make_tensor(name=node.input[1] + '__',
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=B.tolist())

                    model.graph.initializer.append(B_) 

                    node.input[1] = node.input[1] + '__'
                else:
                    values.set_tensor_value(init, B, dims_)

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

        if alpha_proc == False:
            for n in model.graph.node:
                if node.input[1] == n.name:
                    attributes = n.attribute
                    for attr in attributes:
                        if attr.name == 'value':
                            value = attr.t.dims
                            print('value:', value)

                    break
    elif transB == 1:
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

                break
    ################
    del node.attribute[:]

    node.op_type = 'MatMul'
    input_0 = node.input[0]
    input_1 = node.input[1]

    output_0 = node.output[0]

    if length == 3:
        del node.input[2:]

    if outputA != '':
        node.input[0] = outputA

    matmul_output_name = node.output[0] + '_matmul_'
    #node.output[0] = matmul_output_name

    #############
    if length == 3:
        beta_name = matmul_output_name + 'const_beta'
        add_name = matmul_output_name + '_add_'
        add_element = matmul_output_name
        add_name_c = matmul_output_name + '_add_c_'
        mul_name_c = matmul_output_name + '_mul_c_'

        if beta != 1.0 and beta > 0.0:
            node.output[0] = matmul_output_name
            for vi in model.graph.value_info:
                if vi.name == c_name:
                    type_ = vi.type.tensor_type.elem_type

                    if len(vi.type.tensor_type.shape.dim) > 0:
                        shape_ = [s.dim_value for s in vi.type.tensor_type.shape.dim]
                        print('c_name: ', c_name, ', shape: ', shape_)

                        mul_c_output = mul_name_c + '_output_'

                        const_beta = onnx.helper.make_tensor(name=beta_name,
                                            data_type=type_,
                                            dims=(),
                                            vals=[beta])

                        model.graph.initializer.append(const_beta) 

                        mul_c = onnx.helper.make_tensor_value_info(mul_c_output, type_, shape_)                   

                        mul_node_c = onnx.helper.make_node(
                                    'Mul',
                                    name=mul_name_c,
                                    inputs=[beta_name, c_name],
                                    outputs=[mul_c_output])

                        model.graph.node.insert(node_index, mul_node_c)
                        node_index = node_index + 1
                        skip = skip + 1 

                        add_node_c = onnx.helper.make_node(
                            'Add',
                            name=add_name_c,
                            inputs=[mul_c_output, add_element],
                            outputs=[output_0]) 

                        model.graph.node.insert(node_index, add_node_c)
                        node_index = node_index + 1
                        skip = skip + 1  

                    break       
        elif beta == 1.0:
            node.output[0] = matmul_output_name
            add_node = onnx.helper.make_node(
                'Add',
                name=add_name,
                inputs=[c_name, add_element],
                outputs=[output_0]) 

            model.graph.node.insert(node_index, add_node)
            node_index = node_index + 1
            skip = skip + 1

    return skip      