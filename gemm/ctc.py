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

def handle_constant_node(model, node, transpose, alpha):
    for n in model.graph.node:
        if node.input[0] == n.output[0]:
            attributes = n.attribute
            for attr in attributes:
                if attr.name == 'value':
                    if is_shared_constant(model, node.input[0]):
                        new_node = copy.deepcopy(n)
                        new_name = n.name + '__'
                        new_node.name = new_name
                        new_node.output[0] = new_node.output[0] + '__'
                        attrs = new_node.attribute

                        for attr_ in attrs:
                            if attr_.name == 'value':
                                v_ = values.get_tensor_value(attr_.t)
                                shape_ = attr_.t.dims
                                dims_= [shape_[0], shape_[1]]

                                if isinstance(v_, np.ndarray) != True:
                                    A = np.array(v_)

                                A = v_.reshape(*shape_)

                                if alpha != 1.0:
                                    A = v_ * alpha

                                if transpose == True:
                                    A = A.transpose()
                                    dims_= [shape_[1], shape_[0]]

                                A = A.tolist()

                                values.set_tensor_value(attr_.t, A, dims_)    

                                break

                        node.input[0] = new_node.output[0]
                        model.graph.node.append(new_node)            
                    else: 
                        v = values.get_tensor_value(attr.t)
                        if isinstance(v, np.ndarray) != True:
                            A = np.array(v)

                        shape = attr.t.dims
                        dims = [shape[0], shape[1]]

                        A = v.reshape(*shape)

                        if alpha != 1.0:
                            A = v * alpha

                        if transpose == True:    
                            A = A.transpose()
                            dims = [shape_[1], shape_[0]]

                        A = A.tolist()

                        values.set_tensor_value(attr.t, A, dims)   
                    break         
            break     

def proc_gemm_ctc(model, node_id, node, attr):
    print('proc_gemm_ctc-----------')
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    length = len(node.input)
    if length == 3: 
        c_name = node.input[2]

    skip = 0

    node_index = node_id

    input_0 = node.input[0]
    input_1 = node.input[1]

    if transB != 1:
        print('proc_gemm_case_4, Do TransB', node_id)

        output = node.input[1] + '_transpose_'
        #transpose_output = onnx.helper.make_tensor_value_info(output, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=output,
                    inputs=[node.input[1]],
                    outputs=[onnx.helper.make_tensor_value_info(output, TensorProto.UNDEFINED, ['a', 'b'])])

        #node.input[1] = output
        node.input[0] = output

        model.graph.node.insert(node_index, transpose_node)
        node_index = node_index + 1
        skip = skip + 1

        attributes = node.attribute
        for attr in attributes:
            if attr.name == 'transA':
                attr.i = 0              
    else:
        node.input[0] = input_1

        attributes = node.attribute
        for attr in attributes:
            if attr.name == 'transA':
                attr.i = 0  

    if transA != 0 and alpha != 1.0:
        A_name = node.input[0] + '__'
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[0] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                print('000 init shape:', init.dims[0], init.dims[1])
                print('000 init value:', init.name)

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    print('+++A.shape:', A.shape)
                    A = A.flatten()
                    A = A * alpha
                else:    
                    A = np.array(v).reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    print('---B.shape:', B.shape)
                    A = A.flatten()
                    A = A * alpha
                    A = A.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    A_ = onnx.helper.make_tensor(name=A_name,
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=A)

                    model.graph.initializer.append(A_) 

                    node.input[1] = A_name
                else:
                    values.set_tensor_value(init, A, dims_)
                    node.input[1] = input_0

                break

        if alpha_proc == False:
            handle_constant_node(model, node, True, alpha)

        attributes = node.attribute
        found = False
        for attr in attributes:
            if attr.name == 'transB':
                found = True
                attr.i = 1
        
        if found == False:
            attr = onnx.helper.make_attribute('transB', 1)
            node.attribute.append(attr)        
    elif alpha != 1.0:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[0] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                print('111 init shape:', init.dims[0], init.dims[1])
                print('111 init value:', init.name)

                if isinstance(v, np.ndarray) == True:
                    A = v * alpha
                else:    
                    A = np.array(v) * alpha
                    #B = B.reshape(init.dims[0], init.dims[1])
                    print('A.shape:', A.shape)
                    A = A.tolist()

                A_name = node.input[0] + '__'

                if is_shared_init(model, init.name, node.name) == True:
                    A_ = onnx.helper.make_tensor(name=A_name,
                                        data_type=init.data_type,
                                        dims=[init.dims[0], init.dims[1]],
                                        vals=A)

                    model.graph.initializer.append(A_) 

                    node.input[1] = A_name
                else:
                    values.set_tensor_value(init, A)
                    node.input[1] = input_0

                break

        if alpha_proc == False:
            handle_constant_node(model, node, False, alpha)

        attributes = node.attribute
        found = False
        for attr in attributes:
            if attr.name == 'transB':
                found = True
                attr.i = 1
        
        if found == False:
            attr = onnx.helper.make_attribute('transB', 1)
            node.attribute.append(attr)        
    elif transA != 0:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[0] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                print('222 init shape:', init.dims[0], init.dims[1])
                print('222 init value:', init.name)

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    print('=== A.shape:', A.shape)
                    A = A.flatten()
                else:    
                    A = np.array(v).reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    print('!!!! B.shape:', A.shape)
                    A = A.flatten()
                    A = A.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    A_ = onnx.helper.make_tensor(name=A_name,
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=A)

                    model.graph.initializer.append(A_) 

                    node.input[1] = A_name
                else:
                    values.set_tensor_value(init, A, dims_)
                    node.input[1] = input_0
                break

        if alpha_proc == False:
            handle_constant_node(model, node, True, 1.0)

        attributes = node.attribute
        found = False
        for attr in attributes:
            if attr.name == 'transB':
                found = True
                attr.i = 1
        
        if found == False:
            attr = onnx.helper.make_attribute('transB', 1)
            node.attribute.append(attr)    

    output_0 = node.output[0]

    gemm_output = node.name + '_gemm_output_'
    del node.output[:]
    node.output.append(gemm_output)

    #######
    transpose_name = gemm_output + '_transpose_'
    transpose_output = transpose_name + '_output_'

    transpose_node = onnx.helper.make_node(
                'Transpose',
                name=transpose_name,
                inputs=[gemm_output],
                outputs=[onnx.helper.make_tensor_value_info(transpose_output, TensorProto.UNDEFINED, ['a', 'b'])])

    model.graph.node.insert(node_index, transpose_node)
    node_index = node_index + 1
    skip = skip + 1
    ############

    mul_name_c = node.name + '_mul_c_'
    beta_name = node.name + '_const_beta_'
    add_name_c = node.name + '_add_c_'
    add_element_c = transpose_output

    if beta != 1.0:
        beta_proc = False 
        if length == 3:
            for init in model.graph.initializer:
                if node.input[2] == init.name:
                    beta_proc = True
                    v = values.get_init_value(model, init.name)

                    if isinstance(v, np.ndarray) == True:
                        C = v * beta
                    else:    
                        print('---init shape:', init.dims[0])
                        #print('---init value:', init.name, v)
                        C = np.array(v) * beta
                        print('C.shape:', C.shape)
                        C = C.tolist()

                    if is_shared_init(model, init.name, node.name) == True:    
                        C_ = onnx.helper.make_tensor(name=node.input[2] + '__',
                                            data_type=init.data_type,
                                            dims=[init.dims[0]],
                                            vals=C)

                        model.graph.initializer.append(C_) 

                        node.input[2] = node.input[2] + '__'
                    else:
                         values.set_tensor_value(init, C)   

                    break

            if beta_proc == False:
                for n in model.graph.node:
                    if node.input[2] == n.output[0]:
                        attributes = n.attribute
                        for attr in attributes:
                            if attr.name == 'value':
                                if is_shared_constant(model, node.input[2]):
                                    new_node = copy.deepcopy(n)
                                    new_name = n.name + '__'
                                    new_node.name = new_name
                                    new_node.output[0] = new_node.output[0] + '__'
                                    attrs = new_node.attribute

                                    for attr_ in attrs:
                                        if attr_.name == 'value':
                                            v_ = values.get_tensor_value(attr_.t)

                                            if isinstance(v_, np.ndarray) == True:
                                                C = v_ * beta
                                            else:
                                                C = [i * beta for i in v_]

                                            values.set_tensor_value(attr_.t, C)    

                                            break

                                    node.input[2] = new_node.output[0]
                                    model.graph.node.append(new_node)            
                                else: 
                                    v = values.get_tensor_value(attr.t)
                                    if isinstance(v, np.ndarray) == True:
                                        C = v * beta
                                    else:
                                        C = [i * beta for i in v]   

                                    values.set_tensor_value(attr.t, C)   
                                break         
                        break

    return skip