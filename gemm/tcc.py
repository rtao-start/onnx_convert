import onnx
import sys, os
import numpy as np
import copy
import utils
from onnx import TensorProto

sys.path.append(os.path.abspath('..'))
import values

def proc_gemm_tcc(model, node_id, node, attr):
    in_shape, _ = utils.got_input_shape(model, node.input[0])

    print('proc_gemm_tcc, got input shape:', in_shape)

    if in_shape > 32:
        print('in_shape > 32, goto proc_gemm_tcc_matmul')
        return proc_gemm_tcc_matmul(model, node_id, node, attr)

    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    length = len(node.input)

    skip = 0

    if transA != 0:
        print('transA != 0, goto proc_gemm_tcc_matmul')
        return proc_gemm_tcc_matmul(model, node_id, node, attr)
        
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
                    B = B.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if utils.is_shared_init(model, init.name, node.name) == True:
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

                for attr in attributes:
                    if attr.name == 'alpha':
                        attr.f = 1    

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

                if utils.is_shared_init(model, init.name, node.name) == True:
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

                if utils.is_shared_init(model, init.name, node.name) == True:
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

    if beta != 1.0:
        beta_proc = False 
        if length == 3:
            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'beta':
                    attr.f = 1  

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

                    if utils.is_shared_init(model, init.name, node.name) == True:    
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
                                if utils.is_shared_constant(model, node.input[2]):
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

def proc_gemm_tcc_matmul(model, node_id, node, attr):
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
        print('proc_gemm_tcc_matmul, Do TransA', node_id)
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

                if utils.is_shared_init(model, init.name, node.name) == True:
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

                if utils.is_shared_init(model, init.name, node.name) == True:
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

                if utils.is_shared_init(model, init.name, node.name) == True:
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
        add_name = matmul_output_name + '_add_'
        add_element = matmul_output_name
        add_name_c = matmul_output_name + '_add_c_'

        if beta > 1.0:
            node.output[0] = matmul_output_name
            beta_proc = False 
            for init in model.graph.initializer:
                if c_name == init.name:
                    beta_proc = True
                    v = values.get_init_value(model, init.name)

                    if isinstance(v, np.ndarray) == True:
                        C = v * beta
                    else:    
                        print('---init shape:', init.dims[0])
                        #print('---init value:', init.name, v)
                        C = np.array(v) * beta
                        print('C.shape:', C.shape)

                    if utils.is_shared_init(model, init.name, node.name) == True: 
                        new_name = c_name + '__'   
                        C_ = onnx.helper.make_tensor(name=new_name,
                                            data_type=init.data_type,
                                            dims=[init.dims[0]],
                                            vals=C.tolist())

                        model.graph.initializer.append(C_) 

                        add_node = onnx.helper.make_node(
                                'Add',
                                name=add_name,
                                inputs=[add_element, new_name],
                                outputs=[output_0])
                    else:
                        values.set_tensor_value(init, C)

                        add_node = onnx.helper.make_node(
                                'Add',
                                name=add_name,
                                inputs=[add_element, init.name],
                                outputs=[output_0])   

                    model.graph.node.insert(node_index, add_node)
                    node_index = node_index + 1

                    break

            if beta_proc == False:
                for n in model.graph.node:
                    if c_name == n.output[0]:
                        attributes = n.attribute
                        for attr in attributes:
                            if attr.name == 'value':
                                if utils.is_shared_constant(model, c_name):
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

                                    add_node = onnx.helper.make_node(
                                                    'Add',
                                                    name=add_name,
                                                    inputs=[add_element, new_node.output[0]],
                                                    outputs=[output_0])

                                    model.graph.node.append(new_node)         
                                else: 
                                    v = values.get_tensor_value(attr.t)
                                    if isinstance(v, np.ndarray) == True:
                                        C = v * beta
                                    else:
                                        C = [i * beta for i in v]   

                                    values.set_tensor_value(attr.t, C)  

                                    add_node = onnx.helper.make_node(
                                                        'Add',
                                                        name=add_name,
                                                        inputs=[add_element, c_name],
                                                        outputs=[output_0]) 

                                model.graph.node.insert(node_index, add_node)
                                node_index = node_index + 1
                                skip = skip + 1

                                break         
                        break
        elif beta == 1.0:
            node.output[0] = matmul_output_name
            print('proc_gemm_tcc_matmul, beta is 1.0')
            beta_proc = False 
            for init in model.graph.initializer:
                if c_name == init.name:
                    beta_proc = True
                    add_node = onnx.helper.make_node(
                                'Add',
                                name=add_name,
                                inputs=[add_element, init.name],
                                outputs=[output_0])   

                    model.graph.node.insert(node_index, add_node)
                    node_index = node_index + 1

                    break

            if beta_proc == False:
                for n in model.graph.node:
                    if c_name == n.output[0]:
                        add_node = onnx.helper.make_node(
                                                'Add',
                                                name=add_name,
                                                inputs=[add_element, c_name],
                                                outputs=[output_0]) 

                        model.graph.node.insert(node_index, add_node)
                        node_index = node_index + 1
                        skip = skip + 1   

                        break   

    return skip