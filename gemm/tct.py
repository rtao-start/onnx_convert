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
    else:
        

    return skip
