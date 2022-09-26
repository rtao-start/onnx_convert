import onnx
import sys
import values
import numpy as np
import copy
from onnx import TensorProto

constant_combination_case_1A = [False, False, False]
constant_combination_case_1B = [False, False, True]
constant_combination_case_1C = [False, False]

constant_combination_case_1 = [constant_combination_case_1A, constant_combination_case_1B, constant_combination_case_1C]

constant_combination_case_2 = [[False, True, False]]
#input_combination_case_2B = [False, True]

constant_combination_case_3A = [False, True, True]
constant_combination_case_3B = [False, True]

constant_combination_case_3 = [constant_combination_case_3A, constant_combination_case_3B]

constant_combination_case_4 = [[True, False, False]]
#input_combination_case_4B = [True, False]

constant_combination_case_5A = [True, False, True]
constant_combination_case_5B = [True, False]

constant_combination_case_5 = [constant_combination_case_5A, constant_combination_case_5B]

constant_combination_case = [constant_combination_case_1, constant_combination_case_2, constant_combination_case_3, constant_combination_case_4, constant_combination_case_5]

transpose_combination_case_A = [0, 0]
transpose_combination_case_B = [0, 1]
transpose_combination_case_C = [1, 0]
transpose_combination_case_D = [1, 1]

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

def proc_gemm_case_1(model, node_id, node, attr):
    pass

def proc_gemm_case_2(model, node_id, node, attr):
    pass

def proc_gemm_case_3(model, node_id, node, attr):
    print('proc_gemm_case_3-----------')
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    length = len(node.input)

    insert = False

    if transA != 30000:
        print('proc_gemm_case_3, Do TransA', node_id)
        insert = True
        output = node.input[0] + '_transpose_'
        transpose_output = onnx.helper.make_tensor_value_info(output, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=output,
                    inputs=[node.input[0]],
                    outputs=[output])

        node.input[0] = output
        #model.graph.node.append(transpose_node)
        model.graph.node.insert(node_id, transpose_node)

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
                for attr in attributes:
                    if attr.name == 'transB':
                        attr.i = 1

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
                                        C = v * 2 * 1 #beta
                                    else:
                                        C = [i * 1 * 2 for i in v]   

                                    values.set_tensor_value(attr.t, C)   
                                break         
                        break
    return insert

def proc_gemm_case_4(model, node_id, node, attr):
    pass        

def proc_gemm_case_5(model, node_id, node, attr):
    pass

proc_gemm = {
    "case_1": proc_gemm_case_1,
    "case_2": proc_gemm_case_2,
    "case_3": proc_gemm_case_3,
    "case_4": proc_gemm_case_4,
    "case_5": proc_gemm_case_5
}

def gemm_convert(model, output):
    dict_sm = {}
    dict_mul = {}

    got_swish = False

    init_list = []

    const_list = []

    for init in model.graph.initializer:
        init_list.append(init.name)

    for node in model.graph.node:
        if node.op_type == 'Constant': 
            const_list.append(node.output[0])

    skip = False            

    for node_id, node in enumerate(model.graph.node):
        print('loop model', node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                 ", op:", node.op_type)

        if skip == True:
            skip = False
            continue         

        input_const_flag = [False, False]
        length = len(node.input)
        if length == 3:
            input_const_flag.append(False)

        print('input_const_flag:', input_const_flag)    

        if node.op_type == 'Gemm':
            print('===== got gemm', node.name, node.input, node.output)

            alpha = 1.0
            beta = 1.0
            transA = 0
            transB = 0

            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'alpha':
                    alpha = attr.f
                    print('alpha:', alpha)
                
                if attr.name == 'beta':
                    beta = attr.f
                    print('beta:', beta)

                if attr.name == 'transA':
                    transA  = attr.i
                    print('transA :', transA)

                if attr.name == 'transB':
                    transB  = attr.i
                    print('transB :', transB) 

            gemm_attr = {}
            gemm_attr['alpha'] = alpha
            gemm_attr['beta'] = beta
            gemm_attr['transA'] = transA
            gemm_attr['transB'] = transB                   

            for i, input in enumerate(node.input):
                if input in init_list:
                    print('init input:', input, i)
                    input_const_flag[i] = True
                elif input in const_list:
                    print('const input:', input, i)
                    input_const_flag[i] = True        

            print('--- input_const_flag:', input_const_flag) 

            for index, c in enumerate(constant_combination_case):
                for e in c:
                    if e == input_const_flag:
                        print('index = ', index)
                        ii = index + 1
                        insert = proc_gemm['case_' + str(ii)](model, node_id, node, gemm_attr)
                        onnx.save(model, output)

                        if insert == True:
                            skip = True


model = onnx.load('./gemm2.onnx')

gemm_convert(model, './tmp.onnx')