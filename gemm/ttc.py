import onnx
import sys, os
import numpy as np
import copy
from onnx import TensorProto
import utils

sys.path.append(os.path.abspath('..'))
import values

def proc_gemm_ttc_ttt(model, node_id, node, attr):
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    in_shape, _ = utils.got_input_shape(model, node.input[0])

    print('proc_gemm_ttc_ttt, got input shape:', in_shape)

    if in_shape < 32 and transA == 0 and transB == 1:
        print('in_shape < 32, goto proc_gemm_ttc_ttt_fc')
        return proc_gemm_ttc_ttt_fc(model, node_id, node, attr)

    length = len(node.input)
    if length == 3: 
        c_name = node.input[2]

    skip = 0

    node_index = node_id

    outputA = ''
    outputB = ''

    if transA != 0:
        print('proc_gemm_case_1, Do TransA', node_id)
        skip = skip + 1
        outputA = node.input[0] + '_transpose_'
        transpose_output = onnx.helper.make_tensor_value_info(outputA, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=outputA,
                    inputs=[node.input[0]],
                    outputs=[outputA])

        #node.input[0] = output
        #model.graph.node.append(transpose_node)
        model.graph.node.insert(node_index, transpose_node)

        node_index = node_index + 1

    if transB != 0:
        print('proc_gemm_case_1, Do TransB', node_id)
        skip = skip + 1
        outputB = node.input[1] + '_transpose_'
        transpose_output = onnx.helper.make_tensor_value_info(outputB, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=outputB,
                    inputs=[node.input[1]],
                    outputs=[outputB])

        #node.input[1] = outputB

        model.graph.node.insert(node_index, transpose_node)
        node_index = node_index + 1

    del node.attribute[:]

    node.op_type = 'MatMul'
    input_0 = node.input[0]
    input_1 = node.input[1]

    output_0 = node.output[0]

    if length == 3:
        del node.input[2:]

    if outputA != '':
        node.input[0] = outputA

    if outputB != '':
        node.input[1] = outputB

    matmul_output_name = node.output[0] + '_matmul_'
    #node.output[0] = matmul_output_name

    mul_node_name = ''
    mul_node_output = ''

    input_type = onnx.TensorProto.FLOAT

    for vi in model.graph.value_info:
        if vi.name == input_0:
            input_type = vi.type.tensor_type.elem_type
            print('XXX get type', input_type, input_0)

    if alpha != 1.0:
        node.output[0] = matmul_output_name
        mul_node_name = matmul_output_name + '_mul_'
        mul_node_output = mul_node_name + 'output_'
        if beta == 0:
            mul_node_output = output_0

        alpha_name = matmul_output_name + 'const_alpha'
        vals_=[alpha]
        #if input_type == 10:
        #    vals_=[int(alpha)]

        const_alpha = onnx.helper.make_tensor(name=alpha_name,
                            data_type=input_type,
                            dims=(),
                            vals=vals_)

        model.graph.initializer.append(const_alpha)                     

        mul_node = onnx.helper.make_node(
                    'Mul',
                    name=mul_node_name,
                    inputs=[alpha_name, matmul_output_name],
                    outputs=[mul_node_output])

        model.graph.node.insert(node_index, mul_node)
        node_index = node_index + 1

        skip = skip + 1

    #matmul_output_tensor = onnx.helper.make_tensor_value_info(matmul_output_name, TensorProto.UNDEFINED, ['a', 'b'])      

    if length == 3:
        add_name = matmul_output_name + '_add_'
        add_element = matmul_output_name
        mul_name_c = matmul_output_name + '_mul_c_'
        beta_name = matmul_output_name + 'const_beta'
        add_name_c = matmul_output_name + '_add_c_'

        if mul_node_output != '':
            add_name = mul_node_output + '_add_'
            add_element = mul_node_output
            mul_name_c = mul_node_output + '_mul_c_'
            beta_name = mul_node_output + 'const_beta'
            add_name_c = mul_node_output + '_add_c_'

        if beta != 1.0:
            if beta > 0.0:
                node.output[0] = matmul_output_name  
                beta_proc = False 
                for init in model.graph.initializer:
                    if c_name == init.name: # C is initializer
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
                            new_name = c_name + '__'    
                            C_ = onnx.helper.make_tensor(name=new_name,
                                                data_type=init.data_type,
                                                dims=[init.dims[0]],
                                                vals=C)

                            model.graph.initializer.append(C_) 

                            add_node = onnx.helper.make_node(
                                'Add',
                                name=add_name,
                                inputs=[new_name, add_element],
                                outputs=[output_0])
                        else:
                            values.set_tensor_value(init, C)

                            add_node = onnx.helper.make_node(
                                'Add',
                                name=add_name,
                                inputs=[init.name, add_element],
                                outputs=[output_0])

                        model.graph.node.insert(node_index, add_node)
                        node_index = node_index + 1           

                        break

                if beta_proc == False:
                    for n in model.graph.node:
                        if c_name == n.output[0] and node.op_type == 'Constant': #C is Constant
                            beta_proc = True
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

                                        #node.input[2] = new_node.output[0]
                                        
                                        add_node = onnx.helper.make_node(
                                                        'Add',
                                                        name=add_name,
                                                        inputs=[new_node.output[0], add_element],
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
                                                        inputs=[c_name, add_element],
                                                        outputs=[output_0])

                                    model.graph.node.insert(node_index, add_node)
                                    node_index = node_index + 1
                                    skip = skip + 1

                                    break         
                            break

                # C is Tensor
                if beta_proc == False:
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
        else:
            node.output[0] = matmul_output_name
            C_proc = False
            for init in model.graph.initializer:
                if c_name == init.name:
                    C_proc = True
                    add_node = onnx.helper.make_node(
                                'Add',
                                name=add_name,
                                inputs=[init.name, add_element],
                                outputs=[output_0])

                    model.graph.node.insert(node_index, add_node)
                    node_index = node_index + 1    

                    break    
            
            if C_proc == False:
                for n in model.graph.node:
                    if c_name == n.output[0]:
                        C_proc = True
                        attributes = n.attribute
                        for attr in attributes:
                            if attr.name == 'value':
                                add_node = onnx.helper.make_node(
                                                'Add',
                                                name=add_name,
                                                inputs=[c_name, add_element],
                                                outputs=[output_0])

                            model.graph.node.insert(node_index, add_node)
                            node_index = node_index + 1
                            skip = skip + 1
                            break         
                        break    

            # C is Tensor
            if C_proc == False:
                add_node = onnx.helper.make_node(
                    'Add',
                    name=add_name,
                    inputs=[c_name, add_element],
                    outputs=[output_0]) 

                model.graph.node.insert(node_index, add_node)
                node_index = node_index + 1
                skip = skip + 1  
                
    return skip

def proc_gemm_ttc_ttt_fc(model, node_id, node, attr):
    print('######## proc_gemm_ttc_ttt_fc, node.name:', node.name)

    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    length = len(node.input)
    if length == 3: 
        c_name = node.input[2]

    skip = 0

    node_index = node_id

    input_type = onnx.TensorProto.FLOAT

    for vi in model.graph.value_info:
        if vi.name == node.input[0]:
            input_type = vi.type.tensor_type.elem_type
            print('XXX get type', input_type, node.input[0])

    if alpha != 1.0:
        mul_node_name = node.input[0] + '_mul_'
        mul_node_output = mul_node_name + 'output_'
        alpha_name = mul_node_name + 'const_alpha'
        const_alpha = onnx.helper.make_tensor(name=alpha_name,
                            data_type=input_type,
                            dims=(),
                            vals=[alpha])

        model.graph.initializer.append(const_alpha)                     

        mul_node = onnx.helper.make_node(
                    'Mul',
                    name=mul_node_name,
                    inputs=[node.input[0], alpha_name],
                    outputs=[mul_node_output])

        model.graph.node.insert(node_index, mul_node)
        node_index = node_index + 1
        skip = skip + 1

        node.input[0] = mul_node_output

        attributes = node.attribute
        for attr in attributes:
            if attr.name == 'alpha':
                attr.f = 1

    if length == 3:
        if beta != 1.0:
            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'beta':
                    attr.f = 1

            beta_proc = False 
            for init in model.graph.initializer:
                if c_name == init.name: # C is initializer
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
                        new_name = c_name + '__'    
                        C_ = onnx.helper.make_tensor(name=new_name,
                                            data_type=init.data_type,
                                            dims=[init.dims[0]],
                                            vals=C)

                        model.graph.initializer.append(C_) 
                        node.input[2] = new_name
                    else:
                        values.set_tensor_value(init, C)

                    break

            if beta_proc == False:
                for n in model.graph.node:
                    if c_name == n.output[0] and node.op_type == 'Constant': #C is Constant
                        beta_proc = True
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
            # C is Tensor
            if beta_proc == False:
                beta_name = node.input[2] + '_const_beta'
                mul_name_c = node.input[2] + '_mul_'
                mul_c_output = mul_name_c + 'output_'

                for vi in model.graph.value_info:
                    if vi.name == c_name:
                        type_ = vi.type.tensor_type.elem_type

                        if len(vi.type.tensor_type.shape.dim) > 0:
                            shape_ = [s.dim_value for s in vi.type.tensor_type.shape.dim]
                            print('c_name: ', c_name, ', shape: ', shape_)

                            const_beta = onnx.helper.make_tensor(name=beta_name,
                                                data_type=type_,
                                                dims=(),
                                                vals=[beta])

                            model.graph.initializer.append(const_beta) 

                            mul_c = onnx.helper.make_tensor_value_info(mul_c_output, type_, shape_)                   

                            mul_node_c = onnx.helper.make_node(
                                        'Mul',
                                        name=mul_name_c,
                                        inputs=[c_name, beta_name],
                                        outputs=[mul_c_output])

                            node.input[2] = mul_c_output             

                            model.graph.node.insert(node_index, mul_node_c)
                            node_index = node_index + 1
                            skip = skip + 1 

                        break
                                                                          
    return skip