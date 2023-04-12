import onnx
import sys, os
import numpy as np
import copy
from .utils import got_input_shape, is_shared_init, is_shared_constant
from onnx import TensorProto

sys.path.append(os.path.abspath('..'))
import values
import log

logger = log.getLogger(__name__, log.INFO)

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

def handle_common(model, node, attr, replace=True):
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    if transA != 0 and alpha != 1.0:
        A_name = node.input[0] + '__'
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[0] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                logger.debug('000 init shape: {} {}'.format(init.dims[0], init.dims[1]))
                logger.debug('000 init value: {}'.format(init.name))

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    logger.debug('+++A.shape: {}'.format(A.shape))
                    A = A.flatten()
                    A = A * alpha
                else:    
                    A = np.array(v).reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    logger.debug('---B.shape: {}'.format(B.shape))
                    A = A.flatten()
                    A = A * alpha
                    #A = A.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    A_ = onnx.helper.make_tensor(name=A_name,
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=A.tolist())

                    model.graph.initializer.append(A_) 

                    node.input[0] = A_name
                else:
                    values.set_tensor_value(init, A, dims_)

                break

        if alpha_proc == False:
            handle_constant_node(model, node, True, alpha)

        if replace == True:    
            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'transA':
                    found = True
                    attr.i = 0
            
            if found == False:
                attr = onnx.helper.make_attribute('transA', 0)
                node.attribute.append(attr)

            for attr in attributes:
                if attr.name == 'alpha':
                    attr.f = 1           
    elif alpha != 1.0:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[0] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                logger.debug('111 init shape: {} {}'.format(init.dims[0], init.dims[1]))
                logger.debug('111 init value: {}'.format(init.name))

                if isinstance(v, np.ndarray) == True:
                    A = v * alpha
                else:    
                    A = np.array(v) * alpha
                    #B = B.reshape(init.dims[0], init.dims[1])
                    logger.debug('A.shape: {}'.format(A.shape))
                    #A = A.tolist()

                A_name = node.input[0] + '__'

                if is_shared_init(model, init.name, node.name) == True:
                    A_ = onnx.helper.make_tensor(name=A_name,
                                        data_type=init.data_type,
                                        dims=[init.dims[0], init.dims[1]],
                                        vals=A.tolist())

                    model.graph.initializer.append(A_) 
                    node.input[0] = A_name
                else:
                    values.set_tensor_value(init, A)

                break

        if alpha_proc == False:
            handle_constant_node(model, node, False, alpha)  

        if replace == True:    
            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'alpha':
                    attr.f = 1      
    elif transA != 0:
        alpha_proc = False
        for init in model.graph.initializer:
            if node.input[0] == init.name:
                alpha_proc = True

                v = values.get_init_value(model, init.name)
                logger.debug('222 init shape: {} {}'.format(init.dims[0], init.dims[1]))
                logger.debug('222 init value: {}'.format(init.name))

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    logger.debug('=== A.shape: {}'.format(A.shape))
                    A = A.flatten()
                else:    
                    A = np.array(v).reshape(init.dims[0], init.dims[1])
                    A = A.transpose()
                    logger.debug('!!!! B.shape: {}'.format(A.shape))
                    A = A.flatten()
                    #A = A.tolist()

                dims_= [init.dims[1], init.dims[0]]

                if is_shared_init(model, init.name, node.name) == True:
                    A_ = onnx.helper.make_tensor(name=A_name,
                                        data_type=init.data_type,
                                        dims=dims_,
                                        vals=A.tolist())

                    model.graph.initializer.append(A_) 

                    node.input[0] = A_name
                else:
                    values.set_tensor_value(init, A, dims_)
                break

        if alpha_proc == False:
            handle_constant_node(model, node, True, 1.0)

        if replace == True:
            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'transA':
                    found = True
                    attr.i = 0
            
            if found == False:
                attr = onnx.helper.make_attribute('transA', 0)
                node.attribute.append(attr)    

def proc_gemm_ctc(model, node_id, node, attr):
    in_shape, _ = got_input_shape(model, node.input[0])

    logger.debug('proc_gemm_ctc, got input shape: {}'.format(in_shape))

    if in_shape > 32:
        logger.debug('in_shape > 32, goto proc_gemm_ctc_matmul')
        return proc_gemm_ctc_matmul(model, node_id, node, attr)

    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    if transB != 1:
        logger.debug('transB != 1, goto proc_gemm_ctc_matmul')
        return proc_gemm_ctc_matmul(model, node_id, node, attr)

    length = len(node.input)
    if length == 3: 
        c_name = node.input[2]

    skip = 0

    handle_common(model, node, attr)

    input_0 = node.input[0]
    input_1 = node.input[1]

    if length == 3:
        if beta != 1.0:
            beta_proc = False 
            for init in model.graph.initializer:
                if node.input[2] == init.name:
                    beta_proc = True
                    v = values.get_init_value(model, init.name)

                    if isinstance(v, np.ndarray) == True:
                        C = v * beta
                    else:    
                        logger.debug('---init shape: {}'.format(init.dims[0]))
                        #print('---init value:', init.name, v)
                        C = np.array(v) * beta
                        logger.debug('C.shape: {}'.format(C.shape))
                        #C = C.tolist()

                    if is_shared_init(model, init.name, node.name) == True:    
                        C_ = onnx.helper.make_tensor(name=node.input[2] + '__',
                                            data_type=init.data_type,
                                            dims=[init.dims[0]],
                                            vals=C.tolist())

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

def proc_gemm_ctc_matmul(model, node_id, node, attr): 
    alpha = attr['alpha']
    beta = attr['beta']
    transA = attr['transA']
    transB = attr['transB']

    node_index = node_id

    length = len(node.input)
    c_name = ''
    if length == 3: 
        c_name = node.input[2]

    outputB = ''

    skip = 0

    if transB != 0:
        logger.debug('proc_gemm_ctc_matmul, Do TransB : {}'.format(node_id))
        skip = skip + 1
        outputB = node.input[1] + '_transpose_'
        transpose_output = onnx.helper.make_tensor_value_info(outputB, TensorProto.UNDEFINED, ['a', 'b'])      

        transpose_node = onnx.helper.make_node(
                    'Transpose',
                    name=outputB,
                    inputs=[node.input[1]],
                    outputs=[outputB],
                    perm=[1,0])

        model.graph.node.insert(node_index, transpose_node)
        node_index = node_index + 1
        skip = skip + 1

    handle_common(model, node, attr, False)
    ################
    del node.attribute[:]

    node.op_type = 'MatMul'
    input_0 = node.input[0]
    input_1 = node.input[1]

    output_0 = node.output[0]

    if length == 3:
        del node.input[2:]

    if outputB != '':
        node.input[1] = outputB

    matmul_output_name = node.output[0] + '_matmul_'

    #############
    if length == 3:
        add_name = matmul_output_name + '_add_'
        add_element = matmul_output_name
        add_name_c = matmul_output_name + '_add_c_'

        if beta != 1.0 and beta > 0.0:
            node.output[0] = matmul_output_name
            beta_proc = False 
            for init in model.graph.initializer:
                if c_name == init.name:
                    beta_proc = True
                    v = values.get_init_value(model, init.name)

                    if isinstance(v, np.ndarray) == True:
                        C = v * beta
                    else:    
                        logger.debug('---init shape: {}'.format(init.dims[0]))
                        #print('---init value:', init.name, v)
                        C = np.array(v) * beta
                        logger.debug('C.shape: {}'.format(C.shape))

                    if is_shared_init(model, init.name, node.name) == True: 
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
                    skip = skip + 1

                    break

            if beta_proc == False:
                for n in model.graph.node:
                    if c_name == n.output[0]:
                        attributes = n.attribute
                        for attr in attributes:
                            if attr.name == 'value':
                                if is_shared_constant(model, c_name):
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
            logger.debug('proc_gemm_ctc_matmul, beta is 1.0')
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
                    skip = skip + 1

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
   