import onnx
import sys
import values, operation
import numpy as np
import log

logger = log.getLogger(__name__, log.DEBUG)

transpose_node_map = {}
reshape_node_map = {}

def get_prev_node_by_input(model, input_):
    n = model.graph.node[0]
    for node in model.graph.node:
        if input_ in node.output:
            return node, 0

    return n, -1

def get_next_node_by_output(model, output):
    n = model.graph.node[0]
    for node in model.graph.node:
        if output in node.input:
            return node, 0

    return n, -1

def get_all_next_node_by_output(model, output):
    node_list = []
    ok = -1

    for node in model.graph.node:
        if output in node.input:
            node_list.append(node)
            ok = 0

    return node_list, ok

def insert_node(model, insert_node, follow_up_node):
    for follow_up_node_index, _follow_up_node in enumerate(model.graph.node):
        if _follow_up_node == follow_up_node:
            logger.debug("follow_up_node_index: {}".format(follow_up_node_index))
            model.graph.node.insert(follow_up_node_index, insert_node)
            break

#Matmul-->Add->Reshape-->Transpose
def get_matmul_input_path_pattern_one(model, input_name):
    res = -1

    #node_list = []
    node_dict = {}

    logger.debug('get_matmul_input_path_pattern_one, input_name: {}'.format(input_name))

    input_pre, ok = get_prev_node_by_input(model, input_name)
    if ok == 0 and input_pre.op_type == 'Transpose':
        logger.debug('got match Transpose node: {}'.format(input_pre.name))
        attributes = input_pre.attribute
        for attr in attributes:
            if attr.name == 'perm':
                v = values.get_tensor_value(attr.t)
                logger.debug('got transpose shape{} for{}'.format(v, input_pre.name))
                break
        
        input_p_pre, ok = get_prev_node_by_input(model, input_pre.input[0])
        if ok == 0 and input_p_pre.op_type == 'Reshape':
            #####################
            data, shape = values.get_init_value_and_shape(model, input_p_pre.input[1])
            if isinstance(data, list) and data == []:
                logger.debug('reshape_data is not in initilizer')
                data = values.get_constant_value(model, input_p_pre.input[1])

            if len(data) == 4 or len(data) == 3:
                logger.debug('got match Reshape node: {}'.format(input_p_pre.name))
                ##################
                input_pp_pre, ok = get_prev_node_by_input(model, input_p_pre.input[0])
                if ok == 0 and input_pp_pre.op_type == 'Add':
                    ################
                    addA_name = input_pp_pre.input[0]
                    addA, shapeA = values.get_init_value_and_shape(model, input_pp_pre.input[0])
                    '''
                    if isinstance(addA, list) and addA == []:
                        print('addA is not in initilizer')
                        addA = values.get_constant_value(model, input_pp_pre.input[1])
                    '''
                    add_tensor_two = True
                    if len(shapeA) == 0:
                        addA_name = input_pp_pre.input[1]
                        add_tensor_two = False
                        addA, shapeA = values.get_init_value_and_shape(model, input_pp_pre.input[1])

                    if len(shapeA) == 1:
                        logger.debug('got match Add node: {}'.format(input_pp_pre.name))
                        ###########
                        add_input = input_pp_pre.input[1]
                        if add_tensor_two == False:
                            add_input = input_pp_pre.input[0]

                        input_ppp_pre, ok = get_prev_node_by_input(model, add_input)
                        logger.debug('----got matmul node: {}'.format(input_ppp_pre.name))
                        if ok == 0 and input_ppp_pre.op_type == 'MatMul':
                            ############################
                            shapeA = values.get_tensor_shape_by_name(model, input_ppp_pre.input[0])

                            inputB, shapeB = values.get_init_value_and_shape(model, input_ppp_pre.input[1])

                            if isinstance(inputB, list) and inputB == []:
                                logger.debug('inputB is not in initilizer')
                                inputB = values.get_constant_value(model, input_ppp_pre.input[1])

                            if (len(shapeA) == 3 or len(shapeA) == 2) and len(shapeB) == 2:
                                logger.debug('got match MatMul node: {}'.format(input_ppp_pre.name))
                                res = 0
                                node_list = [input_ppp_pre, input_pp_pre, input_p_pre, input_pre]
                                node_dict['node_list'] = node_list
                                node_dict['addA'] = addA_name
                                node_dict['matmul_AShape'] = shapeA
                                node_dict['inputB'] = inputB
                                node_dict['matmul_BShape'] = shapeB

                                input_pppp_pre, ok = get_prev_node_by_input(model, input_ppp_pre.input[0])
                                if ok == 0:
                                    node_dict['prev'] = input_pppp_pre.output[0]
                                    logger.debug('--- map key: {}'.format(input_pppp_pre.output[0]))
                                else:
                                    node_dict['prev'] = input_ppp_pre.input[0]
                                    logger.debug('pre node maybe input: {}'.format(input_ppp_pre.input[0]))    
    elif ok == 0:
        res = 1
        node_list = [input_pre]
        node_dict['node_list'] = node_list

    return node_dict, res

#Transpose-->Reshape-->Matmul-->Add-->Add
def get_matmul_input_path_pattern_two(model, input_name):
    res = -1

    node_dict = {}

    logger.debug('get_matmul_input_path_pattern_two, input_name: {}'.format(input_name))

    next_node, ok = get_next_node_by_output(model, input_name)
    if ok == 0 and next_node.op_type == 'Reshape':
        data, shape = values.get_init_value_and_shape(model, next_node.input[1])
        if isinstance(data, list) and data == []:
            logger.debug('---reshape_data is not in initilizer')
            data = values.get_constant_value(model, next_node.input[1])

        if len(data) == 3 or len(data) == 2:
            logger.debug('----got match Reshape node: {}'.format(next_node.name))

            n_next_node, ok = get_next_node_by_output(model, next_node.output[0])
            if ok == 0 and n_next_node.op_type == 'MatMul':
                #####################
                shapeA = values.get_tensor_shape_by_name(model, n_next_node.input[0])

                inputB, shapeB = values.get_init_value_and_shape(model, n_next_node.input[1])

                logger.debug('++++++++++++++++++shapeA, shapeB: {} {}'.format(shapeA, shapeB))

                if isinstance(inputB, list) and inputB == []:
                    logger.debug('inputB is not in initilizer')
                    inputB = values.get_constant_value(model, n_next_node.input[1])

                if (len(shapeA) == 3 or len(shapeA) == 2) and len(shapeB) == 2:
                    node_dict['matmul_AShape'] = shapeA
                    node_dict['inputB'] = inputB
                    node_dict['matmul_BShape'] = shapeB

                    logger.debug('----got match Matmul node: {}'.format(n_next_node.name))
                    ####################
                    nn_next_node, ok = get_next_node_by_output(model, n_next_node.output[0])
                    if ok == 0 and nn_next_node.op_type == 'Add':
                        ################
                        addA_name = nn_next_node.input[0]
                        node_dict['addA'] = addA_name
                        node_dict['addFirst'] = True
                        addA, shapeA = values.get_init_value_and_shape(model, nn_next_node.input[0])
                        
                        '''
                        if isinstance(addA, list) and addA == []:
                            print('addA is not in initilizer')
                            addA = values.get_constant_value(model, nn_next_node.input[1])
                        '''
                        if len(shapeA) == 0:
                            addA_name = nn_next_node.input[1]
                            node_dict['addA'] = addA_name
                            node_dict['addFirst'] = False
                            addA, shapeA = values.get_init_value_and_shape(model, nn_next_node.input[1])
                        
                        if len(shapeA) == 1:
                            logger.debug('---got match Add node: {}'.format(nn_next_node.name))
                            ###########
                            nnn_next_node, ok = get_next_node_by_output(model, nn_next_node.output[0])
                            if ok == 0 and nnn_next_node.op_type == 'Add':
                                logger.debug('----got match Add node2: {}'.format(n_next_node.name))
                                res = 0
                                node_list = [next_node, n_next_node, nn_next_node]
                                node_dict['node_list'] = node_list

    return node_dict, res

def get_add_combination_pattern_one(model):
    rm_list = []
    sub_list = []
    add_list = []

    ars_list = []

    for node in model.graph.node:
        if node.op_type == 'ReduceMean':
            rm_list.append(node)

        if node.op_type == 'Sub':
            sub_list.append(node) 

        if node.op_type == 'Add':
            add_list.append(node)  

    #print('rm_input_list:', rm_input_list)
    #print('sub_input_list:', sub_input_list)  
    #print('add_input_list:', add_input_list)

    for node in model.graph.node:
        if node.op_type == 'Add':
            match_rm = False
            match_sub = False
            match_add = False

            output = node.output[0]
            for rm_node in rm_list:
                if output in rm_node.input:
                    match_rm = True
                    match_rm_node = rm_node
                    break

            for sub_node in sub_list:
                if output in sub_node.input:
                    match_sub = True
                    match_sub_node = sub_node
                    break        

            for add_node in add_list:
                if output in add_node.input:
                    match_add = True
                    match_add_node = add_node
                    break

            if match_rm == True and match_sub == True and match_add == True:
                logger.debug('found match add node: {}'.format(node.name))
                ars = {}
                ars['nextAdd'] = match_add_node
                ars['currentAdd'] = node
                ars['ReduceMean'] = match_rm_node 
                ars['Sub'] = match_sub_node
                ars_list.append(ars) 

    return ars_list                     

def get_add_combination_pattern_two(model):
    matmul_list = []
    add_list = []
    rm_list = []
    sub_list = []

    am_list = []
    asr_list = []

    for node in model.graph.node:
        if node.op_type == 'Add':
            add_list.append(node)

        if node.op_type == 'MatMul':
            matmul_list.append(node)

        if node.op_type == 'ReduceMean':
            rm_list.append(node) 

        if node.op_type == 'Sub':
            sub_list.append(node)           

    for node in model.graph.node:
        if node.op_type == 'Add':
            match_add_node = None
            match_matmul_node_list = []

            nextAddInput1 = False

            output = node.output[0]
            for add_node in add_list:
                if output in add_node.input:
                    match_add_node = add_node
                    if match_add_node.input[1] == output:
                        nextAddInput1 = True
                    break

            for mm_node in matmul_list:
                if output in mm_node.input:
                    match_matmul_node_list.append(mm_node)

            if match_add_node != None and len(match_matmul_node_list) == 3:
                logger.debug('found match add node: {}'.format(node.name))
                am = {}
                am['nextAdd'] = match_add_node
                am['nextAddInput1'] = nextAddInput1
                am['currentAdd'] = node
                am['MatMulList'] = match_matmul_node_list 
                #am_list.append(am)

                match_rm_node = None
                match_sub_node = None

                next_add_output = match_add_node.output[0]
                for rm_node in rm_list:
                    if next_add_output in rm_node.input:
                        match_rm_node = rm_node
                        break

                for sub_node in sub_list:
                    if next_add_output in sub_node.input:
                        match_sub_node = sub_node
                        break        

                if match_rm_node != None and match_sub_node != None:
                    am['Sub'] = match_sub_node
                    am['ReduceMean'] = match_rm_node

                    logger.debug('got sub and reducemean: {} {}'.format(match_sub_node.name, match_rm_node.name))

                am_list.append(am)
 
    return am_list 


def get_add_combination_pattern_four(model):
    matmul_list = []
    add_list = []
    rm_list = []
    sub_list = []

    am_list = []
    asr_list = []

    for node in model.graph.node:
        if node.op_type == 'Add':
            add_list.append(node)

        if node.op_type == 'MatMul':
            matmul_list.append(node)

        if node.op_type == 'ReduceMean':
            rm_list.append(node) 

        if node.op_type == 'Sub':
            sub_list.append(node)           

    for node in model.graph.node:
        if node.op_type == 'Reshape':
            match_add_node = None
            match_matmul_node_list = []

            output = node.output[0]
            for add_node in add_list:
                if output in add_node.input:
                    match_add_node = add_node
                    break

            for mm_node in matmul_list:
                if output in mm_node.input:
                    match_matmul_node_list.append(mm_node)

            if match_add_node != None and len(match_matmul_node_list) == 3:
                logger.debug('found match reshape node: {}'.format(node.name))
                am = {}
                am['nextAdd'] = match_add_node
                am['currentAdd'] = node
                am['MatMulList'] = match_matmul_node_list 
                #am_list.append(am)

                match_rm_node = None
                match_sub_node = None

                next_add_output = match_add_node.output[0]
                for rm_node in rm_list:
                    if next_add_output in rm_node.input:
                        match_rm_node = rm_node
                        break

                for sub_node in sub_list:
                    if next_add_output in sub_node.input:
                        match_sub_node = sub_node
                        break        

                if match_rm_node != None and match_sub_node != None:
                    am['Sub'] = match_sub_node
                    am['ReduceMean'] = match_rm_node

                    logger.debug('got sub and reducemean: {} {}'.format(match_sub_node.name, match_rm_node.name))

                am_list.append(am)
 
    return am_list

def get_add_combination_pattern_seven(model):
    add_list = []
    rm_list = []
    sub_list = []

    am_list = []
    asr_list = []

    for node in model.graph.node:
        if node.op_type == 'Add':
            add_list.append(node)

        if node.op_type == 'ReduceMean':
            rm_list.append(node) 

        if node.op_type == 'Sub':
            sub_list.append(node)           

    for node in model.graph.node:
        if node.op_type == 'Add':
            match_add_node = None

            nextAddInput1 = False

            output = node.output[0]
            for add_node in add_list:
                if output in add_node.input:
                    match_add_node = add_node
                    if match_add_node.input[1] == output:
                        nextAddInput1 = True
                    break

            if match_add_node != None:
                logger.debug('found match add node: {}'.format(node.name))
                am = {}
                am['nextAdd'] = match_add_node
                am['nextAddInput1'] = nextAddInput1
                am['currentAdd'] = node

                '''
                match_rm_node = None
                match_sub_node = None

                next_add_output = match_add_node.output[0]
                for rm_node in rm_list:
                    if next_add_output in rm_node.input:
                        match_rm_node = rm_node
                        break

                for sub_node in sub_list:
                    if next_add_output in sub_node.input:
                        match_sub_node = sub_node
                        break  
                '''              

                if True:#match_rm_node != None and match_sub_node != None:
                    #am['nextSub'] = match_sub_node
                    #am['nextReduceMean'] = match_rm_node

                    #logger.debug('got sub and reducemean: {} {}'.format(match_sub_node.name, match_rm_node.name))

                    all_next_node, _ = get_all_next_node_by_output(model, node.output[0])
                    if len(all_next_node) == 3:
                        for n in all_next_node:
                            print('+++++++ next node: ', n.name)
                            if n.op_type == 'Sub':
                                print('Got sub')
                                am['currentSub'] = n
                            elif n.op_type == 'ReduceMean':
                                print('Got ReduceMean')
                                am['currentReduceMean'] = n    

                        am_list.append(am)
 
    return am_list 

def get_add_combination_pattern_eight(model):
    block_list = []

    for node in model.graph.node:
        if node.op_type == 'Add':
            all_next_node, ok = get_all_next_node_by_output(model, node.output[0])
            if ok == 0 and len(all_next_node) == 3:
                asra = {}
                asra['currentAdd'] = node

                for n in all_next_node:
                    asra[n.op_type] = n
                
                if 'Sub' in asra.keys() and 'ReduceMean' in asra.keys() and 'Add' in asra.keys():
                    block_list.append(asra)

    return block_list                

def get_add_combination_pattern_five(model):
    pass

#MatMul->Transpose->Reshape->MatMul->Add
def match_mtrma(model, add_node):
    ret = -1

    add_node_prev, ok = get_prev_node_by_input(model, add_node.input[1])
    if ok == 0 and add_node_prev.op_type == 'Add':
        mm_node, ok = get_prev_node_by_input(model, add_node_prev.input[1])
        if ok == 0 and mm_node.op_type == 'MatMul':
            rs_node, ok = get_prev_node_by_input(model, mm_node.input[0])
            if ok == 0 and rs_node.op_type == 'Reshape':
                ts_node, ok = get_prev_node_by_input(model, rs_node.input[0])
                if ok == 0 and ts_node.op_type == 'Transpose':
                    mm_node, ok = get_prev_node_by_input(model, ts_node.input[0])
                    if ok == 0 and mm_node.op_type == 'MatMul':
                        ret = 0
                        logger.debug('match_mtrma, return true {}'.format(mm_node.name))

    return ret

def handle_add_combination_pattern_seven(model):
    am_list = get_add_combination_pattern_seven(model)

    logger.debug('handle_add_combination_pattern_seven------------')

    last_add_node = None

    #if len(am_list):
    for idx, am in enumerate(am_list):
        add_node = am['currentAdd']
        next_add_node = am['nextAdd']
        last_add_node = next_add_node
        nextAddInput1 = am['nextAddInput1'] 

        logger.debug('handle_add_combination_pattern_seven, add_node: {}, next_add_node: {}'.format(add_node.name, next_add_node.name))

        if idx == 0:
            ###add transpose
            ts_name = add_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
            ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            
            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[add_node.output[0]],
                                                outputs=[ts_output_name],
                                                perm=[0,2,1])

            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node, add_node)

            ###add reshape-1
            rs_name = add_node.name + '_reshape_1_'
            rs_output_name = rs_name + '_output_'
            rs_output_shape = [ts_output_shape[1], ts_output_shape[2]] #TBD

            rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

            const_shape_name = add_node.name + '_reshape_data_'
            
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs_output_shape)],
                                vals=rs_output_shape)

            model.graph.initializer.append(const_shape_tensor)

            rs_node = onnx.helper.make_node(
                                                'Reshape',
                                                name=rs_name,
                                                inputs=[ts_output_name, const_shape_name],
                                                outputs=[rs_output_name])

            model.graph.value_info.append(rs_output)

            insert_node(model, rs_node, ts_node)

            ###add reshape-2
            rs2_name = add_node.name + '_reshape_2_'
            rs2_output_name = rs2_name + '_output_'
            rs2_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
            rs2_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

            const_shape2_name = add_node.name + '_reshape2_data_'
            
            const_shape2_tensor = onnx.helper.make_tensor(name=const_shape2_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs2_output_shape)],
                                vals=rs2_output_shape)

            model.graph.initializer.append(const_shape2_tensor)


            ######################################################
            ######################################################
            rs2_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[rs_output_name, const_shape2_name],
                                            outputs=[rs2_output_name])

            model.graph.value_info.append(rs2_output)

            insert_node(model, rs2_node, rs_node)

            if nextAddInput1 == True:
                next_add_node.input[1] = rs2_output_name
            else:    
                next_add_node.input[0] = rs2_output_name

        #ret = match_mtrma(model, add_node)
        logger.debug('---handle Add node {}'.format(add_node.name))
        if 'currentSub' in am.keys():
            logger.debug('+++handle Add node {}'.format(add_node.name))
        #if True: #ret == 0 or idx == 0:
            sub_node = am['currentSub']
            rm_node = am['currentReduceMean']

            ts_name = sub_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'

            add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
            ts_output_shape = add_output_shape#[add_output_shape[0], add_output_shape[1], add_output_shape[2]]
            
            if idx != 0:
                add_output_shape_new = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
                update_tensor_shape(model, add_node.output[0], add_output_shape_new)
            
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            
            input_name = add_node.output[0]
            if idx == 0:
                input_name = rs2_output_name

            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[input_name],
                                                outputs=[ts_output_name],
                                                perm=[0,2,1])

            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node, sub_node)

            sub_node.input[0] = ts_output_name
            rm_node.input[0] = ts_output_name

    if last_add_node != None:
        print('last_add_node:', last_add_node.name)
        last_output = last_add_node.output[0]
        for output in model.graph.output:
            if output.name == last_output:
                ###add transpose
                ts_name = last_add_node.name + '_transpose_'
                ts_output_name = ts_name + '_output_'
                add_output_shape = values.get_tensor_shape_by_name(model, last_add_node.output[0])
                if len(add_output_shape) == 0:
                    add_output_shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]

                print('add_output_shape:', add_output_shape)
                ts_output_shape = [add_output_shape[0], add_output_shape[1], add_output_shape[2]]
                transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
                
                ts_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[last_add_node.output[0]],
                                                    outputs=[ts_output_name],
                                                    perm=[0,2,1])

                model.graph.value_info.append(transpose_output)

                insert_node(model, ts_node, last_add_node)

                del model.graph.output[:]
                #model.graph.output.extend([onnx.ValueInfoProto(name=ts_output_name)])
                model.graph.output.extend([transpose_output])

                break

#       |-->Squeeze-----------------|
#Split----->Squeeze-->Transpose-->MatMul-->Mul-->Softmax-----|
#       |-->Squeeze---------------------------------------->MatMul
def get_matmul_block_pattern_seven(model, outputs, node_dict):
    for out in outputs:
        node, _ = get_next_node_by_output(model, out)
        next_node, _ = get_next_node_by_output(model, node.output[0])
        if next_node.op_type == 'Transpose':
            node_dict['Squeeze_K'] = node
            node_dict['Transpose'] = next_node
            mm_node, _ = get_next_node_by_output(model, next_node.output[0])
            if 'MatMulQK' not in node_dict.keys():
                node_dict['MatMul_QK'] = mm_node
        elif next_node.op_type == 'MatMul':
            ts_mul_node, _ = get_next_node_by_output(model, next_node.output[0])
            if ts_mul_node.op_type == 'Transpose':
                node_dict['MatMul_V'] = next_node
                node_dict['Squeeze_V'] = node
            elif ts_mul_node.op_type == 'Mul':
                node_dict['Mul'] = ts_mul_node
                node_dict['Softmax'], _ = get_next_node_by_output(model, ts_mul_node.output[0]) 
                if 'MatMulQK' not in node_dict.keys():
                    node_dict['MatMul_QK'] = next_node
                    node_dict['Squeeze_Q'] = node 

#Matmul-->Add->Reshape-->Transpose->Split
def get_matmul_input_path_pattern_seven(model):
    res = -1
    node_list = []
    
    logger.debug('get_matmul_input_path_pattern_seven')

    for node in model.graph.node:
        if node.op_type == 'MatMul':
            inputB = node.input[1]

            is_init = False

            for init in model.graph.initializer:
                if init.name == inputB:
                    is_init = True
                    break

            if is_init == False:
                dataB = values.get_constant_value(model, inputB)

                if dataB != []:
                    is_init = True

            if is_init == False:
                logger.debug('skip MatMul: {}'.format(node.name))
                continue

            add_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and add_node.op_type == 'Add':
                mulA = values.get_init_value(model, add_node.input[0])
                logger.debug('matmul input is Add: {}'.format(add_node.name))

                if isinstance(mulA, list) and mulA == []:
                    logger.debug('mulA is not in initilizer')
                    mulA = values.get_constant_value(model, add_node.input[1])

                if len(mulA) > 0 :
                    logger.debug('this is the add-node which we wanted...')
                    inputA = add_node.input[0]

                    rs_node, ok = get_next_node_by_output(model, add_node.output[0])
                    if ok == 0 and rs_node.op_type == 'Reshape':
                        ts_node, ok = get_next_node_by_output(model, rs_node.output[0])
                        if ok == 0 and ts_node.op_type == 'Transpose':
                            split_node, ok = get_next_node_by_output(model, ts_node.output[0])
                            if ok == 0 and split_node.op_type == 'Split':
                                logger.debug('this is the split-node which we wanted...')
                                if len(split_node.output) == 3:
                                    squeeze_cnt = 0
                                    for out in split_node.output:
                                        squeeze_node, ok = get_next_node_by_output(model, out)
                                        if ok == 0 and squeeze_node.op_type == 'Squeeze':
                                            squeeze_cnt = squeeze_cnt + 1

                                    if squeeze_cnt == 3:
                                        node_dict = {}
                                        node_dict['Split'] = split_node
                                        node_dict['FirstMatMul'] = node
                                        node_dict['Add'] = add_node
                                        logger.debug('got split block: {}'.format(split_node.name))
                                        get_matmul_block_pattern_seven(model, split_node.output, node_dict)
                                        node_list.append(node_dict)

    for nn in node_list:
        logger.debug('----got block, q: {}, k: {}, v: {}'.format(nn['Squeeze_Q'].name, nn['Squeeze_K'].name, nn['Squeeze_V'].name))

    return node_list


#MatMul-->Transpose-->Reshape-->MatMul-->Add
def handle_trma(model, matmul_node):
    tp_node, _ = get_next_node_by_output(model, matmul_node.output[0])
    rs_node, _ = get_next_node_by_output(model, tp_node.output[0])
    mm_node, _ = get_next_node_by_output(model, rs_node.output[0])
    add_node, _ = get_next_node_by_output(model, mm_node.output[0])

    ##Transpose-->Reshape
    tp_node.op_type = 'Reshape'
    del tp_node.attribute[:]

    const_shape_name = tp_node.output[0] + '_reshape_data_'
    tp_input_shape = values.get_tensor_shape_by_name(model, tp_node.input[0])
    tp_output_shape = [tp_input_shape[0], tp_input_shape[1]*tp_input_shape[2], tp_input_shape[3]]
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(tp_output_shape)],
                        vals=tp_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    tp_node.input.append(const_shape_name)

    tp_node.input[1] = const_shape_name

    update_tensor_shape(model, tp_node.output[0], tp_output_shape)

    ##Reshape-->Reshape
    const_shape_name = tp_node.output[0] + '_reshape_data2_'

    rs_output_shape = [tp_output_shape[0], tp_output_shape[1], 1, tp_output_shape[2]]
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    rs_node.input[1] = const_shape_name

    update_tensor_shape(model, rs_node.output[0], rs_output_shape)

    ##MatMul-->Conv
    mm_node.op_type = 'Conv'
    logger.debug('-----reuse MatMul to Conv')
    const_x_name = mm_node.name + '_to_conv_x_'

    inputB, shapeB = values.get_init_value_and_shape(model, mm_node.input[1])

    v = inputB
    old_dims = [shapeB[0], shapeB[1]]
    dims_ = [shapeB[1], shapeB[0], 1, 1]

    operation.remove_initializer_if_necessary_by_name(model, mm_node.input[1], mm_node)
        
    if isinstance(v, np.ndarray) == True:
        A = v.reshape(*old_dims)
        A = A.transpose()
        A = A.reshape(*dims_)
        logger.debug('+++A.shape: {}'.format(A.shape))
        A = A.flatten()
    else:    
        A = np.array(v).reshape(*old_dims)
        A = A.transpose()
        A = A.reshape(*dims_)
        logger.debug('---A.shape: {}'.format(A.shape))
        A = A.flatten()

    A = A.tolist()  
    const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=dims_,
                        vals=A)

    model.graph.initializer.append(const_x_tensor)
    mm_node.input[1] = const_x_name

    attr = onnx.helper.make_attribute('dilations', [1, 1])
    mm_node.attribute.append(attr)

    attr = onnx.helper.make_attribute('group', 1)
    mm_node.attribute.append(attr)

    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
    mm_node.attribute.append(attr)

    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
    mm_node.attribute.append(attr)

    attr = onnx.helper.make_attribute('strides', [1,1])
    mm_node.attribute.append(attr)        

    mm_node.input.append(add_node.input[0])   

    output_shape = values.get_tensor_shape_by_name(model, mm_node.output[0])
    conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]]

    update_tensor_shape(model, mm_node.output[0], conv_output_shape) 

    #Add--->Reshape
    add_node.op_type = 'Reshape'

    del add_node.attribute[:]

    rs_name = add_node.name + '_reshape_1_'
    rs_output_name = rs_name + '_output_'
    rs_output_shape = [conv_output_shape[0], conv_output_shape[1], conv_output_shape[3]]
    logger.debug('-----rs_output_shape: {}'.format(rs_output_shape))

    const_shape_name = add_node.name + '_reshape_data_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    add_node.input[0] = add_node.input[1]
    add_node.input[1] = const_shape_name

    update_tensor_shape(model, add_node.output[0], rs_output_shape)

#        |->ReduceMean
#MatMul---->Sub
#        |->Add
def get_msra_block(model):
    msra_list = []

    for node in model.graph.node:
        if node.op_type == 'MatMul':
            msra = {}
            all_next_node, _ = get_all_next_node_by_output(model, node.output[0])
            if len(all_next_node) == 3:
                for n in all_next_node:
                   msra[n.op_type] = n
                
                if 'Add' in msra.keys() and 'Sub' in msra.keys() and 'ReduceMean' in msra.keys():
                    add_node, _ = get_prev_node_by_input(model, node.input[0]) 
                    msra['preAdd'] = add_node
                    msra['MatMul'] = node
                    msra_list.append(msra)      

    return msra_list

def handle_msra_block(model):
    msra_list = get_msra_block(model)
    for msra in msra_list:
        add_node = msra['preAdd']
        matmul_node = msra['MatMul']

        ###insert transpose
        ts_name = add_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'

        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[add_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        model.graph.value_info.append(transpose_output)

        insert_node(model, ts_node, add_node)

        ###add reshape-1
        rs_name = add_node.output[0] + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add_node.output[0] + '_reshape_data_1'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, ts_node)

        #MatMul1--->Conv
        matmul_node.op_type = 'Conv'
        matmul_node.input[0] = rs_output_name
        logger.debug('-----reuse MatMul to Conv')
        const_x_name = matmul_node.name + '_to_conv_x_'

        inputB, shapeB = values.get_init_value_and_shape(model, matmul_node.input[1])
        v = inputB
        old_dims = [shapeB[0], shapeB[1]]
        dims_ = [shapeB[1], shapeB[0],1,1]

        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        operation.remove_initializer_if_necessary_by_name(model, matmul_node.input[1], matmul_node)
        
        model.graph.initializer.append(const_x_tensor)
        matmul_node.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul_node.attribute.append(attr)        

        output_shape = values.get_tensor_shape_by_name(model, matmul_node.output[0])
        conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]]

        update_tensor_shape(model, matmul_node.output[0], conv_output_shape)

        ###add reshape-2
        rs_name = matmul_node.output[0] + '_reshape_2_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [conv_output_shape[0], conv_output_shape[1], conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = matmul_node.output[0] + '_reshape_data_2'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[matmul_node.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, matmul_node)

        next_add_node = msra['Add']
        next_add_node.input[0] = rs_output_name

        ###insert transpose2
        ts_name = rs_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'

        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[rs_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        add_output_shape = values.get_tensor_shape_by_name(model, rs_node.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        model.graph.value_info.append(transpose_output)

        insert_node(model, ts_node, rs_node)

        sub_node = msra['Sub']
        rm_node = msra['ReduceMean']

        sub_node.input[0] = ts_output_name
        rm_node.input[0] = ts_output_name

        ###insert transpose3
        insert_transpose = True
        all_next_node, _ = get_all_next_node_by_output(model, next_add_node.output[0])
        for n in all_next_node:
            if n.op_type == 'Transpose':
                insert_transpose = False
                break

        if insert_transpose == True:
            ts_name = next_add_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'

            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[next_add_node.output[0]],
                                                outputs=[ts_output_name],
                                                perm=[0,2,1])

            add_output_shape = values.get_tensor_shape_by_name(model, next_add_node.output[0])
            ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node, next_add_node)

            all_next_node, _ = get_all_next_node_by_output(model, next_add_node.output[0])
            for n in all_next_node:
                if n.name != ts_name:
                    n.input[0] = ts_output_name


#             |->Slice->| 
#             |->Slice->|
#Add-->Reshape-->Slice--->Concat-->Reshape
#             |->Slice->|
def get_ars4cr_block(model):
    asr4cr_list = []
    for node in model.graph.node:
        if node.op_type == 'Add':
            rs_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and rs_node.op_type == 'Reshape':
                all_next_node, _ = get_all_next_node_by_output(model, rs_node.output[0])
                if len(all_next_node) == 4:
                    ars4cr = {}
                    ars4cr['Add'] = node
                    ars4cr['Reshape'] = rs_node
                    ars4cr['SliceList'] = []

                    for n in all_next_node:
                        if n.op_type == 'Slice':
                            ars4cr['SliceList'].append(n)

                    if len(ars4cr['SliceList']) == 4:
                        logger.debug('got asr4: {}'.format(node.name))
                        concat_node, _ = get_next_node_by_output(model, ars4cr['SliceList'][0].output[0])
                        ars4cr['Concat'] = concat_node
                        rs_node2, _ = get_next_node_by_output(model, concat_node.output[0])
                        ars4cr['Reshape2'] = rs_node2

                        asr4cr_list.append(ars4cr)

    return asr4cr_list

def handle_ars4cr(model):
    ars4cr_list = get_ars4cr_block(model)
    for ars4cr in ars4cr_list:
        rs_node = ars4cr['Reshape']
        concat_node = ars4cr['Concat']
        rs_node2 = ars4cr['Reshape2']

        rs_output_shape_old = values.get_tensor_shape_by_name(model, rs_node.output[0])
        rs_output_shape = [rs_output_shape_old[0], rs_output_shape_old[3], rs_output_shape_old[1], rs_output_shape_old[2]]
        update_tensor_shape(model, rs_node.output[0], rs_output_shape)

        const_shape_name = rs_node.output[0] + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        update_tensor_shape(model, rs_node.output[0], rs_output_shape)

        operation.remove_initializer_if_necessary_by_name(model, rs_node.input[1], rs_node)

        rs_node.input[1] = const_shape_name

        ####
        const_axes_name = rs_node.output[0] + '_axes_'

        const_axes_tensor = onnx.helper.make_tensor(name=const_axes_name,
                    data_type=onnx.TensorProto.INT64,
                    dims=[2],
                    vals=[2,3])

        model.graph.initializer.append(const_axes_tensor)

        slice_list = ars4cr['SliceList']
        for slice_node in slice_list:
            operation.remove_initializer_if_necessary_by_name(model, slice_node.input[3], slice_node)
            slice_node.input[3] = const_axes_name

            slice_output_shape_old = values.get_tensor_shape_by_name(model, slice_node.output[0])
            slice_output_shape = [slice_output_shape_old[0], slice_output_shape_old[3], slice_output_shape_old[1], slice_output_shape_old[2]]
            update_tensor_shape(model, slice_node.output[0], slice_output_shape)

        del concat_node.attribute[:]

        attr = onnx.helper.make_attribute('axis', 1)
        concat_node.attribute.append(attr)

        concat_output_shape_old = values.get_tensor_shape_by_name(model, concat_node.output[0])
        concat_output_shape = [concat_output_shape_old[0], concat_output_shape_old[3], concat_output_shape_old[1], concat_output_shape_old[2]]
        update_tensor_shape(model, concat_node.output[0], concat_output_shape)

        ###update reshape-2
        rs_output_shape_old = values.get_tensor_shape_by_name(model, rs_node2.output[0])
        rs_output_shape = [rs_output_shape_old[0], rs_output_shape_old[2], rs_output_shape_old[1]]

        const_shape_name = rs_node2.output[0] + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        update_tensor_shape(model, rs_node2.output[0], rs_output_shape)

        operation.remove_initializer_if_necessary_by_name(model, rs_node2.input[1], rs_node2)

        rs_node2.input[1] = const_shape_name

        ###insert transpose
        ts_name = rs_node2.name + '_transpose_'
        ts_output_name = ts_name + '_output_'

        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[rs_node2.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        add_output_shape = values.get_tensor_shape_by_name(model, rs_node2.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        model.graph.value_info.append(transpose_output)

        insert_node(model, ts_node, rs_node2)

        all_next_node, _ = get_all_next_node_by_output(model, rs_node2.output[0])
        for n in all_next_node:
            if n.name != ts_name:
                n.input[0] = ts_output_name
        
#MatMul-->Transpose-->Reshape-->MatMul-->Add
def handle_trma_pattern_eight(model, matmul_node):
    handle_trma(model, matmul_node)

    tp_node, _ = get_next_node_by_output(model, matmul_node.output[0])
    rs_node, _ = get_next_node_by_output(model, tp_node.output[0])
    mm_node, _ = get_next_node_by_output(model, rs_node.output[0])
    add_node, _ = get_next_node_by_output(model, mm_node.output[0])
    rs_node, _ = get_next_node_by_output(model, add_node.output[0])
    rs_node2, _ = get_next_node_by_output(model, rs_node.output[0])

    ts_node2, _ = get_next_node_by_output(model, rs_node2.output[0])

    rs_node3, _ = get_next_node_by_output(model, ts_node2.output[0])


    rs_node4, _ = get_next_node_by_output(model, rs_node3.output[0])

    next_node, _ = get_next_node_by_output(model, rs_node4.output[0])

    ts_name = add_node.name + '_transpose_'
    ts_output_name = ts_name + '_output_'

    ts_node = onnx.helper.make_node(
                                        'Transpose',
                                        name=ts_name,
                                        inputs=[add_node.output[0]],
                                        outputs=[ts_output_name],
                                        perm=[1,0,2])

    add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
    ts_output_shape = [add_output_shape[1], add_output_shape[0], add_output_shape[2]]
    transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
    model.graph.value_info.append(transpose_output)

    insert_node(model, ts_node, add_node)
    rs_node.input[0] = ts_output_name

    ###update reshape-1
    rs_output_shape_old = values.get_tensor_shape_by_name(model, rs_node.output[0])
    rs_output_shape = [rs_output_shape_old[3], rs_output_shape_old[0], rs_output_shape_old[1], rs_output_shape_old[2]]

    const_shape_name = rs_node.output[0] + '_reshape_data_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    update_tensor_shape(model, rs_node.output[0], rs_output_shape)

    operation.remove_initializer_if_necessary_by_name(model, rs_node.input[1], rs_node)

    rs_node.input[1] = const_shape_name

    ###update reshape-2
    rs_output_shape_old = values.get_tensor_shape_by_name(model, rs_node2.output[0])
    rs_output_shape = [rs_output_shape_old[0], rs_output_shape_old[5], rs_output_shape_old[1], rs_output_shape_old[2], rs_output_shape_old[3], rs_output_shape_old[4]]

    const_shape_name = rs_node2.output[0] + '_reshape_data_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    update_tensor_shape(model, rs_node2.output[0], rs_output_shape)

    operation.remove_initializer_if_necessary_by_name(model, rs_node2.input[1], rs_node2)

    rs_node2.input[1] = const_shape_name

    ###update transpose node
    ts_output_shape_old = values.get_tensor_shape_by_name(model, ts_node2.output[0])
    ts_output_shape = [ts_output_shape_old[0], ts_output_shape_old[5],ts_output_shape_old[1],ts_output_shape_old[2],ts_output_shape_old[3],ts_output_shape_old[4]]
    del ts_node2.attribute[:]

    attr = onnx.helper.make_attribute('perm', [0,1,2,4,3,5])
    ts_node2.attribute.append(attr)

    update_tensor_shape(model, ts_node2.output[0], ts_output_shape)

    ###update reshape-3
    rs_output_shape_old = values.get_tensor_shape_by_name(model, rs_node3.output[0])
    rs_output_shape = [rs_output_shape_old[0], rs_output_shape_old[3], rs_output_shape_old[1], rs_output_shape_old[2]]

    const_shape_name = rs_node3.output[0] + '_reshape_data_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    update_tensor_shape(model, rs_node3.output[0], rs_output_shape)

    operation.remove_initializer_if_necessary_by_name(model, rs_node3.input[1], rs_node3)

    rs_node3.input[1] = const_shape_name

    ###update reshape-4
    if rs_node4.op_type == 'Reshape':
        rs_output_shape_old = values.get_tensor_shape_by_name(model, rs_node4.output[0])
        rs_output_shape = [rs_output_shape_old[0], rs_output_shape_old[2], rs_output_shape_old[1]]

        const_shape_name = rs_node4.output[0] + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        update_tensor_shape(model, rs_node4.output[0], rs_output_shape)

        operation.remove_initializer_if_necessary_by_name(model, rs_node4.input[1], rs_node4)

        rs_node4.input[1] = const_shape_name
        
        next_add_output_shape = values.get_tensor_shape_by_name(model, next_node.output[0]) 
        update_tensor_shape(model, next_node.output[0], [next_add_output_shape[0], next_add_output_shape[2], next_add_output_shape[1]])  
    else:
        all_next_node, _ = get_all_next_node_by_output(model, rs_node3.output[0])
        if len(all_next_node) == 2:
            concat_node = None
            const_axes_name = ''
            for node in all_next_node:
                if node.op_type == 'Slice':
                    slice_output_shape_old = values.get_tensor_shape_by_name(model, node.output[0])
                    slice_output_shape = [slice_output_shape_old[0], slice_output_shape_old[3], slice_output_shape_old[1], slice_output_shape_old[2]]
                    update_tensor_shape(model, node.output[0], slice_output_shape)

                    if const_axes_name == '':
                        const_axes_name = node.name + '_' + node.input[3] + '_axes_'
            
                        const_axes_tensor = onnx.helper.make_tensor(name=const_axes_name,
                                            data_type=onnx.TensorProto.INT64,
                                            dims=[1],
                                            vals=[2])

                        model.graph.initializer.append(const_axes_tensor)

                    operation.remove_initializer_if_necessary_by_name(model, node.input[3], node)

                    node.input[3] = const_axes_name

                    if concat_node == None:
                        concat_node, _ = get_next_node_by_output(model, node.output[0])
                        del concat_node.attribute[:]

                        attr = onnx.helper.make_attribute('axis', 2)
                        concat_node.attribute.append(attr)

                        concat_output_shape_old = values.get_tensor_shape_by_name(model, concat_node.output[0])
                        concat_output_shape = [concat_output_shape_old[0], concat_output_shape_old[3], concat_output_shape_old[1], concat_output_shape_old[2]]
                        update_tensor_shape(model, concat_node.output[0], concat_output_shape)

            all_next_node, _ = get_all_next_node_by_output(model, concat_node.output[0])
            if len(all_next_node) == 2:
                concat_node = None
                const_axes_name = ''
                for node in all_next_node:
                    if node.op_type == 'Slice':
                        slice_output_shape_old = values.get_tensor_shape_by_name(model, node.output[0])
                        slice_output_shape = [slice_output_shape_old[0], slice_output_shape_old[3], slice_output_shape_old[1], slice_output_shape_old[2]]
                        update_tensor_shape(model, node.output[0], slice_output_shape)

                        if const_axes_name == '':
                            const_axes_name = node.name + '_' + node.input[3] + '_axes_'
                
                            const_axes_tensor = onnx.helper.make_tensor(name=const_axes_name,
                                                data_type=onnx.TensorProto.INT64,
                                                dims=[1],
                                                vals=[3])

                            model.graph.initializer.append(const_axes_tensor)

                        operation.remove_initializer_if_necessary_by_name(model, node.input[3], node)

                        node.input[3] = const_axes_name

                        if concat_node == None:
                            concat_node, _ = get_next_node_by_output(model, node.output[0])

                            del concat_node.attribute[:]

                            attr = onnx.helper.make_attribute('axis', 3)
                            concat_node.attribute.append(attr)

                            concat_output_shape_old = values.get_tensor_shape_by_name(model, concat_node.output[0])
                            concat_output_shape = [concat_output_shape_old[0], concat_output_shape_old[3], concat_output_shape_old[1], concat_output_shape_old[2]]
                            update_tensor_shape(model, concat_node.output[0], concat_output_shape)

                            reshape_node, _ = get_next_node_by_output(model, concat_node.output[0])
                            reshape_output_shape_old = values.get_tensor_shape_by_name(model, reshape_node.output[0])
                            reshape_output_shape = [reshape_output_shape_old[0], reshape_output_shape_old[2], reshape_output_shape_old[1]]  

                            ###
                            const_shape_name = reshape_node.name + '_' + reshape_node.input[1] + '_reshape_'
        
                            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                                data_type=onnx.TensorProto.INT64,
                                                dims=[len(reshape_output_shape)],
                                                vals=reshape_output_shape)

                            model.graph.initializer.append(const_shape_tensor)

                            update_tensor_shape(model, reshape_node.output[0], reshape_output_shape)

                            operation.remove_initializer_if_necessary_by_name(model, reshape_node.input[1], reshape_node)

                            reshape_node.input[1] = const_shape_name

                            next_add_node, _ = get_next_node_by_output(model, reshape_node.output[0])
                            add_output_shape_old = values.get_tensor_shape_by_name(model, next_add_node.output[0])
                            add_output_shape = [add_output_shape_old[0], add_output_shape_old[2], add_output_shape_old[1]]

                            update_tensor_shape(model, next_add_node.output[0], add_output_shape)


def handle_add_combination_pattern_eight(model):
    am_list = get_add_combination_pattern_eight(model)

    logger.debug('handle_add_combination_pattern_eight------------')

    for idx, am in enumerate(am_list):
        add_node = am['currentAdd']
        sub_node = am['Sub']
        rm_node = am['ReduceMean']
        next_add = am['Add']

        logger.debug('get_add_combination_pattern_eight, add_node: {}'.format(add_node.name))

        if idx != 0:
            ###add transpose
            ts_name = add_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
            ts_output_shape = [add_output_shape[0], add_output_shape[1], add_output_shape[2]]
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            
            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[add_node.output[0]],
                                                outputs=[ts_output_name],
                                                perm=[0,2,1])

            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node, add_node)

            sub_node.input[0] = ts_output_name
            rm_node.input[0] = ts_output_name
            #next_add.input[0] = ts_output_name
        else:
            ###add transpose
            ts_name = add_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
            ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            
            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[add_node.output[0]],
                                                outputs=[ts_output_name],
                                                perm=[0,2,1])

            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node, add_node)

            #sub_node.input[0] = ts_output_name
            #rm_node.input[0] = ts_output_name

            ###add reshape-1
            rs_name = add_node.output[0] + '_reshape_1_'
            rs_output_name = rs_name + '_output_'
            rs_output_shape = [ts_output_shape[1], ts_output_shape[2]]

            rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

            const_shape_name = add_node.output[0] + '_reshape_data_1'
            
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs_output_shape)],
                                vals=rs_output_shape)

            model.graph.initializer.append(const_shape_tensor)

            rs_node = onnx.helper.make_node(
                                                'Reshape',
                                                name=rs_name,
                                                inputs=[ts_output_name, const_shape_name],
                                                outputs=[rs_output_name])

            model.graph.value_info.append(rs_output)

            insert_node(model, rs_node, ts_node)

            ###add reshape-2
            rs_name = add_node.output[0] + '_reshape_2_'
            rs_output_name2 = rs_name + '_output_'
            rs_output_shape = [ts_output_shape[0], ts_output_shape[1], ts_output_shape[2]]

            rs_output = onnx.helper.make_tensor_value_info(rs_output_name2, onnx.TensorProto.FLOAT, rs_output_shape)

            const_shape_name = add_node.output[0] + '_reshape_data_2'
            
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs_output_shape)],
                                vals=rs_output_shape)

            model.graph.initializer.append(const_shape_tensor)

            rs_node2 = onnx.helper.make_node(
                                                'Reshape',
                                                name=rs_name,
                                                inputs=[rs_output_name, const_shape_name],
                                                outputs=[rs_output_name2])

            model.graph.value_info.append(rs_output)

            insert_node(model, rs_node2, rs_node)

            next_add.input[0] = rs_output_name2

            ###add transpose
            ts_name = add_node.name + '_transpose_2_'
            ts_output_name = ts_name + '_output_'
            ts_output_shape = [rs_output_shape[0], rs_output_shape[2], rs_output_shape[1]]
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            
            ts_node2 = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[rs_output_name2],
                                                outputs=[ts_output_name],
                                                perm=[0,2,1])

            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node2, add_node)

            sub_node.input[0] = ts_output_name
            rm_node.input[0] = ts_output_name




def handle_split_block_pattern_seven(model):
    node_list = get_matmul_input_path_pattern_seven(model)

    for node_block in node_list:
        matmul_node = node_block['FirstMatMul']
        split_node = node_block['Split']
        add_node = node_block['Add']
        reshape_node, _ = get_next_node_by_output(model, add_node.output[0])
        transpose_node, _ = get_next_node_by_output(model, reshape_node.output[0])

        split_node.input[0] = matmul_node.input[0]

        split_node.op_type = 'Transpose'

        del split_node.attribute[:]

        attr = onnx.helper.make_attribute('perm', [0,2,1])
        split_node.attribute.append(attr)

        matmul_input_shape = values.get_tensor_shape_by_name(model, matmul_node.input[0])
        ts_output_shape = [matmul_input_shape[0], matmul_input_shape[2], matmul_input_shape[1]]

        update_tensor_shape(model, split_node.output[0], ts_output_shape)

        del split_node.output[1:]

        ###add reshape-1
        rs_name = matmul_node.output[0] + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = matmul_node.output[0] + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[split_node.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, split_node)

        ####
        inputB, shapeB = values.get_init_value_and_shape(model, matmul_node.input[1])

        v = inputB
        old_dims = [shapeB[0], shapeB[1]]
        dims_ = [shapeB[1], shapeB[0], 1, 1]

        operation.remove_initializer_if_necessary_by_name(model, matmul_node.input[1], matmul_node)
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            #A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            #A = A.flatten()

        addA, shapeA = values.get_init_value_and_shape(model, add_node.input[0])
        if isinstance(addA, np.ndarray) == True:
            logger.debug('+++A.shape: {}'.format(A.shape))
            #add_A = addA.flatten()
        else:    
            A = np.array(addA)
            logger.debug('---A.shape: {}'.format(A.shape))
            #add_A = addA.flatten()

        #### path V
        squeeze_v = node_block['Squeeze_V']

        #Squeeze--->Conv
        squeeze_v.input[0] = rs_output_name
        squeeze_v.op_type = 'Conv'
        logger.debug('-----reuse Squeeze to Conv')
        const_w_name = squeeze_v.name + '_to_conv_w_'
        const_b_name = squeeze_v.name + '_to_conv_b_'

        index = int(2*dims_[0]/3)
        weight = A[index:,:,:,:]

        print('weight.shape:', weight.shape)

        weight = weight.flatten()
        weight = weight.tolist()

        print('weight.len:', len(weight))

        const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                            vals=weight)

        index = int(2*shapeA[0]/3)
        bias = addA[index:]
        bias = bias.flatten()
        bias = bias.tolist()

        const_b_tensor = onnx.helper.make_tensor(name=const_b_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[int(shapeA[0]/3)],
                            vals=bias)

        model.graph.initializer.append(const_w_tensor)
        model.graph.initializer.append(const_b_tensor)

        #squeeze_v.input[1] = const_w_name
        squeeze_v.input.append(const_w_name) 


        del squeeze_v.attribute[:]

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        squeeze_v.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        squeeze_v.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        squeeze_v.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        squeeze_v.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        squeeze_v.attribute.append(attr)        

        squeeze_v.input.append(const_b_name)   

        conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

        squeeze_v_output_shape = values.get_tensor_shape_by_name(model, squeeze_v.output[0])

        update_tensor_shape(model, squeeze_v.output[0], conv_output_shape)

        ###add reshape-2
        rs_name = squeeze_v.output[0] + '_reshape_2_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [squeeze_v_output_shape[0], squeeze_v_output_shape[1], int(conv_output_shape[1]/squeeze_v_output_shape[1]),conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = squeeze_v.output[0] + '_reshape_data2_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[squeeze_v.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, squeeze_v)

        matmul_v = node_block['MatMul_V']

        matmul_v.input[1] = rs_output_name

        operation.remove_onnx_node(model, transpose_node)
        operation.remove_onnx_node(model, reshape_node)
        operation.remove_onnx_node(model, add_node)
        operation.remove_onnx_node(model, matmul_node) 

        ########################
        ##########################
        ###add reshape-3
        rs_name = matmul_node.output[0] + '_reshape_3_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = matmul_node.output[0] + '_reshape_data3_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[split_node.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, split_node)

        #### path Q
        squeeze_q = node_block['Squeeze_Q']

        #Squeeze--->Conv
        squeeze_q.input[0] = rs_output_name
        squeeze_q.op_type = 'Conv'
        logger.debug('-----reuse Squeeze to Conv')
        const_w_name = squeeze_q.name + '_to_conv_w_'
        const_b_name = squeeze_q.name + '_to_conv_b_'

        index_end = int(dims_[0]/3)
        weight = A[0:index_end,:,:,:]

        print('weight.shape:', weight.shape)

        weight = weight.flatten()
        weight = weight.tolist()

        print('weight.len:', len(weight))

        const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                            vals=weight)

        index_end = int(shapeA[0]/3)
        bias = addA[0:index_end]
        bias = bias.flatten()
        bias = bias.tolist()

        const_b_tensor = onnx.helper.make_tensor(name=const_b_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[int(shapeA[0]/3)],
                            vals=bias)

        model.graph.initializer.append(const_w_tensor)
        model.graph.initializer.append(const_b_tensor)

        #squeeze_v.input[1] = const_w_name
        squeeze_q.input.append(const_w_name) 

        del squeeze_q.attribute[:]

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        squeeze_q.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        squeeze_q.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        squeeze_q.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        squeeze_q.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        squeeze_q.attribute.append(attr)        

        squeeze_q.input.append(const_b_name)   

        conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

        squeeze_q_output_shape = values.get_tensor_shape_by_name(model, squeeze_q.output[0])

        update_tensor_shape(model, squeeze_q.output[0], conv_output_shape)

        ###add reshape-2
        rs_name = squeeze_q.output[0] + '_reshape_2_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [squeeze_q_output_shape[0], squeeze_q_output_shape[1], int(conv_output_shape[1]/squeeze_q_output_shape[1]),conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = squeeze_q.output[0] + '_reshape_data2_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[squeeze_q.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, squeeze_q)

        matmul_qk = node_block['MatMul_QK']

        matmul_qk.input[1] = rs_output_name

        ###################################
        ########################################
        ###add reshape-4
        rs_name = matmul_node.output[0] + '_reshape_4_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = matmul_node.output[0] + '_reshape_data4_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[split_node.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, split_node)

        #### path K
        squeeze_k = node_block['Squeeze_K']

        #Squeeze--->Conv
        squeeze_k.input[0] = rs_output_name
        squeeze_k.op_type = 'Conv'
        logger.debug('-----reuse Squeeze to Conv')
        const_w_name = squeeze_k.name + '_to_conv_w_'
        const_b_name = squeeze_k.name + '_to_conv_b_'

        index_begin = int(dims_[0]/3)
        index_end = int(2*dims_[0]/3)
        weight = A[index_begin:index_end,:,:,:]

        print('weight.shape:', weight.shape)

        weight = weight.flatten()
        weight = weight.tolist()

        print('weight.len:', len(weight))

        const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                            vals=weight)

        index_begin = int(shapeA[0]/3)
        index_end = int(2*shapeA[0]/3)
        bias = addA[index_begin:index_end]
        bias = bias.flatten()
        bias = bias.tolist()

        const_b_tensor = onnx.helper.make_tensor(name=const_b_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=[int(shapeA[0]/3)],
                            vals=bias)

        model.graph.initializer.append(const_w_tensor)
        model.graph.initializer.append(const_b_tensor)

        #squeeze_v.input[1] = const_w_name
        squeeze_k.input.append(const_w_name) 

        del squeeze_k.attribute[:]

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        squeeze_k.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        squeeze_k.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        squeeze_k.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        squeeze_k.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        squeeze_k.attribute.append(attr)        

        squeeze_k.input.append(const_b_name)   

        conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

        squeeze_k_output_shape = values.get_tensor_shape_by_name(model, squeeze_k.output[0])

        update_tensor_shape(model, squeeze_k.output[0], conv_output_shape)

        ###add reshape-2
        rs_name = squeeze_k.output[0] + '_reshape_2_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [squeeze_k_output_shape[0], squeeze_k_output_shape[1], int(conv_output_shape[1]/squeeze_k_output_shape[1]),conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = squeeze_k.output[0] + '_reshape_data2_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[squeeze_k.output[0], const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, squeeze_k)

        transpose_node = node_block['Transpose']
        transpose_node.input[0] = rs_output_name

        ts_output_shape = values.get_tensor_shape_by_name(model, transpose_node.output[0])
        update_tensor_shape(model, transpose_node.output[0], [ts_output_shape[0], ts_output_shape[1], ts_output_shape[3], ts_output_shape[2]])

        matmul_qk = node_block['MatMul_QK']
        matmul_qk.input[0] = transpose_node.output[0]

        ##################
        mul_node = node_block['Mul']
        ts_name = mul_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'

        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[mul_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,1,3,2])

        mul_output_shape = values.get_tensor_shape_by_name(model, mul_node.output[0])
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, mul_output_shape)
        model.graph.value_info.append(transpose_output)

        insert_node(model, ts_node, mul_node)
        ########################

        softmax_node = node_block['Softmax']
        softmax_node.input[0] = ts_output_name

        ts2_name = softmax_node.name + '_transpose_'
        ts2_output_name = ts2_name + '_output_'

        ts2_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts2_name,
                                            inputs=[softmax_node.output[0]],
                                            outputs=[ts2_output_name],
                                            perm=[0,1,3,2])

        transpose_output2 = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, mul_output_shape)
        model.graph.value_info.append(transpose_output2)

        insert_node(model, ts2_node, softmax_node)

        matmul_v = node_block['MatMul_V']
        matmul_v.input[0] = matmul_v.input[1]
        matmul_v.input[1] = ts2_output_name

        matmul_v_output_shape = values.get_tensor_shape_by_name(model, matmul_v.output[0])
        update_tensor_shape(model, matmul_v.output[0], [matmul_v_output_shape[0], matmul_v_output_shape[1], matmul_v_output_shape[3], matmul_v_output_shape[2]])

        handle_trma(model, matmul_v)

def handle_split_block_pattern_eight(model, node_block_list):
    for block in node_block_list:
        ts_node = block['Transpose']
        rs_node_0, ok = get_prev_node_by_input(model, ts_node.input[0])
        if ok == 0 and rs_node_0.op_type == 'Reshape':
            add_node_0, ok = get_prev_node_by_input(model, rs_node_0.input[0])
            if ok == 0 and add_node_0.op_type == 'Add':
                matmul_node_0, ok = get_prev_node_by_input(model, add_node_0.input[1])
                if ok == 0 and matmul_node_0.op_type == 'MatMul':
                    del ts_node.attribute[:]
                    attr = onnx.helper.make_attribute('perm', [0,2,1])
                    ts_node.attribute.append(attr)

                    matmul_input_shape = values.get_tensor_shape_by_name(model, matmul_node_0.input[0])
                    ts_output_shape = [matmul_input_shape[0], matmul_input_shape[2], matmul_input_shape[1]]
                    update_tensor_shape(model, ts_node.output[0], ts_output_shape)
                    
                    ts_node.input[0] = matmul_node_0.input[0]

                    ###add reshape-1
                    rs_name = matmul_node_0.output[0] + '_reshape_1_'
                    rs_output_name = rs_name + '_output_'
                    rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

                    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                    const_shape_name = matmul_node_0.output[0] + '_reshape_data_1_'
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    rs_node = onnx.helper.make_node(
                                                        'Reshape',
                                                        name=rs_name,
                                                        inputs=[ts_node.output[0], const_shape_name],
                                                        outputs=[rs_output_name])

                    model.graph.value_info.append(rs_output)

                    insert_node(model, rs_node, ts_node)

                    ####
                    inputB, shapeB = values.get_init_value_and_shape(model, matmul_node_0.input[1])

                    v = inputB
                    old_dims = [shapeB[0], shapeB[1]]
                    dims_ = [shapeB[1], shapeB[0], 1, 1]

                    operation.remove_initializer_if_necessary_by_name(model, matmul_node_0.input[1], matmul_node_0)
                    
                    if isinstance(v, np.ndarray) == True:
                        A = v.reshape(*old_dims)
                        A = A.transpose()
                        A = A.reshape(*dims_)
                        logger.debug('+++A.shape: {}'.format(A.shape))
                        #A = A.flatten()
                    else:    
                        A = np.array(v).reshape(*old_dims)
                        A = A.transpose()
                        A = A.reshape(*dims_)
                        logger.debug('---A.shape: {}'.format(A.shape))
                        #A = A.flatten()

                    addA, shapeA = values.get_init_value_and_shape(model, add_node_0.input[0])
                    if isinstance(addA, np.ndarray) == True:
                        logger.debug('+++A.shape: {}'.format(A.shape))
                        #add_A = addA.flatten()
                    else:    
                        A = np.array(addA)
                        logger.debug('---A.shape: {}'.format(A.shape))
                        #add_A = addA.flatten()

                    #### path V
                    gather_v = block['gather_v']
                    #squeeze_v = node_block['Squeeze_V']

                    #Squeeze--->Conv
                    gather_v.input[0] = rs_output_name
                    gather_v.op_type = 'Conv'
                    logger.debug('-----reuse Squeeze to Conv')
                    const_w_name = gather_v.name + '_to_conv_w_'
                    const_b_name = gather_v.name + '_to_conv_b_'

                    operation.remove_initializer_if_necessary_by_name(model, gather_v.input[1], gather_v)

                    index = int(2*dims_[0]/3)
                    weight = A[index:,:,:,:]

                    print('weight.shape:', weight.shape)

                    weight = weight.flatten()
                    weight = weight.tolist()

                    print('weight.len:', len(weight))

                    const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                                        vals=weight)

                    index = int(2*shapeA[0]/3)
                    bias = addA[index:]
                    bias = bias.flatten()
                    bias = bias.tolist()

                    const_b_tensor = onnx.helper.make_tensor(name=const_b_name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[int(shapeA[0]/3)],
                                        vals=bias)

                    model.graph.initializer.append(const_w_tensor)
                    model.graph.initializer.append(const_b_tensor)

                    gather_v.input[1] = const_w_name
                    #gather_v.input.append(const_w_name) 

                    del gather_v.attribute[:]

                    attr = onnx.helper.make_attribute('dilations', [1, 1])
                    gather_v.attribute.append(attr)

                    attr = onnx.helper.make_attribute('group', 1)
                    gather_v.attribute.append(attr)

                    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
                    gather_v.attribute.append(attr)

                    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                    gather_v.attribute.append(attr)

                    attr = onnx.helper.make_attribute('strides', [1,1])
                    gather_v.attribute.append(attr)        

                    gather_v.input.append(const_b_name)   

                    conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

                    gather_v_output_shape = values.get_tensor_shape_by_name(model, gather_v.output[0])

                    update_tensor_shape(model, gather_v.output[0], conv_output_shape)

                    ###add reshape-2
                    rs_name = gather_v.output[0] + '_reshape_2_'
                    rs_output_name = rs_name + '_output_'
                    rs_output_shape = [gather_v_output_shape[0], gather_v_output_shape[1], int(conv_output_shape[1]/gather_v_output_shape[1]),conv_output_shape[3]]

                    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                    const_shape_name = gather_v.output[0] + '_reshape_data2_'
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    rs_node = onnx.helper.make_node(
                                                        'Reshape',
                                                        name=rs_name,
                                                        inputs=[gather_v.output[0], const_shape_name],
                                                        outputs=[rs_output_name])

                    model.graph.value_info.append(rs_output)

                    insert_node(model, rs_node, gather_v)

                    matmul_qk_v = block['matmul_qk_v']

                    matmul_qk_v.input[1] = rs_output_name

                    operation.remove_onnx_node(model, matmul_node_0)
                    operation.remove_onnx_node(model, add_node_0)
                    operation.remove_onnx_node(model, rs_node_0) 

                    ###add reshape-3
                    rs_name = matmul_node_0.output[0] + '_reshape_3_'
                    rs_output_name = rs_name + '_output_'
                    rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

                    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                    const_shape_name = matmul_node_0.output[0] + '_reshape_data3_'
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    rs_node = onnx.helper.make_node(
                                                        'Reshape',
                                                        name=rs_name,
                                                        inputs=[ts_node.output[0], const_shape_name],
                                                        outputs=[rs_output_name])

                    model.graph.value_info.append(rs_output)

                    insert_node(model, rs_node, ts_node)

                    #### path Q
                    gather_q = block['gather_q']

                    #Gather--->Conv
                    gather_q.input[0] = rs_output_name
                    gather_q.op_type = 'Conv'
                    logger.debug('-----reuse Squeeze to Conv')
                    const_w_name = gather_q.name + '_to_conv_w_'
                    const_b_name = gather_q.name + '_to_conv_b_'

                    operation.remove_initializer_if_necessary_by_name(model, gather_q.input[1], gather_q)

                    index_end = int(dims_[0]/3)
                    weight = A[:index_end,:,:,:]

                    print('weight.shape:', weight.shape)

                    weight = weight.flatten()
                    weight = weight.tolist()

                    print('weight.len:', len(weight))

                    const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                                        vals=weight)

                    index_end = int(shapeA[0]/3)
                    bias = addA[:index_end]
                    bias = bias.flatten()
                    bias = bias.tolist()

                    const_b_tensor = onnx.helper.make_tensor(name=const_b_name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[int(shapeA[0]/3)],
                                        vals=bias)

                    model.graph.initializer.append(const_w_tensor)
                    model.graph.initializer.append(const_b_tensor)

                    gather_q.input[1] = const_w_name

                    del gather_q.attribute[:]

                    attr = onnx.helper.make_attribute('dilations', [1, 1])
                    gather_q.attribute.append(attr)

                    attr = onnx.helper.make_attribute('group', 1)
                    gather_q.attribute.append(attr)

                    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
                    gather_q.attribute.append(attr)

                    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                    gather_q.attribute.append(attr)

                    attr = onnx.helper.make_attribute('strides', [1,1])
                    gather_q.attribute.append(attr)        

                    gather_q.input.append(const_b_name)   

                    conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

                    gather_q_output_shape = values.get_tensor_shape_by_name(model, gather_q.output[0])

                    update_tensor_shape(model, gather_q.output[0], conv_output_shape)

                    ###add reshape-2
                    rs_name = gather_q.output[0] + '_reshape_2_'
                    rs_output_name = rs_name + '_output_'
                    rs_output_shape = [gather_q_output_shape[0], gather_q_output_shape[1], int(conv_output_shape[1]/gather_q_output_shape[1]),conv_output_shape[3]]

                    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                    const_shape_name = gather_q.output[0] + '_reshape_data2_'
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    rs_node = onnx.helper.make_node(
                                                        'Reshape',
                                                        name=rs_name,
                                                        inputs=[gather_q.output[0], const_shape_name],
                                                        outputs=[rs_output_name])

                    model.graph.value_info.append(rs_output)

                    insert_node(model, rs_node, gather_q)

                    mul_node = block['Mul']
                    mul_node.input[0] = rs_output_name

                    update_tensor_shape(model, mul_node.output[0], rs_output_shape)

                    ###add reshape-4
                    rs_name = matmul_node_0.output[0] + '_reshape_4_'
                    rs_output_name = rs_name + '_output_'
                    rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

                    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                    const_shape_name = matmul_node_0.output[0] + '_reshape_data4_'
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    rs_node = onnx.helper.make_node(
                                                        'Reshape',
                                                        name=rs_name,
                                                        inputs=[ts_node.output[0], const_shape_name],
                                                        outputs=[rs_output_name])

                    model.graph.value_info.append(rs_output)

                    insert_node(model, rs_node, ts_node)
                    #### path K
                    gather_k = block['gather_k']

                    #Gather--->Conv
                    gather_k.input[0] = rs_output_name
                    gather_k.op_type = 'Conv'
                    logger.debug('-----reuse Squeeze to Conv')
                    const_w_name = gather_k.name + '_to_conv_w_'
                    const_b_name = gather_k.name + '_to_conv_b_'

                    operation.remove_initializer_if_necessary_by_name(model, gather_k.input[1], gather_k)

                    index_begin = int(dims_[0]/3)
                    index_end = int(2*dims_[0]/3)
                    weight = A[index_begin:index_end,:,:,:]

                    print('weight.shape:', weight.shape)

                    weight = weight.flatten()
                    weight = weight.tolist()

                    print('weight.len:', len(weight))

                    const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                                        vals=weight)

                    index_begin = int(shapeA[0]/3)
                    index_end = int(2*shapeA[0]/3)
                    bias = addA[index_begin:index_end]
                    bias = bias.flatten()
                    bias = bias.tolist()

                    const_b_tensor = onnx.helper.make_tensor(name=const_b_name,
                                        data_type=onnx.TensorProto.FLOAT,
                                        dims=[int(shapeA[0]/3)],
                                        vals=bias)

                    model.graph.initializer.append(const_w_tensor)
                    model.graph.initializer.append(const_b_tensor)

                    gather_k.input[1] = const_w_name

                    del gather_k.attribute[:]

                    attr = onnx.helper.make_attribute('dilations', [1, 1])
                    gather_k.attribute.append(attr)

                    attr = onnx.helper.make_attribute('group', 1)
                    gather_k.attribute.append(attr)

                    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
                    gather_k.attribute.append(attr)

                    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                    gather_k.attribute.append(attr)

                    attr = onnx.helper.make_attribute('strides', [1,1])
                    gather_k.attribute.append(attr)        

                    gather_k.input.append(const_b_name)   

                    conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

                    gather_k_output_shape = values.get_tensor_shape_by_name(model, gather_k.output[0])

                    update_tensor_shape(model, gather_k.output[0], conv_output_shape)

                    ###add reshape-2
                    rs_name = gather_k.output[0] + '_reshape_2_'
                    rs_output_name = rs_name + '_output_'
                    rs_output_shape = [gather_k_output_shape[0], gather_k_output_shape[1], int(conv_output_shape[1]/gather_k_output_shape[1]),conv_output_shape[3]]

                    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                    const_shape_name = gather_k.output[0] + '_reshape_data2_'
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    rs_node = onnx.helper.make_node(
                                                        'Reshape',
                                                        name=rs_name,
                                                        inputs=[gather_k.output[0], const_shape_name],
                                                        outputs=[rs_output_name])

                    model.graph.value_info.append(rs_output)

                    insert_node(model, rs_node, gather_k)

                    
                    transpose_node = block['TransposeK']
                    transpose_node.input[0] = rs_output_name
                    ts_output_shape = values.get_tensor_shape_by_name(model, transpose_node.output[0])
                    print('~~~ts_output_shape:', ts_output_shape)
                    update_tensor_shape(model, transpose_node.output[0], [ts_output_shape[0], ts_output_shape[1], ts_output_shape[3], ts_output_shape[2]])
                    

                    matmul_qk = block['matmul_qk']
                    matmul_qk.input[1] = matmul_qk.input[0]
                    matmul_qk.input[0] = transpose_node.output[0]

                    ##################
                    ts_name = matmul_qk.name + '_transpose_'
                    ts_output_name = ts_name + '_output_'

                    ts_node = onnx.helper.make_node(
                                                        'Transpose',
                                                        name=ts_name,
                                                        inputs=[matmul_qk.output[0]],
                                                        outputs=[ts_output_name],
                                                        perm=[0,1,3,2])

                    mul_output_shape = values.get_tensor_shape_by_name(model, matmul_qk.output[0])
                    transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, mul_output_shape)
                    model.graph.value_info.append(transpose_output)

                    insert_node(model, ts_node, matmul_qk)

                    add_node = block['Add']
                    add_node.input[0] = ts_output_name
                    ########################

                    softmax_node = block['Softmax']
                    #softmax_node.input[0] = ts_output_name

                    ts2_name = softmax_node.name + '_transpose_'
                    ts2_output_name = ts2_name + '_output_'

                    ts2_node = onnx.helper.make_node(
                                                        'Transpose',
                                                        name=ts2_name,
                                                        inputs=[softmax_node.output[0]],
                                                        outputs=[ts2_output_name],
                                                        perm=[0,1,3,2])

                    transpose_output2 = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, mul_output_shape)
                    model.graph.value_info.append(transpose_output2)

                    insert_node(model, ts2_node, softmax_node)

                    matmul_qk_v.input[0] = matmul_qk_v.input[1]
                    matmul_qk_v.input[1] = ts2_output_name

                    matmul_qk_v_output_shape = values.get_tensor_shape_by_name(model, matmul_qk_v.output[0])
                    update_tensor_shape(model, matmul_qk_v.output[0], [matmul_qk_v_output_shape[0], matmul_qk_v_output_shape[1], matmul_qk_v_output_shape[3], matmul_qk_v_output_shape[2]])

                    handle_trma_pattern_eight(model, matmul_qk_v)



def handle_add_combination_pattern_two_three(model):
    am_list = get_add_combination_pattern_two(model)

    logger.debug('handle_add_combination_pattern_two_three------------')

    #if len(am_list):
    for am in am_list:
        #am = am_list[0]
        add_node = am['currentAdd']
        next_add_node = am['nextAdd']
        matmul_node_list = am['MatMulList']

        nextAddInput1 = am['nextAddInput1'] 

        logger.debug('handle_add_combination_pattern_two_three, add_node: {}, next_add_node: {}'.format(add_node.name, next_add_node.name))
        ###add transpose
        ts_name = add_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[add_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs_name = add_node.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[1], ts_output_shape[2]] #TBD

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add_node.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        ###add reshape-2
        rs2_name = add_node.name + '_reshape_2_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        rs2_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape2_name = add_node.name + '_reshape2_data_'
        
        const_shape2_tensor = onnx.helper.make_tensor(name=const_shape2_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape2_tensor)


        ######################################################
        ######################################################
        rs2_node = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs2_name,
                                        inputs=[rs_output_name, const_shape2_name],
                                        outputs=[rs2_output_name])

        model.graph.value_info.append(rs2_output)

        insert_node(model, rs2_node, next_add_node)

        if nextAddInput1 == True:
            next_add_node.input[1] = rs2_output_name
        else:    
            next_add_node.input[0] = rs2_output_name

        matmul_node_list[0].input[0] = rs2_output_name
        matmul_node_list[1].input[0] = rs2_output_name
        matmul_node_list[2].input[0] = rs2_output_name

        insert_node(model, rs_node, rs2_node)

        insert_node(model, ts_node, rs_node)

        ###################################################################
        ###########insert Transpose before ReduceMean and Sub
        sub_node = am['Sub']
        rm_node = am['ReduceMean']

        ts_name = sub_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'

        add_output_shape = values.get_tensor_shape_by_name(model, next_add_node.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[1], add_output_shape[2]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[next_add_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        insert_node(model, ts_node, sub_node)

        sub_node.input[0] = ts_output_name
        rm_node.input[0] = ts_output_name

def handle_add_combination_pattern_four(model):
    am_list = get_add_combination_pattern_two(model)
    am_list2 = get_add_combination_pattern_four(model)
    f = False
    if len(am_list2) > 0:
        am_list.append(am_list2[0])
        f = True

    logger.debug('handle_add_combination_pattern_four------------')

    length = len(am_list)

    #shape_dim = 3

    #if len(am_list):
    for idx, am in enumerate(am_list):
        #am = am_list[0]
        if idx != length - 1:
            add_node = am['currentAdd']
            next_add_node = am['nextAdd']
            matmul_node_list = am['MatMulList'] 

            ###add transpose
            ts_name = add_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
            shape_dim = len(add_output_shape)
            perm_ = [0,2,1]
            if shape_dim == 3:
                ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
            else:
                perm_ = [1,0]
                ts_output_shape = [add_output_shape[1], add_output_shape[0]]

            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
            
            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[add_node.output[0]],
                                                outputs=[ts_output_name],
                                                perm=perm_)

            model.graph.value_info.append(transpose_output)

            if shape_dim == 3:
                ###add reshape-1
                rs_name = add_node.name + '_reshape_1_'
                rs_output_name = rs_name + '_output_'
                rs_output_shape = [ts_output_shape[1], ts_output_shape[2]] #TBD

                rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

                const_shape_name = add_node.name + '_reshape_data_'
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(rs_output_shape)],
                                    vals=rs_output_shape)

                model.graph.initializer.append(const_shape_tensor)

                rs_node = onnx.helper.make_node(
                                                    'Reshape',
                                                    name=rs_name,
                                                    inputs=[ts_output_name, const_shape_name],
                                                    outputs=[rs_output_name])

                model.graph.value_info.append(rs_output)

            ###add reshape-2
            rs2_name = add_node.name + '_reshape_2_'
            rs2_output_name = rs2_name + '_output_'
            const_shape2_name = add_node.name + '_reshape2_data_'

            if shape_dim == 3:
                inputs_ = [rs_output_name, const_shape2_name]
                rs2_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
            else:
                inputs_ = [ts_output_name, const_shape2_name]
                rs2_output_shape = [1, add_output_shape[1], add_output_shape[0]] 
            
            rs2_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

            const_shape2_tensor = onnx.helper.make_tensor(name=const_shape2_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs2_output_shape)],
                                vals=rs2_output_shape)

            model.graph.initializer.append(const_shape2_tensor)

            ######################################################
            ######################################################
            rs2_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=inputs_,
                                            outputs=[rs2_output_name])

            model.graph.value_info.append(rs2_output)

            insert_node(model, rs2_node, next_add_node)

            if shape_dim == 3:
                next_add_node.input[0] = rs2_output_name
            else:
                next_add_node.input[1] = rs2_output_name    

            matmul_node_list[0].input[0] = rs2_output_name
            matmul_node_list[1].input[0] = rs2_output_name
            matmul_node_list[2].input[0] = rs2_output_name

            if shape_dim == 3:
                insert_node(model, rs_node, rs2_node)
                insert_node(model, ts_node, rs_node)
            else:
                insert_node(model, ts_node, rs2_node)

            update_tensor_shape(model, next_add_node.output[0], rs2_output_shape)    
        else:
            next_add_node = am['nextAdd']
            rs_node = am['currentAdd']
            rs_shape = values.get_tensor_shape_by_name(model, rs_node.input[0])
            rs_node.op_type = 'Transpose'
            attr = onnx.helper.make_attribute('perm', [0,2,1])
            rs_node.attribute.append(attr)
            del rs_node.input[1:]
            update_tensor_shape(model, rs_node.output[0], [rs_shape[0], rs_shape[2], rs_shape[1]])
            update_tensor_shape(model, next_add_node.output[0], [rs_shape[0], rs_shape[2], rs_shape[1]])

        ###################################################################
        ###add reshape
        logger.debug('------shape_dim: {}'.format(shape_dim))

        if shape_dim == 2:
            rs_name_ = next_add_node.name + '_reshape_'
            rs_output_name_ = rs_name_ + '_output_'
            const_shape_name_ = next_add_node.name + '_reshape_data_'

            inputs_ = [next_add_node.output[0], const_shape_name_]
            s = values.get_tensor_shape_by_name(model, next_add_node.input[1])
            rs_output_shape_ = [s[1], s[2]]

            rs_output_ = onnx.helper.make_tensor_value_info(rs_output_name_, onnx.TensorProto.FLOAT, rs_output_shape_)

            const_shape_tensor_ = onnx.helper.make_tensor(name=const_shape_name_,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs_output_shape_)],
                                vals=rs_output_shape_)

            model.graph.initializer.append(const_shape_tensor_)

            rs_node_ = onnx.helper.make_node(
                                                'Reshape',
                                                name=rs_name_,
                                                inputs=inputs_,
                                                outputs=[rs_output_name_])

            model.graph.value_info.append(rs_output_)

        ###########insert Transpose before ReduceMean/Sub/Mul
        mul_node = None
        msr_node_list, ok = get_all_next_node_by_output(model, next_add_node.output[0])
        if ok == 0:
            for n in msr_node_list:
                if n.op_type == 'Mul':
                    mul_node = n
                    break

        sub_node = am['Sub']
        rm_node = am['ReduceMean']

        ts_name = sub_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'

        add_output_shape = values.get_tensor_shape_by_name(model, next_add_node.output[0])
        dims = shape_dim #len(add_output_shape)
        if dims == 3:
            perm_ = [0,2,1]
            inputs_ = [next_add_node.output[0]]
            ts_output_shape = [add_output_shape[0], add_output_shape[1], add_output_shape[2]]
        else:
            perm_=[1,0]
            inputs_ = [rs_output_name_]
            ts_output_shape = [add_output_shape[2], add_output_shape[1]]

        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=inputs_,
                                            outputs=[ts_output_name],
                                            perm=perm_)

        model.graph.value_info.append(transpose_output)

        insert_node(model, ts_node, sub_node)
        if shape_dim == 2:
            logger.debug('insert_node rs_node----')
            insert_node(model, rs_node_, ts_node) 

        sub_node.input[0] = ts_output_name
        rm_node.input[0] = ts_output_name

        if mul_node != None:
            mul_node.input[0] = ts_output_name

def get_matmul_block_one(model, matmul_node):
    logger.debug('into get_matmul_block_one')

    res = -1
    node_dict = {}

    #input_next, ok = get_next_node_by_output(model, input_)
    input_next = matmul_node
    if input_next.op_type == 'MatMul':
        shapeA = values.get_tensor_shape_by_name(model, input_next.input[0])
        inputB, shapeB = values.get_init_value_and_shape(model, input_next.input[1])

        if isinstance(inputB, list) and inputB == []:
            logger.debug('inputB is not in initilizer')
            inputB = values.get_constant_value(model, input_next.input[1])

        if len(shapeA) == 3 and len(shapeB) == 2:
            logger.debug('--- got MatMul node: {}'.format(input_next.name))
            #node_list = [input_next, input_pp_pre, input_p_pre, input_pre]
            #node_dict['node_list'] = node_list
            node_dict['MatMul1'] = input_next
            node_dict['matmulA1_Shape'] = shapeA
            node_dict['inputB1'] = inputB
            node_dict['inputB1_name'] = input_next.input[1]
            node_dict['matmulB1_Shape'] = shapeB

            input_nnext, ok = get_next_node_by_output(model, input_next.output[0])
            if ok == 0 and input_nnext.op_type == 'Add':
                addA_name = input_nnext.input[0]
                addA, shapeA = values.get_init_value_and_shape(model, input_nnext.input[0])
                node_dict['addFirst'] = True

                if len(shapeA) == 0:
                    addA_name = input_nnext.input[1]
                    addA, shapeA = values.get_init_value_and_shape(model, input_nnext.input[1])
                    node_dict['addFirst'] = False

                if len(shapeA) == 1:
                    node_dict['Add1'] = input_nnext
                    logger.debug('--- got Add1 node: {}'.format(input_nnext.name))

                    input_nnnext, ok = get_all_next_node_by_output(model, input_nnext.output[0])
                    if len(input_nnnext) == 2:
                        if (input_nnnext[0].op_type == 'Div' and input_nnnext[1].op_type == 'Mul') or \
                                (input_nnnext[0].op_type == 'Mul' and input_nnnext[0].op_type == 'Div'):
                            mul_node = input_nnnext[0]
                            div_node = input_nnnext[1]
                            if input_nnnext[1].op_type == 'Mul':
                                mul_node = input_nnnext[1]
                                div_node = input_nnnext[0]
                            
                            node_dict['Div'] = div_node
                            node_dict['Mul'] = mul_node

                            logger.debug('--- got Div node: {}'.format(div_node.name))

                            input_nnnnext, ok = get_next_node_by_output(model, mul_node.output[0])
                            if ok == 0 and input_nnnnext.op_type == 'Mul':
                                mulB, shapeB = values.get_init_value_and_shape(model, input_nnnnext.input[1])
                                if len(mulB) > 0:
                                    #######################
                                    ##############################
                                    logger.debug('--- got mul2 node: {}'.format(input_nnnnext.name))
                                    input_nnnnnext, ok = get_next_node_by_output(model, input_nnnnext.output[0])
                                    if ok == 0 and input_nnnnnext.op_type == 'MatMul':
                                        shapeA = values.get_tensor_shape_by_name(model, input_nnnnnext.input[0])
                                        inputB, shapeB = values.get_init_value_and_shape(model, input_nnnnnext.input[1])

                                        if isinstance(inputB, list) and inputB == []:
                                            logger.debug('inputB is not in initilizer')
                                            inputB = values.get_constant_value(model, input_nnnnnext.input[1])

                                        if len(shapeA) == 3 and len(shapeB) == 2:
                                            logger.debug('--- got MatMul2 node: {}'.format(input_nnnnnext.name))
                                            #node_list = [input_nnnnnext, input_pp_pre, input_p_pre, input_pre]
                                            #node_dict['node_list'] = node_list
                                            node_dict['MatMul2'] = input_nnnnnext
                                            node_dict['matmulA2_Shape'] = shapeA
                                            node_dict['inputB2'] = inputB
                                            node_dict['inputB2_name'] = input_nnnnnext.input[1]
                                            node_dict['matmulB2_Shape'] = shapeB

                                            input_nnnnnnext, ok = get_next_node_by_output(model, input_nnnnnext.output[0])
                                            if ok == 0 and input_nnnnnnext.op_type == 'Add':
                                                logger.debug('--- got Add2 node: {}'.format(input_nnnnnnext.name))
                                            
                                            ##########
                                            addA_name = input_nnnnnnext.input[0]
                                            addA, shapeA = values.get_init_value_and_shape(model, input_nnnnnnext.input[0])
                                            node_dict['addFirst2'] = True

                                            if len(shapeA) == 0:
                                                addA_name = input_nnnnnnext.input[1]
                                                addA, shapeA = values.get_init_value_and_shape(model, input_nnnnnnext.input[1])
                                                node_dict['addFirst2'] = False

                                            if len(shapeA) == 1:
                                                node_dict['Add2'] = input_nnnnnnext
                                                next_node, ok = get_next_node_by_output(model, input_nnnnnnext.output[0])
                                                if ok == 0 and next_node.op_type == 'Add':
                                                    logger.debug('--- got last Add node: {}'.format(next_node.name))
                                                    res = 0
                                                    node_dict['NextAdd'] = next_node
                                                    node_dict['NextAddInput1'] = False
                                                    if next_node.input[0] == input_nnnnnnext.output[0]:
                                                        node_dict['NextAddInput1'] = True

    return node_dict, res

def get_mul_add_block(model):
    logger.debug('into get_mul_add_block')

    node_list = []
    for node in model.graph.node:
        if node.op_type == 'Mul':
            #print('got mul:', node.name)

            is_init = False

            for init in model.graph.initializer:
                if init.name == node.input[0] or init.name == node.input[1]:
                    is_init = True
                    break

            if is_init == False:
                dataA = values.get_constant_value(model, node.input[0])
                if len(dataA) == 0:
                    dataA = values.get_constant_value(model, node.input[1])

                if dataA != []:
                    is_init = True

            if is_init == True:
                #print('----got mul:', node.name)
                next_node, ok = get_next_node_by_output(model, node.output[0])
                if ok == 0 and next_node.op_type == 'Add':
                    ##############
                    #print('----got add:', next_node.name)
                    is_init = False

                    for init in model.graph.initializer:
                        if init.name == next_node.input[1]:
                            is_init = True
                            break

                    if is_init == False:
                        dataA = values.get_constant_value(model, next_node.input[1])
                        if dataA != []:
                            is_init = True

                if is_init == True:
                    #print('get_all_next_node_by_output---', next_node.output, node.name)
                    next_node_list, ok = get_all_next_node_by_output(model, next_node.output[0])
                    if ok == 0:
                        #print('next_node_list:', len(next_node_list))
                        if len(next_node_list) == 2:
                            #print('got next_node_list:', next_node_list[0].op_type, next_node_list[1].op_type)

                            if (next_node_list[0].op_type == 'Add' and next_node_list[1].op_type == 'MatMul') or \
                                (next_node_list[0].op_type == 'MatMul' and next_node_list[1].op_type == 'Add'):
                                logger.debug('got it~')
                                matmul_node = next_node_list[0]
                                if next_node_list[1].op_type == 'MatMul':
                                    matmul_node = next_node_list[1]

                                node_dict, ret = get_matmul_block_one(model, matmul_node)
                                if ret == 0:
                                    #print('got node dict:', node_dict)
                                    node_dict['currentAdd'] = next_node
                                    node_list.append(node_dict)
                        elif len(next_node_list) == 1: #for telecom transform model
                            if next_node_list[0].op_type == 'MatMul':
                                logger.debug('got Add~~')

                                matmul_node = next_node_list[0]
                                node_dict, ret = get_matmul_block_one(model, matmul_node)
                                if ret == 0:
                                    #print('got node dict:', node_dict)
                                    node_dict['currentAdd'] = next_node
                                    node_list.append(node_dict)

    return node_list

def handle_mul_add_block(model, pattern):
    node_list = get_mul_add_block(model)

    #if len(node_list) > 0:
    for node_dict in node_list:
        logger.debug('++++++++++++++++++++++')
        logger.debug('Add1: {}'.format(node_dict['Add1'].name))
        logger.debug('Add2: {}'.format(node_dict['Add2'].name))
        logger.debug('++++++++++++++++++++++')

        matmul1 = node_dict['MatMul1']
        add1 = node_dict['Add1']

        matmul2 = node_dict['MatMul2']
        add2 = node_dict['Add2']

        currentAdd = node_dict['currentAdd']
        nextAdd = node_dict['NextAdd']

        nextAddInput1 = node_dict['NextAddInput1']

        div_node = node_dict['Div']

        ###add transpose
        ts_name = currentAdd.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        add_output_shape = values.get_tensor_shape_by_name(model, currentAdd.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[currentAdd.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs_name = currentAdd.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = currentAdd.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        #########################
        insert_node(model, rs_node, matmul1)
        matmul1.input[0] = rs_output_name

        insert_node(model, ts_node, rs_node)

        if pattern != 5 and pattern != 7 and pattern != 8:
            if nextAddInput1 == True:
                nextAdd.input[1] = ts_output_name
            else:    
                nextAdd.input[0] = ts_output_name

        #MatMul1--->Conv
        matmul1.op_type = 'Conv'
        logger.debug('-----reuse MatMul to Conv')
        const_x_name = matmul1.name + '_to_conv_x_'

        v = node_dict['inputB1']
        old_dims = [node_dict['matmulB1_Shape'][0], node_dict['matmulB1_Shape'][1]]
        dims_ = [node_dict['matmulB1_Shape'][1], node_dict['matmulB1_Shape'][0],1,1]

        operation.remove_initializer_if_necessary_by_name(model, node_dict['inputB1_name'], matmul1)
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul1.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul1.attribute.append(attr)        

        if node_dict['addFirst'] == True:
            matmul1.input.append(add1.input[0])
        else:
            matmul1.input.append(add1.input[1])   

        output_shape = values.get_tensor_shape_by_name(model, matmul1.output[0])
        conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]]

        if pattern == 5:
            conv_output_shape = [output_shape[1], output_shape[2], 1, output_shape[0]]

        update_tensor_shape(model, matmul1.output[0], conv_output_shape) 

        #Add1--->Reshape
        add1.op_type = 'Reshape'

        del add1.attribute[:]

        rs_name = add1.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [conv_output_shape[0], conv_output_shape[1], conv_output_shape[3]]
        logger.debug('-----rs_output_shape: {}'.format(rs_output_shape))

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add1.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        if node_dict['addFirst'] == True:
            add1.input[0] = add1.input[1]

        add1.input[1] = const_shape_name

        update_tensor_shape(model, add1.output[0], rs_output_shape)

        #################################
        #################################
        ###add reshape-1
        rs2_name = matmul2.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [rs_output_shape[0], rs_output_shape[1], 1, rs_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = matmul2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs2_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[matmul2.input[0], const_shape_name],
                                            outputs=[rs2_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs2_node, matmul2)
        matmul2.input[0] = rs2_output_name

        #MatMul2--->Conv
        matmul2.op_type = 'Conv'
        logger.debug('++++++reuse MatMul to Conv')
        const_x_name = matmul2.name + '_to_conv_x_'

        v = node_dict['inputB2']
        old_dims = [node_dict['matmulB2_Shape'][0], node_dict['matmulB2_Shape'][1]]
        dims_ = [node_dict['matmulB2_Shape'][1], node_dict['matmulB2_Shape'][0],1,1]

        operation.remove_initializer_if_necessary_by_name(model, node_dict['inputB2_name'], matmul2)
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul2.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul2.attribute.append(attr)        

        if node_dict['addFirst2'] == True:
            B = add2.input[0]
        else:
            B = add2.input[1]   

        matmul2.input.append(B)

        output_shape = values.get_tensor_shape_by_name(model, matmul2.output[0])
        conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]]

        if pattern == 5:
            conv_output_shape = [output_shape[1], output_shape[2], 1, output_shape[0]] 

        update_tensor_shape(model, matmul2.output[0], conv_output_shape) 

        #Add2--->Reshape
        add2.op_type = 'Reshape'

        del add2.attribute[:]

        rs2_name = add2.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [conv_output_shape[0], conv_output_shape[1], conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = add2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        if node_dict['addFirst2'] == True:
            add2.input[0] = add2.input[1]

        add2.input[1] = const_shape_name

        update_tensor_shape(model, add2.output[0], rs2_output_shape)

        ######update tensor shape
        div_output_shape = values.get_tensor_shape_by_name(model, div_node.output[0])
        new_shape = [div_output_shape[0], div_output_shape[2], div_output_shape[1]]
        if pattern == 5:
            new_shape = [div_output_shape[1], div_output_shape[2], div_output_shape[0]]

        update_tensor_shape(model, div_node.output[0], new_shape)

        erf_node, ok = get_next_node_by_output(model, div_node.output[0])
        if ok == 0 and erf_node.op_type == 'Erf':
            erf_output_shape = values.get_tensor_shape_by_name(model, erf_node.output[0])
            new_shape = [erf_output_shape[0], erf_output_shape[2], erf_output_shape[1]]
            if pattern == 5:
                new_shape = [erf_output_shape[1], erf_output_shape[2], erf_output_shape[0]]

            update_tensor_shape(model, erf_node.output[0], new_shape)

            add_node_internal, ok = get_next_node_by_output(model, erf_node.output[0])
            if ok == 0 and add_node_internal.op_type == 'Add':
                addi_output_shape = values.get_tensor_shape_by_name(model, add_node_internal.output[0])
                new_shape = [addi_output_shape[0], addi_output_shape[2], addi_output_shape[1]]

                if pattern == 5:
                    new_shape = [addi_output_shape[1], addi_output_shape[2], addi_output_shape[0]]

                update_tensor_shape(model, add_node_internal.output[0], new_shape)
        
            mul_node1, ok = get_next_node_by_output(model, add_node_internal.output[0])
            if ok == 0 and mul_node1.op_type == 'Mul':
                mul1_output_shape = values.get_tensor_shape_by_name(model, mul_node1.output[0])
                new_shape = [mul1_output_shape[0], mul1_output_shape[2], mul1_output_shape[1]]

                if pattern == 5:
                    new_shape = [mul1_output_shape[1], mul1_output_shape[2], mul1_output_shape[0]]

                update_tensor_shape(model, mul_node1.output[0], new_shape)

            mul_node2, ok = get_next_node_by_output(model, mul_node1.output[0])
            if ok == 0 and mul_node2.op_type == 'Mul':    
                mul2_output_shape = values.get_tensor_shape_by_name(model, mul_node2.output[0])
                new_shape = [mul2_output_shape[0], mul2_output_shape[2], mul2_output_shape[1]]

                if pattern == 5:
                    new_shape = [mul2_output_shape[1], mul2_output_shape[2], mul2_output_shape[0]]

                update_tensor_shape(model, mul_node2.output[0], new_shape)

        ######insert Transpose before ReduceMean and Sub
        if pattern == 5: #or pattern == 8:
            ###add transpose
            ts3_name = nextAdd.name + '_transpose_'
            ts3_output_name = ts3_name + '_output_'
            ts3_output_shape = [rs2_output_shape[0], rs2_output_shape[2], rs2_output_shape[1]]
            ts3_output = onnx.helper.make_tensor_value_info(ts3_output_name, onnx.TensorProto.FLOAT, ts3_output_shape)
            
            ts3_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts3_name,
                                                inputs=[add2.output[0]],
                                                outputs=[ts3_output_name],
                                                perm=[0,2,1])

            model.graph.value_info.append(ts3_output)

            insert_node(model, ts3_node, add2) 
            nextAdd.input[1] = ts3_output_name
        else: 
            update_tensor_shape(model, nextAdd.output[0], rs2_output_shape)

            rm_sub, ok = get_all_next_node_by_output(model, nextAdd.output[0])
            if ok == 0 and len(rm_sub) == 2:
                logger.debug('----got reducemean and sub node--- {}'.format(nextAdd.name))
                sub_node = rm_sub[0]
                rm_node = rm_sub[1]

                if (rm_sub[0].op_type == 'ReduceMean' and rm_sub[1].op_type == 'Sub') or \
                    (rm_sub[1].op_type == 'ReduceMean' and rm_sub[0].op_type == 'Sub'):
                    
                    if rm_sub[0].op_type == 'ReduceMean':
                        sub_node = rm_sub[1]
                        rm_node = rm_sub[0]

                    ###add transpose
                    ts3_name = nextAdd.name + '_transpose_'
                    ts3_output_name = ts3_name + '_output_'
                    add3_output_shape = values.get_tensor_shape_by_name(model, nextAdd.output[0])
                    ts3_output_shape = [add3_output_shape[0], add3_output_shape[2], add3_output_shape[1]]
                    ts3_output = onnx.helper.make_tensor_value_info(ts3_output_name, onnx.TensorProto.FLOAT, ts3_output_shape)
                    
                    ts3_node = onnx.helper.make_node(
                                                        'Transpose',
                                                        name=ts3_name,
                                                        inputs=[nextAdd.output[0]],
                                                        outputs=[ts3_output_name],
                                                        perm=[0,2,1])

                    model.graph.value_info.append(ts3_output)

                    insert_node(model, ts3_node, sub_node) 
                    sub_node.input[0] = ts3_output_name
                    rm_node.input[0] = ts3_output_name

def get_matmul_block_two(model, matmul_node):
    logger.debug('into get_matmul_block_two')

    res = -1
    node_dict = {}

    #input_next, ok = get_next_node_by_output(model, input_)
    input_next = matmul_node
    if input_next.op_type == 'MatMul':
        shapeA = values.get_tensor_shape_by_name(model, input_next.input[0])
        inputB, shapeB = values.get_init_value_and_shape(model, input_next.input[1])

        if isinstance(inputB, list) and inputB == []:
            logger.debug('inputB is not in initilizer')
            inputB = values.get_constant_value(model, input_next.input[1])

        if len(shapeA) == 3 and len(shapeB) == 2:
            logger.debug('++++ got MatMul node: {}'.format(input_next.name))
            #node_list = [input_next, input_pp_pre, input_p_pre, input_pre]
            #node_dict['node_list'] = node_list
            node_dict['MatMul1'] = input_next
            node_dict['matmulA1_Shape'] = shapeA
            node_dict['inputB1'] = inputB
            node_dict['matmulB1_Shape'] = shapeB

            input_nnext, ok = get_next_node_by_output(model, input_next.output[0])
            if ok == 0 and input_nnext.op_type == 'Add':
                addA_name = input_nnext.input[0]
                addA, shapeA = values.get_init_value_and_shape(model, input_nnext.input[0])

                if len(shapeA) == 1:
                    node_dict['Add1'] = input_nnext
                    logger.debug('++++ got Add1 node: {}'.format(input_nnext.name))

                    input_nnnext, ok = get_next_node_by_output(model, input_nnext.output[0])
                    if ok == 0 and input_nnnext.op_type == 'Relu':
                        node_dict['Relu'] = input_nnnext
                        logger.debug('++++ got Relu node: {}'.format(input_nnnext.name))

                        input_nnnnext, ok = get_next_node_by_output(model, input_nnnext.output[0])
                        if ok == 0 and input_nnnnext.op_type == 'MatMul':
                            shapeA = values.get_tensor_shape_by_name(model, input_nnnnext.input[0])
                            inputB, shapeB = values.get_init_value_and_shape(model, input_nnnnext.input[1])

                            if isinstance(inputB, list) and inputB == []:
                                logger.debug('inputB is not in initilizer')
                                inputB = values.get_constant_value(model, input_nnnnext.input[1])

                            if len(shapeA) == 3 and len(shapeB) == 2:
                                logger.debug('++++ got MatMul2 node: {}'.format(input_nnnnext.name))
                                #node_list = [input_nnnnext, input_pp_pre, input_p_pre, input_pre]
                                #node_dict['node_list'] = node_list
                                node_dict['MatMul2'] = input_nnnnext
                                node_dict['matmulA2_Shape'] = shapeA
                                node_dict['inputB2'] = inputB
                                node_dict['matmulB2_Shape'] = shapeB

                                input_nnnnnext, ok = get_next_node_by_output(model, input_nnnnext.output[0])
                                if ok == 0 and input_nnnnnext.op_type == 'Add':
                                    logger.debug('++++ got Add2 node: {}'.format(input_nnnnnext.name))
                                    #addA_name = input_nnnnnext.input[0]
                                    #if len(shapeA) == 1:
                                    node_dict['Add2'] = input_nnnnnext
                                    next_node, ok = get_next_node_by_output(model, input_nnnnnext.output[0])
                                    if ok == 0 and next_node.op_type == 'Add':
                                        logger.debug('++++ got last Add node: {}'.format(next_node.name))
                                        res = 0
                                        node_dict['NextAdd'] = next_node 

    return node_dict, res

#Mul->Add->MatMul->Add->Relu->MatMul->Add
def get_mul_add_block_two(model):
    node_list = []
    for node in model.graph.node:
        if node.op_type == 'Mul':
            #print('got mul:', node.name)

            is_init = False

            for init in model.graph.initializer:
                if init.name == node.input[1]:
                    is_init = True
                    break

            if is_init == False:
                dataA = values.get_constant_value(model, node.input[1])
                if dataA != []:
                    is_init = True

            if is_init == True:
                #print('----got mul:', node.name)
                next_node, ok = get_next_node_by_output(model, node.output[0])
                if ok == 0 and next_node.op_type == 'Add':
                    ##############
                    #print('----got add:', next_node.name)
                    is_init = False

                    for init in model.graph.initializer:
                        if init.name == next_node.input[1]:
                            is_init = True
                            break

                    if is_init == False:
                        dataA = values.get_constant_value(model, next_node.input[1])
                        if dataA != []:
                            is_init = True

                if is_init == True:
                    #print('get_all_next_node_by_output---', next_node.output, node.name)
                    matmul_node, ok = get_next_node_by_output(model, next_node.output[0])
                    if ok == 0 and matmul_node.op_type == 'MatMul':
                        logger.debug('got match MatMul~')

                        node_dict, ret = get_matmul_block_two(model, matmul_node)
                        if ret == 0:
                            #print('got node dict:', node_dict)
                            node_dict['currentAdd'] = next_node
                            node_list.append(node_dict)

    return node_list

#Mul->Add->MatMul->Add->Relu->MatMul->Add
def handle_mul_add_block_two(model):
    node_list = get_mul_add_block_two(model)

    #if len(node_list) > 0:
    for node_dict in node_list:
        logger.debug('##############################')
        logger.debug('Add1: {}'.format(node_dict['Add1'].name))
        logger.debug('Add2: {}'.format(node_dict['Add2'].name))
        logger.debug('###############################')

        matmul1 = node_dict['MatMul1']
        add1 = node_dict['Add1']

        matmul2 = node_dict['MatMul2']
        add2 = node_dict['Add2']

        currentAdd = node_dict['currentAdd']
        nextAdd = node_dict['NextAdd']

        relu_node = node_dict['Relu']

        ###add transpose
        ts_name = currentAdd.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        add_output_shape = values.get_tensor_shape_by_name(model, currentAdd.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[currentAdd.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs_name = currentAdd.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = currentAdd.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        #########################
        insert_node(model, rs_node, matmul1)
        matmul1.input[0] = rs_output_name

        insert_node(model, ts_node, rs_node)

        #nextAdd.input[0] = ts_output_name

        #MatMul1--->Conv
        matmul1.op_type = 'Conv'
        logger.debug('+++++ reuse MatMul to Conv')
        const_x_name = matmul1.name + '_to_conv_x_'

        v = node_dict['inputB1']
        old_dims = [node_dict['matmulB1_Shape'][0], node_dict['matmulB1_Shape'][1]]
        dims_ = [node_dict['matmulB1_Shape'][1], node_dict['matmulB1_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape:', A.shape)
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul1.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul1.attribute.append(attr)        

        matmul1.input.append(add1.input[0])

        output_shape = values.get_tensor_shape_by_name(model, matmul1.output[0])
        conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]] 

        update_tensor_shape(model, matmul1.output[0], conv_output_shape) 

        #Add1--->Reshape
        add1.op_type = 'Reshape'

        del add1.attribute[:]

        rs_name = add1.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [conv_output_shape[0], conv_output_shape[1], conv_output_shape[3]]
        logger.debug('-----rs_output_shape: {}'.format(rs_output_shape))

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add1.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        add1.input[0] = add1.input[1]
        add1.input[1] = const_shape_name

        update_tensor_shape(model, add1.output[0], rs_output_shape)

        update_tensor_shape(model, relu_node.output[0], rs_output_shape)

        #################################
        #################################
        ###add reshape-1
        rs2_name = matmul2.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [rs_output_shape[0], rs_output_shape[1], 1, rs_output_shape[2]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = matmul2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs2_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[matmul2.input[0], const_shape_name],
                                            outputs=[rs2_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs2_node, matmul2)
        matmul2.input[0] = rs2_output_name

        #MatMul2--->Conv
        matmul2.op_type = 'Conv'
        logger.debug('++++++reuse MatMul2 to Conv')
        const_x_name = matmul2.name + '_to_conv_x_'

        v = node_dict['inputB2']
        old_dims = [node_dict['matmulB2_Shape'][0], node_dict['matmulB2_Shape'][1]]
        dims_ = [node_dict['matmulB2_Shape'][1], node_dict['matmulB2_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul2.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul2.attribute.append(attr)        

        B = add2.input[0]
        matmul2.input.append(B)

        output_shape = values.get_tensor_shape_by_name(model, matmul2.output[0])
        conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]] 

        update_tensor_shape(model, matmul2.output[0], conv_output_shape) 

        #Add2--->Reshape
        add2.op_type = 'Reshape'

        del add2.attribute[:]

        rs2_name = add2.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [conv_output_shape[0], conv_output_shape[1], conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = add2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        add2.input[0] = add2.input[1]

        add2.input[1] = const_shape_name

        update_tensor_shape(model, add2.output[0], rs2_output_shape)

        update_tensor_shape(model, nextAdd.output[0], rs2_output_shape)

        nextAdd_next_list, ok = get_all_next_node_by_output(model, nextAdd.output[0])
        if ok == 0:
            where_node = None
            tp_node = None
            add_node = None
            for node in nextAdd_next_list:
                logger.debug('----nextAdd_next_list, node: {}'.format(node.name))
                if node.op_type == 'Where':
                    where_node = node

                if node.op_type == 'Transpose':
                    tp_node = node

                if node.op_type == 'Add':
                    add_node = node 

            if where_node != None and tp_node != None and add_node != None:
                where_node.input[1] = tp_node.output[0]
            elif where_node != None and tp_node == None:
                ###add transpose
                tp_name = nextAdd.name + '_transpose_'
                tp_output_name = tp_name + '_output_'
                add_output_shape = values.get_tensor_shape_by_name(model, nextAdd.output[0])
                tp_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
                tp_output = onnx.helper.make_tensor_value_info(tp_output_name, onnx.TensorProto.FLOAT, tp_output_shape)
                
                tp_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=tp_name,
                                                    inputs=[nextAdd.output[0]],
                                                    outputs=[tp_output_name],
                                                    perm=[0,2,1])

                model.graph.value_info.append(tp_output)

                insert_node(model, tp_node, where_node)

                where_node.input[1] = tp_output_name

def get_last_group(model):
    graph_output = []
    node_dict = {}
    res = -1

    for o in model.graph.output:
        graph_output.append(o.name)

    for node in model.graph.node:
        if node.output[0] in graph_output:
            #print('got mul:', node.name)
            if node.op_type == 'LogSoftmax' or node.op_type == 'Softmax':
                logger.debug('got LogSoftmax node: {}'.format(node.name))
                node_dict['LogSoftmax'] = node

                add_node, ok = get_prev_node_by_input(model, node.input[0])
                if ok == 0 and add_node.op_type == 'Add':
                    addA_name = add_node.input[0]
                    addA, shapeA = values.get_init_value_and_shape(model, add_node.input[0])

                    if len(shapeA) == 1:
                        node_dict['Add'] = add_node
                        logger.debug('!!!!! got Add node: {}'.format(add_node.name))

                        matmul_node, ok = get_prev_node_by_input(model, add_node.input[1])
                        if ok == 0 and matmul_node.op_type == 'MatMul':
                            logger.debug('got MatMul node: {}'.format(matmul_node.name))
                            shapeA = values.get_tensor_shape_by_name(model, matmul_node.input[0])
                            inputB, shapeB = values.get_init_value_and_shape(model, matmul_node.input[1])

                            if isinstance(inputB, list) and inputB == []:
                                logger.debug('inputB is not in initilizer')
                                inputB = values.get_constant_value(model, matmul_node.input[1])

                            if len(shapeA) == 3 and len(shapeB) == 2:
                                logger.debug('++++ got MatMul2 node: {}'.format(matmul_node.name))
                                node_dict['MatMul'] = matmul_node
                                node_dict['matmulA_Shape'] = shapeA
                                node_dict['inputB'] = inputB
                                node_dict['matmulB_Shape'] = shapeB

                                add_node2, ok = get_prev_node_by_input(model, matmul_node.input[0])
                                if ok == 0 and add_node2.op_type == 'Add':
                                    logger.debug('++++ got Add2 node: {}'.format(add_node2.name))
                                    node_dict['Add2'] = add_node2
                                    res = 0
                                    break

    return node_dict, res

#Mul->Add->MatMul->Add->LogSoftmax
def handle_last_group(model):
    node_dict, ok = get_last_group(model)
    if ok == 0:
        logger.debug('start handle_last_group')
        matmul_node = node_dict['MatMul']
        add_node = node_dict['Add']
        add2_node = node_dict['Add2']
        ls_node = node_dict['LogSoftmax']

        ###add transpose
        ts_name = add2_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        add_output_shape = values.get_tensor_shape_by_name(model, add2_node.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[add2_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        ###add reshape
        rs_name = add2_node.name + '_reshape_2_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]
        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape2_name = add2_node.name + '_reshape2_data_'
        
        const_shape2_tensor = onnx.helper.make_tensor(name=const_shape2_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape2_tensor)

        rs_node = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs_name,
                                        inputs=[ts_output_name, const_shape2_name],
                                        outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, matmul_node)
        matmul_node.input[0] = rs_output_name 

        insert_node(model, ts_node, rs_node)

        #MatMul-->Conv
        matmul_node.op_type = 'Conv'
        const_x_name = matmul_node.name + '_to_conv_x_'

        v = node_dict['inputB']
        old_dims = [node_dict['matmulB_Shape'][0], node_dict['matmulB_Shape'][1]]
        dims_ = [node_dict['matmulB_Shape'][1], node_dict['matmulB_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul_node.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul_node.attribute.append(attr)        

        matmul_node.input.append(add_node.input[0])

        mm_output_shape = values.get_tensor_shape_by_name(model, matmul_node.output[0])
        conv_output_shape = [mm_output_shape[0], mm_output_shape[2], 1, mm_output_shape[1]]
        update_tensor_shape(model, matmul_node.output[0], conv_output_shape) 

        ###########
        add_node.op_type = 'Reshape'
        reshape_output = add_node.output[0]

        const_shape_name = add_node.name + '_to_reshape_'

        add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
        rs2_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]

        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        add_node.input[0] = add_node.input[1]
        add_node.input[1] = const_shape_name

        update_tensor_shape(model, add_node.output[0], rs2_output_shape)

        ###add transpose
        ts2_name = add_node.name + '_transpose_'
        ts2_output_name = ts2_name + '_output_'
        ts2_output_shape = add_output_shape
        transpose_output = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, ts2_output_shape)
        
        ts2_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts2_name,
                                            inputs=[add_node.output[0]],
                                            outputs=[ts2_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        insert_node(model, ts2_node, ls_node)
        ls_node.input[0] = ts2_output_name

def handle_add_combination_pattern_one(model):
    ars_list = get_add_combination_pattern_one(model)
    #print('handle_add_combination_pattern_one,ars_list:', ars_list)

    if len(ars_list):
        ars = ars_list[0]

        add_node = ars['currentAdd']
        next_add_node = ars['nextAdd']
        sub_node = ars['Sub'] 
        rm_node = ars['ReduceMean']  

        ###add transpose
        ts_name = add_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
        ts_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[add_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs_name = add_node.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [ts_output_shape[0], ts_output_shape[1]] #TBD


        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add_node.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        ###add reshape-2
        rs2_name = add_node.name + '_reshape_2_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [add_output_shape[0], add_output_shape[2], add_output_shape[1]]
        rs2_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape2_name = add_node.name + '_reshape2_data_'
        
        const_shape2_tensor = onnx.helper.make_tensor(name=const_shape2_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape2_tensor)


        rs2_node = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs2_name,
                                        inputs=[rs_output_name, const_shape2_name],
                                        outputs=[rs2_output_name])

        model.graph.value_info.append(rs2_output)

        ###add transpose2
        ts2_name = add_node.name + '_transpose2_'
        ts2_output_name = ts2_name + '_output_'
        ts2_output = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, [add_output_shape[0], add_output_shape[1], add_output_shape[2]])

        model.graph.value_info.append(ts2_output)

        ts2_node = onnx.helper.make_node(
                                        'Transpose',
                                        name=ts2_name,
                                        inputs=[rs2_output_name],
                                        outputs=[ts2_output_name],
                                        perm=[0,2,1])

        insert_node(model, ts2_node, rm_node)
        rm_node.input[0] = ts2_output_name 
        sub_node.input[0] = ts2_output_name

        insert_node(model, rs2_node, ts2_node)

        insert_node(model, rs_node, rs2_node)

        insert_node(model, ts_node, rs_node)

        next_add_node.input[0] = rs2_output_name

        ars_list2 = ars_list[1:]

        for ars in ars_list2:
            add_node = ars['currentAdd']
            next_add_node = ars['nextAdd']
            sub_node = ars['Sub'] 
            rm_node = ars['ReduceMean']

            ###add transpose
            add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
            ts2_name = add_node.name + '_transpose2_'
            ts2_output_name = ts2_name + '_output_'
            ts2_output = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, add_output_shape)

            model.graph.value_info.append(ts2_output)

            ts2_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts2_name,
                                            inputs=[add_node.output[0]],
                                            outputs=[ts2_output_name],
                                            perm=[0,2,1])

            insert_node(model, ts2_node, rm_node)
            rm_node.input[0] = ts2_output_name 
            sub_node.input[0] = ts2_output_name

def print_matmul_input_path(node_list, desp):
    node_print = ''
    first = True

    for n in node_list:
        if first == True:
            node_print = n.name
            first = False
        else: 
            node_print = node_print + '-->'   
            node_print = node_print + n.name

    logger.debug('{}:{}'.format(desp, node_print))

def update_tensor_shape(model, tensor_name, target_shape_list):
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            dim = vi.type.tensor_type.shape.dim[0]
            del vi.type.tensor_type.shape.dim[:]#[0]
            # dim_proto_input.dim_param = 'bs'
            for ss in target_shape_list:
                dim.dim_value = ss
                vi.type.tensor_type.shape.dim.append(dim)
            break   


def do_convert_pattern_one(model, matmul_dict, isInputA):
    orig_reshape_name = ''
    orig_matmul_name = ''
    reshape_output = ''

    bert_mode = 1

    current_inputA_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[0])
    current_inputB_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[1])

    logger.debug('current: {}, inputA_shape: {}, inputB_shape: {}'.format(matmul_dict['current'].name, current_inputA_shape, current_inputB_shape)) 

    map_key = ''

    if isInputA == True:
        inputA_shape = matmul_dict['A_matmul_AShape']
        inputB_shape = matmul_dict['A_matmul_BShape']
        path_node = matmul_dict['pathA']
        if 'A_prev' in matmul_dict.keys():
            map_key = matmul_dict['A_prev']
    else:
        inputA_shape = matmul_dict['B_matmul_AShape']
        inputB_shape = matmul_dict['B_matmul_BShape']
        path_node = matmul_dict['pathB']
        if 'B_prev' in matmul_dict.keys():
            map_key = matmul_dict['B_prev']

    logger.debug('A inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))
    logger.debug('B inputA shape:{}, inputB shape:{}'.format(matmul_dict['B_matmul_AShape'], matmul_dict['B_matmul_BShape']))
    
    remove_matmul = False
    remove_add = False
    matmul_input0 = ''
    matmul_output0 = ''
    matmul_input0_shape = []
    add_input1 = ''
    add_input1_shape = []

    reuse_transpose = False
    reuse_reshape = False
    
    if len(current_inputA_shape) == 3 and len(current_inputB_shape) == 3:
        for node in path_node:
            if node.op_type == 'MatMul':
                matmul_input0 = node.input[0]
                matmul_output0 = node.output[0]
                matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)
                
                if inputA_shape[1] != inputB_shape[0]:
                    if map_key in transpose_node_map.keys():
                        #transpose_node_map[map_key]
                        logger.debug('------ found transpose_node_map, key: {}'.format(map_key))
                        #model.graph.node.remove(node)
                        operation.remove_onnx_node(model, node)
                        reuse_transpose = True
                        bert_mode = 0
                    else:
                        orig_matmul_name = node.name
                        logger.debug('---matmul+add-->conv, need same channel: {} {}, node.name: {}'.format(inputA_shape[1], inputB_shape[0], node.name))
                        node.op_type = 'Transpose'
                        attr = onnx.helper.make_attribute('perm', [0,2,1])
                        node.attribute.append(attr)
                        del node.input[1:]
                        update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]])

                        transpose_node_map[map_key] = node
                        logger.debug('---map_key is {}'.format(map_key)) 
                else:
                    logger.debug('----Delete MatMul node: {}'.format(node.name))
                    #matmul_input0 = node.input[0]
                    #matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)

                    #model.graph.node.remove(node)
                    operation.remove_onnx_node(model, node)
                    remove_matmul = True
                    bert_mode = 0

            '''
            ###########
            if node.op_type == 'Add':
                logger.debug('reuse Add to Reshape')
                orig_reshape_name = node.name
                node.op_type = 'Reshape'
                const_shape_name = node.name + '_to_reshape_'
                rs_output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                if remove_matmul == True:
                    rs_output_shape = [matmul_input0_shape[1], matmul_input0_shape[2], 1, matmul_input0_shape[0]]  
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(rs_output_shape)],
                                    vals=rs_output_shape)

                model.graph.initializer.append(const_shape_tensor)

                if reuse_transpose == False:
                    node.input[0] = matmul_output0
                else:
                    node.input[0] = transpose_node_map[map_key].output[0]

                node.input[1] = const_shape_name 
                update_tensor_shape(model, node.output[0], rs_output_shape)
                ##################
                '''

            if node.op_type == 'Add':
                rs_output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                if remove_matmul == True:
                    rs_output_shape = [matmul_input0_shape[1], matmul_input0_shape[2], 1, matmul_input0_shape[0]]

                if map_key in reshape_node_map.keys():
                    logger.debug('------ found reshape_node_map, key: {}'.format(map_key))
                    #model.graph.node.remove(node)
                    model.graph.node.remove(node)
                    reuse_reshape = True
                else:
                    logger.debug('reuse Add to Reshape')
                    orig_reshape_name = node.name
                    node.op_type = 'Reshape'
                    const_shape_name = node.name + '_to_reshape_'
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)

                    if reuse_transpose == False:
                        node.input[0] = matmul_output0
                    else:
                        node.input[0] = transpose_node_map[map_key].output[0]

                    node.input[1] = const_shape_name 
                    update_tensor_shape(model, node.output[0], rs_output_shape)

                    reshape_node_map[map_key] = node
                    ##################    

            if node.op_type == 'Reshape' and node.name != orig_reshape_name:
                rs1_output_shape = values.get_tensor_shape_by_name(model, node.output[0])

                logger.debug('-----reuse Reshape to Conv')
                node.op_type = 'Conv'
                const_x_name = node.name + '_to_conv_x_'

                v = matmul_dict['A_inputB']
                old_dims = [matmul_dict['A_matmul_BShape'][0], matmul_dict['A_matmul_BShape'][1]]
                dims_ = [matmul_dict['A_matmul_BShape'][1], matmul_dict['A_matmul_BShape'][0],1,1]

                if reuse_reshape == True:
                    node.input[0] = reshape_node_map[map_key].output[0]

                operation.remove_initializer_if_necessary_by_name(model, node.input[1], node)
                
                if isInputA == False:
                    v = matmul_dict['B_inputB']
                    old_dims = [matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1]]
                    dims_ = [matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1]

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(*old_dims)
                    A = A.transpose()
                    A = A.reshape(*dims_)
                    logger.debug('+++A.shape: {}'.format(A.shape))
                    A = A.flatten()
                else:    
                    A = np.array(v).reshape(*old_dims)
                    A = A.transpose()
                    A = A.reshape(*dims_)
                    logger.debug('---A.shape: {}'.format(A.shape))
                    A = A.flatten()

                A = A.tolist()  
                const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=dims_,#[matmul_dict['A_matmul_BShape'][1], matmul_dict['A_matmul_BShape'][0],1,1],
                                    vals=A)

                model.graph.initializer.append(const_x_tensor)
                node.input[1] = const_x_name

                del node.attribute[:]

                attr = onnx.helper.make_attribute('dilations', [1, 1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('group', 1)
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('kernel_shape', [1, 1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('strides', [1,1])
                node.attribute.append(attr)        

                if isInputA == True:
                    node.input.append(matmul_dict['A_addA'])
                else:
                    node.input.append(matmul_dict['B_addA'])

                output_shape = rs_output_shape #[inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                update_tensor_shape(model, node.output[0], output_shape) 
        
            if node.op_type == 'Transpose' and node.name != orig_matmul_name:
                logger.debug('-----reuse Transpose to Reshape, node.name: {}'.format(node.name))
                node.op_type = 'Reshape'

                del node.attribute[:]
                
                reshape_output = node.output[0]

                tp_output_shape = values.get_tensor_shape_by_name(model, reshape_output)

                const_shape_name = node.name + '_to_reshape_'
                if isInputA == True:
                    output_shape = [rs1_output_shape[1], rs1_output_shape[2], rs1_output_shape[0]] 
                else:
                    output_shape = [rs1_output_shape[1], rs1_output_shape[2], rs1_output_shape[0]]#[current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[2]] 
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(output_shape)],
                                    vals=output_shape)

                model.graph.initializer.append(const_shape_tensor)
                node.input.append(const_shape_name)
                update_tensor_shape(model, node.output[0], output_shape)

                follow_up_node = node
                next_div_node, ok = get_next_node_by_output(model, reshape_output)
                if ok == 0 and next_div_node.op_type == 'Div':
                    shape = values.get_tensor_shape_by_name(model, next_div_node.output[0])
                    update_tensor_shape(model, next_div_node.output[0], [shape[0], shape[2], shape[1]])


        if isInputA == False:
            ts_name = const_shape_name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [tp_output_shape[0], tp_output_shape[2],tp_output_shape[1]])

            transpose_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[reshape_output],
                                                    outputs=[ts_output_name],
                                                    perm=[0,2,1])

            insert_node(model, transpose_node, follow_up_node)

            model.graph.value_info.append(transpose_output)

            matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            matmul_dict['current'].input[0] = ts_output_name

            output_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].output[0])
            update_tensor_shape(model, matmul_dict['current'].output[0], [output_shape[0], output_shape[2], output_shape[1]])
        '''
        else:
            ts_name = const_shape_name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, tp_output_shape)

            transpose_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[reshape_output],
                                                    outputs=[ts_output_name],
                                                    perm=[1,0,2])

            insert_node(model, transpose_node, follow_up_node)

            next_div_node.input[0] = ts_output_name

            model.graph.value_info.append(transpose_output)

            #matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            #matmul_dict['current'].input[0] = ts_output_name

            #output_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].output[0])
            #update_tensor_shape(model, matmul_dict['current'].output[0], [output_shape[0], output_shape[2], output_shape[1]])    
        '''
    else:
        for node in path_node:
            if node.op_type == 'MatMul':
                if inputA_shape[1] != inputB_shape[0]:
                    orig_matmul_name = node.name
                    logger.debug('matmul+add-->conv, need same channel: {} {}'.format(inputA_shape[1], inputB_shape[0]))
                    node.op_type = 'Transpose'
                    attr = onnx.helper.make_attribute('perm', [0,2,1])
                    node.attribute.append(attr)
                    del node.input[1:]
                    update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]])

                    transpose_node_map[map_key] = node
                    logger.debug('map_key is {}'.format(map_key)) 
                else:
                    logger.debug('Delete MatMul node: {}'.format(node.name))
                    matmul_input0 = node.input[0]
                    matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)

                    #model.graph.node.remove(node)
                    operation.remove_onnx_node(model, node)
                    remove_matmul = True
                    bert_mode = 0

            if node.op_type == 'Add':
                logger.debug('reuse Add to Reshape')
                orig_reshape_name = node.name
                node.op_type = 'Reshape'
                const_shape_name = node.name + '_to_reshape_'
                rs_output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                if remove_matmul == True:
                    rs_output_shape = [matmul_input0_shape[0], matmul_input0_shape[1], 1, matmul_input0_shape[2]]  
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(rs_output_shape)],
                                    vals=rs_output_shape)

                model.graph.initializer.append(const_shape_tensor)
                node.input[0] = node.input[1]
                if remove_matmul == True:
                    node.input[0] = matmul_input0

                node.input[1] = const_shape_name 
                update_tensor_shape(model, node.output[0], rs_output_shape)

            if node.op_type == 'Reshape' and node.name != orig_reshape_name:
                logger.debug('reuse Reshape to Conv')
                node.op_type = 'Conv'
                const_x_name = node.name + '_to_conv_x_'

                v = matmul_dict['A_inputB']
                old_dims = [matmul_dict['A_matmul_BShape'][0], matmul_dict['A_matmul_BShape'][1]]
                dims_ = [matmul_dict['A_matmul_BShape'][1], matmul_dict['A_matmul_BShape'][0],1,1]
                
                if isInputA == False:
                    v = matmul_dict['B_inputB']
                    old_dims = [matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1]]
                    dims_ = [matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0],1,1]

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(*old_dims)
                    A = A.transpose()
                    A = A.reshape(*dims_)
                    logger.debug('+++A.shape: {}'.format(A.shape))
                    A = A.flatten()
                else:    
                    A = np.array(v).reshape(*old_dims)
                    A = A.transpose()
                    A = A.reshape(*dims_)
                    logger.debug('---A.shape: {}'.format(A.shape))
                    A = A.flatten()

                A = A.tolist()  
                const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=dims_,#[matmul_dict['A_matmul_BShape'][1], matmul_dict['A_matmul_BShape'][0],1,1],
                                    vals=A)

                model.graph.initializer.append(const_x_tensor)
                node.input[1] = const_x_name

                attr = onnx.helper.make_attribute('dilations', [1, 1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('group', 1)
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('kernel_shape', [1,1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('strides', [1,1])
                node.attribute.append(attr)        

                if isInputA == True:
                    node.input.append(matmul_dict['A_addA'])
                else:
                    node.input.append(matmul_dict['B_addA'])

                output_shape = rs_output_shape #[inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                update_tensor_shape(model, node.output[0], output_shape) 
        
            if node.op_type == 'Transpose' and node.name != orig_matmul_name:
                logger.debug('reuse Transpose to Reshape, node.name: {}'.format(node.name))
                node.op_type = 'Reshape'
                del node.attribute[:]
                reshape_output = node.output[0]
                const_shape_name = node.name + '_to_reshape_'
                if isInputA == True:
                    output_shape = [current_inputA_shape[0], current_inputA_shape[1], current_inputA_shape[3], current_inputA_shape[2]] 
                else:
                    #output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                    output_shape = [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[2], current_inputB_shape[3]] 
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(output_shape)],
                                    vals=output_shape)

                model.graph.initializer.append(const_shape_tensor)
                node.input.append(const_shape_name)
                update_tensor_shape(model, node.output[0], output_shape)

                follow_up_node = node

        if isInputA == False:
            ts_name = const_shape_name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]])

            transpose_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[reshape_output],
                                                    outputs=[ts_output_name],
                                                    perm=[0,1,3,2])

            insert_node(model, transpose_node, follow_up_node)

            model.graph.value_info.append(transpose_output)

            matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            matmul_dict['current'].input[0] = ts_output_name

            output_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].output[0])
            update_tensor_shape(model, matmul_dict['current'].output[0], [output_shape[0], output_shape[1], output_shape[3], output_shape[2]])
        
    return bert_mode

def do_convert_pattern_four(model, matmul_dict, isInputA):
    orig_reshape_name = ''
    orig_matmul_name = ''
    reshape_output = ''

    bert_mode = 1

    current_inputA_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[0])
    current_inputB_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[1])

    logger.debug('---current_inputA_shape: {}'.format(current_inputA_shape)) 
    logger.debug('---current_inputB_shape: '.format(current_inputB_shape))

    map_key = ''

    if isInputA == True:
        inputA_shape = matmul_dict['A_matmul_AShape']
        inputB_shape = matmul_dict['A_matmul_BShape']
        path_node = matmul_dict['pathA']
        if 'A_prev' in matmul_dict.keys():
            map_key = matmul_dict['A_prev']
    else:
        inputA_shape = matmul_dict['B_matmul_AShape']
        inputB_shape = matmul_dict['B_matmul_BShape']
        path_node = matmul_dict['pathB']
        if 'B_prev' in matmul_dict.keys():
            map_key = matmul_dict['B_prev']

    logger.debug('A inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))
    logger.debug('B inputA shape:{}, inputB shape:{}'.format(matmul_dict['B_matmul_AShape'], matmul_dict['B_matmul_BShape']))
    
    remove_matmul = False
    matmul_input0 = ''
    matmul_input0_shape = []

    for node in path_node:
        if node.op_type == 'MatMul':
            if inputA_shape[1] != inputB_shape[0]:
                orig_matmul_name = node.name
                logger.debug('matmul+add-->conv, need same channel'.format(inputA_shape[1], inputB_shape[0]))
                node.op_type = 'Transpose'
                attr = onnx.helper.make_attribute('perm', [0,2,1])
                node.attribute.append(attr)
                del node.input[1:]
                update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]])

                transpose_node_map[map_key] = node
                logger.debug('map_key is {}'.format(map_key)) 
            else:
                logger.debug('Delete MatMul node: {}'.format(node.name))
                matmul_input0 = node.input[0]
                matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)

                #model.graph.node.remove(node)
                operation.remove_onnx_node(model, node)
                remove_matmul = True
                bert_mode = 0

        if node.op_type == 'Add':
            logger.debug('reuse Add to Reshape')
            orig_reshape_name = node.name
            node.op_type = 'Reshape'
            const_shape_name = node.name + '_to_reshape_'
            rs_output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
            if remove_matmul == True:
               rs_output_shape = [matmul_input0_shape[0], matmul_input0_shape[1], 1, matmul_input0_shape[2]]  
            
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs_output_shape)],
                                vals=rs_output_shape)

            model.graph.initializer.append(const_shape_tensor)
            node.input[0] = node.input[1]
            if remove_matmul == True:
               node.input[0] = matmul_input0
            node.input[1] = const_shape_name 
            update_tensor_shape(model, node.output[0], rs_output_shape)

        if node.op_type == 'Reshape' and node.name != orig_reshape_name:
            logger.debug('reuse Reshape to Conv')
            node.op_type = 'Conv'
            const_x_name = node.name + '_to_conv_x_'

            v = matmul_dict['A_inputB']
            old_dims = [matmul_dict['A_matmul_BShape'][0], matmul_dict['A_matmul_BShape'][1]]
            dims_ = [matmul_dict['A_matmul_BShape'][1], matmul_dict['A_matmul_BShape'][0],1,1]
            
            if isInputA == False:
                v = matmul_dict['B_inputB']
                old_dims = [matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1]]
                dims_ = [matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0],1,1]

            if isinstance(v, np.ndarray) == True:
                A = v.reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                logger.debug('+++A.shape: {}'.format(A.shape))
                A = A.flatten()
            else:    
                A = np.array(v).reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                logger.debug('---A.shape: {}'.format(A.shape))
                A = A.flatten()

            A = A.tolist()  
            const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                                data_type=onnx.TensorProto.FLOAT,
                                dims=dims_,#[matmul_dict['A_matmul_BShape'][1], matmul_dict['A_matmul_BShape'][0],1,1],
                                vals=A)

            model.graph.initializer.append(const_x_tensor)
            node.input[1] = const_x_name

            attr = onnx.helper.make_attribute('dilations', [1, 1])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('group', 1)
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('kernel_shape', [1,1])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('pads', [0,0,0,0])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('strides', [1,1])
            node.attribute.append(attr)        

            if isInputA == True:
                node.input.append(matmul_dict['A_addA'])
            else:
                node.input.append(matmul_dict['B_addA'])

            output_shape = rs_output_shape #[inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
            update_tensor_shape(model, node.output[0], output_shape) 
    
        if node.op_type == 'Transpose' and node.name != orig_matmul_name:
            logger.debug('reuse Transpose to Reshape')
            node.op_type = 'Reshape'
            del node.attribute[:]
            reshape_output = node.output[0]
            const_shape_name = node.name + '_to_reshape_'
            if isInputA == True:
                output_shape = [current_inputA_shape[0], current_inputA_shape[1], current_inputA_shape[3], current_inputA_shape[2]] 
            else:
                #output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                output_shape = [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[2], current_inputB_shape[3]] 
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(output_shape)],
                                vals=output_shape)

            model.graph.initializer.append(const_shape_tensor)
            node.input.append(const_shape_name)
            update_tensor_shape(model, node.output[0], output_shape)

            follow_up_node = node

    if isInputA == False:
        ts_name = const_shape_name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]])

        transpose_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[reshape_output],
                                                outputs=[ts_output_name],
                                                perm=[0,1,3,2])

        insert_node(model, transpose_node, follow_up_node)

        model.graph.value_info.append(transpose_output)

        matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
        matmul_dict['current'].input[0] = ts_output_name

        output_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].output[0])
        update_tensor_shape(model, matmul_dict['current'].output[0], [output_shape[0], output_shape[1], output_shape[3], output_shape[2]])
    
    return bert_mode

def do_convert_pattern_two(model, matmul_dict):
    orig_reshape_name = ''
    orig_matmul_name = ''
    reshape_output = ''

    current_inputA_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[0])
    current_inputB_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[1])

    logger.debug('do_convert_pattern_two current_inputA_shape: {}'.format(current_inputA_shape)) 
    logger.debug('do_convert_pattern_twocurrent_inputB_shape: {}'.format(current_inputB_shape))

    inputA_shape = matmul_dict['B_matmul_AShape']
    inputB_shape = matmul_dict['B_matmul_BShape']
    path_node = matmul_dict['pathB']

    logger.debug('do_convert_pattern_two A inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))
    logger.debug('do_convert_pattern_two B inputA shape:{}, inputB shape:{}'.format(matmul_dict['B_matmul_AShape'], matmul_dict['B_matmul_BShape']))
    
    reuse_transpose = False

    remove_matmul = False
    matmul_input0 = ''
    matmul_input0_shape = []
    add_input1 = ''

    if len(current_inputA_shape) == 3 and len(current_inputB_shape) == 3:
        map_key = ''
        reuse_reshape = False
        for node in path_node:
            if node.op_type == 'MatMul':
                map_key = node.input[0]
                matmul_input0 = node.input[0]
                matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)
                
                logger.debug('--handle MatMul: {}'.format(node.name))
                if inputA_shape[1] != inputB_shape[0]:
                    #map_key = node.input[0]
                    if map_key in transpose_node_map.keys():
                        #transpose_node_map[map_key]
                        logger.debug('found transpose_node_map, key: {}'.format(map_key))
                        #model.graph.node.remove(node)
                        operation.remove_onnx_node(model, node)
                        reuse_transpose = True
                    else:     
                        orig_matmul_name = node.name
                        logger.debug('#### matmul+add-->conv, need same channel: {} {}'.format(inputA_shape[1], inputB_shape[0]))
                        node.op_type = 'Transpose'
                        attr = onnx.helper.make_attribute('perm', [0,2,1])
                        node.attribute.append(attr)
                        del node.input[1:]
                        update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]]) 
                else:
                    logger.debug('###### Delete MatMul node: {}'.format(node.name))

                    #model.graph.node.remove(node)
                    operation.remove_onnx_node(model, node)
                    remove_matmul = True

            if node.op_type == 'Add':
                if map_key in reshape_node_map.keys():
                    logger.debug('++++++ found reshape_node_map, key: {}'.format(map_key))
                    #model.graph.node.remove(node)
                    model.graph.node.remove(node)
                    reuse_reshape = True
                else:    
                    logger.debug('----delete add node: {}'.format(node.name))
                    #add_input1 = node.input[1]
                    #model.graph.node.remove(node)

                    #'''
                    orig_reshape_name = node.name
                    node.op_type = 'Reshape'
                    const_shape_name = node.name + '_to_reshape_'
                    rs_output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]] 
                    if remove_matmul == True:
                        rs_output_shape = [matmul_input0_shape[1], matmul_input0_shape[2], 1, matmul_input0_shape[0]]  
                    
                    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                        data_type=onnx.TensorProto.INT64,
                                        dims=[len(rs_output_shape)],
                                        vals=rs_output_shape)

                    model.graph.initializer.append(const_shape_tensor)
                    if remove_matmul == True:
                        node.input[0] = matmul_input0
                    else:
                        if reuse_transpose == False:
                            node.input[0] = node.input[1]
                        else:
                            node.input[0] = transpose_node_map[map_key].output[0]

                    node.input[1] = const_shape_name 
                    update_tensor_shape(model, node.output[0], rs_output_shape)
                    #'''

            if node.op_type == 'Reshape' and node.name != orig_reshape_name:
                logger.debug('reuse Reshape to Conv')
                node.op_type = 'Conv'

                const_x_name = node.name + '_to_conv_x_'

                v = matmul_dict['B_inputB']

                rs1_output_shape = values.get_tensor_shape_by_name(model, node.output[0])

                if reuse_reshape == True:
                    node.input[0] = reshape_node_map[map_key].output[0]

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1])
                    A = A.transpose()
                    A = A.reshape(matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1)
                    logger.debug('+++A.shape: {}'.format(A.shape))
                    A = A.flatten()
                else:    
                    A = np.array(v).reshape(matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1])
                    A = A.transpose()
                    A = A.reshape(matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1)
                    logger.debug('---A.shape: {}'.format(A.shape))
                    A = A.flatten()

                A = A.tolist()  
                const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=[matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0],1, 1],
                                    vals=A)

                model.graph.initializer.append(const_x_tensor)
                node.input[1] = const_x_name

                del node.attribute[:]

                attr = onnx.helper.make_attribute('dilations', [1, 1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('group', 1)
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('kernel_shape', [1, 1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('strides', [1,1])
                node.attribute.append(attr)        

                node.input.append(matmul_dict['B_addA'])
                output_shape = values.get_tensor_shape_by_name(model, node.input[0]) #[inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                update_tensor_shape(model, node.output[0], output_shape) 

            if node.op_type == 'Transpose' and node.name != orig_matmul_name:
                logger.debug('reuse Transpose to Reshape')
                node.op_type = 'Reshape'
                del node.attribute[:]

                reshape_output = node.output[0]
                const_shape_name = node.name + '_to_reshape_'

                shape = values.get_tensor_shape_by_name(model, node.output[0])

                output_shape = [shape[0], shape[2], shape[1]] 
                
                if remove_matmul == True:
                    output_shape = output_shape #[current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]]
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(output_shape)],
                                    vals=output_shape)

                model.graph.initializer.append(const_shape_tensor)
                node.input.append(const_shape_name)
                update_tensor_shape(model, node.output[0], output_shape)

                follow_up_node = node

        output_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].output[0])
        #update_tensor_shape(model, matmul_dict['current'].output[0], [output_shape[0], output_shape[2], output_shape[1]])
        
        if remove_matmul == True:
            tmp = matmul_dict['current'].input[1]
            matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            matmul_dict['current'].input[0] = tmp

        if remove_matmul == False:#isInputA == False:
            ts_name = const_shape_name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[2]])

            transpose_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[reshape_output],
                                                    outputs=[ts_output_name],
                                                    perm=[0,2,1])

            insert_node(model, transpose_node, follow_up_node)

            model.graph.value_info.append(transpose_output)

            #matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            matmul_dict['current'].input[1] = ts_output_name
    else:
        for node in path_node:
            if node.op_type == 'MatMul':
                if inputA_shape[1] != inputB_shape[0]:
                    map_key = node.input[0]
                    if map_key in transpose_node_map.keys():
                        #transpose_node_map[map_key]
                        #model.graph.node.remove(node)
                        operation.remove_onnx_node(model, node)
                        reuse_transpose = True
                    else:     
                        orig_matmul_name = node.name
                        logger.debug('matmul+add-->conv, need same channel: {} {}'.format(inputA_shape[1], inputB_shape[0]))
                        node.op_type = 'Transpose'
                        attr = onnx.helper.make_attribute('perm', [0,2,1])
                        node.attribute.append(attr)
                        del node.input[1:]
                        update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]]) 
                else:
                    logger.debug('------Delete MatMul node: {}'.format(node.name))
                    matmul_input0 = node.input[0]
                    matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)

                    #model.graph.node.remove(node)
                    operation.remove_onnx_node(model, node)
                    remove_matmul = True


            if node.op_type == 'Add':
                logger.debug('----reuse Add to Reshape: {}'.format(node.name))
                orig_reshape_name = node.name
                node.op_type = 'Reshape'
                const_shape_name = node.name + '_to_reshape_'
                rs_output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]] 
                if remove_matmul == True:
                    rs_output_shape = [matmul_input0_shape[0], matmul_input0_shape[1], 1, matmul_input0_shape[2]]  
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(rs_output_shape)],
                                    vals=rs_output_shape)

                model.graph.initializer.append(const_shape_tensor)
                if remove_matmul == True:
                    node.input[0] = matmul_input0
                else:
                    if reuse_transpose == False:
                        node.input[0] = node.input[1]
                    else:
                        node.input[0] = transpose_node_map[map_key].output[0]

                node.input[1] = const_shape_name 
                update_tensor_shape(model, node.output[0], rs_output_shape)

            if node.op_type == 'Reshape' and node.name != orig_reshape_name:
                logger.debug('reuse Reshape to Conv')
                node.op_type = 'Conv'
                const_x_name = node.name + '_to_conv_x_'

                v = matmul_dict['B_inputB']

                if isinstance(v, np.ndarray) == True:
                    A = v.reshape(matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1])
                    A = A.transpose()
                    A = A.reshape(matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1)
                    logger.debug('+++A.shape: {}'.format(A.shape))
                    A = A.flatten()
                else:    
                    A = np.array(v).reshape(matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1])
                    A = A.transpose()
                    A = A.reshape(matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1)
                    logger.debug('---A.shape: {}'.format(A.shape))
                    A = A.flatten()

                A = A.tolist()  
                const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                                    data_type=onnx.TensorProto.FLOAT,
                                    dims=[matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0],1,1],
                                    vals=A)

                model.graph.initializer.append(const_x_tensor)
                node.input[1] = const_x_name

                attr = onnx.helper.make_attribute('dilations', [1, 1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('group', 1)
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('kernel_shape', [1,1])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('pads', [0,0,0,0])
                node.attribute.append(attr)

                attr = onnx.helper.make_attribute('strides', [1,1])
                node.attribute.append(attr)        

                node.input.append(matmul_dict['B_addA'])
                output_shape = rs_output_shape #[inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
                update_tensor_shape(model, node.output[0], output_shape) 
        
            if node.op_type == 'Transpose' and node.name != orig_matmul_name:
                logger.debug('reuse Transpose to Reshape')
                node.op_type = 'Reshape'
                del node.attribute[:]
                reshape_output = node.output[0]
                const_shape_name = node.name + '_to_reshape_'
                output_shape = [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[2], current_inputB_shape[3]] 
                if remove_matmul == True:
                    output_shape = [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]]
                
                const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                    data_type=onnx.TensorProto.INT64,
                                    dims=[len(output_shape)],
                                    vals=output_shape)

                model.graph.initializer.append(const_shape_tensor)
                node.input.append(const_shape_name)
                update_tensor_shape(model, node.output[0], output_shape)

                follow_up_node = node

        output_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].output[0])
        update_tensor_shape(model, matmul_dict['current'].output[0], [output_shape[0], output_shape[1], output_shape[3], output_shape[2]])
        
        if remove_matmul == True:
            tmp = matmul_dict['current'].input[1]
            matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            matmul_dict['current'].input[0] = tmp

        if remove_matmul == False:#isInputA == False:
            ts_name = const_shape_name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]])

            transpose_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[reshape_output],
                                                    outputs=[ts_output_name],
                                                    perm=[0,1,3,2])

            insert_node(model, transpose_node, follow_up_node)

            model.graph.value_info.append(transpose_output)

            matmul_dict['current'].input[1] = matmul_dict['current'].input[0]
            matmul_dict['current'].input[0] = ts_output_name

    
def do_convert_pattern_three(model, matmul_dict, ts_node):
    orig_reshape_name = ''
    orig_matmul_name = ''
    reshape_output = ''

    inputA_shape = matmul_dict['matmul_AShape']
    inputB_shape = matmul_dict['matmul_BShape']
    path_node = matmul_dict['node_list']

    logger.debug('----- inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))

    orig_reshape_name = ts_node.name

    ts_node.op_type = 'Reshape'

    del ts_node.attribute[:]

    const_shape_name = ts_node.name + '_to_reshape_'

    ts_input_shape = values.get_tensor_shape_by_name(model, ts_node.input[0])
    rs_output_shape = [ts_input_shape[0], ts_input_shape[1]*ts_input_shape[2], ts_input_shape[3]]

    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)
    ts_node.input.append(const_shape_name)

    del ts_node.attribute[:]

    update_tensor_shape(model, ts_node.output[0], rs_output_shape)

    for node in path_node:
        if node.op_type == 'Reshape':
            logger.debug('----reuse Reshape')

            rs2_output_shape = [rs_output_shape[0], rs_output_shape[1], 1, rs_output_shape[2]] 
            const_shape_name = node.name + '_reshape_data_'
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(rs2_output_shape)],
                                vals=rs2_output_shape)

            model.graph.initializer.append(const_shape_tensor)

            node.input[1] = const_shape_name

            update_tensor_shape(model, node.output[0], rs2_output_shape)

        if node.op_type == 'MatMul':
            logger.debug('---- reuse Matmul to Conv')

            node.op_type = 'Conv'
            const_x_name = node.name + '_to_conv_x_'

            v = matmul_dict['inputB']

            old_dims = [matmul_dict['matmul_BShape'][0], matmul_dict['matmul_BShape'][1]]

            dims_ = [matmul_dict['matmul_BShape'][1], matmul_dict['matmul_BShape'][0],1,1]
            
            if isinstance(v, np.ndarray) == True:
                A = v.reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                logger.debug('+++    A.shape: {}'.format(A.shape))
                A = A.flatten()
            else:    
                A = np.array(v).reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                logger.debug('---    A.shape: {}'.format(A.shape))
                A = A.flatten()

            A = A.tolist()  
            const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                                data_type=onnx.TensorProto.FLOAT,
                                dims=dims_,
                                vals=A)

            model.graph.initializer.append(const_x_tensor)
            node.input[1] = const_x_name

            attr = onnx.helper.make_attribute('dilations', [1, 1])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('group', 1)
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('kernel_shape', [1,1])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('pads', [0,0,0,0])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('strides', [1,1])
            node.attribute.append(attr)        

            node.input.append(matmul_dict['addA'])

            update_tensor_shape(model, node.output[0], rs2_output_shape) 

        if node.op_type == 'Add':
            logger.debug('----- reuse Add to Reshape')

            node.op_type = 'Reshape'

            add_first = matmul_dict['addFirst']

            const_shape_name = node.name + '_to_reshape_'

            output_shape = [rs2_output_shape[0], rs2_output_shape[1], rs2_output_shape[3]] 

            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(output_shape)],
                                vals=output_shape)

            model.graph.initializer.append(const_shape_tensor)

            if add_first == True:
                node.input[0] = node.input[1]

            node.input[1] = const_shape_name

            update_tensor_shape(model, node.output[0], output_shape)

            next_node, ok = get_next_node_by_output(model, node.output[0])
            update_tensor_shape(model, next_node.output[0], output_shape)

def cvt_matmul_add_to_conv(model, matmul_dict, pattern):
    if matmul_dict['next'][0].op_type == 'Div' or matmul_dict['next'][0].op_type == 'Add' or  matmul_dict['next'][0].op_type == 'Mul':
        bert_mode = -1
        logger.debug('cvt_matmul_add_to_conv, next: {}, current: {}'.format(matmul_dict['next'][0].name, matmul_dict['current'].name))
        
        if pattern == 4:
            if matmul_dict['A_MatMul_Add'] == True:
                bert_mode = do_convert_pattern_four(model, matmul_dict, True)

            if matmul_dict['B_MatMul_Add'] == True:
                bert_mode = do_convert_pattern_four(model, matmul_dict, False)
        else:
            if matmul_dict['A_MatMul_Add'] == True:
                bert_mode = do_convert_pattern_one(model, matmul_dict, True)

            if matmul_dict['B_MatMul_Add'] == True:
                bert_mode = do_convert_pattern_one(model, matmul_dict, False)

        ########### next node is Add
        if matmul_dict['next'][0].op_type == 'Add':
            current_node = matmul_dict['current']
            next_node = matmul_dict['next'][0]
            shape = values.get_tensor_shape_by_name(model, next_node.output[0])
            update_tensor_shape(model, next_node.output[0], [shape[0], shape[1], shape[3], shape[2]])

            ###add transpose
            ts_name = next_node.name + '_transpose_'
            ts_output_name = ts_name + '_output_'
            add_output_shape = values.get_tensor_shape_by_name(model, next_node.output[0])
            transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, shape)
            
            ts_node = onnx.helper.make_node(
                                                'Transpose',
                                                name=ts_name,
                                                inputs=[current_node.output[0]],
                                                outputs=[ts_output_name],
                                                perm=[0,1,3,2])

            model.graph.value_info.append(transpose_output)

            insert_node(model, ts_node, next_node)
            next_node.input[0] = ts_output_name
        else:
            next_node = matmul_dict['next'][0]
            shape = values.get_tensor_shape_by_name(model, next_node.output[0])
            logger.debug('next_node.name: {}, shape: {}'.format(next_node.name, shape))

            if len(shape) == 3:
                update_tensor_shape(model, next_node.output[0], [shape[0], shape[2], shape[1]])

                ###add transpose
                ts_name = next_node.name + '_transpose_'
                ts_output_name = ts_name + '_output_'
                add_output_shape = values.get_tensor_shape_by_name(model, next_node.output[0])
                transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, shape)
                
                ts_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[next_node.output[0]],
                                                    outputs=[ts_output_name],
                                                    perm=[0,2,1])

                model.graph.value_info.append(transpose_output)

                current_node = matmul_dict['current']

                logger.debug('insert ts_name: {}, current_node: {}'.format(ts_name, current_node.name))

                insert_node(model, ts_node, matmul_dict['nnext'][0])

                if matmul_dict['nnext'][0].op_type == 'Add':
                    if matmul_dict['nnext'][0].input[0] == next_node.output[0]:
                        matmul_dict['nnext'][0].input[0] = ts_output_name
                    else:
                        matmul_dict['nnext'][0].input[1] = ts_output_name
                else:
                    if bert_mode == 0:
                        matmul_dict['nnext'][0].input[0] = ts_output_name
                    else:    
                        matmul_dict['nnext'][0].input[1] = ts_output_name
            else:    
                update_tensor_shape(model, next_node.output[0], [shape[0], shape[1], shape[3], shape[2]])

                ###add transpose
                ts_name = next_node.name + '_transpose_'
                ts_output_name = ts_name + '_output_'
                add_output_shape = values.get_tensor_shape_by_name(model, next_node.output[0])
                transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, shape)
                
                ts_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[next_node.output[0]],
                                                    outputs=[ts_output_name],
                                                    perm=[0,1,3,2])

                model.graph.value_info.append(transpose_output)

                insert_node(model, ts_node, matmul_dict['nnext'][0])

                if bert_mode == 0:
                    matmul_dict['nnext'][0].input[0] = ts_output_name
                else:    
                    matmul_dict['nnext'][0].input[1] = ts_output_name
    elif matmul_dict['next'][0].op_type == 'Transpose':
        if matmul_dict['B_MatMul_Add'] == True:
            logger.debug('cvt_matmul_add_to_conv, BBBBBBBBBBBBBBBBBBBB')
            do_convert_pattern_two(model, matmul_dict)

        if matmul_dict['A_MatMul_Add'] == False:
            logger.debug('cvt_matmul_add_to_conv, CCCCCCCCCCCCCCCCCCCCC')
            path_node = matmul_dict['pathA']

            node= path_node[0]

            if node.op_type == 'Where' or node.op_type == 'Softmax':
                shape = values.get_tensor_shape_by_name(model, node.output[0])

                if len (shape) == 3:
                    return

                ###add transpose
                logger.debug('insert Transpose before: {}'.format(node.name))
                ts_name = node.name + '_transpose_'
                ts_output_name = ts_name + '_output_'

                perm_ = [0,1,3,2]

                if len (shape) == 3:
                    perm_ = [0, 2, 1]
                    transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [shape[0], shape[2],shape[1]])
                else:
                    transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [shape[0], shape[1],shape[3],shape[2]])
                
                ts_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[node.output[0]],
                                                    outputs=[ts_output_name],
                                                    perm=perm_)

                model.graph.value_info.append(transpose_output)

                insert_node(model, ts_node, matmul_dict['current'])

                matmul_dict['current'].input[1] = ts_output_name

                next_node = matmul_dict['next'][0]
                nnext_node = matmul_dict['nnext'][0]

                logger.debug('pattern two, Matmul name: {}, next_node: {}'.format(matmul_dict['current'].name, next_node.name))

                if len (shape) == 3:
                    del next_node.attribute[:]
                    attr = onnx.helper.make_attribute('perm', [2,0,1])
                    next_node.attribute.append(attr)

                op_dict, ok = get_matmul_input_path_pattern_two(model, next_node.output[0])

                if op_dict and ok == 0:
                    for node in op_dict['node_list']: 
                        logger.debug('got matmul+add path(pattern 2): {}'.format(node.name))
                         
                    do_convert_pattern_three(model, op_dict, next_node)            

def get_mul_add_transpose_matmul_block(model):
    matm_list = []

    for node in model.graph.node:
        if node.op_type == 'Mul':
            add_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and add_node.op_type == 'Add':
                tp_node, ok = get_next_node_by_output(model, add_node.output[0])
                if ok == 0 and tp_node.op_type == 'Transpose':
                    mm_node_list, ok = get_all_next_node_by_output(model, tp_node.output[0])
                    if ok == 0 and len(mm_node_list) == 3:
                        matm = {}
                        matm['Add'] = add_node
                        matm['Tp'] = tp_node
                        matm['mm1'] = mm_node_list[0]
                        matm['mm2'] = mm_node_list[1]
                        matm['mm3'] = mm_node_list[2]
                        matm_list.append(matm)

    return matm_list                    

def gen_mul_add_block_by_rm_transpose(model):
    logger.debug('into gen_mul_add_block_by_rm_transpose')

    node_list = []
    for node in model.graph.node:
        node_dict = {}
        if node.op_type == 'Mul':
            #print('got mul:', node.name)

            is_init = False

            for init in model.graph.initializer:
                if init.name == node.input[0] or init.name == node.input[1]:
                    is_init = True
                    break

            if is_init == False:
                dataA = values.get_constant_value(model, node.input[0])
                if len(dataA) == 0:
                    dataA = values.get_constant_value(model, node.input[1])

                if dataA != []:
                    is_init = True

            if is_init == True:
                #print('----got mul:', node.name)
                next_node, ok = get_next_node_by_output(model, node.output[0])
                if ok == 0 and next_node.op_type == 'Add':
                    ##############
                    #print('----got add:', next_node.name)
                    is_init = False

                    for init in model.graph.initializer:
                        if init.name == next_node.input[1]:
                            is_init = True
                            break

                    if is_init == False:
                        dataA = values.get_constant_value(model, next_node.input[1])
                        if dataA != []:
                            is_init = True

                if is_init == True:
                    #print('get_all_next_node_by_output---', next_node.output, node.name)
                    tp_node, ok = get_next_node_by_output(model, next_node.output[0])
                    if ok == 0 and tp_node.op_type== 'Transpose':
                        mm_node1, ok = get_next_node_by_output(model, tp_node.output[0])
                        if ok == 0 and mm_node1.op_type == 'MatMul':
                            add_node1, ok = get_next_node_by_output(model, mm_node1.output[0])
                            if ok == 0 and add_node1.op_type == 'Add':
                                next_node_list, ok = get_all_next_node_by_output(model, add_node1.output[0])
                                #print('next_node_list:', len(next_node_list))
                                if len(next_node_list) == 2:
                                    #print('got next_node_list:', next_node_list[0].op_type, next_node_list[1].op_type)

                                    if (next_node_list[0].op_type == 'Div' and next_node_list[1].op_type == 'Mul') or \
                                        (next_node_list[0].op_type == 'Mul' and next_node_list[1].op_type == 'Div'):
                                        logger.debug('---got it~')
                                        mul_node1 = next_node_list[0]
                                        if next_node_list[1].op_type == 'Mul':
                                            mul_node1 = next_node_list[1]

                                        mul_node2, ok = get_next_node_by_output(model, mul_node1.output[0])
                                        if ok == 0 and mul_node2.op_type == 'Mul':
                                            mm_node2, ok = get_next_node_by_output(model, mul_node2.output[0])
                                            if ok == 0 and mm_node2.op_type == 'MatMul':
                                                add_node2, ok = get_next_node_by_output(model, mm_node2.output[0])
                                                if ok == 0 and add_node2.op_type == 'Add':
                                                    tp_node2, ok = get_next_node_by_output(model, add_node2.output[0])
                                                    if ok == 0 and tp_node2.op_type == 'Transpose':
                                                        add_node3, ok = get_next_node_by_output(model, tp_node2.output[0])
                                                        if ok == 0 and add_node3.op_type == 'Add':
                                                            logger.debug('got match transpose block')
                                                            node_dict['Add'] = add_node3
                                                            node_dict['Transpose2'] = tp_node2
                                                            node_dict['MatMul'] = mm_node1
                                                            node_dict['Transpose'] = tp_node
                                                            node_list.append(node_dict)

    for nd in node_list:
        logger.debug('gen_mul_add_block_by_rm_transpose working...')
        tp_node = nd['Transpose']
        tp_node2 = nd['Transpose2']
        add_node = nd['Add']
        mm_node = nd['MatMul']

        mm_node.input[0] = tp_node.input[0]
        #model.graph.node.remove(tp_node)
        logger.debug('gen_mul_add_block_by_rm_transpose, remove transpose node: {}'.format(tp_node.name))
        operation.remove_onnx_node(model, tp_node)

        add_node.input[1] = tp_node2.input[0]
        logger.debug('gen_mul_add_block_by_rm_transpose, remove transpose node2: {}'.format(tp_node2.name))
        #model.graph.node.remove(tp_node2)
        operation.remove_onnx_node(model, tp_node2)

def gen_mul_add_block_by_rm_transpose2(model):
    tramt_list = []

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            rs_node, ok = get_prev_node_by_input(model, node.input[0])
            if ok == 0 and rs_node.op_type == 'Reshape':
                add_node, ok = get_prev_node_by_input(model, rs_node.input[0])
                if ok == 0 and add_node.op_type == 'Add':
                    mm_node, ok = get_prev_node_by_input(model, add_node.input[1])
                    if ok == 0 and mm_node.op_type == 'MatMul':
                        tp_node2, ok = get_prev_node_by_input(model, mm_node.input[0])
                        if ok == 0 and tp_node2.op_type == 'Transpose':
                            logger.debug('got tramt {}'.format(mm_node.name))
                            tramt = {}
                            tramt['Transpose'] = tp_node2
                            '''
                            tramt['Reshape'] = rs_node
                            tramt['Add'] = add_node
                            tramt['MatMul'] = mm_node
                            tramt['Transpose2'] = tp_node2
                            '''
                            if tramt not in tramt_list:
                                tramt_list.append(tramt)

    for tramt in tramt_list:
        tp_node = tramt['Transpose']

        tp_next_node_list, _ = get_all_next_node_by_output(model, tp_node.output[0])

        for node in tp_next_node_list:
            node.input[0] = tp_node.input[0]

        logger.debug('gen_mul_add_block_by_rm_transpose2, remove transpose node: {}'.format(tp_node.name))
        model.graph.node.remove(tp_node)    

def correct_reshape_expand_reshape_pattern(model):
    logger.debug('into correct_reshape_expand_reshape_pattern')

    node_list = []

    for node in model.graph.node:
        node_dict = {}
        if node.op_type == 'Reshape':
            expend_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and expend_node.op_type == 'Expand':
                reshape_node, ok = get_next_node_by_output(model, expend_node.output[0])
                if ok == 0 and reshape_node.op_type == 'Reshape':
                    node_dict['Reshape1'] = node
                    node_dict['Expand'] = expend_node
                    node_dict['Reshape2'] = reshape_node

                    node_list.append(node_dict)

    for index, nd in enumerate(node_list):
        rs_node1 = nd['Reshape1']
        rs_node2 = nd['Reshape2']
        expand_node = nd['Expand']

        len_reshape1_input = len(values.get_tensor_shape_by_name(model, rs_node1.input[0]))
        len_reshape2_output = len(values.get_tensor_shape_by_name(model, rs_node2.output[0]))

        if len_reshape1_input < 4:
            diff = 4 - len_reshape1_input
            logger.info('Corrent Reshape+Expand+Reshape {}'.format(len_reshape1_input))

            input_shape = values.get_tensor_shape_by_name(model, rs_node1.input[0])
            new_shape = [1,1,1,1]

            for idx, v in enumerate(input_shape):
                new_shape[idx+diff] = v

            logger.debug('got new_shape: {}'.format(new_shape))

            if True:
                shape_tensor_name = rs_node1.name + '_reshape_data_' + str(index)
                const_shape = onnx.helper.make_tensor(shape_tensor_name, onnx.TensorProto.INT64, [4], new_shape)
                model.graph.initializer.append(const_shape)

                output_tensor_name = rs_node1.name + '_reshape_' + str(index)
                output_tensor = onnx.helper.make_tensor_value_info(output_tensor_name, onnx.TensorProto.FLOAT, new_shape)

                model.graph.value_info.append(output_tensor)

                prev_node, _ = get_prev_node_by_input(model, rs_node1.input[0])

                rs_node = onnx.helper.make_node(
                    name=rs_node1.name+'__Reshape__'+ str(index),
                    op_type='Reshape', 
                    inputs=[rs_node1.input[0], shape_tensor_name],
                    outputs=[output_tensor_name]
                    )

                insert_node(model, rs_node, prev_node)
                rs_node1.input[0] = output_tensor_name
                ###############################################
                shape_tensor_name = rs_node2.name + '_reshape_data_' + str(index)
                new_shape = values.get_tensor_shape_by_name(model, expand_node.output[0])
                const_shape = onnx.helper.make_tensor(shape_tensor_name, onnx.TensorProto.INT64, [4], new_shape)
                model.graph.initializer.append(const_shape)

                output_tensor_name = rs_node2.name + '_reshape_' + str(index)
                output_tensor = onnx.helper.make_tensor_value_info(output_tensor_name, onnx.TensorProto.FLOAT, new_shape)

                model.graph.value_info.append(output_tensor)

                rs_node = onnx.helper.make_node(
                    name=rs_node2.name+'__Reshape__'+ str(index),
                    op_type='Reshape', 
                    inputs=[expand_node.output[0], shape_tensor_name],
                    outputs=[output_tensor_name]
                    )

                insert_node(model, rs_node, expand_node)
                rs_node2.input[0] = output_tensor_name

#Transpose-->Reshape-->MatMul-->Add-->Reshape-->Transpose
def handle_matmul_add_child_block(model):
    logger.debug('into handle_matmul_add_child_block')

    node_list = []

    for node in model.graph.node:
        node_dict = {}
        if node.op_type == 'Transpose':
            logger.debug('into handle_matmul_add_child_block, step 1')
            rs_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and rs_node.op_type == 'Reshape':
                logger.debug('into handle_matmul_add_child_block, step 2')
                mm_node, ok = get_next_node_by_output(model, rs_node.output[0])
                if ok == 0 and mm_node.op_type == 'MatMul':
                    logger.debug('into handle_matmul_add_child_block, step 3')
                    add_node, ok = get_next_node_by_output(model, mm_node.output[0])
                    if ok == 0 and add_node.op_type == 'Add':
                        logger.debug('into handle_matmul_add_child_block, step 4')
                        rs_node2, ok = get_next_node_by_output(model, add_node.output[0])
                        if ok == 0 and rs_node2.op_type == 'Reshape':
                            tp_node2, ok = get_next_node_by_output(model, rs_node2.output[0])
                            if ok == 0 and tp_node2.op_type == 'Transpose':  
                                logger.debug('got match matmul+add child block~~')
                                node_dict['Transpose'] = node
                                node_dict['Transpose2'] = tp_node2
                                node_dict['MatMul'] = mm_node
                                node_dict['Add'] = add_node
                                node_dict['Reshape'] = rs_node
                                node_dict['Reshape2'] = rs_node2
                                node_list.append(node_dict)

    for nd in node_list:
        tp_node = nd['Transpose']
        tp_node2 = nd['Transpose2']
        add_node = nd['Add']
        mm_node = nd['MatMul']
        rs_node = nd['Reshape']
        rs_node2 = nd['Reshape2']

        ###add transpose
        ts_name = tp_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        shape = values.get_tensor_shape_by_name(model, rs_node.output[0])
        ts_output_shape = [shape[1], shape[0]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[rs_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[1,0])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs2_name = rs_node.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        shape = values.get_tensor_shape_by_name(model, rs_node.output[0])
        rs2_output_shape = [1, shape[1], 1, shape[0]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = rs_node.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node_insert1 = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs2_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, ts_node, rs_node)
        insert_node(model, rs_node_insert1, ts_node)
        
        mm_node.input[0] = rs2_output_name

        ###MatMul-->Conv
        inputB, shapeB = values.get_init_value_and_shape(model, mm_node.input[1])
        if isinstance(inputB, list) and inputB == []:
            logger.debug('-- inputB is not in initilizer')
            inputB = values.get_constant_value(model, input_next.input[1])

        mm_node.op_type = 'Conv'
        logger.debug('=== reuse MatMul to Conv')
        const_x_name = mm_node.name + '_to_conv_x_'

        v = inputB
        old_dims = [shapeB[0], shapeB[1]]
        dims_ = [shapeB[1], shapeB[0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++--- A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---+++ A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        operation.remove_initializer_if_necessary_by_name(model, mm_node.input[1], mm_node)                    

        model.graph.initializer.append(const_x_tensor)
        mm_node.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        mm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        mm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        mm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        mm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        mm_node.attribute.append(attr)        

        B = add_node.input[1]   

        mm_node.input.append(B)

        conv_output_shape = rs2_output_shape

        update_tensor_shape(model, mm_node.output[0], conv_output_shape)     

        #Add--->Reshape
        add_node.op_type = 'Reshape'

        del add_node.attribute[:]

        rs_name = add_node.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [conv_output_shape[1], conv_output_shape[2], conv_output_shape[3]]
        logger.debug('-----+++ rs_output_shape: {}'.format(rs_output_shape))

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add_node.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        add_node.input[1] = const_shape_name

        update_tensor_shape(model, add_node.output[0], rs_output_shape)

        #Reshape2
        operation.remove_onnx_node(model, rs_node2)

        #Transpose
        tp_node2.input[0] = add_node.output[0]

        del tp_node2.attribute[:]

        attr = onnx.helper.make_attribute('perm', [1, 2, 0])
        tp_node2.attribute.append(attr)

#Transpose-->Reshape-->Gemm-->Reshape-->Transpose
def handle_matmul_add_child_block2(model):
    logger.debug('into handle_matmul_add_child_block2')

    node_list = []

    for node in model.graph.node:
        node_dict = {}
        if node.op_type == 'Transpose':
            logger.debug('into handle_matmul_add_child_block2, step 1')
            rs_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and rs_node.op_type == 'Reshape':
                logger.debug('into handle_matmul_add_child_block2, step 2')
                gemm_node, ok = get_next_node_by_output(model, rs_node.output[0])
                if ok == 0 and gemm_node.op_type == 'Gemm':
                    logger.debug('into handle_matmul_add_child_block2, step 3')
                    rs_node2, ok = get_next_node_by_output(model, gemm_node.output[0])
                    if ok == 0 and rs_node2.op_type == 'Reshape':
                        tp_node2, ok = get_next_node_by_output(model, rs_node2.output[0])
                        if ok == 0 and tp_node2.op_type == 'Transpose':  
                            logger.debug('got match matmul+add child block~~')
                            node_dict['Transpose'] = node
                            node_dict['Transpose2'] = tp_node2
                            node_dict['Gemm'] = gemm_node
                            node_dict['Reshape'] = rs_node
                            node_dict['Reshape2'] = rs_node2
                            node_list.append(node_dict)

    for nd in node_list:
        tp_node = nd['Transpose']
        tp_node2 = nd['Transpose2']
        gemm_node = nd['Gemm']
        rs_node = nd['Reshape']
        rs_node2 = nd['Reshape2']

        ###add transpose
        ts_name = tp_node.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        shape = values.get_tensor_shape_by_name(model, rs_node.output[0])
        ts_output_shape = [shape[1], shape[0]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[rs_node.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[1,0])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs2_name = rs_node.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        shape = values.get_tensor_shape_by_name(model, rs_node.output[0])
        rs2_output_shape = [1, shape[1], 1, shape[0]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = rs_node.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node_insert1 = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs2_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, ts_node, rs_node)
        insert_node(model, rs_node_insert1, ts_node)
        
        gemm_node.input[0] = rs2_output_name

        ###Gemm-->Conv
        inputB, shapeB = values.get_init_value_and_shape(model, gemm_node.input[1])
        if isinstance(inputB, list) and inputB == []:
            logger.debug('-- inputB is not in initilizer')
            inputB = values.get_constant_value(model, input_next.input[1])

        gemm_node.op_type = 'Conv'
        logger.debug('=== reuse MatMul to Conv')
        const_x_name = gemm_node.name + '_to_conv_x_'

        transB = 0
        attributes = gemm_node.attribute
        for attr in attributes:
            #TBD
            '''
            if attr.name == 'alpha':
                alpha = attr.f
                logger.debug('alpha: {}'.format(alpha))
            
            if attr.name == 'beta':
                beta = attr.f
                logger.debug('beta: {}'.format(beta))

            if attr.name == 'transA':
                transA  = attr.i
                logger.debug('transA: {}'.format(transA))
            '''    

            if attr.name == 'transB':
                transB = attr.i
                logger.debug('got transB: {}'.format(transB)) 


        del gemm_node.attribute[:]

        v = inputB
        old_dims = [shapeB[0], shapeB[1]]
        dims_ = [shapeB[1], shapeB[0],1,1]

        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)

            if transB == 0:
                A = A.transpose()

            A = A.reshape(*dims_)
            logger.debug('+++--- A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)

            if transB == 0:
                A = A.transpose()

            A = A.reshape(*dims_)
            logger.debug('---+++ A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        operation.remove_initializer_if_necessary_by_name(model, gemm_node.input[1], gemm_node)                    

        model.graph.initializer.append(const_x_tensor)
        gemm_node.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        gemm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        gemm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        gemm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        gemm_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        gemm_node.attribute.append(attr)        

        #B = gemm_node.input[2]   
        #gemm_node.input.append(B)

        conv_output_shape = rs2_output_shape

        update_tensor_shape(model, gemm_node.output[0], conv_output_shape)     

        #Reshape2
        rs_output_shape = [conv_output_shape[1], conv_output_shape[2], conv_output_shape[3]]
        logger.debug('-----+++ rs_output_shape: {}'.format(rs_output_shape))

        const_shape_name = rs_node2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        operation.remove_initializer_if_necessary_by_name(model, rs_node2.input[1], rs_node2)

        rs_node2.input[1] = const_shape_name

        update_tensor_shape(model, rs_node2.output[0], rs_output_shape)

        #Transpose
        del tp_node2.attribute[:]

        attr = onnx.helper.make_attribute('perm', [1, 2, 0])
        tp_node2.attribute.append(attr)

def get_matmul_split_reshape_block_list(model):
    msr_list = []
    for node in model.graph.node:
        if node.op_type == 'MatMul':
            #print('got matmul:', node.name)
            inputB, shapeB = values.get_init_value_and_shape(model, node.input[1])

            if isinstance(inputB, list) and inputB == []:
                logger.debug('get_matmul_split_reshape_block_list, inputB is not in initilizer: {}'.format(node.name))
                inputB = values.get_constant_value(model, node.input[1])

            if len(inputB) > 0:
                #print('--got matmul:', node.name)
                split_node, ok = values.get_next_node_by_output(model, node.output[0])
                if ok == 0 and split_node.op_type == 'Split':
                    print('--got matmul:', node.name)
                    if len(split_node.output) == 3:
                        match_reshape_node_cnt = 0
                        msr = {}
                        for idx, output in enumerate(split_node.output):
                            next_node, ok = values.get_next_node_by_output(model, output)
                            if ok == 0 and next_node.op_type == 'Reshape':
                                msr['Reshape_' + str(idx)] = next_node
                                match_reshape_node_cnt = match_reshape_node_cnt + 1

                        if match_reshape_node_cnt == 3:
                            msr['MatMul'] = node
                            msr['Split'] = split_node
                            msr['inputB'] = inputB
                            msr['shapeB'] = shapeB

                            for idx in range(3):
                                rs_node = msr['Reshape_' + str(idx)]
                                ts_node, _ = get_next_node_by_output(model, rs_node.output[0])
                                mm_node, _ = get_next_node_by_output(model, ts_node.output[0])
                                ts_node2, _ = get_next_node_by_output(model, mm_node.output[0])
                                if ts_node2.op_type == 'Transpose':
                                    msr['MatMul_QK_V'] = mm_node
                                    msr['Transpose_V'] = ts_node
                                    msr['Reshape_V'] = msr.pop('Reshape_' + str(idx))
                                    msr['Softmax'], _ = get_prev_node_by_input(model, mm_node.input[0])
                                else:
                                    msr['MatMul_QK'] = mm_node
                                    msr['Mul'] = ts_node2
                                    if ts_node.output[0] == mm_node.input[0]:
                                        msr['Transpose_Q'] = ts_node
                                        msr['Reshape_Q'] = msr.pop('Reshape_' + str(idx))
                                    else:
                                        msr['Transpose_K'] = ts_node
                                        msr['Reshape_K'] = msr.pop('Reshape_' + str(idx)) 

                            msr_list.append(msr)

    return msr_list                       

def get_matmul_add_block_list(model):
    ma_list = []

    for node in model.graph.node:
        if node.op_type == 'MatMul':
            inputB, shapeB = values.get_init_value_and_shape(model, node.input[1])

            if isinstance(inputB, list) and inputB == []:
                logger.debug('get_matmul_add_block_list, inputB is not in initilizer: {}'.format(node.name))
                inputB = values.get_constant_value(model, node.input[1])

            if len(inputB) > 0:
                #print('--got matmul:', node.name)
                add_node, ok = values.get_next_node_by_output(model, node.output[0])
                if ok == 0 and add_node.op_type == 'Add':
                    addA,shapeAddA = values.get_init_value_and_shape(model, add_node.input[0])
                    if isinstance(addA, list) and addA == []:
                        logger.debug('addA is not in initilizer')
                        addA = values.get_constant_value(model, add_node.input[0])

                    matmul_output_shape = values.get_tensor_shape_by_name(model, node.output[0])
                    if len(addA) > 0 and len(shapeAddA) == 1 and shapeAddA[0] == matmul_output_shape[-1]:
                        logger.debug('got matmul+add block')
                        ma = {}
                        ma['Add'] = add_node
                        ma['MatMul'] = node
                        ma['inputB'] = inputB
                        ma['shapeB'] = shapeB
                        ma_list.append(ma)

    return ma_list  

def handle_matmul_pattern_six_old(model, matmul_dict, index):
    matmul_node = matmul_dict['MatMul']
    next_node, _ = get_next_node_by_output(model, matmul_node.output[0])

    prev_node, ok = values.get_prev_node_by_input(model, matmul_node.input[0])
    if ok == 0:
        matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_node.input[0])
        ###add transpose
        tp_name = matmul_node.name + '_transpose_' + str(index)
        tp_output_name = tp_name + '_output_'
        tp_output_shape = [matmul_input0_shape[0], matmul_input0_shape[2], matmul_input0_shape[1]]
        transpose_output = onnx.helper.make_tensor_value_info(tp_output_name, onnx.TensorProto.FLOAT, tp_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=tp_name,
                                            inputs=[matmul_node.input[0]],
                                            outputs=[tp_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose_output) 

        insert_node(model, ts_node, prev_node)

        ###add reshape
        rs_name = matmul_node.name + '_reshape_' + str(index)
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [matmul_input0_shape[0], matmul_input0_shape[2], 1, matmul_input0_shape[1]]
        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)
        
        const_shape_name = matmul_node.name + '_reshape_data_' + str(index)
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[tp_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.initializer.append(const_shape_tensor)
        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node, ts_node)

        #MatMul--->Conv
        matmul_node.op_type = 'Conv'
        matmul_node.input[0] = rs_output_name

        logger.debug('-----reuse MatMul to Conv: {}'.format(matmul_node.name))
        const_x_name = matmul_node.name + '_to_conv_x_' + str(index)

        v = matmul_dict['inputB']
        old_dims = [matmul_dict['shapeB'][0], matmul_dict['shapeB'][1]]
        dims_ = [matmul_dict['shapeB'][1], matmul_dict['shapeB'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul_node.input[1] = const_x_name

        if 'Add' in matmul_dict.keys():
            matmul_node.input.append(matmul_dict['Add'].input[0])

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul_node.attribute.append(attr)        

        output_shape = values.get_tensor_shape_by_name(model, matmul_node.output[0])
        conv_output_shape = [output_shape[0], output_shape[2], 1, output_shape[1]]#[1, output_shape[1], 1, output_shape[0]] 

        update_tensor_shape(model, matmul_node.output[0], conv_output_shape)

        #add Reshape2
        rs2_name = matmul_node.name + '_reshape_2_' + str(index)
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [output_shape[0], output_shape[2], output_shape[1]]
        logger.debug('-----rs_output_shape: {}'.format(rs2_output_shape))

        rs2_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name2 = matmul_node.name + '_reshape_data2_' + str(index)
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name2,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        rs2_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[matmul_node.output[0], const_shape_name2],
                                            outputs=[rs2_output_name])

        model.graph.initializer.append(const_shape_tensor)
        model.graph.value_info.append(rs2_output)

        insert_node(model, rs2_node, matmul_node)

        ########add transpose2
        tp2_name = matmul_node.name + '_transpose2_' + str(index)
        tp2_output_name = tp2_name + '_output_'
        tp2_output_shape = [rs2_output_shape[0], rs2_output_shape[2], rs2_output_shape[1]]

        transpose2_output = onnx.helper.make_tensor_value_info(tp2_output_name, onnx.TensorProto.FLOAT, tp2_output_shape)
        
        ts2_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=tp2_name,
                                            inputs=[rs2_output_name],
                                            outputs=[tp2_output_name],
                                            perm=[0,2,1])

        model.graph.value_info.append(transpose2_output) 

        insert_node(model, ts2_node, rs2_node)

        if 'Add' in matmul_dict.keys():
            add_node = matmul_dict['Add']
            all_next_nodes, _ = get_all_next_node_by_output(model, add_node.output[0])

            for n in all_next_nodes:
                for idx, input_ in enumerate(n.input):
                    if input_ == add_node.output[0]:
                        n.input[idx] = tp2_output_name
                        break

            operation.remove_onnx_node(model, add_node)
        else:    
            next_node.input[0] = tp2_output_name

def handle_matmul_pattern_six(model, matmul_dict):
    matmul_node = matmul_dict['MatMul']
    split_node = matmul_dict['Split']

    split_node.input[0] = matmul_node.input[0]
    split_node.op_type = 'Transpose'

    del split_node.attribute[:]

    attr = onnx.helper.make_attribute('perm', [0,2,1])
    split_node.attribute.append(attr)

    matmul_input_shape = values.get_tensor_shape_by_name(model, matmul_node.input[0])
    ts_output_shape = [matmul_input_shape[0], matmul_input_shape[2], matmul_input_shape[1]]

    update_tensor_shape(model, split_node.output[0], ts_output_shape)

    del split_node.output[1:]
    del split_node.input[1]

    ####
    inputB = matmul_dict['inputB']
    shapeB = matmul_dict['shapeB']

    v = inputB
    old_dims = [shapeB[0], shapeB[1]]
    dims_ = [shapeB[1], shapeB[0], 1, 1]

    if isinstance(v, np.ndarray) == True:
        A = v.reshape(*old_dims)
        A = A.transpose()
        A = A.reshape(*dims_)
        logger.debug('+++A.shape: {}'.format(A.shape))
        #A = A.flatten()
    else:    
        A = np.array(v).reshape(*old_dims)
        A = A.transpose()
        A = A.reshape(*dims_)
        logger.debug('---A.shape: {}'.format(A.shape))
        #A = A.flatten()

    ########################
    ##########################
    ###path Q
    rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]
    const_shape_name = matmul_node.output[0] + '_reshape_data3_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    rs_node = matmul_dict['Reshape_Q']

    operation.remove_initializer_if_necessary_by_name(model, rs_node.input[1], rs_node)

    rs_node.input[0] = split_node.output[0]
    rs_node.input[1] = const_shape_name
    update_tensor_shape(model, rs_node.output[0], rs_output_shape)

    #Squeeze--->Conv
    ts_q = matmul_dict['Transpose_Q']
    #ts_q.input[0] = rs_output_name
    ts_q.op_type = 'Conv'
    logger.debug('-----reuse Squeeze to Conv')
    const_w_name = ts_q.name + '_to_conv_w_'
    const_b_name = ts_q.name + '_to_conv_b_'

    index_end = int(dims_[0]/3)
    weight = A[0:index_end,:,:,:]

    print('weight.shape:', weight.shape)

    weight = weight.flatten()
    weight = weight.tolist()

    print('weight.len:', len(weight))

    const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                        vals=weight)

    model.graph.initializer.append(const_w_tensor)

    ts_q.input.append(const_w_name) 

    del ts_q.attribute[:]

    attr = onnx.helper.make_attribute('dilations', [1, 1])
    ts_q.attribute.append(attr)

    attr = onnx.helper.make_attribute('group', 1)
    ts_q.attribute.append(attr)

    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
    ts_q.attribute.append(attr)

    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
    ts_q.attribute.append(attr)

    attr = onnx.helper.make_attribute('strides', [1,1])
    ts_q.attribute.append(attr)        

    conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

    ts_q_output_shape = values.get_tensor_shape_by_name(model, ts_q.output[0])

    update_tensor_shape(model, ts_q.output[0], conv_output_shape)

    ###add reshape-2
    rs_name = ts_q.output[0] + '_reshape_2_'
    rs_output_name = rs_name + '_output_'
    rs_output_shape = [ts_q_output_shape[0], ts_q_output_shape[1], int(conv_output_shape[1]/ts_q_output_shape[1]),conv_output_shape[3]]

    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

    const_shape_name = ts_q.output[0] + '_reshape_data2_'
    print('222 const_shape_name', const_shape_name)
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    rs_node = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs_name,
                                        inputs=[ts_q.output[0], const_shape_name],
                                        outputs=[rs_output_name],
                                        allowzero=0)

    model.graph.initializer.append(const_shape_tensor)

    model.graph.value_info.append(rs_output)

    insert_node(model, rs_node, ts_q)

    matmul_qk = matmul_dict['MatMul_QK']

    matmul_qk.input[1] = rs_output_name

    ###################################
    ########################################
    ###Path K
    rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]
    const_shape_name = matmul_node.output[0] + '_reshape_data4_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    rs_node = matmul_dict['Reshape_K']

    operation.remove_initializer_if_necessary_by_name(model, rs_node.input[1], rs_node)

    rs_node.input[0] = split_node.output[0]
    rs_node.input[1] = const_shape_name
    update_tensor_shape(model, rs_node.output[0], rs_output_shape)

    #Squeeze--->Conv
    ts_k = matmul_dict['Transpose_K']
    #ts_k.input[0] = rs_output_name
    ts_k.op_type = 'Conv'
    logger.debug('-----reuse Squeeze to Conv')
    const_w_name = ts_k.name + '_to_conv_w_'
    const_b_name = ts_k.name + '_to_conv_b_'

    index_begin = int(dims_[0]/3)
    index_end = int(2*dims_[0]/3)
    weight = A[index_begin:index_end,:,:,:]

    print('weight.shape:', weight.shape)

    weight = weight.flatten()
    weight = weight.tolist()

    print('weight.len:', len(weight))

    const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                        vals=weight)

    model.graph.initializer.append(const_w_tensor)

    ts_k.input.append(const_w_name) 

    del ts_k.attribute[:]

    attr = onnx.helper.make_attribute('dilations', [1, 1])
    ts_k.attribute.append(attr)

    attr = onnx.helper.make_attribute('group', 1)
    ts_k.attribute.append(attr)

    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
    ts_k.attribute.append(attr)

    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
    ts_k.attribute.append(attr)

    attr = onnx.helper.make_attribute('strides', [1,1])
    ts_k.attribute.append(attr)        

    conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

    ts_k_output_shape = values.get_tensor_shape_by_name(model, ts_k.output[0])

    update_tensor_shape(model, ts_k.output[0], conv_output_shape)

    ###add reshape-2
    rs_name = ts_k.output[0] + '_reshape_2_'
    rs_output_name = rs_name + '_output_'
    rs_output_shape = [ts_k_output_shape[0], ts_k_output_shape[1], int(conv_output_shape[1]/ts_k_output_shape[1]),conv_output_shape[3]]

    rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

    const_shape_name = ts_k.output[0] + '_reshape_data2_'
    print('333 const_shape_name', const_shape_name)

    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    rs_node = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs_name,
                                        inputs=[ts_k.output[0], const_shape_name],
                                        outputs=[rs_output_name])

    attr = onnx.helper.make_attribute('allowzero', 0)
    rs_node.attribute.append(attr)

    model.graph.initializer.append(const_shape_tensor)

    model.graph.value_info.append(rs_output)

    insert_node(model, rs_node, ts_k)

    ###path V
    rs_node = matmul_dict['Reshape_V']
    rs_output_shape = [ts_output_shape[0], ts_output_shape[1], 1, ts_output_shape[2]]

    const_shape_name = rs_node.output[0] + '_reshape_data_'
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    model.graph.initializer.append(const_shape_tensor)

    operation.remove_initializer_if_necessary_by_name(model, rs_node.input[1], rs_node)

    rs_node.input[0] = split_node.output[0]
    rs_node.input[1] = const_shape_name
    update_tensor_shape(model, rs_node.output[0], rs_output_shape)

    #Squeeze--->Conv
    ts_v = matmul_dict['Transpose_V']
    #ts_v.input[0] = rs_output_name
    ts_v.op_type = 'Conv'
    logger.debug('-----reuse Squeeze to Conv')
    const_w_name = ts_v.name + '_to_conv_w_'
    const_b_name = ts_v.name + '_to_conv_b_'

    index = int(2*dims_[0]/3)
    weight = A[index:,:,:,:]

    print('weight.shape:', weight.shape)

    weight = weight.flatten()
    weight = weight.tolist()

    print('weight.len:', len(weight))

    const_w_tensor = onnx.helper.make_tensor(name=const_w_name,
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[int(shapeB[1]/3), shapeB[0], 1, 1],
                        vals=weight)


    model.graph.initializer.append(const_w_tensor)

    ts_v.input.append(const_w_name) 

    del ts_v.attribute[:]

    attr = onnx.helper.make_attribute('dilations', [1, 1])
    ts_v.attribute.append(attr)

    attr = onnx.helper.make_attribute('group', 1)
    ts_v.attribute.append(attr)

    attr = onnx.helper.make_attribute('kernel_shape', [1,1])
    ts_v.attribute.append(attr)

    attr = onnx.helper.make_attribute('pads', [0,0,0,0])
    ts_v.attribute.append(attr)

    attr = onnx.helper.make_attribute('strides', [1,1])
    ts_v.attribute.append(attr)        

    conv_output_shape = [rs_output_shape[0], int(shapeB[1]/3), rs_output_shape[2], rs_output_shape[3]]

    ts_v_output_shape = values.get_tensor_shape_by_name(model, ts_v.output[0])

    update_tensor_shape(model, ts_v.output[0], conv_output_shape)

    ###add reshape-2
    rs_nameV = ts_v.output[0] + '_reshape_2_'
    rs_output_nameV = rs_nameV + '_output_'
    rs_output_shape = [ts_v_output_shape[0], ts_v_output_shape[1], int(conv_output_shape[1]/ts_v_output_shape[1]),conv_output_shape[3]]

    rs_output = onnx.helper.make_tensor_value_info(rs_output_nameV, onnx.TensorProto.FLOAT, rs_output_shape)

    const_shape_nameV = ts_v.output[0] + '_reshape_data2_'
    print('111 const_shape_name', const_shape_nameV)
    
    const_shape_tensor = onnx.helper.make_tensor(name=const_shape_nameV,
                        data_type=onnx.TensorProto.INT64,
                        dims=[len(rs_output_shape)],
                        vals=rs_output_shape)

    rs_node = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs_nameV,
                                        inputs=[ts_v.output[0], const_shape_nameV],
                                        outputs=[rs_output_nameV],
                                        allowzero=0)

    model.graph.initializer.append(const_shape_tensor)                                    

    model.graph.value_info.append(rs_output)

    insert_node(model, rs_node, ts_v)

    matmul_qk_v = matmul_dict['MatMul_QK_V']

    matmul_qk_v.input[0] = rs_output_nameV

#################
    ts_name = rs_output_name + '_transpose_'
    ts_output_name = ts_name + '_output_'

    ts_node = onnx.helper.make_node(
                                        'Transpose',
                                        name=ts_name,
                                        inputs=[rs_output_name],
                                        outputs=[ts_output_name],
                                        perm=[0,1,3,2])

    ts_output_shape = [rs_output_shape[0], rs_output_shape[1], rs_output_shape[3], rs_output_shape[2]]
    transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
    model.graph.value_info.append(transpose_output)

    insert_node(model, ts_node, rs_node)

    matmul_qk.input[0] = ts_output_name

    ##################
    mul_node = matmul_dict['Mul']
    ts_name = mul_node.name + '_transpose_'
    ts_output_name = ts_name + '_output_'

    ts_node = onnx.helper.make_node(
                                        'Transpose',
                                        name=ts_name,
                                        inputs=[mul_node.output[0]],
                                        outputs=[ts_output_name],
                                        perm=[0,1,3,2])

    mul_output_shape = values.get_tensor_shape_by_name(model, mul_node.output[0])
    transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, mul_output_shape)
    model.graph.value_info.append(transpose_output)

    insert_node(model, ts_node, mul_node)
    ########################

    softmax_node = matmul_dict['Softmax']#get_prev_node_by_input(model, matmul_qk_v.input[1])#matmul_dict['Softmax']
    where_node, _ = get_prev_node_by_input(model, softmax_node.input[0])
    where_node.input[2] = ts_output_name

    ts2_name = softmax_node.name + '_transpose_'
    ts2_output_name = ts2_name + '_output_'

    ts2_node = onnx.helper.make_node(
                                        'Transpose',
                                        name=ts2_name,
                                        inputs=[softmax_node.output[0]],
                                        outputs=[ts2_output_name],
                                        perm=[0,1,3,2])

    transpose_output2 = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, mul_output_shape)
    model.graph.value_info.append(transpose_output2)

    insert_node(model, ts2_node, softmax_node)

    matmul_qk_v.input[1] = ts2_output_name

    matmul_v_output_shape = values.get_tensor_shape_by_name(model, matmul_qk_v.output[0])
    update_tensor_shape(model, matmul_qk_v.output[0], [matmul_v_output_shape[0], matmul_v_output_shape[1], matmul_v_output_shape[3], matmul_v_output_shape[2]])

    #handle_trma(model, matmul_v)

    next_ts_node, _ = get_next_node_by_output(model, matmul_qk_v.output[0])
    del next_ts_node.attribute[:]

    attr = onnx.helper.make_attribute('perm', [0,3,1,2])
    next_ts_node.attribute.append(attr)

    operation.remove_initializer_if_necessary_by_name(model, matmul_node.input[1], matmul_node)
    operation.remove_onnx_node(model, matmul_node)

def handle_pattern_six(model):
    msr_list = get_matmul_split_reshape_block_list(model)
    for idx, msr in enumerate(msr_list):
        handle_matmul_pattern_six(model, msr)

    '''
    ma_list = get_matmul_add_block_list(model)
    for idx, ma in enumerate(ma_list):
        handle_matmul_pattern_six(model, ma, idx)
    '''
    return model    

def mha_optimizer(model):
    pattern = -1
    ret1 = match_mha_block_pattern_one(model) #for decoder_model_bs10.onnx
    ret2 = match_mha_block_pattern_two(model) #for bert_cls_sim1.onnx/bert_sst2_wm.onnx
    ret3 = match_mha_block_pattern_three(model) #for bert_sst2_sim.onnx
    ret4 = match_mha_block_pattern_four(model) #for bert_squad_v1_sim1.onnx
    ret5 = match_mha_block_pattern_five(model) #for 
    ret6 = match_mha_block_pattern_six(model) #for platerecognition_model_v5.1.3_sim2
    ret7 = match_mha_block_pattern_seven(model) #for deit_small_patch16_224.onnx
    ret8, node_block_list = match_mha_block_pattern_eight(model) #for swin_tiny_patch4_window7_224.onnx

    if ret1 == 0:
        pattern = 1
    elif ret2 == 0:
        pattern = 2    
    elif ret3 == 0:
        pattern = 3
    elif ret4 == 0:
        pattern = 4
    elif ret5 == 0:
        pattern = 5
    elif ret6 == 0:
        pattern = 6
    elif ret7 == 0:
        pattern = 7
    elif ret8 == 0:
        pattern = 8   

    if pattern == -1:
        logger.debug('This is not a mha model---')
        return model 

    if pattern == 6:
        return handle_pattern_six(model) 

    if pattern == 5:
        matm_list = get_mul_add_transpose_matmul_block(model)
        for matm in matm_list:
            print('got matm~')
            matm['mm1'].input[0] = matm['Add'].output[0]
            matm['mm2'].input[0] = matm['Add'].output[0]
            matm['mm3'].input[0] = matm['Add'].output[0]

            #model.graph.node.remove(matm['Tp'])
            operation.remove_onnx_node(model, matm['Tp'])  

        del model.graph.value_info[:]
        model = onnx.shape_inference.infer_shapes(model)
        model = onnx.shape_inference.infer_shapes(model)
        gen_mul_add_block_by_rm_transpose(model)
        gen_mul_add_block_by_rm_transpose2(model)
        handle_matmul_add_child_block(model)
        handle_matmul_add_child_block2(model)

    matmul_list = []

    logger.debug('mha_optimizer, pattern = {}'.format(pattern))

    if pattern == 1:
        handle_add_combination_pattern_one(model)
    elif pattern == 2 or pattern == 3:   
        handle_add_combination_pattern_two_three(model)
    elif pattern == 7:   
        handle_add_combination_pattern_seven(model)
        handle_mul_add_block(model, pattern)
        handle_split_block_pattern_seven(model)
        return model   
    elif pattern == 8:
        handle_add_combination_pattern_eight(model)   
        handle_mul_add_block(model, pattern)
        handle_split_block_pattern_eight(model, node_block_list)
        handle_ars4cr(model)
        handle_msra_block(model)
        return model   

    if pattern != 4:   
        handle_mul_add_block(model, pattern)
    else:
        logger.debug('handle pattern 4')
        handle_add_combination_pattern_four(model)    
        handle_mul_add_block_pattern_four(model)

    for node in model.graph.node:
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #        ", op:", node.op_type)
        if node.op_type == 'MatMul':
            inputA = node.input[0]
            inputB = node.input[1]

            is_init = False

            for init in model.graph.initializer:
                if init.name == inputA or init.name == inputB:
                    is_init = True
                    break

            if is_init == False:
                dataA = values.get_constant_value(model, inputA)
                dataB = values.get_constant_value(model, inputB)

                if dataA != [] or dataB != []:
                    is_init = True

            if is_init == True:
                logger.debug('skip MatMul: {}'.format(node.name))
                continue

            matmul_dict = {}

            mul_node, ok = get_prev_node_by_input(model, inputA)
            if ok == 0 and mul_node.op_type == 'Mul':
                mulB = values.get_init_value(model, mul_node.input[1])
                logger.debug('matmul input is Mul: {}'.format(mul_node.name, mulB[0]))

                if isinstance(mulB, list) and mulB == []:
                    logger.debug('mulB is not in initilizer')
                    mulB = values.get_constant_value(model, mul_node.input[1])

                if len(mulB) > 0 and abs(mulB[0] - 0.125) < 0.00001:
                    logger.debug('this is the mul-node which we wanted(value B is 0.125)...')
                    matmul_dict['AMul'] = mul_node
                    inputA = mul_node.input[0]

            div_node, ok = get_prev_node_by_input(model, inputA)
            if ok == 0 and div_node.op_type == 'Div':
                divB = values.get_init_value(model, div_node.input[1])
                logger.debug('matmul input is Div: {}'.format(div_node.name, divB[0]))

                if isinstance(divB, list) and mulB == []:
                    logger.debug('divB is not in initilizer')
                    divB = values.get_constant_value(model, div_node.input[1])

                if len(divB) > 0 and abs(divB[0] - 8.0) < 0.00001:
                    logger.debug('this is the div-node which we wanted(value B is 8)...')
                    matmul_dict['AMul'] = div_node
                    inputA = div_node.input[0]

            node_dictA, res1 = get_matmul_input_path_pattern_one(model, inputA)
            node_dictB, res2 = get_matmul_input_path_pattern_one(model, inputB)

            if res1 > -1:
                print_matmul_input_path(node_dictA['node_list'], 'node_listA')
            if res2 > -1:
                print_matmul_input_path(node_dictB['node_list'], 'node_listB')

            if res1 > -1 or res2 > -1:
                next_node, _ = get_next_node_by_output(model, node.output[0])
                nnext_node, _ = get_next_node_by_output(model, next_node.output[0])

                #matmul_dict = {}
                matmul_dict['name'] = node.name
                matmul_dict['current'] = node
                matmul_dict['pathA'] = node_dictA['node_list']
                matmul_dict['A_MatMul_Add'] = False
                if res1 == 0:
                    matmul_dict['A_MatMul_Add'] = True
                    matmul_dict['A_addA'] = node_dictA['addA']
                    matmul_dict['A_matmul_AShape'] = node_dictA['matmul_AShape']
                    matmul_dict['A_inputB'] = node_dictA['inputB']
                    matmul_dict['A_matmul_BShape'] = node_dictA['matmul_BShape']
                    if 'prev' in node_dictA.keys():
                        matmul_dict['A_prev'] = node_dictA['prev']

                matmul_dict['pathB'] = node_dictB['node_list']
                matmul_dict['B_MatMul_Add'] = False
                if res2 == 0:
                    matmul_dict['B_MatMul_Add'] = True
                    matmul_dict['B_addA'] = node_dictB['addA']
                    matmul_dict['B_matmul_AShape'] = node_dictB['matmul_AShape']
                    matmul_dict['B_inputB'] = node_dictB['inputB']
                    matmul_dict['B_matmul_BShape'] = node_dictB['matmul_BShape']
                    if 'prev' in node_dictB.keys():
                        matmul_dict['B_prev'] = node_dictB['prev']

                matmul_dict['next'] = [next_node]
                matmul_dict['nnext'] = [nnext_node]
                matmul_list.append(matmul_dict)      

    for ll in matmul_list:
        logger.debug('stat MatMul: {}, next: {}, op_type: {}'.format(ll['name'], ll['next'][0].name,ll['next'][0].op_type))
        logger.debug('------pathA:')
        for node in ll['pathA']:
            logger.debug('    {}'.format(node.name))

        logger.debug('------pathB:')
        for node in ll['pathB']:
            logger.debug('    {}'.format(node.name))

        cvt_matmul_add_to_conv(model, ll, pattern)

    if pattern == 1:
        handle_mul_add_block_two(model)

    if pattern == 4:
        pass
        #handle_last_group_pattern_four(model)
    else:
        handle_last_group(model)

    return model

def match_mha_block_common(model):
    common_dict = {}

    logger.debug('into match_mha_block_common')

    for node in model.graph.node:
        if node.op_type == 'Add':
            mul_node, ok = get_prev_node_by_input(model, node.input[0])
            if ok == 0 and mul_node.op_type == 'Mul':
                div_node = None
                div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])
                if ok1 == 0 and div_node_case1.op_type == 'Div':
                    div_node = div_node_case1
                elif ok2 == 0 and div_node_case2.op_type == 'Div':
                    div_node = div_node_case2

                #logger.debug('into match_mha_block_common, step 1')    

                if div_node != None:
                    sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                    sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])
                    if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                        add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            #logger.debug('into match_mha_block_common, step 2')
                            rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                            if ok == 0 and rm_node.op_type == 'ReduceMean':
                                pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                if ok == 0 and pow_node.op_type == 'Pow':
                                    #logger.debug('into match_mha_block_common, step 3')
                                    sub_node, ok = get_prev_node_by_input(model, pow_node.input[0])
                                    if ok == 0 and sub_node.op_type == 'Sub':
                                        add_node, ok1 = get_prev_node_by_input(model, sub_node.input[0])
                                        rm_node, ok2 = get_prev_node_by_input(model, sub_node.input[1])
                                        if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                            #logger.debug('into match_mha_block_common, step 4')
                                            add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                            if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                add_node_1, ok1 = get_prev_node_by_input(model, add_node_.input[0])
                                                add_node_2, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                #logger.debug('into match_mha_block_common, step 5')
                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                    mm_node = None
                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_2.input[0])
                                                    mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_2.input[1])
                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                        mm_node = mm_node_case1
                                                    elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                        mm_node = mm_node_case2

                                                    if mm_node == None:
                                                        mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_1.input[0])
                                                        mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_1.input[1])
                                                        if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                            mm_node = mm_node_case1
                                                        elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                            mm_node = mm_node_case2

                                                    logger.debug('into match_mha_block_common, step 6')
                                                    if mm_node != None:
                                                        #logger.debug('into match_mha_block_common, step 6.1')
                                                        mul_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                        if ok == 0 and mul_node.op_type == 'Mul':
                                                            #logger.debug('into match_mha_block_common, step 6.2') 
                                                            mul_node_2, ok = get_prev_node_by_input(model, mul_node.input[0])
                                                            if ok == 0 and mul_node_2.op_type == 'Mul':
                                                                add_node_1, ok1 = get_prev_node_by_input(model, mul_node_2.input[0])
                                                                add_node_2, ok2 = get_prev_node_by_input(model, mul_node_2.input[1])

                                                                #logger.debug('into match_mha_block_common, step 7')

                                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                                    erf_node, ok = get_prev_node_by_input(model, add_node_2.input[0])
                                                                    if ok == 0 and erf_node.op_type == 'Erf':
                                                                        #logger.debug('into match_mha_block_common, step 8')
                                                                        div_node, ok = get_prev_node_by_input(model, erf_node.input[0])
                                                                        if ok == 0 and div_node.op_type == 'Div':
                                                                            #logger.debug('into match_mha_block_common, step 9')
                                                                            add_node, ok = get_prev_node_by_input(model, div_node.input[0])
                                                                            if ok == 0 and add_node.op_type == 'Add' and add_node == add_node_1:
                                                                                #logger.debug('into match_mha_block_common, step 10')
                                                                                mm_node = None
                                                                                mm_node_case1, ok1 = get_prev_node_by_input(model, add_node.input[0])
                                                                                mm_node_case2, ok2 = get_prev_node_by_input(model, add_node.input[1])

                                                                                if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                    mm_node = mm_node_case1
                                                                                elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                    mm_node = mm_node_case2

                                                                                if mm_node != None:
                                                                                    #logger.debug('into match_mha_block_common, step 11')
                                                                                    add_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                    if ok == 0 and add_node.op_type == 'Add':
                                                                                        mul_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                        if ok == 0 and mul_node.op_type == 'Mul':
                                                                                            div_node = None
                                                                                            div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                                                                                            div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])

                                                                                            if ok1 == 0 and div_node_case1.op_type == 'Div':
                                                                                                div_node = div_node_case1
                                                                                            elif ok2 == 0 and div_node_case2.op_type == 'Div':
                                                                                                div_node = div_node_case2

                                                                                            if div_node != None:
                                                                                                #logger.debug('into match_mha_block_common, step 12')
                                                                                                sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                                                                                                sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])

                                                                                                if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                                                                                                    add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                                                                                                    if ok == 0 and add_node.op_type == 'Add':
                                                                                                        #logger.debug('into match_mha_block_common, step 13')
                                                                                                        rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                                        if ok == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                            #logger.debug('into match_mha_block_common, step 14')
                                                                                                            pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                            if ok == 0 and pow_node.op_type == 'Pow':
                                                                                                                #logger.debug('into match_mha_block_common, step 15')
                                                                                                                sub_node_, ok = get_prev_node_by_input(model, pow_node.input[0])
                                                                                                                if ok == 0 and sub_node_.op_type == 'Sub' and sub_node_ == sub_node:
                                                                                                                    add_node, ok1 = get_prev_node_by_input(model, sub_node_.input[0])
                                                                                                                    rm_node, ok2 = get_prev_node_by_input(model, sub_node_.input[1]) 

                                                                                                                    #logger.debug('into match_mha_block_common, step 16')
                                                                                                                    if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                                        add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                                        if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                                                                                            logger.debug('into match_mha_block_common, step 17')
                                                                                                                            add_node_1, ok1 = get_prev_node_by_input(model, add_node_.input[0])                                                                                   
                                                                                                                            add_node_2, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                                                                                            if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                                                                                                mm_node = None
                                                                                                                                mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_2.input[0])
                                                                                                                                mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_2.input[1])

                                                                                                                                if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                    mm_node = mm_node_case1
                                                                                                                                elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                    mm_node = mm_node_case2

                                                                                                                                if mm_node == None:
                                                                                                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_1.input[0])
                                                                                                                                    mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_1.input[1])

                                                                                                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                        mm_node = mm_node_case1
                                                                                                                                    elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                        mm_node = mm_node_case2

                                                                                                                                if mm_node != None:
                                                                                                                                    logger.debug('into match_mha_block_common, step 18')
                                                                                                                                    reshape_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                                                                    if ok == 0 and reshape_node.op_type == 'Reshape':
                                                                                                                                        tp_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                                                                                                                        if ok == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                            mm_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                                                                            if ok == 0 and mm_node.op_type == 'MatMul':
                                                                                                                                                softmax_node, ok1 = get_prev_node_by_input(model, mm_node.input[0])                                                                                   
                                                                                                                                                tp_node, ok2 = get_prev_node_by_input(model, mm_node.input[1])

                                                                                                                                                if ok1 == 0 and softmax_node.op_type == 'Softmax' and ok2 == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                                    reshape_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                                                                                    if ok == 0 and reshape_node.op_type == 'Reshape':
                                                                                                                                                        add_node_branch1, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                                                                                                                                        if ok == 0 and add_node_branch1.op_type == 'Add':
                                                                                                                                                            mm_node = None
                                                                                                                                                            mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_branch1.input[0])
                                                                                                                                                            mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_branch1.input[1])

                                                                                                                                                            if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                                                mm_node = mm_node_case1
                                                                                                                                                            elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                                                mm_node = mm_node_case2

                                                                                                                                                            if mm_node != None:
                                                                                                                                                                add_node_last, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                                                                                                if ok == 0 and add_node_last.op_type == 'Add':
                                                                                                                                                                    add_node_common, ok = get_prev_node_by_input(model, softmax_node.input[0])
                                                                                                                                                                    if ok == 0 and add_node_common.op_type == 'Add':
                                                                                                                                                                        common_dict['add_node_last'] = add_node_last
                                                                                                                                                                        common_dict['add_node_common'] = add_node_common
                                                                                                                                                                        logger.debug('got common mha block')
                                                                                                                                                                        break
    return common_dict

def match_mha_block_common_two(model):
    ret = -1
    logger.debug('into match_mha_block_common_two')

    for node in model.graph.node:
        if node.op_type == 'Add':
            mul_node, ok = get_prev_node_by_input(model, node.input[0])
            if ok == 0 and mul_node.op_type == 'Mul':
                div_node = None
                div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])
                if ok1 == 0 and div_node_case1.op_type == 'Div':
                    div_node = div_node_case1
                elif ok2 == 0 and div_node_case2.op_type == 'Div':
                    div_node = div_node_case2

                logger.debug('into match_mha_block_common_two, step 1 {}'.format(mul_node.name))    

                if div_node != None:
                    sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                    sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])
                    if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                        add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            #logger.debug('into match_mha_block_common_two, step 2')
                            rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                            if ok == 0 and rm_node.op_type == 'ReduceMean':
                                pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                if ok == 0 and pow_node.op_type == 'Pow':
                                    #logger.debug('into match_mha_block_common_two, step 3')
                                    sub_node, ok = get_prev_node_by_input(model, pow_node.input[0])
                                    if ok == 0 and sub_node.op_type == 'Sub':
                                        add_node, ok1 = get_prev_node_by_input(model, sub_node.input[0])
                                        rm_node, ok2 = get_prev_node_by_input(model, sub_node.input[1])
                                        if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                            #logger.debug('into match_mha_block_common_two, step 4')
                                            add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                            if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                add_node_1, ok1 = get_prev_node_by_input(model, add_node_.input[0])
                                                add_node_2, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                logger.debug('into match_mha_block_common_two, step 5 {}'.format(add_node_.name))
                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                    mm_node = None
                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_2.input[0])
                                                    mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_2.input[1])
                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                        mm_node = mm_node_case1
                                                    elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                        mm_node = mm_node_case2

                                                    if mm_node == None:
                                                        mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_1.input[0])
                                                        mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_1.input[1])
                                                        if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                            mm_node = mm_node_case1
                                                        elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                            mm_node = mm_node_case2

                                                    logger.debug('into match_mha_block_common_two, step 6 {}'.format(add_node_1.name))
                                                    if mm_node != None:
                                                        #logger.debug('into match_mha_block_common_two, step 6.1')
                                                        mul_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                        if ok == 0 and mul_node.op_type == 'Mul':
                                                            #logger.debug('into match_mha_block_common_two, step 6.2') 
                                                            mul_node_2, ok = get_prev_node_by_input(model, mul_node.input[0])
                                                            if ok == 0 and mul_node_2.op_type == 'Mul':
                                                                add_node_1, ok1 = get_prev_node_by_input(model, mul_node_2.input[0])
                                                                add_node_2, ok2 = get_prev_node_by_input(model, mul_node_2.input[1])

                                                                #logger.debug('into match_mha_block_common_two, step 7')

                                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                                    erf_node, ok = get_prev_node_by_input(model, add_node_2.input[0])
                                                                    if ok == 0 and erf_node.op_type == 'Erf':
                                                                        #logger.debug('into match_mha_block_common_two, step 8')
                                                                        div_node, ok = get_prev_node_by_input(model, erf_node.input[0])
                                                                        if ok == 0 and div_node.op_type == 'Div':
                                                                            #logger.debug('into match_mha_block_common_two, step 9')
                                                                            add_node, ok = get_prev_node_by_input(model, div_node.input[0])
                                                                            if ok == 0 and add_node.op_type == 'Add' and add_node == add_node_1:
                                                                                #logger.debug('into match_mha_block_common_two, step 10')
                                                                                mm_node = None
                                                                                mm_node_case1, ok1 = get_prev_node_by_input(model, add_node.input[0])
                                                                                mm_node_case2, ok2 = get_prev_node_by_input(model, add_node.input[1])

                                                                                if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                    mm_node = mm_node_case1
                                                                                elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                    mm_node = mm_node_case2

                                                                                if mm_node != None:
                                                                                    #logger.debug('into match_mha_block_common_two, step 11')
                                                                                    add_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                    if ok == 0 and add_node.op_type == 'Add':
                                                                                        mul_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                        if ok == 0 and mul_node.op_type == 'Mul':
                                                                                            div_node = None
                                                                                            div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                                                                                            div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])

                                                                                            if ok1 == 0 and div_node_case1.op_type == 'Div':
                                                                                                div_node = div_node_case1
                                                                                            elif ok2 == 0 and div_node_case2.op_type == 'Div':
                                                                                                div_node = div_node_case2

                                                                                            if div_node != None:
                                                                                                logger.debug('into match_mha_block_common_two, step 12 {}'.format(div_node.name))
                                                                                                sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                                                                                                sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])

                                                                                                if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                                                                                                    add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                                                                                                    if ok == 0 and add_node.op_type == 'Add':
                                                                                                        #logger.debug('into match_mha_block_common_two, step 13')
                                                                                                        rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                                        if ok == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                            #logger.debug('into match_mha_block_common_two, step 14')
                                                                                                            pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                            if ok == 0 and pow_node.op_type == 'Pow':
                                                                                                                #logger.debug('into match_mha_block_common, step 15')
                                                                                                                sub_node_, ok = get_prev_node_by_input(model, pow_node.input[0])
                                                                                                                if ok == 0 and sub_node_.op_type == 'Sub' and sub_node_ == sub_node:
                                                                                                                    add_node, ok1 = get_prev_node_by_input(model, sub_node_.input[0])
                                                                                                                    rm_node, ok2 = get_prev_node_by_input(model, sub_node_.input[1]) 

                                                                                                                    logger.debug('into match_mha_block_common_two, step 16')
                                                                                                                    if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                                        add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                                        if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                                                                                            logger.debug('into match_mha_block_common_two, step 17 {}'.format(add_node_.name))
                                                                                                                            add_node_1, ok1 = get_prev_node_by_input(model, add_node_.input[0])                                                                                   
                                                                                                                            add_node_2, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                                                                                            if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                                                                                                mm_node = None
                                                                                                                                mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_2.input[0])
                                                                                                                                mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_2.input[1])

                                                                                                                                if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                    mm_node = mm_node_case1
                                                                                                                                elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                    mm_node = mm_node_case2

                                                                                                                                if mm_node == None:
                                                                                                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_1.input[0])
                                                                                                                                    mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_1.input[1])

                                                                                                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                        mm_node = mm_node_case1
                                                                                                                                    elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                        mm_node = mm_node_case2

                                                                                                                                if mm_node != None:
                                                                                                                                    logger.debug('into match_mha_block_common_two, step 18')
                                                                                                                                    reshape_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                                                                    if ok == 0 and reshape_node.op_type == 'Reshape':
                                                                                                                                        tp_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                                                                                                                        if ok == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                            logger.debug('into match_mha_block_common_two, step 19 {}'.format(tp_node.name))
                                                                                                                                            mm_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                                                                            if ok == 0 and mm_node.op_type == 'MatMul':
                                                                                                                                                softmax_node, ok1 = get_prev_node_by_input(model, mm_node.input[0])                                                                                   
                                                                                                                                                squeeze_node, ok2 = get_prev_node_by_input(model, mm_node.input[1])
                                                                                                                                                logger.debug('into match_mha_block_common_two, step 20 {} {}'.format(softmax_node.name, squeeze_node.name))
                                                                                                                                                if ok1 == 0 and softmax_node.op_type == 'Softmax' and ok2 == 0 and squeeze_node.op_type == 'Squeeze':
                                                                                                                                                    logger.debug('into match_mha_block_common_two, step 21')
                                                                                                                                                    split_node, ok = get_prev_node_by_input(model, squeeze_node.input[0])
                                                                                                                                                    if ok == 0 and split_node.op_type == 'Split':
                                                                                                                                                        tp_node, ok = get_prev_node_by_input(model, split_node.input[0])
                                                                                                                                                        if ok == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                                            reshape_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                                                                                            if ok == 0 and reshape_node.op_type == 'Reshape':
                                                                                                                                                                add_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                                                                                                                                                if ok == 0 and add_node.op_type == 'Add':
                                                                                                                                                                    mm_node, ok = get_prev_node_by_input(model, add_node.input[1])
                                                                                                                                                                    if ok == 0 and mm_node.op_type == 'MatMul':                                                                                                                           
                                                                                                                                                                        logger.debug('got common mha block two')
                                                                                                                                                                        ret = 0
                                                                                                                                                                        break
    return ret

def match_mha_block_pattern_five(model):
    ret = -1

    logger.debug('into match_mha_block_pattern_five')

    for node in model.graph.node:
        if node.op_type == 'Add':
            mul_node, ok = get_prev_node_by_input(model, node.input[0])
            if ok == 0 and mul_node.op_type == 'Mul':
                div_node = None
                div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])
                if ok1 == 0 and div_node_case1.op_type == 'Div':
                    div_node = div_node_case1
                elif ok2 == 0 and div_node_case2.op_type == 'Div':
                    div_node = div_node_case2

                #logger.debug('into match_mha_block_pattern_five, step 1')    

                if div_node != None:
                    sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                    sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])
                    if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                        add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            #logger.debug('into match_mha_block_pattern_five, step 2')
                            rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                            if ok == 0 and rm_node.op_type == 'ReduceMean':
                                pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                if ok == 0 and pow_node.op_type == 'Pow':
                                    #logger.debug('into match_mha_block_pattern_five, step 3')
                                    sub_node, ok = get_prev_node_by_input(model, pow_node.input[0])
                                    if ok == 0 and sub_node.op_type == 'Sub':
                                        add_node, ok1 = get_prev_node_by_input(model, sub_node.input[0])
                                        rm_node, ok2 = get_prev_node_by_input(model, sub_node.input[1])
                                        if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                            #logger.debug('into match_mha_block_pattern_five, step 4')
                                            add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                            if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                add_node_1, ok1 = get_prev_node_by_input(model, add_node_.input[0])
                                                tp_node, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                logger.debug('into match_mha_block_pattern_five, step 5')
                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and tp_node.op_type == 'Transpose':
                                                    add_node_2, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                    if ok == 0 and add_node_2.op_type == 'Add':  
                                                        mm_node = None
                                                        mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_2.input[0])
                                                        mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_2.input[1])
                                                        if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                            mm_node = mm_node_case1
                                                        elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                            mm_node = mm_node_case2

                                                        logger.debug('into match_mha_block_pattern_five, step 6')
                                                        if mm_node != None:
                                                            #logger.debug('into match_mha_block_pattern_five, step 6.1')
                                                            mul_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                            if ok == 0 and mul_node.op_type == 'Mul':
                                                                #logger.debug('into match_mha_block_pattern_five, step 6.2') 
                                                                mul_node_2, ok = get_prev_node_by_input(model, mul_node.input[0])
                                                                if ok == 0 and mul_node_2.op_type == 'Mul':
                                                                    add_node_1, ok1 = get_prev_node_by_input(model, mul_node_2.input[0])
                                                                    add_node_2, ok2 = get_prev_node_by_input(model, mul_node_2.input[1])

                                                                    #logger.debug('into match_mha_block_pattern_five, step 7')

                                                                    if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                                        erf_node, ok = get_prev_node_by_input(model, add_node_2.input[0])
                                                                        if ok == 0 and erf_node.op_type == 'Erf':
                                                                            #logger.debug('into match_mha_block_pattern_five, step 8')
                                                                            div_node, ok = get_prev_node_by_input(model, erf_node.input[0])
                                                                            if ok == 0 and div_node.op_type == 'Div':
                                                                                #logger.debug('into match_mha_block_pattern_five, step 9')
                                                                                add_node, ok = get_prev_node_by_input(model, div_node.input[0])
                                                                                if ok == 0 and add_node.op_type == 'Add' and add_node == add_node_1:
                                                                                    logger.debug('into match_mha_block_pattern_five, step 10')
                                                                                    mm_node = None
                                                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, add_node.input[0])
                                                                                    mm_node_case2, ok2 = get_prev_node_by_input(model, add_node.input[1])

                                                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                        mm_node = mm_node_case1
                                                                                    elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                        mm_node = mm_node_case2

                                                                                    if mm_node != None:
                                                                                        logger.debug('into match_mha_block_pattern_five, step 11')
                                                                                        tp_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                        if ok == 0 and tp_node.op_type == 'Transpose':
                                                                                            #################
                                                                                            add_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                            if ok == 0 and add_node.op_type == 'Add':
                                                                                                mul_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                                if ok == 0 and mul_node.op_type == 'Mul':
                                                                                                    div_node = None
                                                                                                    div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                                                                                                    div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])

                                                                                                    if ok1 == 0 and div_node_case1.op_type == 'Div':
                                                                                                        div_node = div_node_case1
                                                                                                    elif ok2 == 0 and div_node_case2.op_type == 'Div':
                                                                                                        div_node = div_node_case2

                                                                                                    if div_node != None:
                                                                                                        logger.debug('into match_mha_block_pattern_five, step 12')
                                                                                                        sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                                                                                                        sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])

                                                                                                        if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                                                                                                            add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                                                                                                            if ok == 0 and add_node.op_type == 'Add':
                                                                                                                #logger.debug('into match_mha_block_pattern_five, step 13')
                                                                                                                rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                                                if ok == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                                    #logger.debug('into match_mha_block_pattern_five, step 14')
                                                                                                                    pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                                    if ok == 0 and pow_node.op_type == 'Pow':
                                                                                                                        #logger.debug('into match_mha_block_pattern_five, step 15')
                                                                                                                        sub_node_, ok = get_prev_node_by_input(model, pow_node.input[0])
                                                                                                                        if ok == 0 and sub_node_.op_type == 'Sub' and sub_node_ == sub_node:
                                                                                                                            add_node, ok1 = get_prev_node_by_input(model, sub_node_.input[0])
                                                                                                                            rm_node, ok2 = get_prev_node_by_input(model, sub_node_.input[1]) 

                                                                                                                            #logger.debug('into match_mha_block_pattern_five, step 16')
                                                                                                                            if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                                                add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                                                if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                                                                                                    logger.debug('got common mha block')
                                                                                                                                    ret = 0

    return ret

def match_mha_block_pattern_six(model):
    ret = -1

    logger.debug('into match_mha_block_pattern_six')

    for node in model.graph.node:
        if node.op_type == 'Add':
            mul_node, ok = get_prev_node_by_input(model, node.input[0])
            if ok == 0 and mul_node.op_type == 'Mul':
                div_node = None
                div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])
                if ok1 == 0 and div_node_case1.op_type == 'Div':
                    div_node = div_node_case1
                elif ok2 == 0 and div_node_case2.op_type == 'Div':
                    div_node = div_node_case2

                #logger.debug('into match_mha_blmatch_mha_block_pattern_sixock_common, step 1')    

                if div_node != None:
                    sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                    sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])
                    if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                        add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            #logger.debug('into match_mha_block_pattern_six, step 2')
                            rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                            if ok == 0 and rm_node.op_type == 'ReduceMean':
                                pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                if ok == 0 and pow_node.op_type == 'Pow':
                                    #logger.debug('into match_mha_block_pattern_six, step 3')
                                    sub_node, ok = get_prev_node_by_input(model, pow_node.input[0])
                                    if ok == 0 and sub_node.op_type == 'Sub':
                                        add_node, ok1 = get_prev_node_by_input(model, sub_node.input[0])
                                        rm_node, ok2 = get_prev_node_by_input(model, sub_node.input[1])
                                        if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                            #logger.debug('into match_mha_block_pattern_six, step 4')
                                            add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                            if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                add_node_1, ok1 = get_prev_node_by_input(model, add_node_.input[0])
                                                add_node_2, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                #logger.debug('into match_mha_block_pattern_six, step 5')
                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                    mm_node = None
                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_2.input[0])
                                                    mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_2.input[1])
                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                        mm_node = mm_node_case1
                                                    elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                        mm_node = mm_node_case2

                                                    if mm_node == None:
                                                        mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_1.input[0])
                                                        mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_1.input[1])
                                                        if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                            mm_node = mm_node_case1
                                                        elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                            mm_node = mm_node_case2

                                                    logger.debug('into match_mha_block_pattern_six, step 6')
                                                    if mm_node != None:
                                                        #logger.debug('into match_mha_block_pattern_six, step 6.1')
                                                        mul_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                        if ok == 0 and mul_node.op_type == 'Mul':
                                                            #logger.debug('into match_mha_block_pattern_six, step 6.2') 
                                                            mul_node_2, ok = get_prev_node_by_input(model, mul_node.input[0])
                                                            if ok == 0 and mul_node_2.op_type == 'Mul':
                                                                add_node_1, ok1 = get_prev_node_by_input(model, mul_node_2.input[0])
                                                                add_node_2, ok2 = get_prev_node_by_input(model, mul_node_2.input[1])

                                                                #logger.debug('into match_mha_block_pattern_six, step 7')

                                                                if ok1 == 0 and add_node_1.op_type == 'Add' and ok2 == 0 and add_node_2.op_type == 'Add':
                                                                    erf_node, ok = get_prev_node_by_input(model, add_node_2.input[0])
                                                                    if ok == 0 and erf_node.op_type == 'Erf':
                                                                        #logger.debug('into match_mha_block_pattern_six, step 8')
                                                                        div_node, ok = get_prev_node_by_input(model, erf_node.input[0])
                                                                        if ok == 0 and div_node.op_type == 'Div':
                                                                            #logger.debug('into match_mha_block_pattern_six, step 9')
                                                                            add_node, ok = get_prev_node_by_input(model, div_node.input[0])
                                                                            if ok == 0 and add_node.op_type == 'Add' and add_node == add_node_1:
                                                                                #logger.debug('into match_mha_block_pattern_six, step 10')
                                                                                mm_node = None
                                                                                mm_node_case1, ok1 = get_prev_node_by_input(model, add_node.input[0])
                                                                                mm_node_case2, ok2 = get_prev_node_by_input(model, add_node.input[1])

                                                                                if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                    mm_node = mm_node_case1
                                                                                elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                    mm_node = mm_node_case2

                                                                                if mm_node != None:
                                                                                    #logger.debug('into match_mha_block_pattern_six, step 11')
                                                                                    add_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                    if ok == 0 and add_node.op_type == 'Add':
                                                                                        mul_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                        if ok == 0 and mul_node.op_type == 'Mul':
                                                                                            div_node = None
                                                                                            div_node_case1, ok1 = get_prev_node_by_input(model, mul_node.input[0])
                                                                                            div_node_case2, ok2 = get_prev_node_by_input(model, mul_node.input[1])

                                                                                            if ok1 == 0 and div_node_case1.op_type == 'Div':
                                                                                                div_node = div_node_case1
                                                                                            elif ok2 == 0 and div_node_case2.op_type == 'Div':
                                                                                                div_node = div_node_case2

                                                                                            if div_node != None:
                                                                                                #logger.debug('into match_mha_block_pattern_six, step 12')
                                                                                                sub_node, ok1 = get_prev_node_by_input(model, div_node.input[0])
                                                                                                sqrt_node, ok2 = get_prev_node_by_input(model, div_node.input[1])

                                                                                                if ok1 == 0 and sub_node.op_type == 'Sub' and ok2 == 0 and sqrt_node.op_type == 'Sqrt':
                                                                                                    add_node, ok = get_prev_node_by_input(model, sqrt_node.input[0])
                                                                                                    if ok == 0 and add_node.op_type == 'Add':
                                                                                                        #logger.debug('into match_mha_block_pattern_six, step 13')
                                                                                                        rm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                                                                                        if ok == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                            #logger.debug('into match_mha_block_pattern_six, step 14')
                                                                                                            pow_node, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                            if ok == 0 and pow_node.op_type == 'Pow':
                                                                                                                #logger.debug('into match_mha_block_pattern_six, step 15')
                                                                                                                sub_node_, ok = get_prev_node_by_input(model, pow_node.input[0])
                                                                                                                if ok == 0 and sub_node_.op_type == 'Sub' and sub_node_ == sub_node:
                                                                                                                    add_node, ok1 = get_prev_node_by_input(model, sub_node_.input[0])
                                                                                                                    rm_node, ok2 = get_prev_node_by_input(model, sub_node_.input[1]) 

                                                                                                                    logger.debug('into match_mha_block_pattern_six, step 16 {}'.format(sub_node_.name))
                                                                                                                    if ok1 == 0 and add_node.op_type == 'Add' and ok2 == 0 and rm_node.op_type == 'ReduceMean':
                                                                                                                        add_node_, ok = get_prev_node_by_input(model, rm_node.input[0])
                                                                                                                        if ok == 0 and add_node_.op_type == 'Add' and add_node_ == add_node:
                                                                                                                            logger.debug('into match_mha_block_pattern_six, step 17 {}'.format(add_node_.name))                                                                                

                                                                                                                            mm_node = None
                                                                                                                            mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_.input[0])
                                                                                                                            mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_.input[1])

                                                                                                                            if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                mm_node = mm_node_case1
                                                                                                                            elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                mm_node = mm_node_case2

                                                                                                                            if mm_node == None:
                                                                                                                                mm_node_case1, ok1 = get_prev_node_by_input(model, add_node_1.input[0])
                                                                                                                                mm_node_case2, ok2 = get_prev_node_by_input(model, add_node_1.input[1])

                                                                                                                                if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                    mm_node = mm_node_case1
                                                                                                                                elif ok2 == 0 and mm_node_case2.op_type == 'MatMul':
                                                                                                                                    mm_node = mm_node_case2

                                                                                                                            if mm_node != None:
                                                                                                                                logger.debug('into match_mha_block_pattern_six, step 18 {}'.format(mm_node.name))
                                                                                                                                reshape_node, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                                                                                                if ok == 0 and reshape_node.op_type == 'Reshape':
                                                                                                                                    tp_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                                                                                                                    if ok == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                        logger.debug('into match_mha_block_pattern_six, step 19')
                                                                                                                                        mm_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                                                                        if ok == 0 and mm_node.op_type == 'MatMul':
                                                                                                                                            softmax_node, ok1 = get_prev_node_by_input(model, mm_node.input[0])                                                                                   
                                                                                                                                            tp_node, ok2 = get_prev_node_by_input(model, mm_node.input[1])

                                                                                                                                            if ok1 == 0 and softmax_node.op_type == 'Softmax' and ok2 == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                                logger.debug('into match_mha_block_pattern_six, step 20')
                                                                                                                                                reshape_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                                                                                                                                if ok == 0 and reshape_node.op_type == 'Reshape':
                                                                                                                                                    logger.debug('into match_mha_block_pattern_six, step 21')

                                                                                                                                                    mm_node = None
                                                                                                                                                    mm_node_case1, ok1 = get_prev_node_by_input(model, reshape_node.input[0])

                                                                                                                                                    if ok1 == 0 and mm_node_case1.op_type == 'MatMul':
                                                                                                                                                        mm_node = mm_node_case1

                                                                                                                                                    if mm_node != None:
                                                                                                                                                        logger.debug('into match_mha_block_pattern_six, step 23')
                                                                                                                                                        tp_node, ok = get_prev_node_by_input(model, mm_node.input[0])

                                                                                                                                                        if ok == 0 and tp_node.op_type == 'Transpose':
                                                                                                                                                            add_node_common, ok = get_prev_node_by_input(model, softmax_node.input[0])
                                                                                                                                                            if ok == 0 and add_node_common.op_type == 'Where':
                                                                                                                                                                ret = 0
                                                                                                                                                                print('got pattern six mha_block')
                                                                                                                                                                break
    return ret

def get_node_group(model, input_name, num, index):
    node_list = []
    name = input_name
    for i in range(num):
        node, ok = get_prev_node_by_input(model, name)
        if ok == 0 and len(node.input) > index[i]:
            name = node.input[index[i]]
            node_list.append(node)
        else:
            break

    return node_list  

def match_mha_block_pattern_two(model):
    common_dict = match_mha_block_common(model)
    if len(common_dict):
        add_node_last = common_dict['add_node_last']
        add_node_common = common_dict['add_node_common']

        div_node, ok = get_prev_node_by_input(model, add_node_common.input[0])
        if ok == 0 and div_node.op_type == 'Div':
            mm_node, ok = get_prev_node_by_input(model, div_node.input[0])
            if ok == 0 and mm_node.op_type == 'MatMul':
                tp_node_1, ok1 = get_prev_node_by_input(model, mm_node.input[0])                    
                tp_node_2, ok2 = get_prev_node_by_input(model, mm_node.input[1])

                if ok1 == 0 and tp_node_1.op_type == 'Transpose' and ok2 == 0 and tp_node_2.op_type == 'Transpose':
                    reshape_node, ok = get_prev_node_by_input(model, tp_node_1.input[0])
                    if ok == 0 and reshape_node.op_type == 'Reshape':
                        add_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            mm_node, ok = get_prev_node_by_input(model, add_node.input[1])
                            if ok == 0 and mm_node.op_type == 'MatMul':
                                add_node_branchA, ok = get_prev_node_by_input(model, mm_node.input[0])
                                if ok == 0 and add_node_branchA.op_type == 'Add' and add_node_branchA == add_node_last:
                                    ########################
                                    reshape_node, ok = get_prev_node_by_input(model, tp_node_2.input[0])
                                    if ok == 0 and reshape_node.op_type == 'Reshape':
                                        add_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                        if ok == 0 and add_node.op_type == 'Add':
                                            mm_node, ok = get_prev_node_by_input(model, add_node.input[1])
                                            if ok == 0 and mm_node.op_type == 'MatMul':
                                                add_node_branchB, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                if ok == 0 and add_node_branchB.op_type == 'Add' and add_node_branchB == add_node_last:
                                                    logger.debug('match mha block pattern two success')
                                                    return 0

    return -1

def match_mha_block_pattern_seven(model):
    ret = match_mha_block_common_two(model)

    return ret


def match_mha_block_pattern_eight(model):
    ret = -1
    node_list = []
    qkv_array = ['q', 'k', 'v']

    for node in model.graph.node:
        if node.op_type == 'Transpose':
            all_next_node, ok = get_all_next_node_by_output(model, node.output[0])
            if ok == 0 and len(all_next_node) == 3:
                gather_cnt = 0
                tgm = {}
                for n in all_next_node:
                    if n.op_type == 'Gather':
                        indices, shape = values.get_init_value_and_shape(model, n.input[1])
                        if len(shape) == 0:
                            gather_cnt = gather_cnt + 1
                            logger.debug('got gather({}) index: {}'.format(n.name, indices[0]))
                            tgm['gather_' + qkv_array[indices[0]]] = n

                if gather_cnt == 3:
                    tgm['Transpose'] = node
                    gather_node_v = tgm['gather_v']
                    matmul_qk_v, ok = get_next_node_by_output(model, gather_node_v.output[0])
                    if ok == 0 and matmul_qk_v.op_type == 'MatMul':
                        tgm['matmul_qk_v'] = matmul_qk_v
                        gather_node_q = tgm['gather_q']  
                        mul_node, ok = get_next_node_by_output(model, gather_node_q.output[0])
                        if ok == 0 and mul_node.op_type == 'Mul':
                            tgm['Mul'] = mul_node
                            matmul_qk, ok = get_next_node_by_output(model, mul_node.output[0])
                            if ok == 0 and matmul_qk.op_type == 'MatMul':
                                tgm['matmul_qk'] = matmul_qk
                                gather_node_k = tgm['gather_k']
                                transpose_k, ok = get_next_node_by_output(model, gather_node_k.output[0])
                                if ok == 0 and transpose_k.op_type == 'Transpose':
                                    tgm['TransposeK'] = transpose_k
                                    mm_node, ok = get_next_node_by_output(model, transpose_k.output[0])
                                    if ok == 0 and mm_node == matmul_qk:
                                        logger.debug('Match MatMul_QK node: {}'.format(mm_node.name))
                                        add_node, ok = get_next_node_by_output(model, mm_node.output[0])
                                        if ok == 0 and add_node.op_type == 'Add':
                                            tgm['Add'] = add_node
                                            softmax_node, ok = get_next_node_by_output(model, add_node.output[0])
                                            if ok == 0 and softmax_node.op_type == 'Softmax':
                                                tgm['Softmax'] = softmax_node
                                                mm_node, ok = get_next_node_by_output(model, softmax_node.output[0])
                                                if ok == 0 and mm_node == matmul_qk_v:
                                                    logger.debug('match pattern eight~')
                                                    node_list.append(tgm)
                                            else:
                                                if ok == 0 and softmax_node.op_type == 'Reshape':
                                                    tgm['Reshape1'] = softmax_node
                                                    add_node2, ok = get_next_node_by_output(model, softmax_node.output[0])
                                                    if ok == 0 and add_node2.op_type == 'Add':
                                                        tgm['Add2'] = add_node2
                                                        rs_node2, ok = get_next_node_by_output(model, add_node2.output[0])
                                                        if ok == 0 and rs_node2.op_type == 'Reshape':
                                                            tgm['Reshape2'] = rs_node2
                                                            softmax_node, ok = get_next_node_by_output(model, rs_node2.output[0])
                                                            if ok == 0 and softmax_node.op_type == 'Softmax':
                                                                tgm['Softmax'] = softmax_node
                                                                mm_node, ok = get_next_node_by_output(model, softmax_node.output[0])
                                                                if ok == 0 and mm_node == matmul_qk_v:
                                                                    logger.debug('~~~ match pattern eight~')
                                                                    node_list.append(tgm)

    if len(node_list) > 0:
        ret = 0

    return ret, node_list

def match_mha_block_pattern_three(model):
    common_dict = match_mha_block_common(model)
    if len(common_dict):
        add_node_last = common_dict['add_node_last']
        add_node_common = common_dict['add_node_common']

        mm_node, ok = get_prev_node_by_input(model, add_node_common.input[0])
        if ok == 0 and mm_node.op_type == 'MatMul':
            mul_node, ok1 = get_prev_node_by_input(model, mm_node.input[0])                    
            tp_node, ok2 = get_prev_node_by_input(model, mm_node.input[1])

            if ok1 == 0 and mul_node.op_type == 'Mul' and ok2 == 0 and tp_node.op_type == 'Transpose':
                tp_node2, ok = get_prev_node_by_input(model, mul_node.input[0])
                if ok == 0 and tp_node2.op_type == 'Transpose':
                    reshape_node, ok = get_prev_node_by_input(model, tp_node2.input[0])
                    if ok == 0 and reshape_node.op_type == 'Reshape':
                        add_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                        if ok == 0 and add_node.op_type == 'Add':
                            mm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                            if ok == 0 and mm_node.op_type == 'MatMul':
                                add_node_branchA, ok = get_prev_node_by_input(model, mm_node.input[0])
                                if ok == 0 and add_node_branchA.op_type == 'Add' and add_node_branchA == add_node_last:
                                    ########################
                                    reshape_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                                    if ok == 0 and reshape_node.op_type == 'Reshape':
                                        add_node, ok = get_prev_node_by_input(model, reshape_node.input[0])
                                        if ok == 0 and add_node.op_type == 'Add':
                                            mm_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                            if ok == 0 and mm_node.op_type == 'MatMul':
                                                add_node_branchB, ok = get_prev_node_by_input(model, mm_node.input[0])
                                                if ok == 0 and add_node_branchB.op_type == 'Add' and add_node_branchB == add_node_last:
                                                    logger.debug('match mha block pattern three success')
                                                    return 0

    return -1

def match_mha_block_pattern_one(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Add':
            node_list = get_node_group(model, node.input[1], 6, [0,0,1,0,0,0])
            if len(node_list) == 6:
                expected_pattern = ['MatMul', 'Relu', 'Add', 'MatMul', 'Add', 'Mul']
                for idx1, n in enumerate(node_list):
                    #print('node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        break

                if idx1 == 5:
                    node_list2 = get_node_group(model, node_list[5].input[0], 6, [1,0,0,0,0,0])
                    if len(node_list2) == 6:
                        expected_pattern = ['Div', 'Sqrt', 'Add', 'ReduceMean', 'Pow', 'Sub']
                        for idx2, n in enumerate(node_list2):
                            if n.op_type != expected_pattern[idx2]:
                                break        

                        if idx2 == 5:
                            node_list3 = get_node_group(model, node_list2[5].input[1], 7, [0,1,1,0,0,0,0])
                            if len(node_list3) == 7:
                                expected_pattern = ['ReduceMean', 'Add', 'Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul']
                                for idx3, n in enumerate(node_list3):
                                    if n.op_type != expected_pattern[idx3]:
                                        break

                                if idx3 == 6:
                                    node_list4 = get_node_group(model, node_list3[6].input[1], 4, [0,0,1,0])
                                    if len(node_list4) == 4:
                                        expected_pattern = ['Transpose', 'Reshape', 'Add', 'MatMul']
                                        for idx4, n in enumerate(node_list4):
                                            if n.op_type != expected_pattern[idx4]:
                                                break

                                        if idx4 == 3:
                                            node_list5 = get_node_group(model, node_list3[6].input[0], 12, [1,0,1,0,0,0,0,1,0,0,0,0])
                                            if len(node_list5) == 12:
                                                expected_pattern = ['Where', 'Softmax', 'Where', 'Div', 'MatMul', 'Transpose', 'Reshape','Add', 'MatMul', 'Add', 'Mul', 'Div']
                                                for idx5, n in enumerate(node_list5):
                                                    if n.op_type != expected_pattern[idx5]:
                                                        break

                                                if idx5 == 11:
                                                    node_list6 = get_node_group(model, node_list3[6].input[0], 9, [1,0,1,0,1,0,0,1,0])
                                                    if len(node_list6) == 9:
                                                        expected_pattern = ['Where', 'Softmax', 'Where', 'Div', 'MatMul', 'Transpose', 'Reshape','Add', 'MatMul']
                                                        for idx6, n in enumerate(node_list6):
                                                            if n.op_type != expected_pattern[idx6]:
                                                                break

                                                        if idx6 == 8:
                                                            res = 0
                                                            logger.debug('match_mha_block_pattern_one, success')
                                                            break

    return res                                                                 

def match_mha_block_pattern_four(model):
    res = -1

    for node in model.graph.node:
        if node.op_type == 'Add':
            node_list = get_node_group(model, node.input[1], 11, [1,1,0,0,0,0,0,0,1,0,0])
            if len(node_list) == 11:
                expected_pattern = ['Sub', 'Mul', 'Mul', 'Reciprocal', 'Sqrt', 'Add', 'ReduceMean', 'Mul', 'Sub', 'ReduceMean', 'Add']      
                for idx1, n in enumerate(node_list):
                    #print('node:', idx1, n.op_type, expected_pattern[idx1])
                    if n.op_type != expected_pattern[idx1]:
                        break

                if idx1 == 10:
                    logger.debug('match_mha_block_pattern_four, success 1')

                    node_list1 = get_node_group(model, node_list[10].input[0], 13, [0,0,1,1,1,0,1,1,1,0,0,0,0])
                    if len(node_list1) == 13:
                        expected_pattern = ['Add', 'MatMul', 'Mul', 'Mul', 'Add', 'Tanh', 'Mul', 'Add', 'Mul', 'Pow', 'Add', 'MatMul', 'Add']      
                        for idx2, n in enumerate(node_list1):
                            #print('node:', idx1, n.op_type, expected_pattern[idx1])
                            if n.op_type != expected_pattern[idx2]:
                                break

                        if idx2 == 12:
                            logger.debug('match_mha_block_pattern_four, success 2')
                            node_list2 = get_node_group(model, node_list1[12].input[1], 11, [1,1,0,0,0,0,0,0,1,0,0])
                            if len(node_list2) == 11:
                                expected_pattern = ['Sub', 'Mul', 'Mul', 'Reciprocal', 'Sqrt', 'Add', 'ReduceMean', 'Mul', 'Sub', 'ReduceMean', 'Add']
                                for idx3, n in enumerate(node_list2):
                                    #print('node:', idx1, n.op_type, expected_pattern[idx1])
                                    if n.op_type != expected_pattern[idx3]:
                                        break

                                if idx3 == 10:
                                    logger.debug('match_mha_block_pattern_four, success 3')
                                    last_add_node, ok = get_prev_node_by_input(model, node_list2[10].input[1])
                                    if ok == 0 and last_add_node.op_type == 'Add':
                                        node_list3 = get_node_group(model, node_list2[10].input[0], 5, [0,0,0,0,0])
                                        if len(node_list3) == 5:
                                            expected_pattern = ['Add', 'MatMul', 'Reshape', 'Transpose', 'MatMul']
                                            for idx4, n in enumerate(node_list3):
                                                #print('node:', idx1, n.op_type, expected_pattern[idx1])
                                                if n.op_type != expected_pattern[idx4]:
                                                    break

                                            if idx4 == 4:
                                                logger.debug('match_mha_block_pattern_four, success 4')
                                                node_list4 = get_node_group(model, node_list3[4].input[1], 5, [0,0,0,0,0])
                                                if len(node_list4) == 5:
                                                    expected_pattern = ['Transpose', 'Reshape', 'Add', 'MatMul', 'Add']
                                                    for idx5, n in enumerate(node_list4):
                                                        #print('node:', idx1, n.op_type, expected_pattern[idx1])
                                                        if n.op_type != expected_pattern[idx5]:
                                                            break

                                                    if idx5 == 4 and node_list4[4] == last_add_node:
                                                        logger.debug('match_mha_block_pattern_four, success 5')
                                                        node_list5 = get_node_group(model, node_list3[4].input[0], 4, [0,0,0,0])
                                                        if len(node_list5) == 4:
                                                            expected_pattern = ['Softmax', 'Add', 'Mul', 'MatMul']
                                                            for idx5, n in enumerate(node_list5):
                                                                #print('node:', idx1, n.op_type, expected_pattern[idx1])
                                                                if n.op_type != expected_pattern[idx5]:
                                                                    break

                                                            if idx5 == 3:
                                                                logger.debug('match_mha_block_pattern_four, success 6')

                                                                match_times = 0
                                                                
                                                                for i in range(2):
                                                                    node_list_ = get_node_group(model, node_list5[3].input[i], 5, [0,0,0,0,0])
                                                                    if len(node_list_) == 5:
                                                                        expected_pattern = ['Transpose', 'Reshape', 'Add', 'MatMul', 'Add']
                                                                        for idx_, n in enumerate(node_list_):
                                                                            #print('node:', idx1, n.op_type, expected_pattern[idx1])
                                                                            if n.op_type != expected_pattern[idx_]:
                                                                                break

                                                                        if idx_ == 4 and node_list_[4] == last_add_node:
                                                                            match_times = match_times + 1
                                                                            logger.debug('match_mha_block_pattern_four, success 7, match_times: {}'.format(match_times))

                                                                        if match_times == 2:
                                                                            logger.debug('match_mha_block_pattern_four, success!!!!')
                                                                            res = 0
                                                                            break

    return res

def get_matmul_block_pattern_four(model, matmul_node):
    logger.debug('into get_matmul_block_pattern_four')

    res = -1
    node_dict = {}

    #input_next, ok = get_next_node_by_output(model, input_)
    input_next = matmul_node
    if input_next.op_type == 'MatMul':
        shapeA = values.get_tensor_shape_by_name(model, input_next.input[0])
        inputB, shapeB = values.get_init_value_and_shape(model, input_next.input[1])

        if isinstance(inputB, list) and inputB == []:
            logger.debug('inputB is not in initilizer')
            inputB = values.get_constant_value(model, input_next.input[1])

        if len(shapeA) == 2 and len(shapeB) == 2:
            logger.debug('--- got MatMul node {}'.format(input_next.name))
            #node_list = [input_next, input_pp_pre, input_p_pre, input_pre]
            #node_dict['node_list'] = node_list
            node_dict['MatMul1'] = input_next
            node_dict['matmulA1_Shape'] = shapeA
            node_dict['inputB1'] = inputB
            node_dict['matmulB1_Shape'] = shapeB

            input_nnext, ok = get_next_node_by_output(model, input_next.output[0])
            if ok == 0 and input_nnext.op_type == 'Add':
                addA_name = input_nnext.input[0]
                addA, shapeA = values.get_init_value_and_shape(model, input_nnext.input[0])
                node_dict['addFirst'] = True

                if len(shapeA) == 0:
                    addA_name = input_nnext.input[1]
                    addA, shapeA = values.get_init_value_and_shape(model, input_nnext.input[1])
                    node_dict['addFirst'] = False

                if len(shapeA) == 1:
                    node_dict['Add1'] = input_nnext
                    logger.debug('--- got Add1 node {}'.format(input_nnext.name))

                    input_nnnext, ok = get_all_next_node_by_output(model, input_nnext.output[0])
                    if len(input_nnnext) == 3:
                        got_match_op = 0

                        for n in input_nnnext:
                            if n.op_type == 'Add':
                                node_dict['AddT'] = n
                                got_match_op = got_match_op + 1

                            if n.op_type == 'Pow':
                                node_dict['Pow'] = n
                                got_match_op = got_match_op + 1

                            if n.op_type == 'Mul':
                                node_dict['Mul'] = n
                                got_match_op = got_match_op + 1    

                        if got_match_op == 3:
                            input_nnnnnext, ok = get_next_node_by_output(model, node_dict['Mul'].output[0])
                            if ok == 0 and input_nnnnnext.op_type == 'MatMul':
                                shapeA = values.get_tensor_shape_by_name(model, input_nnnnnext.input[0])
                                inputB, shapeB = values.get_init_value_and_shape(model, input_nnnnnext.input[1])

                                if isinstance(inputB, list) and inputB == []:
                                    logger.debug('inputB is not in initilizer')
                                    inputB = values.get_constant_value(model, input_nnnnnext.input[1])

                                if len(shapeA) == 2 and len(shapeB) == 2:
                                    logger.debug('--- got MatMul2 node: {}'.format(input_nnnnnext.name))
                                    #node_list = [input_nnnnnext, input_pp_pre, input_p_pre, input_pre]
                                    #node_dict['node_list'] = node_list
                                    node_dict['MatMul2'] = input_nnnnnext
                                    node_dict['matmulA2_Shape'] = shapeA
                                    node_dict['inputB2'] = inputB
                                    node_dict['matmulB2_Shape'] = shapeB

                                    input_nnnnnnext, ok = get_next_node_by_output(model, input_nnnnnext.output[0])
                                    if ok == 0 and input_nnnnnnext.op_type == 'Add':
                                        logger.debug('--- got Add2 node: {}'.format(input_nnnnnnext.name))
                                    
                                    ##########
                                    addA_name = input_nnnnnnext.input[0]
                                    addA, shapeA = values.get_init_value_and_shape(model, input_nnnnnnext.input[0])
                                    node_dict['addFirst2'] = True

                                    if len(shapeA) == 0:
                                        addA_name = input_nnnnnnext.input[1]
                                        addA, shapeA = values.get_init_value_and_shape(model, input_nnnnnnext.input[1])
                                        node_dict['addFirst2'] = False

                                    if len(shapeA) == 1:
                                        node_dict['Add2'] = input_nnnnnnext
                                        next_node, ok = get_next_node_by_output(model, input_nnnnnnext.output[0])
                                        if ok == 0 and next_node.op_type == 'Add':
                                            logger.debug('--- got last Add node: {}'.format(next_node.name))
                                            res = 0
                                            node_dict['NextAdd'] = next_node 

    return node_dict, res

def get_mul_add_block_pattern_four(model):
    logger.debug('into get_mul_add_block_pattern_four')

    node_list = []
    for node in model.graph.node:
        if node.op_type == 'Mul':
            #print('----got mul:', node.name)
            next_node, ok = get_next_node_by_output(model, node.output[0])
            if ok == 0 and next_node.op_type == 'Add':
                #print('get_all_next_node_by_output---', next_node.output, node.name)
                next_node_list, ok = get_all_next_node_by_output(model, next_node.output[0])
                if ok == 0:
                    #print('next_node_list:', len(next_node_list))
                    if len(next_node_list) == 2:
                        #print('got next_node_list:', next_node_list[0].op_type, next_node_list[1].op_type)

                        if (next_node_list[0].op_type == 'Add' and next_node_list[1].op_type == 'MatMul') or \
                            (next_node_list[0].op_type == 'MatMul' and next_node_list[1].op_type == 'Add'):
                            logger.debug('got it~')
                            matmul_node = next_node_list[0]
                            if next_node_list[1].op_type == 'MatMul':
                                matmul_node = next_node_list[1]

                            node_dict, ret = get_matmul_block_pattern_four(model, matmul_node)
                            if ret == 0:
                                #print('got node dict:', node_dict)
                                node_dict['currentAdd'] = next_node
                                node_list.append(node_dict)

    return node_list

def handle_mul_add_block_pattern_four(model):
    node_list = get_mul_add_block_pattern_four(model)

    #if len(node_list) > 0:
    for node_dict in node_list:
        logger.debug('++++++++++++++++++++++')
        logger.debug('Add1: {}'.format(node_dict['Add1'].name))
        logger.debug('Add2: {}'.format(node_dict['Add2'].name))
        logger.debug('++++++++++++++++++++++')

        matmul1 = node_dict['MatMul1']
        add1 = node_dict['Add1']

        matmul2 = node_dict['MatMul2']
        add2 = node_dict['Add2']

        currentAdd = node_dict['currentAdd']
        nextAdd = node_dict['NextAdd']

        pow_node = node_dict['Pow']

        ###add transpose
        ts_name = currentAdd.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        add_output_shape = values.get_tensor_shape_by_name(model, currentAdd.output[0])
        ts_output_shape = [add_output_shape[1], add_output_shape[0]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[currentAdd.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[1,0])

        model.graph.value_info.append(transpose_output)

        ###add reshape-1
        rs_name = currentAdd.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [1, ts_output_shape[0], 1, ts_output_shape[1]]

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = currentAdd.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs_name,
                                            inputs=[ts_output_name, const_shape_name],
                                            outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        #########################
        insert_node(model, rs_node, matmul1)
        matmul1.input[0] = rs_output_name

        insert_node(model, ts_node, rs_node)

        nextAdd.input[1] = ts_output_name

        #MatMul1--->Conv
        matmul1.op_type = 'Conv'
        logger.debug('-----reuse MatMul to Conv: {}'.format(matmul1.name))
        const_x_name = matmul1.name + '_to_conv_x_'

        v = node_dict['inputB1']
        old_dims = [node_dict['matmulB1_Shape'][0], node_dict['matmulB1_Shape'][1]]
        dims_ = [node_dict['matmulB1_Shape'][1], node_dict['matmulB1_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul1.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul1.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul1.attribute.append(attr)        

        if node_dict['addFirst'] == True:
            matmul1.input.append(add1.input[0])
        else:
            matmul1.input.append(add1.input[1])   

        output_shape = values.get_tensor_shape_by_name(model, matmul1.output[0])
        conv_output_shape = [rs_output_shape[0], node_dict['matmulB1_Shape'][1], rs_output_shape[2], rs_output_shape[3]]#[1, output_shape[1], 1, output_shape[0]] 

        update_tensor_shape(model, matmul1.output[0], conv_output_shape) 

        #Add1--->Reshape
        add1.op_type = 'Reshape'

        del add1.attribute[:]

        rs_name = add1.name + '_reshape_1_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [conv_output_shape[1], conv_output_shape[3]]
        logger.debug('-----rs_output_shape: {}'.format(rs_output_shape))

        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape_name = add1.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        if node_dict['addFirst'] == True:
            add1.input[0] = add1.input[1]

        add1.input[1] = const_shape_name

        update_tensor_shape(model, add1.output[0], rs_output_shape)

        mul_node = node_dict['Mul']
        update_tensor_shape(model, mul_node.output[0], rs_output_shape)

        #################################
        #################################
        ###add reshape-1
        rs2_name = matmul2.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [1, rs_output_shape[0], 1, rs_output_shape[1]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = matmul2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        rs2_node = onnx.helper.make_node(
                                            'Reshape',
                                            name=rs2_name,
                                            inputs=[matmul2.input[0], const_shape_name],
                                            outputs=[rs2_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs2_node, matmul2)
        matmul2.input[0] = rs2_output_name

        #MatMul2--->Conv
        matmul2.op_type = 'Conv'
        logger.debug('++++++reuse MatMul to Conv')
        const_x_name = matmul2.name + '_to_conv_x_'

        v = node_dict['inputB2']
        old_dims = [node_dict['matmulB2_Shape'][0], node_dict['matmulB2_Shape'][1]]
        dims_ = [node_dict['matmulB2_Shape'][1], node_dict['matmulB2_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul2.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul2.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul2.attribute.append(attr)        

        if node_dict['addFirst2'] == True:
            B = add2.input[0]
        else:
            B = add2.input[1]   

        matmul2.input.append(B)

        output_shape = values.get_tensor_shape_by_name(model, matmul2.output[0])
        conv_output_shape = [rs2_output_shape[0], node_dict['matmulB2_Shape'][1], rs2_output_shape[2], rs2_output_shape[3]]#[1, output_shape[1], 1, output_shape[0]] 

        update_tensor_shape(model, matmul2.output[0], conv_output_shape) 

        #Add2--->Reshape
        add2.op_type = 'Reshape'

        del add2.attribute[:]

        rs2_name = add2.name + '_reshape_1_'
        rs2_output_name = rs2_name + '_output_'
        rs2_output_shape = [conv_output_shape[1], conv_output_shape[3]]

        rs_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.FLOAT, rs2_output_shape)

        const_shape_name = add2.name + '_reshape_data_'
        
        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        if node_dict['addFirst2'] == True:
            add2.input[0] = add2.input[1]

        add2.input[1] = const_shape_name

        update_tensor_shape(model, add2.output[0], rs2_output_shape)

        ######update tensor shape
        pow_output_shape = values.get_tensor_shape_by_name(model, pow_node.output[0])
        new_shape = [pow_output_shape[1], pow_output_shape[0]]
        update_tensor_shape(model, pow_node.output[0], new_shape)

        mul_node, ok = get_next_node_by_output(model, pow_node.output[0])
        if ok == 0 and mul_node.op_type == 'Mul':
            mul_output_shape = values.get_tensor_shape_by_name(model, mul_node.output[0])
            new_shape = [mul_output_shape[1], mul_output_shape[0]]
            update_tensor_shape(model, mul_node.output[0], new_shape)

            add_node_internal, ok = get_next_node_by_output(model, mul_node.output[0])
            if ok == 0 and add_node_internal.op_type == 'Add':
                addi_output_shape = values.get_tensor_shape_by_name(model, add_node_internal.output[0])
                new_shape = [addi_output_shape[1], addi_output_shape[0]]
                update_tensor_shape(model, add_node_internal.output[0], new_shape)
        
                mul_node1, ok = get_next_node_by_output(model, add_node_internal.output[0])
                if ok == 0 and mul_node1.op_type == 'Mul':
                    mul1_output_shape = values.get_tensor_shape_by_name(model, mul_node1.output[0])
                    new_shape = [mul1_output_shape[1], mul1_output_shape[0]]
                    update_tensor_shape(model, mul_node1.output[0], new_shape)

                    tanh_node, ok = get_next_node_by_output(model, mul_node1.output[0])
                    if ok == 0 and tanh_node.op_type == 'Tanh':
                        tanh_output_shape = values.get_tensor_shape_by_name(model, tanh_node.output[0])
                        new_shape = [tanh_output_shape[1], tanh_output_shape[0]]
                        update_tensor_shape(model, tanh_node.output[0], new_shape)

                        add_node2, ok = get_next_node_by_output(model, tanh_node.output[0])
                        if ok == 0 and add_node2.op_type == 'Add':    
                            add_output_shape = values.get_tensor_shape_by_name(model, add_node2.output[0])
                            new_shape = [add_output_shape[1], add_output_shape[0]]
                            update_tensor_shape(model, add_node2.output[0], new_shape)

                            mul_node2, ok = get_next_node_by_output(model, add_node2.output[0])
                            if ok == 0 and mul_node2.op_type == 'Mul':    
                                mul2_output_shape = values.get_tensor_shape_by_name(model, mul_node2.output[0])
                                new_shape = [mul2_output_shape[1], mul2_output_shape[0]]
                                update_tensor_shape(model, mul_node2.output[0], new_shape)

        ######insert Transpose before ReduceMean and Sub
        update_tensor_shape(model, nextAdd.output[0], rs2_output_shape)

        rm_sub, ok = get_all_next_node_by_output(model, nextAdd.output[0])
        if ok == 0 and len(rm_sub) == 3:
            logger.debug('got reducemean and sub node---')
            sub_node = None
            rm_node = None
            mul_node = None

            for n in rm_sub:
                if n.op_type == 'Sub':
                    sub_node = n

                if n.op_type == 'ReduceMean':
                    rm_node = n

                if n.op_type == 'Mul':
                    mul_node = n

            if sub_node != None and rm_node != None and mul_node != None:
                ###add transpose
                ts3_name = nextAdd.name + '_transpose_'
                ts3_output_name = ts3_name + '_output_'
                add3_output_shape = values.get_tensor_shape_by_name(model, nextAdd.output[0])
                ts3_output_shape = [add3_output_shape[1], add3_output_shape[0]]
                ts3_output = onnx.helper.make_tensor_value_info(ts3_output_name, onnx.TensorProto.FLOAT, ts3_output_shape)
                
                ts3_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts3_name,
                                                    inputs=[nextAdd.output[0]],
                                                    outputs=[ts3_output_name],
                                                    perm=[1,0])

                model.graph.value_info.append(ts3_output)

                insert_node(model, ts3_node, sub_node) 
                sub_node.input[0] = ts3_output_name
                rm_node.input[0] = ts3_output_name
                mul_node.input[0] = ts3_output_name

def get_last_group_pattern_four(model):
    graph_output = []
    node_dict = {}
    res = -1

    for o in model.graph.output:
        graph_output.append(o.name)

    for node in model.graph.node:
        if node.output[0] in graph_output:
            #print('got mul:', node.name)
            if node.op_type == 'Squeeze':
                split_node, ok = get_prev_node_by_input(model, node.input[0])
                if ok == 0 and split_node.op_type == 'Split':
                    logger.debug('got Split node: {}'.format(split_node.name))
                    node_dict['Split'] = split_node

                    tp_node, ok = get_prev_node_by_input(model, split_node.input[0])
                    if ok == 0 and tp_node.op_type == 'Transpose':
                        rs_node, ok = get_prev_node_by_input(model, tp_node.input[0])
                        if ok == 0 and rs_node.op_type == 'Reshape':
                            node_dict['Reshape'] = rs_node
                            add_node, ok = get_prev_node_by_input(model, rs_node.input[0])
                            if ok == 0 and add_node.op_type == 'Add':
                                logger.debug('get_last_group_pattern_four, got Add node: {}'.format(add_node.name))
                                node_dict['Add'] = add_node

                                matmul_node, ok = get_prev_node_by_input(model, add_node.input[0])
                                if ok == 0 and matmul_node.op_type == 'MatMul':
                                    logger.debug('get_last_group_pattern_four, got MatMul node: {}'.format(matmul_node.name))
                                    shapeA = values.get_tensor_shape_by_name(model, matmul_node.input[0])
                                    inputB, shapeB = values.get_init_value_and_shape(model, matmul_node.input[1])

                                    if isinstance(inputB, list) and inputB == []:
                                        logger.debug('inputB is not in initilizer')
                                        inputB = values.get_constant_value(model, matmul_node.input[1])

                                    if len(shapeA) == 2 and len(shapeB) == 2:
                                        logger.debug('get_last_group_pattern_four, got MatMul node: {}'.format(matmul_node.name))
                                        node_dict['MatMul'] = matmul_node
                                        node_dict['matmulA_Shape'] = shapeA
                                        node_dict['inputB'] = inputB
                                        node_dict['matmulB_Shape'] = shapeB

                                        rs_node2, ok = get_prev_node_by_input(model, matmul_node.input[0])
                                        if ok == 0 and rs_node2.op_type == 'Reshape':
                                            logger.debug('get_last_group_pattern_four, got Reshape node2: {}'.format(rs_node2.name))
                                            node_dict['Reshape2'] = rs_node2
                                            res = 0
                                            break

    return node_dict, res

#Reshape->MatMul->Add->Reshape->Transpose->Split
def handle_last_group_pattern_four(model):
    node_dict, ok = get_last_group_pattern_four(model)
    if ok == 0:
        logger.debug('start handle_last_group')
        matmul_node = node_dict['MatMul']
        rs_node = node_dict['Reshape']
        add_node = node_dict['Add']
        rs_node2 = node_dict['Reshape2']

        ###add transpose
        ts_name = rs_node2.name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        rs2_output_shape = values.get_tensor_shape_by_name(model, rs_node2.output[0])
        ts_output_shape = [rs2_output_shape[1], rs2_output_shape[0]]
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, ts_output_shape)
        
        ts_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts_name,
                                            inputs=[rs_node2.output[0]],
                                            outputs=[ts_output_name],
                                            perm=[1,0])

        model.graph.value_info.append(transpose_output)

        ###add reshape
        rs_name = rs_node2.name + '_reshape_2_'
        rs_output_name = rs_name + '_output_'
        rs_output_shape = [1, ts_output_shape[0], 1, ts_output_shape[1]]
        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.FLOAT, rs_output_shape)

        const_shape2_name = rs_node2.name + '_reshape2_data_'
        
        const_shape2_tensor = onnx.helper.make_tensor(name=const_shape2_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs_output_shape)],
                            vals=rs_output_shape)

        model.graph.initializer.append(const_shape2_tensor)

        rs_node_ = onnx.helper.make_node(
                                        'Reshape',
                                        name=rs_name,
                                        inputs=[ts_output_name, const_shape2_name],
                                        outputs=[rs_output_name])

        model.graph.value_info.append(rs_output)

        insert_node(model, rs_node_, matmul_node)
        matmul_node.input[0] = rs_output_name 

        insert_node(model, ts_node, rs_node_)

        #MatMul-->Conv
        matmul_node.op_type = 'Conv'
        const_x_name = matmul_node.name + '_to_conv_x_'

        v = node_dict['inputB']
        old_dims = [node_dict['matmulB_Shape'][0], node_dict['matmulB_Shape'][1]]
        dims_ = [node_dict['matmulB_Shape'][1], node_dict['matmulB_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('+++A.shape: {}'.format(A.shape))
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            logger.debug('---A.shape: {}'.format(A.shape))
            A = A.flatten()

        A = A.tolist()  
        const_x_tensor = onnx.helper.make_tensor(name=const_x_name,
                            data_type=onnx.TensorProto.FLOAT,
                            dims=dims_,
                            vals=A)

        model.graph.initializer.append(const_x_tensor)
        matmul_node.input[1] = const_x_name

        attr = onnx.helper.make_attribute('dilations', [1, 1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('group', 1)
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('kernel_shape', [1,1])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('pads', [0,0,0,0])
        matmul_node.attribute.append(attr)

        attr = onnx.helper.make_attribute('strides', [1,1])
        matmul_node.attribute.append(attr)        

        matmul_node.input.append(add_node.input[1])

        #mm_output_shape = values.get_tensor_shape_by_name(model, matmul_node.output[0])
        conv_output_shape = rs_output_shape#[mm_output_shape[0], mm_output_shape[2], 1, mm_output_shape[1]]
        conv_output_shape[1] = node_dict['matmulB_Shape'][1]
        update_tensor_shape(model, matmul_node.output[0], conv_output_shape) 

        ###########
        add_node.op_type = 'Reshape'
        reshape_output = add_node.output[0]

        const_shape_name = add_node.name + '_to_reshape_'

        add_output_shape = values.get_tensor_shape_by_name(model, add_node.output[0])
        rs2_output_shape = [add_output_shape[1], add_output_shape[0]]

        const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(rs2_output_shape)],
                            vals=rs2_output_shape)

        model.graph.initializer.append(const_shape_tensor)

        add_node.input[1] = const_shape_name

        update_tensor_shape(model, add_node.output[0], rs2_output_shape)

        ###add transpose
        ts2_name = add_node.name + '_transpose_'
        ts2_output_name = ts2_name + '_output_'
        ts2_output_shape = [rs2_output_shape[1], rs2_output_shape[0]]
        transpose_output = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.FLOAT, ts2_output_shape)
        
        ts2_node = onnx.helper.make_node(
                                            'Transpose',
                                            name=ts2_name,
                                            inputs=[add_node.output[0]],
                                            outputs=[ts2_output_name],
                                            perm=[1,0])

        model.graph.value_info.append(transpose_output)

        insert_node(model, ts2_node, rs_node)
        rs_node.input[0] = ts2_output_name

'''
if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/bert_sst2_sim.onnx')
    #model = onnx.load('./bert_sst2_sub1.onnx')
    #model = onnx.load('./decoder_model_bs10_sim.onnx')
    model = onnx.load('./bert_squad_v1_sim1.onnx')
    #model = onnx.load('./bert_sub2.onnx')
    #model = onnx.load('/home/zqiu/models/bert_cls_sim1.onnx')

    mha_optimizer(model)

    onnx.save(model, './hs3.onnx')
'''
    