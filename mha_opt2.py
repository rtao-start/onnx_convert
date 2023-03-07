import onnx
import sys
import values
import numpy as np

transpose_node_map = {}

def get_prev_node_by_input(model, input):
    n = model.graph.node[0]
    for node in model.graph.node:
        if input in node.output:
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
    # 根据插入Node的输出修改后续node的输入
    #follow_up_node.input[0] = insert_node.output[0]
    # 找到后续Node的索引位置，并将插入节点插入到graph中
    for follow_up_node_index, _follow_up_node in enumerate(model.graph.node):
        if _follow_up_node == follow_up_node:
            print("follow_up_node_index: ", follow_up_node_index)
            model.graph.node.insert(follow_up_node_index, insert_node)
            break

def get_matmul_input_path_pattern_one(model, input_name):
    res = -1

    #node_list = []
    node_dict = {}

    print('get_matmul_input_path_pattern_one, input_name:', input_name)

    input_pre, ok = get_prev_node_by_input(model, input_name)
    if ok == 0 and input_pre.op_type == 'Transpose':
        print('got match Transpose node:', input_pre.name)
        attributes = input_pre.attribute
        for attr in attributes:
            if attr.name == 'perm':
                v = values.get_tensor_value(attr.t)
                print('got transpose shape{} for{}'.format(v, input_pre.name))
                break
        
        input_p_pre, ok = get_prev_node_by_input(model, input_pre.input[0])
        if ok == 0 and input_p_pre.op_type == 'Reshape':
            #####################
            data, shape = values.get_init_value_and_shape(model, input_p_pre.input[1])
            if isinstance(data, list) and data == []:
                print('reshape_data is not in initilizer')
                data = values.get_constant_value(model, input_p_pre.input[1])

            if len(data) == 4:
                print('got match Reshape node:', input_p_pre.name)
                ##################
                input_pp_pre, ok = get_prev_node_by_input(model, input_p_pre.input[0])
                if ok == 0 and input_pp_pre.op_type == 'Add':
                    ################
                    addA_name = input_pp_pre.input[0]
                    addA, shapeA = values.get_init_value_and_shape(model, input_pp_pre.input[0])
                    if isinstance(addA, list) and addA == []:
                        print('addA is not in initilizer')
                        addA = values.get_constant_value(model, input_pp_pre.input[1])

                    if len(shapeA) == 1:
                        print('got match Add node:', input_pp_pre.name)
                        ###########
                        input_ppp_pre, ok = get_prev_node_by_input(model, input_pp_pre.input[1])
                        if ok == 0 and input_ppp_pre.op_type == 'MatMul':
                            ############################
                            shapeA = values.get_tensor_shape_by_name(model, input_ppp_pre.input[0])

                            inputB, shapeB = values.get_init_value_and_shape(model, input_ppp_pre.input[1])

                            if isinstance(inputB, list) and inputB == []:
                                print('inputB is not in initilizer')
                                inputB = values.get_constant_value(model, input_ppp_pre.input[1])

                            if len(shapeA) == 3 and len(shapeB) == 2:
                                print('got match MatMul node', input_ppp_pre.name)
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
                                    print('--- map key:', input_pppp_pre.output[0])
                                else:
                                    node_dict['prev'] = input_ppp_pre.input[0]
                                    print('pre node maybe input', input_ppp_pre.input[0])    
    elif ok == 0:
        res = 1
        node_list = [input_pre]
        node_dict['node_list'] = node_list

    return node_dict, res

def get_matmul_input_path_pattern_two(model, input_name):
    res = -1

    node_dict = {}

    print('get_matmul_input_path_pattern_two, input_name:', input_name)

    next_node, ok = get_next_node_by_output(model, input_name)
    if ok == 0 and next_node.op_type == 'Reshape':
        data, shape = values.get_init_value_and_shape(model, next_node.input[1])
        if isinstance(data, list) and data == []:
            print('---reshape_data is not in initilizer')
            data = values.get_constant_value(model, next_node.input[1])

        if len(data) == 3:
            print('----got match Reshape node:', next_node.name)

            n_next_node, ok = get_next_node_by_output(model, next_node.output[0])
            if ok == 0 and n_next_node.op_type == 'MatMul':
                #####################
                shapeA = values.get_tensor_shape_by_name(model, n_next_node.input[0])

                inputB, shapeB = values.get_init_value_and_shape(model, n_next_node.input[1])

                print('++++++++++++++++++shapeA, shapeB:', shapeA, shapeB)

                if isinstance(inputB, list) and inputB == []:
                    print('inputB is not in initilizer')
                    inputB = values.get_constant_value(model, n_next_node.input[1])

                if len(shapeA) == 3 and len(shapeB) == 2:
                    node_dict['matmul_AShape'] = shapeA
                    node_dict['inputB'] = inputB
                    node_dict['matmul_BShape'] = shapeB

                    print('----got match Matmul node:', n_next_node.name)
                    ####################
                    nn_next_node, ok = get_next_node_by_output(model, n_next_node.output[0])
                    if ok == 0 and nn_next_node.op_type == 'Add':
                        ################
                        addA_name = nn_next_node.input[0]
                        node_dict['addA'] = addA_name
                        addA, shapeA = values.get_init_value_and_shape(model, nn_next_node.input[0])
                        if isinstance(addA, list) and addA == []:
                            print('addA is not in initilizer')
                            addA = values.get_constant_value(model, nn_next_node.input[1])

                        if len(shapeA) == 1:
                            print('---got match Add node:', nn_next_node.name)
                            ###########
                            nnn_next_node, ok = get_next_node_by_output(model, nn_next_node.output[0])
                            if ok == 0 and nnn_next_node.op_type == 'Add':
                                print('----got match Add node2:', n_next_node.name)
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
                print('found match add node:', node.name)
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

            output = node.output[0]
            for add_node in add_list:
                if output in add_node.input:
                    match_add_node = add_node
                    break

            for mm_node in matmul_list:
                if output in mm_node.input:
                    match_matmul_node_list.append(mm_node)

            if match_add_node != None and len(match_matmul_node_list) == 3:
                print('found match add node:', node.name)
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

                    print('got sub and reducemean:', match_sub_node.name, match_rm_node.name)

                am_list.append(am)
 
    return am_list 

def handle_add_combination_pattern_two(model):
    am_list = get_add_combination_pattern_two(model)

    print('handle_add_combination_pattern_two------------')

    #if len(am_list):
    for am in am_list:
        #am = am_list[0]
        add_node = am['currentAdd']
        next_add_node = am['nextAdd']
        matmul_node_list = am['MatMulList'] 

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

def get_matmul_block_one(model, matmul_node):
    print('into get_matmul_block_one')

    res = -1
    node_dict = {}

    #input_next, ok = get_next_node_by_output(model, input_)
    input_next = matmul_node
    if input_next.op_type == 'MatMul':
        shapeA = values.get_tensor_shape_by_name(model, input_next.input[0])
        inputB, shapeB = values.get_init_value_and_shape(model, input_next.input[1])

        if isinstance(inputB, list) and inputB == []:
            print('inputB is not in initilizer')
            inputB = values.get_constant_value(model, input_next.input[1])

        if len(shapeA) == 3 and len(shapeB) == 2:
            print('--- got MatMul node', input_next.name)
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
                    print('--- got Add1 node', input_nnext.name)

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

                            print('--- got Div node', div_node.name)

                            input_nnnnext, ok = get_next_node_by_output(model, mul_node.output[0])
                            if ok == 0 and input_nnnnext.op_type == 'Mul':
                                #print('AAAAAAAAAAAAAAAAAAAA', input_nnnnext.name)
                                mulB, shapeB = values.get_init_value_and_shape(model, input_nnnnext.input[1])
                                if len(mulB) > 0:
                                    #######################
                                    ##############################
                                    print('--- got mul2 node', input_nnnnext.name)
                                    input_nnnnnext, ok = get_next_node_by_output(model, input_nnnnext.output[0])
                                    if ok == 0 and input_nnnnnext.op_type == 'MatMul':
                                        shapeA = values.get_tensor_shape_by_name(model, input_nnnnnext.input[0])
                                        inputB, shapeB = values.get_init_value_and_shape(model, input_nnnnnext.input[1])

                                        if isinstance(inputB, list) and inputB == []:
                                            print('inputB is not in initilizer')
                                            inputB = values.get_constant_value(model, input_nnnnnext.input[1])

                                        if len(shapeA) == 3 and len(shapeB) == 2:
                                            print('--- got MatMul2 node', input_nnnnnext.name)
                                            #node_list = [input_nnnnnext, input_pp_pre, input_p_pre, input_pre]
                                            #node_dict['node_list'] = node_list
                                            node_dict['MatMul2'] = input_nnnnnext
                                            node_dict['matmulA2_Shape'] = shapeA
                                            node_dict['inputB2'] = inputB
                                            node_dict['matmulB2_Shape'] = shapeB

                                            input_nnnnnnext, ok = get_next_node_by_output(model, input_nnnnnext.output[0])
                                            if ok == 0 and input_nnnnnnext.op_type == 'Add':
                                                print('--- got Add2 node:', input_nnnnnnext.name)
                                                #addA_name = input_nnnnnnext.input[0]
                                                #if len(shapeA) == 1:
                                                node_dict['Add2'] = input_nnnnnnext
                                                next_node, ok = get_next_node_by_output(model, input_nnnnnnext.output[0])
                                                if ok == 0 and next_node.op_type == 'Add':
                                                    print('--- got last Add node:', next_node.name)
                                                    res = 0
                                                    node_dict['NextAdd'] = next_node 

    return node_dict, res

def get_mul_add_block(model):
    node_list = []
    for node in model.graph.node:
        if node.op_type == 'Mul':
            #print('got mul:', node.name)

            is_init = False

            for init in model.graph.initializer:
                if init.name == node.input[0]:
                    is_init = True
                    break

            if is_init == False:
                dataA = values.get_constant_value(model, node.input[0])
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
                                print('got it~')
                                matmul_node = next_node_list[0]
                                if next_node_list[1].op_type == 'MatMul':
                                    matmul_node = next_node_list[1]

                                node_dict, ret = get_matmul_block_one(model, matmul_node)
                                if ret == 0:
                                    #print('got node dict:', node_dict)
                                    node_dict['currentAdd'] = next_node
                                    node_list.append(node_dict)

    return node_list

def handle_mul_add_block(model):
    node_list = get_mul_add_block(model)

    #if len(node_list) > 0:
    for node_dict in node_list:
        print('++++++++++++++++++++++')
        print('Add1:', node_dict['Add1'].name)
        print('Add2:', node_dict['Add2'].name)
        print('++++++++++++++++++++++')

        matmul1 = node_dict['MatMul1']
        add1 = node_dict['Add1']

        matmul2 = node_dict['MatMul2']
        add2 = node_dict['Add2']

        currentAdd = node_dict['currentAdd']
        nextAdd = node_dict['NextAdd']

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

        nextAdd.input[0] = ts_output_name

        #MatMul1--->Conv
        matmul1.op_type = 'Conv'
        print('-----reuse MatMul to Conv')
        const_x_name = matmul1.name + '_to_conv_x_'

        v = node_dict['inputB1']
        old_dims = [node_dict['matmulB1_Shape'][0], node_dict['matmulB1_Shape'][1]]
        dims_ = [node_dict['matmulB1_Shape'][1], node_dict['matmulB1_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            print('+++A.shape:', A.shape)
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            print('---A.shape:', A.shape)
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
        print('-----rs_output_shape:', rs_output_shape)

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
        print('++++++reuse MatMul to Conv')
        const_x_name = matmul2.name + '_to_conv_x_'

        v = node_dict['inputB2']
        old_dims = [node_dict['matmulB2_Shape'][0], node_dict['matmulB2_Shape'][1]]
        dims_ = [node_dict['matmulB2_Shape'][1], node_dict['matmulB2_Shape'][0],1,1]
        
        if isinstance(v, np.ndarray) == True:
            A = v.reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            print('+++A.shape:', A.shape)
            A = A.flatten()
        else:    
            A = np.array(v).reshape(*old_dims)
            A = A.transpose()
            A = A.reshape(*dims_)
            print('---A.shape:', A.shape)
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
        print('ZZZZ HHHHH----', add2.input[0], B)
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

        print('ZZZZ HHHHH+++++++', add2.input[0])
        add2.input[0] = add2.input[1]
        print('ZZZZ HHHHH----', add2.input[0])
        add2.input[1] = const_shape_name

        update_tensor_shape(model, add2.output[0], rs2_output_shape)

        ######update tensor shape
        div_output_shape = values.get_tensor_shape_by_name(model, div_node.output[0])
        new_shape = [div_output_shape[0], div_output_shape[2], div_output_shape[1]]
        update_tensor_shape(model, div_node.output[0], new_shape)

        erf_node, ok = get_next_node_by_output(model, div_node.output[0])
        if ok == 0 and erf_node.op_type == 'Erf':
            erf_output_shape = values.get_tensor_shape_by_name(model, erf_node.output[0])
            new_shape = [erf_output_shape[0], erf_output_shape[2], erf_output_shape[1]]
            update_tensor_shape(model, erf_node.output[0], new_shape)

            add_node_internal, ok = get_next_node_by_output(model, erf_node.output[0])
            if ok == 0 and add_node_internal.op_type == 'Add':
                addi_output_shape = values.get_tensor_shape_by_name(model, add_node_internal.output[0])
                new_shape = [addi_output_shape[0], addi_output_shape[2], addi_output_shape[1]]
                update_tensor_shape(model, add_node_internal.output[0], new_shape)
        
            mul_node1, ok = get_next_node_by_output(model, add_node_internal.output[0])
            if ok == 0 and mul_node1.op_type == 'Mul':
                mul1_output_shape = values.get_tensor_shape_by_name(model, mul_node1.output[0])
                new_shape = [mul1_output_shape[0], mul1_output_shape[2], mul1_output_shape[1]]
                update_tensor_shape(model, mul_node1.output[0], new_shape)

            mul_node2, ok = get_next_node_by_output(model, mul_node1.output[0])
            if ok == 0 and mul_node2.op_type == 'Mul':    
                mul2_output_shape = values.get_tensor_shape_by_name(model, mul_node2.output[0])
                new_shape = [mul2_output_shape[0], mul2_output_shape[2], mul2_output_shape[1]]
                update_tensor_shape(model, mul_node2.output[0], new_shape)

        ######insert Transpose before ReduceMean and Sub
        update_tensor_shape(model, nextAdd.output[0], rs2_output_shape)

        rm_sub, ok = get_all_next_node_by_output(model, nextAdd.output[0])
        if ok == 0 and len(rm_sub) == 2:
            print('got reducemean and sub node---')
            sub_node = rm_sub[0]
            rm_node = rm_sub[1]

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

def handle_add_combination_pattern_one(model):
    ars_list = get_add_combination_pattern_one(model)
    print('handle_add_combination_pattern_one,ars_list:', ars_list)
    sys.exit(0)

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


        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.INT64, rs_output_shape)

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

    print('{}:{}'.format(desp, node_print))

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

    print('current_inputA_shape: ', current_inputA_shape) 
    print('current_inputB_shape: ', current_inputB_shape)

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

    print('A inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))
    print('B inputA shape:{}, inputB shape:{}'.format(matmul_dict['B_matmul_AShape'], matmul_dict['B_matmul_BShape']))
    
    remove_matmul = False
    matmul_input0 = ''
    matmul_input0_shape = []

    for node in path_node:
        if node.op_type == 'MatMul':
            if inputA_shape[1] != inputB_shape[0]:
                orig_matmul_name = node.name
                print('matmul+add-->conv, need same channel', inputA_shape[1], inputB_shape[0])
                node.op_type = 'Transpose'
                attr = onnx.helper.make_attribute('perm', [0,2,1])
                node.attribute.append(attr)
                del node.input[1:]
                update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]])

                transpose_node_map[map_key] = node
                print('map_key is', map_key) 
            else:
                print('Delete MatMul node:', node.name)
                matmul_input0 = node.input[0]
                matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)

                model.graph.node.remove(node)
                remove_matmul = True
                bert_mode = 0

        if node.op_type == 'Add':
            print('reuse Add to Reshape')
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
            print('reuse Reshape to Conv')
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
                print('+++A.shape:', A.shape)
                A = A.flatten()
            else:    
                A = np.array(v).reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                print('---A.shape:', A.shape)
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
            print('reuse Transpose to Reshape')
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

    print('current_inputA_shape: ', current_inputA_shape) 
    print('current_inputB_shape: ', current_inputB_shape)

    inputA_shape = matmul_dict['B_matmul_AShape']
    inputB_shape = matmul_dict['B_matmul_BShape']
    path_node = matmul_dict['pathB']

    print('A inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))
    print('B inputA shape:{}, inputB shape:{}'.format(matmul_dict['B_matmul_AShape'], matmul_dict['B_matmul_BShape']))
    
    reuse_transpose = False

    remove_matmul = False
    matmul_input0 = ''
    matmul_input0_shape = []

    for node in path_node:
        if node.op_type == 'MatMul':
            if inputA_shape[1] != inputB_shape[0]:
                map_key = node.input[0]
                if map_key in transpose_node_map.keys():
                    #transpose_node_map[map_key]
                    model.graph.node.remove(node)
                    reuse_transpose = True
                else:     
                    orig_matmul_name = node.name
                    print('matmul+add-->conv, need same channel', inputA_shape[1], inputB_shape[0])
                    node.op_type = 'Transpose'
                    attr = onnx.helper.make_attribute('perm', [0,2,1])
                    node.attribute.append(attr)
                    del node.input[1:]
                    update_tensor_shape(model, node.output[0], [inputA_shape[0], inputA_shape[2], inputA_shape[1]]) 
            else:
                print('------Delete MatMul node:', node.name)
                matmul_input0 = node.input[0]
                matmul_input0_shape = values.get_tensor_shape_by_name(model, matmul_input0)

                model.graph.node.remove(node)
                remove_matmul = True


        if node.op_type == 'Add':
            print('----reuse Add to Reshape', node.name)
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
            print('reuse Reshape to Conv')
            node.op_type = 'Conv'
            const_x_name = node.name + '_to_conv_x_'

            v = matmul_dict['B_inputB']

            if isinstance(v, np.ndarray) == True:
                A = v.reshape(matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1])
                A = A.transpose()
                A = A.reshape(matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1)
                print('+++A.shape:', A.shape)
                A = A.flatten()
            else:    
                A = np.array(v).reshape(matmul_dict['B_matmul_BShape'][0], matmul_dict['B_matmul_BShape'][1])
                A = A.transpose()
                A = A.reshape(matmul_dict['B_matmul_BShape'][1], matmul_dict['B_matmul_BShape'][0], 1, 1)
                print('---A.shape:', A.shape)
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
            print('reuse Transpose to Reshape')
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

    print('----- inputA shape:{}, inputB shape:{}'.format(inputA_shape, inputB_shape))

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
            print('----reuse Reshape')

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
            print('---- reuse Matmul to Conv')

            node.op_type = 'Conv'
            const_x_name = node.name + '_to_conv_x_'

            v = matmul_dict['inputB']

            old_dims = [matmul_dict['matmul_BShape'][0], matmul_dict['matmul_BShape'][1]]

            dims_ = [matmul_dict['matmul_BShape'][1], matmul_dict['matmul_BShape'][0],1,1]
            
            if isinstance(v, np.ndarray) == True:
                A = v.reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                print('+++    A.shape:', A.shape)
                A = A.flatten()
            else:    
                A = np.array(v).reshape(*old_dims)
                A = A.transpose()
                A = A.reshape(*dims_)
                print('---    A.shape:', A.shape)
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
            print('----- reuse Add to Reshape')

            node.op_type = 'Reshape'

            const_shape_name = node.name + '_to_reshape_'

            output_shape = [rs2_output_shape[0], rs2_output_shape[1], rs2_output_shape[3]] 

            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(output_shape)],
                                vals=output_shape)

            model.graph.initializer.append(const_shape_tensor)

            node.input[0] = node.input[1]
            node.input[1] = const_shape_name 
            update_tensor_shape(model, node.output[0], output_shape)

            next_node, ok = get_next_node_by_output(model, node.output[0])
            update_tensor_shape(model, next_node.output[0], output_shape)

def cvt_matmul_add_to_conv(model, matmul_dict):
    if matmul_dict['next'][0].op_type == 'Div':
        bert_mode = -1
        print('cvt_matmul_add_to_conv, AAAAAAAAAAAAAAAAA', matmul_dict['nnext'][0].name)
        if matmul_dict['A_MatMul_Add'] == True:
            bert_mode = do_convert_pattern_one(model, matmul_dict, True)

        if matmul_dict['B_MatMul_Add'] == True:
            bert_mode = do_convert_pattern_one(model, matmul_dict, False)

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
            print('cvt_matmul_add_to_conv, BBBBBBBBBBBBBBBBBBBB')
            do_convert_pattern_two(model, matmul_dict)

        if matmul_dict['A_MatMul_Add'] == False:
            print('cvt_matmul_add_to_conv, CCCCCCCCCCCCCCCCCCCCC')
            path_node = matmul_dict['pathA']

            node= path_node[0]

            if node.op_type == 'Where' or node.op_type == 'Softmax':
                shape = values.get_tensor_shape_by_name(model, node.output[0])

                ###add transpose
                print('insert Transpose before', node.name)
                ts_name = node.name + '_transpose_'
                ts_output_name = ts_name + '_output_'
                transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.FLOAT, [shape[0], shape[1],shape[3],shape[2]])
                
                ts_node = onnx.helper.make_node(
                                                    'Transpose',
                                                    name=ts_name,
                                                    inputs=[node.output[0]],
                                                    outputs=[ts_output_name],
                                                    perm=[0,1,3,2])

                model.graph.value_info.append(transpose_output)

                insert_node(model, ts_node, matmul_dict['current'])

                matmul_dict['current'].input[1] = ts_output_name

                next_node = matmul_dict['next'][0]
                nnext_node = matmul_dict['nnext'][0]

                op_dict, ok = get_matmul_input_path_pattern_two(model, next_node.output[0])

                if op_dict and ok == 0:
                    for node in op_dict['node_list']: 
                        print('got matmul+add path(pattern 2):', node.name)
                         
                    do_convert_pattern_three(model, op_dict, next_node)            

def mha_optimizer(model):
    matmul_list = []

    #handle_add_combination_pattern_one(model)
    handle_add_combination_pattern_two(model)

    handle_mul_add_block(model)

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
                print('skip MatMul:', node.name)
                continue

            node_dictA, res1 = get_matmul_input_path_pattern_one(model, inputA)
            node_dictB, res2 = get_matmul_input_path_pattern_one(model, inputB)

            if res1 > -1:
                print_matmul_input_path(node_dictA['node_list'], 'node_listA')
            if res2 > -1:
                print_matmul_input_path(node_dictB['node_list'], 'node_listB')

            if res1 > -1 or res2 > -1:
                next_node, _ = get_next_node_by_output(model, node.output[0])
                nnext_node, _ = get_next_node_by_output(model, next_node.output[0])

                matmul_dict = {}
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
        print('stat MatMul: {}, next: {}, op_type: {}'.format(ll['name'], ll['next'][0].name,ll['next'][0].op_type))
        print('------pathA:')
        for node in ll['pathA']:
            print('   ', node.name)

        print('------pathB:')
        for node in ll['pathB']:
            print('   ', node.name)

        cvt_matmul_add_to_conv(model, ll)

if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/decoder_model_bs10.onnx')
    #model = onnx.load('./decoder_sub2.onnx')
    #model = onnx.load('./bert_sub1.onnx')
    model = onnx.load('/home/zqiu/models/bert_cls_sim1.onnx')
    mha_optimizer(model)
    #get_matmul_list(model)
    onnx.save(model, './sim1.onnx')

    