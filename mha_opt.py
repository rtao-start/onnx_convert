import onnx
import sys
import values
import numpy as np

transpose_node_map = {}

def get_node_by_output(model, output):
    n = model.graph.node[0]
    for node in model.graph.node:
        if output in node.output:
            return node, 0

    return n, -1

def get_node_by_pre_output(model, output):
    n = model.graph.node[0]
    for node in model.graph.node:
        if output in node.input:
            return node, 0

    return n, -1

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

    input_pre, ok = get_node_by_output(model, input_name)
    if ok == 0 and input_pre.op_type == 'Transpose':
        print('got match Transpose node:', input_pre.name)
        attributes = input_pre.attribute
        for attr in attributes:
            if attr.name == 'perm':
                v = values.get_tensor_value(attr.t)
                print('got transpose shape{} for{}'.format(v, input_pre.name))
                break
        
        input_p_pre, ok = get_node_by_output(model, input_pre.input[0])
        if ok == 0 and input_p_pre.op_type == 'Reshape':
            #####################
            data, shape = values.get_init_value_and_shape(model, input_p_pre.input[1])
            if isinstance(data, list) and data == []:
                print('reshape_data is not in initilizer')
                data = values.get_constant_value(model, input_p_pre.input[1])

            if len(data) == 4:
                print('got match Reshape node:', input_p_pre.name)
                ##################
                input_pp_pre, ok = get_node_by_output(model, input_p_pre.input[0])
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
                        input_ppp_pre, ok = get_node_by_output(model, input_pp_pre.input[1])
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

                                input_pppp_pre, ok = get_node_by_output(model, input_ppp_pre.input[0])
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

def handle_add_combination_pattern_one(model):
    ars_list = get_add_combination_pattern_one(model)
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
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.UNDEFINED, [add_output_shape[0], add_output_shape[2], add_output_shape[1]])
        
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
        rs_output_shape = [add_output_shape[2], add_output_shape[1]] 
        rs_output = onnx.helper.make_tensor_value_info(rs_output_name, onnx.TensorProto.UNDEFINED, rs_output_shape)

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
        rs2_output = onnx.helper.make_tensor_value_info(rs2_output_name, onnx.TensorProto.UNDEFINED, rs2_output_shape)

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
        ts2_output = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.UNDEFINED, [add_output_shape[0], add_output_shape[1], add_output_shape[2]])

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
            ts2_output = onnx.helper.make_tensor_value_info(ts2_output_name, onnx.TensorProto.UNDEFINED, add_output_shape)

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
        
        if node.op_type == 'Add':
            print('reuse Add to Reshape')
            orig_reshape_name = node.name
            node.op_type = 'Reshape'
            const_shape_name = node.name + '_to_reshape_'
            output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]] 
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(output_shape)],
                                vals=output_shape)

            model.graph.initializer.append(const_shape_tensor)
            node.input[0] = node.input[1]
            node.input[1] = const_shape_name 
            update_tensor_shape(model, node.output[0], output_shape)

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

            attr = onnx.helper.make_attribute('pads', [1,1,1,1])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('strides', [1,1])
            node.attribute.append(attr)        

            if isInputA == True:
                node.input.append(matmul_dict['A_addA'])
            else:
                node.input.append(matmul_dict['B_addA'])

            output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
            update_tensor_shape(model, node.output[0], output_shape) 
    
        if node.op_type == 'Transpose' and node.name != orig_matmul_name:
            print('reuse Transpose to Reshape')
            node.op_type = 'Reshape'
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
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.UNDEFINED, [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]])

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
            
        if node.op_type == 'Add':
            print('reuse Add to Reshape')
            orig_reshape_name = node.name
            node.op_type = 'Reshape'
            const_shape_name = node.name + '_to_reshape_'
            output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]] 
            const_shape_tensor = onnx.helper.make_tensor(name=const_shape_name,
                                data_type=onnx.TensorProto.INT64,
                                dims=[len(output_shape)],
                                vals=output_shape)

            model.graph.initializer.append(const_shape_tensor)
            if reuse_transpose == False:
                node.input[0] = node.input[1]
            else:
                node.input[0] = transpose_node_map[map_key].output[0]

            node.input[1] = const_shape_name 
            update_tensor_shape(model, node.output[0], output_shape)

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

            attr = onnx.helper.make_attribute('pads', [1,1,1,1])
            node.attribute.append(attr)

            attr = onnx.helper.make_attribute('strides', [1,1])
            node.attribute.append(attr)        

            node.input.append(matmul_dict['B_addA'])
            output_shape = [inputA_shape[0], inputA_shape[2], 1, inputA_shape[1]]
            update_tensor_shape(model, node.output[0], output_shape) 
    
        if node.op_type == 'Transpose' and node.name != orig_matmul_name:
            print('reuse Transpose to Reshape')
            node.op_type = 'Reshape'
            reshape_output = node.output[0]
            const_shape_name = node.name + '_to_reshape_'
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

    if 1==1:#isInputA == False:
        ts_name = const_shape_name + '_transpose_'
        ts_output_name = ts_name + '_output_'
        transpose_output = onnx.helper.make_tensor_value_info(ts_output_name, onnx.TensorProto.UNDEFINED, [current_inputB_shape[0], current_inputB_shape[1], current_inputB_shape[3], current_inputB_shape[2]])

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
    
def cvt_matmul_add_to_conv(model, matmul_dict):
    if matmul_dict['next'][0].op_type != 'Transpose':
        print('cvt_matmul_add_to_conv, AAAAAAAAAAAAAAAAA')
        if matmul_dict['A_MatMul_Add'] == True:
            do_convert_pattern_one(model, matmul_dict, True)

        if matmul_dict['B_MatMul_Add'] == True:
            do_convert_pattern_one(model, matmul_dict, False)
    else:
        if matmul_dict['B_MatMul_Add'] == True:
            print('cvt_matmul_add_to_conv, BBBBBBBBBBBBBBBBBBBB')
            do_convert_pattern_two(model, matmul_dict)           

def mha_optimizer(model):
    matmul_list = []

    handle_add_combination_pattern_one(model)

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
                next_node, ok = get_node_by_pre_output(model, node.output[0])
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
    #model = onnx.load('/home/zqiu/models/decoder_sub1.onnx')
    model = onnx.load('./decoder_sub2.onnx')
    mha_optimizer(model)
    #get_matmul_list(model)
    onnx.save(model, './hs.onnx')

    