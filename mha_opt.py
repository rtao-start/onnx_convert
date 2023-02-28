import onnx
import sys
import values
import numpy as np

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
                                node_dict['addA'] = addA
                                node_dict['matmul_AShape'] = shapeA
                                node_dict['inputB'] = inputB
                                node_dict['matmul_BShape'] = shapeB
    elif ok == 0:
        res = 1
        node_list = [input_pre]
        node_dict['node_list'] = node_list

    return node_dict, res

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

def cvt_matmul_add_to_conv(model, matmul_dict):
    if matmul_dict['next'][0].op_type != 'Transpose':
        current_inputA_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[0])
        current_inputB_shape = values.get_tensor_shape_by_name(model, matmul_dict['current'].input[1])

        print('current_inputA_shape: ', current_inputA_shape) 
        print('current_inputB_shape: ', current_inputB_shape)

        if matmul_dict['A_MatMul_Add'] == True:
            print('A inputA shape:{}, inputB shape:{}'.format(matmul_dict['A_matmul_AShape'], matmul_dict['A_matmul_BShape']))
            print('B inputA shape:{}, inputB shape:{}'.format(matmul_dict['B_matmul_AShape'], matmul_dict['B_matmul_BShape']))
            for node in matmul_dict['pathA']:
                if node.op_type == 'Matmul':
                    break

def get_matmul_list(model):
    matmul_list = []

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

                matmul_dict['pathB'] = node_dictB['node_list']
                matmul_dict['B_MatMul_Add'] = False
                if res2 == 0:
                    matmul_dict['B_MatMul_Add'] = True
                    matmul_dict['B_addA'] = node_dictB['addA']
                    matmul_dict['B_matmul_AShape'] = node_dictB['matmul_AShape']
                    matmul_dict['B_inputB'] = node_dictB['inputB']
                    matmul_dict['B_matmul_BShape'] = node_dictB['matmul_BShape']

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

def mha_optimizer(model):
    return model

if __name__ == "__main__":
    #model = onnx.load('/home/zqiu/models/decoder_sub1.onnx')
    model = onnx.load('./decoder_sub2.onnx')
    #mha_optimizer(model)
    get_matmul_list(model)
    #onnx.save(model, './hs.onnx')

    