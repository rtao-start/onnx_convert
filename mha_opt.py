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

def get_matmul_input_path_pattern_one(model, input_name):
    res = -1

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

                            ###############################
    return res

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

            res1 = get_matmul_input_path_pattern_one(model, inputA)
            res2 = get_matmul_input_path_pattern_one(model, inputB)     

def mha_optimizer(model):
    dict_matmul = {}
    dict_add = {}
    dict_reshape = {}
    dict_transpose = {}

    wait_for_finish = 0

    list_matmul = [{}]*5
    list_add = [{}]*5
    list_reshape = [{}]*5

    list_transpose = [{}]*5

    got_swish = False

    search = True

    while search == True:
        search = False
        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type)

            if node.op_type == 'MatMul':
                shapeA = values.get_tensor_shape_by_name(model, node.input[0])
                if len(shapeA) != 3:
                    print('len(inputA) is not 3, ignore this MAtMul~', len(shapeA))
                    continue

                inputB, shapeB = values.get_init_value_and_shape(model, node.input[1])

                if isinstance(inputB, list) and inputB == []:
                    print('inputB is not in initilizer')
                    inputB = values.get_constant_value(model, node.input[1])
                    if inputB == []:
                        print('inputB is not in constant node list')
                        continue

                if len(shapeB) != 2:
                    print('this is not the matmul-node which we wanted(len(shapeB) is not 2)...')
                    continue

                list_matmul[wait_for_finish%5]['input'] = node.input
                list_matmul[wait_for_finish%5]['output'] = node.output
                list_matmul[wait_for_finish%5]['id'] = node_id

                #dict_matmul['input'] = node.input
                #dict_matmul['output'] = node.output
                #dict_matmul['id'] = node_id

                print('got match MatMul node:', node.name, wait_for_finish)

            if node.op_type == 'Add':
                #if dict_matmul and node.input[1] == dict_matmul['output'][0]:
                if list_matmul[wait_for_finish%5] and node.input[1] == list_matmul[wait_for_finish%5]['output'][0]:    
                    addA, shapeA = values.get_init_value_and_shape(model, node.input[0])
                    if isinstance(addA, list) and addA == []:
                        print('addA is not in initilizer')
                        addA = values.get_constant_value(model, node.input[1])
                        if addA == []:
                            #dict_matmul = {}
                            list_matmul[wait_for_finish%5] = {}
                            print('addA is not in constant node list~')
                            continue

                    if len(shapeA) != 1:
                        print('this is not the add-node which we wanted(len(shapeA) is not 1)...')
                        #dict_matmul = {}
                        list_matmul[wait_for_finish%5] = {}
                        continue

                    list_add[wait_for_finish%5]['input'] = node.input
                    list_add[wait_for_finish%5]['output'] = node.output
                    list_add[wait_for_finish%5]['id'] = node_id

                    #dict_add['input'] = node.input
                    #dict_add['output'] = node.output
                    #dict_add['id'] = node_id

                    print('got match add node:', node.name, wait_for_finish)
                else:
                    print('clear dict 1')
                    #dict_matmul = {}
                    list_matmul[wait_for_finish%5] = {}    

            if node.op_type == 'Reshape':
                #if dict_add and node.input[0] == dict_add['output'][0]:
                if list_add[wait_for_finish%5] and node.input[0] == list_add[wait_for_finish%5]['output'][0]: 
                    data, shape = values.get_init_value_and_shape(model, node.input[1])
                    if isinstance(data, list) and data == []:
                        print('reshape_data is not in initilizer')
                        data = values.get_constant_value(model, node.input[1])
                        if data == []:
                            #dict_matmul = {}
                            #dict_add = {}
                            list_matmul[wait_for_finish%5] = {}
                            list_add[wait_for_finish%5] = {}
                            print('reshape_data is not in constant node list~')
                            continue

                    if len(data) != 4:
                        print('this is not the reshape-node which we wanted(len(shape) is not 4)...', len(data))
                        #dict_matmul = {}
                        #dict_add = {}
                        list_matmul[wait_for_finish%5] = {}
                        list_add[wait_for_finish%5] = {}

                        continue

                    list_reshape[wait_for_finish%5]['input'] = node.input
                    list_reshape[wait_for_finish%5]['output'] = node.output
                    list_reshape[wait_for_finish%5]['id'] = node_id

                    #dict_reshape['input'] = node.input
                    #dict_reshape['output'] = node.output
                    #dict_reshape['id'] = node_id

                    wait_for_finish = wait_for_finish + 1 

                    print('got match Reshape node:', node.name, wait_for_finish)  
                else:
                    print('clear dict 2')
                    #dict_matmul = {}
                    #dict_add = {}
                    list_matmul[wait_for_finish%5] = {}
                    list_add[wait_for_finish%5] = {}

            if node.op_type == 'Transpose':
                print('got a Transpose node----', node.input[0])
                for ll in list_reshape:
                    if ll and node.input[0] == ll['output'][0]:
                    #if dict_reshape and node.input[0] == dict_reshape['output'][0]:
                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'value':
                                v = values.get_tensor_value(attr.t)
                                print('got transpose shape{} for{}'.format(v, node.name))

                        dict_transpose['input'] = node.input
                        dict_transpose['output'] = node.output
                        dict_transpose['id'] = node_id

                        got_swish = True
                        search = True

                        print('got MatMul+Add+Reshape+Transpose:', node.name)

                        break       
                    else:
                        print('clear dict 3')
                        #dict_matmul = {}
                        #dict_add = {}
                        #dict_reshape = {}
                        list_matmul[wait_for_finish%5] = {}
                        list_add[wait_for_finish%5] = {}
                        list_reshape[wait_for_finish%5] = {}

    return model

if __name__ == "__main__":
    model = onnx.load('/home/zqiu/models/decoder_sub1.onnx')
    #mha_optimizer(model)
    get_matmul_list(model)
    #onnx.save(model, './hs.onnx')

    