import onnx
import sys
import values
import numpy as np

def merge_swish(model):
    dict_sm = {}
    dict_mul = {}

    got_swish = False

    for node_id, node in enumerate(model.graph.node):
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #         ", op:", node.op_type, ', len(input):', len(node.input))

        if node.op_type == 'Sigmoid':
            dict_sm['input'] = node.input
            dict_sm['output'] = node.output
            dict_sm['id'] = node_id

        if node.op_type == 'Mul':
            if len(dict_sm) > 0 and node.input[0] == dict_sm['input'][0] and node.input[1] == dict_sm['output'][0]:
                dict_mul['input'] = node.input
                dict_mul['output'] = node.output
                dict_mul['id'] = node_id

                print('got swish pair:', dict_sm['input'], dict_sm['output'])
                print('got swish pair:', dict_mul['input'], dict_mul['output'])

                got_swish = True

                old_node = model.graph.node[dict_sm['id']] 
                model.graph.node.remove(old_node)

                swish_node = onnx.helper.make_node(
                                        name = '',
                                        op_type='Swish',
                                        inputs=dict_sm['input'],
                                        outputs=dict_mul['output'],
                                        domain='com.metax-tech',
                                        )

                model.graph.node.insert(dict_sm['id'], swish_node)

                old_node = model.graph.node[dict_mul['id']] 
                model.graph.node.remove(old_node)

                dict_sm = {}
                dict_mul = {} 
            else:
                print('clear Sigmoid and Tanh')
                dict_sm = {}

    if got_swish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1
        
        #onnx.save(model, output)

def merge_hard_swish(model):
    dict_sm = {}
    dict_mul = {}

    got_hard_swish = False

    for node_id, node in enumerate(model.graph.node):
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #         ", op:", node.op_type, ', len(input):', len(node.input))

        if node.op_type == 'HardSigmoid':
            dict_sm['input'] = node.input
            dict_sm['output'] = node.output
            dict_sm['id'] = node_id

        if node.op_type == 'Mul':
            if len(dict_sm) > 0 and node.input[0] == dict_sm['input'][0] and node.input[1] == dict_sm['output'][0]:
                dict_mul['input'] = node.input
                dict_mul['output'] = node.output
                dict_mul['id'] = node_id

                print('got hard_swish pair:', dict_sm['input'], dict_sm['output'])
                print('got hard_swish pair:', dict_mul['input'], dict_mul['output'])

                got_hard_swish = True

                old_node = model.graph.node[dict_sm['id']] 
                model.graph.node.remove(old_node)

                swish_node = onnx.helper.make_node(
                                        name = '',
                                        op_type='HardSwish',
                                        inputs=dict_sm['input'],
                                        outputs=dict_mul['output'],
                                        domain='com.metax-tech',
                                        )

                model.graph.node.insert(dict_sm['id'], swish_node)

                old_node = model.graph.node[dict_mul['id']] 
                model.graph.node.remove(old_node)

                dict_sm = {}
                dict_mul = {} 
            else:
                print('clear HardSigmoid')
                dict_sm = {}

    if got_hard_swish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1
        
        #onnx.save(model, output)

def merge_hard_swish2(model):
    dict_add = {}
    dict_clip = {}
    dict_mul = {}
    dict_div = {}

    got_swish = False

    search = True

    while search == True:
        search = False
        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            found_add = False
            if node.op_type == 'Add':
                addB = values.get_init_value(model, node.input[1])
                print('addB:', addB)

                if isinstance(addB, list) and addB == []:
                    print('addB is not in initilizer')
                    continue

                if addB[0] != 3:
                    print('this is not the add-node which we wanted(value B is not 3)...')
                    continue

                if isinstance(addB, np.ndarray) == True:
                    if addB.shape != (1, ):
                        print('this is not the add-node which we wanted(shape is wrong)...')
                        continue
                else:        
                    if len(addB) != 1:
                        print('this is not the add-node which we wanted(list len is wrong)...')
                        continue

                dict_add['input'] = node.input
                dict_add['output'] = node.output
                dict_add['id'] = node_id
                print('got match add node:', node.name)

            if node.op_type == 'Clip':
                if dict_add and node.input[0] == dict_add['output'][0]:
                    clip_min = values.get_init_value(model, node.input[1])
                    print('clip_min:', clip_min)

                    clip_max = values.get_init_value(model, node.input[2])
                    print('clip_max:', clip_max)

                    if (isinstance(clip_min, list) and clip_min == []) or (isinstance(clip_max, list) and clip_max == []):
                        print('clip_min or clip_max is not in initilizer')
                        continue

                    if clip_min[0] != 0:
                        print('this is not the clip-node which we wanted(min is not 0)...')
                        dict_add = {}
                        continue

                    if isinstance(clip_min, np.ndarray) == True:
                        if clip_min.shape != (1, ):
                            print('this is not the clip-node which we wanted(shape is wrong)...')
                            dict_add = {}
                            continue
                    else:        
                        if len(clip_min) != 1:
                            print('this is not the clip-node which we wanted(list len is wrong)...')
                            dict_add = {}
                            continue    

                    if clip_max[0] != 6:
                        print('this is not the clip-node which we wanted(max is not 6)...')
                        continue

                    if isinstance(clip_max, np.ndarray) == True:
                        if clip_max.shape != (1, ):
                            print('this is not the clip-node which we wanted(shape is wrong)...')
                            dict_add = {}
                            continue
                    else:        
                        if len(clip_max) != 1:
                            print('this is not the clip-node which we wanted(list len is wrong)...')
                            dict_add = {}
                            continue           

                    dict_clip['input'] = node.input
                    dict_clip['output'] = node.output
                    dict_clip['id'] = node_id

                    print('got first pair:', dict_clip['input'], dict_clip['output'])
                else:
                    print('clear dict_add:', dict_add)
                    dict_add = {}    

            if node.op_type == 'Mul':
                if dict_add and dict_clip and node.input[1] == dict_clip['output'][0] and node.input[0] == dict_add['input'][0]:
                    dict_mul['input'] = node.input
                    dict_mul['output'] = node.output
                    dict_mul['id'] = node_id

                    print('got second pair:', dict_mul['input'], dict_mul['output'])

                else:
                    print('clear dict_add and dict_clip')
                    print('dict_add:', dict_add)
                    print('dict_clip:', dict_clip)
                    dict_add = {}
                    dict_clip = {} 

            if node.op_type == 'Div':
                if dict_mul and node.input[0] == dict_mul['output'][0]:
                    dict_div['input'] = node.input
                    dict_div['output'] = node.output
                    dict_div['id'] = node_id

                    divB = values.get_init_value(model, node.input[1])
                    print('divB:', divB)

                    if divB[0] != 6:
                        print('this is not the div-node which we wanted(value B is not 6)...')
                        continue

                    if isinstance(divB, np.ndarray) == True:
                        if divB.shape != (1, ):
                            print('this is not the div-node which we wanted(shape is wrong)...')
                            continue
                    else:        
                        if len(divB) != 1:
                            print('this is not the div-node which we wanted(list len is wrong)...')
                            continue

                    ###################################
                    old_node = model.graph.node[dict_add['id']] 
                    model.graph.node.remove(old_node)

                    swish_node = onnx.helper.make_node(
                                            name = '',
                                            op_type='HardSwish',
                                            inputs=[dict_add['input'][0]],
                                            outputs=dict_div['output'],
                                            domain='com.metax-tech'
                                            )

                    model.graph.node.insert(dict_add['id'], swish_node)

                    old_node = model.graph.node[dict_div['id']] 
                    model.graph.node.remove(old_node)

                    old_node = model.graph.node[dict_mul['id']] 
                    model.graph.node.remove(old_node)

                    old_node = model.graph.node[dict_clip['id']] 
                    model.graph.node.remove(old_node)

                    dict_add = {}
                    dict_clip = {}
                    dict_mul = {} 
                    ###############################

                    got_swish = True
                    search = True
                    break       
                else:
                    print('clear dict_add and dict_clip, ')
                    print('dict_add:', dict_add)
                    print('dict_clip:', dict_clip)
                    print('dict_mul:', dict_mul)
                    dict_add = {}
                    dict_clip = {}
                    dict_mul = {} 

    if got_swish == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1

    return model

def merge_swish_and_hard_swish(model):
    merge_swish(model)
    merge_hard_swish(model)
    merge_hard_swish2(model)

    return model

    