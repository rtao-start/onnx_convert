import onnx
import sys
import values
import numpy as np

def mha_optimizer(model):
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
                    #continue
                    addB = values.get_constant_value(model, node.input[1])
                    if addB == []:
                        print('addB is not in constant node list')
                        continue
                    else:
                        print('addB is', addB, type(addB))    

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
                    if isinstance(clip_min, list) and clip_min == []:
                        print('clip_min is not in initilizer')
                        clip_min = values.get_constant_value(model, node.input[1])
                        if clip_min == []:
                            dict_add = {}
                            print('clip_min is not in constant node list~')
                            continue

                    print('clip_min:', clip_min)

                    clip_max = values.get_init_value(model, node.input[2])
                    if isinstance(clip_max, list) and clip_max == []:
                        print('clip_max is not in initilizer')
                        clip_max = values.get_constant_value(model, node.input[2])
                        if clip_max == []:
                            dict_add = {}
                            print('clip_max is not in constant node list~')
                            continue

                    print('clip_max:', clip_max)

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

            if node.op_type == 'Div':
                if dict_clip and node.input[0] == dict_clip['output'][0]:
                    dict_div['input'] = node.input
                    dict_div['output'] = node.output
                    dict_div['id'] = node_id

                    divB = values.get_init_value(model, node.input[1])
                    if isinstance(divB, list) and divB == []:
                        print('divB is not in initilizer')
                        divB = values.get_constant_value(model, node.input[1])
                        if divB == []:
                            dict_add = {}
                            dict_clip = {}
                            dict_div = {}
                            print('divB is not in constant node list~')
                            continue

                    print('divB:', divB)

                    if divB[0] != 6:
                        print('this is not the div-node which we wanted(value B is not 6)...')
                        dict_add = {}
                        dict_clip = {}
                        dict_div = {}
                        continue

                    if isinstance(divB, np.ndarray) == True:
                        if divB.shape != (1, ):
                            print('this is not the div-node which we wanted(shape is wrong)...')
                            dict_add = {}
                            dict_clip = {}
                            dict_div = {}
                            continue
                    else:        
                        if len(divB) != 1:
                            print('this is not the div-node which we wanted(list len is wrong)...')
                            dict_add = {}
                            dict_clip = {}
                            dict_div = {}
                            continue

                    ###################################
                    old_node = model.graph.node[dict_add['id']] 
                    model.graph.node.remove(old_node)

                    swish_node = onnx.helper.make_node(
                                            name = '',
                                            op_type='HardSigmoid',
                                            inputs=[dict_add['input'][0]],
                                            outputs=dict_div['output'],
                                            alpha=1.0/6,
                                            beta=0.5
                                            )

                    model.graph.node.insert(dict_add['id'], swish_node)

                    old_node = model.graph.node[dict_div['id']] 
                    model.graph.node.remove(old_node)

                    old_node = model.graph.node[dict_clip['id']] 
                    model.graph.node.remove(old_node)

                    dict_add = {}
                    dict_clip = {}
                    ###############################

                    got_swish = True
                    search = True
                    break       
                else:
                    print('clear dict_add and dict_clip, ')
                    print('dict_add:', dict_add)
                    print('dict_clip:', dict_clip)
                    dict_add = {}
                    dict_clip = {}

    return model

if __name__ == "__main__":
    model = onnx.load('/home/zqiu/models/decoder_sub1.onnx')
    mha_optimizer(model)
    onnx.save(model, './hs.onnx')

    