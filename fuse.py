import onnx
import correct_batch
import numpy as np

def fuse_pad_to_pool(model, output):
    dict_pad = {}
    dict_pool = {}
    dict_mul = {}

    got_pad_pool = False

    for node_id, node in enumerate(model.graph.node):
        #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
        #         ", op:", node.op_type, ', len(input):', len(node.input))

        if node.op_type == 'Pad':
            dict_pad['input'] = node.input
            dict_pad['output'] = node.output
            dict_pad['id'] = node_id

        if node.op_type == 'MaxPool':
            if len(dict_pad) > 0 and node.input == dict_pad['output']:
                dict_pool['input'] = node.input
                dict_pool['output'] = node.output
                dict_pool['id'] = node_id
                print('got pad+pool pair, pad:', dict_pad['input'], dict_pad['output'])
                print('got pad+pool pair, pool:', dict_pool['input'], dict_pool['output'])
                pads = []

                got_pad_pool = True

                for init in model.graph.initializer:
                    if init.name == dict_pad['input'][1]:
                        print('got init(pads):', init.name)
                        dtype = init.data_type
                        np_dtype = correct_batch.convert_ort_type_2_np(dtype)
                        if init.raw_data:
                            params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                            for p in params_list:
                                pads.append(p)
                        else:
                            data_list = correct_batch.get_data_list(dtype, init)
                            for p in data_list:
                                pads.append(p)
                    elif init.name == dict_pad['input'][2]:
                        print('got init(constane_value):', init.name)  

                pads_real = [pads[2], pads[3], pads[6], pads[7]]

                for attr in node.attribute:
                    if attr.name == 'pads':
                        del attr.ints[:]
                        attr.ints.extend(pads_real)
                        #print('pads:---', attr.ints)
                        break
     
                node.input[0] = dict_pad['input'][0]

                old_node = model.graph.node[dict_pad['id']] 
                model.graph.node.remove(old_node)

                dict_pad = {}
                dict_pool = {}
            else:
                #print('clear pad dict')
                dict_pad = {}    

    if got_pad_pool == True:
        print('got pad+pool node------------')
        onnx.save(model, output)

    return model    