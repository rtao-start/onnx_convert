import onnx
import correct_batch
import numpy as np
import values

def get_constant_value(model, name):
    shape = []
    for n in model.graph.node:
        if name == n.output[0]:
            attributes = n.attribute
            for attr in attributes:
                if attr.name == 'value':
                    v = values.get_tensor_value(attr.t)
                    dims = len(v)
                    print('get_constant_value:', v, dims)
                    shape = v
                    break
            break

    return shape                     

def fuse_pad_to_pool(model):
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

        if node.op_type == 'MaxPool' or node.op_type == 'AveragePool':
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
                    #elif init.name == dict_pad['input'][2]:
                    #    print('got init(constane_value):', init.name)  

                if pads == []:
                    pads = get_constant_value(model, dict_pad['input'][1])

                print('got pads:', pads)

                pads_real = [pads[2], pads[3], pads[6], pads[7]]
            
                found_pads_attr = False
                for attr in node.attribute:
                    if attr.name == 'pads':
                        del attr.ints[:]
                        attr.ints.extend(pads_real)
                        found_pads_attr = True
                        #print('pads:---', attr.ints)
                        break

                if found_pads_attr == False:
                    attr = onnx.helper.make_attribute('pads', pads_real)
                    node.attribute.append(attr) 

                if node.op_type == 'AveragePool':
                    found_cip_attr = False
                    for attr in node.attribute:
                        if attr.name == 'count_include_pad':
                            found_cip_attr = True
                            attr.i = 1
                            break
                    
                    if found_cip_attr == False:
                        attr = onnx.helper.make_attribute('count_include_pad', 1)
                        node.attribute.append(attr)           
     
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

    return model    