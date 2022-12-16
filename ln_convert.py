import onnx
import sys
import argparse
import values, sys
import numpy as np       
from onnx import onnx_pb as onnx_proto

def make_fp32_tensor_from_fp16(fp16_tensor, fp32_tensor_name):
    float_list = []
    if fp16_tensor.int32_data:
        print('make_fp32_tensor_from_fp16, int32_data')
        num = np.array(fp16_tensor.int32_data)
        float_list = num.astype(np.float).tolist()

    if fp16_tensor.raw_data:
        print('make_fp32_tensor_from_fp16, raw_data')
        float_list = np.fromstring(fp16_tensor.raw_data, dtype='float16')
        num = np.array(float_list)
        float_list = num.astype(np.float).tolist()

    t = onnx.helper.make_tensor(name=fp32_tensor_name,
                                    data_type=onnx_proto.TensorProto.FLOAT,
                                    dims=fp16_tensor.dims,
                                    vals=float_list)

    return t                                

def is_unused_init(model, init):
    for node in model.graph.node:
        if init.name in node.input:
            return False

    return True

def remove_unused_initializer(model, unused_init_list):
    for init in unused_init_list:
        if is_unused_init(model, init):
            print('remove unused init:', init.name)
            model.graph.initializer.remove(init)

# pattern 1:
#                                 ---     ---     ---      ---        ---       ---    ---    --
#                               |                                                              |
# ***(0) --- ReduceMean(1) --- Sub(2) --- Pow(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Div(7) --- Mul(8) --- (Add)(9)
#      |                     |
#       ---   ---   ---   ---    

def merge_layernorm_pattern_1(model):
    got_ln = False
    search = True
    unused_init_list = []
    loop = 0

    while search == True:
        loop = loop + 1

        dict_rm = {}
        dict_sub = {}
        dict_pow = {}
        dict_rm2 = {}
        dict_add = {}
        dict_sqrt = {}
        dict_div = {}
        dict_mul = {}
        dict_add2 = {}

        rm1_axes = [-1]
        rm2_axes = [-1]

        search = False
        ready = False
        ready2 =  False

        for node_id, node in enumerate(model.graph.node):
            loop = loop + 1
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'ReduceMean':
                if ready == True:
                    if node.input == dict_pow['output']:
                        dict_rm2['input'] = node.input
                        dict_rm2['output'] = node.output
                        dict_rm2['id'] = node_id

                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'axes':
                                rm2_axes = attr.ints
                                print('rm2_axes: ', rm2_axes)
                                if len(rm2_axes) != 1 or rm2_axes != rm1_axes:
                                    print('--This ReduceMean IsNot we are looking for...')
                                    dict_rm = {}
                                    dict_sub = {}
                                    dict_pow = {}
                                    dict_rm2 = {}
                                    ready = False
                                    ready2 =  False
                    else: 
                        print('clear ReduceMean Sub Pow')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        print('dict_pow:', dict_pow)
                        dict_rm = {}
                        dict_sub = {}
                        dict_pow = {}
                        ready = False
                        ready2 = False      
                else: 
                    dict_rm['input'] = node.input
                    dict_rm['output'] = node.output
                    dict_rm['id'] = node_id

                    attributes = node.attribute
                    for attr in attributes:
                        if attr.name == 'axes':
                            rm1_axes = attr.ints
                            print('rm1_axes: ', rm1_axes)
                            if len(rm1_axes) != 1:
                                print('This ReduceMean IsNot we are looking for...')
                                dict_rm = {}

                            break

            if node.op_type == 'Sub':
                if dict_rm and node.input[0] == dict_rm['input'][0] and node.input[1] == dict_rm['output'][0]:
                    dict_sub['input'] = node.input
                    dict_sub['output'] = node.output
                    dict_sub['id'] = node_id
                    print('got first pair:', dict_sub['input'], dict_sub['output'])
                else:
                    print('clear ReduceMean, dict_rm:', dict_rm)
                    dict_rm = {}
                    ready = False
                    ready2 = False     

            if node.op_type == 'Pow':
                if dict_rm and dict_sub and node.input[0] == dict_sub['output'][0]:
                    v = values.get_init_value(model, node.input[1])
                    print('pow exp:', v, type(v))
                    if v == 2:
                        ready = True
                        dict_pow['input'] = node.input
                        dict_pow['output'] = node.output
                        dict_pow['id'] = node_id

                        print('got second pair:', dict_pow['input'], dict_pow['output'])
                    else:
                        print('--clear ReduceMean and Sub')
                        print('--dict_rm:', dict_rm)
                        print('--dict_sub:', dict_sub)
                        dict_rm = {}
                        dict_sub = {}
                        ready = False 
                        ready2 = False         
                else:
                    print('clear ReduceMean and Sub')
                    print('dict_rm:', dict_rm)
                    print('dict_sub:', dict_sub)
                    dict_rm = {}
                    dict_sub = {}
                    ready = False
                    ready2 = False  

            if node.op_type == 'Add':
                if ready == True and ready2 == True:
                    if node.input[0] == dict_mul['output'][0]:
                        dict_add2['input'] = node.input
                        dict_add2['output'] = node.output
                        dict_add2['id'] = node_id

                        search = True
                        got_ln = True
                        print('Got a LayerNorm op')
                        ###
                        rm_node = model.graph.node[dict_rm['id']]
                        sub_node = model.graph.node[dict_sub['id']]
                        pow_node = model.graph.node[dict_pow['id']]
                        rm2_node = model.graph.node[dict_rm2['id']]
                        add_node = model.graph.node[dict_add['id']]
                        sqrt_node = model.graph.node[dict_sqrt['id']]
                        div_node = model.graph.node[dict_div['id']]
                        mul_node = model.graph.node[dict_mul['id']]
                        add2_node = model.graph.node[dict_add2['id']]

                        model.graph.node.remove(rm_node)


                        scale_name = dict_mul['input'][1]
                        beta_name = dict_add2['input'][1]
                        for init in model.graph.initializer:
                            if init.name == dict_mul['input'][1]:
                                if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                    scale_name = scale_name + '_to_fp32__'
                                    t = make_fp32_tensor_from_fp16(init, scale_name)
                                    model.graph.initializer.append(t)

                                    if init not in unused_init_list:
                                        unused_init_list.append(init)

                            elif init.name == dict_add2['input'][1]:
                                if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                    beta_name = beta_name + '_to_fp32__'
                                    t = make_fp32_tensor_from_fp16(init, beta_name)
                                    model.graph.initializer.append(t)

                                    if init not in unused_init_list:
                                        unused_init_list.append(init)            

                        ln_node = onnx.helper.make_node(
                                                name = node.name + '_to_layernorm_' + str(loop),
                                                op_type='LayerNorm',
                                                inputs=[dict_rm['input'][0], scale_name, beta_name],
                                                outputs=dict_add2['output'],
                                                axis=rm1_axes[0],
                                                epsilon=1e-5,
                                                stash_type=0,
                                                domain='com.metax-tech'
                                                )

                        model.graph.node.insert(dict_rm['id'], ln_node)

                        model.graph.node.remove(sub_node)
                        model.graph.node.remove(pow_node)
                        model.graph.node.remove(rm2_node)
                        model.graph.node.remove(add_node)
                        model.graph.node.remove(sqrt_node)
                        model.graph.node.remove(div_node)
                        model.graph.node.remove(mul_node)
                        model.graph.node.remove(add2_node)

                        break
                    else:
                        print('--clear ReduceMean Sub Pow ReduceMean2')
                        print('--dict_rm:', dict_rm)
                        print('--dict_sub:', dict_sub)
                        print('--dict_pow:', dict_pow)
                        print('--dict_rm2:', dict_rm2)
                        dict_rm = {}
                        dict_sub = {}
                        dict_rm = {}
                        dict_pow = {}
                        dict_rm2 = {}
                        ready = False
                        ready2 = False  
                else:
                    if dict_rm and dict_sub and dict_pow and dict_rm2 and node.input[0] == dict_rm2['output'][0]:
                        dict_add['input'] = node.input
                        dict_add['output'] = node.output
                        dict_add['id'] = node_id
                        print('got third pair:', dict_add['input'], dict_add['output'])
                    else:
                        print('clear ReduceMean Sub Pow ReduceMean2')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        print('dict_pow:', dict_pow)
                        print('dict_rm2:', dict_rm2)
                        dict_rm = {}
                        dict_sub = {}
                        dict_rm = {}
                        dict_pow = {}
                        dict_rm2 = {}
                        ready = False
                        ready2 = False  

            if node.op_type == 'Sqrt':
                if dict_rm and dict_sub and dict_pow and dict_rm2 and  \
                        dict_add and node.input == dict_add['output']:
                    dict_sqrt['input'] = node.input
                    dict_sqrt['output'] = node.output
                    dict_sqrt['id'] = node_id
                    print('got forth pair:', dict_sqrt['input'], dict_sqrt['output'])
                else:
                    print('clear ReduceMean Sub Pow ReduceMean2 Add')
                    print('dict_rm:', dict_rm)
                    print('dict_sub:', dict_sub)
                    print('dict_pow:', dict_pow)
                    print('dict_rm2:', dict_rm2)
                    print('dict_add:', dict_add)
                    dict_rm = {}
                    dict_sub = {}
                    dict_rm = {}
                    dict_pow = {}
                    dict_rm2 = {}
                    dict_add = {}
                    ready = False
                    ready2 = False 

            if node.op_type == 'Div':
                if dict_rm and dict_sub and dict_pow and dict_rm2 and  \
                        dict_add and dict_sqrt and node.input[0] == dict_sub['output'][0] \
                        and node.input[1] == dict_sqrt['output'][0]:
                    dict_div['input'] = node.input
                    dict_div['output'] = node.output
                    dict_div['id'] = node_id
                    print('got fifth pair:', dict_div['input'], dict_div['output'])
                else:
                    print('-clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                    print('dict_rm:', dict_rm)
                    print('dict_sub:', dict_sub)
                    print('dict_pow:', dict_pow)
                    print('dict_rm2:', dict_rm2)
                    print('dict_add:', dict_add)
                    print('dict_sqrt:', dict_sqrt)
                    dict_rm = {}
                    dict_sub = {}
                    dict_rm = {}
                    dict_pow = {}
                    dict_rm2 = {}
                    dict_add = {}
                    dict_sqrt = {}
                    ready = False
                    ready2 = False 

            if node.op_type == 'Mul':
                if dict_rm and dict_sub and dict_pow and dict_rm2 and  \
                        dict_add and dict_sqrt and dict_div and node.input[0] == dict_div['output'][0]:
                    dict_mul['input'] = node.input
                    dict_mul['output'] = node.output
                    dict_mul['id'] = node_id
                    ready2 = True

                    print('got sixth pair:', dict_mul['input'], dict_mul['output'])
                    #print('got scale:', scale)
                else:
                    print('--clear ReduceMean Sub Pow ReduceMean2 Add Sqrt Div')
                    print('dict_rm:', dict_rm)
                    print('dict_sub:', dict_sub)
                    print('dict_pow:', dict_pow)
                    print('dict_rm2:', dict_rm2)
                    print('dict_add:', dict_add)
                    print('dict_sqrt:', dict_sqrt)
                    print('dict_mul:', dict_mul)
                    dict_rm = {}
                    dict_sub = {}
                    dict_rm = {}
                    dict_pow = {}
                    dict_rm2 = {}
                    dict_add = {}
                    dict_sqrt = {}
                    dict_mul = {}
                    ready = False
                    ready2 = False                              

    if got_ln == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1
        
        #onnx.save(model, export_onnx)

    return model

# pattern 2:
#  ---   ---  ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  -- Mul(11)   ---   ---   Add(12) ---
#  ^                             |                                                                                ^                     ^
#  |                             v                                                                                |                     |
# ***(0) --- ReduceMean(1) --- Sub(2) --- Mul(3) --- ReduceMean(4) --- Add(5) --- Sqrt(6) --- Reciprocal(7) --- Mul(8) --- Mul(9) --- Sub(10)
#             |                                                                                                             |
#             v                                                                                                             ^
#             --   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---   ---  ---  --- --

def merge_layernorm_pattern_2(model):
    got_ln = False
    search = True
    loop = 0
    unused_init_list = []

    while search == True:
        loop = loop + 1
        dict_rm = {}
        dict_sub = {}
        dict_sub2 = {}
        dict_mul = {}
        dict_rm2 = {}
        dict_add = {}
        dict_sqrt = {}
        dict_div = {}
        dict_mul2 = {}
        dict_mul3 = {}
        dict_mul4 = {}
        dict_add2 = {}

        rm1_axes = [-1]
        rm2_axes = [-1]

        search = False
        ready_for_reducemean = False
        ready_for_mul_first = False
        ready_for_mul_second = False
        ready_for_mul_third = False
        ready_for_sub = False
        got_rm_axes = False
        ready2 =  False

        for node_id, node in enumerate(model.graph.node):
            loop = loop + 1
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'ReduceMean' or node.op_type == 'GlobalAveragePool':
                if ready_for_reducemean == True:
                    if node.input == dict_mul['output']:
                        dict_rm2['input'] = node.input
                        dict_rm2['output'] = node.output
                        dict_rm2['id'] = node_id

                        if node.op_type == 'ReduceMean':
                            attributes = node.attribute
                            for attr in attributes:
                                if attr.name == 'axes':
                                    rm2_axes = attr.ints
                                    print('rm2_axes: ', rm2_axes)
                                    if len(rm2_axes) != 1 or rm2_axes != rm1_axes:
                                        print('--This ReduceMean IsNot we are looking for...')
                                        dict_rm = {}
                                        dict_sub = {}
                                        dict_mul = {}
                                        dict_rm2 = {}
                                        ready_for_reducemean = False
                                        ready2 =  False
                                    else:
                                        got_rm_axes = True    
                    else: 
                        print('clear ReduceMean Sub Pow')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        print('dict_mul:', dict_mul)
                        dict_rm = {}
                        dict_sub = {}
                        dict_mul = {}
                        ready_for_reducemean = False
                        ready2 = False      
                else: 
                    dict_rm['input'] = node.input
                    dict_rm['output'] = node.output
                    dict_rm['id'] = node_id

                    if node.op_type == 'ReduceMean':
                        attributes = node.attribute
                        for attr in attributes:
                            if attr.name == 'axes':
                                rm1_axes = attr.ints
                                print('rm1_axes: ', rm1_axes)
                                if len(rm1_axes) != 1:
                                    print('This ReduceMean IsNot we are looking for...')
                                    dict_rm = {}

                                break

            if node.op_type == 'Sub':
                if ready_for_sub == True:
                    if node.input[1] == dict_mul3['output'][0]:
                        dict_sub2['input'] = node.input
                        dict_sub2['output'] = node.output
                        dict_sub2['id'] = node_id
                        ready2 = True
                        print('got eighth pair:', dict_sub2['input'], dict_sub2['output'])
                    else:
                        print('---clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        print('dict_mul:', dict_mul)
                        print('dict_rm2:', dict_rm2)
                        print('dict_add:', dict_add)
                        print('dict_sqrt:', dict_sqrt)
                        dict_rm = {}
                        dict_sub = {}
                        dict_rm2 = {}
                        dict_mul = {}
                        dict_mul2 = {}
                        dict_mul3 = {}
                        dict_mul4 = {}
                        dict_add = {}
                        dict_sqrt = {}
                        dict_div = {}
                        ready_for_reducemean = False
                        ready_for_mul_first = False
                        ready_for_mul_second = False
                        ready_for_mul_third = False
                        ready_for_sub = False
                        ready2 = False    
                else:    
                    if dict_rm and node.input[0] == dict_rm['input'][0] and node.input[1] == dict_rm['output'][0]:
                        dict_sub['input'] = node.input
                        dict_sub['output'] = node.output
                        dict_sub['id'] = node_id
                        print('got first pair:', dict_sub['input'], dict_sub['output'])
                    else:
                        print('---clear ReduceMean, dict_rm:', dict_rm, node.name)
                        dict_rm = {}
                        ready_for_reducemean = False
                        ready2 = False     

            if node.op_type == 'Mul':
                if ready_for_mul_first == True and ready_for_mul_second == True:
                    if  node.input[0] == dict_rm['output'][0] and node.input[1] == dict_mul2['output'][0]:
                        dict_mul3['input'] = node.input
                        dict_mul3['output'] = node.output
                        dict_mul3['id'] = node_id
                        ready_for_sub = True
                        print('got seventh pair:', dict_mul3['input'], dict_mul3['output'])
                    elif  node.input[0] == dict_rm['input'][0] and node.input[1] == dict_mul2['output'][0]:
                        dict_mul4['input'] = node.input
                        dict_mul4['output'] = node.output
                        dict_mul4['id'] = node_id
                        ready_for_mul_third = True
                        print('got nineth pair:', dict_mul4['input'], dict_mul4['output'])    
                    else:
                        print('---clear ReduceMean and Sub')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        dict_rm = {}
                        dict_sub = {}
                        ready_for_reducemean = False
                        ready_for_mul_first = False
                        ready_for_mul_second = False
                        ready_for_mul_third = False
                        ready2 = False    
                elif ready_for_mul_first == True and ready_for_mul_second == False and ready_for_mul_third == False:
                    if  node.input[0] == dict_div['output'][0]:
                        dict_mul2['input'] = node.input
                        dict_mul2['output'] = node.output
                        dict_mul2['id'] = node_id
                        ready_for_mul_second = True
                        print('got sixth pair:', dict_mul2['input'], dict_mul2['output'])
                    else:
                        print('---clear ReduceMean and Sub')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        dict_rm = {}
                        dict_sub = {}
                        ready_for_reducemean = False
                        ready_for_mul_first = False
                        ready2 = False     
                else:    
                    if dict_rm and dict_sub and node.input[0] == dict_sub['output'][0]  and node.input[1] == dict_sub['output'][0]:
                        ready_for_reducemean = True
                        dict_mul['input'] = node.input
                        dict_mul['output'] = node.output
                        dict_mul['id'] = node_id

                        print('got second pair:', dict_mul['input'], dict_mul['output'])   
                    else:
                        print('clear ReduceMean and Sub')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        dict_rm = {}
                        dict_sub = {}
                        ready_for_reducemean = False
                        ready2 = False  

            if node.op_type == 'Add':
                if ready2 == True:
                    if node.input[0] == dict_mul4['output'][0] and node.input[1] == dict_sub2['output'][0]:
                        dict_add2['input'] = node.input
                        dict_add2['output'] = node.output
                        dict_add2['id'] = node_id

                        search = True
                        got_ln = True
                        print('Got a LayerNorm op')
                        ###
                        rm_node = model.graph.node[dict_rm['id']]
                        sub_node = model.graph.node[dict_sub['id']]
                        sub2_node = model.graph.node[dict_sub2['id']]
                        mul_node = model.graph.node[dict_mul['id']]
                        mul2_node = model.graph.node[dict_mul2['id']]
                        mul3_node = model.graph.node[dict_mul3['id']]
                        mul4_node = model.graph.node[dict_mul4['id']]
                        rm2_node = model.graph.node[dict_rm2['id']]
                        add_node = model.graph.node[dict_add['id']]
                        sqrt_node = model.graph.node[dict_sqrt['id']]
                        div_node = model.graph.node[dict_div['id']]
                        add2_node = model.graph.node[dict_add2['id']]

                        model.graph.node.remove(rm_node)

                        if got_rm_axes == True:
                            axis_ = rm1_axes[0]
                        else:
                            axis_ = -1

                        print('get axis_ = ', axis_)

                        scale_name = dict_mul2['input'][1]
                        beta_name = dict_sub2['input'][0]
                        for init in model.graph.initializer:
                            if init.name == dict_mul2['input'][1]:
                                if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                    scale_name = scale_name + '_to_fp32__'
                                    t = make_fp32_tensor_from_fp16(init, scale_name)
                                    model.graph.initializer.append(t)
                                    if init not in unused_init_list:
                                        unused_init_list.append(init) 

                            elif init.name == dict_sub2['input'][0]:
                                if init.data_type == onnx_proto.TensorProto.FLOAT16:
                                    beta_name = beta_name + '_to_fp32__'
                                    t = make_fp32_tensor_from_fp16(init, beta_name)
                                    model.graph.initializer.append(t)

                                    if init not in unused_init_list:
                                        unused_init_list.append(init)               

                        ln_node = onnx.helper.make_node(
                                                name = node.name + '_to_layernorm_' + str(loop),
                                                op_type='LayerNorm',
                                                inputs=[dict_rm['input'][0], scale_name, beta_name],
                                                outputs=dict_add2['output'],
                                                axis=axis_,
                                                epsilon=1e-5,
                                                stash_type=0,
                                                domain='com.metax-tech'
                                                )

                        model.graph.node.insert(dict_rm['id'], ln_node)

                        model.graph.node.remove(sub_node)
                        model.graph.node.remove(sub2_node)
                        model.graph.node.remove(rm2_node)
                        model.graph.node.remove(add_node)
                        model.graph.node.remove(sqrt_node)
                        model.graph.node.remove(div_node)
                        model.graph.node.remove(mul_node)
                        model.graph.node.remove(mul2_node)
                        model.graph.node.remove(mul3_node)
                        model.graph.node.remove(mul4_node)
                        model.graph.node.remove(add2_node)

                        break
                    else:
                        print('--clear ReduceMean Sub Pow ReduceMean2')
                        print('--dict_rm:', dict_rm)
                        print('--dict_sub:', dict_sub)
                        print('--dict_pow:', dict_pow)
                        print('--dict_rm2:', dict_rm2)
                        dict_rm = {}
                        dict_sub = {}
                        dict_rm = {}
                        dict_pow = {}
                        dict_rm2 = {}
                        ready_for_reducemean = False
                        ready2 = False  
                else:
                    if dict_rm and dict_sub and dict_mul and dict_rm2 and node.input[0] == dict_rm2['output'][0]:
                        dict_add['input'] = node.input
                        dict_add['output'] = node.output
                        dict_add['id'] = node_id
                        print('got third pair:', dict_add['input'], dict_add['output'])
                    else:
                        print('clear ReduceMean Sub Mul ReduceMean2')
                        print('dict_rm:', dict_rm)
                        print('dict_sub:', dict_sub)
                        print('dict_mul:', dict_mul)
                        print('dict_rm2:', dict_rm2)
                        dict_rm = {}
                        dict_sub = {}
                        dict_rm = {}
                        dict_mul = {}
                        dict_rm2 = {}
                        ready_for_reducemean = False
                        ready2 = False  

            if node.op_type == 'Sqrt':
                if dict_rm and dict_sub and dict_mul and dict_rm2 and  \
                        dict_add and node.input == dict_add['output']:
                    dict_sqrt['input'] = node.input
                    dict_sqrt['output'] = node.output
                    dict_sqrt['id'] = node_id
                    print('got forth pair:', dict_sqrt['input'], dict_sqrt['output'])
                else:
                    print('clear ReduceMean Sub Pow ReduceMean2 Add')
                    print('dict_rm:', dict_rm)
                    print('dict_sub:', dict_sub)
                    print('dict_mul:', dict_mul)
                    print('dict_rm2:', dict_rm2)
                    print('dict_add:', dict_add)
                    dict_rm = {}
                    dict_sub = {}
                    dict_rm = {}
                    dict_mul = {}
                    dict_rm2 = {}
                    dict_add = {}
                    ready_for_reducemean = False
                    ready2 = False 

            if node.op_type == 'Reciprocal':
                if dict_rm and dict_sub and dict_mul and dict_rm2 and  \
                        dict_add and dict_sqrt and node.input[0] == dict_sqrt['output'][0]:
                    dict_div['input'] = node.input
                    dict_div['output'] = node.output
                    dict_div['id'] = node_id
                    ready_for_mul_first = True
                    print('got fifth pair:', dict_div['input'], dict_div['output'])
                else:
                    print('----clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
                    print('dict_rm:', dict_rm)
                    print('dict_sub:', dict_sub)
                    print('dict_mul:', dict_mul)
                    print('dict_rm2:', dict_rm2)
                    print('dict_add:', dict_add)
                    print('dict_sqrt:', dict_sqrt)
                    dict_rm = {}
                    dict_sub = {}
                    dict_rm = {}
                    dict_mul = {}
                    dict_rm2 = {}
                    dict_add = {}
                    dict_sqrt = {}
                    ready_for_reducemean = False
                    ready2 = False 
                   
    if got_ln == True:
        op_set = model.opset_import.add()
        op_set.domain = 'com.metax-tech'
        op_set.version = 1
        
        #onnx.save(model, export_onnx)

    remove_unused_initializer(model, unused_init_list)

    return model

def merge_layernorm(model):
    model = merge_layernorm_pattern_1(model)
    model = merge_layernorm_pattern_2(model)

    return model

model = onnx.load('./bert_fp16_test.onnx')
#model = onnx.load('./bert_sub.onnx')
m = merge_layernorm(model)
onnx.save(m, 'ln.onnx')   
  
