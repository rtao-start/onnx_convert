import onnx
import sys
import argparse
import values
       
def merge_layernorm(model):
    got_ln = False
    search = True

    while search == True:
        dict_rm = {}
        dict_sub = {}
        dict_pow = {}
        dict_rm2 = {}
        dict_add = {}
        dict_sqrt = {}
        dict_div = {}
        dict_mul = {}
        dict_add2 = {}

        search = False
        ready = False
        ready2 =  False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'ReduceMean':
                if ready == True:
                    if node.input == dict_pow['output']:
                        dict_rm2['input'] = node.input
                        dict_rm2['output'] = node.output
                        dict_rm2['id'] = node_id
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

                        '''
                        got_ln = True

                        old_node = model.graph.node[dict_rm['id']] 
                        model.graph.node.remove(old_node)

                        mish_node = onnx.helper.make_node(
                                                name = '',
                                                op_type='Mish',
                                                inputs=dict_rm['input'],
                                                outputs=dict_pow['output'],
                                                domain='com.metax-tech'
                                                )

                        model.graph.node.insert(dict_rm['id'], mish_node)

                        old_node = model.graph.node[dict_pow['id']] 
                        model.graph.node.remove(old_node)

                        old_node = model.graph.node[dict_sub['id']] 
                        model.graph.node.remove(old_node)

                        dict_rm = {}
                        dict_sub = {}
                        dict_pow = {} 

                        search = True
                        break
                        '''
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
                        search = True
                        got_ln = True
                        print('Got a LayerNorm op')
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
                        dict_add and dict_sqrt and node.input[0] == dict_sub['output'][0] |
                        and node.input[1] == dict_sqrt['output'][0]:
                    dict_div['input'] = node.input
                    dict_div['output'] = node.output
                    dict_div['id'] = node_id
                    print('got fifth pair:', dict_div['input'], dict_div['output'])
                else:
                    print('clear ReduceMean Sub Pow ReduceMean2 Add Sqrt')
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
                else:
                    print('clear ReduceMean Sub Pow ReduceMean2 Add Sqrt Div')
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

model = onnx.load('./xxx.onnx')
merge_layernorm(model)   
  
