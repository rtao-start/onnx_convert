import onnx
import sys

def merge_swish(model, output):
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
        
        onnx.save(model, output)

def merge_hard_swish(model, output):
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
        
        onnx.save(model, output)

def merge_swish_and_hard_swish(model, output):
    merge_swish(model, output)
    merge_hard_swish(model, output)

    