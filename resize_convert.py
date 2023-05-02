import onnx
import values,operation
import sys
import numpy as np
import log

logger = log.getLogger(__name__, log.INFO)

def is_unused_init(model, init):
    for node in model.graph.node:
        if init.name in node.input:
            return False

    return True

def remove_unused_initializer(model, unused_init_list):
    for init in unused_init_list:
        if is_unused_init(model, init):
            logger.debug('remove unused init: {}'.format(init.name))
            model.graph.initializer.remove(init)

def merge_resize_old(model):
    dict_reshape = {}
    dict_expand = {}
    dict_reshape2 = {}
    unused_init_list = []

    search = True

    while search == True:
        search = False
        ready = False

        for node_id, node in enumerate(model.graph.node):
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            if node.op_type == 'Reshape':
                if ready == True:
                    if dict_reshape and dict_expand and node.input[0] == dict_expand['output'][0]:
                        dict_reshape2['input'] = node.input
                        dict_reshape2['output'] = node.output
                        dict_reshape2['id'] = node_id

                        shape = values.get_init_value(model, node.input[1])
                        logger.debug('got Second Reshape shape: {}'.format(shape))
                        shape_list = shape
                        shape_dims = []

                        for init in model.graph.initializer:
                            if init.name == node.input[1]:
                                shape_dims = init.dims
                                if init not in unused_init_list:
                                    unused_init_list.append(init)

                        if isinstance(shape, np.ndarray):
                            shape_list = shape.tolist()

                        input_shape = values.get_tensor_shape_by_name(model, dict_reshape['input'][0])

                        if len(input_shape) != len(shape_list):
                            logger.info('ignore Reshape+Expand+Reshape==>Resize, because input dim {} is not equal sizes dim {}'.format(len(input_shape), len(shape_list)))
                            dict_reshape = {}
                            dict_expand = {}
                            dict_reshape2 = {}
                            ready = False 
                            continue    

                        ###################################
                        old_node = model.graph.node[dict_reshape['id']] 
                        old_node_reshape2 = model.graph.node[dict_reshape2['id']]
                        old_node_expand = model.graph.node[dict_expand['id']] 

                        model.graph.node.remove(old_node)

                        empty0_name = f'empty0_{node_id}'
                        empty1_name = f'empty1_{node_id}'

                        empty0 = onnx.helper.make_tensor(empty0_name, onnx.TensorProto.FLOAT, [0], [])
                        empty1 = onnx.helper.make_tensor(empty1_name, onnx.TensorProto.FLOAT, [0], [])
                        
                        const_sizes_name = f'const_sizes_{node_id}'

                        const_sizes = onnx.helper.make_tensor(name=const_sizes_name,
                                                                data_type=onnx.TensorProto.INT64,
                                                                dims=shape_dims,
                                                                vals=shape_list)

                        model.graph.initializer.append(empty0)
                        model.graph.initializer.append(empty1)
                        model.graph.initializer.append(const_sizes)    

                        resize_node = onnx.helper.make_node(
                                                name = f'Resize_{node_id}',
                                                op_type='Resize',
                                                inputs=[dict_reshape['input'][0], empty0_name, empty1_name, const_sizes_name],
                                                outputs=dict_reshape2['output']
                                                )

                        model.graph.node.insert(dict_reshape['id'], resize_node)

                        model.graph.node.remove(old_node_expand)
                        model.graph.node.remove(old_node_reshape2)

                        dict_reshape = {}
                        dict_expand = {}
                        dict_reshape2 = {}
                        ready = False 
                        ###############################

                        search = True
                        break       
                    else:
                        logger.debug('clear dict_reshape and dict_expand, dict_reshapes')
                        logger.debug('dict_reshape: {}'.format(dict_reshape))
                        logger.debug('dict_expand: {}'.format(dict_expand))
                        logger.debug('dict_reshape2: {}'.format(dict_reshape2))
                        dict_reshape = {}
                        dict_expand = {}
                        dict_reshape2 = {} 
                        unused_init_list = []
                else:
                    dict_reshape['input'] = node.input
                    dict_reshape['output'] = node.output
                    dict_reshape['id'] = node_id

                    for init in model.graph.initializer:
                        if init.name == node.input[1]:
                            if init not in unused_init_list:
                                unused_init_list.append(init)

                    logger.debug('got match Reshape node: {}'.format(node.name))
                
            if node.op_type == 'Expand':
                if dict_reshape and node.input[0] == dict_reshape['output'][0]:
                    dict_expand['input'] = node.input
                    dict_expand['output'] = node.output
                    dict_expand['id'] = node_id

                    for init in model.graph.initializer:
                        if init.name == node.input[1]:
                            if init not in unused_init_list:
                                unused_init_list.append(init)

                    ready = True

                    logger.debug('got match Expand node: {}'.format(node.name))
                else:
                    logger.debug('clear dict_reshape: {}'.format(dict_reshape))
                    dict_reshape = {}
                    unused_init_list = []           

    remove_unused_initializer(model, unused_init_list)

    return model

def merge_resize(model):
    unused_init_list = []

    #while search == True:
    if True:
        search = False
        ready = False

        rer_list = []

        for node in model.graph.node:
            #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
            #        ", op:", node.op_type, ', len(input):', len(node.input))

            rer_dict = {}

            if node.op_type == 'Reshape':
                expand_node, ok = values.get_next_node_by_output(model, node.output[0])
                if ok == 0 and expand_node.op_type == 'Expand':
                    reshape_node2, ok = values.get_next_node_by_output(model, expand_node.output[0])
                    if ok == 0 and reshape_node2.op_type == 'Reshape':
                        rer_dict['Reshape'] = node
                        rer_dict['Reshape2'] = reshape_node2
                        rer_dict['Expand'] = expand_node

                        rer_list.append(rer_dict)

        for idx, rer in enumerate(rer_list):
            rs_node = rer['Reshape']
            rs_node2 = rer['Reshape2']
            expand_node = rer['Expand']

            shape = values.get_init_value(model, rs_node2.input[1])
            shape_list = shape
            shape_dims = []

            if isinstance(shape, np.ndarray):
                shape_list = shape.tolist()

            input_shape = values.get_tensor_shape_by_name(model, rs_node.input[0])

            if len(input_shape) != len(shape_list):
                logger.info('ignore Reshape+Expand+Reshape==>Resize, because input dim {} is not equal sizes dim {}'.format(len(input_shape), len(shape_list)))
                continue   

            for init in model.graph.initializer:
                if init.name == rs_node2.input[1]:
                    shape_dims = init.dims
                    if init not in unused_init_list:
                        unused_init_list.append(init)

            for init in model.graph.initializer:
                if init.name == rs_node.input[1] or init.name == expand_node.input[1]:
                    if init not in unused_init_list:
                        unused_init_list.append(init)

            prev_node, _ = values.get_prev_node_by_input(model, rs_node.input[0])
            next_node, _ = values.get_next_node_by_output(model, rs_node2.output[0])

            empty0_name = f'empty0_{idx}'
            empty1_name = f'empty1_{idx}'

            empty0 = onnx.helper.make_tensor(empty0_name, onnx.TensorProto.FLOAT, [0], [])
            empty1 = onnx.helper.make_tensor(empty1_name, onnx.TensorProto.FLOAT, [0], [])
            
            const_sizes_name = f'const_sizes_{idx}'

            const_sizes = onnx.helper.make_tensor(name=const_sizes_name,
                                                    data_type=onnx.TensorProto.INT64,
                                                    dims=shape_dims,
                                                    vals=shape_list)

            model.graph.initializer.append(empty0)
            model.graph.initializer.append(empty1)
            model.graph.initializer.append(const_sizes)    

            resize_node = onnx.helper.make_node(
                                    name = f'Resize_{idx}',
                                    op_type='Resize',
                                    inputs=[rs_node.input[0], empty0_name, empty1_name, const_sizes_name],
                                    outputs=rs_node2.output
                                    )

            operation.insert_onnx_node(model, resize_node, prev_node)

            model.graph.node.remove(rs_node2)
            model.graph.node.remove(expand_node)
            model.graph.node.remove(rs_node)                        
      
    remove_unused_initializer(model, unused_init_list)

    return model

'''
model = onnx.load('./expand.onnx') 
merge_resize(model)
onnx.save(model, './rs.onnx')
'''