import yaml
from yaml.loader import SafeLoader
import onnx
from onnx import helper
from onnx import TensorProto
import sys
import log

logger = log.getLogger(__name__, log.INFO)

#rgb-->rgba: taanspose(1, 2, 0) + padding

def parse_postproc_yaml(yaml_file):
    std_list = []
    mean_list = []
    alpha_list = []
    control_list = []

    # Open the file and load the file
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)
        print(type(data))

        if 'norm' in data.keys():
            logger.debug('got norm---')
            norm_list = data['norm']
            for n in norm_list:
                logger.debug('n: {}'.format(n))
                if 'std' in n.keys():
                    std_list_ = n['std']
                    if len(std_list_) == 3 or len(std_list_) == 1:
                        for n in std_list_:
                            if n > 0.0:
                                std_list.append(1.0/n)
                            else:
                                std_list.append(1.0/1e-6)    

                        logger.debug('got std values: {}'.format(std_list))

                    continue    

                if 'mean' in n.keys():
                    mean_list_ = n['mean']
                    if len(mean_list_) == 3 or len(mean_list_) == 1:
                        for n in mean_list_:
                            mean_list.append(int(n))

                        logger.debug('got mean values: {}'.format(mean_list)) 

                    continue        

        if 'postproc' in data.keys():
            logger.debug('got postproc---')
            postproc_list = data['postproc']
            for p in postproc_list:
                if 'alpha' in p.keys():
                    alpha_list_ = p['alpha']
                    #if len(alpha_list_) == 3:
                    alpha_list = alpha_list_
                    logger.debug('got alpha values: {}'.format(alpha_list)) 
                
                if 'control' in p.keys():
                    control_list = p['control']
                    logger.debug('got control values: {}'.format(control_list))

    if len(std_list) == 0 or len(mean_list) == 0 or len(std_list) != len(mean_list):
        return {}

    if len(control_list) == 0:
        return {}

    postproc_dict = {} 

    postproc_dict['std'] = std_list
    postproc_dict['mean'] = mean_list

    if len(alpha_list) > 0:
        postproc_dict['alpha'] = alpha_list

    postproc_dict['control'] = control_list

    return postproc_dict    

def insert_postproc_node(model, postproc_dict):
    graph = model.graph
    output_name = graph.output[0].name
    #input_name = ''
    last_id = 0

    logger.debug('output_name: {}'.format(output_name))

    '''
    const_mean = onnx.helper.make_tensor(name='const_mean',
                            data_type=onnx.TensorProto.UINT8,
                            dims=[len(postproc_dict['mean'])],
                            vals=postproc_dict['mean'])

    graph.initializer.append(const_mean)                        

    const_std = onnx.helper.make_tensor(name='const_std',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[len(postproc_dict['std'])],
                        vals=postproc_dict['std'])

    graph.initializer.append(const_std)     
    '''
    
    
    for node_id, node in enumerate(graph.node):
        if node.output[0] == output_name:
            last_id = node_id
            #input_name = node.input[0]
            logger.debug('last_id: {}, name: {}'.format(last_id, node.name))

    #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
    #        ", op:", node.op_type, ', len(input):', len(node.input))

    const_alpha = TensorProto()

    b_alpha = False
    if 'alpha' in postproc_dict.keys() and postproc_dict['control'][1] == 1:
        b_alpha = True

    if b_alpha == True:
        const_alpha = onnx.helper.make_tensor(name='const_alpha',
                            data_type=onnx.TensorProto.UINT8,
                            dims=[len(postproc_dict['alpha'])],
                            vals=postproc_dict['alpha'])

        graph.initializer.append(const_alpha)                        
                  
    const_control_post = onnx.helper.make_tensor(name='const_control_post',
                        data_type=onnx.TensorProto.INT32,
                        dims=[len(postproc_dict['control'])],
                        vals=postproc_dict['control'])   #fp32-->uint8, rgb-->rgba   

    graph.initializer.append(const_control_post)                                  

    #x = helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 3, 576, 720])
    output_shape = graph.output[0].type.tensor_type.shape

    if b_alpha == True:
        post_process_output = helper.make_tensor_value_info('post_process_output', TensorProto.UINT8, [-1, (output_shape.dim[1].dim_value + 1)*output_shape.dim[2].dim_value*output_shape.dim[3].dim_value])      
    else:
        post_process_output = helper.make_tensor_value_info('post_process_output', TensorProto.UINT8, [-1, output_shape.dim[1].dim_value*output_shape.dim[2].dim_value*output_shape.dim[3].dim_value//2])
    
    if b_alpha == True:
        post_process_node = onnx.helper.make_node(
                        'PostProc',
                        name='postprocess',
                        inputs=[output_name, 'const_alpha', 'const_control_post'],
                        outputs=['post_process_output'],
                        domain='com.metax-tech')
    else:
        post_process_node = onnx.helper.make_node(
                        'PostProc',
                        name='postprocess',
                        inputs=[output_name, 'const_control_post'],
                        outputs=['post_process_output'],
                        domain='com.metax-tech')
 
    graph.node.insert(last_id + 1, post_process_node)

    del model.graph.output[:]

    model.graph.output.extend([onnx.ValueInfoProto(name='post_process_output')])

    new_graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    model = onnx.helper.make_model(new_graph)
    #model = onnx.shape_inference.infer_shapes(model)
 
    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    #onnx.checker.check_model(onnx_model)
    #onnx.save(model, output)
  
def postproc(model, postproc_yaml):
    post_dict = parse_postproc_yaml(postproc_yaml) 

    print('---------------------------------')

    for k, v in post_dict.items():
        print(k, ':', v)   

    if len(post_dict) > 0:
        insert_postproc_node(model, post_dict) 

    return model       
'''
if __name__ == "__main__":
    model = onnx.load('./r1.onnx')
    postproc(model, './post.onnx')
'''    
