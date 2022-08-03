import yaml
from yaml.loader import SafeLoader
import onnx
from onnx import helper
from onnx import TensorProto
import sys

#rgb-->rgba: taanspose(1, 2, 0) + padding

def parse_postproc_yaml(yaml_file):
    alpha_list = []
    control_list = []

    # Open the file and load the file
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)
        print(type(data))

        if 'alpha' in data.keys():
            alpha_list_ = data['alpha']
            if len(alpha_list_) == 3:
                alpha_list = alpha_list_
                print('got alpha values:', alpha_list) 
        
        if 'control' in data.keys():
            control_list = data['control']
            print('got control values:', control_list)

    if len(control_list) == 0:
        return {}

    postproc_dict = {} 

    if len(alpha_list) > 0:
        postproc_dict['alpha'] = alpha_list

    postproc_dict['control'] = control_list

    return postproc_dict    

def insert_postproc_node(model, postproc_dict, output):
    graph = model.graph
    output_name = graph.output[0].name
    #input_name = ''
    last_id = 0

    print('output_name:', output_name)

    for node_id, node in enumerate(graph.node):
        if node.output[0] == output_name:
            last_id = node_id
            #input_name = node.input[0]
            print('last_id:', last_id, ', name:', node.name)

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
                  
    const_control = onnx.helper.make_tensor(name='const_control',
                        data_type=onnx.TensorProto.INT32,
                        dims=[len(postproc_dict['control'])],
                        vals=postproc_dict['control'])   #fp32-->uint8, rgb-->rgba   

    graph.initializer.append(const_control)                                  

    #x = helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 3, 576, 720])
    output_shape = graph.output[0].type.tensor_type.shape

    if b_alpha == True:
        post_process_output = helper.make_tensor_value_info('post_process_output', TensorProto.UINT8, [-1, output_shape.dim[1].dim_value*output_shape.dim[2].dim_value*(output_shape.dim[3].dim_value+1)])      
    else:
        post_process_output = helper.make_tensor_value_info('post_process_output', TensorProto.UINT8, [-1, output_shape.dim[1].dim_value*output_shape.dim[2].dim_value*output_shape.dim[3].dim_value//2])
    
    if b_alpha == True:
        post_process_node = onnx.helper.make_node(
                        'PostProc',
                        name='postprocess',
                        inputs=[output_name, 'const_alpha', 'const_control'],
                        outputs=['post_process_output'],
                        domain='com.metax-tech')
    else:
        post_process_node = onnx.helper.make_node(
                        'PostProc',
                        name='postprocess',
                        inputs=[output_name, 'const_control'],
                        outputs=['post_process_output'],
                        domain='com.metax-tech')
 
    graph.node.insert(last_id + 1, post_process_node)

    del model.graph.output[:]

    model.graph.output.extend([onnx.ValueInfoProto(name='post_process_output')])

    new_graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    model = onnx.helper.make_model(new_graph)
    model = onnx.shape_inference.infer_shapes(model)
 
    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    #onnx.checker.check_model(onnx_model)
    onnx.save(model, output)
  
def postproc(model, output):
    post_dict = parse_postproc_yaml('./postproc.yaml') 

    print('---------------------------------')

    for k, v in post_dict.items():
        print(k, ':', v)   

    if len(post_dict) > 0:
        insert_postproc_node(model, post_dict, output)    

if __name__ == "__main__":
    model = onnx.load('./r1.onnx')
    postproc(model, './post.onnx')
