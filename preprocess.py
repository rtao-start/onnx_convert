# import pyyaml module
import yaml, sys
from yaml.loader import SafeLoader
import onnx
from onnx import helper
from onnx import TensorProto

def parse_yaml(yaml_file):
    hw_list = []
    std_list = []
    mean_list = []
    resize_list = []
    crop_list = []
    control_list = []

    # Open the file and load the file
    with open(yaml_file) as f:
        data = yaml.load(f, Loader=SafeLoader)
        print(data)
        print(type(data))

        if 'norm' in data.keys():
            print('got norm---')
            norm_list = data['norm']
            for n in norm_list:
                print('n:', n)
                if 'std' in n.keys():
                    std_list_ = n['std']
                    if len(std_list_) == 3 or len(std_list_) == 1:
                        for n in std_list_:
                            if n > 0.0:
                                std_list.append(1.0/n)
                                print('add std:', 1.0/n)
                            else:
                                std_list.append(1.0/1e-6)    

                        print('got std values:', std_list)

                    continue    

                if 'mean' in n.keys():
                    mean_list_ = n['mean']
                    if len(mean_list_) == 3 or len(mean_list_) == 1:
                        for n in mean_list_:
                            mean_list.append(int(n))

                        print('got mean values:', mean_list) 

                    continue         

        if 'preproc' in data.keys():
            print('got preproc---')
            preproc_list = data['preproc']
            for p in preproc_list:
                print('p:',p)
                if 'hw' in p.keys():
                    hw_list_ = p['hw']
                    if len(hw_list_) == 2:
                        hw_list = hw_list_
                        print('got hw values:', hw_list)
                    continue    

                if 'resize' in p.keys():
                    resize_list_ = p['resize']
                    if len(resize_list_) == 2 or len(resize_list_) == 1:
                        resize_list = resize_list_
                        print('got resize values:', resize_list)
                    continue

                if 'crop' in p.keys():
                    crop_list_ = p['crop']
                    if len(crop_list_) == 4:
                        crop_list = crop_list_
                        print('got crop values:', crop_list)
                    continue

                if 'control' in p.keys():
                    control_list_ = p['control']
                    if len(control_list_) == 4 or len(control_list_) == 5:
                        control_list = control_list_
                        print('got control values:', control_list)
                    continue

    if len(std_list) == 0 or len(mean_list) == 0 or len(std_list) != len(mean_list):
        return {}

    preproc_dict = {} 

    preproc_dict['std'] = std_list
    preproc_dict['mean'] = mean_list

    if len(hw_list) == 0:
        hw_list = [-1, -1]

    preproc_dict['hw'] = hw_list

    if len(resize_list) == 0:
        resize_list=[-1, -1]

    preproc_dict['resize'] = resize_list

    if len(crop_list) == 0:
        crop_list = [-1, -1, -1, -1]

    preproc_dict['crop'] = crop_list

    if len(control_list) == 0:
        control_list = [False, True, False, False]

    preproc_dict['control'] = control_list

    return preproc_dict    

def insert_preproc_node(model, preproc_dict):
    graph = model.graph
    input_name = graph.input[0].name

    h = preproc_dict['resize'][0]
    w = preproc_dict['resize'][1]

    preproc_dict['resize'].append(graph.input[0].type.tensor_type.shape.dim[2].dim_value)
    preproc_dict['resize'].append(graph.input[0].type.tensor_type.shape.dim[3].dim_value)

    if h == -1:
        h = graph.input[0].type.tensor_type.shape.dim[2].dim_value

    if w == -1:
        w = graph.input[0].type.tensor_type.shape.dim[3].dim_value    

    print('type(preproc_dict[mean])', type(preproc_dict['mean']))

    const_mean_r = onnx.helper.make_tensor(name='const_mean_r',
                            data_type=onnx.TensorProto.UINT8,
                            dims=[1],
                            vals=[preproc_dict['mean'][0]])

    graph.initializer.append(const_mean_r) 

    const_mean_g = onnx.helper.make_tensor(name='const_mean_g',
                            data_type=onnx.TensorProto.UINT8,
                            dims=[1],
                            vals=[preproc_dict['mean'][1]])

    graph.initializer.append(const_mean_g) 

    const_mean_b = onnx.helper.make_tensor(name='const_mean_b',
                            data_type=onnx.TensorProto.UINT8,
                            dims=[1],
                            vals=[preproc_dict['mean'][2]])

    graph.initializer.append(const_mean_b) 

    print('preproc_dict[std]:', preproc_dict['std'])
    std_list = []
    for v in preproc_dict['std']:
        print('std:', v)
        std_list.append(v)                        

    const_std_r = onnx.helper.make_tensor(name='const_std_r',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[1],
                        vals=[std_list[0]])
                        #vals=preproc_dict['std'])

    graph.initializer.append(const_std_r)   

    const_std_g = onnx.helper.make_tensor(name='const_std_g',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[1],
                        vals=[std_list[1]])
                        #vals=preproc_dict['std'])

    graph.initializer.append(const_std_g)   

    const_std_b = onnx.helper.make_tensor(name='const_std_b',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[1],
                        vals=[std_list[2]])
                        #vals=preproc_dict['std'])

    graph.initializer.append(const_std_b)   

    const_resize = onnx.helper.make_tensor(name='const_resize',
                        data_type=onnx.TensorProto.INT32,
                        dims=[len(preproc_dict['resize'])],
                        vals=preproc_dict['resize'])

    graph.initializer.append(const_resize)   

    const_crop = onnx.helper.make_tensor(name='const_crop',
                        data_type=onnx.TensorProto.INT32,
                        dims=[len(preproc_dict['crop'])],
                        vals=preproc_dict['crop'])

    graph.initializer.append(const_crop)                    

    const_control = onnx.helper.make_tensor(name='const_control',
                        data_type=onnx.TensorProto.INT32,
                        dims=[len(preproc_dict['control'])],
                        vals=preproc_dict['control'])   #rgb2bgr, crop_first, norm_first, float16   

    graph.initializer.append(const_control)                                  

    #x = helper.make_tensor_value_info('x', TensorProto.UINT8, [1, 3, 576, 720])
    pre_process_output = helper.make_tensor_value_info('pre_process_output', TensorProto.FLOAT, [1, graph.input[0].type.tensor_type.shape.dim[1].dim_value, h, w])      

    model.graph.value_info.append(pre_process_output)

    pre_process_node = onnx.helper.make_node(
                    'PreProc',
                    name='preprocess',
                    inputs=[input_name, 'const_std_r', 'const_std_g', 'const_std_b', 'const_mean_r', 'const_mean_g','const_mean_b','const_resize','const_crop', 'const_control'],
                    outputs=['pre_process_output'],
                    domain='com.metax-tech')

    print('before insert node 0')                 

    first_node = graph.node[1]
    for node in graph.node:
        if node.input[0] == graph.input[0].name:
            first_node = node
            print('first node name:', node.name)
            break

    graph.node.insert(0, pre_process_node)

    print('after insert node 0')

    first_node.input[0]='pre_process_output'

    print('before change input 0')
             
    graph.input[0].type.tensor_type.elem_type = 2
    graph.input[0].type.tensor_type.shape.dim[2].dim_value = preproc_dict['hw'][0]
    graph.input[0].type.tensor_type.shape.dim[3].dim_value = preproc_dict['hw'][1]

    #rgb packet input shape must be [1,1,h,w*3]
    if preproc_dict['control'][3] == 1:
        graph.input[0].type.tensor_type.shape.dim[1].dim_value = 1
        graph.input[0].type.tensor_type.shape.dim[2].dim_value = preproc_dict['hw'][0]
        graph.input[0].type.tensor_type.shape.dim[3].dim_value = 3*preproc_dict['hw'][1]

    print('before make graph')

    new_graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    
    print('before make model')

    model = onnx.helper.make_model(new_graph)

    print('before make shape_inference')

    #model = onnx.shape_inference.infer_shapes(model)

    print('before change opset')
 
    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    #onnx.checker.check_model(onnx_model)
    #onnx.save(model, output)
  
def preproc(model, preproc_yaml):
    preproc_dict = parse_yaml(preproc_yaml) 

    print('---------------------------------')

    for k, v in preproc_dict.items():
        print(k, ':', v)   

    if len(preproc_dict) > 0:
        insert_preproc_node(model, preproc_dict)

    return model        