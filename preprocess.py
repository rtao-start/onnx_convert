# import pyyaml module
import yaml
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

        if 'hw' in data.keys():
            hw_list_ = data['hw']
            if len(hw_list_) == 2:
                hw_list = hw_list_
                print('got std values:', hw_list)

        if 'std' in data.keys():
            std_list_ = data['std']
            if len(std_list_) == 3 or len(std_list_) == 1:
                for n in std_list_:
                    if n > 0.0:
                        std_list.append(1.0/n)
                    else:
                        std_list.append(1.0/1e-6)    

                print('got std values:', std_list)

        if 'mean' in data.keys():
            mean_list_ = data['mean']
            if len(mean_list_) == 3 or len(mean_list_) == 1:
                for n in mean_list_:
                    mean_list.append(int(n))

                print('got mean values:', mean_list)        
        
        if 'resize' in data.keys():
            resize_list_ = data['resize']
            if len(resize_list_) == 2 or len(resize_list_) == 1:
                resize_list = resize_list_
                print('got resize values:', resize_list)

        if 'crop' in data.keys():
            crop_list_ = data['crop']
            if len(crop_list_) == 4:
                crop_list = crop_list_
                print('got crop values:', crop_list)

        if 'control' in data.keys():
            control_list_ = data['control']
            if len(control_list_) == 4:
                control_list = control_list_
                print('got control values:', control_list)

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

def insert_preproc_node(model, preproc_dict, output):
    graph = model.graph
    input_name = graph.input[0].name

    h = preproc_dict['resize'][0]
    w = preproc_dict['resize'][1]

    if h == -1:
        h = graph.input[0].type.tensor_type.shape.dim[2].dim_value

    if w == -1:
        w = graph.input[0].type.tensor_type.shape.dim[3].dim_value    

    const_mean = onnx.helper.make_tensor(name='const_mean',
                            data_type=onnx.TensorProto.UINT8,
                            dims=[len(preproc_dict['mean'])],
                            vals=preproc_dict['mean'])

    graph.initializer.append(const_mean)                        

    const_std = onnx.helper.make_tensor(name='const_std',
                        data_type=onnx.TensorProto.FLOAT,
                        dims=[len(preproc_dict['std'])],
                        vals=preproc_dict['std'])

    graph.initializer.append(const_std)                     

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
    pre_process_output = helper.make_tensor_value_info('pre_process_output', TensorProto.FLOAT, [-1, graph.input[0].type.tensor_type.shape.dim[1].dim_value, h, w])      

    pre_process_node = onnx.helper.make_node(
                    'PreProc',
                    name='preprocess',
                    inputs=[input_name, 'const_std', 'const_mean', 'const_resize','const_crop', 'const_control'],
                    outputs=['pre_process_output'],
                    domain='com.metax-tech')

    graph.node.insert(0, pre_process_node)

    graph.node[1].input[0]='pre_process_output'
             
    graph.input[0].type.tensor_type.elem_type = 2
    graph.input[0].type.tensor_type.shape.dim[2].dim_value = preproc_dict['hw'][0]
    graph.input[0].type.tensor_type.shape.dim[3].dim_value = preproc_dict['hw'][1]

    new_graph = onnx.helper.make_graph(graph.node, graph.name, graph.input, graph.output, graph.initializer)
    model = onnx.helper.make_model(new_graph)
    model = onnx.shape_inference.infer_shapes(model)
 
    op_set = model.opset_import.add()
    op_set.domain = 'com.metax-tech'
    op_set.version = 1

    #onnx.checker.check_model(onnx_model)
    onnx.save(model, output)
  
def preproc(model, output):
    preproc_dict = parse_yaml('./preproc.yaml') 

    print('---------------------------------')

    for k, v in preproc_dict.items():
        print(k, ':', v)   

    if len(preproc_dict) > 0:
        insert_preproc_node(model, preproc_dict, output)    