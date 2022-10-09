import torch
import sys
import importlib
import importlib.util
import torchvision

def check_module(module_name):
    """
    Checks if module can be imported without actually
    importing it
    """
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        print("Module: {} not found".format(module_name))
        return None
    else:
        print("Module: {} can be imported".format(module_name))
        return module_spec

def convert_pt_model_and_params_2_onnx(model_path, output, op_set, input_shape_list,
                           model_def_file, model_class_name, output_num):
    print('Begin converting pytorch to onnx...')

    #input_num = len(input_shape_list)
    input_shape_list_ = []

    for input_shape in input_shape_list:
        input_shape=input_shape.strip('[')
        input_shape=input_shape.strip(']')
        input_shape=input_shape.split(',')
        print('got shape:', input_shape)
        input_shape_list_.append(input_shape)

    print('got input_shape_list:', input_shape_list_)

    input_shape_list_int = []

    for input_shape in input_shape_list_:
        shape = [int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
        input_shape_list_int.append(shape)

    print('got input_shape_list_int:', input_shape_list_int)

    input_tensor_list = []
    input_name_list = []
    output_name_list = []

    for i in range(output_num):
        output_name = 'output_' + str(i)
        output_name_list.append(output_name)

    for idx, input_shape in enumerate(input_shape_list_int):
        input_tensor_list.append(torch.randn(*input_shape))
        input_name = 'input_' + str(idx)
        input_name_list.append(input_name)

    input_tensor_tuple = tuple(input_tensor_list)

    print('---input_name_list:', input_name_list)

    dynamic_axes_dict = {}
    for input_name in input_name_list:
        dynamic_axes_dict[input_name] = {0:'-1'}

    for output_name in output_name_list:
        dynamic_axes_dict[output_name] = {0:'-1'}

    #in_shape=[int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
    
    out = output.split('.onnx')[-2]
    #print('out is ', out)

    target_module = ''
    
    if '.' in model_class_name:
        n = model_class_name.rindex('.')
        m = model_class_name[:n]
        model_class_name = model_class_name.split('.')[-1]
        '''
        m = model_class_name[:n+1]
        last = model_class_name.split('.')[-1]
        model_class_name = last
        last_ = last.lower()
        m = m + last_
        print('m is ', m)
        '''
        target_module = m
    else:
        target_module = model_def_file.split('/')[-1]
        target_module = target_module.split('.')[-2]

    print('convert_pt_model_and_params_2_onnx, target_module:', target_module)

    module_find = check_module(target_module)
    if module_find != None:
        print('----get module: ', module_find)
        module = importlib.import_module(target_module)
        cls = getattr(module, model_class_name, None)
        if cls != None:
            model = cls()
            m = torch.load(model_path, map_location=torch.device('cpu'))
            m = m.cpu() #cuda()
            
            torch.onnx.export(
                m,
                input_tensor_tuple, #(x),
                output,
                opset_version=op_set, 
                do_constant_folding=True,   # 是否执行常量折叠优化
                input_names=input_name_list, #["input"],    # 模型输入名
                output_names=output_name_list, #["output"],  # 模型输出名
                #dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}}
                dynamic_axes=dynamic_axes_dict #{'input_0':{0:'-1'}, 'output':{0:'-1'}}
            )
        else:
            print('There is no', model_class_name, ' in', model_def_file)   
    else:
        print('Cound not find', model_def_file)

    #sys.exit() 

def convert_pt_state_dict_2_onnx(model_path, output, op_set, input_shape_list,
                           model_def_file, model_class_name, model_weights_file, output_num):
    print('Begin converting pytorch state dict to onnx...')

    #input_num = len(input_shape_list)
    input_shape_list_ = []

    for input_shape in input_shape_list:
        input_shape=input_shape.strip('[')
        input_shape=input_shape.strip(']')
        input_shape=input_shape.split(',')
        print('got shape:', input_shape)
        input_shape_list_.append(input_shape)

    print('convert_pt_state_dict_2_onnx, got input_shape_list:', input_shape_list_)

    input_shape_list_int = []

    for input_shape in input_shape_list_:
        shape = [int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
        input_shape_list_int.append(shape)

    print('convert_pt_state_dict_2_onnx, got input_shape_list_int:', input_shape_list_int)

    input_tensor_list = []
    input_name_list = []
    output_name_list = []

    for i in range(output_num):
        output_name = 'output_' + str(i)
        output_name_list.append(output_name)

    for idx, input_shape in enumerate(input_shape_list_int):
        input_tensor_list.append(torch.randn(*input_shape))
        input_name = 'input_' + str(idx)
        input_name_list.append(input_name)

    input_tensor_tuple = tuple(input_tensor_list)

    print('convert_pt_state_dict_2_onnx, input_name_list:', input_name_list)

    dynamic_axes_dict = {}
    for input_name in input_name_list:
        dynamic_axes_dict[input_name] = {0:'-1'}

    for output_name in output_name_list:
        dynamic_axes_dict[output_name] = {0:'-1'}

    in_shape=[int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
    out=output.split('.onnx')[-2]
    #print('out is ', out)

    target_module = ''
    
    if '.' in model_class_name:
        n = model_class_name.rindex('.')
        m = model_class_name[:n]
        model_class_name = model_class_name.split('.')[-1]
        '''
        m = model_class_name[:n+1]
        last = model_class_name.split('.')[-1]
        model_class_name = last
        last_ = last.lower()
        m = m + last_
        print('m is ', m)
        '''
        target_module = m
    else:
        target_module = model_def_file.split('/')[-1]
        target_module = target_module.split('.')[-2]

    print('convert_pt_state_dict_2_onnx, target_module:', target_module)

    module_find = check_module(target_module)
    if module_find != None:
        print('----get module: ', module_find)
        module = importlib.import_module(target_module)
        cls = getattr(module, model_class_name, None)
        if cls != None:
            m = cls()
            m.load_state_dict(torch.load(model_weights_file, map_location=torch.device('cpu')))
            m = m.cpu() #cuda()
            #x = torch.randn(int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
            torch.onnx.export(
                m,
                input_tensor_tuple, #x,
                output,
                opset_version=op_set, 
                do_constant_folding=True,   # 是否执行常量折叠优化
                input_names=input_name_list, #["input"],    # 模型输入名
                output_names=output_name_list, #["output"],  # 模型输出名
                dynamic_axes=dynamic_axes_dict #{'input':{0:'-1'}, 'output':{0:'-1'}}
            )
        else:
            print('There is no', model_class_name, ' in', model_def_file)   
    else:
        print('Cound not find', model_def_file)

def convert_pt2onnx(model_path, output, op_set, input_shape_list,
                           model_def_file, model_class_name, model_weights_file, output_num):
    if model_weights_file == '':
        convert_pt_model_and_params_2_onnx(model_path, output, op_set, input_shape_list,
                           model_def_file, model_class_name, output_num)
    else:
        convert_pt_state_dict_2_onnx(model_path, output, op_set, input_shape_list,
                    model_def_file, model_class_name, model_weights_file, output_num)
