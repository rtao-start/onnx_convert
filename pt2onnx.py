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

def convert_pt_model_and_params_2_onnx(model_path, output, op_set, input_shape,
                           model_def_file, model_class_name):
    print('Begin converting pytorch to onnx...')

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
            x = torch.randn(int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
            torch.onnx.export(
                m,
                x,
                output,
                opset_version=op_set, 
                do_constant_folding=True,   # 是否执行常量折叠优化
                input_names=["input"],    # 模型输入名
                output_names=["output"],  # 模型输出名
                #dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}}
                dynamic_axes={'input':{0:'-1'}, 'output':{0:'-1'}}
            )
        else:
            print('There is no', model_class_name, ' in', model_def_file)   
    else:
        print('Cound not find', model_def_file)

    #sys.exit() 

def convert_pt_state_dict_2_onnx(model_path, output, op_set, input_shape,
                           model_def_file, model_class_name, model_weights_file):
    print('Begin converting pytorch state dict to onnx...')

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
            x = torch.randn(int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
            torch.onnx.export(
                m,
                x,
                output,
                opset_version=op_set, 
                do_constant_folding=True,   # 是否执行常量折叠优化
                input_names=["input"],    # 模型输入名
                output_names=["output"],  # 模型输出名
                dynamic_axes={'input':{0:'-1'}, 'output':{0:'-1'}}
            )
        else:
            print('There is no', model_class_name, ' in', model_def_file)   
    else:
        print('Cound not find', model_def_file)

    #sys.exit() 

def convert_pt2onnx(model_path, output, op_set, input_shape,
                           model_def_file, model_class_name, model_weights_file):
    if model_weights_file == '':
        convert_pt_model_and_params_2_onnx(model_path, output, op_set, input_shape,
                           model_def_file, model_class_name)
    else:
        convert_pt_state_dict_2_onnx(model_path, output, op_set, input_shape,
                    model_def_file, model_class_name, model_weights_file)
