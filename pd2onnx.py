import paddle
import importlib
import importlib.util
import sys, os

paddle.disable_signal_handler()

def get_paddle_files(model_path):
    items = os.listdir(model_path)
    has_pdmodel = False
    has_pdiparams = False
    pdmodel = ''
    pdiparams = ''

    for f in items:
        if f.endswith(".pdmodel"):
            has_pdmodel = True
            pdmodel = f
        elif f.endswith(".pdiparams"):
            has_pdiparams = True
            pdiparams = f

        if has_pdmodel == True and has_pdiparams == True:
            if model_path.endswith("/"):
                pdmodel = model_path + pdmodel
                pdiparams = model_path + pdiparams
            else:
                pdmodel = model_path + '/' + pdmodel
                pdiparams = model_path + '/' + pdiparams

            print('got pdmodel:{}, pdiparams:{}'.format(pdmodel, pdiparams))

            break        
    
    return pdmodel, pdiparams  

def convert_pdstatic2onnx(model_path, output, op_set):
    print('Begin converting static paddle to onnx...')
    pdmodel, pdiparams = get_paddle_files(model_path)

    cmd = 'paddle2onnx --model_dir ' + model_path + ' --opset_version ' + str(op_set) + ' --save_file ' + output \
            + ' --model_filename '  + pdmodel + ' --params_filename ' + pdiparams

    print('convert_paddle2onnx: ', cmd)

    os.system(cmd)

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

def convert_pddynamic2onnx(model_path, output, op_set, input_shape,
                           model_def_file, model_class_name, paddle_input_type, model_weights_file):
    print('Begin converting dynamin paddle to onnx...')

    in_shape=[int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3])]
    out=output.split('.onnx')[-2]
    print('out is ', out)

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

    print('convert_pddynamic2onnx, target_module:', target_module)

    module_find = check_module(target_module)
    if module_find != None:
        print('----get module: ', module_find)
        module = importlib.import_module(target_module)
        cls = getattr(module, model_class_name, None)
        if cls != None:
            model = cls()
            model.set_dict(paddle.load(model_weights_file))
            model.eval()
            input_spec = paddle.static.InputSpec(shape=in_shape, dtype=paddle_input_type, name='input')
            paddle.onnx.export(model, out, input_spec=[input_spec], opset_version=op_set)
        else:
            print('There is no', model_class_name, ' in', model_def_file)   
    else:
        print('Cound not find', model_def_file)

    #sys.exit()    

def convert_pd2onnx(model_path, output, op_set, input_shape, model_def_file, model_class_name, paddle_input_type, model_weights_file):
    if is_dynamic_paddle(input_shape, model_def_file, model_class_name, model_weights_file):
        convert_pddynamic2onnx(model_path, output, op_set, input_shape, model_def_file, model_class_name, paddle_input_type, model_weights_file)              
    else:
        convert_pdstatic2onnx(model_path, output, op_set)

def is_dynamic_paddle(input_shape, model_def_file, model_class_name, model_weights_file):
    if model_class_name != '' and '.' not in model_class_name:
        return input_shape != '' and model_def_file != '' and model_weights_file != ''
    elif model_class_name != '' and '.' in model_class_name:
        return input_shape != '' and model_weights_file != ''
    else:
        return False    

               

