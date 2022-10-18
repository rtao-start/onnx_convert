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

module_find = check_module('params')
if module_find != None:
    module = importlib.import_module('params')
    obj = getattr(module, 'param_dict', None)
    if obj != None:
        print('got it~')
        params = obj
        #print('params:', type(params))

        for k, v in params.items():
            print(k, ':', v)

        #e = torchvision.models.EfficientNet(inverted_residual_setting=params['inverted_residual_setting'], dropout=0.2)
        e = torchvision.models.EfficientNet(**params)

        print('test finish~')