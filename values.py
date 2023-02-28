import onnx
import numpy as np
import sys

def convert_ort_type_2_np(ort_data_type):
    #logger.info("convert_ort_type_2_np")
    
    types = {
        1 : np.float32,
        2 : np.uint8,
        3 : np.int8,
        4 : np.uint16,
        5 : np.int16,
        6 : np.int32,
        7 : np.int64,
        8 : "",  #string
        9 : np.bool_,
        10 : np.float16,
        11 : np.float64,
        12 : np.uint32,
        13 : np.uint64,
        14 : np.complex64,
        15 : np.complex_,
        16 : ""
    }

    return types.get(ort_data_type, None)

def get_data_list(dtype, tensor):
    data_list = []

    if dtype == 1: #float
        data_list = tensor.float_data

    if dtype == 2: #uint8
        data_list = tensor.int32_data

    if dtype == 3: #int8
        data_list = tensor.int32_data    

    if dtype == 4: #uint16
        data_list = tensor.int32_data

    if dtype == 5: #int16
        data_list = tensor.int32_data

    if dtype == 6: #int32
        data_list = tensor.int32_data

    if dtype == 7: #int64
        data_list = tensor.int64_data

    if dtype == 8: #string
        data_list = tensor.string_data       

    if dtype == 9: #bool
        data_list = tensor.int32_data    

    if dtype == 10: #float16
        data_list = tensor.int32_data
         
    if dtype == 11: #double
        data_list = tensor.double_data      

    if dtype == 12: #uint32
        data_list = tensor.uint64_data  

    if dtype == 13: #uint64
        data_list = tensor.uint64_data

    return data_list

def get_init_value(model, init_name):
    data_list = []

    for init in model.graph.initializer:
        if init.name == init_name:
            print('init.name', init.name)
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            print('np_dtype is ', np_dtype)
            if init.raw_data:
                data_list = np.fromstring(init.raw_data, dtype=np_dtype)
            else:
                data_list = get_data_list(dtype, init)

    return data_list

def get_init_value_and_shape(model, init_name):
    shape = []
    v = get_init_value(model, init_name)
    if v != []:
        for init in model.graph.initializer:
            if init.name == init_name:
                shape = [s for s in init.dims]
                print('got shape{} for {}'.format(shape, init_name))
                break
                
    return v, shape  

def get_tensor_value(tensor): 
    data_list = []

    print('tensor.name', tensor.name)

    dtype = tensor.data_type

    np_dtype = convert_ort_type_2_np(dtype)
    if tensor.raw_data:
        data_list = np.fromstring(tensor.raw_data, dtype=np_dtype)
    else:
        data_list = get_data_list(dtype, tensor)

    return data_list   

def set_tensor_value(tensor, v, dims=[]): 
    data_list = []

    print('tensor.name', tensor.name)

    dtype = tensor.data_type

    np_dtype = convert_ort_type_2_np(dtype)
    if tensor.raw_data:
        #data_list = np.fromstring(tensor.raw_data, dtype=np_dtype)
        tensor.raw_data = v.tostring()
        print('set raw data')
    else:
        data_list = get_data_list(dtype, tensor)
        del data_list[:]
        data_list[:] = v[:]
        print('set data list')

    if len(dims) > 0:
        del tensor.dims[:]
        tensor.dims[:] = dims[:]     

def get_constant_value(model, name):
    value = []
    for n in model.graph.node:
        if n.op_type == 'Constant':
            if name == n.output[0]:
                attributes = n.attribute
                for attr in attributes:
                    if attr.name == 'value':
                        v = get_tensor_value(attr.t)
                        dims = len(v)
                        print('get constant value:', v, dims)
                        value = v
                        break
                break

    return value   

def get_tensor_shape_by_name(model, name):
    shape = []
    for vi in model.graph.value_info:
        if vi.name == name:
            shape = [d.dim_value for d in vi.type.tensor_type.shape.dim]
            print('++++got tensor shape{} for {}'.format(shape, name))
            break

    if shape == []:
        for input_ in model.graph.input:
            if name == input_.name:
                if len(input_.type.tensor_type.shape.dim) > 0:
                    shape = [d.dim_value for d in input_.type.tensor_type.shape.dim]
                    print('----got tensor shape{} for {}'.format(shape, name))
                    break    

    return shape        
