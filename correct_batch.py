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

def get_data_list(dtype, init):
    data_list = []

    if dtype == 2: #uint8
        data_list = init.uint8_data

    if dtype == 3: #int8
        data_list = init.int8_data    

    if dtype == 4: #uint16
        data_list = init.uint16_data

    if dtype == 5: #int16
        data_list = init.int16_data

    if dtype == 6: #int32
        data_list = init.int32_data

    if dtype == 12: #uint32
        data_list = init.uint32_data  

    if dtype == 7: #int64
        data_list = init.int64_data

    if dtype == 13: #uint64
        data_list = init.uint64_data

    return data_list                      


def correct_batch_for_opset_convert(model):
    input_shape = model.graph.input[0].type.tensor_type.shape.dim
    input_batch = input_shape[0].dim_value
    print('correct_batch_for_opset_convert, input_batch:', input_batch)

    for idx in range(len(model.graph.value_info)):
      dim_proto_input = model.graph.value_info[idx].type.tensor_type.shape.dim[0]
      if dim_proto_input.dim_value != input_batch:
        #print('$$$', dim_proto_input.dim_value, input_batch)
        dim_proto_input.dim_value = input_batch

    reshape_input_list = []
    for node in model.graph.node:
        if node.op_type == 'Reshape':
            if node.input[1] not in reshape_input_list:
                reshape_input_list.append(node.input[1])

    if len(reshape_input_list):
        for reshape_input in reshape_input_list:
            for id, init in enumerate(model.graph.initializer):
                if init.name == reshape_input:
                    print('init.name', init.name)
                    dtype = init.data_type
                    np_dtype = convert_ort_type_2_np(dtype)
                    if init.raw_data:
                        params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                        #print('@@@@', params_list[0], input_batch)
                        if params_list[0] != input_batch:
                            params_list[0] = input_batch
                            init.raw_data = params_list.tostring()
                    else:
                        #print('Only support modify initializer raw data for now, cannot correct this model!!!')    
                        #sys.exit()
                        #break
                        data_list = get_data_list(dtype, init)
                        #print('#####', data_list[0], input_batch)
                        if len(data_list) > 0 and data_list[0] != input_batch:
                            print('data_list:', data_list[0])
                            data_list[0] = input_batch

    output_shape = model.graph.output[0].type.tensor_type.shape.dim
    output_batch = output_shape[0].dim_value

    if output_batch != input_batch:
        output_shape[0].dim_value = input_batch



