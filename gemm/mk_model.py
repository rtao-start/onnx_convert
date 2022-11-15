import onnx
import sys,os
import numpy as np

sys.path.append(os.path.abspath('..'))
from correct_batch import convert_ort_type_2_np, get_data_list

model = onnx.load('../gemm3.onnx')

def mk_transA_alpha_beta(model):
    for init in model.graph.initializer:
        if '662' == init.name:
            print('got it in initializer:', init.int64_data)
            #init.int64_data[0] = -1
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            if init.raw_data:
                params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                print('len(params_list):', len(params_list))
                params_list[0] = 2048
                params_list[1] = 1
                init.raw_data = params_list.tostring()
            else:
                data_list = get_data_list(dtype, init)
                adjust = True
                print('len(data_list):', len(data_list))

                if len(data_list) > 0:
                        data_list[0] = 2048
                        data_list[1] = 1

            break

    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_224':
            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'transA':
                    found = True
                    attr.i = 1
            
            if found == False:
                attr = onnx.helper.make_attribute('transA', 1)
                node.attribute.append(attr)

            found = False
            for attr in attributes:
                if attr.name == 'alpha':
                    found = True
                    attr.f = 3.5
            
            if found == False:
                attr = onnx.helper.make_attribute('alpha', 3.5)
                node.attribute.append(attr)  

            found = False
            for attr in attributes:
                if attr.name == 'beta':
                    found = True
                    attr.f = 5.5
            
            if found == False:
                attr = onnx.helper.make_attribute('beta', 5.5)
                node.attribute.append(attr)            

    onnx.save(model, './gemm_transA_alpha_beta.onnx')

def mk_transA_alpha(model):
    for init in model.graph.initializer:
        if '662' == init.name:
            print('got it in initializer:', init.int64_data)
            #init.int64_data[0] = -1
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            if init.raw_data:
                params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                print('len(params_list):', len(params_list))
                params_list[0] = 2048
                params_list[1] = 1
                init.raw_data = params_list.tostring()
            else:
                data_list = get_data_list(dtype, init)
                adjust = True
                print('len(data_list):', len(data_list))

                if len(data_list) > 0:
                        data_list[0] = 2048
                        data_list[1] = 1

            break

    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_224':
            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'transA':
                    found = True
                    attr.i = 1
            
            if found == False:
                attr = onnx.helper.make_attribute('transA', 1)
                node.attribute.append(attr)

            found = False
            for attr in attributes:
                if attr.name == 'alpha':
                    found = True
                    attr.f = 3.5
            
            if found == False:
                attr = onnx.helper.make_attribute('alpha', 3.5)
                node.attribute.append(attr)  

    onnx.save(model, './gemm_transA_alpha.onnx')


def mk_alpha(model):
    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_224':
            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'alpha':
                    found = True
                    attr.f = 3
            
            if found == False:
                attr = onnx.helper.make_attribute('alpha', 3)
                node.attribute.append(attr)  

    onnx.save(model, './gemm_alpha.onnx')


mk_alpha(model)    


