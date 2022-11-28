import onnx
import sys,os
import numpy as np
from onnx import helper
from onnx import TensorProto

sys.path.append(os.path.abspath('..'))
from correct_batch import convert_ort_type_2_np, get_data_list

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


def mk_tct(model):
    for init in model.graph.initializer:
        if '662' == init.name:
            print('got it in initializer:', init.int64_data)
            #init.int64_data[0] = -1
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            if init.raw_data:
                params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                print('len(params_list):', len(params_list))
                params_list[0] = 1
                params_list[1] = 2048
                init.raw_data = params_list.tostring()
            else:
                data_list = get_data_list(dtype, init)
                adjust = True
                print('len(data_list):', len(data_list))

                if len(data_list) > 0:
                        data_list[0] = 1
                        data_list[1] = 2048

            break

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

            found = False
            for attr in attributes:
                if attr.name == 'beta':
                    found = True
                    attr.f = 2
            
            if found == False:
                attr = onnx.helper.make_attribute('beta', 2)
                node.attribute.append(attr)      

            output_shape = [1000]
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, output_shape)

            const_starts = onnx.helper.make_tensor(name='const_starts',
                        data_type=onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[0])

            const_ends = onnx.helper.make_tensor(name='const_ends',
                        data_type=onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[1000])

            const_axis = onnx.helper.make_tensor(name='const_axis',
                        data_type=onnx.TensorProto.INT64,
                        dims=[1],
                        vals=[1])            

            slice_node = helper.make_node(
                            'Slice', # node name
                            ['663', 'const_starts', 'const_ends', 'const_axis'],
                            ['Y'], # outputs
                            )  

            model.graph.node.insert(idx, slice_node)
            model.graph.initializer.append(const_starts)
            model.graph.initializer.append(const_ends)
            model.graph.initializer.append(const_axis)

            node.input[2] = 'Y'

            break

    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, './gemm_tct.onnx')

def mk_ttt(model):
    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_224':
            output_shape = [1000, 2048]
            Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, output_shape)

            const_repeats = onnx.helper.make_tensor(name='const_repeats',
                        data_type=onnx.TensorProto.INT64,
                        dims=[2],
                        vals=[1000, 1])

            tile_node = helper.make_node(
                            'Tile', # node name
                            ['663', 'const_repeats'],
                            ['Z'], # outputs
                            )  

            model.graph.node.insert(idx+1, tile_node)
            model.graph.initializer.append(const_repeats)

            node.input[1] = 'Z'

            break

    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, './gemm_ttt.onnx')

def mk_ctt(model):
    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_3':
            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'beta':
                    found = True
                    attr.f = 2
            
            if found == False:
                attr = onnx.helper.make_attribute('beta', 2)
                node.attribute.append(attr)  

            output_shape = [1, 1]
            Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT16, output_shape)

            rm_node = helper.make_node(
                            'ReduceMean', # node name
                            ['6'],
                            ['Z'], # outputs
                            axes=[0,1],
                            keepdims=1,
                        )  

            model.graph.node.insert(idx+1, rm_node)
            node.input[2] = 'Z'

            break

    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, './gemm_ctt.onnx')

def mk_ttt_transA(model):
    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_224':
            output_shape = [2048, 1]
            TP = helper.make_tensor_value_info('TP', TensorProto.FLOAT, output_shape)

            tp_node = helper.make_node(
                            'Transpose', # node name
                            ['663'],
                            ['TP'], # outputs
                            )  

            model.graph.node.insert(idx+1, tp_node)

            node.input[0] = 'TP'

            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'transA':
                    found = True
                    attr.i = 1
            
            if found == False:
                attr = onnx.helper.make_attribute('transA', 1)
                node.attribute.append(attr)  

            break

    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, './gemm_ttt_transA.onnx')

def mk_ttt_no_trans(model):
    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_224':
            output_shape = [2048, 1000]
            TP = helper.make_tensor_value_info('TPB', TensorProto.FLOAT, output_shape)

            tp_node = helper.make_node(
                            'Transpose', # node name
                            ['Z'],
                            ['TPB'], # outputs
                            )  

            model.graph.node.insert(idx+1, tp_node)

            node.input[1] = 'TPB'

            attributes = node.attribute
            found = False
            for attr in attributes:
                if attr.name == 'transB':
                    found = True
                    attr.i = 0
            
            if found == False:
                attr = onnx.helper.make_attribute('transB', 0)
                node.attribute.append(attr)  

            for attr in attributes:
                if attr.name == 'alpha':
                    attr.f = 1
                    break

            for attr in attributes:
                if attr.name == 'beta':
                    attr.f = 1

            break

    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, './gemm_ttt_transA_only.onnx')

def mk_ttc_beta(model):
    for idx, node in enumerate(model.graph.node): 
        if node.name == 'Gemm_2':
            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'beta':
                    attr.f = 2

            break

    model = onnx.shape_inference.infer_shapes(model)
    onnx.save(model, './gemm_ttc_beta.onnx')

model = onnx.load('./gemm_test/pygcn_fp16_maca.onnx')
mk_ttc_beta(model)    


