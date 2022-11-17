import onnx
import sys, os
import numpy as np
import copy
from onnx import TensorProto

from ttc import proc_gemm_ttc_ttt
from tcc import proc_gemm_tcc
from tct import proc_gemm_tct
from ctt import proc_gemm_ctt
from ctc import proc_gemm_ctc

sys.path.append(os.path.abspath('..'))

constant_combination_case_1A = [False, False, False]
constant_combination_case_1B = [False, False, True]
constant_combination_case_1C = [False, False]

constant_combination_case_1 = [constant_combination_case_1A, constant_combination_case_1B, constant_combination_case_1C]

constant_combination_case_2 = [[False, True, False]]
#input_combination_case_2B = [False, True]

constant_combination_case_3A = [False, True, True]
constant_combination_case_3B = [False, True]

constant_combination_case_3 = [constant_combination_case_3A, constant_combination_case_3B]

constant_combination_case_4 = [[True, False, False]]
#input_combination_case_4B = [True, False]

constant_combination_case_5A = [True, False, True]
constant_combination_case_5B = [True, False]

constant_combination_case_5 = [constant_combination_case_5A, constant_combination_case_5B]

constant_combination_case = [constant_combination_case_1, constant_combination_case_2, constant_combination_case_3, constant_combination_case_4, constant_combination_case_5]

transpose_combination_case_A = [0, 0]
transpose_combination_case_B = [0, 1]
transpose_combination_case_C = [1, 0]
transpose_combination_case_D = [1, 1]

def proc_gemm_case_4(model, node_id, node, attr):
    pass        

def proc_gemm_case_5(model, node_id, node, attr):
    pass

proc_gemm = {
    "case_1": proc_gemm_ttc_ttt,
    "case_2": proc_gemm_tct,
    "case_3": proc_gemm_tcc,
    "case_4": proc_gemm_ctt,
    "case_5": proc_gemm_ctc
}

def gemm_convert(model, output):
    dict_sm = {}
    dict_mul = {}

    got_swish = False

    init_list = []

    const_list = []

    for init in model.graph.initializer:
        init_list.append(init.name)

    for node in model.graph.node:
        if node.op_type == 'Constant': 
            const_list.append(node.output[0])

    skip = False            

    for node_id, node in enumerate(model.graph.node):
        print('loop model', node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
                 ", op:", node.op_type)

        if skip > 0:
            skip = skip - 1
            continue         

        input_const_flag = [False, False]
        length = len(node.input)
        if length == 3:
            input_const_flag.append(False)

        print('input_const_flag:', input_const_flag)    

        if node.op_type == 'Gemm':
            print('===== got gemm', node.name, node.input, node.output)

            alpha = 1.0
            beta = 1.0
            transA = 0
            transB = 0

            attributes = node.attribute
            for attr in attributes:
                if attr.name == 'alpha':
                    alpha = attr.f
                    print('alpha:', alpha)
                
                if attr.name == 'beta':
                    beta = attr.f
                    print('beta:', beta)

                if attr.name == 'transA':
                    transA  = attr.i
                    print('transA :', transA)

                if attr.name == 'transB':
                    transB = attr.i
                    print('transB :', transB) 

            gemm_attr = {}
            gemm_attr['alpha'] = alpha
            gemm_attr['beta'] = beta
            gemm_attr['transA'] = transA
            gemm_attr['transB'] = transB                   

            for i, input in enumerate(node.input):
                if input in init_list:
                    print('init input:', input, i)
                    input_const_flag[i] = True
                elif input in const_list:
                    print('const input:', input, i)
                    input_const_flag[i] = True        

            print('--- input_const_flag:', input_const_flag) 

            for index, c in enumerate(constant_combination_case):
                for e in c:
                    if e == input_const_flag:
                        print('index = ', index)
                        ii = index + 1
                        skip = proc_gemm['case_' + str(ii)](model, node_id, node, gemm_attr)
                        onnx.save(model, output)

model = onnx.load('./gemm_ttt.onnx')
#model = onnx.load('../gemm3.onnx')
model = onnx.shape_inference.infer_shapes(model) 

gemm_convert(model, './tmp.onnx')