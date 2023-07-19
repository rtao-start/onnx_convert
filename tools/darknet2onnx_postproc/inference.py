import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
import subprocess
import torch
import argparse
import cv2

from collections import OrderedDict
from onnx import shape_inference
from onnx import numpy_helper, helper

from detect import (DecodeBox, non_max_suppression, yolo_correct_boxes)

logging.basicConfig(level=logging.INFO, filename='./inference.log', filemode='w')

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

logger = logging.getLogger("[INFERENCE]")

'''
  --------------------ONNX Data Type-----------------
  enum DataType {
    UNDEFINED = 0;
    // Basic types.
    FLOAT = 1;   // float
    UINT8 = 2;   // uint8_t
    INT8 = 3;    // int8_t
    UINT16 = 4;  // uint16_t
    INT16 = 5;   // int16_t
    INT32 = 6;   // int32_t
    INT64 = 7;   // int64_t
    STRING = 8;  // string
    BOOL = 9;    // bool

    // IEEE754 half-precision floating-point format (16 bits wide).
    // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
    FLOAT16 = 10;

    DOUBLE = 11;
    UINT32 = 12;
    UINT64 = 13;
    COMPLEX64 = 14;     // complex with float32 real and imaginary components
    COMPLEX128 = 15;    // complex with float64 real and imaginary components

    // Non-IEEE floating-point format based on IEEE754 single-precision
    // floating-point number truncated to 16 bits.
    // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
    BFLOAT16 = 16;

    // Future extensions go here.
  }
'''

network_size = (416,416)

def convert_ort_type_2_np(ort_data_type):

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


def get_tensor_type_by_data_type(dtype):

    print('get_tensor_type_by_data_type: ', dtype.name)

    types__ = {
        'float16' : TensorProto.FLOAT16,
        'float32' : TensorProto.FLOAT,
        'int8' : TensorProto.INT8,
        'int16' : TensorProto.INT16,
        'int32' : TensorProto.INT32,
        'int64' : TensorProto.INT64,
        'uint8' : TensorProto.UINT8,
        'uint16' : TensorProto.UINT16,
        'uint32' : TensorProto.UINT32,
        'uint64' : TensorProto.UINT64,
        'float64' : TensorProto.DOUBLE
    }

    t = types__.get(dtype.name, None) 
    #print('t = ', t)

    return t 

def get_output(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output = p.stdout.decode("ascii").strip()
    return output

def get_cosine(gpu_array, cpu_array):
    x = np.square(gpu_array)
    x = np.sum(x) 
    x = np.sqrt(x)

    y = np.square(cpu_array)
    y = np.sum(y) 
    y = np.sqrt(y)

    z = gpu_array * cpu_array
    z = sum(z)

    print('x y z:', x, y, z)

    cosine_sim  = (z + 1e-7) / ((x * y) + 1e-7) # eps

    cosine_sim = max(cosine_sim, 1.0)

    cosine = np.mean(cosine_sim)

    print('-----cosine:', cosine)

    #cosine = max(cosine, 1.0)

    cosine = 1.0 - cosine

    #cosine = max(0, cosine)

    print('+++++cosine:', cosine)

    return cosine  

def get_mse(gpu_array, cpu_array):
    diff_array = np.subtract(cpu_array, gpu_array)
    x = np.square(diff_array)
    mse = np.mean(x)

    print('mse:', mse)

    return mse  

def get_snr(gpu_array, cpu_array):
    diff_array = np.subtract(cpu_array, gpu_array)
    x = np.square(diff_array)
    x = np.sum(x)

    y = np.square(cpu_array)
    y = np.sum(y) 

    snr = (x) / (y + 1e-7)

    snr = np.mean(snr)

    print('snr:', snr)

    return snr  
    
precision_cmp_method = {
    "mse": get_mse,
    "cosine": get_cosine,
    "snr": get_snr
}

precision_cmp_str = 'snr'
precision_threshold = 0.1 

import math 

def compare_result(ort_outs_onnx, ort_outs_origin): 
    match=True 
    seq = 0


    v1 = ort_outs_onnx.flatten()
    print('v1:', v1.tolist()[:32])

    #print('ort_outs_origin:', ort_outs_origin[:, 21660:21665, 4:10])
  
    v2 = ort_outs_origin.flatten()
    print('v2:', v2.tolist()[:32])

    cmp_value = get_snr(v1, v2) 
    if cmp_value > 0.001:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('WARNING: output is abnormal, please check it~~')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        return False

    return True    

def generate_onnx_model_input(model, image_npy):
    ort_inputs = {}

    initializer = []
    for init in model.graph.initializer:
        #: ', init.name)
        initializer.append(init.name)

    for input_ in model.graph.input:
        if input_.name not in initializer:
            print('-----input_ name is', input_.name)

            ort_inputs[input_.name] = image_npy

    return ort_inputs

def run_onnx_model_for_darknet(model_path, detect_img):
    model_input_h = 416
    model_input_w = 416

    try:
        model = onnx.load(model_path)
    except BaseException as e:
        print('The model cannot be load for: %s' % e)
        return False
    else:
        print('Load mode success')

    img_path = input_pic
    img = cv2.imread(img_path)
    target_size = (model_input_h, model_input_w)
    mean = [0.0, 0.0, 0.0]
    orign_size = [img.shape[0], img.shape[1]]

    image_blob = cv2.dnn.blobFromImage(img, 1 / 255.0, target_size, mean=mean, swapRB=True, crop=False)

    ori_outputs = [x.name for x in model.graph.output]
    ori_outputs_backup=model.graph.output[:]

    del model.graph.output[:]

    intermedia_list = ['106_convolutional', '094_convolutional', '082_convolutional']
    for node in model.graph.node:
        for output in node.output:
            if output not in ori_outputs and output in intermedia_list:
                print('set intermedia output', output)
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                
    EP_list = ['CPUExecutionProvider']

    try:
        ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=EP_list)
    except BaseException as e:
        print('Create InferenceSession Failed: %s' % e)
        return False
    else:
        print('Create InferenceSession success')

    ort_inputs = generate_onnx_model_input(model, image_blob)

    print('ort_inputs:', ort_inputs)

    outputs = [x.name for x in ort_session.get_outputs()]

    print('output list:')
    print(outputs)

    print('begin run cpu......')

    try:
        ort_outs = ort_session.run(outputs, ort_inputs)
    except BaseException as e:
        print('Cannot run model for: %s' % e)
        return False
    else:
        print('Run model success')
    
    ort_outs = OrderedDict(zip(outputs, ort_outs))

    print('outputs[0].shape:', ort_outs[outputs[0]].shape)
    print('outputs[1].shape:', ort_outs[outputs[1]].shape)
    print('outputs[2].shape:', ort_outs[outputs[2]].shape)

    ort_outs_0 = ort_outs[outputs[0]]
    ort_outs_1 = ort_outs[outputs[1]]
    ort_outs_2 = ort_outs[outputs[2]]

    output_list = []

    anchors = [[147, 153], [323, 111], [278, 257]]
    db = DecodeBox(anchors, 1, [model_input_h, model_input_w])
    ort_outs_0 = torch.from_numpy(ort_outs_0)
    output_13x13 = db.detect(ort_outs_0)
    output_list.append(output_13x13)

    anchors = [[130, 48], [183, 61], [241, 77]]
    db = DecodeBox(anchors, 1, [model_input_h, model_input_w])
    ort_outs_1 = torch.from_numpy(ort_outs_1)
    output_26x26 = db.detect(ort_outs_1)
    output_list.append(output_26x26)

    anchors = [[24, 14], [54, 25], [88, 36]]
    db = DecodeBox(anchors, 1, [model_input_h, model_input_w])
    ort_outs_2 = torch.from_numpy(ort_outs_2)
    output_52x52 = db.detect(ort_outs_2)
    output_list.append(output_52x52)

    detect_output = torch.cat(output_list, 1)
    batch_detections = non_max_suppression(detect_output, 1,
                                                conf_thres=0.5,
                                                nms_thres=0.45)

    for batch_detection in batch_detections:
        if isinstance(batch_detection, type(None)) == False:
            batch_detection = batch_detection.numpy()

            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            top_index = batch_detection[:,4] * batch_detection[:,5] > 0.5
            top_conf = batch_detection[top_index,4]*batch_detection[top_index,5]
            top_label = np.array(batch_detection[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detection[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([model_input_h, model_input_w]), np.array(orign_size))
            
            box_corner = np.ones_like(boxes)
            box_corner[:, 0] = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
            box_corner[:, 1] = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            box_corner[:, 2] = boxes[:, 3] - boxes[:, 1]
            box_corner[:, 3] = boxes[:, 2] - boxes[:, 0]
            #boxes[:, :, :4] = box_corner[:, :, :4]

            extra = np.empty([box_corner.shape[0], 2]) 

            for i, c in enumerate(top_label):
                score = top_conf[i]

                top, left, bottom, right = box_corner[i]

                extra[i] = np.array([score, c])

            result = np.concatenate((box_corner, extra), axis=1)

            print('detect result:', result)

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Convert caffe/tensorflow/torch/paddle/darknet model to ONNX.')

    parser.add_argument("--onnx_model_path",
                        type=str)

    parser.add_argument("--input_pic",
                        type=str)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    onnx_model_path = args.onnx_model_path
    input_pic = args.input_pic

    print('onnx_model_path:{}, input_pic:{}'.format(onnx_model_path, input_pic))

    dn_out = run_onnx_model_for_darknet(onnx_model_path, input_pic)

    