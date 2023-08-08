import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
import subprocess
import argparse
import cv2
import colorsys
from collections import OrderedDict
from onnx import shape_inference
from onnx import numpy_helper, helper
from PIL import Image, ImageDraw, ImageFont
from detect_new import (DecodeBox, non_max_suppression, yolo_correct_boxes)

logging.basicConfig(level=logging.INFO, filename='./inference.log', filemode='w')

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

logger = logging.getLogger("[INFERENCE]")

def get_output(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output = p.stdout.decode("ascii").strip()
    return output

def generate_onnx_model_input(model, image_npy):
    ort_inputs = {}

    initializer = []
    for init in model.graph.initializer:
        #: ', init.name)
        initializer.append(init.name)

    for input_ in model.graph.input:
        if input_.name not in initializer:
            ort_inputs[input_.name] = image_npy

    return ort_inputs

def get_class(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        print('class:')
        print(class_names)
        return class_names

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    #双三次插值算子
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def run_onnx_model_for_darknet(model_path, detect_img):
    #模型输入的宽和高，需根据实际修改
    model_input_h = 416#608
    model_input_w = 416#608

    #分类数，需根据实际修改
    cls_num = 1#80

    try:
        model = onnx.load(model_path)
    except BaseException as e:
        print('The model cannot be load for: %s' % e)
        return False
    else:
        print('Load mode success')

    #'''
    img_path = detect_img
    img = cv2.imread(img_path)
    target_size = (model_input_h, model_input_w)
    mean = [0.0, 0.0, 0.0]
    orign_size = [img.shape[0], img.shape[1]]
    print('000 orign_size:', orign_size)

    image_blob = cv2.dnn.blobFromImage(img, 1 / 255.0, target_size, mean=mean, swapRB=True, crop=False)
    #'''

    '''###
    img = Image.open(detect_img)
    width, height = img.size
    orign_size = [height, width]
    print('orign_size:', orign_size)
    img = np.array(letterbox_image(img, (model_input_h, model_input_w)), dtype=np.float32)
    img = img.transpose([2, 0, 1])
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    image_blob = img/255.0
    '''###
    
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

    #print('ort_inputs:', ort_inputs)

    outputs = [x.name for x in ort_session.get_outputs()]

    print('output list:')
    print(outputs)

    print('begin run onnx model......')

    try:
        ort_outs = ort_session.run(outputs, ort_inputs)
    except BaseException as e:
        print('Cannot run model for: %s' % e)
        return False
    else:
        print('Run model success')
    
    ort_outs = OrderedDict(zip(outputs, ort_outs))

    for out_name, out_value in ort_outs.items():
        print('output: {}, shape: {}'.format(out_name, out_value.shape))

    ort_outs_0 = ort_outs[outputs[0]]
    ort_outs_1 = ort_outs[outputs[1]]
    ort_outs_2 = ort_outs[outputs[2]]

    output_list = []

    ##三个尺度的预设anchors，需根据实际修改
    anchors = [[147, 153], [323, 111], [278, 257]]
    #anchors = [[116,90],[156,198],[373,326]]
    db = DecodeBox(anchors, cls_num, [model_input_h, model_input_w])
    output_13x13 = db.detect(ort_outs_0)
    output_list.append(output_13x13)

    anchors = [[130, 48], [183, 61], [241, 77]]
    #anchors = [[30,61],[62,45],[59,119]]
    db = DecodeBox(anchors, cls_num, [model_input_h, model_input_w])
    output_26x26 = db.detect(ort_outs_1)
    output_list.append(output_26x26)

    anchors = [[24, 14], [54, 25], [88, 36]]
    #anchors = [[10,13],[16,30],[33,23]]
    db = DecodeBox(anchors, cls_num, [model_input_h, model_input_w])
    output_52x52 = db.detect(ort_outs_2)
    output_list.append(output_52x52)

    detect_output = np.concatenate(output_list, 1)

    '''
    batch_detections = non_max_suppression(detect_output, cls_num,
                                                conf_thres=0.5,
                                                nms_thres=0.45)
    '''

    batch_detections = non_max_suppression(detect_output, cls_num,
                                                conf_thres=0.28,
                                                nms_thres=0.45)                                            

    for batch_detection in batch_detections:
        if isinstance(batch_detection, type(None)) == False:
            batch_detection = batch_detection#.numpy()

            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            top_index = batch_detection[:,4] * batch_detection[:,5] > 0.28
            top_conf = batch_detection[top_index,4]*batch_detection[top_index,5]
            top_label = np.array(batch_detection[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detection[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([model_input_h, model_input_w]), np.array(orign_size), False)
            
            #如果需要在原图画框，可执行以下代码
            image = Image.open(detect_img)
            font = ImageFont.truetype(font='./simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
            thickness = max((np.shape(image)[0] + np.shape(image)[1]) // model_input_h, 1)

            class_names = get_class('./coco_classes.txt')
            hsv_tuples = [(x / len(class_names), 1., 1.)
                        for x in range(len(class_names))]

            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))

            colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

            for i, c in enumerate(top_label):
                predicted_class = class_names[c]
                score = top_conf[i]

                top, left, bottom, right = boxes[i]
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5

                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
                right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                #label_size = draw.textsize(label, font)
                text_bbox = draw.textbbox((0, 0), label, font=font)
                label_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
                label = label.encode('utf-8')

                print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=colors[class_names.index(predicted_class)])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[class_names.index(predicted_class)])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

                #image.show()

                old_img = detect_img.split('/')[-1]
                output_path = 'inference_' + old_img 
                image.save(output_path)

            #####画框、保存 finish    
            ###########################################################################


            #如果不需要画框，只需要得到推理结果(x,y,w,h,prob,class)，可执行以下代码
            '''
            box_corner = np.ones_like(boxes)
            box_corner[:, 0] = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
            box_corner[:, 1] = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
            box_corner[:, 2] = boxes[:, 3] - boxes[:, 1]
            box_corner[:, 3] = boxes[:, 2] - boxes[:, 0]
            #boxes[:, :, :4] = box_corner[:, :, :4]

            extra = np.empty([box_corner.shape[0], 2]) 

            for i, c in enumerate(top_label):
                score = top_conf[i]
                extra[i] = np.array([score, c])

            result = np.concatenate((box_corner, extra), axis=1)

            print('detect result:', result)

            return result
            '''

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='')

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

    run_onnx_model_for_darknet(onnx_model_path, input_pic)

    