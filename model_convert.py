# -*- coding: utf-8 -*-

import os
import onnx
from onnx import version_converter
import copy
import numpy as np
import logging
import log
import onnxruntime
import sys, getopt
import json
import argparse
import h5py
#import tensorflow as tf
import time
import fuse
from swish_convert import merge_swish_and_hard_swish
from mish_convert import merge_mish
from hard_sigmoid_convert import merge_hard_sigmiod
from gelu_fuse import merge_gelu
import bn2conv
import values
import version_check
import operation
import copy

from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
from caffe2onnx.src.caffe2onnx import Caffe2Onnx
from onnxsim.onnx_simplifier import simplify
from float16 import convert_float_to_float16
from preprocess import preproc
from postprocess import postproc
from correct_batch import correct_batch_for_opset_convert, convert_ort_type_2_np, get_data_list
#from pd2onnx import convert_pd2onnx, is_dynamic_paddle
#from pt2onnx import convert_pt2onnx
from gemm.gemm_cvt import gemm_convert
from resize_convert import merge_resize
from ln_convert import merge_layernorm
from matmul2gemm import matmul_2_gemm
from mha_optimization import mha_optimizer
from input_fp32_to_uint8 import fp32_to_uint8
from common import MXC_CONFIG

using_wheel = False

support_mish = 0

concat_result = 0

inputs_as_nchw = ''

#logging.basicConfig(level=logging.INFO, filename='./convert.log', filemode='w')
#logger = logging.getLogger("[MacaConverter]")

logger = log.getLogger('[MacaConverter]', log.INFO)
file_handler = logging.FileHandler('./convert.log')
file_handler.setLevel(logging.CRITICAL)
logger.addHandler(file_handler)

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

import argparse

valid_model_type = ['caffe', 'pytorch', 'tf-h5', 'tf-ckpt', 'tf-sm', 'tf-graph', 'darknet', 'onnx', 'paddle']

optimization_op_list = ['Max', 'Min', 'Sum', 'Mean']

############ Error Code Define #################
exit_code_normal = 0
exit_code_no_caffe_cfg_file = -1
exit_code_sm2onnx = -2
exit_code_h52onnx = -3
exit_code_ckpt2onnx = -4
exit_code_pb2onnx = -5
exit_code_no_darknet_cfg_or_weights = -6
exit_code_convert_darknet2onnx = -7
exit_code_fuse_mish = -8
exit_code_check_optimization_op = -9
exit_code_check_modify_onnx2dynamic = -10
exit_code_check_convert_gap_2_ap = -11
exit_code_model_not_exist = -12
exit_code_invalid_model_type = -13
exit_code_pytorch_no_input_shape = -14
exit_code_tensorflow_no_inputs_or_outputs = -15
exit_code_extract_sub_no_inputs_or_outputs = -16
exit_code_model_type_not_onnx = -17
#########################################################

def set_using_wheel():
   global using_wheel
   using_wheel = True

def parse_args():
   parser = argparse.ArgumentParser(description='Convert caffe/tensorflow/torch/paddle/darknet model to ONNX.')

   parser.add_argument("--model_path",
                        type=str,  
                        help="Input path(model file or folder)")

   parser.add_argument("--model_type",
                        type=str,
                        help="Input model type(ex: caffe/pytorch/tf-h5/...)")

   parser.add_argument("--output",
                        type=str,
                        help="Output path(ex: ./output.onnx)")

   parser.add_argument("--concat_result",
                        type=int, required=False,
                        choices=[0, 1],
                        default=0,
                        help="When convert darknet to onnx, if set 1, the tool will concat the results")

   parser.add_argument("--op_set",
                        type=int, required=False,
                        help="Set op_set version(default: 11)")

   #for simplify
   parser.add_argument("--simplify",
                        type=int, required=False,
                        choices=[0, 1, 2],
                        default=1,
                        help="Simplify the model(0:no simplify;1:do simplify; 2:for dynamic model)")     

   parser.add_argument("--simplify_hw",
                        type=str, 
                        required=False,
                        default='',
                        help="When h/w is -1, you can specify h/w as you expected(together with --simplify 2)")  

   #force simplify
   parser.add_argument("--force_simplify",
                        type=int, required=False,
                        choices=[0, 1, 2],
                        default=0,
                        help="Force simplify the model(0:no simplify;1:do simplify; 2:for dynamic model)")   

   #for pytorch/dynamic_paddle
   parser.add_argument("--input_shape",
                        type=str, 
                        required=False,
                        default='',
                        help="Input shape for pytorch/paddle(ex: [1,3,224,224] or [1,3,224,224]/[1,3,56,56])")

   #######
   parser.add_argument("--inputs",
                        type=str, 
                        required=False,
                        default='',
                        help="When do checkpoint2ONNX/graph2ONNX/onnx_sub_graph, you should specify inputs(ex: --inputs image:0)")

   parser.add_argument("--outputs",
                        type=str, 
                        required=False,
                        default='',
                        help="When do checkpoint2ONNX/graph2ONNX/onnx_sub_graph, you should specify outputs(ex: --outputs predict:0)")

   #for extract sub graph
   parser.add_argument("--extract_sub",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will extract sub graph by specify inputs/outputs") 

   #for dynamic batch size
   parser.add_argument("--dynamic_batch",
                        type=int, 
                        required=False,
                        default=0,
                        choices=[0, 1],
                        help="If set 1, the tool will convert batch size to -1")

   #for fp32-->fp16
   parser.add_argument("--fp32_to_fp16",
                        type=int, 
                        required=False,
                        default=0,
                        choices=[0, 1],
                        help="If set 1, the tool will convert fp32 to fp16 in the model")

   parser.add_argument("--support_mish",
                        type=int, 
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Softplus+Tanh+Mul to Mish") 

   #insert preproc node
   parser.add_argument("--preproc_yaml",
                        type=str, 
                        required=False,
                        default='',
                        help="If specify preprocess yaml file, the tool will insert preproc node in the beginning of the model")

   parser.add_argument("--postproc_yaml",
                        type=str, 
                        required=False,
                        default='',
                        help="If specify postprocess yaml file, the tool will insert postproc node in the ending of the model")                     

   #for paddle dynamic model or pytorch
   parser.add_argument("--model_def_file",
                        type=str, 
                        required=False,
                        default='',
                        help="Paddle/pytorch model definition file location(ex: --model_def_file ./cnn.py)")

   parser.add_argument("--model_weights_file",
                        type=str, 
                        required=False,
                        default='',
                        help="Paddle/pytorch model weights file location(ex: --model_weights_file ./0.99667.pth)")                     

   parser.add_argument("--model_class_name",
                        type=str, 
                        required=False,
                        default='',
                        help="Paddle/pytorch model calss name(ex: --model_class_name CNN)")

   parser.add_argument("--model_input_type",
                        type=str, 
                        required=False,
                        #choices=['float', 'float32', 'float16', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'bool'],
                        default='',
                        help="Paddle/pytorch input type(default float, choice is ['float', 'float32', 'float16', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'uint64', 'int64', 'bool'])")

   parser.add_argument("--params_file",
                        type=str, 
                        required=False,
                        default='',
                        help="Paddle/pytorch params declaration file location(ex: --params_file ./params.py)")

   #for tensorflow 
   parser.add_argument("--inputs_as_nchw",
                        type=str, 
                        required=False,
                        default='',
                        help="When some input of tensorflow model is nhwc, you can use it(ex: --inputs_as_nchw image:0) to convert to nchw")
    
   #for gap-->ap 
   parser.add_argument("--gap_to_ap",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert GlobalAveragePool to AveragePool for hardware acceleration") 

   #for pad+pool fuse
   parser.add_argument("--fuse_pad_pool",
                        type=int, 
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Pad into Pool") 

   #for merge swish
   parser.add_argument("--support_swish",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert Sigmoid+Mul to Swish; HardSigmoid+Mul to HardSwish")   

   #for convert BN to GroupConv(1x1)
   parser.add_argument("--bn_to_conv",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert BN to group 1x1_Conv")          

   #for pytorch/paddle
   parser.add_argument("--output_num",
                        type=int, 
                        required=False,
                        default=1,
                        help="If output num of pytorch model > 1, you can specify it by --output_num")                                                                                                               

   #for pytorch
   parser.add_argument("--keep_batch",
                        type=int, 
                        choices=[0, 1],
                        required=False,
                        default=1,
                        help="For pytorch, if set 1, the tool will keep model batch size(if 0, set it to dynamic(-1))") 

   #for convert gemm
   parser.add_argument("--gemm_optimization",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert gemm to fc or matmul+add+mul")     
   
   #for convert Reshape+Expand+Reshape to Resize
   parser.add_argument("--expand_to_resize",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert Reshape+Expand+Reshape to Resize")    

   #reset model value_info(some model(batch=-1) may have wrong value info for middle node)) 
   parser.add_argument("--reset_value_info",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will try correct wrong value info") 

   #fuse match ops to LayerNorm 
   parser.add_argument("--fuse_layernorm",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will fuse match ops to LayerNorm")                      

   #convert matmul to gemm()
   parser.add_argument("--matmul_to_gemm",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=1,
                        help="If set 1, the tool will convert Matmul to Gemm(when A shape[0] < 32 and B is Constant)")   

   #reset model value_info(some model(batch=-1) may have wrong value info for middle node)) 
   parser.add_argument("--reset_batch",
                        type=str, 
                        required=False,
                        nargs='*',
                        default='', #should be 'input_batch,output_batch'
                        help="If set 1, the tool will try reset model batch_size") 

   #convert Add+Clip+Div to HardSigmoid
   parser.add_argument("--fuse_hard_sigmoid",
                        type=int, 
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will merge Add+Cilp+Div to HardSigmoid") 

   #fuse Gelu
   parser.add_argument("--fuse_gelu",
                        type=int, 
                        required=False,
                        default=1,
                        choices=[0, 1],
                        help="If set 1, the tool will fuse Gelu") 

   #Disable all optimization
   parser.add_argument("--disable_all_optimizer",
                        type=int, 
                        required=False,
                        default=0,
                        choices=[0, 1],
                        help="If set 1, the tool will force all optimization value to 0") 

   #for mha optimization
   parser.add_argument("--mha_optimization",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will do some optimization for mha structure")     
   
   #for caffe pooling
   parser.add_argument("--ceil_floor_reverse",
                        type=int, 
                        required=False,
                        choices=[0, 1],
                        default=0,
                        help="If set 1, the tool will set ceil=1 in caffe pooling")     

   #fp32-->u8(for input type)
   parser.add_argument("--fp32_to_u8",
                        type=int, 
                        required=False,
                        default=0,
                        help="If set 1, the tool will change input type from float to uint8")     
   
   #show version
   parser.add_argument('--version', '-v',
                        action='store_true',
                        default=False,
                        help='Show current version')
       
   args = parser.parse_args()

   return args

def get_caffe_files(model_path):
   items = os.listdir(model_path)
   prototxt_cnt = 0
   caffemodel_cnt = 0
   prototxt_file = ''
   caffemodel_file = ''

   for f in items:
      if f.endswith(".prototxt"):
         prototxt_cnt = prototxt_cnt + 1
         prototxt_file = f
      elif f.endswith(".caffemodel"):
         caffemodel_cnt = caffemodel_cnt + 1
         caffemodel_file = f

   if prototxt_cnt == 1 and caffemodel_cnt == 1:
      if model_path.endswith("/"):
         prototxt_file = model_path + prototxt_file
         caffemodel_file = model_path + caffemodel_file
      else:
         prototxt_file = model_path + '/' + prototxt_file
         caffemodel_file = model_path + '/' + caffemodel_file
      logger.info('got prototxt_file:{}, caffemodel_file:{}'.format(prototxt_file, caffemodel_file))
   elif prototxt_cnt > 1 or caffemodel_cnt > 1:
      prototxt_file = ''
      caffemodel_file = ''
      logger.error('ERROR: prototxt_cnt > 1 or caffemodel_cnt > 1')
   elif prototxt_cnt == 0 or caffemodel_cnt == 0:
      prototxt_file = ''
      caffemodel_file = ''
      logger.error('ERROR: No .prototxt file or no .caffemodel file')       
   
   return prototxt_file, caffemodel_file  

def convert_caffe2onnx(model_path, output, op_set, ceil_floor_reverse):
      logger.info('Begin converting caffe to onnx...')
      prototxt_file, caffemodel_file = get_caffe_files(model_path)

      if prototxt_file == '' or caffemodel_file == '':
         sys.exit(exit_code_no_caffe_cfg_file)

      onnxmodel_path = output

      graph, params = loadcaffemodel(prototxt_file, caffemodel_file)
      c2o = Caffe2Onnx(graph, params, onnxmodel_path, op_set, ceil_floor_reverse)
      onnxmodel = c2o.createOnnxModel(op_set) #qiuzy debug

      saveonnxmodel(onnxmodel, onnxmodel_path)

def convert_sm2onnx(model_path, output, op_set):
      logger.info('Begin converting tf-savemodel to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_sm2onnx)

      version_check.check('tensorflow', tensorflow.__version__)

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --saved-model ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --saved-model ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      
      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw
      logger.info('convert_tfsm2onnx: {}'.format(cmd))
      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_sm2onnx failed')
         sys.exit(exit_code_sm2onnx)

def convert_h52onnx(model_path, output, op_set):
      logger.info('Begin converting tf-savemodel to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_h52onnx)

      version_check.check('tensorflow', tensorflow.__version__)   

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --keras ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --keras ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      
      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw
      logger.info('convert_tfh52onnx: {}'.format(cmd))
      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_h52onnx failed')
         sys.exit(exit_code_h52onnx)     

def convert_ckpt2onnx(model_path, output, op_set, inputs, outputs):
      logger.info('Begin converting tf-ckpt to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_ckpt2onnx)

      version_check.check('tensorflow', tensorflow.__version__)

      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --checkpoint ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
               + ' --inputs '  + inputs + ' --outputs ' + outputs
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --checkpoint ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
               + ' --inputs '  + inputs + ' --outputs ' + outputs
      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw

      logger.info('convert_ckpt2onnx: {}'.format(cmd))

      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_ckpt2onnx failed')
         sys.exit(exit_code_ckpt2onnx)   

def convert_graph2onnx(model_path, output, op_set, inputs, outputs):
      logger.info('Begin converting tf-graph to onnx...')

      try:
         import tensorflow
      except Exception as e:
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         print(e)
         print('Please install tensorflow(pip install tensorflow==2.4.0)')
         print('If numpy version > 1.19.5, tensorflow version should be 2.7.4')
         print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
         sys.exit(exit_code_pb2onnx)

      version_check.check('tensorflow', tensorflow.__version__)
      
      if using_wheel == False:
         cmd = 'python -m tf2onnx.convert --graphdef ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs
      else:
         cmd = 'python -m maca_converter.tf2onnx.convert --graphdef ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs

      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw

      logger.info('convert_graph2onnx: {}'.format(cmd))
      
      r = os.system(cmd)
      if r != 0:
         logger.error('ERROR: convert_graph2onnx failed')
         sys.exit(exit_code_pb2onnx)         

def get_darknet_files(model_path):
   items = os.listdir(model_path)
   cfg_cnt = 0
   weights_cnt = 0
   cfg_file = ''
   weights_file = ''

   for f in items:
      if f.endswith(".cfg"):
         cfg_cnt = cfg_cnt + 1
         cfg_file = f
      elif f.endswith(".weights"):
         weights_cnt = weights_cnt + 1
         weights_file = f

   if cfg_cnt == 1 and weights_cnt == 1:
      if model_path.endswith("/"):
         cfg_file = model_path + cfg_file
         weights_file = model_path + weights_file
      else:
         cfg_file = model_path + '/' + cfg_file
         weights_file = model_path + '/' + weights_file

      logger.info('got cfg_file:{}, weights_file:{}'.format(cfg_file, weights_file))
   elif cfg_cnt > 1 or weights_cnt > 1:
      cfg_file = ''
      weights_file = ''
      logger.error('ERROR: cfg_cnt > 1 or weights_cnt > 1')
   elif cfg_cnt == 0 or weights_cnt == 0:
      cfg_file = ''
      weights_file = ''
      logger.error('ERROR: No .cfg file or no .weights file')   
     
   return cfg_file, weights_file  

#tf.compat.v1.disable_v2_behavior()

def ckpt2h5(trained_checkpoint_prefix):
   #trained_checkpoint_prefix = sys.argv[1] 
   export_dir = './savemodel' 
   graph = tf.Graph()
   config=tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
   with tf.compat.v1.Session(graph=graph, config=config) as sess:
      # Restore from checkpoint
      loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
      loader.restore(sess, trained_checkpoint_prefix)

   # Export checkpoint to SavedModel
   builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
   builder.add_meta_graph_and_variables(sess, [tf.saved_model.TRAINING, tf.saved_model.SERVING],strip_default_attrs=False)
   builder.save()

def undo_darknet_concat(output):
   model = onnx.load(output)
   for node in model.graph.node:
      if node.name == 'outputs' and node.op_type == 'Concat':
         del model.graph.output[:]
         
         for i in node.input:
            model.graph.output.extend([onnx.ValueInfoProto(name=i)])

         logger.info('undo concat, new outputs: {}'.format(model.graph.output))
         model.graph.node.remove(node)
         onnx.save(model, output)
         break

def convert_dn2onnx(model_path, output, op_set):
   global support_mish
   global concat_result

   logger.info('Begin converting darknet to onnx...... support_mish: {}'.format(support_mish))
   cfg_file, weights_file = get_darknet_files(model_path)
   if cfg_file == '' or weights_file == '':
      sys.exit(exit_code_no_darknet_cfg_or_weights)

   if using_wheel == False:
      cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
   else:
      cmd = 'python -m maca_converter.darknet2onnx --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --output_file ' + output + ' --support_mish ' + str(support_mish)

   if '-tiny' in cfg_file or '-tiny' in weights_file:
      if using_wheel == False:
         cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
      else:
         cmd = 'python -m maca_converter.darknet2onnx --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output + ' --support_mish ' + str(support_mish)   
   elif 'yolov3' in cfg_file or 'yolov3' in weights_file:
      if using_wheel == False:
         cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output
      else:
         cmd = 'python -m maca_converter.darknet2onnx --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output
   logger.info('convert_dn2onnx: {}'.format(cmd))

   r = os.system(cmd)
   if r != 0:
      logger.error('ERROR: convert_dn2onnx failed')
      sys.exit(exit_code_convert_darknet2onnx)

   if concat_result == 0:
      undo_darknet_concat(output)

def convert_mish(model_path, output, op_set):
   global support_mish
   logger.info('Begin converting mish')

   if using_wheel == False:
      cmd = 'python ./mish_convert.py --onnx_file ' + model_path + ' --output_file ' + output
   else:
      cmd = 'python -m maca_converter.mish_convert --onnx_file ' + model_path + ' --output_file ' + output
   
   logger.info('convert_mish: {}'.format(cmd))
   r = os.system(cmd)
   if r != 0:
      logger.error('ERROR: convert_mish failed')
      sys.exit(exit_code_fuse_mish)      

def convert(model_path, model_type, output, op_set, input_shape_list, inputs, outputs, 
               model_def_file,
               model_class_name,
               model_input_type,
               model_weights_file,
               output_num,
               keep_batch,
               params_file,
               ceil_floor_reverse):

   if model_type == 'caffe':
      convert_caffe2onnx(model_path, output, op_set, ceil_floor_reverse)

   if model_type == 'tf-sm':
      convert_sm2onnx(model_path, output, op_set)   

   if model_type == 'tf-h5':
      convert_h52onnx(model_path, output, op_set) 

   if model_type == 'tf-ckpt':
      convert_ckpt2onnx(model_path, output, op_set, inputs, outputs)

   if model_type == 'tf-graph':
      convert_graph2onnx(model_path, output, op_set, inputs, outputs)               

   if model_type == 'darknet':
      convert_dn2onnx(model_path, output, op_set)    

   if model_type == 'pytorch':
      #convert_pt2onnx(model_path, output, op_set, input_shape)
      from pt2onnx import convert_pt2onnx
      convert_pt2onnx(model_path, output, op_set, input_shape_list,
                           model_def_file, model_class_name, model_weights_file, output_num, model_input_type, keep_batch, params_file)

   if model_type == 'paddle':
      from pd2onnx import convert_pd2onnx
      convert_pd2onnx(model_path, output, op_set, input_shape_list, model_def_file, model_class_name, model_input_type, model_weights_file)              

def optimization_op(model):
   #model = onnx.load(onnxfile)

   delete_node_id = 0
   delete = False
   #export_onnx = onnxfile

   for node_id, node in enumerate(model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #         ", op:", node.op_type, ', len(input):', len(node.input))

      if node.op_type in optimization_op_list and len(node.input) == 1:
         delete_node_id = node_id
         delete = True
         break

   if delete == True:     
      log.debug('delete: {}'.format(delete_node_id))
      delete_node = model.graph.node[delete_node_id]
      log.debug('delete node op: {}'.format(delete_node.op_type))
      next_node = model.graph.node[delete_node_id+1]

      for i, n_ in enumerate(next_node.input):
         #print('next input:', n_)
         if n_ == delete_node.output[0]:
               log.debug('got it: {]}'.format(n_))
               next_node.input[i] = delete_node.input[0]

      model.graph.node.remove(delete_node)
      export_onnx = onnxfile

      try:
         onnx.checker.check_model(model)
      except onnx.checker.ValidationError as e:
         logger.warning('The model cannot be saved for: {}'.format(e))
         if 'No Op registered for Mish' in str(e):
               logger.warning('ignore mish warning, continue saving~')
         else:
               logger.error('ERROR: check model failed')
               sys.exit(exit_code_check_optimization_op)    
      else:
         logger.info('---Begin saving model...')

      ###################
      #onnx.checker.check_model(model)
      #onnx.save(model, export_onnx)

   return delete

def correct_output_shape(model):
   for output in model.graph.output:
      if len(output.type.tensor_type.shape.dim) > 0:
         output_shape = output.type.tensor_type.shape.dim
         output_shape = [x.dim_value for x in output_shape]

         dynamic_output_shape_ = any(d==-1 or d==0 for d in output_shape)
         if dynamic_output_shape_ == True:
            logger.info('The model output is dynamic, output: {} {}'.format(output.name, output_shape))
            output.type.tensor_type.shape.dim[0].dim_value = 1

   for output in model.graph.output:
      if len(output.type.tensor_type.shape.dim) > 0:
         output_shape = output.type.tensor_type.shape.dim
         output_shape = [x.dim_value for x in output_shape]
         logger.debug('The model output is dynamic, output_shape: {}'.format(output_shape))

def reset_model_value_info(model):
   model_bak = copy.deepcopy(model)

   del model_bak.graph.value_info[:]

   try:
      new_model = onnx.shape_inference.infer_shapes(model_bak)
   except BaseException as e:
      logger.warning('reset_model_value_info, the model cannot be inferenced for: {}'.format(e))
      return model    
   else:
      new_model = onnx.shape_inference.infer_shapes(model_bak)
      new_model = onnx.shape_inference.infer_shapes(new_model)
      return new_model

def reset_batch_size(model, input_batch, output_batch):
   for input_ in model.graph.input:
      if len(input_.type.tensor_type.shape.dim) > 0:
         dim_proto = input_.type.tensor_type.shape.dim[0]
         dim_proto.dim_value = input_batch

   for output_ in model.graph.output:
      if len(output_.type.tensor_type.shape.dim) > 0:
         dim_proto = output_.type.tensor_type.shape.dim[0]
         dim_proto.dim_value = output_batch

   del model.graph.value_info[:]

   try:
      new_model = onnx.shape_inference.infer_shapes(model)
   except BaseException as e:
      logger.warning('reset_batch_size, the model cannot be inferenced for: {}'.format(e))
      new_model = model    
   else:
      new_model = onnx.shape_inference.infer_shapes(new_model)

   return new_model

def model_simplify(onnx_model, simplify_model, simplify_hw):
   #onnx_model = onnx.load(model_path)
   is_dynamic_input_shape = False

   init_list = []
   input_shapes_ = {}

   for init in onnx_model.graph.initializer:
      init_list.append(init.name)

   h = -1
   w = -1   

   for input_ in onnx_model.graph.input:
      #print('graph_input_name:', input_.name)
      if input_.name not in init_list:
         if len(input_.type.tensor_type.shape.dim) > 0:
            input_shape = input_.type.tensor_type.shape.dim
            input_shape = [x.dim_value for x in input_shape]

            if len(input_shape) < 2:
               continue

            h = input_shape[-2]
            w = input_shape[-1]

            dynamic_input_shape_ = any(d==-1 or d==0 for d in input_shape)
            if dynamic_input_shape_ == True:
               is_dynamic_input_shape = True
               logger.info('The model input is dynamic, input: {} {}'.format(input_.name, input_shape))
               input_shape[0] = 1
               if simplify_hw != '':
                  hw_list = simplify_hw.split(',')
                  input_shape[-1] = int(hw_list[1])
                  input_shape[-2] = int(hw_list[0])
                  logger.debug('input_shape: {}'.format(input_shape))

            input_shapes_[input_.name] = input_shape
            logger.info('----- input_shapes_: {}'.format(input_shapes_))

   skip_constant_folding_ = False
   if h <= 0 or w <= 0:
      skip_constant_folding_ = True

   if simplify_model == 2:
      if is_dynamic_input_shape == True:
         model_simp, check = simplify(onnx_model, input_shapes=input_shapes_, skip_constant_folding=skip_constant_folding_)
         if simplify_hw == '':
            #correct_batch_for_opset_convert(model_simp)
            correct_output_shape(model_simp)
            try:
               model_simp = reset_model_value_info(model_simp)
            except Exception as e:
               logger.warning('Cannot do reset_value operation~')
      else:   
         model_simp, check = simplify(onnx_model, dynamic_input_shape=False)
   else:
      model_simp, check = simplify(onnx_model, dynamic_input_shape=is_dynamic_input_shape, skip_constant_folding=skip_constant_folding_)

   #onnx.save(model_simp, model_path)
   if model_simp.producer_version != '':
      model_simp.producer_version = model_simp.producer_version + '(simplified by macaConverter)'
   else:   
      model_simp.producer_name = model_simp.producer_name + '(simplified by macaConverter)'

   return model_simp

def modify_onnx2dynamic(onnx_model):
   for idx in range(len(onnx_model.graph.input)):
      if len(onnx_model.graph.input[idx].type.tensor_type.shape.dim) > 0:
         dim_proto_input = onnx_model.graph.input[idx].type.tensor_type.shape.dim[0]
         # dim_proto_input.dim_param = 'bs'
         dim_proto_input.dim_value = -1

   for idx in range(len(onnx_model.graph.value_info)):
      if len(onnx_model.graph.value_info[idx].type.tensor_type.shape.dim) > 0:
         logger.debug('value info name: {}'.format(onnx_model.graph.value_info[idx].name))
         dim_proto_input = onnx_model.graph.value_info[idx].type.tensor_type.shape.dim[0]
         # dim_proto_input.dim_param = 'bs'
         dim_proto_input.dim_value = -1   
  
   for idx in range(len(onnx_model.graph.output)):
      if len(onnx_model.graph.output[idx].type.tensor_type.shape.dim):
         dim_proto_output = onnx_model.graph.output[idx].type.tensor_type.shape.dim[0]
         # dim_proto_output.dim_param = 'bs'
         dim_proto_output.dim_value = -1

   ### for Reshape
   reshape_param = []
   for node_id, node in enumerate(onnx_model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #         ", op:", node.op_type, ', len(input):', len(node.input))
      if node.op_type == 'Reshape':
         logger.debug('Reshape, input: {}'.format(node.input))
         if node.input[1] not in reshape_param:
            reshape_param.append(node.input[1])

   for n in reshape_param:
      for init in onnx_model.graph.initializer:
         logger.info('loop init.name: {}'.format(init.name))
         if n == init.name:
            logger.info('got it in initializer: {} {}'.format(n, init.int64_data))
            #init.int64_data[0] = -1
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            if init.raw_data:
               params_list = np.fromstring(init.raw_data, dtype=np_dtype)
               logger.debug('len(params_list): {}'.format(len(params_list)))
               adjust = True
               for val in params_list:
                  if val == -1:
                     adjust = False

               if adjust == True and params_list[0] != -1:
                     params_list[0] = -1
                     init.raw_data = params_list.tostring()
            else:
               data_list = get_data_list(dtype, init)
               adjust = True
               logger.debug('len(data_list): {}'.format(len(data_list)))

               for val in data_list:
                  if val == -1:
                     adjust = False

               if adjust == True and len(data_list) > 0 and data_list[0] != -1:
                     data_list[0] = -1

############# for constant node
   for n in reshape_param:
      for node in onnx_model.graph.node:
         if node.op_type == 'Constant':
            if node.output[0] == n:
               logger.info('got constant output: {}'.format(node.output))
               attributes = node.attribute
               for attr in attributes:
                     if attr.name == 'value':
                        v = values.get_tensor_value(attr.t)
                        #print('got type v:', type(v))
                        adjust = True
                        for val in v:
                           if val == -1:
                              adjust = False
                              break

                        if adjust == True:
                           v[0] = -1 
                           vv = [v_ for v_ in v]
                           #print('-----new vv:', vv, type(vv))
                           if isinstance(v, np.ndarray) == True:
                              values.set_tensor_value(attr.t, v)
                           else:     
                              values.set_tensor_value(attr.t, vv)   
########################

   #onnx_model = onnx.shape_inference.infer_shapes(onnx_model)                  
   try:
      onnx.checker.check_model(onnx_model)
   except onnx.checker.ValidationError as e:
      print('*** The model cannot be modified for: %s' % e)
      if 'No Op registered for Mish' in str(e):
         logger.warning('ignore mish warning, continue saving~')
      else:
         logger.error('ERROR: check model failed in modify_onnx2dynamic')
         #sys.exit(exit_code_check_modify_onnx2dynamic)    
   else:
      logger.info('*** The model is modified!')

   return onnx_model
    
def convert_gap_2_ap(model):
   #model = onnx.load(onnxfile)

   node_list = []

   has_global_average_pool = False
   need_convert = False

   for node_id, node in enumerate(model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #         ", op:", node.op_type)

      if node.op_type == 'GlobalAveragePool':
         has_global_average_pool = True 
         dict={'id':node_id, 'input':node.input, 'output':node.output, 'op':node.op_type}   
         node_list.append(dict)

   if has_global_average_pool == True:
      node_list2 = []

      for v in model.graph.value_info:
         input_shape = v.type.tensor_type.shape.dim
         input_shape = [x.dim_value for x in input_shape]

         if len(input_shape) >= 4:
            dict2 = {'name':v.name, 'shape':input_shape}
            node_list2.append(dict2)

         #print("+++++++++++ name:", v.name, input_shape)

      for d in node_list:
         if d['op'] == 'GlobalAveragePool':
               logger.debug('op id: {}, op input: {}'.format(d['id'], d['input']))
               for v in node_list2:
                  #print('v.name:', v['name'])
                  if d['input'][0] == v['name']:
                     logger.debug('got GlobalAveragePool, shape: {} {}'.format(v['shape'], v['name']))
                     if v['shape'][2] <= 15 and v['shape'][3] <= 15:
                           need_convert = True
                           logger.info('GlobalAveragePool===>AveragePool......') 
                           old_node = model.graph.node[d['id']] 
                           model.graph.node.remove(old_node)
                           new_node = onnx.helper.make_node(
                                                   name = '',
                                                   op_type="AveragePool",
                                                   inputs=d['input'],
                                                   outputs=d['output'],
                                                   kernel_shape=[v['shape'][2], v['shape'][3]], 
                                                   )

                           model.graph.node.insert(d['id'], new_node)

   if need_convert == True:
         #onnx.checker.check_model(model)
         try:
            onnx.checker.check_model(model)
         except onnx.checker.ValidationError as e:
            print('The model cannot be saved for: %s' % e)
            if 'No Op registered for Mish' in str(e):
                  logger.warning('ignore mish warning, continue saving~')
            else:
                  logger.error('ERROR: check model failed in convert_gap_2_ap')
                  sys.exit(exit_code_check_convert_gap_2_ap)    
         else:
            logger.info('+++ Begin saving model...')

         #onnx.save(model, onnxfile)

   return need_convert      

def post_process(new_model, inference_success, gap_to_ap):
   start_time = time.time()

   debug_print = False

   delete = optimization_op(new_model)
   while delete == True:
      debug_print = True
      delete = optimization_op(new_model)

   end_time1 = time.time()

   if debug_print == True:
      logger.info('optimization_op cost {} seconds'.format(end_time1 - start_time))

   debug_print = False

   if gap_to_ap == 1:
      if inference_success == True:
         debug_print = convert_gap_2_ap(new_model)
      else:
         logger.warning('Cannot do inference, so skip global_average_pool-->average_pool')    

   end_time2 = time.time()

   if debug_print == True:
      logger.info('convert_gap_2_ap cost {} seconds'.format(end_time2 - end_time1))  

def my_extract_model(
        input_path,  # type: Text
        output_path,  # type: Text
        input_names,  # type: List[Text]
        output_names  # type: List[Text]
):  # type: (...) -> None
   """Extracts sub-model from an ONNX model.

   The sub-model is defined by the names of the input and output tensors *exactly*.

   Note: For control-flow operators, e.g. If and Loop, the _boundary of sub-model_,
   which is defined by the input and output tensors, should not _cut through_ the
   subgraph that is connected to the _main graph_ as attributes of these operators.

   Arguments:
   input_path (string): The path to original ONNX model.
   output_path (string): The path to save the extracted ONNX model.
   input_names (list of string): The names of the input tensors that to be extracted.
   output_names (list of string): The names of the output tensors that to be extracted.
   """
   if not os.path.exists(input_path):
      raise ValueError("Invalid input model path: %s" % input_path)
   if not output_path:
      raise ValueError("Output model path shall not be empty!")
   if not output_names:
      raise ValueError("Output tensor names shall not be empty!")

   #onnx.checker.check_model(input_path)
   try:
      onnx.checker.check_model(input_path)
   except onnx.checker.ValidationError as e:
      print('Extract warning:: %s' % e)
   else:
      logger.info('~~~~ Begin extracting model...')

   model = onnx.load(input_path)

   e = onnx.utils.Extractor(model)
   extracted = e.extract_model(input_names, output_names)

   onnx.save(extracted, output_path)
   #onnx.checker.check_model(output_path)
   try:
      onnx.checker.check_model(output_path)
   except onnx.checker.ValidationError as e:
      print('Extracted warning: %s' % e)
   else:
      logger.info('^^^^ Finish extracting model...')

   return True   

def extract_sub_graph(input_path, output_path, input_names, output_names):
   logger.info('input_names: {}, output_names: {}'.format(input_names, output_names))
   input_list = input_names.split(',')
   output_list = output_names.split(',')
   #onnx.utils.extract_model(input_path, output_path, input_list, output_list)
   return my_extract_model(input_path, output_path, input_list, output_list)

   '''
   try:
      onnx.utils.extract_model(input_path, output_path, input_list, output_list)
   except BaseException as e:
      print('The model cannot be extracted for: %s' % e)
      return False   
   else:
      print('Inference success---')
      return True
   '''

def process(args):
   global support_mish
   global inputs_as_nchw
   global concat_result

   model_path = args.model_path
   model_type = args.model_type
   output = args.output
   concat_result = args.concat_result
   op_set = args.op_set
   input_shape = args.input_shape
   inputs = args.inputs
   outputs = args.outputs
   simplify_model = args.simplify
   extract_sub = args.extract_sub
   dynamic_batch = args.dynamic_batch
   fp32_to_fp16 = args.fp32_to_fp16
   support_mish = args.support_mish
   preproc_yaml = args.preproc_yaml
   postproc_yaml = args.postproc_yaml
   model_def_file = args.model_def_file
   model_class_name = args.model_class_name
   model_input_type = args.model_input_type
   model_weights_file = args.model_weights_file
   inputs_as_nchw = args.inputs_as_nchw
   gap_to_ap = args.gap_to_ap
   fuse_pad_pool = args.fuse_pad_pool
   support_swish = args.support_swish
   bn_to_conv = args.bn_to_conv
   output_num = args.output_num
   keep_batch = args.keep_batch
   params_file = args.params_file
   simplify_hw = args.simplify_hw
   force_simplify = args.force_simplify
   gemm_optimization = args.gemm_optimization
   expand_to_resize = args.expand_to_resize
   reset_value_info = args.reset_value_info
   fuse_layernorm = args.fuse_layernorm
   matmul_to_gemm = args.matmul_to_gemm
   reset_batch = args.reset_batch
   fuse_hard_sigmoid = args.fuse_hard_sigmoid
   fuse_gelu = args.fuse_gelu
   disable_all_optimizer = args.disable_all_optimizer
   mha_optimization = args.mha_optimization
   fp32_to_u8 = args.fp32_to_u8
   ceil_floor_reverse = args.ceil_floor_reverse

   if args.version:
      print('maca_converter version:', MXC_CONFIG.VERSION)
      print('last modified:', MXC_CONFIG.LAST_MODIFIED)
      exit(0)

   if model_path == None or model_type == None or output == None:
      print('WARNING: model_path/model_type/output COULD NOT be null')
      exit(-1)

   if disable_all_optimizer == 1:
      print('------- disable all optimazation')
      support_mish = 0
      gap_to_ap = 0
      fuse_pad_pool = 0
      support_swish = 0 
      bn_to_conv = 0 
      gemm_optimization = 0 
      expand_to_resize = 0
      fuse_layernorm = 0 
      matmul_to_gemm = 0 
      fuse_hard_sigmoid = 0 
      fuse_gelu = 0
      ceil_floor_reverse = 0

   logger.info('model_path:{}, model_type:{}, output:{}'.format(model_path, model_type, output))

   if model_type == 'tf-ckpt' or model_type == 'tf-graph' :
      logger.debug('checkpoint: {} {}'.format(inputs, outputs))

   logger.info('---input_shape: {}'.format(input_shape))

   input_shape_list = input_shape.split('/') 
   logger.info('---input_shape_list: {}'.format(input_shape_list)) 

   #input_shape = input_shape_list[0]  

   dynamic_paddle = False
   if model_type == 'paddle':
         from pd2onnx import is_dynamic_paddle
         dynamic_paddle = is_dynamic_paddle(input_shape_list, model_def_file, model_class_name, model_weights_file)
         #if dynamic_paddle == True and model_input_type == '':
         #   model_input_type = 'float32' 

   can_ignore_model_path = False
   if model_type == 'pytorch':
      #if '.' in model_class_name:
      if model_weights_file != '':
         can_ignore_model_path = True

   if dynamic_paddle == False and can_ignore_model_path == False and not os.path.exists(model_path):
      logger.error('ERROR: {} is not exist'.format(model_path))
      sys.exit(exit_code_model_not_exist)

   if model_type not in valid_model_type:
      logger.error('Valid mode type is {}'.format(valid_model_type))
      logger.error('ERROR: {} is not valid mode type'.format(model_type))
      sys.exit(exit_code_invalid_model_type)

   op_set_default = 11

   if op_set != None and op_set < op_set_default:
      op_set_default = op_set

   if model_type == 'pytorch' and args.input_shape == '':
      logger.warning('WARNNIG: when converting pytorch model, you must tell the input shape(ex: --input_shape [1, 3, 32, 32])')
      logger.warning('WARNNIG: also, you should provide model definition file')
      sys.exit(exit_code_pytorch_no_input_shape)

   if (model_type == 'tf-ckpt' or model_type == 'tf-graph') and (args.inputs == '' or args.outputs == ''):
      logger.warning('WARNNIG: When converting checkpoint/graph, you must tell the inputs(ex: --inputs input0:0,input1:0) and outputs(ex: --outputs output0:0)')
      sys.exit(exit_code_tensorflow_no_inputs_or_outputs)

   if extract_sub == 1:
      if args.inputs == '' or args.outputs == '':
         logger.warning('WARNNIG: When extract sub graph, you must tell the inputs(ex: --inputs input0:0,input1:0) and outputs(ex: --outputs output0:0)')
         sys.exit(exit_code_extract_sub_no_inputs_or_outputs)

      if model_type != 'onnx':
         logger.warning('WARNNING: only onnx model supports extracting...')
         sys.exit(exit_code_model_type_not_onnx)

      r = extract_sub_graph(model_path, output, inputs, outputs)
      if r == True:
         logger.critical('Convert Success!')
         
      sys.exit(exit_code_normal)                

   logger.info('begin convert..')

   begin_time = time.time()

   if model_type != 'onnx':
      convert(model_path, 
               model_type, 
               output, 
               op_set_default, 
               input_shape_list, 
               inputs, 
               outputs,
               model_def_file,
               model_class_name,
               model_input_type,
               model_weights_file,
               output_num,
               keep_batch,
               params_file,
               ceil_floor_reverse)

   end_time1 = time.time()
  
   logger.info('finish convert, it cost {} seconds'.format(end_time1 - begin_time))

   if model_type != 'onnx':
      model = onnx.load(output)
   else:
      model = onnx.load(model_path)

   if op_set != None :
      if model_type == 'onnx':
         logger.info('ONNX, add_value_info_for_constants...')
         correct_batch_for_opset_convert(model)
         operation.add_value_info_for_constants(model)
         model = version_converter.convert_version(model, op_set)
      elif op_set != op_set_default:
         correct_batch_for_opset_convert(model)
         operation.add_value_info_for_constants(model)
         model = version_converter.convert_version(model, op_set)

      operation.eliminate_unused_input_initializer(model)   

   inference_success = False
   new_model = model

   #new_model = onnx.shape_inference.infer_shapes(model)
   try:
      new_model = onnx.shape_inference.infer_shapes(model)
   except BaseException as e:
      print('The model cannot be inferenced for: %s' % e)
      new_model = model    
   else:
      logger.info('Inference success---')
      inference_success = True

   #onnx.checker.check_model(new_model)
   try:
      onnx.checker.check_model(new_model)
   except BaseException as e: #onnx.checker.ValidationError as e:
      logger.warning('ignore warning(check_model), continue saving~')  
   else:
      logger.info('### Begin saving model...')

   #onnx.save(new_model, output)

   if dynamic_batch == 1:
      logger.info('modify model to dynamic batch...')
      new_model = modify_onnx2dynamic(new_model)

   end_time2 = time.time()

   logger.info('generate inference shape model, it cost {} seconds'.format(end_time2 - end_time1))

   #post_process(new_model, inference_success, gap_to_ap)

   if simplify_model == 1 or simplify_model == 2:
      logger.info('begin doing simplify...')

      producer_name = new_model.producer_name
      producer_version = new_model.producer_version
      simplify_flag = '(simplified by macaConverter)'
      if simplify_flag not in producer_name and simplify_flag not in producer_version:
         new_model = model_simplify(new_model, simplify_model, simplify_hw)
      else:
         if force_simplify != 0:
            simplify_model = force_simplify
            new_model = model_simplify(new_model, simplify_model, simplify_hw)
         else:
            logger.info('The model has been simplified by macaConverter, ignore this operation~~')  

   post_process(new_model, inference_success, gap_to_ap)

   if mha_optimization == 1:
      new_model = mha_optimizer(new_model)

   if reset_batch != '':
      batchs = reset_batch #.split(' ')
      input_batch = int(batchs[0])
      output_batch = input_batch
      if len(batchs) >= 2:
         output_batch = int(batchs[1])

      if input_batch == 0:
         input_batch = -1

      if output_batch == 0:
         output_batch = -1

      logger.info('got batchs: {}'.format(batchs))            

      new_model = reset_batch_size(new_model, input_batch, output_batch)

   if fuse_pad_pool == 1:
      logger.info('begin doing fuse_pad_to_pool...')
      new_model = fuse.fuse_pad_to_pool(new_model)   

   '''
   if fp32_to_fp16 == 1:
      print('begin doing fp32-->fp16...')
      new_model = convert_float_to_float16(new_model, keep_io_types=True)
   '''

   #if model_type == 'onnx' and support_mish == 1:
   if support_mish == 1:   
      new_model = merge_mish(new_model)

   #if model_type == 'onnx' and support_swish == 1:
   if support_swish == 1:   
      new_model = merge_swish_and_hard_swish(new_model)   

   #if model_type == 'onnx' and gemm_optimization == 1:
   if gemm_optimization == 1:   
      new_model = gemm_convert(new_model)  

   if model_type == 'onnx' and preproc_yaml != '':
      if os.path.exists(preproc_yaml):
         new_model = preproc(new_model, preproc_yaml)
      else:
         logger.warning('pre_proc yaml file {} is not exist'.format(preproc_yaml))    

   if model_type == 'onnx' and postproc_yaml != '':
      if os.path.exists(postproc_yaml):
         new_model = postproc(new_model, postproc_yaml)
      else:
         logger.warning('post_proc yaml file {} is not exist'.format(postproc_yaml))

   if bn_to_conv == 1 and simplify_model != 0:
      new_model = bn2conv.bn2conv(new_model)

   #if model_type == 'onnx' and expand_to_resize == 1:
   if expand_to_resize == 1:   
      new_model = merge_resize(new_model)           

   if model_type == 'onnx' and reset_value_info == 1:
      new_model = reset_model_value_info(new_model) 

   #if model_type == 'onnx' and fuse_layernorm == 1:
   if fuse_layernorm == 1:   
      new_model = merge_layernorm(new_model) 

   #if model_type == 'onnx' and matmul_to_gemm == 1:
   if matmul_to_gemm == 1:
      new_model = matmul_2_gemm(new_model)

   if fuse_hard_sigmoid == 1:
      new_model = merge_hard_sigmiod(new_model)

   if fuse_gelu== 1:
      new_model = merge_gelu(new_model)

   if fp32_to_u8 == 1:
      new_model = fp32_to_uint8(new_model)

   if fp32_to_fp16 == 1:
      logger.info('begin doing fp32-->fp16...')
      new_model = convert_float_to_float16(new_model, keep_io_types=True)

   delete = operation.eliminate_redundant_reshape(new_model)
   while delete == True:
      delete = operation.eliminate_redundant_reshape(new_model)   

   operation.eliminate_unused_input_initializer(new_model)
   operation.eliminate_unused_constant_node(new_model)
   operation.remove_unused_initializer(new_model)

   onnx.save(new_model, output)                 

   end_time3 = time.time()

   logger.info('The whole progress cost {} seconds'.format(end_time3 - begin_time))

   logger.critical('Convert Success!')

def usage():
    print('python model_convert.py --model_path ./my_model --model_type caffe --output ./c2o.onnx')
    print('or') 
    print('python model_convert.py --model_path ./my_model \
            --model_type tf-h5  \
            --output ./t2o.onnx  \
            --op_set 12 \
            --datatype_convert fp32_to_fp16 \
            --q_dataset_file ./quantization.npy \
            --q_onnx_file ./quantization.onnx')    

def main(args):
   #clear log file
   with open("./convert.log", 'r+') as file:
      file.truncate(0)

   process(args)

if __name__ == "__main__":
   args = parse_args()
   main(args)
