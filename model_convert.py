# -*- coding: utf-8 -*-

import os
import onnx
from onnx import version_converter
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
import argparse
import h5py
import tensorflow as tf
import time
import fuse
from swish_convert import merge_swish_and_hard_swish

from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
from caffe2onnx.src.caffe2onnx import Caffe2Onnx
from onnxsim.onnx_simplifier import simplify
from float16 import convert_float_to_float16
from preprocess import preproc
from postprocess import postproc
from correct_batch import correct_batch_for_opset_convert, convert_ort_type_2_np, get_data_list
from pd2onnx import convert_pd2onnx, is_dynamic_paddle
from pt2onnx import convert_pt2onnx

support_mish = 0

inputs_as_nchw = ''

logging.basicConfig(level=logging.INFO, filename='./convert.log', filemode='w')

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

logger = logging.getLogger("[MacaConverter]")

import argparse

valid_model_type = ['caffe', 'pytorch', 'tf-h5', 'tf-ckpt', 'tf-sm', 'tf-graph', 'darknet', 'onnx', 'paddle']

optimization_op_list = ['Max', 'Min', 'Sum', 'Mean']

def parse_args():
   parser = argparse.ArgumentParser(description='Convert xxx model to ONNX.')

   parser.add_argument("--model_path",
                        type=str,  required=True,
                        help="input model folder")

   parser.add_argument("--model_type",
                        type=str,
                        required=True,
                        help="input model type(ex: caffe/pytorch/tf-h5)")

   parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="output .onnx(ex: output.onnx)")

   parser.add_argument("--op_set",
                        type=int, required=False,
                        help="op_set version(default: 11)")

   #for simplify
   parser.add_argument("--simplify",
                        type=int, required=False,
                        default=1,
                        help="simplify the model")                              

   #for pytorch/dynamic_paddle
   parser.add_argument("--input_shape",
                        type=str, 
                        required=False,
                        default='',
                        help="input shape")

   #######
   parser.add_argument("--inputs",
                        type=str, 
                        required=False,
                        default='',
                        help="checkpoint/graph/onnx_sub_graph input")

   parser.add_argument("--outputs",
                        type=str, 
                        required=False,
                        default='',
                        help="checkpoint/graph/onnx_sub_graph output")

   #for extract sub graph
   parser.add_argument("--extract_sub",
                        type=int, 
                        required=False,
                        default=0,
                        help="extract sub graph") 

   #for dynamic batch size
   parser.add_argument("--dynamic_batch",
                        type=int, 
                        required=False,
                        default=0,
                        help="dynamic batch size")

   #for fp32-->fp16
   parser.add_argument("--fp32_to_fp16",
                        type=int, 
                        required=False,
                        default=0,
                        help="fp32-->fp16")

   parser.add_argument("--support_mish",
                        type=int, 
                        required=False,
                        default=0,
                        help="hardware support mish") 

   #insert preproc node
   parser.add_argument("--preproc_yaml",
                        type=str, 
                        required=False,
                        default='',
                        help="preprocess yaml file")

   parser.add_argument("--postproc_yaml",
                        type=str, 
                        required=False,
                        default='',
                        help="postprocess yaml file")                     

   #for paddle dynamic model or pytorch
   parser.add_argument("--model_def_file",
                        type=str, 
                        required=False,
                        default='',
                        help="paddle/pytorch model definition file location")

   parser.add_argument("--model_weights_file",
                        type=str, 
                        required=False,
                        default='',
                        help="paddle/pytorch model weights file")                     

   parser.add_argument("--model_class_name",
                        type=str, 
                        required=False,
                        default='',
                        help="paddle/pytorch model calss name")

   parser.add_argument("--paddle_input_type",
                        type=str, 
                        required=False,
                        default='',
                        help="paddle/pytorch input type")

   #for tensorflow 
   parser.add_argument("--inputs_as_nchw",
                        type=str, 
                        required=False,
                        default='',
                        help="tensorflow nchw")

   #for gap-->ap 
   parser.add_argument("--gap_to_ap",
                        type=int, 
                        required=False,
                        default=0,
                        help="GlobalAveragePool-->AveragePool") 

   #for pad+pool fuse
   parser.add_argument("--fuse_pad_pool",
                        type=int, 
                        required=False,
                        default=0,
                        help="fuse pad+pool") 

   #for merge swish
   parser.add_argument("--support_swish",
                        type=int, 
                        required=False,
                        default=0,
                        help="Sigmoid+Mul-->swish; HardSigmoid+Mul-->HardSwish")                                                                       
                                                                                                                                                                                                                                                                                                                                     
   args = parser.parse_args()

   return args

def get_caffe_files(model_path):
   items = os.listdir(model_path)
   has_prototxt = False
   has_caffemodel = False
   prototxt_file = ''
   caffemodel_file = ''

   for f in items:
      if f.endswith(".prototxt"):
         has_prototxt = True
         prototxt_file = f
      elif f.endswith(".caffemodel"):
         has_caffemodel = True
         caffemodel_file = f

      if has_prototxt == True and has_caffemodel == True:
         if model_path.endswith("/"):
            prototxt_file = model_path + prototxt_file
            caffemodel_file = model_path + caffemodel_file
         else:
            prototxt_file = model_path + '/' + prototxt_file
            caffemodel_file = model_path + '/' + caffemodel_file

         print('got prototxt_file:{}, caffemodel_file:{}'.format(prototxt_file, caffemodel_file))

         break        
   
   return prototxt_file, caffemodel_file  

def convert_caffe2onnx(model_path, output, op_set):
      print('Begin converting caffe to onnx...')
      prototxt_file, caffemodel_file = get_caffe_files(model_path)

      if prototxt_file == '' or caffemodel_file == '':
         print('ERROR: prototxt file or caffemodel file is None')
         sys.exit()

      onnxmodel_path = output

      graph, params = loadcaffemodel(prototxt_file, caffemodel_file)
      c2o = Caffe2Onnx(graph, params, onnxmodel_path)
      onnxmodel = c2o.createOnnxModel(op_set) #qiuzy debug

      saveonnxmodel(onnxmodel, onnxmodel_path)

def convert_sm2onnx(model_path, output, op_set):
      print('Begin converting tf-savemodel to onnx...')
      cmd = 'python -m tf2onnx.convert --saved-model ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw
      print('convert_tfsm2onnx: ', cmd)
      os.system(cmd)

def convert_h52onnx(model_path, output, op_set):
      print('Begin converting tf-savemodel to onnx...')
      cmd = 'python -m tf2onnx.convert --keras ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw
      print('convert_tfh52onnx: ', cmd)
      os.system(cmd)     

def convert_ckpt2onnx(model_path, output, op_set, inputs, outputs):
      print('Begin converting tf-ckpt to onnx...')
      cmd = 'python -m tf2onnx.convert --checkpoint ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs

      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw

      print('convert_ckpt2onnx: ', cmd)

      os.system(cmd)

def convert_graph2onnx(model_path, output, op_set, inputs, outputs):
      print('Begin converting tf-graph to onnx...')
      cmd = 'python -m tf2onnx.convert --graphdef ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs

      if inputs_as_nchw != '':
         cmd += ' --inputs-as-nchw ' + inputs_as_nchw

      print('convert_graph2onnx: ', cmd)
      
      os.system(cmd)      

def get_darknet_files(model_path):
   items = os.listdir(model_path)
   has_cfg = False
   has_weights = False
   cfg_file = ''
   weights_file = ''

   for f in items:
      if f.endswith(".cfg"):
         has_cfg = True
         cfg_file = f
      elif f.endswith(".weights"):
         has_weights = True
         weights_file = f

      if has_cfg == True and has_weights == True:
         if model_path.endswith("/"):
            cfg_file = model_path + cfg_file
            weights_file = model_path + weights_file
         else:
            cfg_file = model_path + '/' + cfg_file
            weights_file = model_path + '/' + weights_file

         print('got cfg_file:{}, weights_file:{}'.format(cfg_file, weights_file))

         break        
   
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

def convert_dn2onnx(model_path, output, op_set):
   global support_mish
   print('Begin converting darknet to onnx...... support_mish：', support_mish)
   cfg_file, weights_file = get_darknet_files(model_path)

   cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
   
   if '-tiny' in cfg_file or '-tiny' in weights_file:
      cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output + ' --support_mish ' + str(support_mish)
   elif 'yolov3' in cfg_file or 'yolov3' in weights_file:
      cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --strides 32 16 8 ' + ' --neck FPN ' + ' --output_file ' + output
   
   print('convert_dn2onnx: ', cmd)

   os.system(cmd)

def convert_mish(model_path, output, op_set):
   global support_mish
   print('Begin converting mish')

   cmd = 'python ./mish_convert.py --onnx_file ' + model_path + ' --output_file ' + output
   print('convert_mish: ', cmd)
   os.system(cmd)     

def convert(model_path, model_type, output, op_set, input_shape, inputs, outputs, 
               model_def_file,
               model_class_name,
               paddle_input_type,
               model_weights_file):

   if model_type == 'caffe':
      convert_caffe2onnx(model_path, output, op_set)

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
      convert_pt2onnx(model_path, output, op_set, input_shape,
                           model_def_file, model_class_name, model_weights_file)

   if model_type == 'paddle':
      convert_pd2onnx(model_path, output, op_set, input_shape, model_def_file, model_class_name, paddle_input_type, model_weights_file)              

def optimization_op(onnxfile):
   model = onnx.load(onnxfile)

   delete_node_id = 0
   delete = False
   export_onnx = onnxfile

   for node_id, node in enumerate(model.graph.node):
      print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
               ", op:", node.op_type, ', len(input):', len(node.input))

      if node.op_type in optimization_op_list and len(node.input) == 1:
         delete_node_id = node_id
         delete = True
         break

   if delete == True:     
      print('delete: ', delete_node_id)
      delete_node = model.graph.node[delete_node_id]
      print('delete node op:', delete_node.op_type)
      next_node = model.graph.node[delete_node_id+1]

      for i, n_ in enumerate(next_node.input):
         #print('next input:', n_)
         if n_ == delete_node.output[0]:
               print('got it:', n_)
               next_node.input[i] = delete_node.input[0]

      model.graph.node.remove(delete_node)
      export_onnx = onnxfile

      try:
         onnx.checker.check_model(model)
      except onnx.checker.ValidationError as e:
         print('---The model cannot be saved for: %s' % e)
         if 'No Op registered for Mish' in str(e):
               print('ignore mish warning, continue saving~')
         else:
               sys.exit()    
      else:
         print('---Begin saving model...')

      ###################
      #onnx.checker.check_model(model)
      onnx.save(model, export_onnx)

   return delete, export_onnx

def model_simplify(model_path):
   onnx_model = onnx.load(model_path)
   dynamic_input_shape_ = False

   init_list = []

   for init in onnx_model.graph.initializer:
      init_list.append(init.name)

   for idx in range(len(onnx_model.graph.input)):
      print('graph_input_name:', onnx_model.graph.input[idx].name)
      if onnx_model.graph.input[idx].name not in init_list:
         if len(onnx_model.graph.input[idx].type.tensor_type.shape.dim) > 0:
            dim_proto_input = onnx_model.graph.input[idx].type.tensor_type.shape.dim[0]
            if dim_proto_input.dim_value == -1 or dim_proto_input.dim_value == 0:
               print('The model input is dynamic~~~~~~')
               dynamic_input_shape_ = True
               break

   model_simp, check = simplify(onnx_model, dynamic_input_shape=dynamic_input_shape_)
   onnx.save(model_simp, model_path)

   return model_simp

def modify_onnx2dynamic(onnx_model, model_path):
   for idx in range(len(onnx_model.graph.input)):
      dim_proto_input = onnx_model.graph.input[idx].type.tensor_type.shape.dim[0]
      # dim_proto_input.dim_param = 'bs'
      dim_proto_input.dim_value = -1

   for idx in range(len(onnx_model.graph.value_info)):
      dim_proto_input = onnx_model.graph.value_info[idx].type.tensor_type.shape.dim[0]
      # dim_proto_input.dim_param = 'bs'
      dim_proto_input.dim_value = -1   

   for idx in range(len(onnx_model.graph.output)):
      dim_proto_output = onnx_model.graph.output[idx].type.tensor_type.shape.dim[0]
      # dim_proto_output.dim_param = 'bs'
      dim_proto_output.dim_value = -1

   ### for Reshape
   reshape_param = []
   for node_id, node in enumerate(onnx_model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #         ", op:", node.op_type, ', len(input):', len(node.input))
      if node.op_type == 'Reshape':
         print('Reshape, input:', node.input)
         if node.input[1] not in reshape_param:
            reshape_param.append(node.input[1])

   for n in reshape_param:
      for init in onnx_model.graph.initializer:
         if n == init.name:
            print('got it in initializer:', n, init.int64_data)
            #init.int64_data[0] = -1
            dtype = init.data_type
            np_dtype = convert_ort_type_2_np(dtype)
            if init.raw_data:
               params_list = np.fromstring(init.raw_data, dtype=np_dtype)
               if params_list[0] != -1:
                     params_list[0] = -1
                     init.raw_data = params_list.tostring()
            else:
               data_list = get_data_list(dtype, init)
               if len(data_list) > 0 and data_list[0] != -1:
                     data_list[0] = -1
   try:
      onnx.checker.check_model(onnx_model)
   except onnx.checker.ValidationError as e:
      print('*** The model cannot be modified for: %s' % e)
      if 'No Op registered for Mish' in str(e):
         print('ignore mish warning, continue saving~')
      else:
         sys.exit()    
   else:
      print('*** The model is modified!')

   onnx.save(onnx_model, model_path)
    
def convert_gap_2_ap(onnxfile):
   model = onnx.load(onnxfile)

   node_list = []

   has_global_average_pool = False
   need_convert = False

   for node_id, node in enumerate(model.graph.node):
      print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
               ", op:", node.op_type)

      if node.op_type == 'GlobalAveragePool':
         has_global_average_pool = True 
         dict={'id':node_id, 'input':node.input, 'output':node.output, 'op':node.op_type}   
         node_list.append(dict)

   if has_global_average_pool == True:
      node_list2 = []

      for v in model.graph.value_info:
         input_shape = v.type.tensor_type.shape.dim
         input_shape = [x.dim_value for x in input_shape]

         dict2 = {'name':v.name, 'shape':input_shape}
         node_list2.append(dict2)

         print("+++++++++++ name:", v.name, input_shape)

      for d in node_list:
         if d['op'] == 'GlobalAveragePool':
               print('op id:', d['id'], ', op input:', d['input'])
               for v in node_list2:
                  #print('v.name:', v['name'])
                  if d['input'][0] == v['name']:
                     print('got GlobalAveragePool, shape:', v['shape'])
                     if v['shape'][2] <= 15 and v['shape'][3] <= 15:
                           need_convert = True
                           print('GlobalAveragePool===>AveragePool......') 
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
            print('+++ The model cannot be saved for: %s' % e)
            if 'No Op registered for Mish' in str(e):
                  print('ignore mish warning, continue saving~')
            else:
                  sys.exit()    
         else:
            print('+++ Begin saving model...')

         onnx.save(model, onnxfile)

def post_process(onnxfile, inference_success, gap_to_ap):
   start_time = time.time()

   delete, post_process_file = optimization_op(onnxfile)
   while delete == True:
      delete, post_process_file = optimization_op(post_process_file)

   delete, post_process_file = eliminate_redundant_reshape(post_process_file)
   while delete == True:
      delete, post_process_file = eliminate_redundant_reshape(post_process_file)   

   end_time1 = time.time()

   print('optimization_op cost', end_time1 - start_time, ' seconds')   

   if gap_to_ap == 1:
      if inference_success == True:
         convert_gap_2_ap(post_process_file)
      else:
         print('Cannot do inference, so skip global_average_pool-->average_pool')    

   end_time2 = time.time()

   print('convert_gap_2_ap cost', end_time2 - end_time1, ' seconds')  

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
      print('~~~~ The model cannot be extracted for: %s' % e)
      if 'No Op registered for Mish' in str(e):
            print('ignore mish warning, continue~')
      else:
            sys.exit()    
   else:
      print('~~~~ Begin extracting model...')

   model = onnx.load(input_path)

   e = onnx.utils.Extractor(model)
   extracted = e.extract_model(input_names, output_names)

   onnx.save(extracted, output_path)
   #onnx.checker.check_model(output_path)
   try:
      onnx.checker.check_model(output_path)
   except onnx.checker.ValidationError as e:
      print('^^^^ The model cannot be extracted for: %s' % e)
      if 'No Op registered for Mish' in str(e):
            print('ignore mish warning, continue~')
      else:
            sys.exit()    
   else:
      print('^^^^ Finish extracting model...')

def extract_sub_graph(input_path, output_path, input_names, output_names):
   print('input_names:', input_names, ', output_names:', output_names)
   input_list = input_names.split(',')
   output_list = output_names.split(',')
   #onnx.utils.extract_model(input_path, output_path, input_list, output_list)
   #my_extract_model(input_path, output_path, input_list, output_list)

   try:
      onnx.utils.extract_model(input_path, output_path, input_list, output_list)
   except BaseException as e:
      print('The model cannot be extracted for: %s' % e)
      return False   
   else:
      print('Inference success---')
      return True

def add_value_info_for_constants(model : onnx.ModelProto):
   # All (top-level) constants will have ValueInfos before IRv4 as they are all inputs
   if model.ir_version < 4:
      return

   def add_const_value_infos_to_graph(graph : onnx.GraphProto):
      inputs = {i.name for i in graph.input}
      in_ = {i.name: i for i in graph.input}
      for init in graph.initializer:
         # Check it really is a constant, not an input
         if init.name in inputs:
               continue

         # The details we want to add
         elem_type = init.data_type
         shape = init.dims

         # Get existing or create new value info for this constant
         vi = in_.get(init.name)
         if vi is None:
               vi = graph.input.add()
               vi.name = init.name

         # Even though it would be weird, we will not overwrite info even if it doesn't match
         tt = vi.type.tensor_type
         if tt.elem_type == onnx.TensorProto.UNDEFINED:
               tt.elem_type = elem_type
         if not tt.HasField("shape"):
               # Ensure we set an empty list if the const is scalar (zero dims)
               tt.shape.dim.extend([])
               for dim in shape:
                  tt.shape.dim.add().dim_value = dim

      # Handle subgraphs
      for node in graph.node:
         for attr in node.attribute:
               # Ref attrs refer to other attrs, so we don't need to do anything
               if attr.ref_attr_name != "":
                  continue

               if attr.type == onnx.AttributeProto.GRAPH:
                  add_const_value_infos_to_graph(attr.g)
               if attr.type == onnx.AttributeProto.GRAPHS:
                  for g in attr.graphs:
                     add_const_value_infos_to_graph(g)

   return add_const_value_infos_to_graph(model.graph)

def eliminate_unused_input_initializer(model, output):
   init_list = []
   for init in model.graph.initializer:
      #print("init name:", init.name)
      init_list.append(init.name)   

   #print('==================================++++++++++++++++++')

   real_input_init = []
   node = model.graph.node[0]    
   for n in node.input:
      if n in init_list:
         real_input_init.append(n)

   for n in real_input_init:
      print("real_input_init:", n)

   #ValueInfoProto 
   vip = []
   need_eliminate = False

   for input in model.graph.input:
      if input.name in real_input_init:
         vip.append(input)
      elif input.name not in init_list:
         vip.append(input)
      elif input.name in init_list and input.name not in real_input_init:
         need_eliminate = True

   if need_eliminate == True:
      print('need to eliminate invalid initializer in input')

      del model.graph.input[:]

      model.graph.input.extend(vip)

      #for input in model.graph.input:
      #   print("xxx got  input name:", input.name)

      #onnx.checker.check_model(model)
      onnx.save(model, output)

def eliminate_redundant_reshape(onnxfile):
   model = onnx.load(onnxfile)
   reshape_input = []
   reshape_output = []

   delete_node_id = 0
   delete = False

   for node_id, node in enumerate(model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #   ", op:", node.op_type, ', len(input):', len(node.input))

      if node.op_type == 'Reshape':
         print('eliminate_redundant_reshape, got Reshape node:', node.input)
         reshape_input.extend(node.input)
         reshape_output.extend(node.output)
         delete_node_id = node_id
         break

   if len(reshape_input) > 0:
      got_value = False
      reshape_input_shape = []

      for v in model.graph.value_info:
         if v.name == reshape_input[0]:
               print('got value info:', reshape_input) 
               got_value = True
               for d in v.type.tensor_type.shape.dim:
                  reshape_input_shape.append(d.dim_value)
                  
               break

      if got_value == True:
         shape_list = []
         for init in model.graph.initializer:
               if init.name == reshape_input[1]:
                  #print('-------')
                  #print('init.name', init.name)
                  dtype = init.data_type
                  np_dtype = convert_ort_type_2_np(dtype)
                  if init.raw_data:
                     params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                     for p in params_list:
                           #print('p:', p)
                           shape_list.append(p)
                  else:
                     data_list = get_data_list(dtype, init)
                     for p in data_list:
                           #print('---p:', p)
                           shape_list.append(p)

                  if reshape_input_shape == shape_list and len(shape_list) > 0:
                     print('need eliminate_reshape')
                     delete = True

                  break            

   if delete == True:     
      print('eliminate_redundant_reshape, delete: ', delete_node_id)
      delete_node = model.graph.node[delete_node_id]

      last_node = True

      for node_id, node in enumerate(model.graph.node):
         if node.input[0] == reshape_output[0]:
               print('got reshape next node:', node.name)
               next_node = model.graph.node[node_id]
               next_node.input[0] = delete_node.input[0]
               last_node = False
               break

      model.graph.node.remove(delete_node)

      if last_node == True:
         #model.graph.output.extend()
         for node_id, node in enumerate(model.graph.node):
               #print('+++++====', node.input[0], reshape_output[0])
               if node.output[0] == reshape_input[0]:
                  print('eliminate_redundant_reshape, got reshape prev node:', node.name)
                  prev_node = model.graph.node[node_id]
                  prev_node.output[0] = reshape_output[0]
                  break

      ###################
      #onnx.checker.check_model(model)
      onnx.save(model, onnxfile)

   return delete, onnxfile

def process(args):
   global support_mish
   global inputs_as_nchw

   model_path = args.model_path
   model_type = args.model_type
   output = args.output
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
   paddle_input_type = args.paddle_input_type
   model_weights_file = args.model_weights_file
   inputs_as_nchw = args.inputs_as_nchw
   gap_to_ap = args.gap_to_ap
   fuse_pad_pool = args.fuse_pad_pool
   support_swish = args.support_swish

   print('model_path:{}, model_type:{}, output:{}'.format(model_path, model_type, output))

   if model_type == 'tf-ckpt' or model_type == 'tf-graph' :
      print('checkpoint:', inputs, outputs)

   dynamic_paddle = False
   if model_type == 'paddle':
         dynamic_paddle = is_dynamic_paddle(input_shape, model_def_file, model_class_name, model_weights_file)
         if dynamic_paddle == True and paddle_input_type == '':
            paddle_input_type = 'float32' 

   torchvision_model = False
   if model_type == 'pytorch':
      if '.' in model_class_name:
         torchvision_model = True

   if dynamic_paddle == False and torchvision_model == False and not os.path.exists(model_path):
      print('ERROR: {} is not exist'.format(model_path))
      sys.exit()

   if model_type not in valid_model_type:
      print('ERROR: {} is not valid mode type'.format(model_type))
      print('valid mode type is {}'.format(valid_model_type))
      sys.exit()

   op_set_default = 11

   #if op_set == None:
   #   op_set = op_set_default

   if model_type == 'pytorch' and args.input_shape == '':
      print('When converting pytorch model, you must tell the input shape(ex: --input_shape [1, 3, 32, 32])')
      print('Also, you should provide model definition file')
      sys.exit()

   if (model_type == 'tf-ckpt' or model_type == 'tf-graph') and (args.inputs == '' or args.outputs == ''):
      print('When converting checkpoint/graph, you must tell the inputs(ex: --inputs input0:0,input1:0) and outputs(ex: --outputs output0:0)')
      sys.exit()

   if extract_sub == 1:
      if args.inputs == '' or args.outputs == '':
         print('When extract sub graph, you must tell the inputs(ex: --inputs input0:0,input1:0) and outputs(ex: --outputs output0:0)')
         sys.exit()

      if model_type != 'onnx':
         print('WARNNING: only onnx model supports extracting...')
         sys.exit()

      r = extract_sub_graph(model_path, output, inputs, outputs)
      if r == True:
         logger.info('Convert Success!')
         
      sys.exit()                

   if model_type == 'pytorch' or model_type == 'paddle':
      input_shape=args.input_shape.strip('[')
      input_shape=input_shape.strip(']')
      input_shape=input_shape.split(',')
      #print(type(input_shape[0])) 
      print('got shape:', input_shape)

   print('begin convert..')

   begin_time = time.time()

   if model_type != 'onnx':
      convert(model_path, 
               model_type, 
               output, 
               op_set_default, 
               input_shape, 
               inputs, 
               outputs,
               model_def_file,
               model_class_name,
               paddle_input_type,
               model_weights_file)

   end_time1 = time.time()
  
   print('finish convert, it cost', end_time1 - begin_time, ' seconds')

   if model_type != 'onnx':
      model = onnx.load(output)
   else:
      model = onnx.load(model_path)

   if op_set != None :
      if model_type == 'onnx':
         print('ONNX, add_value_info_for_constants...')
         correct_batch_for_opset_convert(model)
         add_value_info_for_constants(model)
         model = version_converter.convert_version(model, op_set)
      elif op_set != op_set_default:
         model = version_converter.convert_version(model, op_set)

   inference_success = False
   new_model = model
   #new_model = onnx.shape_inference.infer_shapes(model)
   try:
      new_model = onnx.shape_inference.infer_shapes(model)
   except BaseException as e:
      print('The model cannot be inferenced for: %s' % e)
      new_model = model    
   else:
      print('Inference success---')
      inference_success = True

   #onnx.checker.check_model(new_model)
   try:
      onnx.checker.check_model(new_model)
   except onnx.checker.ValidationError as e:
      print('### The model cannot be saved for: %s' % e)
      if 'No Op registered for Mish' in str(e) or 'No opset import for domain' in str(e) :
            print('ignore warning, continue saving~')  
      else:
            sys.exit()
   else:
      print('### Begin saving model...')

   onnx.save(new_model, output)

   if dynamic_batch == 1:
      print('modify model to dynamic batch...')
      modify_onnx2dynamic(new_model, output)

   end_time2 = time.time()

   print('generate inference shape model, it cost', end_time2 - end_time1, ' seconds')

   post_process(output, inference_success, gap_to_ap)
   new_model = onnx.load(output)

   if simplify_model == 1:
      print('begin doing simplify...')
      new_model = model_simplify(output)

   if fuse_pad_pool == 1:
      print('begin doing fuse_pad_to_pool...')
      new_model = fuse.fuse_pad_to_pool(new_model, output)   

   if fp32_to_fp16 == 1:
      print('begin doing fp32-->fp16...')
      new_model = convert_float_to_float16(new_model, keep_io_types=True)
      onnx.save(new_model, output)

   if model_type == 'onnx' and support_mish == 1:
      convert_mish(model_path, output, op_set)

   if model_type == 'onnx' and support_swish == 1:
      merge_swish_and_hard_swish(new_model, output)   

   if model_type == 'onnx' and preproc_yaml != '':
      if os.path.exists(preproc_yaml):
         preproc(new_model, output)
      else:
         print(preproc_yaml, 'is not exist')    

   if model_type == 'onnx' and postproc_yaml != '':
      if os.path.exists(postproc_yaml):
         postproc(new_model, output)
      else:
         print(postproc_yaml, 'is not exist')

   eliminate_unused_input_initializer(new_model, output)                 

   end_time3 = time.time()

   print('The whole progress cost', end_time3 - begin_time, ' seconds')

   logger.info('Convert Success!')


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
   #print(args)
   process(args)

if __name__ == "__main__":
   args = parse_args()
   main(args)
