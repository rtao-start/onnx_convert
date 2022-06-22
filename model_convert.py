import os
import onnx
import copy
import numpy as np
import logging
import onnxruntime
import sys, getopt
import json
import argparse
import h5py
import tensorflow as tf
import torch
import time

from caffe2onnx.src.load_save_model import loadcaffemodel, saveonnxmodel
from caffe2onnx.src.caffe2onnx import Caffe2Onnx
from onnxsim.onnx_simplifier import simplify

logging.basicConfig(level=logging.INFO)

from onnx import shape_inference, TensorProto, version_converter, numpy_helper

logger = logging.getLogger("[Any2ONNX]")

import argparse

valid_model_type = ['caffe', 'pytorch', 'tf-h5', 'tf-ckpt', 'tf-sm', 'tf-graph', 'darknet', 'onnx']

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

   parser.add_argument("--datatype_convert",
                        type=str, required=False,
                        help="data type convert(ex: fp32_to_fp16)") 

   parser.add_argument("--q_dataset_file",
                        type=str, required=False,
                        help="quantization prepare file(ex: ./1.npy)")

   parser.add_argument("--q_onnx_file",
                        type=str, required=False,
                        help="quantization output onnx file(ex: ./output_q.onnx)") 

   parser.add_argument("--simplify",
                        type=int, required=False,
                        default=1,
                        help="simplify the model")                              

   #for pytorch
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

   #for pytorch
   parser.add_argument("--extract_sub",
                        type=int, 
                        required=False,
                        default=0,
                        help="extract sub graph")                                                                    
                                                                  
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

def convert_caffe2onnx(model_path, output):
      print('Begin converting caffe to onnx...')
      prototxt_file, caffemodel_file = get_caffe_files(model_path)

      if prototxt_file == '' or caffemodel_file == '':
         print('ERROR: prototxt file or caffemodel file is None')
         sys.exit()

      onnxmodel_path = output

      graph, params = loadcaffemodel(prototxt_file, caffemodel_file)
      c2o = Caffe2Onnx(graph, params, onnxmodel_path)
      onnxmodel = c2o.createOnnxModel()

      saveonnxmodel(onnxmodel, onnxmodel_path)

def convert_sm2onnx(model_path, output, op_set):
      print('Begin converting tf-savemodel to onnx...')
      cmd = 'python -m tf2onnx.convert --saved-model ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      print('convert_tfsm2onnx: ', cmd)
      os.system(cmd)

def convert_h52onnx(model_path, output, op_set):
      print('Begin converting tf-savemodel to onnx...')
      cmd = 'python -m tf2onnx.convert --keras ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output
      print('convert_tfh52onnx: ', cmd)
      os.system(cmd)     

def convert_ckpt2onnx(model_path, output, op_set, inputs, outputs):
      print('Begin converting tf-ckpt to onnx...')
      cmd = 'python -m tf2onnx.convert --checkpoint ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs

      print('convert_ckpt2onnx: ', cmd)

      os.system(cmd)

def convert_graph2onnx(model_path, output, op_set, inputs, outputs):
      print('Begin converting tf-graph to onnx...')
      cmd = 'python -m tf2onnx.convert --graphdef ' + model_path + ' --opset ' + str(op_set) + ' --output ' + output \
              + ' --inputs '  + inputs + ' --outputs ' + outputs

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

def convert_pt2onnx(model_path, output, op_set, input_shape):
      print('Begin converting pytorch to onnx...')
      #cfg_file, weights_file = get_darknet_files(model_path)

      m = torch.load(model_path)
      m = m.cpu() #cuda()

      x = torch.randn(int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))
      torch.onnx.export(
         m,
         x,
         output,
         opset_version=op_set, 
         do_constant_folding=True,   # 是否执行常量折叠优化
         input_names=["input"],    # 模型输入名
         output_names=["output"],  # 模型输出名
         dynamic_axes={'input':{0:'batch_size'}, 'output':{0:'batch_size'}}
      )

def convert_dn2onnx(model_path, output, op_set):
      print('Begin converting darknet to onnx...')
      cfg_file, weights_file = get_darknet_files(model_path)

      cmd = 'python ./darknet2onnx.py --cfg_file ' + cfg_file + ' --weights_file ' + weights_file + ' --output ' + output
     
      print('convert_dn2onnx: ', cmd)

      os.system(cmd) 

def convert(model_path, model_type, output, op_set, input_shape, inputs, outputs):
   if model_type == 'caffe':
      convert_caffe2onnx(model_path, output)

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
      convert_pt2onnx(model_path, output, op_set, input_shape)       

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
      export_onnx = onnxfile #"./tmp.onnx"
      onnx.checker.check_model(model)
      onnx.save(model, export_onnx)

   return delete, export_onnx

def model_simplify(model_path):
   onnx_model = onnx.load(model_path)
   model_simp, check = simplify(onnx_model)
   onnx.save(model_simp, model_path)

def modify_onnx2dymnamic(model_path):
   onnx_model = onnx.load(model_path)

   for idx in range(len(onnx_model.graph.input)):
      dim_proto_input = onnx_model.graph.input[idx].type.tensor_type.shape.dim[0]
      # dim_proto_input.dim_param = 'bs'
      dim_proto_input.dim_value = -1

   for idx in range(len(onnx_model.graph.output)):
      dim_proto_output = onnx_model.graph.output[idx].type.tensor_type.shape.dim[0]
      # dim_proto_output.dim_param = 'bs'
      dim_proto_output.dim_value = -1

   try:
      onnx.checker.check_model(onnx_model)
   except onnx.checker.ValidationError as e:
      print('The model cannot be modified for: %s' % e)
   else:
      print('The model is modified!')
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
         onnx.checker.check_model(model)
         onnx.save(model, onnxfile)

def post_process(onnxfile):
   start_time = time.time()

   delete, post_process_file = optimization_op(onnxfile)

   while delete == True:
      delete, post_process_file = optimization_op(post_process_file)

   end_time1 = time.time()

   print('optimization_op cost', end_time1 - start_time, ' seconds')   

   convert_gap_2_ap(post_process_file) 

   end_time2 = time.time()

   print('convert_gap_2_ap cost', end_time2 - end_time1, ' seconds')     

def extract_sub_graph(input_path, output_path, input_names, output_names):
   print('input_names:', input_names, ', output_names:', output_names)
   input_list = input_names.split(',')
   output_list = output_names.split(',')
   onnx.utils.extract_model(input_path, output_path, input_list, output_list)

def process(args):
   model_path = args.model_path
   model_type = args.model_type
   output = args.output
   op_set = args.op_set
   input_shape = args.input_shape
   inputs = args.inputs
   outputs = args.outputs
   simplify_model = args.simplify
   extract_sub = args.extract_sub

   print('model_path:{}, model_type:{}, output:{}'.format(model_path, model_type, output))

   if model_type == 'tf-ckpt' or model_type == 'tf-graph' :
      print('checkpoint:', inputs, outputs)

   if not os.path.exists(model_path):
      print('ERROR: {} is not exist'.format(model_path))
      sys.exit()

   if model_type not in valid_model_type:
      print('ERROR: {} is not valid mode type'.format(model_type))
      print('valid mode type is {}'.format(valid_model_type))
      sys.exit()

   if op_set == None:
      op_set = 11

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

      extract_sub_graph(model_path, output, inputs, outputs)
      sys.exit()                

   if model_type == 'pytorch':
      input_shape=args.input_shape.strip('[')
      input_shape=input_shape.strip(']')
      input_shape=input_shape.split(',')
      #print(type(input_shape[0])) 
      #print('got shape:', input_shape)

   print('begin convert..')

   begin_time = time.time()

   if model_type != 'onnx':
      convert(model_path, model_type, output, op_set, input_shape, inputs, outputs)

   end_time1 = time.time()

   print('finish convert, it cost', end_time1 - begin_time, ' seconds')

   if model_type != 'onnx':
      model = onnx.load(output)
   else:
      model = onnx.load(model_path)
      
   new_model = onnx.shape_inference.infer_shapes(model)
   onnx.checker.check_model(new_model)
   onnx.save(new_model, output)

   end_time2 = time.time()

   print('generate inference shape model, it cost', end_time2 - end_time1, ' seconds')

   post_process(output)

   if simplify_model == 1:
      print('begin doing simplify...')
      model_simplify(output)

   end_time3 = time.time()

   print('The whole progress cost', end_time3 - begin_time, ' seconds')

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
