import onnx
import values 
import numpy as np
import log

logger = log.getLogger(__name__, log.INFO)

def remove_node(model, inputs, outputs):
    for node in model.graph.node:
        if node.input == inputs and node.output == outputs:
            model.graph.node.remove(node)
            logger.debug('remove node: {}'.format(node.name))
            break

def is_unused_init(model, init):
    for node in model.graph.node:
        if init.name in node.input:
            return False

    return True

def is_unused_init2(model, init, node_):
    for node in model.graph.node:
        if init.name in node.input and node != node_:
            return False

    return True

def remove_unused_initializer_list(model, unused_init_list):
    for init in unused_init_list:
        if is_unused_init(model, init):
            logger.debug('remove unused init: {}'.format(init.name))
            model.graph.initializer.remove(init) 

def remove_unused_initializer(model):
   for init in model.graph.initializer:
      print('====got init: ', init.name)
      if is_unused_init(model, init):
         logger.debug('---remove unused init: {}'.format(init.name))
         model.graph.initializer.remove(init)                        

def eliminate_unused_input_initializer(model):
   init_list = []
   for init in model.graph.initializer:
      #print("init name:", init.name)
      init_list.append(init.name)   

   #print('==================================++++++++++++++++++')

   real_input_init = []
   '''
   node = model.graph.node[0]    
   for n in node.input:
      if n in init_list:
         real_input_init.append(n)
   '''      

   #for n in real_input_init:
   #   print("real_input_init:", n)

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
      logger.debug('Now eliminate invalid initializer in input')

      del model.graph.input[:]

      model.graph.input.extend(vip)

      for input in model.graph.input:
         logger.debug('last input: {}'.format(input.name))

def eliminate_unused_constant_node(model):
   constant_idx_name = []
   for node in model.graph.node:
      if node.op_type == 'Constant':
         logger.debug('eliminate_unused_constant_node, node.name: {}'.format(node.name))
         dict_ = {}
         dict_['output'] = node.output[0]
         dict_['del'] = True
         constant_idx_name.append(dict_)

   for node in model.graph.node:
      if node.op_type != 'Constant':
         for d in constant_idx_name:
            if d['output'] in node.input:
               d['del'] = False

   for output in model.graph.output:
      for d in constant_idx_name:
         if d['output'] == output.name:
            d['del'] = False
            break

   del_constant_output = []
   for d in constant_idx_name:
      if d['del'] == True:
         del_constant_output.append(d['output'])

   for node in reversed(model.graph.node):
      if node.op_type == 'Constant':
         if node.output[0] in del_constant_output:
            model.graph.node.remove(node)

def eliminate_redundant_reshape(model):
   reshape_input = []
   reshape_output = []

   delete_node_id = 0
   delete = False

   for node_id, node in enumerate(model.graph.node):
      #print(node_id, ", name:", node.name, ", input:", node.input, ", output:", node.output,  \
      #   ", op:", node.op_type, ', len(input):', len(node.input))

      if node.op_type == 'Reshape':
         logger.debug('eliminate_redundant_reshape, got Reshape node: {}'.format(node.input))
         reshape_input.extend(node.input)
         reshape_output.extend(node.output)
         delete_node_id = node_id
         break

   if len(reshape_input) > 0:
      got_value = False
      reshape_input_shape = []

      for v in model.graph.value_info:
         if v.name == reshape_input[0]:
               logger.debug('got value info: {}'.format(reshape_input)) 
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
                  np_dtype = values.convert_ort_type_2_np(dtype)
                  if init.raw_data:
                     params_list = np.fromstring(init.raw_data, dtype=np_dtype)
                     for p in params_list:
                           #print('p:', p)
                           shape_list.append(p)
                  else:
                     data_list = values.get_data_list(dtype, init)
                     for p in data_list:
                           #print('---p:', p)
                           shape_list.append(p)

                  if reshape_input_shape == shape_list and len(shape_list) > 0:
                     logger.debug('need eliminate_reshape')
                     delete = True

                  break            

   if delete == True:     
      logger.debug('eliminate_redundant_reshape, delete: {}'.format(delete_node_id))
      delete_node = model.graph.node[delete_node_id]

      last_node = True

      for node_id, node in enumerate(model.graph.node):
         if len(node.input) > 0 and node.input[0] == reshape_output[0]:
            logger.debug('got reshape next node: {}'.format(node.name))
            next_node = model.graph.node[node_id]
            next_node.input[0] = delete_node.input[0]
            last_node = False
            break
         #elif len(node.input) == 0:
         #   print('Got a constant node:', node.name, ',', node.input, ', ', node.output)   

      model.graph.node.remove(delete_node)

      if last_node == True:
         #model.graph.output.extend()
         for node_id, node in enumerate(model.graph.node):
               #print('+++++====', node.input[0], reshape_output[0])
               if node.output[0] == reshape_input[0]:
                  logger.debug('eliminate_redundant_reshape, got reshape prev node: {}'.format(node.name))
                  prev_node = model.graph.node[node_id]
                  prev_node.output[0] = reshape_output[0]
                  break

      ###################
      #onnx.save(model, onnxfile)

   return delete

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


def get_all_next_node_by_output(model, output):
    node_list = []
    ok = -1

    for node in model.graph.node:
        if output in node.input:
            node_list.append(node)
            ok = 0

    return node_list, ok

def insert_onnx_node(model, insert_node, follow_up_node):
    # 根据插入Node的输出修改后续node的输入
    #follow_up_node.input[0] = insert_node.output[0]
    # 找到后续Node的索引位置，并将插入节点插入到graph中
    for follow_up_node_index, _follow_up_node in enumerate(model.graph.node):
        if _follow_up_node == follow_up_node:
            logger.debug("follow_up_node_index: {}".format(follow_up_node_index))
            model.graph.node.insert(follow_up_node_index, insert_node)
            break

def remove_onnx_node(model, node):
   init_name_list = []
   init_name_map = {}

   for init in model.graph.initializer:
      init_name_list.append(init.name)
      init_name_map[init.name] = init

   for input_ in node.input:
      if input_ in init_name_list:
         if is_unused_init2(model, init_name_map[input_], node):
            model.graph.initializer.remove(init_name_map[input_])

   model.graph.node.remove(node)

def remove_initializer_if_necessary(model, init, node):
   if is_unused_init2(model, init, node):
      model.graph.initializer.remove(init)


