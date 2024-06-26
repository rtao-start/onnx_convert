import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint: xxx.ckpt(千万不要加后面的xxx.ckpt.data这种，到ckpt就行了!)
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    #output_node_names = "rois/Reshape/shape" # 模型输入节点，根据情况自定义
    output_node_names = 'y_pred'
    #clearsess
    #if sess 中有模型
    #saver = tf.train.Saver()
    #else:
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    
    graph = tf.compat.v1.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint) # 恢复图并得到数据
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开
 
        #with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
        with tf.io.gfile.GFile(output_graph, "wb") as f: #保存模型            
            f.write(output_graph_def.SerializeToString()) #序列化输出

freeze_graph('./ckpt_models/dog-cat.ckpt-7950', './1.pb')