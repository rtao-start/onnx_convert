import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import time

config = tf.compat.v1.ConfigProto()

sess = tf.compat.v1.Session(config=config)
with gfile.FastGFile('../graph.pb', 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    opname = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node] 
    print(opname)

# 获取输入tensor
x = tf.compat.v1.get_default_graph().get_tensor_by_name("x:0")    # 不知道输入名时通过节点名查，一般情况下是每一个节点tf.get_default_graph().as_graph_def().node[0].name,名字构成后有个:0
print("input:", x)

# 获取预测tensor
y_pred = tf.compat.v1.get_default_graph().get_tensor_by_name("y_pred:0")  # tf.get_default_graph().as_graph_def().node[-1].name，有可能不是是最后一一个
print(y_pred)

data = np.array(np.random.random([1, 64, 64, 3]), dtype = np.float32)

feed_dict_testing = {x: data}
result = sess.run(y_pred, feed_dict=feed_dict_testing)

print('result:', result)

'''
start = time.time()
for tx,ty in zip(x_test,y_test):
    pre = sess.run(pred, feed_dict={x:tx.reshape(1,28,28,1)/255}) # 预测直接run输出，传入输入
    pre = np.argmax(pre)
  #  print("Test Case: " + str(ty))
  #  print("Prediction: " + str(pre))
'''