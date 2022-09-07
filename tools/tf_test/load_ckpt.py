import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

tf.compat.v1.disable_eager_execution()

image_size=64
num_channels=3
images = []

path = '3.jpg'
image = cv2.imread(path)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

in_ = x_batch.transpose(0,3,1,2)
print('x_batch', in_.shape)

np.save('ckpt_input.npy', in_)

## Let us restore the saved model 
sess = tf.compat.v1.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.compat.v1.train.import_meta_graph('./dogs-cats-model/dog-cat.ckpt-7950.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, './dogs-cats-model/dog-cat.ckpt-7950')

# Accessing the default graph which we have restored
graph = tf.compat.v1.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 2)) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)

print('result:', result.shape)

np.save('ckpt_output.npy', result)

# result is of this format [probabiliy_of_rose probability_of_sunflower]
# dog [1 0]
res_label = ['dog','cat']
print('it is:', res_label[result.argmax()])
