import torch
import numpy as np
import sys
import torchvision
from unet import Net

model = torch.load('./mnist_model_cpu.pkl')

model.eval()

np.random.seed(6)

data = np.array(np.random.random([1,1,28,28]), dtype = np.float32)

#print('000 input:', data)

np.save('cnn_input.npy', data)

data = torch.from_numpy(data)

print('000 input:', data)

#data = data.to('cuda')

output = model(data)

print('000 output:', output)

output = output.cpu().detach().numpy()

np.save('cnn_output.npy', output)

#print('output:', output)
max_ = np.argmax(output, axis=1)

print('111 output:', output, max_)


'''
model = Net()

model.load_state_dict(torch.load('./9_epoch_iou_0.9743422508239746.pth'))

np.random.seed(6)

data = np.array(np.random.random([64,3,32,32]), dtype = np.float32)

#print('000 input:', data)

np.save('cnn_input.npy', data)

data = torch.from_numpy(data)

print('000 input:', data)

#data = data.to('cuda')

output = model(data)

print('000 output:', output)

output = output.cpu().detach().numpy()

np.save('cnn_output.npy', output)

#print('output:', output)
max_ = np.argmax(output, axis=1)

print('111 output:', output.shape)
'''


'''
model =  torchvision.models.resnet50()

model.load_state_dict(torch.load('./0.96966957919051920.9139525532770406.pth'))

model.eval()

np.random.seed(6)

data = np.array(np.random.random([16,3,256,256]), dtype = np.float32)

#print('000 input:', data)

np.save('cnn_input.npy', data)

data = torch.from_numpy(data)

print('000 input:', data)

#data = data.to('cuda')

output = model(data)

print('000 output:', output)

output = output.cpu().detach().numpy()

np.save('cnn_output.npy', output)

#print('output:', output)
max_ = np.argmax(output, axis=1)

print('111 output:', max_)
'''

