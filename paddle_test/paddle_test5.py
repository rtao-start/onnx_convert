import paddle
import paddle.nn.functional as F
from paddle.nn import Layer
from paddle.vision.datasets import MNIST
from paddle.metric import Accuracy
from paddle.nn import Conv2D,MaxPool2D,Linear
from paddle.static import InputSpec
from paddle.vision.transforms import ToTensor

train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())

class MyModel(Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = Linear(in_features=16*5*5, out_features=120)
        self.linear2 = Linear(in_features=120, out_features=84)
        self.linear3 = Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

'''
inputs = InputSpec([None, 784], 'float32', 'x')
labels = InputSpec([None, 10], 'float32', 'x')
model = paddle.Model(MyModel(), inputs, labels)
 
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
 
model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy()
    )

model.fit(train_dataset,
        test_dataset,
        epochs=3,
        batch_size=64,
        save_dir='mnist_checkpoint',
        verbose=1
        )
'''
##############################################################
train_dataset = MNIST(mode='train', transform=ToTensor())
test_dataset = MNIST(mode='test', transform=ToTensor())
 
inputs = InputSpec([None, 784], 'float32', 'inputs')
labels = InputSpec([None, 10], 'float32', 'labels')
model = paddle.Model(MyModel(), inputs, labels)
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
model.load("./mnist_checkpoint/final")
model.prepare( 
      optim,
      paddle.nn.loss.CrossEntropyLoss(),
      Accuracy()
      )
      
model.fit(train_data=train_dataset,
        eval_data=test_dataset,
        batch_size=64,
        epochs=2,
        verbose=1
        )