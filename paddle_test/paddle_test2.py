'''
动态图
'''

import paddle, sys
import numpy as np
import paddle.fluid as fluid

epoch_num = 10
BATCH_SIZE = 64
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=60000), batch_size=BATCH_SIZE, drop_last=True)
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128, drop_last=False)

class Conv_Mnist(fluid.dygraph.Layer):
    def __init__(self):
        super(Conv_Mnist, self).__init__()
        self.conv1 = fluid.dygraph.Conv2D(num_channels=1, num_filters=8, filter_size=(3, 3), stride=2, padding=1)
        self.bn1 = fluid.dygraph.BatchNorm(num_channels=8, act="leaky_relu")

        self.conv2 = fluid.dygraph.Conv2D(num_channels=8, num_filters=16, filter_size=(3, 3), stride=2, padding=1)
        self.bn2 = fluid.dygraph.BatchNorm(num_channels=16, act="leaky_relu")

        self.conv3 = fluid.dygraph.Conv2D(num_channels=16, num_filters=32, filter_size=(3, 3), stride=2, padding=1)
        self.bn3 = fluid.dygraph.BatchNorm(num_channels=32, act="leaky_relu")

        self.fc = fluid.dygraph.Linear(input_dim=4*4*32, output_dim=10, act="softmax")

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        conv2 = self.conv2(bn1)
        bn2 = self.bn2(conv2)
        conv3 = self.conv3(bn2)
        bn3 = self.bn3(conv3)
        bn3 = fluid.layers.reshape(bn3, shape=(-1, 4*4*32))
        out = self.fc(bn3)

        return out

from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import Conv2D

def test_dynamic():
    x = np.random.rand(1, 3, 32, 32).astype("float32")

    with fluid.dygraph.guard():
        conv2d = Conv2D(num_channels=3,
                        num_filters=2,
                        filter_size=3,
                        act='relu')
        data = to_variable(x)
        conv = conv2d(data)

        print(conv.numpy())

if __name__ == '__main__':

    test_dynamic()
    sys.exit()

    with fluid.dygraph.guard():
        net = Conv_Mnist()

        learning_rate = fluid.layers.exponential_decay(learning_rate=0.001, decay_steps=1000, decay_rate=0.8)
        lr = fluid.layers.linear_lr_warmup(learning_rate=learning_rate, warmup_steps=500, start_lr=0.0001, end_lr=0.001)
        opt = fluid.optimizer.MomentumOptimizer(learning_rate=lr, momentum=0.9, parameter_list=net.parameters())
        # nll_loss = fluid.dygraph.NLLLoss()  loss为负，且loss和avg_loss一样，fluid.dygraph里貌似没有交叉熵

        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                x = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = fluid.dygraph.to_variable(x)
                label = fluid.dygraph.to_variable(y)

                out = net(img)
                loss = fluid.layers.cross_entropy(out, label)
                avg_loss = fluid.layers.mean(loss)

                _lr = opt.current_step_lr()  # 当前步的学习率

                avg_loss.backward()
                opt.minimize(avg_loss)
                net.clear_gradients()

                if batch_id % 100 == 0:
                    print("epoch {}  step {}  lr {}  Loss {}".format(epoch, batch_id, _lr, avg_loss.numpy()))

            test_num = 0
            rigth_num = 0
            for _, data in enumerate(test_reader()):
                net.eval()
                x = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                y = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

                img = fluid.dygraph.to_variable(x)
                label = fluid.dygraph.to_variable(y)
                out = net(img)

                output = np.argmax(out.numpy(), axis=1)
                _label = y.T[0]

                rigth_num += (output == _label).sum()
                test_num += output.shape[0]

            # print("output:", output[:10])
            # print("label: ", _label[:10])
            acc = rigth_num / test_num
            print("test_acc:", acc)
            print('-' * 60)

            net.train()
