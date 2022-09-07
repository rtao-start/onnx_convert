'''
静态图
'''

import paddle, sys
import numpy as np
import paddle.fluid as fluid

paddle.enable_static()

epoch_num = 10
BATCH_SIZE = 64
train_reader = paddle.batch(paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=128, drop_last=False)


class Mnist(fluid.Layer):
    def __init__(self):
        super(Mnist, self).__init__()
        self.x = fluid.data(name="img", shape=[None, 1, 28, 28], dtype="float32")
        self.y = fluid.data(name="label", shape=[None, 1], dtype="int64")

    def forward(self):
        self.conv_bn1 = fluid.layers.batch_norm(fluid.layers.conv2d(input=self.x,num_filters=8, filter_size=(3,3),
                                                                    stride=2, padding="SAME", act=None),act="leaky_relu")
        self.conv_bn2 = fluid.layers.batch_norm(fluid.layers.conv2d(input=self.conv_bn1,num_filters=16, filter_size=(3,3),
                                                                    stride=2, padding="SAME", act=None),act="leaky_relu")
        print(self.conv_bn2.shape)  # 7*7
        self.conv_bn_pool = fluid.nets.img_conv_group(input=self.conv_bn2, conv_num_filter=(32,32), conv_padding=1, conv_act="leaky_relu",
                                                   conv_filter_size=3, conv_with_batchnorm=True, pool_size=3, pool_stride=2)
        print(self.conv_bn_pool.shape)  # 3*3
        self.feat = fluid.layers.reshape(self.conv_bn_pool, shape=(-1, 3*3*32))
        self.fc = fluid.layers.fc(input=self.feat, size=10, act="softmax")

        # self.output = np.argmax(self.fc, axis=1)
        # 在网络内部做argmax，会报超出维度的错误，猜测是网络里self.fc的形状为(-1,10),不能直接在维度上变换

        return self.fc

    def backward(self):
        loss = fluid.layers.cross_entropy(self.fc, self.y)
        avg_loss = fluid.layers.mean(loss)

        return avg_loss

def test_static():
    data = fluid.layers.data(name='data', shape=[3, 32, 32], dtype='float32')
    param_attr = fluid.ParamAttr(name='conv2d.weight', initializer=fluid.initializer.Xavier(uniform=False), learning_rate=0.001)

    res = fluid.layers.conv2d(input=data, num_filters=2, filter_size=3, act="relu", param_attr=param_attr)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    x = np.random.rand(1, 3, 32, 32).astype("float32")

    output = exe.run(feed={"data": x}, fetch_list=[res])

    print(output)

if __name__ == '__main__':
    test_static()
    sys.exit()
    
    use_cuda = True
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    net = Mnist()

    out = net.forward()
    avg_loss = net.backward()

    boundaries = [4685, 6246, 7808]
    lr_steps = [0.001, 0.0005, 0.00001, 0.00005]
    learning_rate = fluid.layers.piecewise_decay(boundaries, lr_steps)
    lr = fluid.layers.linear_lr_warmup(learning_rate=learning_rate, warmup_steps=500, start_lr=0.0001, end_lr=0.001)

    opt = fluid.optimizer.MomentumOptimizer(learning_rate=lr, momentum=0.9)
    opt.minimize(avg_loss)     # opt.minimize()不能放在训练中求出loss后面，会报错，tf是放在后面的

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())      # 该函数可以获取默认/全局 startup Program (初始化启动程序)。
    #  default_startup_program 只运行一次来初始化参数， default_main_program 在每个mini batch中运行并更新权重。
    main_program = fluid.default_main_program()   # 获取当前用于存储op和variable描述信息的 default main program
    test_program = fluid.default_main_program().clone(for_test=True)

    for epoch in range(epoch_num):
        for batch_id, data in enumerate(train_reader()):
            x = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
            y = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

            _, loss, _lr = exe.run(main_program, feed={"img": x, "label": y}, fetch_list=[out, avg_loss, lr])

            if batch_id % 100 == 0:
                print("epoch {}  step {}  lr {}  Loss {}".format(epoch, batch_id, _lr, loss))

        test_num = 0
        rigth_num = 0
        for _, data in enumerate(test_reader()):
            x = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
            y = np.array([x[1] for x in data]).astype('int64').reshape(-1, 1)

            output, loss = exe.run(test_program, feed={"img": x, "label": y}, fetch_list=[out, avg_loss])

            pred = np.argmax(output, axis=1)
            label = y.T[0]

            rigth_num += (pred == label).sum()
            test_num += pred.shape[0]

        # print("pred: ", pred[:10])
        # print("label:", label[:10])
        acc = rigth_num / test_num
        print("test_acc:", acc)
        print('-' * 40)
