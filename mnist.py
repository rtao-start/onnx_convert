import paddle.fluid as fluid

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

        