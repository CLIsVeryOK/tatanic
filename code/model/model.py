import torch.nn as nn
import torch.nn.functional as F


# 定义Net类，运用nn中的Module模块，Module是继承的父类
class Net(nn.Module):
    # 定义神经网络结构   1x32x32
    def __init__(self):
        # super()调用下一个父类，并返回父类实例的一个方法
        super(Net, self).__init__()

        # 第一层：fc1
        self.fc1 = nn.Linear(11, 32)
        # 第二层：fc2
        self.fc2 = nn.Linear(32, 64)
        # 第三层：fc3
        self.fc3 = nn.Linear(64, 32)
        # 第四层：fc4
        self.fc4 = nn.Linear(32, 1)

    # 定义神经网络数据流向:
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        return x


if __name__ == '__main__':
    # 打印神经网络：
    net = Net()
    print(net)
