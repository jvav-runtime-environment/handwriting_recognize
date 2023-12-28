from torch.utils.data import DataLoader
import torch as th
import torch.nn as nn
import torch.nn.functional as nf
import torch.optim as op
from torchvision import datasets, transforms

# 超参数

# 批处理数据大小
batch_size = 16
# 训练设备
device = th.device("cpu")
# 训练轮次
epochs = 10
# 图片变换
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307), (0.3081))  # 正则化,降低模型复杂度
])

# 下载
train_set = datasets.MNIST(
    "data", train=True, download=True, transform=pipeline)
test_set = datasets.MNIST(
    "data", train=False, download=True, transform=pipeline)

# 加载
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


# 构建模型


class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数，输出通道数，卷积核
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)  # 输入
        x = nf.relu(x)
        x = nf.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = nf.relu(x)

        x = x.view(input_size, -1)

        x = self.fc1(x)
        x = nf.relu(x)

        x = self.fc2(x)

        output = nf.log_softmax(x, dim=1)

        return output


model = Digit().to(device)

optimizer = op.Adam(model.parameters())


def train_modle(model: Digit, device, train_loader, optimizer: op.Adam, epoch):
    model.train()
    for batch_index, (data, tag) in enumerate(train_loader):
        data, tag = data.to(device), tag.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nf.cross_entropy(output, tag)
        loss.backward()
        optimizer.step()
        if batch_index % 1500 == 0:
            print('轮数:{}\t损失:{:.6f}'.format(epoch, loss.item()))


def test_model(model: Digit, device, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with th.no_grad():
        for data, tag in test_loader:
            data, tag = data.to(device), tag.to(device)
            output = model(data)
            test_loss += nf.cross_entropy(output, tag).item()
            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(tag.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('平均损失值:{:.4f}\t准确率{:.3f}'.format(
            test_loss, 100*correct/len(test_loader.dataset)))


for epoch in range(1, epochs+1):
    train_modle(model, device, train_loader, optimizer, epoch)
    test_model(model, device, test_loader)

th.save(model, "model.pt")
