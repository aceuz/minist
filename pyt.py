# 1加载库
import torch
import torch.nn as nn  # 网络
import torch.nn.functional as F
import torch.optim as optim  # 优化器
from torchvision import datasets, transforms
# 2超参数
BATCH_SIZE = 128  # 16 64 128批处理数据
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10  # 训练data轮次 10 20 50 100...
# 3pipeline,图像处理(transforms库函数)
pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 4加载数据
from torch.utils.data import DataLoader
 # DOWNLOAD DATA
train_set = datasets.MNIST('datamn', train=True, download=True, transform=pipeline)
test_set = datasets.MNIST('datamn', train=False, download=True, transform=pipeline)
 # 加载load the data
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# '''查看图片'''
# with open('raw/train-images-idx3-ubyte', 'rb') as f:
#     file = f.read()
# image1 = [int(str(item).encode('ascii'),16)for item in file[16 : 16+784]]
# print(image1)

# import numpy as np
# image_np = np.array(image1,dtype=np.uint8).reshape(28,28,1)
# print(image_np.shape)


# 5网络模型构建
class MODE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20*10*10, 500)  # 输入channel,输出
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        input_size = x.size(0)  # 张量形式:batch*channel*h*w b*1*28*28
        x = self.conv1(x)  # 输入 batch*1*28*28 输出:batch*10*24*24
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)  # kernel,step out:b*10*12*12
        x = self.conv2(x)  # out b*20*10*10
        x = F.relu(x)
        x = x.view(input_size, -1)  # B20*10*10=B*2000
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # shuchu:batch*10
        output = F.log_softmax(x, dim=1)  # 计算分类后每个数字的概率值,10通道dim为0-1-2....具体看张量型状
        return output

# 6定义优化器
model = MODE().to(DEVICE)
optimizer = optim.Adam(model.parameters())  # not only Adam

# 7训练方法定义
def train_model(model,device,train_loader,optimizer,epoch):
    model.train() # 训练
    for batch_index, (data, target) in enumerate(train_loader):  # data:pic traget is target
        data, target = data.to(device), target.to(device)  # 设备部署
        # set grad =0 original
        optimizer.zero_grad()
        # result of train
        output = model(data)
        # compute the loss
        loss = F.cross_entropy(output,target)
        # 反向传播
        loss.backward()
        # 参数优化更新
        optimizer.step()
        if batch_index % 3000 == 0:  # 6w 3000 20ci
            print('Train Epoch: %d  Loss:%.3f '% (epoch, loss.item()))

# 8测试方法定义
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 准确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():
        for data,target in test_loader:
            data,target = data.to(device), target.to(device)
            optput = model(data)
            test_loss += F.cross_entropy(optput,target).item()
            pred = optput.argmax(dim=1)  # 最大概率下标
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('Test__Average loss:{:.4f}, Accuracy:{:.3f}\n'.format(
            test_loss,1 00.0*correct / len(test_loader.dataset)
        ))


# 9调用方法78
for epoch in range(1, EPOCHS+1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)
