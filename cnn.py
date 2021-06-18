import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, SVHN
from torch.nn import functional as F
from torch.autograd import Variable

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from PIL import Image
import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os

LOAD = True
LR = 0.01
EPOCH = 5
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = "./data"
save_path = "./checkpoint/cnn.pth"
print(f"Device: {DEVICE}")

global_epoch = EPOCH
global_lr = LR
global_batchsize = BATCH_SIZE

class CNN(nn.Module):
    def __init__(self, categories=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 89, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(89, 122, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(122 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, categories),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LeNet5(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=4, padding=2)
       self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
       self.fc1 = nn.Linear(2304, 1000)       
       self.fc2 = nn.Linear(1000, 120)
       self.fc3 = nn.Linear(120, 64)
       self.fc4 = nn.Linear(64, 10)
   def forward(self, x):
       x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
       x = x.view(-1, self.num_flat_features(x))
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = F.relu(self.fc3(x))
       x = self.fc4(x)
       return x
   def num_flat_features(self, x):
       size = x.size()[1:]
       num_features = 1
       for s in size:
           num_features *= s
       return num_features


if __name__ == "__main__":

    print("Loading Data...")
    tsfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    dataset_train = SVHN(DATA_PATH, split="train", transform=tsfm, download=False)
    dataset_test = SVHN(DATA_PATH, split="test", transform=tsfm, download=False)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)

    print("Building Model...")
    # net = CNN(categories=100)
    # net = models.alexnet(pretrained=False)
    # net.classifier[6] = nn.Linear(4096, 100)
    net = LeNet5()
    print(net)
    net.to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

    loss_list = []
    acc_list = []
    train_num = len(dataset_train)
    val_num = len(dataset_test)

    if LOAD:
        net.load_state_dict(torch.load(save_path))
    else:
        print("Start training")
        for epoch in range(EPOCH):
            net.train()  # 训练过程中开启 Dropout
            epoch_loss = 0.0  # 每个 epoch 都会对 running_loss 清零
            for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
                images, labels = data  # 获取训练集的图像和标签
                optimizer.zero_grad()  # 清除历史梯度

                outputs = net(images.to(DEVICE))  # 正向传播
                loss = loss_func(outputs, labels.to(DEVICE))  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 优化器更新参数
                epoch_loss += loss.item()

                # 打印训练进度（使训练过程可视化）
                rate = (step + 1) / len(train_loader)  # 当前进度 = 当前step / 训练一轮epoch所需总step
                a = "*" * int(rate * 50)
                b = "." * int((1 - rate) * 50)
                print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
            epoch_loss /= len(train_loader)
            loss_list.append(epoch_loss)

            ########################################## validate ###############################################
            net.eval()  # 验证过程中关闭 Dropout
            acc = 0.0
            with torch.no_grad():
                for val_data in test_loader:
                    val_images, val_labels = val_data
                    # print(val_labels)
                    # print("mark1")
                    outputs = net(val_images.to(DEVICE))
                    # print(f"-----{outputs.shape}")
                    # print(outputs)
                    # print("mark2")
                    predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                    acc += (predict_y == val_labels.to(DEVICE)).sum().item()
                val_accurate = acc / val_num
                acc_list.append(val_accurate)
                print()
                print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
                    (epoch + 1, epoch_loss, val_accurate))
        print('Finished Training')
        torch.save(net.state_dict(), save_path)
    
        loss_acc = zip(loss_list, acc_list)
        dirname = "./log"
        filename = f"cnn_{global_lr}_{global_batchsize}_{global_epoch}_" + time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(dirname, filename)
        f = open(path, 'w')
        f.write("epoch,loss,acc\n")
        for i, value in enumerate(loss_acc):
            f.write(f"{i},{value[0]},{value[1]}\n")
        f.close()

    # 可视化
    target_layer = net.fc4
    input_tensor, target_category = test_loader[0]
    cam = GradCAM(model=net, target_layer=target_layer, use_cuda=(DEVICE=='cuda'))
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(input_tensor.numpy(), grayscale_cam)

    # 打印训练过程中的loss变化情况

    # epoch_list = list(range(EPOCH))
    # fig = plt.figure()
    # plt.plot(epoch_list, loss_list, color='blue')
    # plt.legend('Train Loss', loc='upper right')
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.savefig("./cnn_loss")
    # plt.show()
    # fig2 = plt.figure()
    # plt.plot(epoch_list, acc_list, color='red')
    # plt.legend('Train Loss', loc='upper right')
    # plt.xlabel('Epoch')
    # plt.ylabel('Testing Accuracy')
    # plt.savefig("./cnn_acc")
    # plt.show()
