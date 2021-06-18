import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import utils as vutils
from torchvision.datasets import CIFAR100, SVHN
from torch.nn import functional as F
from torch.autograd import Variable

from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from PIL import Image
import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import cv2

model_name = "cnn_overfitting"
LOAD = True
LR = 0.01
EPOCH = 5
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = "./data"
save_path = f"./checkpoint/{model_name}.pth"
print(f"Device: {DEVICE}")

global_epoch = EPOCH
global_lr = LR
global_batchsize = BATCH_SIZE
global_visual_num = 20

visual_choose_list = [2, 3, 5, 7, 11, 18, 19]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop1 = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 122)
        self.drop2 = nn.Dropout2d()
        self.fc2 = nn.Linear(122, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.bn1(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv3(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv4(x))), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class CNN_Overfiting(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, 122)       
        self.fc2 = nn.Linear(122, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class MyLeNet5(nn.Module):
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
    time1 = time.time()
    print("Loading Data...")
    tsfm = transforms.Compose([
        transforms.Resize(224),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset_train = SVHN(DATA_PATH, split="train", transform=tsfm, download=False)
    dataset_test = SVHN(DATA_PATH, split="test", transform=tsfm, download=False)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)

    print("Building Model...")
    net = CNN_Overfiting()
    # net = MyLeNet5()
    # net = CNN()
    # net = RealLeNet()
    # models.alexnet()
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
                    outputs = net(val_images.to(DEVICE))
                    predict_y = torch.max(outputs, dim=1)[1]  # 以output中值最大位置对应的索引（标签）作为预测输出
                    acc += (predict_y == val_labels.to(DEVICE)).sum().item()
                val_accurate = acc / val_num
                acc_list.append(val_accurate)
                print()
                print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
                    (epoch + 1, epoch_loss, val_accurate))
        print('Finished Training')
        torch.save(net.state_dict(), save_path)
        print(f"Time: {time.time() - time1}s")
    
        loss_acc = zip(loss_list, acc_list)
        dirname = "./log"
        # filename = f"cnn_{global_lr}_{global_batchsize}_{global_epoch}_" + time.strftime("%Y%m%d-%H%M%S")
        filename = f"{model_name}_" + time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(dirname, filename)
        f = open(path, 'w')
        f.write("epoch,loss,acc\n")
        for i, value in enumerate(loss_acc):
            f.write(f"{i},{value[0]},{value[1]}\n")
        f.close()

    target_layer = net.conv2
    for i in range(global_visual_num):
        # if i not in visual_choose_list:
        #     continue
        visual_img, target_category = dataset_train[i]
        img_path = f"./{model_name}/origin_{i}.jpg"
        vutils.save_image(visual_img, img_path)
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], 
                                            std=[0.5, 0.5, 0.5])

        cam = GradCAM(model=net, target_layer=target_layer, use_cuda=(DEVICE=='cuda'))
        grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category)

        grayscale_cam = grayscale_cam[0, :]
        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        cv2.imwrite(f'./{model_name}/gradcam_{i}.jpg', cam_image)