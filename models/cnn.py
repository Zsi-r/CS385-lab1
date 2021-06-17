import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt

LR = 0.01
EPOCH = 10
BATCH_SIZE = 10000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")


class CNN(nn.Module):
    def __init__(self, categories=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=3, padding=1),
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=6400, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=categories))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class MyDataset(Dataset):
    def __init__(self, X_train, y_train, X_test, y_test, train=False):
        super(MyDataset, self).__init__()
        if train:
            self.images, self.labels = X_train, y_train
        else:
            self.images, self.labels = X_test, y_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx, :, :, :]
        image = image.astype(np.float32)
        image = torch.from_numpy(image).div(255.0)
        label = int(self.labels[idx])
        return image, label


if __name__ == "__main__":

    print("Loading Data...")
    f1 = loadmat("../data/train_32x32.mat")
    f2 = loadmat("../data/test_32x32.mat")
    X_train, y_train = f1['X'], f1['y']
    X_train = X_train.transpose([3, 2, 0, 1])
    y_train[y_train == 10] = 0
    X_test, y_test = f2['X'], f2['y']
    X_test = X_test.transpose([3, 2, 0, 1])
    y_test[y_test == 10] = 0

    dataset_train = MyDataset(X_train, y_train, X_test, y_test, train=True)
    dataset_test = MyDataset(X_train, y_train, X_test, y_test, train=False)
    train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, num_workers=0)

    print("Building Model...")
    net = CNN()
    net.to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    train_counter = []
    train_losses = []
    train_num = len(dataset_train)
    val_num = len(dataset_test)

    print("Start training")
    for epoch in range(EPOCH):
        net.train()  # 训练过程中开启 Dropout
        running_loss = 0.0  # 每个 epoch 都会对 running_loss 清零
        time_start = time.perf_counter()  # 对训练一个 epoch 计时

        for step, data in enumerate(train_loader, start=0):  # 遍历训练集，step从0开始计算
            images, labels = data  # 获取训练集的图像和标签
            optimizer.zero_grad()  # 清除历史梯度

            outputs = net(images.to(DEVICE))  # 正向传播
            loss = loss_func(outputs, labels.to(DEVICE))  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            running_loss += loss.item()

            train_losses.append(loss.item())
            train_counter.append((step * BATCH_SIZE) + ((epoch - 1) * len(dataset_train)))

            # 打印训练进度（使训练过程可视化）
            rate = (step + 1) / len(train_loader)  # 当前进度 = 当前step / 训练一轮epoch所需总step
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print('%f s' % (time.perf_counter() - time_start))

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

            # # 保存准确率最高的那次网络参数
            # if val_accurate > best_acc:
            #     best_acc = val_accurate
            #     torch.save(net.state_dict(), save_path)

            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f \n' %
                  (epoch + 1, running_loss / epoch, val_accurate))

    print('Finished Training')

    # 打印训练过程中的loss变化情况

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.legend('Train Loss', loc='upper right')
    plt.xlabel('number of training examples')
    plt.ylabel('loss')
    plt.show()
