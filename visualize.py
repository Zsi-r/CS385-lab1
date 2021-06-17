import csv
import matplotlib.pyplot as plt

f = open("./log/ridge20210617-174342", 'r')
data = list(csv.reader(f))
length = len(data)

x = list()
y = list()
z = list()

for i in range(1, length):
    x.append(int(data[i][0]))
    y.append(float(data[i][1]))
    z.append(float(data[i][2]))

fig = plt.figure(1, dpi=128, figsize=(10, 6))
plt.plot(x, y)
plt.title("Training Loss in Ridge Regression (epoch=300, lr=0.1)")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.savefig('./ridge_loss.png')

fig2 = plt.figure(2, dpi=128, figsize=(10, 6))
plt.plot(x, z)
plt.title("Test Accuracy in Ridge Regression (epoch=300, lr=0.1)")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy rate", fontsize=16)
plt.savefig('./ridge_acc.png')
