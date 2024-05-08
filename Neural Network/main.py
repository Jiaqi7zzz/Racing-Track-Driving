import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

class DrivingNet(nn.Module):
    def __init__(self):
        super(DrivingNet, self).__init__()
        self.fc1 = nn.Linear(4 , 100 , dtype = torch.float64)
        self.fc2 = nn.Linear(100, 500 , dtype = torch.float64)
        self.fc3 = nn.Linear(500, 3 , dtype = torch.float64)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    

net = DrivingNet()
# criterion = nn.L1Loss()
criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

data = pd.read_csv('./data/training_set.csv')
data = data.to_numpy()

x = data[:, 0]
y = data[:, 1]
left = data[:, 2]
right = data[:, 3]
v = data[:, 4]
drive = data[:, 5]
brake = data[:, 6]

# sum = [np.sum(data[:, 4]), np.data[:, 5], np.data[:, 6]]
# for i in range(len(data)):
#     data[i, 4] = data[i, 4] / sum[0]
#     data[i, 5] = data[i, 5] / sum[1]
#     data[i, 6] = data[i, 6] / sum[2]

# data[:, 4] = data[:, 4] / np.sum(data[:, 4])
# data[:, 5] = data[:, 5] / np.sum(data[:, 5])
# data[:, 6] = data[:, 6] / np.sum(data[:, 6])

Input = data[:, 0: 4]
Labels = data[:, 4:]
bs = 10

class Data(Dataset):
    def __init__(self, Input, Labels):
        self.Input = Input
        self.Labels = Labels

    def __len__(self):
        return len(self.Input)
    
    def __getitem__(self, index):
        # sample = np.concatenate([self.Input[index], self.Input[index + 1]]) if index < len(self.Input) - 1 else np.concatenate([self.Input[index], self.Input[0]])
        sample = self.Input[index]
        # sample = torch.cat((self.Input[index], self.Input[index + 1]))
        label = self.Labels[index]
        return sample, label
    
train_set, test_set, train_label, test_label = train_test_split(Input, Labels, test_size = 0.2, random_state = 42)
train_dataset = Data(train_set, train_label)
test_dataset = Data(test_set, test_label)
train_dataloader = DataLoader(train_dataset, batch_size = bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = bs, shuffle=True)

for batch in train_dataloader:
    inputs, labels = batch
for batch in test_dataloader:
    inputs, labels = batch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():

    Epochs = 10000
    avg_loss = 0.0
    loss_history = []

    net.to(device)
    for epoch in range(Epochs):
        epoch_loss = 0.0
        for _, (inputs, labels) in enumerate(train_dataloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss += epoch_loss / len(train_dataloader)
        if not (epoch + 1) % 50:
            print(f"epoch is {epoch + 1} , loss is {epoch_loss:.10f}")

        # loss_history.append(avg_loss)
        loss_history.append(epoch_loss)

    plt.plot(loss_history)
    plt.show()

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(test_dataloader, 0):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            for i in range(bs):
                if abs(labels[i][0].cpu() - outputs[i][0].cpu()) < 2 and abs(labels[i][1].cpu() - outputs[i][1].cpu()) < 10 and abs(labels[i][2].cpu() - outputs[i][2].cpu()) < 10:
                    correct += 1
                total += 1
        acc = correct / total

    print(f"accuracy = {100 * acc}%")

def check():
    input = torch.tensor([216.0184389, 5.187479855, 4.223997415, 5.62814563], dtype = torch.float64)
    input = input.to("cpu")
    output = net(input)
    print(output.item())


if "__main__" == __name__:
    train()
    test()
    check()