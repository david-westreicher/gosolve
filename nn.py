import numpy as np
import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from vis import Vis

# print(calc_norm(trainloader))
DATASET_MEAN = 140.50954406211468
DATASET_STD = 63.4983134178766

def loader(img_path):
    img = Image.open(img_path)
    return img

def calc_norm(loader):
    all_images = []
    for img, _ in loader:
        all_images = np.concatenate((all_images, img.flatten()))
    mean = np.mean(all_images)
    std = np.std(all_images)
    return mean, std

unnormalize = lambda x: x * DATASET_STD + DATASET_MEAN
normalize = lambda x: (x - DATASET_MEAN) / DATASET_STD

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(net, trainloader, epochs):
    net.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    print('train')
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()

            outputs = net(inputs.float().cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

def test(net):
    net.eval()
    correct = 0
    total = 0
    classifier = Classifier()
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.float().cuda())
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()

            for image, p in zip(images, predicted):
                c_predict = classifier.predict(unnormalize(image))
                p2 = classifier.classes.index(c_predict)
                assert p2 == int(p), (c_predict, p2, p)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %s %%' % (
        100.0 * correct / total))
    print(correct, total)

class Classifier:
    def __init__(self):
        print(torch.__version__)
        self.classes = ['black', 'empty', 'white']
        self.net = Net()
        state = torch.load('train/model.pth.tar')
        self.net.load_state_dict(state)

    def predict(self, img):
        img = torch.from_numpy(np.asarray([normalize(np.asarray(img))])).float()
        pred = self.net(img)
        _, pred = torch.max(pred.data, 1)
        pred = int(pred)
        classs = self.classes[pred]
        return classs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(args)

    vis = Vis(unnormalize)
    transform = transforms.Compose([
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(0, [0.2, 0.2]),
        lambda x: np.asarray(x),
        normalize
    ])
    all_dataset = datasets.DatasetFolder('train', loader, ['png'], transform)
    trainset, testset = torch.utils.data.dataset.random_split(all_dataset, [len(all_dataset) - 500, 500])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    vis.showbatch(next(iter(trainloader))[0])

    net = Net().cuda()
    if args.train:
        train(net, trainloader, args.epochs)
        print('Finished Training')
        torch.save(net.state_dict(), 'train/model.pth.tar')
    state = torch.load('train/model.pth.tar')
    net.load_state_dict(state)
    test(net)
