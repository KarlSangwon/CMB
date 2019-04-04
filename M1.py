
#Karl CMB Project

# Karl
# M1

#Update
#TODO: Current Issue. Label and Conv Output dimension does not match. Need to map back to original dimension. (algorithm in paper)

# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pdb
#Karl CMB Project

# Karl
# M1

#Update
#TODO: Current Issue. Label and Conv Output dimension does not match. Need to map back to original dimension. (algorithm in paper)

# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pdb
# Hyperparameters
num_epochs = 5
num_classes = 2
#batch_size = 100
batch_size = 1 #for testing purpose
learning_rate = 0.03
momentum = 0.9
dropout_rate = 0.3
# 0. Loading Data

# 0.A. Creating a Fake Dataset for Experimentation

# can be imported
class FakeDataset3D(data.Dataset):
    def __init__(self, size = 1000, image_size= (1,100,100,100), num_classes = 2, transform = None, target_transform = None, random_offset = 0):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.random_offset = random_offset

    def __getitem__(self, index):

        if index >= len(self):
            raise IndexError('{} is out of range'.format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        #img = torch.randn(*self.image_size)
        img = torch.randint(0, 256, size = self.image_size)
        #target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        target = torch.randint(0, self.num_classes, size = (27,27,28), dtype = torch.long)
        #target = torch.randint(0, self.num_classes, size = self.image_size)
        torch.set_rng_state(rng_state)
​
        # convert to PIL Image
        #img = transforms.ToPILImage()(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.size

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
#currently empty

class TrainDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # Change 0 to the total size of your dataset.
        return 0

#currently empty
class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names.
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # Change 0 to the total size of your dataset.
        return 0
# Data loader.
#train_dataset = TrainDataset()
#train_dataset = torchvision.datasets.FakeData(size=100, image_size=(64,64,3,1), num_classes=2, transform=transforms.ToTensor(), target_transform =None,random_offset=0)
train_dataset = FakeDataset3D(size = 1, image_size = (1,64,64,64), num_classes=2, transform = None, target_transform = None, random_offset = 0)
print(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
for i, (images, labels) in enumerate(train_loader):
    print(images[0][0][0])
    print(labels[0][0][0])
#test_dataset = TestDataset()
test_dataset = FakeDataset3D(size = 1, image_size = (1,64,64,64), num_classes = 2, transform = None, target_transform = None, random_offset = 1)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
for i, (images, labels) in enumerate(test_loader):
    print(labels[0][0][0])

class M1(nn.Module):
    def __init__(self):
        super(M1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1,64, (5,5,3)), nn.Dropout(0.2)) # in_channel, out_channel, kernel
        self.pool = nn.MaxPool3d((2, 2, 2),2) # kernel, stride
        self.conv2 = nn.Sequential(nn.Conv3d(64, 64, (3,3,3)), nn.Dropout(0.3)) # in_channel, out_channel, kernel
        self.conv3 = nn.Sequential(nn.Conv3d(64, 64, (3,3,1)), nn.Dropout(0.3)) # in_channel, out_channel, kernel
        self.fc1 = nn.Sequential(nn.Conv3d(64,150, (2,2,2)), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Conv3d(150,2, (1,1,1)), nn.Dropout(0.3))

    def forward(self,x):
        #print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = F.relu(self.fc1(x))
        #print(x.shape)
        x = F.relu(self.fc2(x))
        #print(x.shape)
        return x

m1 = M1()
# 2. Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(m1.parameters(), lr = learning_rate)
# 3. Training Loop

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
m1.to(device)

total_step = len(train_loader)
for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = m1(images)
        print(outputs.shape)
        #pdb.set_trace()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
torch.Size([1, 2, 27, 27, 28])
Epoch [1/5], Step [1/1], Loss: 1.1786
torch.Size([1, 2, 27, 27, 28])
Epoch [2/5], Step [1/1], Loss: 0.6991
torch.Size([1, 2, 27, 27, 28])
Epoch [3/5], Step [1/1], Loss: 0.6931
torch.Size([1, 2, 27, 27, 28])
Epoch [4/5], Step [1/1], Loss: 0.6931
torch.Size([1, 2, 27, 27, 28])
Epoch [5/5], Step [1/1], Loss: 0.6931
# 4. Testing

m1.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = m1(images)
        _, predicted = torch.max(outputs.data, 1)
        nLabels = labels.size(0)*labels.size(1)*labels.size(2)*labels.size(3)
        total += nLabels
        correct += (predicted == labels).sum().item()

    print(correct)
    print(total)
    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# 5. Save the model checkpoint
#torch.save(m1.state_dict(), 'm1.ckpt')
