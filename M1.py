# Karl
# M1
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
class FakeDataset(data.Dataset):
    def __init__(self, size = 1000, image_size=(512,512,512), num_classes = 2, transform = None, target_transform = None, random_offset = 0):
        self.size = size
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform
        self.target_trasnform = target_transform
        self.random_offset = random_offset

    def __getitem__(self, index):

        if index >= len(self):
            raise IndexError('{} is out of range'.format(self.__class__.__name__))
            rng_state = torch.rng.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        img = torch.randn(*self.image_size)
        target = torch.randint(0, self.num_classes, size=(1,), dtype=torch.long)[0]
        torch.set_rng_state(rng_state)

        # convert to PIL Image
        img = transforms.ToPILImage()(img)
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
train_dataset = torchvision.datasets.FakeData(size=1000, image_size=(64,64,32,3), num_classes=2, transform=transforms.ToTensor(), random_offset=0)
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
#test_dataset = TestDataset()
test_dataset = FakeDataset()
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# 1. Model Architecture

class M1(nn.Module):

    def __init__(self):
        super(M1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(1,64, (5,5,3)), nn.Dropout(0.2)) # in_channel, out_channel, kernel
        self.pool = nn.MaxPool3d((2, 2, 2),2) # kernerl, stride
        self.conv2 = nn.Sequential(nn.Conv3d(64, 64, (3,3,3)), nn.Dropout(0.3)) # in_channel, out_channel, kernel
        self.conv3 = nn.Sequential(nn.Conv3d(64, 64, (3,3,1)), nn.Dropout(0.3)) # in_channel, out_channel, kernel
        self.fc1 = nn.Sequential(nn.Conv3d(64,150, (2,2,2)), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Conv3d(150,2, (1,1,1)), nn.Dropout(0.3))

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

m1 = M1()


# 2. Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(M1.parameters(), lr = learning_rate)

# 3. Training Loop

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
m1.to(device)

total_step = len(train_loader)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        #pdb.set_trace()
        images = images.to(device)
        labels = labels.to(device)

        outputs = m1(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# 4. Testing

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = m1(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# 5. Save the model checkpoint
torch.save(model.state_dict(), 'm1.ckpt')
