import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import resnet


data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

face_dataset = datasets.ImageFolder(root='../data/train/',
                        transform=data_transform)


dataset_loader = torch.utils.data.DataLoader(face_dataset,
                                             batch_size=20, shuffle=True,
                                             num_workers=4)

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
momentum = 0.5
seed = 1
log_interval = 100
torch.manual_seed(seed)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


model = resnet.resnet50(num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train(model, device, train_loder, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loder):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*len(data), len(train_loder.dataset), 100. * batch_idx / len(train_loder), loss.item()))

for epoch in range(1, epochs + 1):
        train(model, device, dataset_loader, optimizer, epoch)

torch.save(model, "../models/model.pth")