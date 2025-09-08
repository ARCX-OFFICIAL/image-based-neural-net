import torch_directml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

for i in range(2):
    dml = torch_directml.device()

    noepoch = 4

    inputTransforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_path = path/to/your/database"
    test_path = "path/to/your/database"

    class ASLModel(nn.Module):
        def __init__(self):
            super(ASLModel, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 26)
            )
        def forward(self, x):
            return self.model(x)
        
        print('looks like the model got defined')
        
    model = ASLModel().to(dml)

    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(noepoch):
        print('training on epoch', epoch+1, 'begun')
        model.train()
        runningLoss = 0

        for images, labels in DataLoader(datasets.ImageFolder(train_path, transform=inputTransforms), batch_size=64, shuffle=True):
            images = images.to(dml)
            labels = labels.to(dml)
            opt.zero_grad()
            out = model(images)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            runningLoss += loss.item()
        print(f"epoch {epoch+1}, loss: {runningLoss/len(DataLoader(datasets.ImageFolder(train_path, transform=transforms), batch_size=64, shuffle=True))}")

    def test():
        model.eval()
        correct = 0
        incorrect = 0
        with torch.no_grad():
            for images, labels in DataLoader(datasets.ImageFolder(test_path, transform=inputTransforms), batch_size=64):
                images = images.to(dml)
                labels = labels.to(dml)
                out = model(images)
                _, preds = torch.max(out, 1)
                correct += (preds == labels).sum().item()
                incorrect += (preds != labels).sum().item()

                print(f'correct: {correct}, incorrect: {incorrect}, percent: {(correct/(correct+incorrect)) * 100}')

    test()

