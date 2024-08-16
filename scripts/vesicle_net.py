import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
class VesicleNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VesicleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))  # Kernel size 1 for depth
        self.layer3 = nn.Sequential(
            nn.Conv3d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))  # Kernel size 1 for depth
        self.fc = nn.Sequential(
            nn.Linear(13888, 128),  # Adjusted input size based on printed shapes
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def create_model(in_channels, num_classes, lr, momentum):
    model = VesicleNet(in_channels=in_channels, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return model, criterion, optimizer

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")
        return start_epoch
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

# Training function with checkpointing
def train_model(model, criterion, optimizer, train_loader, start_epoch, num_epochs, checkpoint_path):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        for inputs, _, targets in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.float().unsqueeze(1)  # Adding channel dimension for grayscale
            outputs = model(inputs)
            targets = targets.long()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")