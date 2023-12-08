import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 color channels, 6 feature maps to output, and a 5x5 kernel. Stride is 1 for the kernel.
        self.pool = nn.MaxPool2d(2, 2) # Pooling filter is 2x2, and a stride of 2.
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input feature maps from conv1 + pool, 16 feature maps to output, 5x5 kernel. Stride of 1.
        self.fc1 = nn.Linear(16*5*5, 120) # 16 for the output layer * the resulting 5x5 image after pooling
        self.fc2 = nn.Linear(120, 84) # 120 input layer, 84 output
        self.fc3 = nn.Linear(84, 10) # 84 input layer, 10 output (10 classes)

        # Formula applied for conv = (Dimension - Filter)/Stride + 1


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # Flattening, because Linear only takes a 1D array
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)    # Don't apply ReLU here, nor softmax. CrossEntropyLoss() applies softmax automatically, and requires raw data.
        return x
def run(X_tr, y_tr, X_te, y_te):
    num_epochs = 5
    batch_size = 5
    learning_rate = 0.001

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # First, converting our data into tensors
    X_tr_transposed = X_tr.transpose((3, 2, 0, 1))

    X_tr_tensor = torch.from_numpy(X_tr_transposed).float().to(device)
    y_tr_tensor = torch.from_numpy(y_tr).long().to(device)

    # Converting our tensors into a Tensor Dataset
    train_dataset = torch.utils.data.TensorDataset(X_tr_tensor, y_tr_tensor)

    # Creating a loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training the model
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            inputs, labels = batch
            labels = labels.squeeze()
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # Zeroing gradient
            optimizer.zero_grad()

            # Forward pass
            # (batch_size, 3, 32, 32)
            # (32, 32, 3, batch_size)
            outputs = model(inputs)

            # Applying loss
            loss = criterion(outputs, labels)

            # Backpropogation
            loss.backward()

            # Update our weights
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}")
            
    # Testing the model
    X_te_transposed = X_te.transpose((3, 2, 0, 1))

    X_te_tensor = torch.from_numpy(X_te_transposed).float().to(device)
    y_te_tensor = torch.from_numpy(y_te).long().to(device)

    # Converting our tensors into a Tensor Dataset
    test_dataset = torch.utils.data.TensorDataset(X_te_tensor, y_te_tensor)

    # Creating a loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    print("testing the model...")
    # Testing the model
    X_te_transposed = X_te.transpose((3, 2, 0, 1))

    X_te_tensor = torch.from_numpy(X_te_transposed).float().to(device)
    y_te_tensor = torch.from_numpy(y_te).long().to(device)

    # Converting our tensors into a Tensor Dataset
    test_dataset = torch.utils.data.TensorDataset(X_te_tensor, y_te_tensor)

    # Creating a loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("testing the model...")
    correct_count, all_count = 0, 0
    for images,labels in test_loader:
        for i in range(len(labels)):
            img = images[i].view(1, 3, 32, 32)
            with torch.no_grad():
                logps = model(img)

        
        ps = torch.exp(logps)
        probab = list(ps.cpu()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.cpu()[i]
        if(true_label == pred_label):
            correct_count += 1
        all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))