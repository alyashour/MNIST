import torch
from torch import nn
from torch.nn import functional
from torchvision import datasets

# CONFIG
EPOCHS = 4
LEARNING_RATE = 0.001
GAMMA = 0.7
TRAINING_LOG_INTERVAL = 10_000
EPOCH_BREAK_ACCURACY = .995
TEST_BATCH_SIZE = 1000
# END CONFIG

# The model architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.conv2(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
def train_model(model, device, data_loader, optimizer, num_epochs):
    # Source: Zain Syed the goat
    train_loss, train_acc = [], [] # 2 arrays to track the loss values and the accuracy of our model
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(data_loader)
        epoch_acc = correct / total

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # over-training break case
        if epoch_acc >= EPOCH_BREAK_ACCURACY: # If the model reaches the break accuracy, we stop training (usually to prevent over-fitting)
            print(f"Model has reached {EPOCH_BREAK_ACCURACY * 100}% accuracy, stopping training")
            break

    return train_loss, train_acc

def test_model(model, data_loader):
    device = torch.device('cpu')
    model.eval()
    
    test_loss = 0
    correct = 0
    
    data_len = len(data_loader.dataset)
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / data_len
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{data_len} ({100 * accuracy}%)')
    
    return accuracy
