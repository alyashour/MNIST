from tkinter import Canvas
import tkinter as tk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageGrab
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 4
LEARNING_RATE = 0.001
GAMMA = 0.7
TRAINING_LOG_INTERVAL = 10_000
DO_SAVE_MODEL = True

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
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % TRAINING_LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train_model():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    torch.manual_seed(5)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)

    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA)
    for epoch in range(1, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if DO_SAVE_MODEL:
        torch.save(model.state_dict(), "mnist_cnn.pt")

# GUI
def run_gui():
    # Load the PyTorch model
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pt', weights_only=True))
    model.eval()

    # test the model before we run the GUI
    device = torch.device("cpu")
    test_dataset = datasets.MNIST('../data', train=False,
                              transform=transform)
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    test(model, device, test_loader)

    # Run the GUI
    root = tk.Tk()
    root.title("MNIST DIGIT CLASSIFIER")

    canvas = Canvas(root, width=300, height=300, bg='white')
    canvas.pack()

    def preprocess(img):
        # resize, grayscale, and flip black and white
        img = img.resize((28, 28)).convert('L')
        img_array = np.array(img)
        img_array = 255 - img_array
        img = Image.fromarray(img_array.astype(np.uint8))

        # apply the same transformation as before
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0) # add a dimension for the batch

        # debug
        # img.show()

        return img_tensor

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    def predict_digit():
        # capture canvas content
        # have to multiply by 2 because of retina display scaling ☝️🤓
        # https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
        x0 = 2 * (root.winfo_rootx() + canvas.winfo_x())
        y0 = 2 * (root.winfo_rooty() + canvas.winfo_y())
        x1 = x0 + 2 * canvas.winfo_width()
        y1 = y0 + 2 * canvas.winfo_height()

        img = ImageGrab.grab()
        img = img.crop((x0, y0, x1, y1))

        # preprocess and predict
        img_tensor = preprocess(img)
        with torch.no_grad():
            model.eval()
            prediction = model(img_tensor)
            probabilities = F.softmax(prediction, dim=1)
            digit = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()

        # display result
        result_label.config(text=f'Prediction: {digit}\nConfidence: {confidence:.2f}')

    def clear_canvas():
        canvas.delete('all')

    canvas.bind("<B1-Motion>", paint)

    btn_predict = tk.Button(root, text="Predict", command=predict_digit)
    btn_predict.pack()

    btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
    btn_clear.pack()

    result_label = tk.Label(root, text="", font=("Helvetica", 16))
    result_label.pack()

    root.mainloop()

if __name__ == '__main__':
    train_model()
    run_gui()