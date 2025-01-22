# Importing relevant libraries for MNIST Training and GUI usage
from tkinter import Canvas
import tkinter as tk
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageGrab
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import seaborn
import matplotlib.pyplot as plt

# Constants that will be used for training
BATCH_SIZE = 64 # Batch size entails how many images we will be inputting into the model at once, they group together images to be trained on
TEST_BATCH_SIZE = 1000 # Test batch size is the number of images we will be testing on
EPOCHS = 4 # Number of times we will be training the model on the SAME dataset
LEARNING_RATE = 0.001 # A custom value we use to how fast the model will learn, lower rates are slower but more stable, higher are fast but can have errors
GAMMA = 0.7 # A custom value we use to decay the learning rate, it is used to prevent the model from overfitting
TRAINING_LOG_INTERVAL = 10_000 # The interval at which we will be logging the training progress
DO_SAVE_MODEL = True # A boolean value that will determine if we will save the model after training

# We are creating a class to define the CNN model that we will be using to train the MNIST dataset
class CNN(nn.Module):

    def __init__(self): # The initialization function allows us to determine the layers and their properties, we aren't training the model here

        super(CNN, self).__init__() # The super keyword allows us to access the parent class's properties, the self keyword refers to the instance of the class

        # A kernel of 3 is simpky a 3x3 matrix that is used to specifically look at features on the 28x28 pixel image
        # The stride of 1 is how many pixels the kernel will move after reading a portion of an image
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # We are defining a convolutional layer with 1 input channel, 32 output channels, a kernel size of 3, and a stride of 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1) # 32 inputs resulting from the previous 32 outputs, and 64 outputs, kernel size of 3, and stride of 1

        # The dropout layer is used to prevent overfitting, it randomly and temporarily disables a percentage of neurons while training
        self.dropout1 = nn.Dropout(0.25) # Drops 25% of Neurons
        self.dropout2 = nn.Dropout(0.5) # Drops 50% of Neurons

        # Fully connected layers are used to condense the size of the data, we can do this by connecting every input layer to every output layer
        self.fc1 = nn.Linear(9216, 128) # We are expected to have 9216 neurons, and we will connect them to 128 neurons
        self.fc2 = nn.Linear(128, 10) # After a couple more functions, we can then connect the 128 neurons to 10 neurons, which will be our output layer

    def forward(self, x): # The forward function is where we define the flow of the data through the model, we use the defined layers and create a flow of data
        x = self.conv1(x) # Passing data through the first convolutional layer we defined above
        # Rectified Linear Unit (ReLU) is an activation function that is used to introduce non-linearity to the model, it is used to make the model more flexible
        # ReLU is the most widely-used activation function, it is differentiable at all points except for x=0, it is computationally efficient
        x = F.relu(x) # Applying the ReLU activation function to the data
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2) # Max Pooling layers allow us to reduce the size of our image, in this case, the 2 represents reducing it by a factor of 2
        x = self.dropout1(x) # Applying the dropout layer defined above to the data
        x = torch.flatten(x, 1) # Flattening layers allows us to reduce our data from multidimensional, to 1D
        x = self.fc1(x) # Passing the data through the first fully connected layer
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x # Returning the data after passing it through the model

transform=transforms.Compose([
    transforms.ToTensor(), # Converts the image to a tensor, which is a multi-dimensional array, because we cannot train with PyTorch using an standard image
    transforms.Normalize((0.1307,), (0.3081,)) # Normalizing the data allows us to scale the data to a range of 0 to 1, which makes it easier to train
    # The normalization equation is: normalized = (x - mean) / standard deviation of MNIST pixels. Normalization makes training much faster and more stable
    # The x in the equation represents the grayscale pixel value on a scale of 0-1
])

def train(model, device, train_loader, optimizer, epochs): # This is the code where we define how our model will be trained

    train_loss, train_acc = [], [] # 2 arrays to track the loss values and the accuracy of our model
    for epoch in range(epochs): # Essentially loops based on the epoch variable we defined earlier
        runningLoss = 0.0 # Stores total loss for the current epoch
        correct = 0 # Stores the number of correct predictions
        total = 0 # Stores the total number of predictions

        for images, labels, in train_loader: # Loops through the training data, both the images and the labels associated with them, train_loader loads in batches instead of all at once
            images, labels = images.to(device), labels.to(device) # Moves the images and labels to the device we defined earlier

            optimizer.zero_grad() # We are resetting the optimizer to zero, this is because PyTorch accumulates gradients, so we need to reset them after each batch
            outputs = model(images)  # Passes inputs through the model to get their predictions
            loss = F.cross_entropy(outputs, labels) # Compares our model's guesses to actual labels to calculate loss
            loss.backward() # Simply calculates gradients, they will allow us to calculate loss
            optimizer.step() # Using our loss function to update the weights of our model in real time

            runningLoss += loss.item() # Simply updating our total loss
            _, predicted = outputs.max(1) # Getting the highest probability claass for a given image, this is going to be our guess
            correct += predicted.eq(labels).sum().item() # Counts the number of correct guesses
            total += labels.size(0) # Counts the total number of guesses

        # Calculating the loss and accuracy for the current epoch
        epoch_loss = runningLoss / len(train_loader)
        epoch_acc = correct / total

        # Appending the loss and accuracy to the arrays we defined earlier
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch_acc >= 0.995: # If the model reaches 99.5% accuracy, we stop training
            print("Model has reached 99.5% accuracy, stopping training")
            break

    return train_loss, train_acc

# Defining a function to test the model
def test(model, device, test_loader):
    model.eval() # Sets the model to evaluation mode, this is because we don't want to update the weights of the model while testing
    test_loss = 0 # Stores the total loss of the model
    correct = 0 # Stores the number of correct predictions

    with torch.no_grad(): # Disables the gradient calculation to speed up testing
        for data, target in test_loader: 
            data, target = data.to(device), target.to(device)
            output = model(data) # Getting the predictions of the model
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # Checking if the model's predictions match the actual labels

    test_loss /= len(test_loader.dataset) # Calculating the average loss

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset))) 

# The following code takes all our predefined functions and parameters and finally trains the model
def train_model():
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    torch.manual_seed(5)

    # Choosing processor based on availability and highest training speed
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
                              transform=transform) # Retreiving Training Data
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform) # Retreiving Testing Data

    # When using new unfamiliar datasets, it is always a good idea to visualize the data to get a better understanding of it, we can do this using seaborn
    # Visualizing the distribution of labels in our training set to ensure they are evenly distributed
    seaborn.countplot(x=numpy.array(dataset1.targets))
    plt.title('Distribution of Labels in Training Set')
    plt.show() # Notice how the labels are somewhat evenly distributed, but 1 and 7 in partifular have more inputs, why is that?

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE) # Defining loss function (and in turn the optimizer) to be used in training

    train_loss, train_acc = train(model, device, train_loader, optimizer, EPOCHS)

    # Plotting the training loss and accuracy
    fig, ax = plt.subplots(2,1)
    ax[0].plot(train_loss, color='b', label="Training Loss")
    ax[0].legend(loc='best', shadow=True)
    ax[0].set_title("Training Loss Curve")

    ax[1].plot(train_acc, color='r', label="Training Accuracy")
    ax[1].legend(loc='best', shadow=True)
    ax[1].set_title("Training Accuracy Curve")

    plt.tight_layout()
    plt.show()

    # Saving our model as a .pt file
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

        return img_tensor

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')

    def predict_digit():
        # capture canvas content
        # have to multiply by 2 because of retina display scaling ☝️🤓
        # https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
        scaling_factor = 2
        x0 = scaling_factor * (root.winfo_rootx() + canvas.winfo_x())
        y0 = scaling_factor * (root.winfo_rooty() + canvas.winfo_y())
        x1 = x0 + scaling_factor * canvas.winfo_width()
        y1 = y0 + scaling_factor * canvas.winfo_height()

        img = ImageGrab.grab()
        img = img.crop((x0, y0, x1, y1))
        img.show()

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