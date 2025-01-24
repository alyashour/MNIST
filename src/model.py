import torch
from torch import nn
from torch.nn import functional

# CONFIG
EPOCHS = 4 # Number of times we will be training the model on the SAME dataset
LEARNING_RATE = 0.001 # A custom value we use to how fast the model will learn, lower rates are slower but more stable, higher are fast but can have errors
# GAMMA = 0.7 # A custom value we use to decay the learning rate, it is used to prevent the model from overfitting
# TRAINING_LOG_INTERVAL = 10_000 # The interval at which we will be logging the training progress
EPOCH_BREAK_ACCURACY = .995 # The target accuracy, we stop training if the model reaches this accuracy
TEST_BATCH_SIZE = 1000 # Test batch size is the number of images we will be testing on
# END CONFIG

# We are creating a class to define the CNN model that we will be using to train the MNIST dataset
class CNN(nn.Module):
    def __init__(self): # The initialization function allows us to determine the layers and their properties, we aren't training the model here

        super(CNN, self).__init__() # The super keyword allows us to access the parent class's properties, the self keyword refers to the instance of the class

        # A kernel of 3 is simpky a 3x3 matrix that is used to specifically look at features on the 28x28 pixel image
        # The stride of 1 is how many pixels the kernel will move after reading a portion of an image
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # We are defining a convolutional layer with 1 input channel, 32 output channels (extract 32 feature maps), a kernel size of 3, and a stride of 1
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
        x = functional.relu(x) # Applying the ReLU activation function to the data
        x = self.conv2(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x, 2) # Max Pooling layers allow us to reduce the size of our image, in this case, the 2 represents reducing it by a factor of 2
        x = self.dropout1(x) # Applying the dropout layer defined above to the data
        x = torch.flatten(x, 1) # Flattening layers allows us to reduce our data from multidimensional, to 1D
        x = self.fc1(x) # Passing the data through the first fully connected layer
        x = functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x # Returning the data after passing it through the model

def train_model(model, device, data_loader, loss_func, optimizer, num_epochs=EPOCHS):
    """
    Trains some model using the given data_loader and optimizer for the given number of epochs

    Original function sourced from Zain Syed

    :param model:       The model to be trained, architecture irrelevant
    :param device:      i.e., a torch.device() object
    :param data_loader: a data loader containing the training dataset
    :param loss_func:   some loss function object instance, see torch.nn. Can also use functional API.
    :param optimizer:   some optimizer, see torch.optim
    :param num_epochs:  the number of training epochs, default=4
    :return:            None
    """

    train_loss, train_acc = [], [] # 2 arrays to track the loss values and the accuracy of our model
    for epoch in range(num_epochs): # Essentially loops based on the epoch variable we defined earlier
        runningLoss = 0.0 # Stores total loss for the current epoch
        correct = 0 # Stores the number of correct predictions
        total = 0 # Stores the total number of predictions

        for images, labels, in data_loader: # Loops through the training data, both the images and the labels associated with them, train_loader loads in batches instead of all at once
            images, labels = images.to(device), labels.to(device) # Moves the images and labels to the device we defined earlier

            optimizer.zero_grad() # We are resetting the optimizer to zero, this is because PyTorch accumulates gradients, so we need to reset them after each batch
            outputs = model(images)  # Passes inputs through the model to get their predictions
            loss = loss_func(outputs, labels) # Compares our model's guesses to actual labels to calculate loss
            loss.backward() # Simply calculates gradients, they will allow us to calculate loss
            optimizer.step() # Using our loss function to update the weights of our model in real time

            runningLoss += loss.item() # Simply updating our total loss
            _, predicted = outputs.max(1) # Getting the highest probability claass for a given image, this is going to be our guess
            correct += predicted.eq(labels).sum().item() # Counts the number of correct guesses
            total += labels.size(0) # Counts the total number of guesses

        # Calculating the loss and accuracy for the current epoch
        epoch_loss = runningLoss / len(data_loader)
        epoch_acc = correct / total

        # Appending the loss and accuracy to the arrays we defined earlier
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch_acc >= 0.995: # If the model reaches 99.5% accuracy, we stop training
            print("Model has reached 99.5% accuracy, stopping training")
            break

    return train_loss, train_acc

# Defining a function to test the model, this is done after training to see how well the model performs on unseen data
def test_model(model, data_loader, device=None):
    # we can always test on the cpu if not given an input
    if device is None:
        device = torch.device('cpu')

    model.eval() # Sets the model to evaluation mode, this is because we don't want to update the weights of the model while testing
    test_loss = 0 # Stores the total loss of the model
    correct = 0 # Stores the number of correct predictions
    
    data_len = len(data_loader.dataset)
    
    with torch.no_grad(): # Disables the gradient calculations to speed up testing
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) # Retrieving predictions of our model
            test_loss += functional.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # Checking correctness of predictions

    test_loss /= len(data_loader.dataset) #Calculating average loss
    accuracy = correct / data_len
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{data_len} ({100 * accuracy}%)')
    
    return accuracy
