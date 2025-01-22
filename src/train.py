import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torchvision import datasets, transforms

from model import CNN, train_model
from lib.util import header, plot_distribution, double_plot

# CONFIG
EPOCHS = 4
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
# END CONFIG

normalization_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # params analyzed from the dataset
])

def det_device_config(train_kwargs, test_kwargs):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    # use the most powerful available device
    if use_cuda:
        print('using cuda capable device')
        device = torch.device("cuda")
    elif use_mps:
        print('using mps')
        device = torch.device("mps")
    else:
        print('using cpu')
        device = torch.device("cpu")
        
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    return device, train_kwargs, test_kwargs

if __name__ == '__main__':
    # create dicts that contain our training and testing args
    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}
    
    # device config (cuda, gpu, cpu, etc.)
    # this config may need to update the training and testing kwargs
    device, train_kwargs, test_kwargs = det_device_config(train_kwargs, test_kwargs)
    
    # create the model and attach it to the device found above
    model = CNN().to(device)
    
    # download/load data and normalize it
    header("Loading datasets...")
    train_data = datasets.MNIST('../data', train=True, download=True, transform=normalization_transform)
    test_data = datasets.MNIST('../data', train=False, transform=normalization_transform)
    print("Done loading datasets")
    
    # "When using new unfamiliar datasets, it is always a good idea to visualize the data to get a better understanding of it, we can do this using seaborn
    # Visualizing the distribution of labels in our training set to ensure they are evenly distributed" - Zain
    plot_distribution('Distribution of Labels in Training Set', train_data)
    plot_distribution('Distribution of Labels in Testing Set', test_data)
    
    # train the model
    header("Training the model...")
    train_loss, train_acc = train_model(
        model,
        device,
        data_loader=DataLoader(train_data,**train_kwargs),
        loss_func=torch.nn.CrossEntropyLoss(),
        optimizer=RMSprop(model.parameters(), lr=LEARNING_RATE),
        num_epochs=EPOCHS
    )
    print("Done training")
    
    # save the model
    header("Saving the model to file...")
    MODEL_PATH = "../mnist_cnn.pt"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    
    # plot the loss and accuracy for us to view
    double_plot(label1="Training Loss", data1=train_loss, label2="Training Accuracy", data2=train_acc)