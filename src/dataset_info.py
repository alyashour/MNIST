from torchvision import transforms

normalization_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # params analyzed from the dataset
])  