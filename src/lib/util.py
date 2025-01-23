import numpy
import seaborn
import matplotlib.pyplot as plt
from torchvision import transforms

normalization_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # params analyzed from the dataset
])

def header(text):
    print(f'\n| {text} |')
    print('========================')

def double_plot(label1, data1, label2, data2):
    fig, ax = plt.subplots(2,1)
    ax[0].plot(data1, color='b', label=label1)
    ax[0].legend(loc='best', shadow=True)
    ax[0].set_title(f"{label1} Curve")

    ax[1].plot(data2, color='r', label=label2)
    ax[1].legend(loc='best', shadow=True)
    ax[1].set_title(f"{label2} Curve")
    plt.tight_layout()
    plt.show()

def plot_distribution(title: str, data):
    """
    Plots a countplot (simple bargraph)
    :param title:   the title of the plot
    :param data:    the data to plot. This should be some kind of array type.
    Source: Zain Syed
    """

    seaborn.countplot(x=numpy.array(data.targets))
    plt.title(title)
    plt.show()