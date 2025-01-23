import platform
import tkinter as tk
from tkinter import Canvas

import numpy as np
import torch
from torch.nn import functional

from PIL import Image, ImageGrab
from torchvision import datasets

from model import CNN, test_model
from train import normalization_transform

# CONFIG
DEBUG = False                           # default False. Turns on some debug screenshots for you to see.
OVERRIDE_SCALING_FACTOR = None          # default None. Change this value if you have cropping issues.
WINDOW_WIDTH = 300                      # default 300
WINDOW_HEIGHT = 300                     # default 300
WINDOW_BG_COLOR = 'white'               # default white
DRAW_COLOR = 'black'                    # default black
MINIMUM_VIABLE_MODEL_ACCURACY = 0.8     # default 0.8 (80%)

# TESTING CONFIG
TEST_BATCH_SIZE = 1000                  # default 1000

# where to read the model .pt file from
MODEL_PATH = '../../mnist_cnn.pt'   # default ../../mnist_cnn.pt
# END CONFIG

def get_scaling_factor():
    """
    Quick solution to the high-dpi retina scaling factor thing.
    Assumes everyone on macOS has a retina display and everyone not on macOS doesn't have a retina display.

    :return:
    """

    if OVERRIDE_SCALING_FACTOR:
        return OVERRIDE_SCALING_FACTOR
    elif platform.system() == 'Darwin':
        return 2
    else:
        return 1

def run_gui(model):
    # Run the GUI
    root = tk.Tk()
    root.title("MNIST DIGIT CLASSIFIER")

    canvas = Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg=WINDOW_BG_COLOR)
    canvas.pack()

    def preprocess(img):
        # resize, grayscale, and flip black and white
        img = img.resize((28, 28)).convert('L') # convert('L') converts to grayscale
        img_array = np.array(img)
        img_array = 255 - img_array # flip black and white since dataset works with black background
        img = Image.fromarray(img_array.astype(np.uint8))

        # apply the same transformation as before
        img_tensor = normalization_transform(img)
        img_tensor = img_tensor.unsqueeze(0) # add a dimension for the batch

        return img_tensor

    def paint(event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill=DRAW_COLOR, outline=DRAW_COLOR)

    def predict_digit():
        # capture canvas content
        # have to multiply by 2 because of retina display scaling ☝️🤓
        # https://pillow.readthedocs.io/en/stable/reference/ImageGrab.html
        scaling_factor = get_scaling_factor()

        x0 = scaling_factor * (root.winfo_rootx() + canvas.winfo_x())
        y0 = scaling_factor * (root.winfo_rooty() + canvas.winfo_y())
        x1 = x0 + scaling_factor * canvas.winfo_width()
        y1 = y0 + scaling_factor * canvas.winfo_height()

        img = ImageGrab.grab()

        # show the full, un-cropped screenshot if debugging
        # should show your full desktop
        if DEBUG:
            img.show(title='Full Image')

        img = img.crop((x0, y0, x1, y1))

        # show the cropped screenshot if debugging
        # should show just the drawing section of the window
        if DEBUG:
            img.show(title='Cropped Image')

        # preprocess and predict
        img_tensor = preprocess(img)
        with torch.no_grad():
            model.eval()
            prediction = model(img_tensor)
            probabilities = functional.softmax(prediction, dim=1)
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



if __name__ == "__main__":
    # Load the PyTorch model
    model = CNN()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    
    # to make sure it's loaded correctly,
    # we'll test it and ensure a high enough accuracy
    # load the test dataset
    test_kwargs = {'batch_size': TEST_BATCH_SIZE}
    test_data = datasets.MNIST('../../data', train=False, transform=normalization_transform)
    test_loader = torch.utils.data.DataLoader(test_data, **test_kwargs)

    # run the test
    accuracy = test_model(model, test_loader)

    if accuracy > MINIMUM_VIABLE_MODEL_ACCURACY:
        run_gui(model)
    else:
        print(f'Accuracy too low! Found {accuracy * 100}%, expected >{MINIMUM_VIABLE_MODEL_ACCURACY * 100}%.'
              f'Model may not have been saved or loaded successfully. Please try again...')
    