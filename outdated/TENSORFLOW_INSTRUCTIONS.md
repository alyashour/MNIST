# Instructions
## 1. Setup your Environment
- [ ] make sure you have python installed. 
  - Any relatively modern version should work fine. I used Python 3.12
- [ ] install tensorflow, numpy, and matplotlib

The second part of this project, the app, includes a GUI.
We make this using Tkinter. Tkinter does not have to be installed - it is a part of your python installation by default.

If, however, you installed python using homebrew on macOS (or any package manager on Linux or Windows I'm pretty sure) verify that it has tcl/tk support by running:
```shell
brew info python
```

If your python does NOT include tcl/tk you'll need to install it separately.
### MacOS
This is how I did it on my Mac. If you're running choco on Windows or apt/yum/etc.
```shell
brew install tcl-tk

# then
export PATH="/usr/local/opt/tcl-tk/bin:$PATH"
export LDFLAGS="-L/usr/local/opt/tcl-tk/lib"
export CPPFLAGS="-I/usr/local/opt/tcl-tk/include"
export PKG_CONFIG_PATH="/usr/local/opt/tcl-tk/lib/pkgconfig"

# and if you're on zsh like i am
source ~/.zshrc
```

### Linux
I have not tested this, but you should be able to follow these instructions:
https://www.geeksforgeeks.org/how-to-install-tkinter-on-linux/

### Testing Tkinter
Before continuing you can test that your installation works correctly by opening a python REPL in whatever venv or path you have python setup and running:
```python
import tkinter
print(tkinter.TkVersion)
```
If you get `python package not found` or `environment is not configured for tkinter` please take a look at the above instructions.


## 2. Testing our env, interpreter, and creating main
- [ ] create a new python file called `main.py` or `cnn.py` or something along those lines.
  - It will contain the code needed to define our cnn (or convolutional neural network, more on that later), download the dataset, then train, and test our finished trained model.
- [ ] start by ensuring tensorflow (tf) is installed.
  - Tensorflow is a massive package - it makes creating, training, and testing deep learning models faster.
  - It is written in C++ with APIs for python, C++, Java, etc. so if you feel like doing it in those languages feel free, your models can transfer at any time.
  - There are alternatives to tensorflow (like pytorch) that you are free to use, the process is almost identical to how tf handles it. Consider this an exercise in learning how these deep models work instead of how tf specifically works. There is much more complexity that I have not touched on at all here.
  - You can test this by adding the following code to your file:
    - If you're not familiar with python I can explain what any of the following code in any step just ask.
```python
import tensorflow
print(tensorflow.__version__)
```
- [ ] get the mnist dataset from tensorflow.
  - This is actually a subset of the new dataset which is actually a merge of 2 old datasets but blah blah
  - [keras datasets documentation](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)
  - hint use the load data function
# 3. Preprocessing
We have to preprocess our data. Why?
Because deep learning and ml models generally are only good at picking up patterns if we make those patterns the only variation between the input data.

What is the format of our data in now?
What format does it need to be in?
### 3.1 Normalization
What should we normalize?
- [x] The size of the images. Reshape every image to be the same size. (this is already done for us by the dataset to be 28x28 pixels)
- [ ] The values in the encoding of the images. Black-white pixels are encoded with a byte each so each pixel has a value between 0 and 255. BUT models work better with values that are *normalized* between 0 and 1 (or really any range) for [reasons](https://stackoverflow.com/questions/48284427/why-should-we-normalize-data-for-deep-learning-in-keras) and [more reasons](https://www.datacamp.com/tutorial/normalization-in-machine-learning)
- 
### 3.2 Label Encoding
A machine learning model does not know what a car or a bike or a *1* or a *2* is.
So we have to be clever with our output representation.

For categorical data we can encode what the output categories is in a bunch of ways.
Since our categorical data here is already numerical why don't we just use that?
- because we are learning how numbers look and how they look and their numerical value has no relation.
- if we just ask the model to output a number and round that number then inside it has to determine a function that takes in the input and gives a bigger answer for certain shapes and a smaller answer for smaller shapes.
- this is not correct. I.e., even though 2 < 3 is true *the shape of 2* is not < *the shape of 3* so we have to encode in a different way.

In this project we'll use one-hot encodings. Basically we have the model output a vector instead of a single number, and before we train it we convert our labels into vectors.
I.e.,
$0 \rightarrow [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  $
$1 \rightarrow [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  $
$2 \rightarrow [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  $
$3 \rightarrow [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  $  
$...  $  
$8 \rightarrow [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]  $
$9 \rightarrow [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  $

This way every output is separate and independent and follows a relatively separate calculation in the model.
It is also directly related to the number of output neurons.
So the next step is...

- [ ] one-hot encode the outputs

<details>
<summary>Solution</summary>
<pre>
<code class="language-python">
train_labels = tf.one_hot(train_labels.astype(np.int32), depth=10)  <br/>
test_labels = tf.one_hot(test_labels.astype(np.int32), depth=10)  
</code>
</pre>
</details>

## 4. OPTIONAL: Visualize the Input Data
```python
index = random.randint(0, train_images.shape[0] - 1)
plt.imshow(train_images[index][:,:,0])
plt.title("Sample Image")
plt.show()
```

# 5. Build the model
- [ ] define hyperparameters batch size, the number of classes, and the number of epochs.
  - What's a "batch"?
    - The amount of data we pass through each forward and backward pass.
  - What's a forward and backwards pass?
    - ...
  - Why not all at once?
    - Memory
    - Noise (is good)
    - Parallelism (also good)
  - What is a class?
    - In categorical models these are the types of things. Here the shapes of the numbers 0-9.
  - What is an epoch?
    - Passing the whole dataset through the model
  - Why use epochs?
    - More epochs gives the model multiple tries to learn the dataset effectively.
      - Up to a point. Cannot learn forever. We'll see after we build our own.
    - Monitor performance
      - Lets engineers observe, tweak, and split training into sessions.
  - Stopping criteria...
    - We can stop epochs based on some criteria to prevent overfitting.
    - Here we'll set one of accuracy on the test data of >99.5%.
  - What is overfitting?
    - ...
- [ ] Define the layers.
  - Just use these. Picking layers is very hard, and I'm still understanding it myself.
```python
layers = tf.keras.layers
model = tf.keras.models.Sequential([
    layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=input_shape),
    layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    layers.MaxPool2D(),
    layers.Dropout(0.25),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPool2D(strides=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```