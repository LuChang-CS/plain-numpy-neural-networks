# Multi-layer Perceptron

## THE MNIST DATABASE of handwritten digits

### Link
[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

### Files
- train-images-idx3-ubyte.gz:  training set images, 60000 images (9912422 bytes)
- train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
- t10k-images-idx3-ubyte.gz:   test set images, 10000 images (1648877 bytes)
- t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

### Features
28 × 28 images

### Category
0 - 9 digits

### Preprocess
We ramdomly select 50000 images as training set and 10000 images as validation set from the original training images.

## Experimental Setting
The settings are applicable to both numpy and pytorch models.

### Hyperparameters
- Hidden unites: [512, 256]
- Epochs: 25
- Learning rate: 1e-3

### Optimizer
Adam optimizer (D. P. Kingma and J. Ba, "[Adam: A method for stochastic optimization](https://arxiv.org/pdf/1412.6980.pdf)," in 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings, Y. Bengio and Y. LeCun, Eds., 2015.)

## Performance
![Performance](https://raw.githubusercontent.com/LuChang-CS/plain-numpy-neural-networks/master/2.%20Multi-layer%20Perceptron/figs/performance.png)
