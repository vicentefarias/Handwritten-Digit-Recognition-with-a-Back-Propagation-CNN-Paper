# Handwritten Digit Recognition with a Back-Propagation Network

This repository contains an implementation of a **Convolutional Neural Network (CNN)** for handwritten digit recognition, trained and evaluated on the **MNIST dataset**. The implementation is inspired by the classical **Back-Propagation Neural Network** model as applied to digit recognition.
https://proceedings.neurips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Evaluation](#evaluation)
6. [References](#references)

---

## Requirements

You need the following libraries installed to run the code:

- **Python 3.7+**
- **PyTorch**
- **Torchvision**
- **NumPy**
- **Matplotlib** (optional)

You can install all the dependencies using the following command:

```bash
pip install torch torchvision numpy matplotlib
```

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/vicentefarias/Handwritten-Digit-Recognition-with-a-Back-Propagation-CNN-Paper.git
    ```

---

## Dataset

This implementation uses the **MNIST dataset**, which consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). The dataset will be automatically downloaded when running the script, so no manual download is required.

---

## Model Architecture

The model used in this implementation is a **Convolutional Neural Network (CNN)** designed for handwritten digit recognition. The architecture consists of the following layers:

1. **Convolutional Layer 1**:
   - Input Channels: 1 (grayscale image)
   - Output Channels: 10
   - Kernel Size: 5x5
   - Activation Function: ReLU

2. **Max Pooling Layer**:
   - Size: 2x2

3. **Convolutional Layer 2**:
   - Input Channels: 10
   - Output Channels: 20
   - Kernel Size: 5x5
   - Activation Function: ReLU

4. **Dropout Layer**:
   - Applied after the second convolutional layer to prevent overfitting.

5. **Max Pooling Layer**:
   - Size: 2x2

6. **Fully Connected Layer 1**:
   - Input Features: 320
   - Output Features: 50
   - Activation Function: ReLU

7. **Dropout Layer**:
   - Applied to the output of the first fully connected layer during training.

8. **Fully Connected Output Layer**:
   - Input Features: 50
   - Output Features: 10 (for digit classes 0-9)

9. **Output Activation**:
   - Log-Softmax for multi-class classification.

## Evaluation

To evaluate the model on the test set, run the following command:

```bash
python train.py
```

## References

This implementation is inspired by the following works:

    LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
    MNIST Dataset: A dataset of handwritten digits available at MNIST.
