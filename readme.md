# Classifying Handwritten Digits: 1 vs. 5

|Example Digit 1      | Example Digit 5     |
|---------------------|---------------------|
|![](./pic/dig_1.png) | ![](./pic/dig_5.png)|

## Preprocessing
- Filtered the original dataset to contation only digits 1 and 5
- Set +1 as the label for a digit 1 and -1 for a digit 5


## Methods
- Defined two features: symmetry and density

![Correlation Between Two Features](./pic/features.png){:height="300px"}

- Linear Regression

![Linear Regression](./pic/LR.png){:height="300px"}

- Linear Regression with Kernel and regration $\lambda = 0.0774$

![Linear Regression with Kernel](./pic/LRwK.png){:height="300px"}

- k-NN

![kNN](./pic/knn.png){:height="300px"}

- RBF-network with Gaussian kernel, $r = 2/\sqrt{k}$

![RBF](./pic/rb.png){:height="300px"}

- Neural Network

![Neural Network](./pic/nn.png){:height="300px"}


- SVM

![SVM](./pic/svm.png){:height="300px"}
