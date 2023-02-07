# pytorch_neural_network
Build different types of neural network with Pytorch 


## Types of neural network

Fully Connected Neural Network

A fully connected neural network (also known as a dense neural network or a multi-layer perceptron) is a type of artificial neural network where each neuron in one layer is connected to all neurons in the subsequent layer. The input is passed through multiple layers of computations, where each layer applies a set of weights and biases to the inputs and then applies an activation function to produce an output. The final layer of the network gives the output, which can be interpreted as a prediction for the desired target. The parameters of the network (weights and biases) are learned during the training process by optimizing the loss function.


Covolutional Neural Network

A Convolutional Neural Network (CNN) is a type of deep learning neural network designed for image classification and computer vision tasks. It is composed of several layers, including convolutional layers, activation functions, pooling layers, and fully connected layers. The convolutional layers extract features from the input image, the activation functions introduce non-linearity to the model, the pooling layers reduce the spatial resolution of the feature maps, and the fully connected layers combine the extracted features to perform classification. The architecture of a CNN is designed in such a way that it is able to automatically learn hierarchical representations of the input data, making it well suited for image classification tasks.

A Convolutional Neural Network (CNN) kernel and pooling:

A kernel, also known as a filter in Convolutional Neural Networks (CNNs), is a small matrix that slides over the input image to extract features and form feature maps. The values in the kernel are learned during the training process and are updated using backpropagation. The kernel performs element-wise multiplication with the overlapping portion of the input image and sums up the results to produce one pixel value in the feature map. By convolving multiple kernels with different sizes, shapes, and orientations, a CNN can capture different levels of abstraction and patterns in the input image.

A Pooling, is a down-sampling operation used in convolutional neural networks (CNNs) to reduce the spatial size of the input volume. The goal of pooling is to reduce the number of parameters and computation in the network, as well as to control over-fitting. Pooling also provides a form of translation invariance, meaning that features learned at a certain location in the input volume can be detected at any location in the output. There are two main types of pooling: max pooling and average pooling. Max pooling takes the maximum value in a set of inputs as the output, while average pooling takes the average of the inputs as the output. The pooling operation is performed over a small neighborhood of the input, such as 2x2 or 3x3 pixels, and the output is reduced by a factor equal to the size of the neighborhood.

## Loss Functions definitions:

Mean Squared Error (MSE) loss: MSE is often used in regression problems, where the goal is to predict a continuous output value. It measures the average squared difference between the predicted and actual values.

Binary Cross-Entropy (BCE) loss: BCE is commonly used in binary classification problems, where the goal is to predict one of two possible classes. It measures the dissimilarity between the predicted probability distribution and the true distribution.

Categorical Cross-Entropy (CCE) loss: CCE is used in multi-class classification problems, where the goal is to predict one of several possible classes. It measures the dissimilarity between the predicted probability distribution and the true distribution.

Hinge loss: Hinge loss is used in maximum-margin classification problems, such as support vector machines (SVMs). It penalizes predictions that are not confident and far from the decision boundary.

Kullback-Leibler (KL) divergence loss: KL divergence is used in generative models, such as variational autoencoders (VAEs) and generative adversarial networks (GANs), where the goal is to match the distribution of the generated data to that of the target data.

Dice loss: Dice loss is used in semantic segmentation problems, where the goal is to assign a class label to each pixel in an image. It measures the overlap between the predicted and target segmentations.

Jaccard loss: Jaccard loss is similar to Dice loss and is also used in semantic segmentation problems. It measures the overlap between the predicted and target segmentations.
