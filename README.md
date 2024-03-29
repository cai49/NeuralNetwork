# Custom Neural Network Lib
There are multiple NN libraries out there that are very powerful. Maybe to much powerful and resource intensive, with a very convoluted architecture that makes debugging and edge deployment awkwardly harder. 

This project aims to be an easy-to-use tiny-featured library that makes working with ML models a breeze with powerful one-liners with the ability to customize the whole learning pipeline and the ability to export models as json files to be used on the edge.

This initial release has simple but powerful features, with the ability to build a perceptron with any number of hidden layers with customizable I/O sizes. One feature is that a NN might be expressed in terms of hidden layers as well as their activation functions.

Currently the following activation functions are available:
- Sigmoid
- Tanh
- ReLU
