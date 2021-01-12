# Builder

Adapted from "Neural Networks from Scratch" by Harrison Kinsley and Daniel Kukiela -- for use in personal projects, to be used as a package for later usage.

Having followed the NNFS material, will it work with real data?

## The Process

Currently, I'm working to flush out and add to the NNFS framework with my own custom functions, adding functionality, and providing docstrings and example cases to classes/methods. I'll be adding material as I require different features in my future projects. Happy to take any suggestions for improvements! 

## Current Functionality

Layers:

* Dense
* Dropout
* Input (utility layer)

Activation Functions:

* Rectified Linear Unit (ReLU)
* Sigmoid
* Softmax
* Linear

Optimizer Functions:

* Stochastic Gradient Descent (SGD)
* Adagrad
* RMSprop
* Adam

Loss Functions:

* Categorical Cross-Entropy
* Binary Cross-Entropy
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

Accuracy Metrics (leverage a loss function):

* Regression
* Categorical

## An Example Case

```python
model = Model()
model.add(LayerDense(2, 128))
model.add(Activation_ReLU())
model.add(LayerDropout(0.1)) # 10% dropout
model.add(LayerDense(128, 10))
model.add(ActivationSoftmax())
```
