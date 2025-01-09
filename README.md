# Renewed Library
A new simple neural network library inspired by kotlin-dl. It tries to replicate how the kotlin-dl library is written because it is a great example of OOP and has good code architecture.
Currently, the library supports the creation of a neural network with dense layers. Everything works except the math of the Stochastic Gradient Descent algorithm. As such this code remains unusable.
I have put a lot of work into deconstructing the kotlin-dl library and replicating how it uses the unique features of the Kotlin language to develop a code architecture that involves a lot of small modules that all work together. However, this sort of distribution made it more difficult to pass information around. The kotlin-dl library used TensorFlow to handle the data of the neural networks which meant I had to discover other ways to pass on information to the next layer about the previous layer.

## Making a network in the new library
```kotlin
val model = Network(
  Input(2),
  Dense(2),
  Activation(Activations.ReLu),
  Dense(2, activation = Activations.Sigmoid)
)
```
<sub>Feel free to contribute</sub>
