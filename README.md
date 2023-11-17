# Renewed Library
A new simple neural network library inspired by kotlin-dl. I tried to replicate the way the kotlin-dl library is written because I think it is a great example of oop and it shows good code architecture.
Currently, the library supports neural network creation with dense layers and everything works perfectly except the math of the Stochastic Gradient Descent algorithm (I couldn't understand the math I'm too young). As such this library remains unusable.
I have put a lot of work into deconstructing the kotlin-dl library and replicating the way it uses the unique features of the Kotlin language to develop a code architecture that basically involves a lot of small modules that all work together. I found it amazing how a lot of the files had just one or two pages of code and they distributed all the different modules into smaller files. However, this sort of distribution made it more difficult to pass information around. The kotlin-dl library used TensorFlow to handle the data of the neural networks which meant I had to discover my own ways to pass on information to the next layer about the previous layer.

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
