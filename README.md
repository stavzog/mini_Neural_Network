# Mini Neural Network
This is a neural network library written in kotlin made by following the logic of 
[these](https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh) videos. 
This liberary allows the creation of a NN with 2 dense layers. This library also includes a basic matrix operations library.

## Getting Started
### Create a neural network
```kotlin
//Use the NeuralNetwork(inputNodes,hiddenNodes,outputNodes) class
val nn = NeuralNetwork(2,3,1)
```
### Train your neural network
```kotlin
//Train your network with backpropagation algorithm using train(inputs: DoubleArray, targetOutput: DoubleArray)
nn.train(doubleArrayOf(1,0),doubleArrayOf(0))
```
### Make predictions
```kotlin
//Get an output with guess(input: DoubleArray)
val output = nn.guess(doubleArrayOf(0,0)) 
```
---
Currently, you can get this library by cloning the repo and running `./gradlew publishToMavenLocal`
